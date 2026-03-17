"""Modele LSTM bidirectionnel avec mecanisme d'Attention pour le trading.

PyTorch est importe dynamiquement. Si indisponible, LSTMTrainer.is_available()
retourne False et le moteur ML fonctionne avec XGBoost + LightGBM uniquement.
"""

import numpy as np

# Import dynamique de torch - ne crash pas si absent/casse
TORCH_AVAILABLE = False
torch = None
nn = None
Dataset = None
DataLoader = None

try:
    import torch as _torch
    import torch.nn as _nn
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader

    torch = _torch
    nn = _nn
    Dataset = _Dataset
    DataLoader = _DataLoader
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"[LSTM] PyTorch non disponible: {e}")
    print("[LSTM] Le bot fonctionnera avec XGBoost + LightGBM uniquement.")


class LSTMTrainer:
    """Entraineur pour le modele LSTM avec support GPU (ROCm/CUDA).

    Si PyTorch n'est pas disponible, is_available() retourne False.
    """

    @staticmethod
    def is_available() -> bool:
        return TORCH_AVAILABLE

    def __init__(self, input_size: int, seq_length: int = 60,
                 hidden_size: int = 128, num_layers: int = 2,
                 learning_rate: float = 0.001, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch non disponible")

        self.seq_length = seq_length
        self.input_size = input_size
        self.device = self._get_device()

        self.model = _LSTMAttentionNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            print(f"GPU detecte : {name}")
            return device
        print("Pas de GPU detecte, utilisation du CPU")
        return torch.device("cpu")

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X)):
            X_seq.append(X[i - self.seq_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 64,
              progress_callback=None) -> dict:
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            raise ValueError("Pas assez de donnees pour creer des sequences")

        train_dataset = _TimeSeriesDataset(X_train_seq, y_train_seq)
        val_dataset = _TimeSeriesDataset(X_val_seq, y_val_seq)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_model_state = None
        patience_counter = 0
        early_stop_patience = 15
        epoch = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            val_loss /= len(val_loader)
            val_acc = correct / total if total > 0 else 0

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, val_loss, val_acc)

            if patience_counter >= early_stop_patience:
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        final_acc = self._evaluate(val_loader)

        return {
            "epochs_trained": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "final_val_accuracy": final_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "device": str(self.device),
        }

    def _evaluate(self, data_loader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        return correct / total if total > 0 else 0

    def predict(self, X: np.ndarray) -> tuple:
        self.model.eval()
        if len(X) < self.seq_length:
            raise ValueError(f"Besoin d'au moins {self.seq_length} lignes")

        X_seq = X[-self.seq_length:].reshape(1, self.seq_length, -1)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            prediction = int(np.argmax(probs))

        return prediction, probs

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "seq_length": self.seq_length,
            "input_size": self.input_size,
            "best_val_loss": self.best_val_loss,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]


# --- Classes internes PyTorch (definies seulement si torch est disponible) ---

if TORCH_AVAILABLE:

    class _TimeSeriesDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class _Attention(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, lstm_output):
            weights = self.attention(lstm_output)
            weights = torch.softmax(weights, dim=1)
            context = torch.sum(weights * lstm_output, dim=1)
            return context, weights

    class _LSTMAttentionNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.attention = _Attention(hidden_size)
            self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            context, _ = self.attention(lstm_out)
            context = self.batch_norm(context)
            return self.classifier(context)
