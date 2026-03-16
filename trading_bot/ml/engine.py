"""Moteur ML hybride : Ensemble LSTM + XGBoost + LightGBM."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

from trading_bot.ml.lstm_model import LSTMTrainer


FEATURE_COLUMNS = [
    "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_High", "BB_Low", "BB_Mid", "ATR",
    "Volume_SMA", "Return_1d", "Return_5d", "Return_10d", "Volatility",
]


class MLEngine:
    """Moteur ML ensemble pour prédire les mouvements de marché.

    Combine 3 modèles :
    - LSTM bidirectionnel avec Attention (PyTorch, GPU)
    - XGBoost (CPU)
    - LightGBM (CPU)

    La prédiction finale est une moyenne pondérée des probabilités.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.lgbm_model = None
        self.lstm_trainer = None
        self.is_trained = False
        self.results = {}
        self.feature_importance = None
        self.weights = {"lstm": 0.4, "xgb": 0.35, "lgbm": 0.25}
        self._features_used = []

    def prepare_data(self, df: pd.DataFrame, horizon: int = 5) -> tuple:
        """Prépare les données pour l'entraînement."""
        df = df.copy()
        df["Target"] = (df["Close"].shift(-horizon) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        self._features_used = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[self._features_used].values
        y = df["Target"].values

        return X, y

    def train(self, df: pd.DataFrame, horizon: int = 5, test_size: float = 0.2,
              lstm_epochs: int = 100, progress_callback=None) -> dict:
        """Entraîne l'ensemble des 3 modèles.

        Args:
            progress_callback: fonction(stage, progress_pct, message)
                stage: "xgb", "lgbm", "lstm"
        """
        X, y = self.prepare_data(df, horizon)

        if len(X) < 100:
            raise ValueError("Pas assez de données (minimum 100 lignes après traitement)")

        # Split chronologique
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.results = {}

        # --- 1. XGBoost ---
        if progress_callback:
            progress_callback("xgb", 0, "Entraînement XGBoost...")

        self.xgb_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss",
            early_stopping_rounds=20,
        )
        self.xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
        xgb_pred = self.xgb_model.predict(X_test_scaled)
        xgb_proba = self.xgb_model.predict_proba(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)

        self.results["xgb"] = {
            "accuracy": xgb_acc,
            "report": classification_report(y_test, xgb_pred, target_names=["Baisse", "Hausse"]),
        }

        if progress_callback:
            progress_callback("xgb", 100, f"XGBoost terminé - Accuracy: {xgb_acc:.1%}")

        # --- 2. LightGBM ---
        if progress_callback:
            progress_callback("lgbm", 0, "Entraînement LightGBM...")

        self.lgbm_model = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        self.lgbm_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
        )
        lgbm_pred = self.lgbm_model.predict(X_test_scaled)
        lgbm_proba = self.lgbm_model.predict_proba(X_test_scaled)
        lgbm_acc = accuracy_score(y_test, lgbm_pred)

        self.results["lgbm"] = {
            "accuracy": lgbm_acc,
            "report": classification_report(y_test, lgbm_pred, target_names=["Baisse", "Hausse"]),
        }

        if progress_callback:
            progress_callback("lgbm", 100, f"LightGBM terminé - Accuracy: {lgbm_acc:.1%}")

        # --- 3. LSTM ---
        if progress_callback:
            progress_callback("lstm", 0, "Entraînement LSTM (GPU si disponible)...")

        n_features = X_train_scaled.shape[1]
        self.lstm_trainer = LSTMTrainer(
            input_size=n_features,
            seq_length=60,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.001,
            dropout=0.3,
        )

        def lstm_progress(epoch, total, train_loss, val_loss, val_acc):
            pct = int(epoch / total * 100)
            if progress_callback:
                progress_callback("lstm", pct,
                                  f"LSTM Epoch {epoch}/{total} - Loss: {val_loss:.4f} - Acc: {val_acc:.1%}")

        lstm_results = self.lstm_trainer.train(
            X_train_scaled, y_train,
            X_val=X_test_scaled, y_val=y_test,
            epochs=lstm_epochs,
            batch_size=64,
            progress_callback=lstm_progress,
        )

        # Évaluer LSTM sur le test set
        lstm_correct = 0
        lstm_proba_list = []
        for i in range(self.lstm_trainer.seq_length, len(X_test_scaled)):
            pred, probs = self.lstm_trainer.predict(X_test_scaled[:i + 1])
            lstm_proba_list.append(probs)
            if pred == y_test[i]:
                lstm_correct += 1

        lstm_test_count = len(X_test_scaled) - self.lstm_trainer.seq_length
        lstm_acc = lstm_correct / lstm_test_count if lstm_test_count > 0 else 0

        self.results["lstm"] = {
            "accuracy": lstm_acc,
            "device": lstm_results["device"],
            "epochs_trained": lstm_results["epochs_trained"],
            "best_val_loss": lstm_results["best_val_loss"],
        }

        if progress_callback:
            progress_callback("lstm", 100, f"LSTM terminé - Accuracy: {lstm_acc:.1%}")

        # --- Ensemble : optimiser les poids ---
        self._optimize_weights(xgb_proba, lgbm_proba, lstm_proba_list, y_test)

        # Feature importance (moyenne XGBoost + LightGBM)
        xgb_imp = self.xgb_model.feature_importances_
        lgbm_imp = self.lgbm_model.feature_importances_
        avg_imp = (xgb_imp + lgbm_imp) / 2
        self.feature_importance = dict(zip(self._features_used, avg_imp))

        # Accuracy ensemble
        ensemble_acc = self._ensemble_accuracy(
            xgb_proba, lgbm_proba, lstm_proba_list, y_test
        )
        self.results["ensemble"] = {
            "accuracy": ensemble_acc,
            "weights": self.weights.copy(),
        }

        self.is_trained = True

        if progress_callback:
            progress_callback("done", 100, f"Entraînement terminé ! Accuracy ensemble: {ensemble_acc:.1%}")

        return {
            "xgb_accuracy": xgb_acc,
            "lgbm_accuracy": lgbm_acc,
            "lstm_accuracy": lstm_acc,
            "ensemble_accuracy": ensemble_acc,
            "weights": self.weights.copy(),
            "feature_importance": self.feature_importance,
            "lstm_device": lstm_results["device"],
            "lstm_epochs": lstm_results["epochs_trained"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def _optimize_weights(self, xgb_proba, lgbm_proba, lstm_proba_list, y_test):
        """Optimise les poids de l'ensemble par grid search."""
        seq_len = self.lstm_trainer.seq_length
        # Aligner les probabilités (LSTM commence après seq_length)
        xgb_p = xgb_proba[seq_len:]
        lgbm_p = lgbm_proba[seq_len:]
        lstm_p = np.array(lstm_proba_list)
        y = y_test[seq_len:]

        if len(lstm_p) == 0:
            return

        best_acc = 0
        best_w = self.weights.copy()

        for w_lstm in np.arange(0.1, 0.7, 0.05):
            for w_xgb in np.arange(0.1, 1.0 - w_lstm, 0.05):
                w_lgbm = 1.0 - w_lstm - w_xgb
                if w_lgbm < 0.05:
                    continue
                ensemble_p = w_lstm * lstm_p + w_xgb * xgb_p + w_lgbm * lgbm_p
                preds = np.argmax(ensemble_p, axis=1)
                acc = accuracy_score(y, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_w = {"lstm": round(w_lstm, 2), "xgb": round(w_xgb, 2), "lgbm": round(w_lgbm, 2)}

        self.weights = best_w

    def _ensemble_accuracy(self, xgb_proba, lgbm_proba, lstm_proba_list, y_test):
        """Calcule l'accuracy de l'ensemble."""
        seq_len = self.lstm_trainer.seq_length
        xgb_p = xgb_proba[seq_len:]
        lgbm_p = lgbm_proba[seq_len:]
        lstm_p = np.array(lstm_proba_list)
        y = y_test[seq_len:]

        if len(lstm_p) == 0:
            return 0

        w = self.weights
        ensemble_p = w["lstm"] * lstm_p + w["xgb"] * xgb_p + w["lgbm"] * lgbm_p
        preds = np.argmax(ensemble_p, axis=1)
        return accuracy_score(y, preds)

    def predict(self, df: pd.DataFrame) -> dict:
        """Fait une prédiction ensemble sur les données récentes."""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas encore entraîné")

        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[available].values
        X_scaled = self.scaler.transform(X)

        # XGBoost
        xgb_proba = self.xgb_model.predict_proba(X_scaled[-1:])
        # LightGBM
        lgbm_proba = self.lgbm_model.predict_proba(X_scaled[-1:])
        # LSTM
        lstm_pred, lstm_proba = self.lstm_trainer.predict(X_scaled)

        # Ensemble
        w = self.weights
        ensemble_proba = (
            w["lstm"] * lstm_proba +
            w["xgb"] * xgb_proba[0] +
            w["lgbm"] * lgbm_proba[0]
        )

        prediction = int(np.argmax(ensemble_proba))
        confidence = float(max(ensemble_proba) * 100)
        direction = "HAUSSE" if prediction == 1 else "BAISSE"

        return {
            "direction": direction,
            "prediction": prediction,
            "confidence": confidence,
            "prob_down": float(ensemble_proba[0] * 100),
            "prob_up": float(ensemble_proba[1] * 100),
            "details": {
                "xgb": {"prob_up": float(xgb_proba[0][1] * 100), "prob_down": float(xgb_proba[0][0] * 100)},
                "lgbm": {"prob_up": float(lgbm_proba[0][1] * 100), "prob_down": float(lgbm_proba[0][0] * 100)},
                "lstm": {"prob_up": float(lstm_proba[1] * 100), "prob_down": float(lstm_proba[0] * 100)},
            },
            "weights": self.weights,
        }

    def save_models(self, path: str = "models"):
        """Sauvegarde tous les modèles."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.xgb_model, os.path.join(path, "xgb_model.pkl"))
        joblib.dump(self.lgbm_model, os.path.join(path, "lgbm_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.weights, os.path.join(path, "weights.pkl"))
        joblib.dump(self._features_used, os.path.join(path, "features.pkl"))
        if self.lstm_trainer:
            self.lstm_trainer.save(os.path.join(path, "lstm_model.pt"))

    def load_models(self, path: str = "models"):
        """Charge les modèles sauvegardés."""
        self.xgb_model = joblib.load(os.path.join(path, "xgb_model.pkl"))
        self.lgbm_model = joblib.load(os.path.join(path, "lgbm_model.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        self.weights = joblib.load(os.path.join(path, "weights.pkl"))
        self._features_used = joblib.load(os.path.join(path, "features.pkl"))
        # LSTM
        n_features = len(self._features_used)
        self.lstm_trainer = LSTMTrainer(input_size=n_features, seq_length=60)
        self.lstm_trainer.load(os.path.join(path, "lstm_model.pt"))
        self.is_trained = True
