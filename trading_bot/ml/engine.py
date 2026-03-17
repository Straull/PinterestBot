"""Moteur ML hybride : Ensemble LSTM + XGBoost + LightGBM.

Si PyTorch n'est pas disponible, le moteur fonctionne avec XGBoost + LightGBM uniquement.
"""

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
    """Moteur ML ensemble pour predire les mouvements de marche.

    Combine jusqu'a 3 modeles :
    - LSTM bidirectionnel avec Attention (PyTorch, GPU) - si disponible
    - XGBoost (CPU)
    - LightGBM (CPU)

    Si PyTorch n'est pas installe, seuls XGBoost + LightGBM sont utilises.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.lgbm_model = None
        self.lstm_trainer = None
        self.is_trained = False
        self.results = {}
        self.feature_importance = None
        self.use_lstm = LSTMTrainer.is_available()
        if self.use_lstm:
            self.weights = {"lstm": 0.4, "xgb": 0.35, "lgbm": 0.25}
        else:
            self.weights = {"xgb": 0.55, "lgbm": 0.45}
            print("[MLEngine] PyTorch indisponible - mode XGBoost + LightGBM uniquement")
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
            progress_callback("lgbm", 100, f"LightGBM termine - Accuracy: {lgbm_acc:.1%}")

        # --- 3. LSTM (si PyTorch disponible) ---
        lstm_acc = 0
        lstm_proba_list = []
        lstm_device = "N/A"
        lstm_epochs_trained = 0

        if self.use_lstm:
            if progress_callback:
                progress_callback("lstm", 0, "Entrainement LSTM (GPU si disponible)...")

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

            # Evaluer LSTM sur le test set
            lstm_correct = 0
            for i in range(self.lstm_trainer.seq_length, len(X_test_scaled)):
                pred, probs = self.lstm_trainer.predict(X_test_scaled[:i + 1])
                lstm_proba_list.append(probs)
                if pred == y_test[i]:
                    lstm_correct += 1

            lstm_test_count = len(X_test_scaled) - self.lstm_trainer.seq_length
            lstm_acc = lstm_correct / lstm_test_count if lstm_test_count > 0 else 0
            lstm_device = lstm_results["device"]
            lstm_epochs_trained = lstm_results["epochs_trained"]

            self.results["lstm"] = {
                "accuracy": lstm_acc,
                "device": lstm_device,
                "epochs_trained": lstm_epochs_trained,
                "best_val_loss": lstm_results["best_val_loss"],
            }

            if progress_callback:
                progress_callback("lstm", 100, f"LSTM termine - Accuracy: {lstm_acc:.1%}")
        else:
            if progress_callback:
                progress_callback("lstm", 100, "LSTM ignore (PyTorch indisponible)")

        # --- Ensemble : optimiser les poids ---
        if self.use_lstm and len(lstm_proba_list) > 0:
            self._optimize_weights(xgb_proba, lgbm_proba, lstm_proba_list, y_test)
        else:
            self._optimize_weights_no_lstm(xgb_proba, lgbm_proba, y_test)

        # Feature importance (moyenne XGBoost + LightGBM)
        xgb_imp = self.xgb_model.feature_importances_
        lgbm_imp = self.lgbm_model.feature_importances_
        avg_imp = (xgb_imp + lgbm_imp) / 2
        self.feature_importance = dict(zip(self._features_used, avg_imp))

        # Accuracy ensemble
        if self.use_lstm and len(lstm_proba_list) > 0:
            ensemble_acc = self._ensemble_accuracy(
                xgb_proba, lgbm_proba, lstm_proba_list, y_test
            )
        else:
            ensemble_acc = self._ensemble_accuracy_no_lstm(
                xgb_proba, lgbm_proba, y_test
            )

        self.results["ensemble"] = {
            "accuracy": ensemble_acc,
            "weights": self.weights.copy(),
        }

        self.is_trained = True

        if progress_callback:
            progress_callback("done", 100, f"Entrainement termine ! Accuracy ensemble: {ensemble_acc:.1%}")

        return {
            "xgb_accuracy": xgb_acc,
            "lgbm_accuracy": lgbm_acc,
            "lstm_accuracy": lstm_acc,
            "ensemble_accuracy": ensemble_acc,
            "weights": self.weights.copy(),
            "feature_importance": self.feature_importance,
            "lstm_device": lstm_device,
            "lstm_epochs": lstm_epochs_trained,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def _optimize_weights(self, xgb_proba, lgbm_proba, lstm_proba_list, y_test):
        """Optimise les poids de l'ensemble par grid search (avec LSTM)."""
        seq_len = self.lstm_trainer.seq_length
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

    def _optimize_weights_no_lstm(self, xgb_proba, lgbm_proba, y_test):
        """Optimise les poids XGBoost + LightGBM (sans LSTM)."""
        best_acc = 0
        best_w = self.weights.copy()

        for w_xgb in np.arange(0.1, 0.95, 0.05):
            w_lgbm = 1.0 - w_xgb
            ensemble_p = w_xgb * xgb_proba + w_lgbm * lgbm_proba
            preds = np.argmax(ensemble_p, axis=1)
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_w = {"xgb": round(w_xgb, 2), "lgbm": round(w_lgbm, 2)}

        self.weights = best_w

    def _ensemble_accuracy(self, xgb_proba, lgbm_proba, lstm_proba_list, y_test):
        """Calcule l'accuracy de l'ensemble (avec LSTM)."""
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

    def _ensemble_accuracy_no_lstm(self, xgb_proba, lgbm_proba, y_test):
        """Calcule l'accuracy de l'ensemble (sans LSTM)."""
        w = self.weights
        ensemble_p = w["xgb"] * xgb_proba + w["lgbm"] * lgbm_proba
        preds = np.argmax(ensemble_p, axis=1)
        return accuracy_score(y_test, preds)

    def predict(self, df: pd.DataFrame) -> dict:
        """Fait une prediction ensemble sur les donnees recentes."""
        if not self.is_trained:
            raise ValueError("Le modele n'est pas encore entraine")

        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[available].values
        X_scaled = self.scaler.transform(X)

        # XGBoost
        xgb_proba = self.xgb_model.predict_proba(X_scaled[-1:])
        # LightGBM
        lgbm_proba = self.lgbm_model.predict_proba(X_scaled[-1:])

        w = self.weights
        details = {
            "xgb": {"prob_up": float(xgb_proba[0][1] * 100), "prob_down": float(xgb_proba[0][0] * 100)},
            "lgbm": {"prob_up": float(lgbm_proba[0][1] * 100), "prob_down": float(lgbm_proba[0][0] * 100)},
        }

        # LSTM (si disponible et entraine)
        if self.use_lstm and self.lstm_trainer:
            lstm_pred, lstm_proba = self.lstm_trainer.predict(X_scaled)
            ensemble_proba = (
                w["lstm"] * lstm_proba +
                w["xgb"] * xgb_proba[0] +
                w["lgbm"] * lgbm_proba[0]
            )
            details["lstm"] = {"prob_up": float(lstm_proba[1] * 100), "prob_down": float(lstm_proba[0] * 100)}
        else:
            ensemble_proba = (
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
            "details": details,
            "weights": self.weights,
        }

    def save_models(self, path: str = "models"):
        """Sauvegarde tous les modeles."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.xgb_model, os.path.join(path, "xgb_model.pkl"))
        joblib.dump(self.lgbm_model, os.path.join(path, "lgbm_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.weights, os.path.join(path, "weights.pkl"))
        joblib.dump(self._features_used, os.path.join(path, "features.pkl"))
        if self.lstm_trainer:
            self.lstm_trainer.save(os.path.join(path, "lstm_model.pt"))

    def load_models(self, path: str = "models"):
        """Charge les modeles sauvegardes."""
        self.xgb_model = joblib.load(os.path.join(path, "xgb_model.pkl"))
        self.lgbm_model = joblib.load(os.path.join(path, "lgbm_model.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        self.weights = joblib.load(os.path.join(path, "weights.pkl"))
        self._features_used = joblib.load(os.path.join(path, "features.pkl"))
        # LSTM (seulement si PyTorch disponible)
        if self.use_lstm:
            lstm_path = os.path.join(path, "lstm_model.pt")
            if os.path.exists(lstm_path):
                n_features = len(self._features_used)
                self.lstm_trainer = LSTMTrainer(input_size=n_features, seq_length=60)
                self.lstm_trainer.load(lstm_path)
        self.is_trained = True
