"""Panneau de résultats et prévisions."""

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QScrollArea, QWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt

from trading_bot.gui.styles import COLORS


class ResultsPanel(QFrame):
    """Panneau affichant les résultats ML et les prévisions en direct."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFixedWidth(320)
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- Prévision principale ---
        pred_card = QFrame()
        pred_card.setObjectName("card")
        pred_layout = QVBoxLayout(pred_card)
        pred_layout.setSpacing(8)

        pred_title = QLabel("PREVISION")
        pred_title.setObjectName("section")
        pred_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(pred_title)

        self.direction_label = QLabel("--")
        self.direction_label.setObjectName("value_neutral")
        self.direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.direction_label)

        self.confidence_label = QLabel("Confiance : --")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.confidence_label)

        # Probabilités
        prob_row = QHBoxLayout()
        self.prob_down_label = QLabel("Baisse: --")
        self.prob_down_label.setStyleSheet(f"color: {COLORS['accent_red']};")
        self.prob_down_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prob_row.addWidget(self.prob_down_label)

        self.prob_up_label = QLabel("Hausse: --")
        self.prob_up_label.setStyleSheet(f"color: {COLORS['accent_green']};")
        self.prob_up_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prob_row.addWidget(self.prob_up_label)
        pred_layout.addLayout(prob_row)

        layout.addWidget(pred_card)

        # --- Détails par modèle ---
        detail_card = QFrame()
        detail_card.setObjectName("card")
        detail_layout = QVBoxLayout(detail_card)

        detail_title = QLabel("DETAILS PAR MODELE")
        detail_title.setObjectName("section")
        detail_layout.addWidget(detail_title)

        self.lstm_detail = QLabel("LSTM : --")
        self.xgb_detail = QLabel("XGBoost : --")
        self.lgbm_detail = QLabel("LightGBM : --")
        self.weights_label = QLabel("Poids : --")
        self.weights_label.setObjectName("muted")

        for w in [self.lstm_detail, self.xgb_detail, self.lgbm_detail, self.weights_label]:
            detail_layout.addWidget(w)

        layout.addWidget(detail_card)

        # --- Prix en direct ---
        live_card = QFrame()
        live_card.setObjectName("card")
        live_layout = QVBoxLayout(live_card)

        live_title = QLabel("MARCHE EN DIRECT")
        live_title.setObjectName("section")
        live_layout.addWidget(live_title)

        self.price_label = QLabel("Prix : --")
        self.price_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        live_layout.addWidget(self.price_label)

        self.change_label = QLabel("Variation : --")
        live_layout.addWidget(self.change_label)

        self.volume_label = QLabel("Volume : --")
        self.volume_label.setObjectName("muted")
        live_layout.addWidget(self.volume_label)

        self.timestamp_label = QLabel("")
        self.timestamp_label.setObjectName("muted")
        live_layout.addWidget(self.timestamp_label)

        layout.addWidget(live_card)

        # --- Résultats ML ---
        ml_card = QFrame()
        ml_card.setObjectName("card")
        ml_layout = QVBoxLayout(ml_card)

        ml_title = QLabel("RESULTATS ENTRAINEMENT")
        ml_title.setObjectName("section")
        ml_layout.addWidget(ml_title)

        self.accuracy_labels = {}
        for name in ["XGBoost", "LightGBM", "LSTM", "Ensemble"]:
            label = QLabel(f"{name} : --")
            ml_layout.addWidget(label)
            self.accuracy_labels[name] = label

        self.samples_label = QLabel("")
        self.samples_label.setObjectName("muted")
        ml_layout.addWidget(self.samples_label)

        layout.addWidget(ml_card)

        # --- Feature Importance ---
        fi_card = QFrame()
        fi_card.setObjectName("card")
        fi_layout = QVBoxLayout(fi_card)

        fi_title = QLabel("IMPORTANCE DES FEATURES")
        fi_title.setObjectName("section")
        fi_layout.addWidget(fi_title)

        self.fi_text = QTextEdit()
        self.fi_text.setReadOnly(True)
        self.fi_text.setMaximumHeight(200)
        fi_layout.addWidget(self.fi_text)

        layout.addWidget(fi_card)

        layout.addStretch()

        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def update_prediction(self, prediction: dict):
        """Met à jour l'affichage de la prévision."""
        direction = prediction["direction"]
        confidence = prediction["confidence"]

        self.direction_label.setText(f"{'▲' if direction == 'HAUSSE' else '▼'} {direction}")
        if direction == "HAUSSE":
            self.direction_label.setObjectName("value_up")
        else:
            self.direction_label.setObjectName("value_down")
        self.direction_label.style().unpolish(self.direction_label)
        self.direction_label.style().polish(self.direction_label)

        self.confidence_label.setText(f"Confiance : {confidence:.1f}%")
        self.prob_down_label.setText(f"Baisse: {prediction['prob_down']:.1f}%")
        self.prob_up_label.setText(f"Hausse: {prediction['prob_up']:.1f}%")

        # Détails par modèle
        if "details" in prediction:
            d = prediction["details"]
            self.lstm_detail.setText(f"LSTM : ▲{d['lstm']['prob_up']:.1f}% / ▼{d['lstm']['prob_down']:.1f}%")
            self.xgb_detail.setText(f"XGBoost : ▲{d['xgb']['prob_up']:.1f}% / ▼{d['xgb']['prob_down']:.1f}%")
            self.lgbm_detail.setText(f"LightGBM : ▲{d['lgbm']['prob_up']:.1f}% / ▼{d['lgbm']['prob_down']:.1f}%")

        if "weights" in prediction:
            w = prediction["weights"]
            self.weights_label.setText(
                f"Poids: LSTM={w['lstm']:.0%} XGB={w['xgb']:.0%} LGBM={w['lgbm']:.0%}"
            )

    def update_live_data(self, data: dict):
        """Met à jour les données de marché en direct."""
        price = data["price"]
        change = data["change"]
        change_pct = data["change_pct"]

        self.price_label.setText(f"${price:,.2f}")

        sign = "+" if change >= 0 else ""
        self.change_label.setText(f"{sign}{change:,.2f} ({sign}{change_pct:.2f}%)")
        color = COLORS["accent_green"] if change >= 0 else COLORS["accent_red"]
        self.change_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        vol = data["volume"]
        if vol >= 1_000_000:
            vol_str = f"{vol / 1_000_000:.1f}M"
        elif vol >= 1_000:
            vol_str = f"{vol / 1_000:.1f}K"
        else:
            vol_str = str(vol)
        self.volume_label.setText(f"Volume : {vol_str}")
        self.timestamp_label.setText(f"Mis à jour : {data['timestamp']}")

    def update_training_results(self, results: dict):
        """Met à jour les résultats d'entraînement."""
        self.accuracy_labels["XGBoost"].setText(f"XGBoost : {results['xgb_accuracy']:.1%}")
        self.accuracy_labels["LightGBM"].setText(f"LightGBM : {results['lgbm_accuracy']:.1%}")
        self.accuracy_labels["LSTM"].setText(
            f"LSTM : {results['lstm_accuracy']:.1%} ({results['lstm_device']})"
        )
        self.accuracy_labels["Ensemble"].setText(f"Ensemble : {results['ensemble_accuracy']:.1%}")
        self.accuracy_labels["Ensemble"].setStyleSheet(
            f"color: {COLORS['accent_blue']}; font-weight: bold; font-size: 14px;"
        )

        self.samples_label.setText(
            f"Train: {results['train_samples']} | Test: {results['test_samples']} | "
            f"LSTM epochs: {results['lstm_epochs']}"
        )

        # Feature importance
        if results.get("feature_importance"):
            sorted_fi = sorted(results["feature_importance"].items(), key=lambda x: x[1], reverse=True)
            lines = []
            for name, imp in sorted_fi[:10]:
                bar = "█" * int(imp * 50)
                lines.append(f"{name:>15s}  {bar} {imp:.3f}")
            self.fi_text.setText("\n".join(lines))

    def clear(self):
        """Réinitialise tous les affichages."""
        self.direction_label.setText("--")
        self.direction_label.setObjectName("value_neutral")
        self.confidence_label.setText("Confiance : --")
        self.prob_down_label.setText("Baisse: --")
        self.prob_up_label.setText("Hausse: --")
        self.lstm_detail.setText("LSTM : --")
        self.xgb_detail.setText("XGBoost : --")
        self.lgbm_detail.setText("LightGBM : --")
        self.weights_label.setText("Poids : --")
        self.price_label.setText("Prix : --")
        self.change_label.setText("Variation : --")
        self.volume_label.setText("Volume : --")
        self.timestamp_label.setText("")
        for label in self.accuracy_labels.values():
            label.setText(label.text().split(":")[0] + " : --")
        self.samples_label.setText("")
        self.fi_text.clear()
