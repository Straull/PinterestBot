"""Panneau de contrôle gauche de l'interface."""

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QSpinBox, QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt

from trading_bot.data.market_data import MarketData


class ControlsPanel(QFrame):
    """Panneau de contrôle avec les paramètres du bot."""

    # Signaux
    load_requested = pyqtSignal(str, str)     # symbol, period
    train_requested = pyqtSignal(int)         # horizon
    live_requested = pyqtSignal(bool)         # start/stop
    save_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFixedWidth(280)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Titre
        title = QLabel("Trading Bot V1")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("ML Ensemble | LSTM + XGBoost + LightGBM")
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # --- Section : Données ---
        self._add_section(layout, "DONNEES")

        # Symbole
        self._add_label(layout, "Symbole boursier")
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Ex: AAPL, MSFT, TSLA...")
        self.symbol_input.setText("AAPL")
        layout.addWidget(self.symbol_input)

        # Période
        self._add_label(layout, "Période historique")
        self.period_combo = QComboBox()
        for label, code in MarketData.PERIODS.items():
            self.period_combo.addItem(label, code)
        self.period_combo.setCurrentIndex(3)  # 1 an par défaut
        layout.addWidget(self.period_combo)

        # Bouton charger
        self.load_btn = QPushButton("Charger les données")
        self.load_btn.setObjectName("primary")
        self.load_btn.clicked.connect(self._on_load)
        layout.addWidget(self.load_btn)

        # Info données
        self.data_info = QLabel("Aucune donnée chargée")
        self.data_info.setObjectName("muted")
        self.data_info.setWordWrap(True)
        layout.addWidget(self.data_info)

        layout.addSpacing(8)

        # --- Section : ML ---
        self._add_section(layout, "MACHINE LEARNING")

        # Horizon de prédiction
        self._add_label(layout, "Horizon de prédiction (jours)")
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 30)
        self.horizon_spin.setValue(5)
        layout.addWidget(self.horizon_spin)

        # Bouton entraîner
        self.train_btn = QPushButton("Entraîner le modèle")
        self.train_btn.setObjectName("success")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._on_train)
        layout.addWidget(self.train_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status entraînement
        self.train_status = QLabel("")
        self.train_status.setObjectName("muted")
        self.train_status.setWordWrap(True)
        layout.addWidget(self.train_status)

        layout.addSpacing(8)

        # --- Section : Live ---
        self._add_section(layout, "SUIVI EN DIRECT")

        self.live_btn = QPushButton("Lancer le suivi live")
        self.live_btn.setObjectName("primary")
        self.live_btn.setEnabled(False)
        self.live_btn.setCheckable(True)
        self.live_btn.clicked.connect(self._on_live)
        layout.addWidget(self.live_btn)

        self.live_status = QLabel("")
        self.live_status.setObjectName("muted")
        self.live_status.setWordWrap(True)
        layout.addWidget(self.live_status)

        layout.addSpacing(8)

        # --- Bouton sauvegarder ---
        self.save_btn = QPushButton("Sauvegarder le modèle")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self.save_btn)

        # Spacer
        layout.addStretch()

    def _add_section(self, layout, text: str):
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: #30363d;")
        layout.addWidget(sep)

        label = QLabel(text)
        label.setObjectName("section")
        layout.addWidget(label)

    def _add_label(self, layout, text: str):
        label = QLabel(text)
        label.setObjectName("muted")
        layout.addWidget(label)

    def _on_load(self):
        symbol = self.symbol_input.text().strip().upper()
        period = self.period_combo.currentData()
        if symbol:
            self.load_requested.emit(symbol, period)

    def _on_train(self):
        horizon = self.horizon_spin.value()
        self.train_requested.emit(horizon)

    def _on_live(self):
        is_active = self.live_btn.isChecked()
        if is_active:
            self.live_btn.setText("Arrêter le suivi")
            self.live_btn.setObjectName("danger")
        else:
            self.live_btn.setText("Lancer le suivi live")
            self.live_btn.setObjectName("primary")
        self.live_btn.style().unpolish(self.live_btn)
        self.live_btn.style().polish(self.live_btn)
        self.live_requested.emit(is_active)

    def set_data_loaded(self, rows: int, symbol: str):
        self.data_info.setText(f"{symbol} : {rows} lignes chargées")
        self.train_btn.setEnabled(True)

    def set_training_progress(self, value: int, message: str = ""):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        if message:
            self.train_status.setText(message)

    def set_training_done(self):
        self.progress_bar.setVisible(False)
        self.live_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def set_live_status(self, message: str):
        self.live_status.setText(message)
