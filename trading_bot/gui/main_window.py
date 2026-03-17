"""Fenêtre principale de l'application Trading Bot."""

import traceback
import pandas as pd
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QMessageBox, QSplitter, QTabWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from trading_bot.gui.styles import DARK_THEME
from trading_bot.gui.chart_widget import ChartWidget
from trading_bot.gui.controls_panel import ControlsPanel
from trading_bot.gui.results_panel import ResultsPanel
from trading_bot.gui.portfolio_tab import PortfolioTab
from trading_bot.data.market_data import MarketData
from trading_bot.ml.engine import MLEngine
from trading_bot.portfolio.manager import PortfolioManager
from trading_bot.portfolio.strategy import TradingStrategy


class DataWorker(QThread):
    """Thread pour le chargement des données."""
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, market_data: MarketData, symbol: str, period: str):
        super().__init__()
        self.market_data = market_data
        self.symbol = symbol
        self.period = period

    def run(self):
        try:
            df = self.market_data.fetch_historical(self.symbol, self.period)
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    """Thread pour l'entraînement ML."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str, int, str)  # stage, pct, message
    error = pyqtSignal(str)

    def __init__(self, engine: MLEngine, df: pd.DataFrame, horizon: int):
        super().__init__()
        self.engine = engine
        self.df = df
        self.horizon = horizon

    def run(self):
        try:
            def callback(stage, pct, message):
                self.progress.emit(stage, pct, message)

            results = self.engine.train(
                self.df, horizon=self.horizon,
                lstm_epochs=100, progress_callback=callback,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class LiveWorker(QThread):
    """Thread pour les données et prévisions en direct."""
    data_received = pyqtSignal(dict)
    prediction_received = pyqtSignal(dict)
    trade_signal = pyqtSignal(dict)  # Signal pour les décisions de trading
    error = pyqtSignal(str)

    def __init__(self, market_data: MarketData, engine: MLEngine,
                 strategy: TradingStrategy, symbol: str):
        super().__init__()
        self.market_data = market_data
        self.engine = engine
        self.strategy = strategy
        self.symbol = symbol

    def run(self):
        try:
            # Données live
            live_data = self.market_data.fetch_live(self.symbol)
            self.data_received.emit(live_data)

            # Prédiction + trading automatique
            if self.engine.is_trained:
                recent = self.market_data.get_recent_data_for_prediction(self.symbol)
                prediction = self.engine.predict(recent)
                self.prediction_received.emit(prediction)

                # Évaluer la stratégie et exécuter le trade
                current_price = live_data.get("price", 0)
                if current_price > 0:
                    decision = self.strategy.evaluate(
                        self.symbol, prediction, current_price
                    )
                    if decision and decision["action"] in ("BUY", "SELL"):
                        result = self.strategy.execute_decision(decision)
                        if result:
                            self.trade_signal.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Fenêtre principale du Trading Bot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot V1 - ML Ensemble")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # Composants métier
        self.market_data = MarketData()
        self.ml_engine = MLEngine()
        self.portfolio = PortfolioManager()
        self.strategy = TradingStrategy(self.portfolio)
        self.current_data = None
        self.live_timer = None
        self._workers = []

        # Appliquer le thème
        self.setStyleSheet(DARK_THEME)

        self._setup_ui()
        self._connect_signals()
        self._setup_statusbar()

        # Mise à jour du portefeuille au démarrage
        QTimer.singleShot(500, self._startup_portfolio_update)

    def _setup_ui(self):
        # Tab widget central
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Onglet 1 : Analyse & Trading ---
        analysis_tab = QWidget()
        analysis_layout = QHBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(8, 8, 8, 8)
        analysis_layout.setSpacing(8)

        # Panneau gauche : contrôles
        self.controls = ControlsPanel()
        analysis_layout.addWidget(self.controls)

        # Centre : graphiques
        self.chart = ChartWidget()
        analysis_layout.addWidget(self.chart, stretch=1)

        # Droite : résultats
        self.results = ResultsPanel()
        analysis_layout.addWidget(self.results)

        self.tabs.addTab(analysis_tab, "Analyse & Trading")

        # --- Onglet 2 : Portefeuille ---
        self.portfolio_tab = PortfolioTab(self.portfolio, self.strategy)
        self.tabs.addTab(self.portfolio_tab, "Portefeuille")

    def _connect_signals(self):
        self.controls.load_requested.connect(self._load_data)
        self.controls.train_requested.connect(self._train_model)
        self.controls.live_requested.connect(self._toggle_live)
        self.controls.save_requested.connect(self._save_model)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Pret. Entrez un symbole et chargez les donnees.")

    def _startup_portfolio_update(self):
        """Met à jour la valeur du portefeuille au démarrage."""
        try:
            self.portfolio_tab.refresh()
            self.statusbar.showMessage(
                f"Portefeuille charge - Cash: {self.portfolio.cash:,.2f}E | "
                f"Pret. Entrez un symbole et chargez les donnees."
            )
        except Exception:
            pass

    def _load_data(self, symbol: str, period: str):
        """Charge les données historiques dans un thread."""
        self.controls.load_btn.setEnabled(False)
        self.statusbar.showMessage(f"Chargement de {symbol}...")

        worker = DataWorker(self.market_data, symbol, period)
        worker.finished.connect(self._on_data_loaded)
        worker.error.connect(self._on_data_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        worker.error.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_data_loaded(self, df: pd.DataFrame):
        self.current_data = df
        self.chart.update_chart(df)
        self.controls.set_data_loaded(len(df), self.market_data.symbol)
        self.controls.load_btn.setEnabled(True)
        self.results.clear()

        info = self.market_data.get_info()
        name = info.get("name", self.market_data.symbol)
        self.statusbar.showMessage(
            f"{name} - {len(df)} lignes chargees | "
            f"Dernier prix: ${df['Close'].iloc[-1]:,.2f}"
        )

    def _on_data_error(self, error: str):
        self.controls.load_btn.setEnabled(True)
        self.statusbar.showMessage(f"Erreur: {error}")
        QMessageBox.warning(self, "Erreur de chargement", error)

    def _train_model(self, horizon: int):
        """Lance l'entraînement ML dans un thread."""
        if self.current_data is None:
            return

        self.controls.train_btn.setEnabled(False)
        self.controls.set_training_progress(0, "Demarrage de l'entrainement...")
        self.statusbar.showMessage("Entrainement en cours...")

        worker = TrainWorker(self.ml_engine, self.current_data, horizon)
        worker.progress.connect(self._on_train_progress)
        worker.finished.connect(self._on_train_done)
        worker.error.connect(self._on_train_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        worker.error.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_train_progress(self, stage: str, pct: int, message: str):
        stage_offsets = {"xgb": 0, "lgbm": 25, "lstm": 50, "done": 100}
        base = stage_offsets.get(stage, 0)
        if stage == "done":
            global_pct = 100
        elif stage == "lstm":
            global_pct = 50 + int(pct * 0.5)
        else:
            global_pct = base + int(pct * 0.25)

        self.controls.set_training_progress(min(global_pct, 100), message)

    def _on_train_done(self, results: dict):
        self.controls.train_btn.setEnabled(True)
        self.controls.set_training_done()
        self.results.update_training_results(results)

        # Faire une première prédiction
        try:
            recent = self.market_data.get_recent_data_for_prediction()
            prediction = self.ml_engine.predict(recent)
            self.results.update_prediction(prediction)
        except Exception:
            pass

        self.statusbar.showMessage(
            f"Entrainement termine | Accuracy ensemble: {results['ensemble_accuracy']:.1%} | "
            f"GPU: {results['lstm_device']}"
        )

    def _on_train_error(self, error: str):
        self.controls.train_btn.setEnabled(True)
        self.controls.set_training_progress(0, "Erreur")
        self.controls.progress_bar.setVisible(False)
        self.statusbar.showMessage("Erreur d'entrainement")
        QMessageBox.critical(self, "Erreur d'entrainement", error)

    def _toggle_live(self, active: bool):
        """Active/désactive le suivi en direct."""
        if active:
            self.live_timer = QTimer()
            self.live_timer.timeout.connect(self._fetch_live)
            self.live_timer.start(30000)
            self._fetch_live()
            # Activer aussi le rafraîchissement du portefeuille
            self.portfolio_tab.start_auto_refresh(60000)
            self.statusbar.showMessage("Suivi en direct active (maj toutes les 30s)")
        else:
            if self.live_timer:
                self.live_timer.stop()
                self.live_timer = None
            self.portfolio_tab.stop_auto_refresh()
            self.statusbar.showMessage("Suivi en direct desactive")

    def _fetch_live(self):
        """Récupère les données live dans un thread."""
        worker = LiveWorker(
            self.market_data, self.ml_engine,
            self.strategy, self.market_data.symbol,
        )
        worker.data_received.connect(self.results.update_live_data)
        worker.prediction_received.connect(self.results.update_prediction)
        worker.trade_signal.connect(self._on_trade_executed)
        worker.error.connect(lambda e: self.controls.set_live_status(f"Erreur: {e}"))
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_trade_executed(self, result: dict):
        """Callback quand un trade est exécuté automatiquement."""
        action = result.get("action", "?")
        symbol = result.get("symbol", "?")
        shares = result.get("shares", 0)
        price = result.get("price", 0)

        self.statusbar.showMessage(
            f"Trade execute: {action} {shares}x {symbol} @ ${price:.2f}"
        )
        # Rafraîchir le portefeuille
        self.portfolio_tab.refresh()

    def _save_model(self):
        """Sauvegarde les modèles."""
        try:
            self.ml_engine.save_models()
            self.statusbar.showMessage("Modeles sauvegardes dans ./models/")
            QMessageBox.information(self, "Sauvegarde", "Modeles sauvegardes avec succes !")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de sauvegarde: {e}")

    def _cleanup_worker(self, worker):
        """Nettoie un worker terminé."""
        if worker in self._workers:
            self._workers.remove(worker)

    def closeEvent(self, event):
        """Nettoyage à la fermeture."""
        if self.live_timer:
            self.live_timer.stop()
        self.portfolio_tab.stop_auto_refresh()
        for worker in self._workers:
            worker.quit()
            worker.wait(2000)
        event.accept()
