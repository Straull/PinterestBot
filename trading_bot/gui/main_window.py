"""Fenetre principale de l'application Trading Bot."""

import traceback
import numpy as np
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


def _df_to_arrays(df: pd.DataFrame) -> dict:
    """Convertit un DataFrame en dict de numpy arrays (thread-safe).

    Doit etre appele dans le meme thread que celui qui a cree le DataFrame.
    """
    result = {}
    for col in df.columns:
        arr = df[col].values.copy()
        if arr.dtype.kind == "f":
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        result[col] = arr
    result["_length"] = len(df)
    return result


class DataWorker(QThread):
    """Thread pour le chargement des donnees."""
    finished = pyqtSignal(dict)  # dict de numpy arrays
    raw_df_ready = pyqtSignal(pd.DataFrame)  # pour le ML (stocke en interne)
    error = pyqtSignal(str)

    def __init__(self, market_data: MarketData, symbol: str, period: str):
        super().__init__()
        self.market_data = market_data
        self.symbol = symbol
        self.period = period

    def run(self):
        try:
            df = self.market_data.fetch_historical(self.symbol, self.period)
            # Nettoyer dans ce thread
            df = df.ffill().fillna(0)
            # Emettre le DataFrame brut pour le ML
            self.raw_df_ready.emit(df)
            # Convertir en numpy arrays pour le GUI (thread-safe)
            arrays = _df_to_arrays(df)
            arrays["_symbol"] = self.market_data.symbol or self.symbol
            arrays["_rows"] = len(df)
            self.finished.emit(arrays)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    """Thread pour l'entrainement ML."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str, int, str)
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
    """Thread pour les donnees et previsions en direct."""
    data_received = pyqtSignal(dict)
    prediction_received = pyqtSignal(dict)
    trade_signal = pyqtSignal(dict)
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
            live_data = self.market_data.fetch_live(self.symbol)
            self.data_received.emit(live_data)

            if self.engine.is_trained:
                recent = self.market_data.get_recent_data_for_prediction(self.symbol)
                prediction = self.engine.predict(recent)
                self.prediction_received.emit(prediction)

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
    """Fenetre principale du Trading Bot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot V1 - ML Ensemble")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        self.market_data = MarketData()
        self.ml_engine = MLEngine()
        self.portfolio = PortfolioManager()
        self.strategy = TradingStrategy(self.portfolio)
        self.current_data = None  # DataFrame pour le ML
        self.live_timer = None
        self._workers = []

        self.setStyleSheet(DARK_THEME)

        self._setup_ui()
        self._connect_signals()
        self._setup_statusbar()

        QTimer.singleShot(500, self._startup_portfolio_update)

    def _setup_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Onglet 1 : Analyse & Trading
        analysis_tab = QWidget()
        analysis_layout = QHBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(8, 8, 8, 8)
        analysis_layout.setSpacing(8)

        self.controls = ControlsPanel()
        analysis_layout.addWidget(self.controls)

        self.chart = ChartWidget()
        analysis_layout.addWidget(self.chart, stretch=1)

        self.results = ResultsPanel()
        analysis_layout.addWidget(self.results)

        self.tabs.addTab(analysis_tab, "Analyse & Trading")

        # Onglet 2 : Portefeuille
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
        try:
            self.portfolio_tab.refresh()
            self.statusbar.showMessage(
                f"Portefeuille charge - Cash: {self.portfolio.cash:,.2f}E | "
                f"Pret. Entrez un symbole et chargez les donnees."
            )
        except Exception:
            pass

    def _load_data(self, symbol: str, period: str):
        self.controls.load_btn.setEnabled(False)
        self.statusbar.showMessage(f"Chargement de {symbol}...")

        worker = DataWorker(self.market_data, symbol, period)
        worker.raw_df_ready.connect(self._store_raw_data)
        worker.finished.connect(self._on_data_loaded)
        worker.error.connect(self._on_data_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        worker.error.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _store_raw_data(self, df: pd.DataFrame):
        """Stocke le DataFrame brut pour l'entrainement ML."""
        self.current_data = df

    def _on_data_loaded(self, data: dict):
        """Recoit les donnees sous forme de numpy arrays (thread-safe)."""
        try:
            self.chart.update_chart(data)
        except Exception as e:
            print(f"Erreur graphique: {e}")

        symbol = data.get("_symbol", "?")
        rows = data.get("_rows", 0)
        self.controls.set_data_loaded(rows, symbol)
        self.controls.load_btn.setEnabled(True)
        self.results.clear()

        last_price = 0
        if "Close" in data and len(data["Close"]) > 0:
            last_price = data["Close"][-1]
        self.statusbar.showMessage(
            f"{symbol} - {rows} lignes chargees | Dernier prix: ${last_price:,.2f}"
        )

    def _on_data_error(self, error: str):
        self.controls.load_btn.setEnabled(True)
        self.statusbar.showMessage(f"Erreur: {error}")
        QMessageBox.warning(self, "Erreur de chargement", error)

    def _train_model(self, horizon: int):
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

        try:
            recent = self.market_data.get_recent_data_for_prediction()
            prediction = self.ml_engine.predict(recent)
            self.results.update_prediction(prediction)
        except Exception:
            pass

        device_info = results.get('lstm_device', 'N/A')
        if device_info == "N/A":
            self.statusbar.showMessage(
                f"Entrainement termine | Accuracy ensemble: {results['ensemble_accuracy']:.1%} | "
                f"Mode: XGBoost + LightGBM"
            )
        else:
            self.statusbar.showMessage(
                f"Entrainement termine | Accuracy ensemble: {results['ensemble_accuracy']:.1%} | "
                f"GPU: {device_info}"
            )

    def _on_train_error(self, error: str):
        self.controls.train_btn.setEnabled(True)
        self.controls.set_training_progress(0, "Erreur")
        self.controls.progress_bar.setVisible(False)
        self.statusbar.showMessage("Erreur d'entrainement")
        QMessageBox.critical(self, "Erreur d'entrainement", error)

    def _toggle_live(self, active: bool):
        if active:
            self.live_timer = QTimer()
            self.live_timer.timeout.connect(self._fetch_live)
            self.live_timer.start(30000)
            self._fetch_live()
            self.portfolio_tab.start_auto_refresh(60000)
            self.statusbar.showMessage("Suivi en direct active (maj toutes les 30s)")
        else:
            if self.live_timer:
                self.live_timer.stop()
                self.live_timer = None
            self.portfolio_tab.stop_auto_refresh()
            self.statusbar.showMessage("Suivi en direct desactive")

    def _fetch_live(self):
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
        action = result.get("action", "?")
        symbol = result.get("symbol", "?")
        shares = result.get("shares", 0)
        price = result.get("price", 0)

        self.statusbar.showMessage(
            f"Trade execute: {action} {shares}x {symbol} @ ${price:.2f}"
        )
        self.portfolio_tab.refresh()

    def _save_model(self):
        try:
            self.ml_engine.save_models()
            self.statusbar.showMessage("Modeles sauvegardes dans ./models/")
            QMessageBox.information(self, "Sauvegarde", "Modeles sauvegardes avec succes !")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de sauvegarde: {e}")

    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    def closeEvent(self, event):
        if self.live_timer:
            self.live_timer.stop()
        self.portfolio_tab.stop_auto_refresh()
        for worker in self._workers:
            worker.quit()
            worker.wait(2000)
        event.accept()
