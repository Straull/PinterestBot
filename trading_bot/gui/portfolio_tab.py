"""Onglet Portefeuille - Affichage des positions, transactions et courbe de valeur."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QPushButton, QSpinBox, QComboBox, QProgressBar, QGroupBox,
    QAbstractItemView, QSizePolicy, QMessageBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg

import pandas as pd
from datetime import datetime

from trading_bot.gui.styles import COLORS
from trading_bot.portfolio.manager import PortfolioManager
from trading_bot.portfolio.strategy import TradingStrategy
from trading_bot.portfolio.universe import CTO_UNIVERSE, get_all_sectors, get_all_markets


class PortfolioWorker(QThread):
    """Thread pour les opérations portefeuille (valuation, trading)."""
    valuation_ready = pyqtSignal(dict)
    trade_executed = pyqtSignal(dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, portfolio: PortfolioManager):
        super().__init__()
        self.portfolio = portfolio
        self._task = None
        self._args = {}

    def update_valuation(self):
        self._task = "valuation"
        self.start()

    def run(self):
        try:
            if self._task == "valuation":
                self.status.emit("Mise à jour des prix...")
                result = self.portfolio.update_valuation()
                self.valuation_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class PortfolioTab(QWidget):
    """Onglet complet du portefeuille."""

    def __init__(self, portfolio: PortfolioManager, strategy: TradingStrategy, parent=None):
        super().__init__(parent)
        self.portfolio = portfolio
        self.strategy = strategy
        self.worker = PortfolioWorker(portfolio)
        self.worker.valuation_ready.connect(self._on_valuation)
        self.worker.error.connect(self._on_error)
        self.worker.status.connect(self._on_status)

        self._setup_ui()

        # Timer de rafraîchissement
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Header avec résumé ----
        header = QFrame()
        header.setObjectName("card")
        header_layout = QHBoxLayout(header)
        header_layout.setSpacing(24)

        # Valeur totale
        total_group = QVBoxLayout()
        total_group.setSpacing(2)
        lbl = QLabel("VALEUR TOTALE")
        lbl.setObjectName("muted")
        total_group.addWidget(lbl)
        self.total_value_label = QLabel("--")
        self.total_value_label.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: {COLORS['text']};"
        )
        total_group.addWidget(self.total_value_label)
        header_layout.addLayout(total_group)

        # P&L
        pnl_group = QVBoxLayout()
        pnl_group.setSpacing(2)
        lbl2 = QLabel("PROFIT / PERTE")
        lbl2.setObjectName("muted")
        pnl_group.addWidget(lbl2)
        self.pnl_label = QLabel("--")
        self.pnl_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        pnl_group.addWidget(self.pnl_label)
        header_layout.addLayout(pnl_group)

        # Cash
        cash_group = QVBoxLayout()
        cash_group.setSpacing(2)
        lbl3 = QLabel("CASH DISPONIBLE")
        lbl3.setObjectName("muted")
        cash_group.addWidget(lbl3)
        self.cash_label = QLabel("--")
        self.cash_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        cash_group.addWidget(self.cash_label)
        header_layout.addLayout(cash_group)

        # Positions value
        pos_group = QVBoxLayout()
        pos_group.setSpacing(2)
        lbl4 = QLabel("EN ACTIONS")
        lbl4.setObjectName("muted")
        pos_group.addWidget(lbl4)
        self.positions_value_label = QLabel("--")
        self.positions_value_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        pos_group.addWidget(self.positions_value_label)
        header_layout.addLayout(pos_group)

        # Stats
        stats_group = QVBoxLayout()
        stats_group.setSpacing(2)
        lbl5 = QLabel("TRADES")
        lbl5.setObjectName("muted")
        stats_group.addWidget(lbl5)
        self.trades_label = QLabel("--")
        self.trades_label.setStyleSheet("font-size: 18px;")
        stats_group.addWidget(self.trades_label)
        header_layout.addLayout(stats_group)

        header_layout.addStretch()

        # Capital de départ
        capital_group = QVBoxLayout()
        capital_group.setSpacing(4)
        cap_lbl = QLabel("CAPITAL DE DEPART")
        cap_lbl.setObjectName("muted")
        capital_group.addWidget(cap_lbl)

        cap_row = QHBoxLayout()
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1_000_000)
        self.capital_spin.setSingleStep(1000)
        self.capital_spin.setPrefix("")
        self.capital_spin.setSuffix(" €")
        self.capital_spin.setDecimals(0)
        self.capital_spin.setValue(self.portfolio.initial_cash)
        cap_row.addWidget(self.capital_spin)

        self.apply_capital_btn = QPushButton("Appliquer")
        self.apply_capital_btn.setObjectName("primary")
        self.apply_capital_btn.clicked.connect(self._apply_capital)
        cap_row.addWidget(self.apply_capital_btn)
        capital_group.addLayout(cap_row)
        header_layout.addLayout(capital_group)

        # Boutons
        btn_group = QVBoxLayout()
        self.refresh_btn = QPushButton("Rafraichir")
        self.refresh_btn.setObjectName("primary")
        self.refresh_btn.clicked.connect(self.refresh)
        btn_group.addWidget(self.refresh_btn)

        self.reset_btn = QPushButton("Reset Portefeuille")
        self.reset_btn.setObjectName("danger")
        self.reset_btn.clicked.connect(self._reset_portfolio)
        btn_group.addWidget(self.reset_btn)
        header_layout.addLayout(btn_group)

        layout.addWidget(header)

        # ---- Splitter principal : graphique + tables ----
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Graphique de valeur
        chart_frame = QFrame()
        chart_frame.setObjectName("card")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(8, 8, 8, 8)

        chart_title = QLabel("COURBE DE VALEUR DU PORTEFEUILLE")
        chart_title.setObjectName("section")
        chart_layout.addWidget(chart_title)

        self.value_chart = pg.PlotWidget()
        self.value_chart.setBackground(COLORS["bg_dark"])
        self.value_chart.showGrid(x=True, y=True, alpha=0.15)
        self.value_chart.setLabel("left", "Valeur (€)")
        self.value_chart.setLabel("bottom", "")
        self.value_chart.addLegend(offset=(10, 10))

        # Ligne de référence (capital initial)
        self.ref_line = pg.InfiniteLine(
            pos=self.portfolio.initial_cash,
            angle=0,
            pen=pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine),
            label=f"Capital initial: {self.portfolio.initial_cash:,.0f}€",
            labelOpts={"color": COLORS["text_muted"], "position": 0.05},
        )
        self.value_chart.addItem(self.ref_line)

        chart_layout.addWidget(self.value_chart)
        splitter.addWidget(chart_frame)

        # Partie basse : positions + transactions côte à côte
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table des positions
        pos_frame = QFrame()
        pos_frame.setObjectName("card")
        pos_layout = QVBoxLayout(pos_frame)
        pos_layout.setContentsMargins(8, 8, 8, 8)

        pos_title = QLabel("POSITIONS OUVERTES")
        pos_title.setObjectName("section")
        pos_layout.addWidget(pos_title)

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(8)
        self.positions_table.setHorizontalHeaderLabels([
            "Symbole", "Nom", "Quantité", "PRU", "Prix actuel", "Valeur", "P&L", "P&L %"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.positions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.positions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.positions_table.verticalHeader().setVisible(False)
        self.positions_table.setAlternatingRowColors(True)
        pos_layout.addWidget(self.positions_table)

        bottom_splitter.addWidget(pos_frame)

        # Table des transactions
        tx_frame = QFrame()
        tx_frame.setObjectName("card")
        tx_layout = QVBoxLayout(tx_frame)
        tx_layout.setContentsMargins(8, 8, 8, 8)

        tx_title = QLabel("HISTORIQUE DES TRANSACTIONS")
        tx_title.setObjectName("section")
        tx_layout.addWidget(tx_title)

        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(7)
        self.transactions_table.setHorizontalHeaderLabels([
            "Date", "Action", "Symbole", "Quantité", "Prix", "Total", "Raison"
        ])
        self.transactions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.transactions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.transactions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.transactions_table.verticalHeader().setVisible(False)
        self.transactions_table.setAlternatingRowColors(True)
        tx_layout.addWidget(self.transactions_table)

        bottom_splitter.addWidget(tx_frame)

        splitter.addWidget(bottom_splitter)
        splitter.setSizes([350, 350])

        layout.addWidget(splitter)

        # Status
        self.status_label = QLabel("")
        self.status_label.setObjectName("muted")
        layout.addWidget(self.status_label)

    def refresh(self):
        """Rafraîchit les données du portefeuille."""
        if not self.worker.isRunning():
            self.refresh_btn.setEnabled(False)
            self.worker.update_valuation()

    def start_auto_refresh(self, interval_ms: int = 60000):
        """Démarre le rafraîchissement automatique."""
        self.refresh_timer.start(interval_ms)
        self.refresh()

    def stop_auto_refresh(self):
        """Arrête le rafraîchissement automatique."""
        self.refresh_timer.stop()

    def _on_valuation(self, data: dict):
        """Callback quand la valuation est prête."""
        self.refresh_btn.setEnabled(True)

        # Header
        total = data["total_value"]
        pnl = data["profit_loss"]
        pnl_pct = data["profit_loss_pct"]
        cash = data["cash"]
        pos_val = data["positions_value"]

        self.total_value_label.setText(f"{total:,.2f} €")

        sign = "+" if pnl >= 0 else ""
        color = COLORS["accent_green"] if pnl >= 0 else COLORS["accent_red"]
        self.pnl_label.setText(f"{sign}{pnl:,.2f} € ({sign}{pnl_pct:.2f}%)")
        self.pnl_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {color};")

        self.cash_label.setText(f"{cash:,.2f} €")
        self.positions_value_label.setText(f"{pos_val:,.2f} €")

        stats = self.portfolio.get_summary_stats()
        self.trades_label.setText(f"{stats['total_trades']} ({stats['total_buys']}A / {stats['total_sells']}V)")

        # Positions table
        positions = data.get("positions", [])
        self.positions_table.setRowCount(len(positions))
        for i, pos in enumerate(positions):
            symbol = pos["symbol"]
            info = CTO_UNIVERSE.get(symbol, {"name": symbol, "sector": ""})

            items = [
                symbol,
                info["name"],
                f"{pos['shares']:.0f}",
                f"{pos['avg_cost']:.2f} €",
                f"{pos['current_price']:.2f} €",
                f"{pos['value']:,.2f} €",
                f"{'+' if pos['pnl'] >= 0 else ''}{pos['pnl']:,.2f} €",
                f"{'+' if pos['pnl_pct'] >= 0 else ''}{pos['pnl_pct']:.2f}%",
            ]

            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j >= 6:  # P&L columns
                    c = QColor(COLORS["accent_green"]) if pos["pnl"] >= 0 else QColor(COLORS["accent_red"])
                    item.setForeground(c)
                self.positions_table.setItem(i, j, item)

        # Transactions table
        tx_df = self.portfolio.get_transactions(limit=200)
        self.transactions_table.setRowCount(len(tx_df))
        for i, (_, tx) in enumerate(tx_df.iterrows()):
            ts = tx["timestamp"][:19].replace("T", " ")
            action = tx["action"]
            items = [
                ts,
                action,
                tx["symbol"],
                f"{tx['shares']:.0f}",
                f"{tx['price']:.2f} €",
                f"{tx['total']:,.2f} €",
                tx.get("reason", ""),
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j == 1:
                    c = QColor(COLORS["accent_green"]) if action == "BUY" else QColor(COLORS["accent_red"])
                    item.setForeground(c)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.transactions_table.setItem(i, j, item)

        # Value chart
        self._update_value_chart()

        self.status_label.setText(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")

    def _update_value_chart(self):
        """Met à jour le graphique de courbe de valeur."""
        history = self.portfolio.get_value_history()
        if history.empty:
            return

        self.value_chart.clear()

        # Re-ajouter la ligne de référence
        self.ref_line = pg.InfiniteLine(
            pos=self.portfolio.initial_cash,
            angle=0,
            pen=pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine),
        )
        self.value_chart.addItem(self.ref_line)

        # Convertir timestamps en indices numériques
        x = list(range(len(history)))
        y_total = history["total_value"].values
        y_cash = history["cash"].values
        y_pos = history["positions_value"].values

        # Courbe principale
        pen_total = pg.mkPen(COLORS["accent_blue"], width=2.5)
        self.value_chart.plot(x, y_total, pen=pen_total, name="Valeur totale")

        # Fill zone profit/perte
        fill_brush_up = pg.mkBrush(COLORS["accent_green"] + "30")
        fill_brush_down = pg.mkBrush(COLORS["accent_red"] + "30")

        initial = self.portfolio.initial_cash
        ref_line_y = [initial] * len(x)

        # Zone au-dessus du capital initial = profit
        fill_up = pg.FillBetweenItem(
            pg.PlotCurveItem(x, y_total),
            pg.PlotCurveItem(x, ref_line_y),
            brush=fill_brush_up,
        )
        self.value_chart.addItem(fill_up)

        # Courbe cash
        pen_cash = pg.mkPen(COLORS["accent_orange"], width=1, style=Qt.PenStyle.DashLine)
        self.value_chart.plot(x, y_cash, pen=pen_cash, name="Cash")

        # Labels d'axe X avec dates
        if len(history) > 1:
            ticks = []
            step = max(1, len(history) // 10)
            for i in range(0, len(history), step):
                ts = history.iloc[i]["timestamp"]
                if hasattr(ts, "strftime"):
                    ticks.append((i, ts.strftime("%d/%m %H:%M")))
                else:
                    ticks.append((i, str(ts)[:16]))

            ax = self.value_chart.getAxis("bottom")
            ax.setTicks([ticks])

    def _on_error(self, error: str):
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"Erreur : {error}")
        self.status_label.setStyleSheet(f"color: {COLORS['accent_red']};")

    def _on_status(self, msg: str):
        self.status_label.setText(msg)

    def _apply_capital(self):
        """Applique un nouveau capital de départ (reset le portefeuille)."""
        new_capital = self.capital_spin.value()
        reply = QMessageBox.question(
            self, "Changer le capital",
            f"Cela va réinitialiser le portefeuille avec {new_capital:,.0f} €.\n\n"
            "Toutes les positions et l'historique seront perdus.\nContinuer ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.portfolio.reset(initial_cash=new_capital)
            self.refresh()

    def _reset_portfolio(self):
        """Remet le portefeuille à zéro."""
        reply = QMessageBox.question(
            self, "Reset Portefeuille",
            "Réinitialiser le portefeuille ?\n"
            "Toutes les positions et l'historique seront perdus.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.portfolio.reset(initial_cash=self.capital_spin.value())
            self.refresh()

    def record_trade(self, trade_result: dict):
        """Enregistre un trade et rafraîchit l'affichage."""
        self.refresh()
