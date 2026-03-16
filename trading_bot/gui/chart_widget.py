"""Widget de graphiques interactifs pour les données financières."""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from trading_bot.gui.styles import COLORS


class ChartWidget(QFrame):
    """Widget de graphiques financiers avec prix, volume, RSI et MACD."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # Configuration pyqtgraph
        pg.setConfigOptions(antialias=True, background=COLORS["chart_bg"])

        # Widget graphique
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground(COLORS["chart_bg"])
        layout.addWidget(self.graph_widget)

        # Graphique principal (prix)
        self.price_plot = self.graph_widget.addPlot(row=0, col=0)
        self._style_plot(self.price_plot, "Prix")
        self.price_plot.setMinimumHeight(250)
        self.price_plot.showGrid(x=True, y=True, alpha=0.15)

        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine))
        self.price_plot.addItem(self.vline, ignoreBounds=True)
        self.price_plot.addItem(self.hline, ignoreBounds=True)

        # Graphique volume
        self.volume_plot = self.graph_widget.addPlot(row=1, col=0)
        self._style_plot(self.volume_plot, "Volume")
        self.volume_plot.setMaximumHeight(80)
        self.volume_plot.setXLink(self.price_plot)

        # Graphique RSI
        self.rsi_plot = self.graph_widget.addPlot(row=2, col=0)
        self._style_plot(self.rsi_plot, "RSI")
        self.rsi_plot.setMaximumHeight(100)
        self.rsi_plot.setXLink(self.price_plot)
        self.rsi_plot.setYRange(0, 100)
        # Lignes de référence RSI
        self.rsi_plot.addItem(pg.InfiniteLine(pos=70, angle=0, pen=pg.mkPen(COLORS["accent_red"], width=1, style=Qt.PenStyle.DotLine)))
        self.rsi_plot.addItem(pg.InfiniteLine(pos=30, angle=0, pen=pg.mkPen(COLORS["accent_green"], width=1, style=Qt.PenStyle.DotLine)))

        # Graphique MACD
        self.macd_plot = self.graph_widget.addPlot(row=3, col=0)
        self._style_plot(self.macd_plot, "MACD")
        self.macd_plot.setMaximumHeight(100)
        self.macd_plot.setXLink(self.price_plot)

        # Mouse tracking
        self.proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def _style_plot(self, plot, title: str):
        """Applique le style à un plot."""
        plot.setLabel("left", title, color=COLORS["text_secondary"], size="10pt")
        plot.getAxis("left").setPen(pg.mkPen(COLORS["border"]))
        plot.getAxis("bottom").setPen(pg.mkPen(COLORS["border"]))
        plot.getAxis("left").setTextPen(pg.mkPen(COLORS["text_muted"]))
        plot.getAxis("bottom").setTextPen(pg.mkPen(COLORS["text_muted"]))
        plot.setContentsMargins(0, 0, 0, 0)

    def _mouse_moved(self, evt):
        pos = evt[0]
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            self.vline.setPos(mouse_point.x())
            self.hline.setPos(mouse_point.y())

    def update_chart(self, df: pd.DataFrame):
        """Met à jour tous les graphiques avec les nouvelles données."""
        if df.empty:
            return

        self.price_plot.clear()
        self.volume_plot.clear()
        self.rsi_plot.clear()
        self.macd_plot.clear()

        # Réajouter crosshair
        self.price_plot.addItem(self.vline, ignoreBounds=True)
        self.price_plot.addItem(self.hline, ignoreBounds=True)

        # Réajouter lignes RSI
        self.rsi_plot.addItem(pg.InfiniteLine(pos=70, angle=0, pen=pg.mkPen(COLORS["accent_red"], width=1, style=Qt.PenStyle.DotLine)))
        self.rsi_plot.addItem(pg.InfiniteLine(pos=30, angle=0, pen=pg.mkPen(COLORS["accent_green"], width=1, style=Qt.PenStyle.DotLine)))

        x = np.arange(len(df))

        # --- Prix avec chandeliers simplifiés ---
        close = df["Close"].values
        open_p = df["Open"].values if "Open" in df.columns else close

        # Ligne de prix principale
        self.price_plot.plot(x, close, pen=pg.mkPen(COLORS["accent_blue"], width=2))

        # Moyennes mobiles
        if "SMA_20" in df.columns:
            sma20 = df["SMA_20"].values
            self.price_plot.plot(x, sma20, pen=pg.mkPen(COLORS["accent_orange"], width=1))
        if "SMA_50" in df.columns:
            sma50 = df["SMA_50"].values
            self.price_plot.plot(x, sma50, pen=pg.mkPen(COLORS["accent_purple"], width=1))

        # Bollinger Bands
        if "BB_High" in df.columns and "BB_Low" in df.columns:
            bb_high = df["BB_High"].values
            bb_low = df["BB_Low"].values
            bb_pen = pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine)
            self.price_plot.plot(x, bb_high, pen=bb_pen)
            self.price_plot.plot(x, bb_low, pen=bb_pen)

            # Fill entre les bandes
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x, bb_high),
                pg.PlotDataItem(x, bb_low),
                brush=pg.mkBrush(88, 166, 255, 15),
            )
            self.price_plot.addItem(fill)

        # --- Volume ---
        if "Volume" in df.columns:
            volume = df["Volume"].values
            colors = [
                QColor(COLORS["chart_up"]) if close[i] >= open_p[i]
                else QColor(COLORS["chart_down"])
                for i in range(len(close))
            ]
            brushes = [pg.mkBrush(c.red(), c.green(), c.blue(), 120) for c in colors]
            bar = pg.BarGraphItem(x=x, height=volume, width=0.6, brushes=brushes)
            self.volume_plot.addItem(bar)

        # --- RSI ---
        if "RSI" in df.columns:
            rsi = df["RSI"].values
            rsi_pen = pg.mkPen(COLORS["accent_purple"], width=1.5)
            self.rsi_plot.plot(x, rsi, pen=rsi_pen)
            self.rsi_plot.setYRange(0, 100)

        # --- MACD ---
        if "MACD" in df.columns:
            macd = df["MACD"].values
            signal = df["MACD_Signal"].values if "MACD_Signal" in df.columns else np.zeros_like(macd)
            hist = df["MACD_Hist"].values if "MACD_Hist" in df.columns else macd - signal

            self.macd_plot.plot(x, macd, pen=pg.mkPen(COLORS["accent_blue"], width=1.5))
            self.macd_plot.plot(x, signal, pen=pg.mkPen(COLORS["accent_orange"], width=1.5))

            # Histogramme MACD
            hist_colors = [
                pg.mkBrush(COLORS["chart_up"]) if h >= 0
                else pg.mkBrush(COLORS["chart_down"])
                for h in hist
            ]
            bar = pg.BarGraphItem(x=x, height=hist, width=0.5, brushes=hist_colors)
            self.macd_plot.addItem(bar)

        self.price_plot.autoRange()
