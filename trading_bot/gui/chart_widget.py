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
        plot.setLabel("left", title, color=COLORS["text_secondary"])
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

    @staticmethod
    def _clean(arr):
        """Remplace NaN et Inf par 0 pour éviter les crashs pyqtgraph."""
        arr = np.array(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def update_chart(self, df: pd.DataFrame):
        """Met à jour tous les graphiques avec les nouvelles données."""
        if df is None or df.empty:
            return

        # Nettoyer les NaN résiduels
        df = df.copy()
        df = df.fillna(method="ffill").fillna(0)

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

        # --- Prix ---
        close = self._clean(df["Close"].values)
        open_p = self._clean(df["Open"].values) if "Open" in df.columns else close

        self.price_plot.plot(x, close, pen=pg.mkPen(COLORS["accent_blue"], width=2))

        # Moyennes mobiles
        if "SMA_20" in df.columns:
            self.price_plot.plot(x, self._clean(df["SMA_20"].values), pen=pg.mkPen(COLORS["accent_orange"], width=1))
        if "SMA_50" in df.columns:
            self.price_plot.plot(x, self._clean(df["SMA_50"].values), pen=pg.mkPen(COLORS["accent_purple"], width=1))

        # Bollinger Bands
        if "BB_High" in df.columns and "BB_Low" in df.columns:
            bb_high = self._clean(df["BB_High"].values)
            bb_low = self._clean(df["BB_Low"].values)
            bb_pen = pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine)
            self.price_plot.plot(x, bb_high, pen=bb_pen)
            self.price_plot.plot(x, bb_low, pen=bb_pen)

            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x, bb_high),
                pg.PlotDataItem(x, bb_low),
                brush=pg.mkBrush(88, 166, 255, 15),
            )
            self.price_plot.addItem(fill)

        # --- Volume ---
        if "Volume" in df.columns:
            volume = self._clean(df["Volume"].values)
            up_brush = pg.mkBrush(COLORS["chart_up"] + "78")
            down_brush = pg.mkBrush(COLORS["chart_down"] + "78")
            brushes = [
                up_brush if close[i] >= open_p[i] else down_brush
                for i in range(len(close))
            ]
            bar = pg.BarGraphItem(x=x, height=volume, width=0.6, brushes=brushes)
            self.volume_plot.addItem(bar)

        # --- RSI ---
        if "RSI" in df.columns:
            rsi = self._clean(df["RSI"].values)
            self.rsi_plot.plot(x, rsi, pen=pg.mkPen(COLORS["accent_purple"], width=1.5))
            self.rsi_plot.setYRange(0, 100)

        # --- MACD ---
        if "MACD" in df.columns:
            macd = self._clean(df["MACD"].values)
            signal = self._clean(df["MACD_Signal"].values) if "MACD_Signal" in df.columns else np.zeros_like(macd)
            hist = self._clean(df["MACD_Hist"].values) if "MACD_Hist" in df.columns else macd - signal

            self.macd_plot.plot(x, macd, pen=pg.mkPen(COLORS["accent_blue"], width=1.5))
            self.macd_plot.plot(x, signal, pen=pg.mkPen(COLORS["accent_orange"], width=1.5))

            up_brush = pg.mkBrush(COLORS["chart_up"])
            down_brush = pg.mkBrush(COLORS["chart_down"])
            hist_colors = [up_brush if h >= 0 else down_brush for h in hist]
            bar = pg.BarGraphItem(x=x, height=hist, width=0.5, brushes=hist_colors)
            self.macd_plot.addItem(bar)

        self.price_plot.autoRange()
