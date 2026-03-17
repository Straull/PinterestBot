"""Widget de graphiques interactifs pour les donnees financieres."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QFrame
from PyQt6.QtCore import Qt

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

        pg.setConfigOptions(antialias=True, background=COLORS["chart_bg"])

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground(COLORS["chart_bg"])
        layout.addWidget(self.graph_widget)

        # Prix
        self.price_plot = self.graph_widget.addPlot(row=0, col=0)
        self._style_plot(self.price_plot, "Prix")
        self.price_plot.setMinimumHeight(250)
        self.price_plot.showGrid(x=True, y=True, alpha=0.15)

        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine))
        self.price_plot.addItem(self.vline, ignoreBounds=True)
        self.price_plot.addItem(self.hline, ignoreBounds=True)

        # Volume
        self.volume_plot = self.graph_widget.addPlot(row=1, col=0)
        self._style_plot(self.volume_plot, "Volume")
        self.volume_plot.setMaximumHeight(80)
        self.volume_plot.setXLink(self.price_plot)

        # RSI
        self.rsi_plot = self.graph_widget.addPlot(row=2, col=0)
        self._style_plot(self.rsi_plot, "RSI")
        self.rsi_plot.setMaximumHeight(100)
        self.rsi_plot.setXLink(self.price_plot)
        self.rsi_plot.setYRange(0, 100)
        self.rsi_plot.addItem(pg.InfiniteLine(pos=70, angle=0,
                              pen=pg.mkPen(COLORS["accent_red"], width=1, style=Qt.PenStyle.DotLine)))
        self.rsi_plot.addItem(pg.InfiniteLine(pos=30, angle=0,
                              pen=pg.mkPen(COLORS["accent_green"], width=1, style=Qt.PenStyle.DotLine)))

        # MACD
        self.macd_plot = self.graph_widget.addPlot(row=3, col=0)
        self._style_plot(self.macd_plot, "MACD")
        self.macd_plot.setMaximumHeight(100)
        self.macd_plot.setXLink(self.price_plot)

        # Mouse tracking
        self.proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved,
                                    rateLimit=60, slot=self._mouse_moved)

    def _style_plot(self, plot, title: str):
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
    def _get(data: dict, key: str) -> np.ndarray:
        """Recupere un array depuis le dict, nettoie NaN/Inf."""
        if key not in data:
            return None
        arr = np.asarray(data[key], dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def update_chart(self, data: dict):
        """Met a jour les graphiques. data = dict de numpy arrays."""
        if not data:
            return

        n = data.get("_length", 0)
        if n == 0:
            return

        self.price_plot.clear()
        self.volume_plot.clear()
        self.rsi_plot.clear()
        self.macd_plot.clear()

        # Re-ajouter crosshair
        self.price_plot.addItem(self.vline, ignoreBounds=True)
        self.price_plot.addItem(self.hline, ignoreBounds=True)

        # Re-ajouter lignes RSI
        self.rsi_plot.addItem(pg.InfiniteLine(pos=70, angle=0,
                              pen=pg.mkPen(COLORS["accent_red"], width=1, style=Qt.PenStyle.DotLine)))
        self.rsi_plot.addItem(pg.InfiniteLine(pos=30, angle=0,
                              pen=pg.mkPen(COLORS["accent_green"], width=1, style=Qt.PenStyle.DotLine)))

        x = np.arange(n)

        # --- Prix ---
        close = self._get(data, "Close")
        if close is None:
            return
        open_p = self._get(data, "Open")
        if open_p is None:
            open_p = close

        self.price_plot.plot(x, close, pen=pg.mkPen(COLORS["accent_blue"], width=2))

        sma20 = self._get(data, "SMA_20")
        if sma20 is not None:
            self.price_plot.plot(x, sma20, pen=pg.mkPen(COLORS["accent_orange"], width=1))

        sma50 = self._get(data, "SMA_50")
        if sma50 is not None:
            self.price_plot.plot(x, sma50, pen=pg.mkPen(COLORS["accent_purple"], width=1))

        # Bollinger Bands
        bb_high = self._get(data, "BB_High")
        bb_low = self._get(data, "BB_Low")
        if bb_high is not None and bb_low is not None:
            bb_pen = pg.mkPen(COLORS["text_muted"], width=1, style=Qt.PenStyle.DashLine)
            self.price_plot.plot(x, bb_high, pen=bb_pen)
            self.price_plot.plot(x, bb_low, pen=bb_pen)

            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x, bb_high),
                pg.PlotDataItem(x, bb_low),
                brush=pg.mkBrush(88, 166, 255, 15),
            )
            self.price_plot.addItem(fill)

        # --- Volume (barres simples, une seule couleur pour eviter le bug brushes) ---
        volume = self._get(data, "Volume")
        if volume is not None:
            bar = pg.BarGraphItem(
                x=x, height=volume, width=0.6,
                brush=pg.mkBrush(COLORS["accent_blue"] + "60"),
                pen=None,
            )
            self.volume_plot.addItem(bar)

        # --- RSI ---
        rsi = self._get(data, "RSI")
        if rsi is not None:
            self.rsi_plot.plot(x, rsi, pen=pg.mkPen(COLORS["accent_purple"], width=1.5))
            self.rsi_plot.setYRange(0, 100)

        # --- MACD ---
        macd = self._get(data, "MACD")
        if macd is not None:
            signal = self._get(data, "MACD_Signal")
            if signal is None:
                signal = np.zeros_like(macd)
            hist = self._get(data, "MACD_Hist")
            if hist is None:
                hist = macd - signal

            self.macd_plot.plot(x, macd, pen=pg.mkPen(COLORS["accent_blue"], width=1.5))
            self.macd_plot.plot(x, signal, pen=pg.mkPen(COLORS["accent_orange"], width=1.5))

            # Histogramme positif et negatif separes
            hist_pos = np.where(hist >= 0, hist, 0)
            hist_neg = np.where(hist < 0, hist, 0)

            if np.any(hist_pos > 0):
                bar_pos = pg.BarGraphItem(
                    x=x, height=hist_pos, width=0.5,
                    brush=pg.mkBrush(COLORS["chart_up"]),
                    pen=None,
                )
                self.macd_plot.addItem(bar_pos)

            if np.any(hist_neg < 0):
                bar_neg = pg.BarGraphItem(
                    x=x, height=hist_neg, width=0.5,
                    brush=pg.mkBrush(COLORS["chart_down"]),
                    pen=None,
                )
                self.macd_plot.addItem(bar_neg)

        self.price_plot.autoRange()
