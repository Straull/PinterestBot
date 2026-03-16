"""Module de récupération des données de marché (historiques et temps réel)."""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta


class MarketData:
    """Gère le téléchargement et le traitement des données de marché."""

    PERIODS = {
        "1 mois": "1mo",
        "3 mois": "3mo",
        "6 mois": "6mo",
        "1 an": "1y",
        "2 ans": "2y",
        "5 ans": "5y",
        "10 ans": "10y",
        "Max": "max",
    }

    def __init__(self):
        self.symbol = None
        self.data = None
        self.ticker = None

    def fetch_historical(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Télécharge les données historiques depuis Yahoo Finance."""
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self.data = self.ticker.history(period=period, auto_adjust=True)

        if self.data.empty:
            raise ValueError(f"Aucune donnée trouvée pour {self.symbol}")

        self.data = self._add_technical_indicators(self.data)
        return self.data

    def fetch_live(self, symbol: str = None) -> dict:
        """Récupère les données en temps réel."""
        if symbol:
            self.symbol = symbol.upper()
            self.ticker = yf.Ticker(self.symbol)

        if not self.ticker:
            raise ValueError("Aucun symbole défini")

        info = self.ticker.fast_info
        hist_1d = self.ticker.history(period="5d", interval="1m")

        if hist_1d.empty:
            raise ValueError("Impossible de récupérer les données en direct")

        latest = hist_1d.iloc[-1]
        prev_close = info.get("previousClose", latest["Close"])

        return {
            "symbol": self.symbol,
            "price": float(latest["Close"]),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "volume": int(latest["Volume"]),
            "prev_close": float(prev_close) if prev_close else float(latest["Close"]),
            "change": float(latest["Close"] - prev_close) if prev_close else 0,
            "change_pct": float((latest["Close"] - prev_close) / prev_close * 100) if prev_close else 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_recent_data_for_prediction(self, symbol: str = None, days: int = 60) -> pd.DataFrame:
        """Récupère les données récentes avec indicateurs pour la prédiction."""
        if symbol:
            self.symbol = symbol.upper()
            self.ticker = yf.Ticker(self.symbol)

        data = self.ticker.history(period=f"{days + 50}d")
        if data.empty:
            raise ValueError("Impossible de récupérer les données récentes")

        data = self._add_technical_indicators(data)
        return data.tail(days)

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques au DataFrame."""
        df = df.copy()

        # Moyennes mobiles
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
        df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)

        # RSI
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
        df["BB_High"] = bollinger.bollinger_hband()
        df["BB_Low"] = bollinger.bollinger_lband()
        df["BB_Mid"] = bollinger.bollinger_mavg()

        # ATR (Average True Range)
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

        # Volume moyen
        df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()

        # Rendements
        df["Return_1d"] = df["Close"].pct_change(1)
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_10d"] = df["Close"].pct_change(10)

        # Volatilité
        df["Volatility"] = df["Return_1d"].rolling(window=20).std()

        df.dropna(inplace=True)
        return df

    def get_info(self) -> dict:
        """Retourne les informations de base sur le ticker."""
        if not self.ticker:
            return {}
        try:
            info = self.ticker.info
            return {
                "name": info.get("longName", self.symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "currency": info.get("currency", "USD"),
            }
        except Exception:
            return {"name": self.symbol, "sector": "N/A", "industry": "N/A", "market_cap": "N/A", "currency": "USD"}
