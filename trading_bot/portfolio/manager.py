"""Gestionnaire de portefeuille avec persistance SQLite.

Gère les positions, transactions, et l'historique de valeur du portefeuille.
Persiste toutes les données dans une base SQLite locale pour survivre aux redémarrages.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "portfolio.db")


class PortfolioManager:
    """Gère un portefeuille virtuel avec persistance SQLite."""

    def __init__(self, db_path: str = None, initial_cash: float = 10000.0):
        self.db_path = db_path or os.path.abspath(DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

        # Si le portefeuille n'existe pas encore, l'initialiser
        if self._get_meta("initialized") is None:
            self._set_meta("initialized", "true")
            self._set_meta("initial_cash", str(initial_cash))
            self._set_meta("cash", str(initial_cash))
            self._set_meta("created_at", datetime.now().isoformat())
            self._snapshot(initial_cash, 0.0)

    def _init_db(self):
        """Crée les tables si elles n'existent pas."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                shares REAL NOT NULL DEFAULT 0,
                avg_cost REAL NOT NULL DEFAULT 0,
                first_buy_date TEXT,
                last_trade_date TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                commission REAL NOT NULL DEFAULT 0,
                reason TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS value_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                profit_loss REAL NOT NULL,
                profit_loss_pct REAL NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _get_meta(self, key: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()

    @property
    def cash(self) -> float:
        return float(self._get_meta("cash") or 0)

    @cash.setter
    def cash(self, value: float):
        self._set_meta("cash", str(value))

    @property
    def initial_cash(self) -> float:
        return float(self._get_meta("initial_cash") or 10000)

    # ---- Opérations de trading ----

    def buy(self, symbol: str, shares: float, price: float,
            commission: float = 0, reason: str = "") -> dict:
        """Achète des actions. Retourne les détails de la transaction."""
        total = shares * price + commission

        if total > self.cash:
            raise ValueError(
                f"Fonds insuffisants: {total:.2f}€ requis, {self.cash:.2f}€ disponible"
            )

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Mise à jour position
        c.execute("SELECT shares, avg_cost FROM positions WHERE symbol = ?", (symbol,))
        row = c.fetchone()
        now = datetime.now().isoformat()

        if row:
            old_shares, old_avg = row
            new_shares = old_shares + shares
            new_avg = (old_shares * old_avg + shares * price) / new_shares
            c.execute(
                "UPDATE positions SET shares=?, avg_cost=?, last_trade_date=? WHERE symbol=?",
                (new_shares, new_avg, now, symbol),
            )
        else:
            c.execute(
                "INSERT INTO positions (symbol, shares, avg_cost, first_buy_date, last_trade_date) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, shares, price, now, now),
            )

        # Transaction
        c.execute(
            "INSERT INTO transactions (timestamp, symbol, action, shares, price, total, commission, reason) "
            "VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?)",
            (now, symbol, shares, price, total, commission, reason),
        )

        # Cash
        new_cash = self.cash - total
        conn.commit()
        conn.close()
        self.cash = new_cash

        return {
            "action": "BUY",
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "total": total,
            "commission": commission,
            "remaining_cash": new_cash,
        }

    def sell(self, symbol: str, shares: float, price: float,
             commission: float = 0, reason: str = "") -> dict:
        """Vend des actions."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT shares, avg_cost FROM positions WHERE symbol = ?", (symbol,))
        row = c.fetchone()

        if not row or row[0] < shares:
            conn.close()
            available = row[0] if row else 0
            raise ValueError(
                f"Pas assez d'actions {symbol}: {shares} demandé, {available} disponible"
            )

        old_shares, avg_cost = row
        total = shares * price - commission
        now = datetime.now().isoformat()

        new_shares = old_shares - shares
        if new_shares < 0.001:  # Considérer comme vendu entièrement
            c.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        else:
            c.execute(
                "UPDATE positions SET shares=?, last_trade_date=? WHERE symbol=?",
                (new_shares, now, symbol),
            )

        # Transaction
        profit = (price - avg_cost) * shares - commission
        c.execute(
            "INSERT INTO transactions (timestamp, symbol, action, shares, price, total, commission, reason) "
            "VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?)",
            (now, symbol, shares, price, total, commission, reason),
        )

        new_cash = self.cash + total
        conn.commit()
        conn.close()
        self.cash = new_cash

        return {
            "action": "SELL",
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "total": total,
            "profit": profit,
            "commission": commission,
            "remaining_cash": new_cash,
        }

    # ---- Lecture du portefeuille ----

    def get_positions(self) -> pd.DataFrame:
        """Retourne les positions actuelles."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM positions WHERE shares > 0.001 ORDER BY symbol", conn
        )
        conn.close()
        return df

    def get_transactions(self, limit: int = 100) -> pd.DataFrame:
        """Retourne l'historique des transactions."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        conn.close()
        return df

    def get_value_history(self) -> pd.DataFrame:
        """Retourne l'historique de valeur du portefeuille."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM value_history ORDER BY timestamp", conn
        )
        conn.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_current_prices(self, symbols: list) -> dict:
        """Récupère les prix actuels via yfinance."""
        prices = {}
        if not symbols:
            return prices
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                prices[symbol] = info.get("lastPrice", info.get("last_price", 0))
            except Exception:
                prices[symbol] = 0
        return prices

    def update_valuation(self) -> dict:
        """Met à jour la valeur du portefeuille avec les prix actuels.

        Appelé au démarrage et périodiquement.
        """
        positions = self.get_positions()
        if positions.empty:
            total_positions = 0.0
            details = []
        else:
            symbols = positions["symbol"].tolist()
            prices = self.get_current_prices(symbols)

            details = []
            total_positions = 0.0
            for _, pos in positions.iterrows():
                symbol = pos["symbol"]
                current_price = prices.get(symbol, 0)
                value = pos["shares"] * current_price
                cost = pos["shares"] * pos["avg_cost"]
                pnl = value - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0

                details.append({
                    "symbol": symbol,
                    "shares": pos["shares"],
                    "avg_cost": pos["avg_cost"],
                    "current_price": current_price,
                    "value": value,
                    "cost": cost,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                total_positions += value

        cash = self.cash
        total_value = cash + total_positions
        initial = self.initial_cash
        profit_loss = total_value - initial
        profit_loss_pct = (profit_loss / initial * 100) if initial > 0 else 0

        # Sauvegarder le snapshot
        self._snapshot(total_value, profit_loss)

        return {
            "cash": cash,
            "positions_value": total_positions,
            "total_value": total_value,
            "initial_cash": initial,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
            "positions": details,
        }

    def _snapshot(self, total_value: float, profit_loss: float):
        """Enregistre un point dans l'historique de valeur."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cash = self.cash
        positions_value = total_value - cash
        initial = self.initial_cash
        pnl_pct = (profit_loss / initial * 100) if initial > 0 else 0

        c.execute(
            "INSERT INTO value_history (timestamp, total_value, cash, positions_value, profit_loss, profit_loss_pct) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), total_value, cash, positions_value, profit_loss, pnl_pct),
        )
        conn.commit()
        conn.close()

    def reset(self, initial_cash: float = 10000.0):
        """Remet le portefeuille à zéro."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM positions")
        c.execute("DELETE FROM transactions")
        c.execute("DELETE FROM value_history")
        c.execute("DELETE FROM meta")
        conn.commit()
        conn.close()
        self._set_meta("initialized", "true")
        self._set_meta("initial_cash", str(initial_cash))
        self._set_meta("cash", str(initial_cash))
        self._set_meta("created_at", datetime.now().isoformat())
        self._snapshot(initial_cash, 0.0)

    def get_summary_stats(self) -> dict:
        """Statistiques résumées du portefeuille."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM transactions WHERE action = 'BUY'")
        total_buys = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM transactions WHERE action = 'SELL'")
        total_sells = c.fetchone()[0]

        # Calcul des trades gagnants/perdants
        c.execute(
            "SELECT symbol, price, shares FROM transactions WHERE action = 'SELL'"
        )
        sells = c.fetchall()

        winning = 0
        losing = 0
        total_profit = 0.0
        for symbol, sell_price, shares in sells:
            c.execute(
                "SELECT avg_cost FROM positions WHERE symbol = ? "
                "UNION ALL "
                "SELECT ? as avg_cost",  # fallback
                (symbol, sell_price),
            )
            # Simplified: use transaction history for avg cost
            pass

        # Win rate approximation from value history
        history = pd.read_sql_query(
            "SELECT profit_loss FROM value_history ORDER BY timestamp", conn
        )

        conn.close()

        created = self._get_meta("created_at") or datetime.now().isoformat()

        return {
            "total_buys": total_buys,
            "total_sells": total_sells,
            "total_trades": total_buys + total_sells,
            "created_at": created,
            "days_active": (datetime.now() - datetime.fromisoformat(created)).days,
        }
