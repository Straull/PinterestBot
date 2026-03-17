"""Stratégie de trading automatique basée sur les prédictions ML.

Gère la logique de décision d'achat/vente en fonction des signaux ML,
des niveaux de confiance, et de la gestion du risque.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from trading_bot.portfolio.manager import PortfolioManager
from trading_bot.portfolio.universe import CTO_UNIVERSE


# Commission par transaction (estimée pour un courtier français type Degiro/Boursorama)
COMMISSION_PCT = 0.001  # 0.1%
MIN_COMMISSION = 0.50   # 0.50€ minimum


class TradingStrategy:
    """Stratégie de trading automatique.

    Règles :
    - Acheter quand le modèle prédit HAUSSE avec confiance > seuil
    - Vendre quand le modèle prédit BAISSE avec confiance > seuil
    - Position sizing : max X% du portefeuille par position
    - Diversification : max N positions simultanées
    - Stop-loss et take-profit automatiques
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        confidence_threshold: float = 60.0,
        max_position_pct: float = 20.0,
        max_positions: int = 10,
        stop_loss_pct: float = -5.0,
        take_profit_pct: float = 15.0,
    ):
        self.portfolio = portfolio
        self.confidence_threshold = confidence_threshold
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._last_decisions = {}

    def evaluate(self, symbol: str, prediction: dict, current_price: float) -> Optional[dict]:
        """Évalue une prédiction et décide d'une action.

        Args:
            symbol: ticker boursier
            prediction: dict avec direction, confidence, prob_up, prob_down
            current_price: prix actuel

        Returns:
            dict avec action recommandée ou None si aucune action
        """
        if current_price <= 0:
            return None

        direction = prediction["direction"]
        confidence = prediction["confidence"]
        positions = self.portfolio.get_positions()
        cash = self.portfolio.cash
        total_value = cash

        # Calculer la valeur totale pour le sizing
        if not positions.empty:
            for _, pos in positions.iterrows():
                total_value += pos["shares"] * current_price  # approximation

        has_position = False
        current_shares = 0
        avg_cost = 0
        if not positions.empty:
            pos_row = positions[positions["symbol"] == symbol]
            if not pos_row.empty:
                has_position = True
                current_shares = pos_row.iloc[0]["shares"]
                avg_cost = pos_row.iloc[0]["avg_cost"]

        # ---- Vérifier stop-loss / take-profit sur positions existantes ----
        if has_position and avg_cost > 0:
            pnl_pct = (current_price - avg_cost) / avg_cost * 100

            if pnl_pct <= self.stop_loss_pct:
                return self._sell_signal(
                    symbol, current_shares, current_price,
                    f"STOP-LOSS déclenché ({pnl_pct:.1f}%)"
                )

            if pnl_pct >= self.take_profit_pct:
                return self._sell_signal(
                    symbol, current_shares, current_price,
                    f"TAKE-PROFIT déclenché ({pnl_pct:.1f}%)"
                )

        # ---- Décision basée sur la prédiction ML ----
        if confidence < self.confidence_threshold:
            return {
                "action": "HOLD",
                "symbol": symbol,
                "reason": f"Confiance insuffisante ({confidence:.1f}% < {self.confidence_threshold}%)",
            }

        if direction == "HAUSSE" and not has_position:
            # Signal d'achat
            n_positions = len(positions) if not positions.empty else 0
            if n_positions >= self.max_positions:
                return {
                    "action": "HOLD",
                    "symbol": symbol,
                    "reason": f"Max positions atteint ({self.max_positions})",
                }

            # Position sizing
            max_amount = total_value * (self.max_position_pct / 100)
            amount = min(max_amount, cash * 0.95)  # Garder 5% de marge
            if amount < 10:
                return {
                    "action": "HOLD",
                    "symbol": symbol,
                    "reason": "Fonds insuffisants",
                }

            shares = int(amount / current_price)
            if shares < 1:
                return {
                    "action": "HOLD",
                    "symbol": symbol,
                    "reason": "Prix trop élevé pour le budget",
                }

            commission = max(shares * current_price * COMMISSION_PCT, MIN_COMMISSION)

            return {
                "action": "BUY",
                "symbol": symbol,
                "shares": shares,
                "price": current_price,
                "commission": commission,
                "total": shares * current_price + commission,
                "reason": f"Signal HAUSSE (confiance {confidence:.1f}%)",
                "confidence": confidence,
            }

        elif direction == "BAISSE" and has_position:
            # Signal de vente
            return self._sell_signal(
                symbol, current_shares, current_price,
                f"Signal BAISSE (confiance {confidence:.1f}%)"
            )

        return {
            "action": "HOLD",
            "symbol": symbol,
            "reason": f"{'Déjà en position' if has_position else 'Pas de signal'} - {direction} {confidence:.1f}%",
        }

    def _sell_signal(self, symbol: str, shares: float, price: float, reason: str) -> dict:
        commission = max(shares * price * COMMISSION_PCT, MIN_COMMISSION)
        return {
            "action": "SELL",
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "commission": commission,
            "total": shares * price - commission,
            "reason": reason,
        }

    def execute_decision(self, decision: dict) -> Optional[dict]:
        """Exécute une décision de trading (achat ou vente)."""
        if decision["action"] == "BUY":
            return self.portfolio.buy(
                symbol=decision["symbol"],
                shares=decision["shares"],
                price=decision["price"],
                commission=decision["commission"],
                reason=decision["reason"],
            )
        elif decision["action"] == "SELL":
            return self.portfolio.sell(
                symbol=decision["symbol"],
                shares=decision["shares"],
                price=decision["price"],
                commission=decision["commission"],
                reason=decision["reason"],
            )
        return None

    def get_watchlist(self) -> list:
        """Retourne la liste des symboles CTO à surveiller."""
        return list(CTO_UNIVERSE.keys())
