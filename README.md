# Trading Bot V1

Bot de trading Python avec interface graphique moderne, Machine Learning hybride (LSTM + XGBoost + LightGBM) et données de marché en temps réel.

## Fonctionnalités

- **Données** : Import historique via Yahoo Finance avec 20+ indicateurs techniques (RSI, MACD, Bollinger, SMA, EMA, ATR...)
- **ML Ensemble** : LSTM bidirectionnel avec Attention (GPU PyTorch/ROCm) + XGBoost + LightGBM
- **Prévisions** : Direction hausse/baisse avec niveau de confiance et détail par modèle
- **Live** : Suivi en temps réel avec mise à jour automatique toutes les 30s
- **GUI** : Interface sombre moderne avec graphiques interactifs (pyqtgraph)

## Architecture ML

| Modèle | Type | Rôle |
|--------|------|------|
| LSTM + Attention | Deep Learning (GPU) | Capture les patterns séquentiels |
| XGBoost | Gradient Boosting | Relations non-linéaires tabulaires |
| LightGBM | Gradient Boosting | Complément rapide et performant |

La prédiction finale est un **vote pondéré** des 3 modèles, avec poids optimisés automatiquement.

## Installation

```bash
pip install -r requirements.txt

# Pour GPU AMD (ROCm) :
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

## Utilisation

```bash
python main.py
```

1. Entrez un symbole boursier (ex: AAPL, MSFT, TSLA)
2. Sélectionnez la période historique
3. Cliquez sur "Charger les données"
4. Entraînez le modèle ML
5. Lancez le suivi en direct pour obtenir des prévisions

## Temps d'entraînement estimés (AMD 7900 GRE)

- LSTM : ~20-40 min (100 epochs)
- XGBoost : ~1-2 min
- LightGBM : ~30 sec
- **Total : < 1h**
