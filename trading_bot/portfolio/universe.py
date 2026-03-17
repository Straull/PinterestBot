"""Univers d'actions éligibles au CTO français.

Actions accessibles depuis un Compte-Titres Ordinaire chez les courtiers français
(Boursorama, Degiro, Trade Republic, etc.).

Inclut des actions US (NYSE/NASDAQ), européennes (Euronext), et ETF populaires.
"""

CTO_UNIVERSE = {
    # ---- Actions US (NYSE / NASDAQ) - Accessibles CTO ----
    "AAPL": {"name": "Apple", "sector": "Tech", "market": "NASDAQ"},
    "MSFT": {"name": "Microsoft", "sector": "Tech", "market": "NASDAQ"},
    "GOOGL": {"name": "Alphabet", "sector": "Tech", "market": "NASDAQ"},
    "AMZN": {"name": "Amazon", "sector": "Tech", "market": "NASDAQ"},
    "NVDA": {"name": "NVIDIA", "sector": "Tech", "market": "NASDAQ"},
    "META": {"name": "Meta", "sector": "Tech", "market": "NASDAQ"},
    "TSLA": {"name": "Tesla", "sector": "Auto", "market": "NASDAQ"},
    "JPM": {"name": "JPMorgan", "sector": "Finance", "market": "NYSE"},
    "V": {"name": "Visa", "sector": "Finance", "market": "NYSE"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Santé", "market": "NYSE"},
    "WMT": {"name": "Walmart", "sector": "Distribution", "market": "NYSE"},
    "PG": {"name": "Procter & Gamble", "sector": "Conso", "market": "NYSE"},
    "MA": {"name": "Mastercard", "sector": "Finance", "market": "NYSE"},
    "DIS": {"name": "Disney", "sector": "Média", "market": "NYSE"},
    "NFLX": {"name": "Netflix", "sector": "Média", "market": "NASDAQ"},
    "AMD": {"name": "AMD", "sector": "Tech", "market": "NASDAQ"},
    "CRM": {"name": "Salesforce", "sector": "Tech", "market": "NYSE"},
    "KO": {"name": "Coca-Cola", "sector": "Conso", "market": "NYSE"},
    "PEP": {"name": "PepsiCo", "sector": "Conso", "market": "NASDAQ"},
    "INTC": {"name": "Intel", "sector": "Tech", "market": "NASDAQ"},

    # ---- Actions Euronext Paris (CAC 40) ----
    "MC.PA": {"name": "LVMH", "sector": "Luxe", "market": "Euronext Paris"},
    "OR.PA": {"name": "L'Oréal", "sector": "Conso", "market": "Euronext Paris"},
    "TTE.PA": {"name": "TotalEnergies", "sector": "Énergie", "market": "Euronext Paris"},
    "SAN.PA": {"name": "Sanofi", "sector": "Santé", "market": "Euronext Paris"},
    "AI.PA": {"name": "Air Liquide", "sector": "Industrie", "market": "Euronext Paris"},
    "SU.PA": {"name": "Schneider Electric", "sector": "Industrie", "market": "Euronext Paris"},
    "BNP.PA": {"name": "BNP Paribas", "sector": "Finance", "market": "Euronext Paris"},
    "ACA.PA": {"name": "Crédit Agricole", "sector": "Finance", "market": "Euronext Paris"},
    "DG.PA": {"name": "Vinci", "sector": "BTP", "market": "Euronext Paris"},
    "AIR.PA": {"name": "Airbus", "sector": "Aéro", "market": "Euronext Paris"},
    "SAF.PA": {"name": "Safran", "sector": "Aéro", "market": "Euronext Paris"},
    "CS.PA": {"name": "AXA", "sector": "Assurance", "market": "Euronext Paris"},
    "KER.PA": {"name": "Kering", "sector": "Luxe", "market": "Euronext Paris"},
    "RI.PA": {"name": "Pernod Ricard", "sector": "Conso", "market": "Euronext Paris"},
    "CAP.PA": {"name": "Capgemini", "sector": "Tech", "market": "Euronext Paris"},

    # ---- ETF populaires (CTO-éligibles) ----
    "SPY": {"name": "SPDR S&P 500 ETF", "sector": "ETF", "market": "NYSE"},
    "QQQ": {"name": "Invesco NASDAQ 100", "sector": "ETF", "market": "NASDAQ"},
    "IWM": {"name": "iShares Russell 2000", "sector": "ETF", "market": "NYSE"},
    "EFA": {"name": "iShares MSCI EAFE", "sector": "ETF", "market": "NYSE"},
    "CW8.PA": {"name": "Amundi MSCI World (EUR)", "sector": "ETF", "market": "Euronext Paris"},
}


def get_symbols_by_market(market: str) -> dict:
    """Filtre l'univers par marché."""
    return {k: v for k, v in CTO_UNIVERSE.items() if v["market"] == market}


def get_symbols_by_sector(sector: str) -> dict:
    """Filtre l'univers par secteur."""
    return {k: v for k, v in CTO_UNIVERSE.items() if v["sector"] == sector}


def get_all_sectors() -> list:
    """Retourne la liste de tous les secteurs."""
    return sorted(set(v["sector"] for v in CTO_UNIVERSE.values()))


def get_all_markets() -> list:
    """Retourne la liste de tous les marchés."""
    return sorted(set(v["market"] for v in CTO_UNIVERSE.values()))
