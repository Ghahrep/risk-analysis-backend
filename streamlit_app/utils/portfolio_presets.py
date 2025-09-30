"""Portfolio Presets for Quick Loading"""

PRESETS = {
    "tech": {
        "name": "Tech Growth",
        "symbols": ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
        "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
        "description": "Growth-focused technology portfolio"
    },
    "diversified": {
        "name": "Diversified Balanced",
        "symbols": ['SPY', 'AGG', 'VNQ', 'GLD', 'TLT'],
        "weights": [0.40, 0.30, 0.15, 0.10, 0.05],
        "description": "Balanced multi-asset portfolio"
    },
    "dividend": {
        "name": "Dividend Income",
        "symbols": ['JNJ', 'PG', 'KO', 'MCD', 'PEP'],
        "weights": [0.20, 0.20, 0.20, 0.20, 0.20],
        "description": "Dividend aristocrats portfolio"
    },
    "faang": {
        "name": "FAANG",
        "symbols": ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
        "weights": [0.20, 0.20, 0.20, 0.20, 0.20],
        "description": "Original FAANG stocks"
    },
    "defensive": {
        "name": "Defensive",
        "symbols": ['JNJ', 'PG', 'WMT', 'VZ', 'KO'],
        "weights": [0.20, 0.20, 0.20, 0.20, 0.20],
        "description": "Low-volatility defensive stocks"
    }
}

def get_preset(preset_key):
    """Get preset portfolio by key"""
    return PRESETS.get(preset_key)

def list_presets():
    """List all available presets"""
    return {k: v["name"] for k, v in PRESETS.items()}