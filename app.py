from flask import Flask, jsonify, request, make_response, send_from_directory
import requests as req
import numpy as np
from datetime import datetime
import traceback
import os

app = Flask(__name__, static_folder='static')

API_KEY = os.environ.get('TD_API_KEY', '')
TD_BASE = 'https://api.twelvedata.com'

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ── Index constituents ──────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","JNJ","JPM","PG","KO","WMT","BAC","CVX",
    "MRK","ABBV","PEP","NKE","MCD","AMGN","TMO","ABT","DHR","INTU",
    "V","MA","UNH","HD","AVGO","TXN","QCOM","ORCL","HON","COST"
]

FTSE100_TICKERS = [
    "SHEL:LSE","AZN:LSE","HSBA:LSE","ULVR:LSE","BP:LSE","GSK:LSE",
    "DGE:LSE","LLOY:LSE","BARC:LSE","NWG:LSE","RIO:LSE","NG:LSE",
    "BATS:LSE","REL:LSE","EXPN:LSE"
]

SECTOR_MAP = {
    "Technology": ["AAPL","MSFT","GOOGL","AVGO","TXN","QCOM","ORCL","INTU"],
    "Healthcare": ["JNJ","MRK","ABBV","AMGN","TMO","ABT","DHR","UNH","AZN:LSE","GSK:LSE"],
    "Financials": ["JPM","BAC","V","MA","HSBA:LSE","LLOY:LSE","BARC:LSE","NWG:LSE"],
    "Consumer Staples": ["PG","KO","WMT","PEP","MCD","NKE","COST","ULVR:LSE","DGE:LSE","BATS:LSE"],
    "Energy": ["CVX","SHEL:LSE","BP:LSE"],
    "Industrials": ["HON","HD","RIO:LSE","NG:LSE"],
    "Communications": ["REL:LSE"],
    "Other": ["EXPN:LSE"],
}

def get_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"

def get_index(ticker):
    return "FTSE 100" if ":LSE" in ticker else "S&P 500"

def display_ticker(ticker):
    return ticker.replace(":LSE", ".L")

def safe(val, default=0):
    if val is None or val == "None" or val == "-" or val == "N/A":
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except Exception:
        return default

def td_get(endpoint, params={}):
    params['apikey'] = API_KEY
    r = req.get(f"{TD_BASE}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get('code') in [400, 401, 403, 429]:
        raise Exception(f"API error: {data.get('message', str(data))}")
    return data

def get_fundamentals(ticker):
    symbol = ticker.split(':')[0]
    exchange = 'LSE' if ':LSE' in ticker else 'NASDAQ,NYSE'
    data = td_get('fundamentals', {'symbol': symbol, 'exchange': exchange})
    return data

def score_dcf(stats, price):
    try:
        eps = safe(stats.get('eps'))
        growth = min(max(safe(stats.get('eps_growth_next_y'), 0.05), -0.05), 0.20)
        pe = safe(stats.get('pe'))
        if not eps or eps <= 0 or not pe:
            return 5, 0
        discount_rate = 0.10
        terminal_growth = 0.025
        intrinsic = 0
        for yr in range(1, 11):
            intrinsic += eps * ((1 + growth) ** yr) / ((1 + discount_rate) ** yr)
        terminal_eps = eps * ((1 + growth) ** 10) * (1 + terminal_growth)
        intrinsic += (terminal_eps / (discount_rate - terminal_growth)) / ((1 + discount_rate) ** 10)
        intrinsic_price = intrinsic * pe / 10
        mos = ((intrinsic_price - price) / price) * 100 if price else 0
        return round(max(0, min(10, 5 + (mos / 6))), 1), round(mos, 1)
    except Exception:
        return 5, 0

def score_quality(stats, financials):
    scores = []
    roe = safe(stats.get('return_on_equity'))
    if roe: scores.append(min(10, max(0, roe * 50)))
    margin = safe(stats.get('net_profit_margin'))
    if margin: scores.append(min(10, max(0, margin * 40)))
    fcf = safe(financials.get('free_cash_flow'))
    mc  = safe(financials.get('market_capitalization'))
    if fcf and mc and mc > 0:
        scores.append(min(10, max(0, (fcf / mc) * 100 * 1.5)))
    return round(np.mean(scores), 1) if scores else 5

def score_balance(stats):
    scores = []
    de = safe(stats.get('debt_to_equity'))
    if de: scores.append(min(10, max(0, 10 - (de / 20))))
    current = safe(stats.get('current_ratio'))
    if current: scores.append(min(10, max(0, current * 4)))
    return round(np.mean(scores), 1) if scores else 5

def score_earnings(stats):
    scores = []
    eg = safe(stats.get('eps_growth_ttm'))
    if eg: scores.append(min(10, max(0, 5 + eg * 20)))
    dps = safe(stats.get('dividend_per_share'))
    eps = safe(stats.get('eps'))
    if dps and eps and eps > 0:
        payout = dps / eps
        scores.append(8 if 0 < payout < 0.8 else 4)
    return round(np.mean(scores), 1) if scores else 5

def score_valuation(stats, sector_pe_avg):
    scores = []
    pe = safe(stats.get('pe'))
    if pe and sector_pe_avg and pe > 0:
        scores.append(min(10, max(0, 10 - (pe / sector_pe_avg - 0.5) * 10)))
    pb = safe(stats.get('pb'))
    if pb: scores.append(min(10, max(0, 10 - pb * 0.8)))
    ev_ebitda = safe(stats.get('ev_to_ebitda'))
    if ev_ebitda: scores.append(min(10, max(0, 10 - ev_ebitda * 0.4)))
    return round(np.mean(scores), 1) if scores else 5

def analyse_ticker(ticker, sector_pe_avgs):
    try:
        data = get_fundamentals(ticker)
        if not data or 'statistics' not in data:
            return None

        stats      = data.get('statistics', {}).get('valuations_metrics', {})
        financials = data.get('statistics', {}).get('financials', {})
        profile    = data.get('profile', {})
        price_data = td_get('price', {'symbol': ticker.split(':')[0], 'exchange': 'LSE' if ':LSE' in ticker else 'NASDAQ,NYSE'})
        price      = safe(price_data.get('price'))

        sector = get_sector(ticker)
        dcf_score, mos  = score_dcf(stats, price)
        quality_score   = score_quality(stats, financials)
        balance_score   = score_balance(stats)
        earnings_score  = score_earnings(stats)
        val_score       = score_valuation(stats, sector_pe_avgs.get(sector, 20))

        composite = round(
            dcf_score*0.30 + quality_score*0.25 +
            balance_score*0.20 + earnings_score*0.15 + val_score*0.10, 1
        )
        status = "value" if composite >= 7.0 and mos > 10 else "watch" if composite >= 5.5 else "fair"

        return {
            "ticker": display_ticker(ticker),
            "name": profile.get("name", ticker),
            "sector": sector,
            "index": get_index(ticker),
            "price": round(price, 2),
            "currency": "GBP" if ":LSE" in ticker else "USD",
            "score": round(composite * 10),
            "mos": round(mos, 1),
            "status": status,
            "pillars": {
                "dcf":       round(dcf_score * 10),
                "quality":   round(quality_score * 10),
                "balance":   round(balance_score * 10),
                "earnings":  round(earnings_score * 10),
                "valuation": round(val_score * 10),
            }
        }
    except Exception as e:
        print(f"Error analysing {ticker}: {e}")
        return None

def compute_sector_pe_avgs(tickers):
    sector_pes = {}
    for ticker in tickers[:10]:
        try:
            data = get_fundamentals(ticker)
            if data and 'statistics' in data:
                pe = safe(data['statistics'].get('valuations_metrics', {}).get('pe'))
                sector = get_sector(ticker)
                if pe and 0 < pe < 200:
                    sector_pes.setdefault(sector, []).append(pe)
        except Exception:
            continue
    return {s: np.mean(v) for s, v in sector_pes.items()}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

@app.route('/test', methods=['GET'])
def test():
    try:
        data = get_fundamentals('AAPL')
        profile = data.get('profile', {})
        price_data = td_get('price', {'symbol': 'AAPL'})
        return jsonify({
            "success": True,
            "name": profile.get('name'),
            "price": price_data.get('price'),
            "keys": list(data.keys())
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})

@app.route('/run-analysis', methods=['POST', 'OPTIONS'])
def run_analysis():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    try:
        all_tickers = SP500_TICKERS + FTSE100_TICKERS
        sector_pe_avgs = compute_sector_pe_avgs(all_tickers)
        results = [r for r in (analyse_ticker(t, sector_pe_avgs) for t in all_tickers) if r]
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return jsonify({"success": True, "timestamp": datetime.utcnow().isoformat(), "count": len(results), "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/refresh-watchlist', methods=['POST', 'OPTIONS'])
def refresh_watchlist():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    try:
        tickers = request.get_json().get("tickers", [])
        if not tickers:
            return jsonify({"success": False, "error": "No tickers provided"}), 400
        sector_pe_avgs = compute_sector_pe_avgs(tickers)
        results = [r for r in (analyse_ticker(t, sector_pe_avgs) for t in tickers) if r]
        return jsonify({"success": True, "timestamp": datetime.utcnow().isoformat(), "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
