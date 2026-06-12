from flask import Flask, jsonify, request, make_response, send_from_directory
import requests as req
import numpy as np
from datetime import datetime
import traceback
import os
import threading
import time

app = Flask(__name__, static_folder='static')

API_KEY = os.environ.get('FINNHUB_API_KEY', '')
FH_BASE = 'https://finnhub.io/api/v1'

# ── Cache ────────────────────────────────────────────────────────────────────
cache = {
    "results":   [],
    "timestamp": None,
    "count":     0,
    "refreshing": False,
}
CACHE_TTL_HOURS = 24

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ── S&P 500 tickers ─────────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","JNJ","JPM","PG","KO","WMT","BAC","CVX",
    "MRK","ABBV","PEP","NKE","MCD","AMGN","TMO","ABT","DHR","INTU",
    "V","MA","UNH","HD","AVGO","TXN","QCOM","ORCL","HON","COST"
]

SECTOR_MAP = {
    "Technology":            ["AAPL","MSFT","GOOGL","AVGO","TXN","QCOM","ORCL","INTU","IBM","AMD","NVDA"],
    "Healthcare":            ["JNJ","MRK","ABBV","AMGN","TMO","ABT","DHR","UNH","BMY","GILD"],
    "Financials":            ["JPM","BAC","V","MA","GS","MS","BLK","AXP"],
    "Consumer Staples":      ["PG","KO","PEP","WMT","MCD","SBUX","MDLZ","COST"],
    "Consumer Discretionary":["NKE","HD","AMZN","TSLA"],
    "Energy":                ["CVX"],
    "Industrials":           ["HON","GE","CAT","RTX","UPS","LIN"],
    "Utilities":             ["NEE"],
}

def get_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"

def safe(val, default=0):
    if val is None or val == "None" or val == "-":
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default

def fh_get(endpoint, params={}):
    params['token'] = API_KEY
    r = req.get(f"{FH_BASE}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_quote(ticker):
    return fh_get('quote', {'symbol': ticker})

def get_profile(ticker):
    return fh_get('stock/profile2', {'symbol': ticker})

def get_financials(ticker):
    return fh_get('stock/metric', {'symbol': ticker, 'metric': 'all'})

# ── Scoring ─────────────────────────────────────────────────────────────────

def score_dcf(metrics, price):
    try:
        # Use free cash flow per share approach
        fcf_per_share = safe(metrics.get('freeCashFlowPerShareTTM'))
        if not fcf_per_share or fcf_per_share <= 0 or not price:
            return 5, 0

        # Conservative growth assumptions
        growth_3y   = safe(metrics.get('epsGrowth3Y'), 0.05)
        growth_rate = min(max(growth_3y, -0.02), 0.15)  # cap between -2% and 15%
        discount    = 0.10
        terminal_g  = 0.025
        years       = 10

        # Project and discount FCF per share
        intrinsic = 0
        for yr in range(1, years + 1):
            intrinsic += fcf_per_share * ((1 + growth_rate) ** yr) / ((1 + discount) ** yr)

        # Terminal value
        terminal_fcf = fcf_per_share * ((1 + growth_rate) ** years) * (1 + terminal_g)
        intrinsic += (terminal_fcf / (discount - terminal_g)) / ((1 + discount) ** years)

        # Margin of safety: how much cheaper is the stock vs intrinsic value
        mos = ((intrinsic - price) / price) * 100

        # Cap MoS display at ±100% to avoid wild numbers
        mos = max(min(mos, 100), -100)
        score = max(0, min(10, 5 + (mos / 20)))
        return round(score, 1), round(mos, 1)
    except Exception:
        return 5, 0

def score_quality(metrics):
    scores = []
    roe = safe(metrics.get('roeTTM'))
    if roe: scores.append(min(10, max(0, roe * 50)))
    margin = safe(metrics.get('netProfitMarginTTM'))
    if margin: scores.append(min(10, max(0, margin * 40)))
    roa = safe(metrics.get('roaTTM'))
    if roa: scores.append(min(10, max(0, roa * 100)))
    return round(np.mean(scores), 1) if scores else 5

def score_balance(metrics):
    scores = []
    de = safe(metrics.get('totalDebt/totalEquityAnnual'))
    if de: scores.append(min(10, max(0, 10 - (de / 20))))
    current = safe(metrics.get('currentRatioAnnual'))
    if current: scores.append(min(10, max(0, current * 4)))
    return round(np.mean(scores), 1) if scores else 5

def score_earnings(metrics):
    scores = []
    eg = safe(metrics.get('epsGrowth3Y'))
    if eg: scores.append(min(10, max(0, 5 + eg * 20)))
    eg_ttm = safe(metrics.get('revenueGrowthTTMYoy'))
    if eg_ttm: scores.append(min(10, max(0, 5 + eg_ttm * 15)))
    return round(np.mean(scores), 1) if scores else 5

def score_valuation(metrics, sector_pe_avg):
    scores = []
    pe = safe(metrics.get('peBasicExclExtraTTM'))
    if pe and sector_pe_avg and pe > 0:
        scores.append(min(10, max(0, 10 - (pe / sector_pe_avg - 0.5) * 10)))
    pb = safe(metrics.get('pbAnnual'))
    if pb: scores.append(min(10, max(0, 10 - pb * 0.8)))
    ev_ebitda = safe(metrics.get('currentEv/freeCashFlowAnnual'))
    if ev_ebitda: scores.append(min(10, max(0, 10 - ev_ebitda * 0.4)))
    return round(np.mean(scores), 1) if scores else 5

def analyse_ticker(ticker, sector_pe_avgs):
    try:
        profile  = get_profile(ticker)
        fin_data = get_financials(ticker)
        quote    = get_quote(ticker)

        if not profile or not profile.get('name'):
            return None

        metrics = fin_data.get('metric', {})
        price   = safe(quote.get('c'))
        sector  = get_sector(ticker)

        dcf_score, mos  = score_dcf(metrics, price)
        quality_score   = score_quality(metrics)
        balance_score   = score_balance(metrics)
        earnings_score  = score_earnings(metrics)
        val_score       = score_valuation(metrics, sector_pe_avgs.get(sector, 20))

        composite = round(
            dcf_score*0.30 + quality_score*0.25 +
            balance_score*0.20 + earnings_score*0.15 + val_score*0.10, 1
        )
        status = "value" if composite >= 7.0 and mos > 10 else "watch" if composite >= 5.5 else "fair"

        return {
            "ticker":   ticker,
            "name":     profile.get('name', ticker),
            "sector":   sector,
            "index":    "S&P 500",
            "price":    round(price, 2),
            "currency": "USD",
            "score":    round(composite * 10),
            "mos":      round(mos, 1),
            "status":   status,
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
    for ticker in tickers[:15]:
        try:
            fin_data = get_financials(ticker)
            metrics  = fin_data.get('metric', {})
            pe       = safe(metrics.get('peBasicExclExtraTTM'))
            sector   = get_sector(ticker)
            if pe and 0 < pe < 200:
                sector_pes.setdefault(sector, []).append(pe)
        except Exception:
            continue
    return {s: np.mean(v) for s, v in sector_pes.items()}

# ── Routes ───────────────────────────────────────────────────────────────────

def run_full_analysis():
    """Runs the full analysis and stores results in cache."""
    if cache["refreshing"]:
        return
    cache["refreshing"] = True
    print(f"[{datetime.utcnow().isoformat()}] Starting analysis...")
    try:
        sector_pe_avgs = compute_sector_pe_avgs(SP500_TICKERS)
        results = [r for r in (analyse_ticker(t, sector_pe_avgs) for t in SP500_TICKERS) if r]
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        cache["results"]   = results
        cache["timestamp"] = datetime.utcnow().isoformat()
        cache["count"]     = len(results)
        print(f"[{datetime.utcnow().isoformat()}] Analysis complete — {len(results)} companies scored.")
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        cache["refreshing"] = False

def background_scheduler():
    """Runs analysis once on startup, then every 24 hours."""
    time.sleep(5)  # brief delay to let server start
    while True:
        run_full_analysis()
        time.sleep(CACHE_TTL_HOURS * 3600)

# Start background thread
scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
scheduler_thread.start()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":     "ok",
        "timestamp":  datetime.utcnow().isoformat(),
        "cached":     cache["timestamp"],
        "count":      cache["count"],
        "refreshing": cache["refreshing"],
    })

@app.route('/test', methods=['GET'])
def test():
    try:
        profile  = get_profile('AAPL')
        quote    = get_quote('AAPL')
        fin_data = get_financials('AAPL')
        metrics  = fin_data.get('metric', {})
        return jsonify({
            "success": True,
            "name":    profile.get('name'),
            "price":   quote.get('c'),
            "pe":      metrics.get('peBasicExclExtraTTM'),
            "roe":     metrics.get('roeTTM'),
            "margin":  metrics.get('netProfitMarginTTM'),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})

@app.route('/run-analysis', methods=['POST', 'OPTIONS'])
def run_analysis():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    try:
        force = request.get_json(silent=True) or {}
        force_refresh = force.get('force', False)

        # If force refresh requested, trigger in background and return current cache
        if force_refresh and not cache["refreshing"]:
            t = threading.Thread(target=run_full_analysis, daemon=True)
            t.start()

        # Return cached results if available
        if cache["results"]:
            return jsonify({
                "success":    True,
                "timestamp":  cache["timestamp"],
                "count":      cache["count"],
                "results":    cache["results"],
                "refreshing": cache["refreshing"],
                "cached":     True,
            })

        # No cache yet — still loading
        if cache["refreshing"]:
            return jsonify({
                "success": False,
                "loading": True,
                "error":   "Analysis is running for the first time, please wait a moment and try again."
            }), 202

        # Fallback: trigger fresh analysis synchronously if cache empty and not refreshing
        run_full_analysis()
        return jsonify({
            "success":   True,
            "timestamp": cache["timestamp"],
            "count":     cache["count"],
            "results":   cache["results"],
            "cached":    False,
        })

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
        return jsonify({
            "success":   True,
            "timestamp": datetime.utcnow().isoformat(),
            "results":   results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
