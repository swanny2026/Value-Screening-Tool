from flask import Flask, jsonify, request, make_response, send_from_directory
import requests as req
import numpy as np
from datetime import datetime
import traceback
import os
import time

app = Flask(__name__, static_folder='static')

API_KEY = os.environ.get('AV_API_KEY', '')
AV_BASE = 'https://www.alphavantage.co/query'

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
# Alpha Vantage free tier = 25 requests/day, so keeping list small
SP500_TICKERS = [
    "AAPL","MSFT","JNJ","JPM","PG","KO","WMT","CVX","MRK","NKE"
]

FTSE100_TICKERS = [
    "SHEL.LON","AZN.LON","HSBA.LON","BP.LON","GSK.LON"
]

SECTOR_MAP = {
    "Technology": ["AAPL","MSFT"],
    "Healthcare": ["JNJ","MRK","AZN.LON","GSK.LON"],
    "Financials": ["JPM","HSBA.LON"],
    "Consumer Staples": ["PG","KO","WMT","NKE"],
    "Energy": ["CVX","SHEL.LON","BP.LON"],
}

def get_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"

def get_index(ticker):
    return "FTSE 100" if ticker.endswith(".LON") else "S&P 500"

def safe(val, default=0):
    if val is None or val == "None" or val == "-":
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default

def av_overview(ticker):
    r = req.get(AV_BASE, params={
        'function': 'OVERVIEW',
        'symbol': ticker,
        'apikey': API_KEY
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    if 'Note' in data or 'Information' in data:
        raise Exception('Rate limit hit: ' + str(data.get('Note') or data.get('Information')))
    return data

def score_dcf(ov):
    try:
        eps = safe(ov.get('EPS'))
        pe  = safe(ov.get('PERatio'))
        if not eps or not pe or eps <= 0:
            return 5, 0
        price = eps * pe
        growth = min(max(safe(ov.get('QuarterlyEarningsGrowthYOY'), 0.05), -0.05), 0.20)
        discount_rate = 0.10
        terminal_growth = 0.025
        intrinsic = 0
        for yr in range(1, 11):
            intrinsic += eps * ((1 + growth) ** yr) / ((1 + discount_rate) ** yr)
        terminal_eps = eps * ((1 + growth) ** 10) * (1 + terminal_growth)
        intrinsic += (terminal_eps / (discount_rate - terminal_growth)) / ((1 + discount_rate) ** 10)
        # Convert per-share intrinsic to price comparison
        mos = ((intrinsic - price) / price) * 100 if price else 0
        return round(max(0, min(10, 5 + (mos / 6))), 1), round(mos, 1)
    except Exception:
        return 5, 0

def score_quality(ov):
    scores = []
    roe = safe(ov.get('ReturnOnEquityTTM'))
    if roe: scores.append(min(10, max(0, roe * 50)))
    margin = safe(ov.get('ProfitMargin'))
    if margin: scores.append(min(10, max(0, margin * 40)))
    return round(np.mean(scores), 1) if scores else 5

def score_balance(ov):
    scores = []
    beta = safe(ov.get('Beta'), 1)
    scores.append(min(10, max(0, 10 - beta * 3)))
    ev = safe(ov.get('EVToEBITDA'))
    if ev: scores.append(min(10, max(0, 10 - ev * 0.4)))
    return round(np.mean(scores), 1) if scores else 5

def score_earnings(ov):
    scores = []
    eg = safe(ov.get('QuarterlyEarningsGrowthYOY'))
    if eg: scores.append(min(10, max(0, 5 + eg * 20)))
    dps = safe(ov.get('DividendPerShare'))
    eps = safe(ov.get('EPS'))
    if dps and eps and eps > 0:
        payout = dps / eps
        scores.append(8 if 0 < payout < 0.8 else 4)
    return round(np.mean(scores), 1) if scores else 5

def score_valuation(ov, sector_pe_avg):
    scores = []
    pe = safe(ov.get('PERatio'))
    if pe and sector_pe_avg:
        scores.append(min(10, max(0, 10 - (pe / sector_pe_avg - 0.5) * 10)))
    pb = safe(ov.get('PriceToBookRatio'))
    if pb: scores.append(min(10, max(0, 10 - pb * 0.8)))
    ev = safe(ov.get('EVToEBITDA'))
    if ev: scores.append(min(10, max(0, 10 - ev * 0.4)))
    return round(np.mean(scores), 1) if scores else 5

def analyse_ticker(ticker, sector_pe_avgs):
    try:
        ov = av_overview(ticker)
        if not ov or 'Symbol' not in ov:
            return None
        sector = get_sector(ticker)
        dcf_score, mos  = score_dcf(ov)
        quality_score   = score_quality(ov)
        balance_score   = score_balance(ov)
        earnings_score  = score_earnings(ov)
        val_score       = score_valuation(ov, sector_pe_avgs.get(sector, 20))
        composite = round(
            dcf_score*0.30 + quality_score*0.25 +
            balance_score*0.20 + earnings_score*0.15 + val_score*0.10, 1
        )
        status = "value" if composite >= 7.0 and mos > 10 else "watch" if composite >= 5.5 else "fair"
        price_str = ov.get('50DayMovingAverage', '0')
        return {
            "ticker": ticker,
            "name": ov.get("Name", ticker),
            "sector": sector,
            "index": get_index(ticker),
            "price": round(safe(price_str), 2),
            "currency": "GBP" if ticker.endswith(".LON") else "USD",
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
    for ticker in tickers[:5]:
        try:
            ov = av_overview(ticker)
            pe = safe(ov.get('PERatio'))
            sector = get_sector(ticker)
            if pe and 0 < pe < 200:
                sector_pes.setdefault(sector, []).append(pe)
            time.sleep(1)  # respect rate limit
        except Exception:
            continue
    return {s: np.mean(v) for s, v in sector_pes.items()}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

@app.route('/test', methods=['GET'])
def test():
    try:
        ov = av_overview('AAPL')
        return jsonify({"success": True, "name": ov.get('Name'), "pe": ov.get('PERatio'), "eps": ov.get('EPS')})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})

@app.route('/run-analysis', methods=['POST', 'OPTIONS'])
def run_analysis():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    try:
        all_tickers = SP500_TICKERS + FTSE100_TICKERS
        sector_pe_avgs = compute_sector_pe_avgs(all_tickers)
        results = []
        for t in all_tickers:
            r = analyse_ticker(t, sector_pe_avgs)
            if r:
                results.append(r)
            time.sleep(1)  # 1 req/sec to stay within free tier
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
        results = []
        for t in tickers:
            r = analyse_ticker(t, sector_pe_avgs)
            if r:
                results.append(r)
            time.sleep(1)
        return jsonify({"success": True, "timestamp": datetime.utcnow().isoformat(), "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
