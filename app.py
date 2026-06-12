from flask import Flask, jsonify, request, make_response, send_from_directory
import requests as req
import numpy as np
from datetime import datetime
import traceback
import os

app = Flask(__name__, static_folder='static')

API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE = 'https://financialmodelingprep.com/api/v3'

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
    "MRK","ABBV","PEP","NKE","MCD","AMGN","TMO","ABT","DHR","INTU"
]

FTSE100_TICKERS = [
    "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","GSK.L","DGE.L",
    "LLOY.L","BARC.L","NWG.L"
]

SECTOR_MAP = {
    "Technology": ["AAPL","MSFT","INTU","DHR","TMO"],
    "Healthcare": ["JNJ","MRK","ABBV","AMGN","ABT","AZN.L","GSK.L"],
    "Financials": ["JPM","BAC","HSBA.L","LLOY.L","BARC.L","NWG.L"],
    "Consumer Staples": ["PG","PEP","KO","WMT","MCD","NKE","ULVR.L","DGE.L"],
    "Energy": ["CVX","SHEL.L","BP.L"],
}

def get_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"

def get_index(ticker):
    return "FTSE 100" if ticker.endswith(".L") else "S&P 500"

def fmp_get(endpoint, params={}):
    params['apikey'] = API_KEY
    r = req.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def safe(val, default=None):
    if val is None or val == 0:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default

def score_dcf(profile, ratios):
    try:
        fcf = safe(profile.get('freeCashFlowPerShare', 0))
        price = safe(profile.get('price', 0))
        shares = safe(profile.get('sharesOutstanding', 0))
        if not fcf or not price or not shares or fcf <= 0:
            return 5, 0
        total_fcf = fcf * shares
        market_cap = price * shares
        growth = min(max(safe(ratios.get('revenueGrowth', 0.05), 0.05), -0.05), 0.20)
        discount_rate = 0.10
        terminal_growth = 0.025
        intrinsic = 0
        for yr in range(1, 11):
            intrinsic += total_fcf * ((1 + growth) ** yr) / ((1 + discount_rate) ** yr)
        terminal_fcf = total_fcf * ((1 + growth) ** 10) * (1 + terminal_growth)
        intrinsic += (terminal_fcf / (discount_rate - terminal_growth)) / ((1 + discount_rate) ** 10)
        mos = ((intrinsic - market_cap) / market_cap) * 100
        return round(max(0, min(10, 5 + (mos / 6))), 1), round(mos, 1)
    except Exception:
        return 5, 0

def score_quality(ratios):
    scores = []
    roe = safe(ratios.get('returnOnEquityTTM'))
    if roe: scores.append(min(10, max(0, roe * 50)))
    margin = safe(ratios.get('netProfitMarginTTM'))
    if margin: scores.append(min(10, max(0, margin * 40)))
    fcf_yield = safe(ratios.get('freeCashFlowYieldTTM'))
    if fcf_yield: scores.append(min(10, max(0, fcf_yield * 150)))
    return round(np.mean(scores), 1) if scores else 5

def score_balance(ratios):
    scores = []
    de = safe(ratios.get('debtEquityRatioTTM'))
    if de: scores.append(min(10, max(0, 10 - (de / 20))))
    current = safe(ratios.get('currentRatioTTM'))
    if current: scores.append(min(10, max(0, current * 4)))
    return round(np.mean(scores), 1) if scores else 5

def score_earnings(ratios):
    scores = []
    eps_growth = safe(ratios.get('epsgrowthTTM') or ratios.get('epsGrowth'))
    if eps_growth: scores.append(min(10, max(0, 5 + eps_growth * 20)))
    payout = safe(ratios.get('payoutRatioTTM'))
    if payout: scores.append(8 if 0 < payout < 0.8 else 4)
    return round(np.mean(scores), 1) if scores else 5

def score_valuation(ratios, sector_pe_avg):
    scores = []
    pe = safe(ratios.get('peRatioTTM'))
    if pe and sector_pe_avg:
        scores.append(min(10, max(0, 10 - (pe / sector_pe_avg - 0.5) * 10)))
    pb = safe(ratios.get('priceToBookRatioTTM'))
    if pb: scores.append(min(10, max(0, 10 - pb * 0.8)))
    ev_ebitda = safe(ratios.get('enterpriseValueMultipleTTM'))
    if ev_ebitda: scores.append(min(10, max(0, 10 - ev_ebitda * 0.4)))
    return round(np.mean(scores), 1) if scores else 5

def analyse_ticker(ticker, sector_pe_avgs):
    try:
        fmp_ticker = ticker.replace('.L', '')
        profile_data = fmp_get(f'profile/{fmp_ticker}')
        ratios_data  = fmp_get(f'ratios-ttm/{fmp_ticker}')
        if not profile_data:
            return None
        profile = profile_data[0]
        ratios  = ratios_data[0] if ratios_data else {}
        sector  = get_sector(ticker)
        dcf_score, mos = score_dcf(profile, ratios)
        quality_score  = score_quality(ratios)
        balance_score  = score_balance(ratios)
        earnings_score = score_earnings(ratios)
        val_score      = score_valuation(ratios, sector_pe_avgs.get(sector, 20))
        composite = round(
            dcf_score*0.30 + quality_score*0.25 +
            balance_score*0.20 + earnings_score*0.15 + val_score*0.10, 1
        )
        status = "value" if composite >= 7.0 and mos > 10 else "watch" if composite >= 5.5 else "fair"
        return {
            "ticker": ticker,
            "name": profile.get("companyName", ticker),
            "sector": sector,
            "index": get_index(ticker),
            "price": round(float(profile.get("price", 0)), 2),
            "currency": "GBP" if ticker.endswith(".L") else "USD",
            "score": round(composite * 10),
            "mos": round(mos, 1),
            "status": status,
            "pillars": {
                "dcf":      round(dcf_score * 10),
                "quality":  round(quality_score * 10),
                "balance":  round(balance_score * 10),
                "earnings": round(earnings_score * 10),
                "valuation":round(val_score * 10),
            }
        }
    except Exception:
        return None

def compute_sector_pe_avgs(tickers):
    sector_pes = {}
    for ticker in tickers[:10]:
        try:
            fmp_ticker = ticker.replace('.L', '')
            ratios = fmp_get(f'ratios-ttm/{fmp_ticker}')
            if ratios:
                pe = safe(ratios[0].get('peRatioTTM'))
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
        data = fmp_get('profile/AAPL')
        return jsonify({"success": True, "name": data[0].get('companyName'), "price": data[0].get('price')})
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
