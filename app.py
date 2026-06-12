from flask import Flask, jsonify, request, make_response
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

# ── Index constituents ──────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","BRK-B","JNJ","JPM","V","PG","MA",
    "UNH","HD","CVX","MRK","ABBV","PEP","KO","BAC","PFE","AVGO",
    "TMO","COST","MCD","ACN","ABT","DHR","WMT","LIN","NKE","TXN",
    "PM","NEE","UPS","MS","RTX","ORCL","QCOM","HON","AMGN","BMY",
    "SBUX","IBM","GE","CAT","GS","BLK","MDLZ","AXP","GILD","INTU"
]

FTSE100_TICKERS = [
    "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","DGE.L",
    "BATS.L","LSEG.L","NG.L","LLOY.L","VOD.L","REL.L","EXPN.L","CPG.L",
    "BARC.L","NWG.L","PRU.L","STAN.L","IMB.L","WPP.L","TSCO.L","SGRO.L",
    "BT-A.L","INF.L","MNDI.L","RKT.L","SKG.L","ANTO.L"
]

SECTOR_MAP = {
    "Technology": ["AAPL","MSFT","AVGO","TXN","QCOM","ORCL","IBM","INTU","ACN"],
    "Healthcare": ["JNJ","UNH","MRK","ABBV","PFE","TMO","DHR","ABT","AMGN","BMY","GILD","AZN.L","GSK.L"],
    "Financials": ["BRK-B","JPM","V","MA","BAC","MS","GS","BLK","AXP","HSBA.L","LLOY.L","BARC.L","NWG.L","STAN.L","PRU.L","LSEG.L"],
    "Consumer Staples": ["PG","PEP","KO","WMT","COST","MCD","SBUX","MDLZ","PM","ULVR.L","BATS.L","DGE.L","IMB.L","TSCO.L","CPG.L"],
    "Energy": ["CVX","SHEL.L","BP.L"],
    "Industrials": ["HD","UPS","RTX","HON","GE","CAT","RIO.L","NG.L"],
    "Consumer Discretionary": ["AMZN","NKE","REL.L","WPP.L","BT-A.L"],
    "Real Estate": ["SGRO.L"],
    "Materials": ["LIN","ANTO.L","MNDI.L","SKG.L"],
    "Communications": ["GOOGL","VOD.L"],
    "Utilities": ["NEE","INF.L"],
}

def get_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"

def get_index(ticker):
    return "FTSE 100" if ticker.endswith(".L") else "S&P 500"

def safe_get(info, key, default=None):
    val = info.get(key, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

def score_dcf_margin(info):
    try:
        fcf = safe_get(info, "freeCashflow")
        market_cap = safe_get(info, "marketCap")
        growth = safe_get(info, "earningsGrowth", 0.05)
        if not fcf or not market_cap or fcf <= 0:
            return 5, 0
        growth = max(min(float(growth), 0.20), -0.05)
        discount_rate = 0.10
        terminal_growth = 0.025
        years = 10
        intrinsic = 0
        for yr in range(1, years + 1):
            projected_fcf = fcf * ((1 + growth) ** yr)
            intrinsic += projected_fcf / ((1 + discount_rate) ** yr)
        terminal_fcf = fcf * ((1 + growth) ** years) * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        intrinsic += terminal_value / ((1 + discount_rate) ** years)
        mos = ((intrinsic - market_cap) / market_cap) * 100
        score = max(0, min(10, 5 + (mos / 6)))
        return round(score, 1), round(mos, 1)
    except Exception:
        return 5, 0

def score_business_quality(info):
    try:
        scores = []
        roe = safe_get(info, "returnOnEquity")
        if roe is not None:
            scores.append(min(10, max(0, float(roe) * 50)))
        margin = safe_get(info, "profitMargins")
        if margin is not None:
            scores.append(min(10, max(0, float(margin) * 40)))
        fcf = safe_get(info, "freeCashflow")
        mc = safe_get(info, "marketCap")
        if fcf and mc and mc > 0:
            fcf_yield = (fcf / mc) * 100
            scores.append(min(10, max(0, fcf_yield * 1.5)))
        return round(np.mean(scores), 1) if scores else 5
    except Exception:
        return 5

def score_balance_sheet(info):
    try:
        scores = []
        de = safe_get(info, "debtToEquity")
        if de is not None:
            scores.append(min(10, max(0, 10 - (float(de) / 20))))
        current = safe_get(info, "currentRatio")
        if current is not None:
            scores.append(min(10, max(0, float(current) * 4)))
        return round(np.mean(scores), 1) if scores else 5
    except Exception:
        return 5

def score_earnings_consistency(info):
    try:
        scores = []
        eps_growth = safe_get(info, "earningsGrowth")
        if eps_growth is not None:
            scores.append(min(10, max(0, 5 + float(eps_growth) * 20)))
        payout = safe_get(info, "payoutRatio")
        if payout is not None and 0 < float(payout) < 0.8:
            scores.append(8)
        elif payout is not None:
            scores.append(4)
        return round(np.mean(scores), 1) if scores else 5
    except Exception:
        return 5

def score_valuation_ratios(info, sector_pe_avg):
    try:
        scores = []
        pe = safe_get(info, "trailingPE")
        if pe is not None and sector_pe_avg:
            ratio = float(pe) / sector_pe_avg
            scores.append(min(10, max(0, 10 - (ratio - 0.5) * 10)))
        pb = safe_get(info, "priceToBook")
        if pb is not None:
            scores.append(min(10, max(0, 10 - float(pb) * 0.8)))
        ev_ebitda = safe_get(info, "enterpriseToEbitda")
        if ev_ebitda is not None:
            scores.append(min(10, max(0, 10 - float(ev_ebitda) * 0.4)))
        return round(np.mean(scores), 1) if scores else 5
    except Exception:
        return 5

def compute_composite_score(dcf_score, quality_score, balance_score, earnings_score, val_score):
    return round(
        dcf_score      * 0.30 +
        quality_score  * 0.25 +
        balance_score  * 0.20 +
        earnings_score * 0.15 +
        val_score      * 0.10,
        1
    )

def status_from_score(score, mos):
    if score >= 7.0 and mos > 10:
        return "value"
    elif score >= 5.5:
        return "watch"
    else:
        return "fair"

def analyse_ticker(ticker, sector_pe_avgs):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = safe_get(info, "longName") or safe_get(info, "shortName") or ticker
        sector = get_sector(ticker)
        index = get_index(ticker)
        sector_pe = sector_pe_avgs.get(sector, 20)
        dcf_score, mos = score_dcf_margin(info)
        quality_score   = score_business_quality(info)
        balance_score   = score_balance_sheet(info)
        earnings_score  = score_earnings_consistency(info)
        val_score       = score_valuation_ratios(info, sector_pe)
        composite = compute_composite_score(dcf_score, quality_score, balance_score, earnings_score, val_score)
        score_100 = round(composite * 10)
        status = status_from_score(composite, mos)
        price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice", 0)
        currency = safe_get(info, "currency", "USD")
        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "index": index,
            "price": round(float(price), 2) if price else 0,
            "currency": currency,
            "score": score_100,
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
    for ticker in tickers[:20]:
        try:
            info = yf.Ticker(ticker).info
            pe = safe_get(info, "trailingPE")
            sector = get_sector(ticker)
            if pe and 0 < float(pe) < 200:
                sector_pes.setdefault(sector, []).append(float(pe))
        except Exception:
            continue
    return {s: np.mean(v) for s, v in sector_pes.items()}

@app.route('/run-analysis', methods=['POST', 'OPTIONS'])
def run_analysis():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    try:
        all_tickers = SP500_TICKERS + FTSE100_TICKERS
        sector_pe_avgs = compute_sector_pe_avgs(all_tickers)
        results = []
        for ticker in all_tickers:
            result = analyse_ticker(ticker, sector_pe_avgs)
            if result:
                results.append(result)
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return jsonify({
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/refresh-watchlist', methods=['POST', 'OPTIONS'])
def refresh_watchlist():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    try:
        data = request.get_json()
        tickers = data.get("tickers", [])
        if not tickers:
            return jsonify({"success": False, "error": "No tickers provided"}), 400
        sector_pe_avgs = compute_sector_pe_avgs(tickers)
        results = []
        for ticker in tickers:
            result = analyse_ticker(ticker, sector_pe_avgs)
            if result:
                results.append(result)
        return jsonify({
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
