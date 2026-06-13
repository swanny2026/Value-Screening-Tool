from flask import Flask, jsonify, request, make_response, send_from_directory
import requests as req
import numpy as np
from datetime import datetime
import traceback
import os
import threading
import time

app = Flask(__name__, static_folder='static')

API_KEY      = os.environ.get('FINNHUB_API_KEY', '')
FH_BASE      = 'https://finnhub.io/api/v1'
SB_URL       = os.environ.get('SUPABASE_URL', '')
SB_KEY       = os.environ.get('SUPABASE_KEY', '')
SB_TABLE     = 'analysis_cache'

# ── Cache (in-memory + Supabase backed) ─────────────────────────────────────
cache = {
    "results":    [],
    "timestamp":  None,
    "count":      0,
    "refreshing": False,
    "progress":   0,       # 0-100
    "progress_label": "",
    "started_at": None,
    "total":      0,
    "done":       0,
}
CACHE_TTL_HOURS = 24

def sb_headers():
    return {
        "apikey":        SB_KEY,
        "Authorization": f"Bearer {SB_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal"
    }

def load_cache_from_db():
    """Load latest results from Supabase into memory on startup."""
    try:
        r = req.get(
            f"{SB_URL}/rest/v1/{SB_TABLE}?order=id.desc&limit=1",
            headers=sb_headers(), timeout=10
        )
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            row = data[0]
            cache["results"]   = row["results"]
            cache["timestamp"] = row["timestamp"]
            cache["count"]     = row["count"]
            print(f"Cache loaded from Supabase — {cache['count']} companies.")
            return True
    except Exception as e:
        print(f"Failed to load cache from Supabase: {e}")
    return False

def save_cache_to_db(results, timestamp, count):
    """Save results to Supabase."""
    try:
        # Delete old rows first to keep table clean
        req.delete(
            f"{SB_URL}/rest/v1/{SB_TABLE}?id=gte.0",
            headers=sb_headers(), timeout=10
        )
        # Insert new results
        r = req.post(
            f"{SB_URL}/rest/v1/{SB_TABLE}",
            headers=sb_headers(),
            json={"results": results, "timestamp": timestamp, "count": count},
            timeout=10
        )
        print(f"Cache saved to Supabase — {count} companies.")
        return True
    except Exception as e:
        print(f"Failed to save cache to Supabase: {e}")
        return False

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
    # Technology
    "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","CSCO","ACN","IBM","INTU",
    "GOOGL","GOOG","META","ADBE","NOW","AMD","QCOM","TXN","MU","AMAT",
    "LRCX","ADI","KLAC","SNPS","CDNS","ANSS","PTC","CTSH","EPAM","FFIV",
    "AKAM","JNPR","NTAP","STX","WDC","HPE","HPQ","DELL","NCR","PSTG",
    # Healthcare
    "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","AMGN","ISRG",
    "VRTX","REGN","SYK","BDX","MDT","ZTS","ELV","CI","HUM","MCK",
    "CVS","PFE","MRNA","BIIB","IQV","A","WAT","PKI","HOLX","DXCM",
    "IDXX","MTD","PODD","RMD","TFX","STE","HSIC","COO","ALGN","TECH",
    # Financials
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW",
    "AXP","C","USB","PNC","TFC","COF","DFS","AIG","MET","PRU",
    "BK","STT","NTRS","IVZ","BEN","AMG","AMP","RJF","SF","CBOE",
    "CME","ICE","NDAQ","MKTX","VIRT","RE","ALL","CB","TRV","PGR",
    "HIG","AFL","LNC","UNM","FNF","FAF","SFI","CINF","GL","TMK",
    # Consumer Staples
    "WMT","PG","KO","PEP","COST","MCD","MDLZ","MO","PM","CL",
    "SBUX","EL","CHD","CLX","SJM","CAG","CPB","GIS","HRL","MKC",
    "K","HSY","MNST","STZ","BF-B","TAP","SAM","FIZZ","COKE","KDP",
    # Consumer Discretionary
    "AMZN","TSLA","HD","NKE","LOW","TJX","ROST","BKNG","MAR","HLT",
    "MHK","LVS","MGM","WYNN","CZR","RCL","CCL","NCLH","HAS","MAT",
    "F","GM","APTV","LEA","BWA","LKQ","AAP","AZO","ORLY","GPC",
    "DIS","NFLX","PARA","FOX","LYV","IMAX","CNK","AMC","CINEMARK",
    # Energy
    "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
    "HAL","BKR","NOV","DVN","FANG","PXD","APA","MRO","OVV","CTRA",
    "LNG","KMI","WMB","OKE","ET","EPD","MPLX","PAA","SHLX","TRGP",
    # Industrials
    "GE","HON","RTX","CAT","DE","MMM","LMT","NOC","GD","BA",
    "UPS","FDX","EMR","ETN","PH","ROK","IR","XYL","PCAR","CUMMINS",
    "CMI","TT","CARR","OTIS","AME","FTV","ROP","VRSK","BR","PAYC",
    "WM","RSG","WCN","CLH","SRCL","CWST","HCCI","ACM","PWR","MAS",
    "SWK","PNR","ALLE","AOS","LII","GNRC","HUBB","FELE","RXN","ATKR",
    # Materials
    "LIN","APD","ECL","SHW","PPG","NEM","FCX","VMC","MLM","NUE",
    "STLD","RS","CMC","ATI","AA","ALB","CE","DD","DOW","LYB",
    "EMN","HUN","OLN","WLK","TROX","RPM","SON","AVY","SEE","BMS",
    # Real Estate
    "PLD","AMT","CCI","EQIX","PSA","SPG","O","WELL","AVB","EQR",
    "ESS","MAA","UDR","CPT","AIV","NNN","WPC","STOR","VICI","MGP",
    # Utilities
    "NEE","SO","DUK","AEP","EXC","SRE","D","PCG","ED","FE",
    "EIX","PEG","XEL","ES","WEC","ETR","CMS","NI","ATO","LNT",
    # Communications
    "GOOGL","META","CMCSA","T","VZ","TMUS","CHTR","DIS","NFLX","PARA",
    "WBD","LUMN","DISH","SIRI","IPG","OMC","PUB","WPP","TTGT","ZETA",
    # Fintech & New Tech
    "PYPL","UBER","LYFT","ABNB","DASH","COIN","RBLX","ZM","SNOW","PLTR",
    "CRWD","PANW","FTNT","NET","DDOG","OKTA","ZS","TWLO","MDB","HUBS",
]

SECTOR_MAP = {
    "Technology":             ["AAPL","MSFT","GOOGL","GOOG","NVDA","AVGO","CSCO","CRM","ACN","TXN","QCOM","INTU","ORCL","IBM","ADBE","NOW","SNOW","PLTR","AMAT","LRCX","ADI","MU","KLAC","TWLO","OKTA","DDOG","CRWD","PANW","FTNT","NET","ZS","MDB","HUBS","SNPS","CDNS","ANSS","PTC","CTSH","EPAM","FFIV","AKAM","JNPR","NTAP","STX","WDC","HPE","HPQ","DELL","PSTG","AMD"],
    "Healthcare":             ["JNJ","UNH","MRK","ABBV","TMO","ABT","AMGN","DHR","LLY","REGN","ISRG","SYK","BDX","MDT","ZTS","ELV","CI","HUM","MCK","CVS","PFE","MRNA","BIIB","VRTX","ILMN","IQV","A","WAT","PKI","HOLX","DXCM","IDXX","MTD","PODD","RMD","TFX","STE","HSIC","COO","ALGN","TECH"],
    "Financials":             ["BRK-B","JPM","V","MA","BAC","GS","MS","AXP","WFC","C","USB","PNC","TFC","COF","DFS","AIG","MET","PRU","SCHW","BK","STT","NTRS","IVZ","BEN","AMG","AMP","CME","ICE","NDAQ","CBOE","MKTX","RJF","RE","ALL","CB","TRV","PGR","HIG","AFL","LNC","UNM","FNF","FAF","CINF","GL","BLK"],
    "Consumer Staples":       ["PG","KO","PEP","WMT","MCD","COST","SBUX","MDLZ","MO","PM","CL","EL","CHD","CLX","SJM","CAG","CPB","GIS","HRL","MKC","K","HSY","MNST","STZ","TAP","KDP"],
    "Consumer Discretionary": ["AMZN","TSLA","HD","NKE","LOW","TJX","ROST","BKNG","MAR","HLT","F","GM","APTV","LEA","BWA","LKQ","AAP","AZO","ORLY","GPC","DIS","NFLX","PARA","LYV","MHK","LVS","MGM","WYNN","RCL","CCL","NCLH","HAS","MAT"],
    "Energy":                 ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES","HAL","BKR","NOV","DVN","PXD","APA","MRO","OVV","CTRA","LNG","KMI","WMB","OKE"],
    "Industrials":            ["GE","HON","RTX","CAT","DE","MMM","LMT","NOC","GD","BA","UPS","FDX","EMR","ETN","PH","ROK","IR","XYL","CMI","TT","CARR","OTIS","AME","FTV","ROP","VRSK","WM","RSG","WCN","CLH","ACM","PWR","MAS","SWK","PNR","ALLE","AOS","LII","GNRC","HUBB","FELE","ATKR"],
    "Materials":              ["LIN","APD","ECL","SHW","PPG","NEM","FCX","VMC","MLM","NUE","STLD","RS","CMC","ATI","AA","ALB","CE","DD","DOW","LYB","EMN","HUN","OLN","WLK","TROX","RPM","SON","AVY","SEE","BMS"],
    "Real Estate":            ["PLD","AMT","CCI","EQIX","PSA","SPG","O","WELL","AVB","EQR","ESS","MAA","UDR","CPT","NNN","WPC","VICI"],
    "Utilities":              ["NEE","SO","DUK","AEP","EXC","SRE","D","PCG","ED","FE","EIX","PEG","XEL","ES","WEC","ETR","CMS","NI","ATO","LNT"],
    "Communications":         ["META","CMCSA","T","VZ","TMUS","CHTR","DISH","WBD","SIRI","IPG","OMC"],
    "Fintech":                ["PYPL","FISV","FIS","GPN","UBER","LYFT","ABNB","DASH","COIN","RBLX","ZM"],
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
        # Try FCF per share first, fall back to EPS
        fcf = safe(metrics.get('freeCashFlowPerShareTTM'))
        eps = safe(metrics.get('epsBasicExclExtraItemsAnnual'))
        per_share = fcf if fcf and fcf > 0 else eps
        if not per_share or per_share <= 0 or not price:
            return 5, 0

        # Conservative growth assumptions
        growth_3y   = safe(metrics.get('epsGrowth3Y'), 0.05)
        growth_rate = min(max(growth_3y, -0.02), 0.15)
        discount    = 0.10
        terminal_g  = 0.025

        # Project and discount per share value
        intrinsic = 0
        for yr in range(1, 11):
            intrinsic += per_share * ((1 + growth_rate) ** yr) / ((1 + discount) ** yr)

        # Terminal value
        terminal = per_share * ((1 + growth_rate) ** 10) * (1 + terminal_g)
        intrinsic += (terminal / (discount - terminal_g)) / ((1 + discount) ** 10)

        # Apply a PE-like multiple to convert earnings to price
        pe = safe(metrics.get('peBasicExclExtraTTM'), 15)
        pe = min(max(pe, 8), 30)  # sensible range
        intrinsic_price = intrinsic * (pe / 15)

        mos = ((intrinsic_price - price) / price) * 100
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

        # Use ticker as name if profile missing — don't drop the company
        name = (profile or {}).get('name') or ticker
        metrics = fin_data.get('metric', {}) if fin_data else {}
        price   = safe(quote.get('c')) if quote else 0

        # Skip if no price at all — can't do anything useful
        if not price:
            return None

        sector = get_sector(ticker)

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
            "name":     name,
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
    """Runs the full analysis and stores results in cache and Supabase."""
    if cache["refreshing"]:
        return
    cache["refreshing"]  = True
    cache["progress"]    = 0
    cache["started_at"]  = datetime.utcnow().isoformat()
    cache["total"]       = len(SP500_TICKERS)
    cache["done"]        = 0
    print(f"[{datetime.utcnow().isoformat()}] Starting analysis...")
    try:
        cache["progress_label"] = "Computing sector averages…"
        sector_pe_avgs = compute_sector_pe_avgs(SP500_TICKERS)
        cache["progress"] = 5

        results = []
        total = len(SP500_TICKERS)
        for i, t in enumerate(SP500_TICKERS):
            r = analyse_ticker(t, sector_pe_avgs)
            if r:
                results.append(r)
            cache["done"]          = i + 1
            cache["progress"]      = 5 + int((i + 1) / total * 90)
            cache["progress_label"] = f"Scoring {t} ({i+1} of {total})…"

        cache["progress_label"] = "Ranking results…"
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        timestamp = datetime.utcnow().isoformat()
        cache["results"]        = results
        cache["timestamp"]      = timestamp
        cache["count"]          = len(results)
        cache["progress"]       = 100
        cache["progress_label"] = f"Complete — {len(results)} companies scored."
        save_cache_to_db(results, timestamp, len(results))
        print(f"[{datetime.utcnow().isoformat()}] Analysis complete — {len(results)} companies scored.")
    except Exception as e:
        cache["progress_label"] = f"Error: {e}"
        print(f"Analysis error: {e}")
    finally:
        cache["refreshing"] = False

def background_scheduler():
    """Load from Supabase first, then refresh every 24 hours."""
    time.sleep(3)
    loaded = load_cache_from_db()
    if not loaded:
        run_full_analysis()
    while True:
        time.sleep(CACHE_TTL_HOURS * 3600)
        run_full_analysis()

# Start background thread
scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
scheduler_thread.start()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/status', methods=['GET'])
def status():
    elapsed = None
    eta = None
    if cache["started_at"] and cache["refreshing"]:
        start = datetime.fromisoformat(cache["started_at"])
        elapsed_secs = (datetime.utcnow() - start).total_seconds()
        elapsed = int(elapsed_secs)
        if cache["done"] > 0:
            secs_per_ticker = elapsed_secs / cache["done"]
            remaining = cache["total"] - cache["done"]
            eta = int(secs_per_ticker * remaining)
    return jsonify({
        "refreshing":      cache["refreshing"],
        "progress":        cache["progress"],
        "progress_label":  cache["progress_label"],
        "done":            cache["done"],
        "total":           cache["total"],
        "elapsed_seconds": elapsed,
        "eta_seconds":     eta,
        "has_results":     len(cache["results"]) > 0,
        "timestamp":       cache["timestamp"],
        "count":           cache["count"],
    })

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
