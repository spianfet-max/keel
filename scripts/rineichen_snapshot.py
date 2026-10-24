"""Snapshot for the Rineichen dashboard: weekly closes from Keel, OECD macro, news sentiment,
and country-ETF sector weights from Yahoo. Writes data/rineichen.json."""
import json, time, datetime as dt, requests

API = "https://keel-api-tqpz.onrender.com"
START = "2005-01-01"
COUNTRIES = "ACWI SPY EWJ EWG EWQ EWU EWL EWI EWP EWN EWD EWC EWA EWH MCHI EWY EWT INDA EWZ EWW EIDO EZA EWS THD".split()
SECTORS = "IXC MXI EXI RXI KXI IXJ IXG IXN IXP JXI REET".split()
TOPIX17 = [f"{c}.T" for c in range(1617, 1634)]
OTHER = ("^VIX ^MOVE HYG IEF LQD USDJPY=X EURJPY=X GBPJPY=X AUDJPY=X CHFJPY=X EURUSD=X CNY=X "
         "GC=F SI=F HG=F CL=F NG=F BTC-USD ETH-USD ^GSPC ^N225 ^STOXX50E ^TPX UST2Y UST10Y JGB10Y").split()

LOG = []
import os, traceback
os.makedirs("data", exist_ok=True)
def log(*a):
    line = " ".join(str(x) for x in a); print(line, flush=True); LOG.append(line)
    with open("data/rineichen.log", "a") as f:
        f.write(line + "\n")

def get(path, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(API + path, params=params, timeout=240)
            if r.status_code == 200:
                return r.json()
            log(path, params.get("symbols", ""), r.status_code, r.text[:300])
        except Exception as e:
            log(path, params.get("symbols", ""), "error", repr(e)[:300])
        time.sleep(15)
    return None

try:
    log("health", bool(get("/health", {}, tries=8)))
    syms = COUNTRIES + SECTORS + TOPIX17 + OTHER
    series, missing = {}, []
    def fetch(chunk):
        j = get("/api/history", {"symbols": ",".join(chunk), "start": START, "interval": "W"})
        if j is None and len(chunk) > 1:  # split and retry
            h = len(chunk) // 2
            fetch(chunk[:h]); fetch(chunk[h:]); return
        if j is None:
            missing.append(chunk[0]); return
        series.update(j["series"]); missing.extend(j.get("missing", []))
        log(chunk, "ok", len(j["series"]), "missing", j.get("missing"))

    for i in range(0, len(syms), 6):
        fetch(syms[i:i + 6])
    macro = get("/api/macro", {"areas": "US,JP,EA,CN,UK,DE", "start": "2000-01"}) or {"series": {}, "asof": None}
    mood = get("/api/newsmood", {"start": "2000-01-01", "daily_days": 30}) or {"monthly": [], "asof": None}

    weights = {}
    try:
        import yfinance as yf
        for t in COUNTRIES:
            try:
                w = yf.Ticker(t).funds_data.sector_weightings
                weights[t] = {k: round(float(v), 4) for k, v in (w or {}).items()}
            except Exception as e:
                log("weights", t, repr(e)[:200])
    except Exception as e:
        log("yfinance", repr(e)[:200])

    out = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "series": series, "missing": missing,
           "macro": macro["series"], "macro_asof": macro.get("asof"), "newsmood": mood["monthly"], "newsmood_asof": mood["asof"],
           "weights": weights, "log": LOG}
    json.dump(out, open("data/rineichen.json", "w"), separators=(",", ":"))
    print("series", len(series), "missing", missing, "weights", len(weights))
except Exception:
    log(traceback.format_exc())
