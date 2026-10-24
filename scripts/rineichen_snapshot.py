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

def get(path, params, tries=8):
    for i in range(tries):
        try:
            r = requests.get(API + path, params=params, timeout=180)
            if r.status_code == 200:
                return r.json()
            print(path, r.status_code, r.text[:200])
        except Exception as e:
            print(path, "error", e)
        time.sleep(20)
    raise SystemExit(f"failed {path}")

get("/health", {})
syms = COUNTRIES + SECTORS + TOPIX17 + OTHER
series, missing = {}, []
for i in range(0, len(syms), 12):
    chunk = syms[i:i + 12]
    j = get("/api/history", {"symbols": ",".join(chunk), "start": START, "interval": "W"})
    series.update(j["series"]); missing += j.get("missing", [])
    print(chunk, "missing", j.get("missing"))
macro = get("/api/macro", {"areas": "US,JP,EA,CN,UK,DE", "start": "2000-01"})
mood = get("/api/newsmood", {"start": "2000-01-01", "daily_days": 30})

weights = {}
try:
    import yfinance as yf
    for t in COUNTRIES:
        try:
            w = yf.Ticker(t).funds_data.sector_weightings
            weights[t] = {k: round(float(v), 4) for k, v in (w or {}).items()}
        except Exception as e:
            print("weights", t, e)
except Exception as e:
    print("yfinance", e)

out = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "series": series, "missing": missing,
       "macro": macro["series"], "macro_asof": macro.get("asof"), "newsmood": mood["monthly"], "newsmood_asof": mood["asof"],
       "weights": weights}
json.dump(out, open("data/rineichen.json", "w"), separators=(",", ":"))
print("series", len(series), "missing", missing, "weights", len(weights))
