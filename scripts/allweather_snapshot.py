"""Snapshot for the AllWeather dashboard: weekly TOTAL-RETURN closes from Yahoo (dividends and
distributions reinvested), plus yields, FX and OECD macro from Keel. Writes data/allweather.json."""
import json, time, os, traceback, datetime as dt
import requests

API = "https://keel-api-tqpz.onrender.com"
os.makedirs("data", exist_ok=True)
LOG = []
def log(*a):
    line = " ".join(str(x) for x in a); print(line, flush=True); LOG.append(line)

ETFS = "SPY EFA EEM 1306.T EWJ IEF TLT TIP LQD HYG VNQ GLD DBC BTC-USD".split()
RAW = "^IRX USDJPY=X".split()          # price/level series (no dividends)
KEEL = "UST2Y UST10Y UST30Y JGB2Y JGB10Y JGB30Y".split()

def keel(path, params, tries=5):
    for i in range(tries):
        try:
            r = requests.get(API + path, params=params, timeout=240)
            if r.status_code == 200:
                return r.json()
            log(path, params.get("symbols", ""), r.status_code)
        except Exception as e:
            log(path, "error", repr(e)[:200])
        time.sleep(15)
    return None

out = {"series": {}, "macro": {}, "log": LOG}
try:
    import yfinance as yf, pandas as pd
    for group, adj in ((ETFS, True), (RAW, False)):
        for s in group:
            try:
                df = yf.download(s, start="2004-01-01", auto_adjust=adj, progress=False, threads=False)
                c = df["Close"]
                if hasattr(c, "columns"):
                    c = c.iloc[:, 0]
                c = c.dropna()
                w = c.resample("W-FRI").last().dropna()
                out["series"][s] = [{"d": str(i.date()), "v": round(float(v), 6)} for i, v in w.items()]
                log(s, "ok", len(w), str(w.index[0].date()))
            except Exception as e:
                log(s, "fail", repr(e)[:200])
    log("health", bool(keel("/health", {}, tries=8)))
    j = keel("/api/history", {"symbols": ",".join(KEEL), "start": "2004-01-01", "interval": "W"})
    if j:
        out["series"].update(j["series"]); log("yields ok", list(j["series"]), "missing", j.get("missing"))
    m = keel("/api/macro", {"areas": "US,JP,DE,UK,CN", "start": "2000-01"})
    if m:
        out["macro"] = m["series"]; log("macro ok", list(m["series"]))
except Exception:
    log(traceback.format_exc())
out["generated"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
json.dump(out, open("data/allweather.json", "w"), separators=(",", ":"))
print("done", len(out["series"]))
