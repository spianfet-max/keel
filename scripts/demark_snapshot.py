"""Snapshot for the DeMark dashboard: daily OHLCV since 2012 from Yahoo (split-adjusted, not
dividend-adjusted, like Keel's /api/ohlc), plus close-only government yields from Keel.
Writes data/demark.json."""
import json, time, os, traceback, datetime as dt
import requests

API = "https://keel-api-tqpz.onrender.com"
os.makedirs("data", exist_ok=True)
LOG = []
def log(*a):
    line = " ".join(str(x) for x in a); print(line, flush=True); LOG.append(line)

SYMS = ("USDJPY=X EURJPY=X GBPJPY=X AUDJPY=X CHFJPY=X EURUSD=X GBPUSD=X AUDUSD=X CNY=X "
        "^N225 1306.T ^GSPC ^NDX ^STOXX50E ^GDAXI ^HSI ^KS11 "
        "GC=F SI=F CL=F HG=F BTC-USD "
        "7974.T 6758.T 9697.T 7832.T 9766.T 6460.T 9684.T 3635.T").split()
out = {"series": {}}
try:
    import yfinance as yf
    for s in SYMS:
        try:
            df = yf.download(s, start="2012-01-01", auto_adjust=False, actions=False, progress=False, threads=False)
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            rows = []
            for i, r in df.iterrows():
                o, h, l, c = (float(r[k]) for k in ("Open", "High", "Low", "Close"))
                if not o or o != o:
                    o = c
                h, l = max(h, o, c), min(l, o, c)
                v = float(r.get("Volume", 0) or 0)
                rows.append([str(i.date()), o, h, l, c, v])
            out["series"][s] = rows
            log(s, "ok", len(rows), rows[0][0])
        except Exception as e:
            log(s, "fail", repr(e)[:200])
    for i in range(6):
        try:
            r = requests.get(API + "/api/history", params={"symbols": "UST10Y,JGB10Y", "start": "2012-01-01", "interval": "D"}, timeout=240)
            if r.status_code == 200:
                for k, a in r.json()["series"].items():
                    out["series"][k] = [[p["d"], p["v"], p["v"], p["v"], p["v"], 0] for p in a]
                    log(k, "ok", len(a))
                break
            log("yields", r.status_code)
        except Exception as e:
            log("yields error", repr(e)[:200])
        time.sleep(20)
except Exception:
    log(traceback.format_exc())
out["generated"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
out["log"] = LOG
json.dump(out, open("data/demark.json", "w"), separators=(",", ":"))
print("done", len(out["series"]))
