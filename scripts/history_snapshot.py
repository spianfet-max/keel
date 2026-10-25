"""Snapshot for the History dashboard: daily closes since 1985 (or first available).
Yahoo for indices, futures, ETFs and recent FX (stale daily FX closes rebuilt from the next open, as Keel does);
FRED for long FX, oil and US 10-year history; Keel (Japan MOF) for the JGB 10-year. Writes data/history.json."""
import io, json, os, time, traceback, datetime as dt
import requests
import pandas as pd
os.makedirs("data", exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

START = "1985-01-01"
YAHOO = {  # key: yahoo symbol
    "SPX": "^GSPC", "NKY": "^N225", "TOPIX": "1306.T", "NDX": "^NDX", "SX5E": "^STOXX50E", "DAX": "^GDAXI", "FTSE": "^FTSE",
    "HSI": "^HSI", "KOSPI": "^KS11", "EEM": "EEM", "VIX": "^VIX", "GOLD": "GC=F", "OIL": "CL=F", "COPPER": "HG=F",
    "BTC": "BTC-USD", "DXY": "DX-Y.NYB", "UST10": "^TNX", "TLT": "TLT", "HYG": "HYG",
    "USDJPY": "USDJPY=X", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "AUDJPY": "AUDJPY=X", "USDCNY": "CNY=X", "USDCHF": "CHF=X", "USDTRY": "TRY=X",
}
FRED = {"USDJPY": "DEXJPUS", "EURUSD": "DEXUSEU", "GBPUSD": "DEXUSUK", "USDCHF": "DEXSZUS", "USDCNY": "DEXCHUS", "OIL": "DCOILWTICO", "UST10": "DGS10"}
out = {"s": {}, "src": {}}

def fx_fix(df):
    flat = (df.High == df.Low) & (df.Open == df.Close)
    df = df[~flat]
    t = df.iloc[-250:]; rng = t.High - t.Low
    med = ((t.Close - t.Open).abs() / rng)[rng > 0].median()
    if med >= 0.08: return df, med
    df = df.copy(); df.loc[df.index[:-1], "Close"] = df.Open.shift(-1).iloc[:-1].values
    return df, med

try:
    import yfinance as yf
    for k, sym in YAHOO.items():
        try:
            df = yf.download(sym, start=START, auto_adjust=False, actions=False, progress=False, threads=False)
            if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"]); df = df[df.Close > 0]
            note = ""
            if sym.endswith("=X"):
                df, med = fx_fix(df); note = f"fx body {med:.2f}"
            c = df.Close
            if sym == "1306.T":  # unadjusted 10:1 split in 2016? guard: drop jumps over 60% in a day
                r = c.pct_change().abs(); bad = r[r > .6]
                if len(bad): log(k, "large jumps", list(bad.index.date[:3]))
            out["s"][k] = {"y": [[str(i.date()), float(v)] for i, v in c.items()]}
            log(k, sym, "ok", len(c), str(c.index[0].date()), note)
        except Exception as e:
            log(k, sym, "fail", repr(e)[:150])
except Exception:
    log(traceback.format_exc()[-500:])

for k, sid in FRED.items():
    for attempt in range(3):
        try:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": START}, timeout=90,
                             headers={"User-Agent": "Mozilla/5.0 (keel research)"})
            r.raise_for_status()
            f = pd.read_csv(io.StringIO(r.text)); f.columns = ["d", "v"]
            f["v"] = pd.to_numeric(f.v, errors="coerce"); f = f.dropna()
            out["s"].setdefault(k, {})["f"] = [[d, float(v)] for d, v in zip(f.d, f.v)]
            log(k, "FRED", sid, len(f), f.d.iloc[0], "->", f.d.iloc[-1]); break
        except Exception as e:
            log(k, "FRED", sid, "attempt", attempt, repr(e)[:120]); time.sleep(10)

for attempt in range(6):
    try:
        r = requests.get("https://keel-api-tqpz.onrender.com/api/history", params={"symbols": "JGB10Y", "start": START, "interval": "D"}, timeout=240)
        if r.status_code == 200:
            a = r.json()["series"].get("JGB10Y", [])
            out["s"]["JGB10"] = {"k": [[p["d"], p["v"]] for p in a]}
            log("JGB10", "keel", len(a), a[0]["d"] if a else "-"); break
        log("JGB10 status", r.status_code)
    except Exception as e:
        log("JGB10 err", repr(e)[:120])
    time.sleep(20)

out["generated"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"); out["log"] = LOG
json.dump(out, open("data/history.json", "w"), separators=(",", ":"))
print("done")
