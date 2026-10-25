"""Probe 9: Keel's closes()/ohlc() for FX via OpenBB, with the fix, vs Yahoo hourly."""
import os, sys
sys.path.insert(0, os.path.abspath("keel-src"))
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
try:
    import yfinance as yf, pandas as pd
    from app import data as D
    o = D.ohlc("USDJPY=X", 0.2); c = D.closes("USDJPY=X", 0.2)
    h = yf.download("USDJPY=X", period="1mo", interval="1h", progress=False)
    if hasattr(h.columns, "levels"): h.columns = h.columns.get_level_values(0)
    hu = h.tz_convert("UTC")
    for dt, r in o.tail(7).iterrows():
        day = pd.Timestamp(dt.date(), tz="UTC"); w = hu[(hu.index >= day) & (hu.index < day + pd.Timedelta(days=1))]
        log(dt.date(), "keel ohlc c %.3f closes %.3f | hourly last %.3f" % (r.close, c.get(dt, float("nan")), w.Close.iloc[-1] if len(w) else float("nan")))
    med = ((o.close - o.open).abs() / (o.high - o.low)).median()
    log("median body/range after fix %.2f" % med)
except Exception:
    import traceback; log(traceback.format_exc()[-800:])
open("data/mood_probe.txt", "w").write("\n".join(out))
