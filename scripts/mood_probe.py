"""Probe 8: Yahoo FX daily bars vs hourly bars (what does the daily close mean?)."""
import os, yfinance as yf, pandas as pd
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
for sym in ["USDJPY=X", "EURUSD=X"]:
    d = yf.download(sym, period="1mo", interval="1d", auto_adjust=False, progress=False)
    h = yf.download(sym, period="1mo", interval="1h", auto_adjust=False, progress=False)
    for x in (d, h):
        if hasattr(x.columns, "levels"): x.columns = x.columns.get_level_values(0)
    log(sym, "daily index tz", d.index.tz, "hourly tz", h.index.tz, "first hourly", h.index[0])
    hu = h.tz_convert("UTC") if h.index.tz is not None else h.tz_localize("UTC")
    for dt, r in d.tail(8).iterrows():
        day = pd.Timestamp(dt.date(), tz="UTC")
        w = hu[(hu.index >= day) & (hu.index < day + pd.Timedelta(days=1))]
        if w.empty: log(dt.date(), "no hourly"); continue
        log(dt.date(), "D o/h/l/c %.3f %.3f %.3f %.3f" % (r.Open, r.High, r.Low, r.Close),
            "| H(UTC day) o/h/l/c %.3f %.3f %.3f %.3f" % (w.Open.iloc[0], w.High.max(), w.Low.min(), w.Close.iloc[-1]),
            "| last hourly bar", w.index[-1].strftime("%H:%M"), "first", w.index[0].strftime("%H:%M"))
open("data/mood_probe.txt", "w").write("\n".join(out))
