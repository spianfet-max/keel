"""Snapshot for the Mood dashboard. Uses Keel's own sentiment/news modules (checked out to keel-src)
for CFTC, MOF, JPX, SF Fed and Google News; Yahoo for prices. Writes data/mood.json."""
import json, os, sys, traceback, datetime as dt
sys.path.insert(0, os.path.abspath("keel-src"))
os.makedirs("data", exist_ok=True)
LOG = []
def log(*a):
    line = " ".join(str(x) for x in a); print(line, flush=True); LOG.append(line)

out = {"w": {}, "rv": {}}
WEEKLY = ("^GSPC ^VIX ^VIX3M ^VIX9D ^SKEW SPY TLT HYG IEF SPHB SPLV XLY XLP AUDJPY=X USDJPY=X "
          "^N225 ^NKVI.OS 1306.T GC=F").split()
try:
    import numpy as np, pandas as pd, yfinance as yf
    for s in WEEKLY:
        try:
            df = yf.download(s, start="2005-01-01", auto_adjust=True, actions=False, progress=False, threads=False)
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            c = df["Close"].dropna()
            c = c[c > 0]
            w = c.resample("W-FRI").last().dropna()
            out["w"][s] = [[str(i.date()), round(float(v), 5)] for i, v in w.items()]
            if s in ("^GSPC", "^N225", "USDJPY=X"):
                rv = (np.log(c).diff().rolling(20).std() * np.sqrt(252) * 100).resample("W-FRI").last().dropna()
                out["rv"][s] = [[str(i.date()), round(float(v), 2)] for i, v in rv.items()]
            log(s, "ok", len(w), out["w"][s][0][0])
        except Exception as e:
            log(s, "fail", repr(e)[:200])
except Exception:
    log(traceback.format_exc()[-600:])

try:
    from app import sentiment as S
    try:
        cot = S.cot(tuple(S.COT_MARKETS), "2006-01-01")
        out["cot"] = {k: [[r["d"], r["pct"], round(r["net"])] for r in v] for k, v in cot.items()}
        out["cot_names"] = {k: [v[1], v[2]] for k, v in S.COT_MARKETS.items()}
        for k in S.COT_MARKETS:
            v = cot.get(k, [])
            log("COT", k, len(v), v[0]["d"] if v else "-", v[-1] if v else "MISSING")
    except Exception:
        log("COT fail", traceback.format_exc()[-500:])
    for k, fn in (("mof", S.mof_week), ("margin", S.jpx_margin), ("investors", S.jpx_investors)):
        try:
            v = fn()
            out[k] = v
            log(k, len(v), v[0] if v else "-", v[-1] if v else "-")
        except Exception:
            log(k, "fail", traceback.format_exc()[-500:])
except Exception:
    log("import fail", traceback.format_exc()[-600:])

try:
    from app import news as N
    s = N.news_sentiment()
    s = s[s.index >= "2005-01-01"].resample("W-FRI").mean().dropna()
    out["news"] = [[str(i.date()), round(float(v), 4)] for i, v in s.items()]
    log("sffed", len(out["news"]), out["news"][-1])
    hl = {}
    for key, q, lang in (("en", "stock market when:3d", "en"), ("ja", "株式市場 日経平均 when:3d", "ja")):
        try:
            items = N._gnews(q, lang)
            seen, keep = set(), []
            for it in sorted(items, key=lambda x: x["d"], reverse=True):
                if it["title"] not in seen:
                    seen.add(it["title"]); keep.append({k: it[k] for k in ("d", "title", "source", "link")})
            hl[key] = keep[:40]
            log("headlines", key, len(keep))
        except Exception as e:
            log("headlines", key, "fail", repr(e)[:150])
    out["headlines"] = hl
except Exception:
    log("news fail", traceback.format_exc()[-500:])


# ---- StockTwits: daily samples back as far as the public API serves (it stops after ~7 months) ----
try:
    import time, httpx
    from datetime import datetime, timedelta, timezone
    from app import sentiment as S
    c = httpx.Client(timeout=15, headers=S.UA, follow_redirects=True)
    deadline, calls = time.time() + 14 * 60, [0]
    def pg(sym, mx=None):
        for k in range(2):
            if time.time() > deadline:
                return []
            try:
                time.sleep(0.8); calls[0] += 1
                return S.stocktwits_page(sym, mx, c)
            except Exception as e:
                log("st retry", sym, mx, repr(e)[:120]); time.sleep(10)
        return []
    ts = lambda m: datetime.fromisoformat(m["t"].replace("Z", "+00:00")).timestamp()
    top = pg("SPY"); id0, t0 = top[0]["id"], ts(top[0])
    old = pg("SPY", id0 - 15_000_000); rate = 15_000_000 / (t0 - ts(old[0])) if old else 1.0  # ids per second
    log("st rate ids/day", round(rate * 86400))
    days, d = [], datetime.now(timezone.utc).date() - timedelta(days=1)
    rows, empty = [], 0
    while empty < 3 and time.time() < deadline:
        if d.weekday() < 5:
            tgt = datetime(d.year, d.month, d.day, 20, tzinfo=timezone.utc).timestamp()
            guess = int(id0 - (t0 - tgt) * rate)
            p = pg("SPY", guess)
            if p:
                err = ts(p[0]) - tgt
                if abs(err) > 3 * 3600:
                    guess = int(guess - err * rate); p = pg("SPY", guess) or p
            if not p:
                empty += 1
            else:
                empty = 0
                row = {"d": d.isoformat(), "id": guess, "t": p[0]["t"]}
                for sym, ms in (("SPY", p), ("QQQ", pg("QQQ", guess))):
                    row[sym] = [sum(m["s"] == 1 for m in ms), sum(m["s"] == -1 for m in ms), len(ms)]
                rows.append(row)
        d -= timedelta(days=1)
        if d < datetime(2024, 1, 1).date():
            break
    rows.reverse()
    out["st_hist"] = rows
    log("st calls", calls[0], "stopped by deadline" if time.time() > deadline else "")
    log("st days", len(rows), rows[0] if rows else "-", rows[-1] if rows else "-")
    now = S.stocktwits(S.ST_SYMBOLS, 2)
    out["st_now"] = {k: v[:60] for k, v in now.items()}
    log("st now", {k: (len(v), sum(m["s"] == 1 for m in v), sum(m["s"] == -1 for m in v)) for k, v in now.items()})
except Exception:
    log("stocktwits fail", traceback.format_exc()[-600:])

out["generated"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
out["log"] = LOG
json.dump(out, open("data/mood.json", "w"), separators=(",", ":"), ensure_ascii=False)
print("done")
