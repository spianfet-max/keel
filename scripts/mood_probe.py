"""Probe which Mood data sources answer from a GitHub runner. Writes data/mood_probe.txt."""
import requests, os, traceback
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = {"User-Agent": "Mozilla/5.0 (keel research)"}
def get(name, url, **kw):
    try:
        r = requests.get(url, headers=H, timeout=60, **kw)
        log(name, r.status_code, r.headers.get("content-type", "")[:40], len(r.content), repr(r.text[:300]).replace("\\n", " ")[:300])
        return r
    except Exception as e:
        log(name, "ERR", repr(e)[:200])
# CFTC legacy futures-only (Socrata)
get("cftc_legacy", "https://publicreporting.cftc.gov/resource/6dca-aqww.json", params={"$where": "cftc_contract_market_code='097741'", "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 2})
get("cftc_tff", "https://publicreporting.cftc.gov/resource/gpe5-46if.json", params={"$where": "cftc_contract_market_code='13874A'", "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 1})
# JPX
get("jpx_investor_type", "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html")
get("jpx_margin", "https://www.jpx.co.jp/markets/statistics-equities/margin/index.html")
get("jpx_investor_type_en", "https://www.jpx.co.jp/english/markets/statistics-equities/investor-type/index.html")
# AAII
get("aaii_xls", "https://www.aaii.com/files/surveys/sentiment.xls")
# Google News RSS
get("gnews", "https://news.google.com/rss/search", params={"q": "USD/JPY when:7d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
# Yahoo symbols
try:
    import yfinance as yf
    for s in "^VIX9D ^VIX ^VIX3M ^VIX6M ^SKEW ^JNIV ^VXJ SPLV SPHB XLY XLP AUDCHF=X ^NKVI.OS".split():
        try:
            d = yf.download(s, start="2005-01-01", progress=False, auto_adjust=True, threads=False)
            log("yahoo", s, len(d), str(d.index[0].date()) if len(d) else "-", str(d.index[-1].date()) if len(d) else "-")
        except Exception as e:
            log("yahoo", s, "ERR", repr(e)[:120])
except Exception:
    log(traceback.format_exc()[-300:])
# FRED graph CSV (no key) for Michigan sentiment
get("fred_umcsent", "https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "UMCSENT"})
open("data/mood_probe.txt", "w").write("\n".join(out))
