"""Probe 2: JPX file links and layouts. Writes data/mood_probe.txt."""
import requests, os, re, io, traceback
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = {"User-Agent": "Mozilla/5.0 (keel research)"}
B = "https://www.jpx.co.jp"
def links(url, pat=r'href="([^"]+\.(?:xls|xlsx|csv|pdf|html))"'):
    r = requests.get(url, headers=H, timeout=60); r.encoding = "utf-8"
    ls = re.findall(pat, r.text)
    return r.text, ls
try:
    for url in [B + "/markets/statistics-equities/investor-type/index.html", B + "/markets/statistics-equities/margin/index.html"]:
        t, ls = links(url)
        log("PAGE", url, len(ls)); [log("  ", l) for l in ls[:40]]
        tl = re.findall(r'<title>(.*?)</title>', t); log("  title", tl[:1])
    # archives
    for url in [B + "/markets/statistics-equities/investor-type/00-00-archives-00.html", B + "/markets/statistics-equities/margin/00-archives-01.html", B + "/markets/statistics-equities/margin/04.html", B + "/markets/statistics-equities/margin/05.html"]:
        try:
            t, ls = links(url); log("ARCH", url, len(ls)); [log("  ", l) for l in ls[:15]]
        except Exception as e:
            log("ARCH", url, "ERR", repr(e)[:100])
except Exception:
    log(traceback.format_exc()[-500:])
# open one investor-type weekly file and one margin file
try:
    import pandas as pd
    t, ls = links(B + "/markets/statistics-equities/investor-type/index.html")
    x = [l for l in ls if re.search(r'stock_val_1_\d+\.xls', l)] or [l for l in ls if l.endswith(('.xls', '.xlsx'))]
    if x:
        u = x[0] if x[0].startswith("http") else B + x[0]; log("OPEN", u)
        df = pd.read_excel(io.BytesIO(requests.get(u, headers=H, timeout=60).content), header=None)
        for i in range(min(60, len(df))):
            row = [str(v)[:14] for v in df.iloc[i].tolist()[:12] if str(v) != "nan"]
            if row: log("  r%02d" % i, " | ".join(row))
    t, ls = links(B + "/markets/statistics-equities/margin/index.html")
    x = [l for l in ls if l.endswith(('.xls', '.xlsx'))]
    if x:
        u = x[0] if x[0].startswith("http") else B + x[0]; log("OPEN", u)
        df = pd.read_excel(io.BytesIO(requests.get(u, headers=H, timeout=60).content), header=None)
        for i in range(min(45, len(df))):
            row = [str(v)[:14] for v in df.iloc[i].tolist()[:12] if str(v) != "nan"]
            if row: log("  m%02d" % i, " | ".join(row))
except Exception:
    log(traceback.format_exc()[-800:])
open("data/mood_probe.txt", "w").write("\n".join(out))
