"""Probe 3: JPX investor-type columns and archive links."""
import requests, os, re, io, traceback
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = {"User-Agent": "Mozilla/5.0 (keel research)"}
B = "https://www.jpx.co.jp"
try:
    import pandas as pd
    u = B + "/markets/statistics-equities/investor-type/t13vrt000002havh-att/stock_1_w_20261013_20261016.xlsx"
    xl = pd.ExcelFile(io.BytesIO(requests.get(u, headers=H, timeout=60).content))
    log("sheets", xl.sheet_names)
    df = xl.parse(xl.sheet_names[0], header=None)
    log("shape", df.shape)
    for i in range(0, 9):
        log("ROW", i, " || ".join(f"{j}:{str(v)[:22]}" for j, v in enumerate(df.iloc[i].tolist()) if str(v) != "nan"))
    for i in (7, 8, 13, 14):
        log("DATA", i, " || ".join(f"{j}:{str(v)[:14]}" for j, v in enumerate(df.iloc[i].tolist()) if str(v) != "nan"))
    for page in ["/markets/statistics-equities/investor-type/00-00-archives-00.html", "/markets/statistics-equities/investor-type/00-00-archives-01.html", "/markets/statistics-equities/investor-type/00-00-archives-05.html", "/markets/statistics-equities/investor-type/00-00-archives-10.html", "/markets/statistics-equities/investor-type/00-00-archives-14.html"]:
        try:
            r = requests.get(B + page, headers=H, timeout=60); r.encoding = "utf-8"
            ls = re.findall(r'href="([^"]*stock[^"]*_w_[^"]*\.(?:xls|xlsx))"', r.text)
            other = re.findall(r'href="(/markets/statistics-equities/investor-type/00-00-archives-\d+\.html)"', r.text)
            log("ARCH", page, r.status_code, len(ls), ls[:3], ls[-2:], "pages:", sorted(set(other))[:30])
        except Exception as e:
            log("ARCH", page, "ERR", repr(e)[:100])
except Exception:
    log(traceback.format_exc()[-800:])
open("data/mood_probe.txt", "w").write("\n".join(out))
