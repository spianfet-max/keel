"""Probe 5: JPX margin history table."""
import requests, os, io, traceback
import pandas as pd
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = {"User-Agent": "Mozilla/5.0 (keel research)"}
B = "https://www.jpx.co.jp/markets/statistics-equities/margin/"
for f in ["tvdivq0000001rq1-att/tvdivq0000015969.xlsx", "tvdivq0000001rq1-att/20260925_mtgk.xls", "tvdivq0000001rk9-att/20261016_mtcurrent.xlsx"]:
    try:
        xl = pd.ExcelFile(io.BytesIO(requests.get(B + f, headers=H, timeout=60).content))
        log("FILE", f, xl.sheet_names)
        df = xl.parse(xl.sheet_names[0], header=None)
        log("shape", df.shape)
        rows = list(range(min(14, len(df)))) + list(range(max(14, len(df) - 4), len(df)))
        for i in rows:
            log("R", i, " | ".join(f"{j}:{str(v)[:18]}" for j, v in enumerate(df.iloc[i].tolist()) if str(v) != "nan")[:600])
    except Exception:
        log(f, traceback.format_exc()[-400:])
open("data/mood_probe.txt", "w").write("\n".join(out))
