"""Probe 4: MOF weekly portfolio flows, JPX archive + margin totals."""
import requests, os, re, io, traceback
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = {"User-Agent": "Mozilla/5.0 (keel research)"}
B = "https://www.jpx.co.jp"
for u in ["https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/week.csv",
          "https://www.mof.go.jp/english/policy/international_policy/reference/itn_transactions_in_securities/week.csv"]:
    try:
        r = requests.get(u, headers=H, timeout=60)
        t = r.content.decode("shift_jis", "replace")
        L = t.splitlines()
        log("MOF", u, r.status_code, len(L)); [log("  ", l[:220]) for l in L[:16]]; [log("  ", l[:220]) for l in L[-3:]]
    except Exception as e: log("MOF ERR", repr(e)[:150])
for page in ["/markets/statistics-equities/investor-type/00-00-archives-01.html", "/markets/statistics-equities/margin/index.html",
             "/markets/statistics-equities/margin/01.html", "/markets/statistics-equities/margin/02.html",
             "/markets/statistics-equities/margin/04.html", "/markets/statistics-equities/margin/05.html",
             "/markets/statistics-equities/margin/00-archives-01.html"]:
    try:
        r = requests.get(B + page, headers=H, timeout=60); r.encoding = "utf-8"
        ls = re.findall(r'href="([^"]+\.(?:xls|xlsx|csv|pdf))"', r.text)
        t = re.sub(r"<[^>]+>", " ", r.text); t = re.sub(r"\s+", " ", t)
        i = t.find("信用")
        log("PAGE", page, r.status_code, len(ls), ls[:6], "|", t[max(0, i-50):i+200])
    except Exception as e: log("PAGE", page, "ERR", repr(e)[:100])
open("data/mood_probe.txt", "w").write("\n".join(out))
