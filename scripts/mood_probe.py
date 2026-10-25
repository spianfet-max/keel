"""Probe 7: StockTwits history by id stepping, symbol names, rate limits."""
import requests, os, time
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = {"User-Agent": "keel-mood/0.1 (market sentiment research)"}
B = "https://api.stocktwits.com/api/2/streams/symbol/"
def page(sym, mx=None):
    r = requests.get(B + sym + ".json", params={"max": mx} if mx else {}, headers=H, timeout=30)
    rl = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower()}
    if r.status_code != 200: return r.status_code, None, rl
    return 200, r.json(), rl
for sym in ["SPY", "QQQ", "IWM", "SPX", "EWJ", "DXJ", "NIKKEI", "N225", "USDJPY", "USD.JPY", "JPY", "TOPIX", "7203.T", "TM"]:
    st, j, rl = page(sym)
    if j:
        ms = j["messages"]; s = [((m.get("entities") or {}).get("sentiment") or {}).get("basic") for m in ms]
        log(sym, st, len(ms), "bull", s.count("Bullish"), "bear", s.count("Bearish"), ms[-1]["created_at"] if ms else "", "->", ms[0]["created_at"] if ms else "", rl)
    else:
        log(sym, st, rl)
    time.sleep(1)
# id stepping: SPY pages at older ids
st, j, _ = page("SPY"); top = j["messages"][0]["id"]
for back in [1e5, 1e6, 5e6, 2e7, 5e7, 1e8, 2e8, 3e8, 4e8]:
    mx = int(top - back); st, j, rl = page("SPY", mx)
    if j and j["messages"]:
        ms = j["messages"]; s = [((m.get("entities") or {}).get("sentiment") or {}).get("basic") for m in ms]
        log("SPY max", mx, st, len(ms), ms[-1]["created_at"], "->", ms[0]["created_at"], "bull", s.count("Bullish"), "bear", s.count("Bearish"), rl)
    else: log("SPY max", mx, st, rl)
    time.sleep(1)
# volume: pages to cover 24h of SPY
mx = None; n = 0; t0 = None
for k in range(40):
    st, j, rl = page("SPY", mx)
    if not j or not j["messages"]: log("vol stop", st, rl); break
    ms = j["messages"]; n += len(ms); t0 = t0 or ms[0]["created_at"]; mx = ms[-1]["id"] - 1
    if ms[-1]["created_at"][:10] < t0[:10] and k > 2 and ms[-1]["created_at"][11:13] < t0[11:13]: break
    time.sleep(.6)
log("SPY 24h-ish pages", k + 1, "msgs", n, t0, "->", ms[-1]["created_at"], rl)
open("data/mood_probe.txt", "w").write("\n".join(out))
