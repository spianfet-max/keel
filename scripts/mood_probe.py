"""Probe 6: StockTwits and Reddit public endpoints."""
import requests, os, json
os.makedirs("data", exist_ok=True)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
UAS = {"browser": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36",
       "bot": "keel-mood/0.1 (market sentiment research; contact spianfet-max on GitHub)"}
URLS = [
    "https://api.stocktwits.com/api/2/streams/symbol/SPY.json",
    "https://api.stocktwits.com/api/2/streams/symbol/EWJ.json",
    "https://api.stocktwits.com/api/2/trending/symbols.json",
    "https://www.reddit.com/r/wallstreetbets/hot.json?limit=25",
    "https://old.reddit.com/r/stocks/new.json?limit=25",
    "https://www.reddit.com/r/wallstreetbets/new/.rss",
    "https://api.reddit.com/r/investing/new?limit=25",
]
for u in URLS:
    for name, ua in UAS.items():
        try:
            r = requests.get(u, headers={"User-Agent": ua, "Accept": "application/json, application/rss+xml, */*"}, timeout=30)
            t = r.text
            info = ""
            if r.status_code == 200 and t.lstrip().startswith("{"):
                j = r.json()
                if "messages" in j:
                    ms = j["messages"]
                    s = [((m.get("entities") or {}).get("sentiment") or {}).get("basic") for m in ms]
                    info = f"msgs={len(ms)} bull={s.count('Bullish')} bear={s.count('Bearish')} first={ms[0]['created_at'] if ms else ''} last={ms[-1]['created_at'] if ms else ''} cursor={j.get('cursor')} | {ms[0]['body'][:80] if ms else ''}"
                elif "symbols" in j:
                    info = "trending " + ",".join(x["symbol"] for x in j["symbols"][:10])
                elif "data" in j:
                    ch = j["data"].get("children", [])
                    info = f"posts={len(ch)} | " + " / ".join(c["data"]["title"][:50] for c in ch[:3])
            log(r.status_code, name, u, len(t), info or t[:150].replace("\n", " "))
        except Exception as e:
            log("ERR", name, u, repr(e)[:150])
open("data/mood_probe.txt", "w").write("\n".join(out))
