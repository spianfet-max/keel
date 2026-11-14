"""Earnings revisions for Rineichen: analysts' EPS estimates now and 7/30/60/90 days ago (Yahoo
`earningsTrend`), for baskets of large companies standing in for world sectors, regions, TOPIX-17
sectors and the gaming peers.

Per company: the 12-month-forward EPS (current and next fiscal year blended by the time left in the
current year), its change over 1 and 2 months, and how many analysts raised or cut in the last 30 days.
Per group: the median 2-month revision, the share of companies revised up and down (the asymmetry),
the net share of analysts raising, and roughly how many weeks ago the direction of revisions last turned.

The network part (`fetch_trend`) is separate from the pure aggregation so tests can feed fixtures.
"""
from __future__ import annotations

import logging
import math
import statistics as st
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

log = logging.getLogger("keel.revisions")

# ------------------------------------------------------------------ universe
# group key -> (English, Japanese, kind, tickers). Keys of the world sectors match the Yahoo sector keys the
# page already uses for ETF weights; TOPIX groups use the NEXT FUNDS ETF code of the sector.
GROUPS: dict[str, tuple[str, str, str, list[str]]] = {
    "energy": ("Energy", "エネルギー", "sector", ["XOM", "CVX", "COP", "SHEL", "TTE", "BP", "EOG", "SLB", "5020.T", "1605.T"]),
    "basic_materials": ("Materials", "素材", "sector", ["LIN", "BHP", "RIO", "SHW", "APD", "FCX", "NEM", "ECL", "4063.T", "BAS.DE"]),
    "industrials": ("Industrials", "資本財", "sector", ["GE", "CAT", "RTX", "HON", "UNP", "ETN", "SIE.DE", "AIR.PA", "6501.T", "8058.T"]),
    "consumer_cyclical": ("Consumer discretionary", "一般消費財", "sector", ["AMZN", "TSLA", "HD", "MCD", "LOW", "BKNG", "NKE", "MC.PA", "7203.T", "9983.T"]),
    "consumer_defensive": ("Consumer staples", "生活必需品", "sector", ["WMT", "PG", "KO", "PEP", "COST", "PM", "NESN.SW", "ULVR.L", "OR.PA", "2914.T"]),
    "healthcare": ("Health care", "ヘルスケア", "sector", ["LLY", "UNH", "JNJ", "ABBV", "MRK", "NVO", "AZN", "ROG.SW", "NOVN.SW", "4502.T"]),
    "financial_services": ("Financials", "金融", "sector", ["JPM", "BAC", "V", "MA", "GS", "BRK-B", "HSBA.L", "ALV.DE", "8306.T", "8316.T"]),
    "technology": ("Information technology", "情報技術", "sector", ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "TSM", "ASML", "SAP", "8035.T", "005930.KS"]),
    "communication_services": ("Communication services", "コミュニケーション", "sector", ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "0700.HK", "9432.T", "9984.T", "7974.T"]),
    "utilities": ("Utilities", "公益", "sector", ["NEE", "SO", "DUK", "CEG", "AEP", "IBE.MC", "ENEL.MI", "NG.L", "9503.T", "9502.T"]),
    "realestate": ("Real estate", "不動産", "sector", ["PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "VNA.DE", "0016.HK", "8801.T", "8802.T"]),
    "1617.T": ("Foods", "食品", "topix", ["2914.T", "2802.T", "2502.T", "2503.T"]),
    "1618.T": ("Energy resources", "エネルギー資源", "topix", ["5020.T", "1605.T", "5019.T", "5021.T"]),
    "1619.T": ("Construction & materials", "建設・資材", "topix", ["1925.T", "1928.T", "1801.T", "5201.T"]),
    "1620.T": ("Raw materials & chemicals", "素材・化学", "topix", ["4063.T", "4452.T", "3407.T", "4901.T"]),
    "1621.T": ("Pharmaceutical", "医薬品", "topix", ["4502.T", "4568.T", "4519.T", "4503.T"]),
    "1622.T": ("Autos & transport equipment", "自動車・輸送機", "topix", ["7203.T", "7267.T", "6902.T", "7269.T"]),
    "1623.T": ("Steel & nonferrous", "鉄鋼・非鉄", "topix", ["5401.T", "5411.T", "5802.T", "5713.T"]),
    "1624.T": ("Machinery", "機械", "topix", ["6367.T", "6301.T", "6273.T", "7011.T"]),
    "1625.T": ("Electric & precision", "電機・精密", "topix", ["6758.T", "6501.T", "8035.T", "6861.T"]),
    "1626.T": ("IT & services", "情報通信・サービス", "topix", ["9432.T", "9433.T", "9984.T", "7974.T"]),
    "1627.T": ("Electric power & gas", "電力・ガス", "topix", ["9503.T", "9501.T", "9502.T", "9531.T"]),
    "1628.T": ("Transport & logistics", "運輸・物流", "topix", ["9022.T", "9020.T", "9101.T", "9104.T"]),
    "1629.T": ("Trading & wholesale", "商社・卸売", "topix", ["8058.T", "8031.T", "8001.T", "8053.T"]),
    "1630.T": ("Retail", "小売", "topix", ["9983.T", "3382.T", "8267.T", "7532.T"]),
    "1631.T": ("Banks", "銀行", "topix", ["8306.T", "8316.T", "8411.T", "8308.T"]),
    "1632.T": ("Financials ex banks", "金融（除く銀行）", "topix", ["8766.T", "8591.T", "8604.T", "8725.T"]),
    "1633.T": ("Real estate", "不動産", "topix", ["8801.T", "8802.T", "8830.T", "3289.T"]),
    "gaming": ("Gaming peers", "ゲーム関連", "peers", ["7974.T", "6758.T", "9697.T", "7832.T", "9766.T", "6460.T", "9684.T", "3635.T", "TTWO", "NTES"]),
}
# Regions are built from every company in the world-sector baskets.
REGIONS = {"US": ("United States", "米国"), "EU": ("Europe", "欧州"), "JP": ("Japan", "日本"), "AS": ("Asia ex Japan", "日本除くアジア")}
_REGION_OVERRIDE = {"SHEL": "EU", "TTE": "EU", "BP": "EU", "BHP": "AS", "RIO": "EU", "NVO": "EU", "AZN": "EU",
                    "TSM": "AS", "ASML": "EU", "SAP": "EU", "NTES": "AS", "LIN": "US"}


def region(t: str) -> str:
    if t in _REGION_OVERRIDE:
        return _REGION_OVERRIDE[t]
    if t.endswith(".T"):
        return "JP"
    if t.endswith((".HK", ".KS", ".TW", ".SS", ".SZ")):
        return "AS"
    if "." in t and not t.startswith("BRK"):
        return "EU"
    return "US"


def all_tickers() -> list[str]:
    seen: list[str] = []
    for g in GROUPS.values():
        for t in g[3]:
            if t not in seen:
                seen.append(t)
    return seen


# ------------------------------------------------------------------ fetching
PTS = ("current", "7daysago", "30daysago", "60daysago", "90daysago")


def _f(v):
    if isinstance(v, dict):
        v = v.get("raw")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_trend(t: str, raw: list) -> dict | None:
    """Yahoo's earningsTrend.trend list -> EPS now and 7/30/60/90 days ago for the current (0y) and next (+1y)
    fiscal year, analysts raising/cutting in the last 30 days, analyst count and the fiscal-year end."""
    out: dict = {"t": t}
    for it in raw or []:
        p = it.get("period")
        k = {"0y": "fy0", "+1y": "fy1"}.get(p)
        if not k:
            continue
        et = {str(a).lower(): b for a, b in (it.get("epsTrend") or {}).items()}
        rv = {str(a).lower(): b for a, b in (it.get("epsRevisions") or {}).items()}
        es = {str(a).lower(): b for a, b in (it.get("earningsEstimate") or {}).items()}
        pts = [_f(et.get(c)) for c in PTS]
        if all(v is None for v in pts):
            continue
        out[k] = pts
        out[k + "_up"] = _f(rv.get("uplast30days"))
        out[k + "_dn"] = _f(rv.get("downlast30days"))
        out[k + "_n"] = _f(es.get("numberofanalysts"))
        if k == "fy0" and it.get("endDate"):
            out["fy0_end"] = str(it["endDate"])[:10]
    return out if "fy0" in out or "fy1" in out else None


def fetch_trend(t: str) -> dict | None:
    """One Yahoo quoteSummary call (module earningsTrend) through yfinance."""
    import yfinance as yf  # noqa: PLC0415

    an = yf.Ticker(t)._analysis  # noqa: SLF001
    an._fetch_earnings_trend()  # noqa: SLF001
    return parse_trend(t, an._earnings_trend)  # noqa: SLF001


# ------------------------------------------------------------------ maths
def _chg(new, old):
    if new is None or old is None or old == 0:
        return None
    c = (new - old) / abs(old)
    return max(-0.5, min(0.5, c))


def forward(rec: dict, today: date | None = None) -> list | None:
    """12-month-forward EPS at each of the five points: FY0 and FY1 blended by the share of FY0 still ahead."""
    f0, f1 = rec.get("fy0"), rec.get("fy1")
    if not f1 and not f0:
        return None
    w0 = 0.5
    end = rec.get("fy0_end")
    if end:
        try:
            d = datetime.strptime(end, "%Y-%m-%d").date()
            w0 = max(0.0, min(1.0, ((d - (today or date.today())).days) / 365.0))
        except ValueError:
            pass
    out = []
    for i in range(5):
        a = f0[i] if f0 else None
        b = f1[i] if f1 else None
        if a is None and b is None:
            out.append(None)
        elif a is None:
            out.append(b)
        elif b is None:
            out.append(a)
        else:
            out.append(w0 * a + (1 - w0) * b)
    return out


def company(rec: dict, today: date | None = None) -> dict:
    fw = forward(rec, today) or [None] * 5
    up = sum(v or 0 for v in (rec.get("fy0_up"), rec.get("fy1_up")))
    dn = sum(v or 0 for v in (rec.get("fy0_dn"), rec.get("fy1_dn")))
    return {"t": rec["t"], "region": region(rec["t"]), "fwd": fw[0],
            "r1m": _chg(fw[0], fw[2]), "r2m": _chg(fw[0], fw[3]), "r3m": _chg(fw[0], fw[4]),
            # change within each window, newest first: 0-1w, 1-4w, 4-9w, 9-13w
            "seg": [_chg(fw[0], fw[1]), _chg(fw[1], fw[2]), _chg(fw[2], fw[3]), _chg(fw[3], fw[4])],
            "up": up, "dn": dn, "n": max(rec.get("fy0_n") or 0, rec.get("fy1_n") or 0) or None}


FLAT = 0.001  # a change smaller than 0.1% counts as unchanged
SEG_START = (0, 1, 4, 9)  # weeks ago at which each window ends (newest first)


def _breadth(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return (sum(x > FLAT for x in v) - sum(x < -FLAT for x in v)) / len(v)


def _sign(b, dead=0.1):
    return 0 if b is None or abs(b) < dead else (1 if b > 0 else -1)


def group_stats(cos: list[dict]) -> dict | None:
    cos = [c for c in cos if c.get("r2m") is not None]
    if not cos:
        return None
    r2 = [c["r2m"] for c in cos]
    r1 = [c["r1m"] for c in cos if c["r1m"] is not None]
    seg = [_breadth([c["seg"][i] for c in cos]) for i in range(4)]
    up, dn = sum(c["up"] for c in cos), sum(c["dn"] for c in cos)
    # Direction now: breadth of 1-month revisions (steadier than the last week alone).
    now = _sign(_breadth(r1)) or _sign(_breadth(r2))
    turn = None
    if now:
        for i in range(1, 4):
            if _sign(seg[i]) == -now:
                turn = SEG_START[i]
                break
    return {"n": len(cos), "rev2m": st.median(r2), "rev1m": st.median(r1) if r1 else None,
            "mean2m": sum(r2) / len(r2),
            "up_share": sum(x > FLAT for x in r2) / len(cos), "dn_share": sum(x < -FLAT for x in r2) / len(cos),
            "an_up": up, "an_dn": dn, "net_analysts": (up - dn) / (up + dn) if up + dn else None,
            "seg_breadth": seg, "direction": now, "weeks_since_turn": turn,
            "turn_note": None if not now else ("13+" if turn is None else None)}


def build(records: dict[str, dict | None], today: date | None = None) -> dict:
    cos = {t: company(r, today) for t, r in records.items() if r}
    groups = {}
    for k, (en, ja, kind, ts) in GROUPS.items():
        g = group_stats([cos[t] for t in ts if t in cos])
        if g:
            groups[k] = {"name": en, "ja": ja, "kind": kind, "members": [t for t in ts if t in cos], **g}
    world = [t for k, v in GROUPS.items() if v[2] == "sector" for t in v[3]]
    world = list(dict.fromkeys(world))
    for rk, (en, ja) in REGIONS.items():
        g = group_stats([cos[t] for t in world if t in cos and cos[t]["region"] == rk])
        if g:
            groups["reg_" + rk] = {"name": en, "ja": ja, "kind": "region", **g}
    g = group_stats([cos[t] for t in world if t in cos])
    if g:
        groups["world"] = {"name": "World (all baskets)", "ja": "世界（全バスケット）", "kind": "region", **g}
    return {"asof": str(today or date.today()), "groups": groups, "companies": cos,
            "missing": [t for t, r in records.items() if not r]}


# ------------------------------------------------------------------ cache + background fetch
_LOCK = threading.Lock()
_STATE: dict = {"data": None, "at": 0.0, "running": False, "error": None}
TTL = 12 * 3600


def _run(fetch=fetch_trend, workers: int = 6):
    try:
        ts = all_tickers()
        recs: dict = {}

        def one(t):
            for attempt in range(2):
                try:
                    return fetch(t)
                except Exception as e:  # noqa: BLE001
                    if attempt:
                        log.warning("revisions %s: %s", t, e)
                    time.sleep(1.5)
            return None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for t, r in zip(ts, ex.map(one, ts)):
                recs[t] = r
        data = build(recs)
        if len(data["companies"]) < len(ts) * 0.3:
            raise RuntimeError(f"only {len(data['companies'])} of {len(ts)} companies answered (Yahoo throttling?)")
        with _LOCK:
            _STATE.update(data=data, at=time.time(), error=None)
    except Exception as e:  # noqa: BLE001
        log.warning("revisions fetch failed: %s", e)
        with _LOCK:
            _STATE["error"] = str(e)
    finally:
        with _LOCK:
            _STATE["running"] = False


def get(wait: float = 45.0, fresh: bool = False, fetch=fetch_trend) -> dict:
    """Cached payload; starts a background fetch when stale and waits up to `wait` seconds for it.
    Returns {'status': 'preparing'} if still running (callers retry), the stale copy if one exists."""
    with _LOCK:
        stale = _STATE["data"] is None or time.time() - _STATE["at"] > TTL or fresh
        if stale and not _STATE["running"]:
            _STATE["running"] = True
            threading.Thread(target=_run, kwargs={"fetch": fetch}, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < wait:
        with _LOCK:
            if not _STATE["running"]:
                break
        time.sleep(0.5)
    with _LOCK:
        if _STATE["data"] is not None and (not _STATE["running"] or not fresh):
            return {**_STATE["data"], "status": "ok" if not _STATE["running"] else "ok_refreshing"}
        if _STATE["running"]:
            return {"status": "preparing", "note": "Collecting analyst estimates for about 170 companies; ask again in 20 seconds."}
        raise RuntimeError(_STATE["error"] or "no data")
