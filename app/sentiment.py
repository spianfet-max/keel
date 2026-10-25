"""Positioning and flow data for sentiment work. Free sources, no keys.

  * CFTC Commitments of Traders (legacy, futures only) via the CFTC's Socrata API: weekly
    large-speculator (non-commercial) longs and shorts per market since 1986.
  * Japan Ministry of Finance, weekly portfolio flows (指定報告機関ベース): non-residents' net
    purchases of Japanese equities and long-term bonds, and residents' net purchases of foreign
    equities, weekly since 2005.
  * JPX weekly margin balances (信用取引現在高): margin longs and shorts for Tokyo and Nagoya,
    weekly since 2002, plus the latest weekly file; the ratio is the 信用倍率.
  * JPX weekly trading by investor type (投資部門別売買状況, TSE Prime, value): net buying by
    foreigners and individuals for the last few weeks the site lists.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date

import httpx
import pandas as pd

from .data import cached

log = logging.getLogger("keel.sentiment")
UA = {"User-Agent": "Mozilla/5.0 (keel market-data service)"}

# --------------------------------------------------------------------------- #
# CFTC
# --------------------------------------------------------------------------- #

COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
COT_MARKETS = {
    # key: (CFTC contract market code, English, Japanese)
    "JPY": ("097741", "Japanese yen", "円"),
    "EUR": ("099741", "Euro", "ユーロ"),
    "GBP": ("096742", "British pound", "英ポンド"),
    "AUD": ("232741", "Australian dollar", "豪ドル"),
    "CHF": ("092741", "Swiss franc", "スイスフラン"),
    "CAD": ("090741", "Canadian dollar", "カナダドル"),
    "MXN": ("095741", "Mexican peso", "メキシコペソ"),
    "SPX": ("13874A", "S&P 500 E-mini", "S&P500ミニ"),
    "NDX": ("209742", "Nasdaq-100 E-mini", "ナスダック100ミニ"),
    "NKY": ("240741", "Nikkei 225 (USD)", "日経225（ドル建）"),
    "TY": ("043602", "US 10-year note", "米10年債"),
    "TU": ("042601", "US 2-year note", "米2年債"),
    "GC": ("088691", "Gold", "金"),
    "CL": ("067651", "WTI crude", "WTI原油"),
    "HG": ("085692", "Copper", "銅"),
    "VX": ("1170E1", "VIX futures", "VIX先物"),
    "BTC": ("133741", "Bitcoin (CME)", "ビットコイン（CME）"),
}
COT_FIELDS = ["report_date_as_yyyy_mm_dd", "cftc_contract_market_code", "open_interest_all",
              "noncomm_positions_long_all", "noncomm_positions_short_all",
              "comm_positions_long_all", "comm_positions_short_all"]


def parse_cot(rows: list[dict]) -> dict[str, list[dict]]:
    """Socrata rows -> {key: [{d, oi, ncl, ncs, net, pct}]} sorted by date; pct = net / OI in %."""
    code2key = {v[0]: k for k, v in COT_MARKETS.items()}
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        k = code2key.get(str(r.get("cftc_contract_market_code", "")).strip())
        if not k:
            continue
        try:
            d = str(r["report_date_as_yyyy_mm_dd"])[:10]
            oi = float(r["open_interest_all"])
            lo, sh = float(r["noncomm_positions_long_all"]), float(r["noncomm_positions_short_all"])
        except (KeyError, TypeError, ValueError):
            continue
        cl, cs = r.get("comm_positions_long_all"), r.get("comm_positions_short_all")
        rec = {"d": d, "oi": oi, "ncl": lo, "ncs": sh, "net": lo - sh,
               "pct": round(100 * (lo - sh) / oi, 2) if oi > 0 else None}
        if cl is not None and cs is not None:
            rec["cnet"] = float(cl) - float(cs)
        out.setdefault(k, {})[d] = rec  # one row per date (last wins)
    return {k: [v[d] for d in sorted(v)] for k, v in out.items()}


@cached(ttl=6 * 3600)
def cot(keys: tuple[str, ...], start: str) -> dict[str, list[dict]]:
    codes = ",".join(f"'{COT_MARKETS[k][0]}'" for k in keys)
    params = {"$select": ",".join(COT_FIELDS),
              "$where": f"cftc_contract_market_code in({codes}) AND report_date_as_yyyy_mm_dd >= '{start}T00:00:00'",
              "$order": "report_date_as_yyyy_mm_dd", "$limit": "50000"}
    with httpx.Client(timeout=60, headers=UA, follow_redirects=True) as c:
        r = c.get(COT_URL, params=params)
        r.raise_for_status()
    return parse_cot(r.json())


# --------------------------------------------------------------------------- #
# MOF weekly portfolio flows
# --------------------------------------------------------------------------- #

MOF_WEEK = "https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/week.csv"
_NUM = re.compile(r"^-?[\d,]+\.?\d*$")


def _period_end(p: str) -> str | None:
    """'2005．1．2〜 1．8' -> '2005-01-08'; a week that crosses the year ('2025．12．28〜 1．3') rolls over."""
    p = p.replace("．", ".").replace("〜", "~").replace("～", "~").replace(" ", "").replace("　", "")
    m = re.match(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})~(?:(\d{4})\.)?(\d{1,2})\.(\d{1,2})", p)
    if not m:
        return None
    y, m1 = int(m.group(1)), int(m.group(2))
    y2 = int(m.group(4)) if m.group(4) else y
    m2, d2 = int(m.group(5)), int(m.group(6))
    if not m.group(4) and m2 < m1:
        y2 = y + 1
    try:
        return date(y2, m2, d2).isoformat()
    except ValueError:
        return None


def parse_mof_week(text: str) -> list[dict]:
    """Rows -> [{d (week end), fe (non-residents' net buying of Japanese equities), fb (of long-term
    bonds), re (residents' net buying of foreign equities)}] in 100 million yen."""
    import csv  # noqa: PLC0415

    out = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 18 or not row[0].strip():
            continue
        d = _period_end(row[0])
        if not d:
            continue

        def num(i):
            s = row[i].strip().replace(",", "").replace("−", "-").replace("△", "-").replace("▲", "-")
            return float(s) if s and _NUM.match(s) else None

        fe, fb, re_ = num(14), num(17), num(3)
        if fe is None:
            continue
        out.append({"d": d, "fe": fe, "fb": fb, "re": re_})
    out.sort(key=lambda x: x["d"])
    return out


@cached(ttl=12 * 3600)
def mof_week() -> list[dict]:
    with httpx.Client(timeout=60, headers=UA, follow_redirects=True) as c:
        r = c.get(MOF_WEEK)
        r.raise_for_status()
    return parse_mof_week(r.content.decode("shift_jis", "replace"))


# --------------------------------------------------------------------------- #
# JPX margin balances and investor-type trading
# --------------------------------------------------------------------------- #

JPX = "https://www.jpx.co.jp"
MARGIN_HIST_PAGE = JPX + "/markets/statistics-equities/margin/05.html"
MARGIN_WEEK_PAGE = JPX + "/markets/statistics-equities/margin/04.html"
INVESTOR_PAGE = JPX + "/markets/statistics-equities/investor-type/00-00-archives-00.html"


def _get(c: httpx.Client, url: str) -> httpx.Response:
    r = c.get(url)
    r.raise_for_status()
    return r


def _links(html: str, pattern: str) -> list[str]:
    return [JPX + h if h.startswith("/") else h for h in re.findall(r'href="([^"]*' + pattern + r')"', html)]


def parse_margin_hist(df: pd.DataFrame) -> list[dict]:
    """信用取引現在高 history sheet (header=None): date, short shares, short value, long shares, long value.
    Takes the Tokyo & Nagoya block (the first one)."""
    out = []
    for _, r in df.iterrows():
        d = pd.to_datetime(r.iloc[0], errors="coerce")
        if pd.isna(d):
            if out and str(r.iloc[0]).startswith("["):
                break  # next market block
            continue
        try:
            ss, sv, ls, lv = (float(r.iloc[i]) for i in (1, 2, 3, 4))
        except (TypeError, ValueError):
            continue
        out.append({"d": str(d.date()), "ss": ss, "sv": sv, "ls": ls, "lv": lv})
    return out


def parse_margin_week(df: pd.DataFrame) -> dict | None:
    """Weekly 信用取引現在高 file: date in row 0, the 合計 Total row has short shares in col 4 and long in col 6.
    Values are on the next row."""
    d = pd.to_datetime(df.iloc[0, 1], errors="coerce")
    for i in range(len(df) - 1):
        if "合計" in str(df.iloc[i, 1]):
            try:
                return {"d": str(d.date()), "ss": float(df.iloc[i, 4]), "ls": float(df.iloc[i, 6]),
                        "sv": float(df.iloc[i + 1, 4]), "lv": float(df.iloc[i + 1, 6])}
            except (TypeError, ValueError):
                return None
    return None


def parse_investor_week(df: pd.DataFrame, fname: str) -> dict | None:
    """TSE Prime value row: foreigners = cols 21 + 25 (institutions + individuals), Japanese individuals =
    cols 13 + 17 (cash + margin), trust banks col 53, investment trusts col 33. Thousand yen -> 100m yen."""
    m = re.search(r"_w_(\d{8})_(\d{8})", fname)
    if not m:
        return None
    for i in range(len(df)):
        if "金額" in str(df.iloc[i, 2]) and ("プライム" in str(df.iloc[i - 1, 1]) or "Prime" in str(df.iloc[i - 1, 1])):
            v = lambda j: float(df.iloc[i, j]) / 1e5  # noqa: E731
            try:
                return {"d": pd.Timestamp(m.group(2)).date().isoformat(), "foreign": round(v(21) + v(25), 1),
                        "indiv": round(v(13) + v(17), 1), "trust_bk": round(v(53), 1), "inv_trust": round(v(33), 1)}
            except (TypeError, ValueError, IndexError):
                return None
    return None


@cached(ttl=12 * 3600)
def jpx_margin() -> list[dict]:
    with httpx.Client(timeout=60, headers=UA, follow_redirects=True) as c:
        hist_links = [u for u in _links(_get(c, MARGIN_HIST_PAGE).text, r"_mtgk\.xls") if u.endswith(".xls")]
        rows = []
        if hist_links:
            df = pd.read_excel(io.BytesIO(_get(c, hist_links[0]).content), header=None)
            rows = parse_margin_hist(df)
        last = rows[-1]["d"] if rows else "2000-01-01"
        for u in _links(_get(c, MARGIN_WEEK_PAGE).text, r"_mtcurrent\.xlsx")[:8]:
            try:
                w = parse_margin_week(pd.read_excel(io.BytesIO(_get(c, u).content), header=None))
            except Exception as e:  # noqa: BLE001
                log.warning("margin week %s: %s", u, e)
                continue
            if w and w["d"] > last:
                rows.append(w)
    rows.sort(key=lambda x: x["d"])
    return rows


@cached(ttl=12 * 3600)
def jpx_investors() -> list[dict]:
    out = []
    with httpx.Client(timeout=60, headers=UA, follow_redirects=True) as c:
        for u in _links(_get(c, INVESTOR_PAGE).text, r"stock_1_w_\d{8}_\d{8}\.xlsx")[:8]:
            try:
                w = parse_investor_week(pd.read_excel(io.BytesIO(_get(c, u).content), header=None), u)
            except Exception as e:  # noqa: BLE001
                log.warning("investor week %s: %s", u, e)
                continue
            if w:
                out.append(w)
    return sorted(out, key=lambda x: x["d"])
