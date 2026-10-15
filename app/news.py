"""News-based measures (after Bybee, Kelly, Manela and Xiu, "Business News and Business Cycles", JF 2024).

Two free sources, no keys:
  * San Francisco Fed Daily News Sentiment Index: sentiment of economics-related articles in
    24 major US newspapers, daily since 1980, updated about weekly (Shapiro, Sudhof and Wilson).
  * Google News RSS search: headlines about yen intervention, used to count daily coverage
    and to read how far official language has escalated (the MOF "verbal ladder").

GDELT would be the obvious source for news attention, but its API refuses most requests
(HTTP 429) at the time of writing, so it is not used.
"""

from __future__ import annotations

import io
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx
import pandas as pd

from .data import _pool, cached

log = logging.getLogger("keel.news")

SFFED_CSV = "https://www.frbsf.org/wp-content/uploads/news-sentiment-chart-1.csv"
GNEWS = "https://news.google.com/rss/search"
UA = {"User-Agent": "Mozilla/5.0 (keel market-data service)"}
JST = timezone(timedelta(hours=9))

# --------------------------------------------------------------------------- #
# SF Fed Daily News Sentiment
# --------------------------------------------------------------------------- #


def parse_sentiment_csv(text: str) -> pd.Series:
    """'date,Daily news sentiment' rows with MM/DD/YYYY dates -> daily Series."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip().lower() for c in df.columns]
    dcol = next(c for c in df.columns if "date" in c)
    vcol = next(c for c in df.columns if c != dcol)
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], format="%m/%d/%Y", errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


@cached(ttl=12 * 3600)
def news_sentiment() -> pd.Series:
    with httpx.Client(timeout=40, headers=UA, follow_redirects=True) as c:
        r = c.get(SFFED_CSV, params={"v": date.today().isoformat()})
        r.raise_for_status()
    s = parse_sentiment_csv(r.text)
    if s.empty:
        raise ValueError("SF Fed news sentiment: empty file")
    return s


# --------------------------------------------------------------------------- #
# Google News RSS
# --------------------------------------------------------------------------- #

LOCALE = {"ja": ("ja", "JP", "JP:ja"), "en": ("en-US", "US", "US:en")}


def parse_rss(text: str) -> list[dict]:
    """Google News RSS -> [{d (JST date), title, source, link}]. The source name is cut from the title."""
    out = []
    root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = (it.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3].strip()
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate") or "")
            d = dt.astimezone(JST).date().isoformat()
        except (TypeError, ValueError):
            continue
        out.append({"d": d, "title": title, "source": src, "link": (it.findtext("link") or "").strip()})
    return out


def _gnews(query: str, lang: str) -> list[dict]:
    hl, gl, ceid = LOCALE[lang]
    url = f"{GNEWS}?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
    for attempt in range(2):
        try:
            with httpx.Client(timeout=25, headers=UA, follow_redirects=True) as c:
                r = c.get(url)
            if r.status_code == 200:
                return parse_rss(r.text)
            if r.status_code in (429, 503) and attempt == 0:
                time.sleep(2.0)
                continue
            raise ValueError(f"Google News HTTP {r.status_code}")
        except httpx.HTTPError as e:
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise ValueError(f"Google News: {e}") from e
    return []


@cached(ttl=24 * 3600)
def _window_old(query: str, lang: str, a: str, b: str) -> list[dict]:
    return _gnews(f"{query} after:{a} before:{b}", lang)


@cached(ttl=3 * 3600)
def _window_recent(query: str, lang: str, a: str, b: str) -> list[dict]:
    return _gnews(f"{query} after:{a} before:{b}", lang)


def coverage(query: str, lang: str, days: int, step: int = 3) -> dict:
    """Daily headline counts over `days`, fetched in `step`-day windows (each window returns at
    most 100 items, so a window that hits 100 is flagged as capped). Items are de-duplicated by
    link and counted on their own publication date (JST)."""
    today = datetime.now(JST).date()
    start = today - timedelta(days=days)
    wins = []
    d = start
    while d <= today:
        e = min(d + timedelta(days=step), today + timedelta(days=1))
        wins.append((d.isoformat(), (e + timedelta(days=1)).isoformat()))
        d = e
    recent_cut = (today - timedelta(days=4)).isoformat()

    def fetch(w):
        a, b = w
        f = _window_recent if b >= recent_cut else _window_old
        return f(query, lang, a, b)

    got = _pool(fetch, wins, workers=3)
    seen, counts, capped, failed = set(), {}, [], 0
    for w in wins:
        items = got.get(w)
        if items is None:
            failed += 1
            continue
        if len(items) >= 100:
            capped.append(w[0])
        for it in items:
            k = it["link"] or (it["d"] + it["title"])
            if k in seen or not (start.isoformat() <= it["d"] <= today.isoformat()):
                continue
            seen.add(k)
            counts[it["d"]] = counts.get(it["d"], 0) + 1
    series = []
    d = start
    while d <= today:
        series.append({"d": d.isoformat(), "n": counts.get(d.isoformat(), 0)})
        d += timedelta(days=1)
    return {"query": query, "lang": lang, "series": series, "capped_windows": capped,
            "windows": len(wins), "failed_windows": failed}


# --------------------------------------------------------------------------- #
# The verbal ladder: how far official language on the yen has escalated
# --------------------------------------------------------------------------- #

# rung -> (score 0-100, patterns). The highest rung a headline matches wins.
LADDER = [
    (5, 100, r"レートチェック|rate[- ]check|覆面介入|stealth intervention|介入に踏み切|介入を実施した|intervened"),
    (4, 80, r"断固たる|断固とした|あらゆる手段|あらゆる選択肢|排除しない|排除せず|躊躇なく|ちゅうちょなく|decisive action|decisive steps|all options|rule out any|won.?t rule out|not rule out"),
    (3, 55, r"過度な変動|投機的|行き過ぎ|適切に対応|適切な対応|一方的な|急速な円安|excessive|speculative|one-sided|appropriate action|rapid moves"),
    (2, 30, r"緊張感|高い緊張感|注視|sense of urgency|closely watch|watching .{0,20}closely"),
]
LADDER_RX = [(rung, score, re.compile(p, re.I)) for rung, score, p in LADDER]
# A headline must be about the yen / FX policy. (Not "finance minister" alone: other countries have them too.)
FX_RX = re.compile(r"為替|円安|円相場|ドル円|ドル・円|yen|USD/JPY|USDJPY|forex|レートチェック|rate[- ]check|介入|intervention|財務官|Mimura|currency diplomat", re.I)
# Reports that something did NOT happen, and talk about yen strength, are not escalation.
NEG_RX = re.compile(r"伝わらず|見送|否定|していない|ではない|なかった|確認されず|no sign|not conducted|denied|denies", re.I)
YEN_UP_RX = re.compile(r"円高")
YEN_DOWN_RX = re.compile(r"円安|円売り|ドル高")

# One search per rung, so frequent rate-check stories do not crowd out rarer warnings (each search returns at most 100).
LADDER_QUERIES = {
    "ja": {
        5: "為替 (レートチェック OR 覆面介入 OR 介入に踏み切)",
        4: "為替 (断固たる OR あらゆる手段 OR 排除しない OR 躊躇なく)",
        3: "為替 (過度な変動 OR 投機的 OR 行き過ぎ OR 適切に対応)",
        2: "為替 緊張感",
    },
    "en": {
        5: 'yen ("rate check" OR "stealth intervention")',
        4: 'yen ("decisive action" OR "all options" OR "rule out")',
        3: 'yen (excessive OR speculative OR "one-sided")',
        2: 'yen "sense of urgency"',
    },
}


def classify(title: str) -> tuple[int, int] | None:
    if not FX_RX.search(title) or NEG_RX.search(title):
        return None
    if YEN_UP_RX.search(title) and not YEN_DOWN_RX.search(title):
        return None
    for rung, score, rx in LADDER_RX:
        if rx.search(title):
            return rung, score
    return None


@cached(ttl=3 * 3600)
def _ladder_search(query: str, lang: str, days: int) -> list[dict]:
    return _gnews(f"{query} when:{days}d", lang)


def ladder_headlines(days: int = 30) -> dict:
    jobs = [(lang, rung, q) for lang, qs in LADDER_QUERIES.items() for rung, q in qs.items()]
    got = _pool(lambda j: _ladder_search(j[2], j[0], days), jobs, workers=3)
    out, errors = [], []
    for j in jobs:
        items = got.get(j)
        if items is None:
            errors.append(f"{j[0]} rung {j[1]}: no answer")
            continue
        for it in items:
            c = classify(it["title"])
            if c:
                out.append({**it, "lang": j[0], "rung": c[0], "score": c[1]})
    seen, dedup = set(), []
    for it in sorted(out, key=lambda x: (x["d"], x["rung"]), reverse=True):
        if it["title"] not in seen:
            seen.add(it["title"])
            dedup.append(it)
    return {"items": dedup, "errors": errors}
