"""Data access for Keel. Every outside call goes through here.

Most data comes from OpenBB's Open Data Platform (`from openbb import obb`):
  * prices, quotes, fundamentals, news, company and economic calendars -> yfinance extension
  * US Treasury curve -> Federal Reserve (H.15), no key needed
  * EUR AAA curve -> ECB, no key needed
  * CPI -> OECD, no key needed
Two gaps are filled directly:
  * JGB curve -> Japan Ministry of Finance CSV (not in OpenBB)
  * option implied vols -> the yfinance library that OpenBB's yfinance extension ships with

Results are cached in memory (Render's free instance sleeps; the first call after
waking is slow, later ones are instant).
"""

from __future__ import annotations

import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import wraps

import httpx
import numpy as np
import pandas as pd

from .analytics import fix_splits, to_percent

log = logging.getLogger("keel.data")

_obb = None


def obb():
    """Import OpenBB lazily so tests and cold start stay fast."""
    global _obb
    if _obb is None:
        from openbb import obb as _o  # noqa: PLC0415

        _obb = _o
        try:
            _o.user.preferences.output_type = "OBBject"
        except Exception:  # pragma: no cover
            pass
    return _obb


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

_cache: dict = {}
_lock = threading.Lock()


def _has_data(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return not v.empty
    if isinstance(v, dict):
        return any(_has_data(x) if isinstance(x, (dict, list)) else x is not None for x in v.values())
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    return True


def cached(ttl: int):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            with _lock:
                hit = _cache.get(key)
                if hit and now - hit[0] < ttl:
                    return hit[1]
            val = fn(*args, **kwargs)
            if _has_data(val):  # never cache an empty answer (e.g. Yahoo throttling)
                with _lock:
                    _cache[key] = (now, val)
            return val

        wrapper.cache_clear = lambda: _cache.clear()  # type: ignore[attr-defined]
        return wrapper

    return deco


def _df(result) -> pd.DataFrame:
    """OBBject -> flat DataFrame with a 'date' column when there is one."""
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        df = result
    elif hasattr(result, "to_dataframe"):
        df = result.to_dataframe()
    elif hasattr(result, "results"):
        df = pd.DataFrame([r.model_dump() if hasattr(r, "model_dump") else r for r in result.results])
    else:
        df = pd.DataFrame(result)
    if df.index.name in ("date", "Date") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
    return df


def _pool(fn, items, workers: int = 6) -> dict:
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {it: ex.submit(fn, it) for it in items}
        for it, f in futs.items():
            try:
                out[it] = f.result()
            except Exception as e:  # noqa: BLE001
                log.warning("fetch %s failed: %s", it, e)
                out[it] = None
    return out


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #


@cached(ttl=900)
def closes(symbol: str, years: float = 3) -> pd.Series:
    """Daily closes for any Yahoo symbol (stocks 7974.T, FX USDJPY=X, indices ^N225)."""
    start = (date.today() - timedelta(days=int(365.25 * years) + 7)).isoformat()
    res = obb().yfinance.equity.price.historical(symbol=symbol, start_date=start)
    df = _df(res)
    if df.empty or "close" not in df:
        raise ValueError(f"no prices for {symbol}")
    s = pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=pd.to_datetime(df["date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s.name = symbol
    if "=" not in symbol and not symbol.startswith("^"):
        s = fix_splits(s)
    return s


def many_closes(symbols: list[str], years: float = 3) -> dict[str, pd.Series]:
    got = _pool(lambda s: closes(s, years), symbols)
    return {k: v for k, v in got.items() if v is not None and len(v) > 0}


@cached(ttl=3600)
def metrics(symbols: tuple[str, ...]) -> dict[str, dict]:
    """Valuation and profile numbers per symbol (yfinance key metrics + quote)."""
    out: dict[str, dict] = {s: {} for s in symbols}
    for attempt in range(2):
        try:
            df = _df(obb().yfinance.equity.fundamental.metrics(symbol=",".join(symbols)))
            for _, row in df.iterrows():
                sym = row.get("symbol")
                if sym in out:
                    out[sym].update({k: _clean(v) for k, v in row.items()})
            break
        except Exception as e:  # noqa: BLE001
            log.warning("metrics failed (attempt %d): %s", attempt + 1, e)
            time.sleep(1.5)
    try:
        q = _df(obb().yfinance.equity.price.quote(symbol=",".join(symbols)))
        for _, row in q.iterrows():
            sym = row.get("symbol")
            if sym in out:
                for k in ("name", "currency", "exchange", "last_price", "prev_close", "year_high", "year_low"):
                    if k in row and row[k] is not None:
                        out[sym].setdefault(k, _clean(row[k]))
    except Exception as e:  # noqa: BLE001
        log.warning("quote failed: %s", e)
    return out


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return str(v)[:10]
    return v


# --------------------------------------------------------------------------- #
# News and calendars
# --------------------------------------------------------------------------- #


# Friendly search terms for symbols whose ticker feed is usually empty (FX, indices, futures).
NEWS_QUERY = {"USDJPY=X": "yen dollar", "EURJPY=X": "euro yen", "GBPJPY=X": "pound yen", "^N225": "Nikkei",
              "^TNX": "Treasury yields", "^FVX": "Treasury yields", "^GSPC": "S&P 500", "GC=F": "gold price",
              "CL=F": "oil prices", "1306.T": "TOPIX"}


def _norm_news(item: dict, sym: str) -> dict | None:
    """Flatten one yfinance news item (same shape OpenBB's company-news fetcher reads)."""
    c = item.get("content") if isinstance(item, dict) else None
    if not isinstance(c, dict):
        return None
    url = None
    for k in ("clickThroughUrl", "canonicalUrl"):
        v = c.get(k)
        if isinstance(v, dict) and v.get("url"):
            url = v["url"]
            break
    prov = c.get("provider")
    title = c.get("title")
    if not title:
        return None
    return {"title": title, "date": c.get("pubDate") or c.get("displayTime"), "url": url or c.get("previewUrl"),
            "source": prov.get("displayName") if isinstance(prov, dict) else None, "symbol": sym}


@cached(ttl=900)
def news(symbol: str, limit: int = 6, query: str | None = None) -> list[dict]:
    """Ticker news feed (yfinance, as OpenBB's company-news fetcher does); falls back to a keyword
    search on `query`, a known phrase for FX/indices, or the company name."""
    import yfinance as yf  # noqa: PLC0415

    items: list[dict] = []
    try:
        items = [n for n in (_norm_news(x, symbol) for x in (yf.Ticker(symbol).get_news(count=limit) or [])) if n]
    except Exception as e:  # noqa: BLE001
        log.warning("news feed %s failed: %s", symbol, e)
    if not items:
        q = query or NEWS_QUERY.get(symbol)
        if q:
            try:
                df = _df(obb().yfinance.news(query=q, limit=limit, fetch_body=False))
                for _, r in df.head(limit).iterrows():
                    items.append({"title": r.get("title"), "date": _clean(r.get("date")), "source": r.get("author"),
                                  "url": r.get("url"), "symbol": symbol})
            except Exception as e:  # noqa: BLE001
                log.warning("news search %s failed: %s", q, e)
    return items[:limit]


CAL_ALIASES = {
    "US": {"US", "USA", "UNITED STATES"},
    "JP": {"JP", "JPN", "JAPAN"},
    "EU": {"EU", "EZ", "EMU", "EA", "EUR", "EURO AREA", "EUROZONE", "EURO ZONE", "EUROPEAN UNION"},
    "DE": {"DE", "DEU", "GERMANY"},
    "GB": {"GB", "UK", "GBR", "UNITED KINGDOM"},
    "CN": {"CN", "CHN", "CHINA"},
}


def country_key(raw) -> str | None:
    s = str(raw or "").strip().upper()
    for k, al in CAL_ALIASES.items():
        if s in al:
            return k
    return None


@cached(ttl=3600)
def econ_calendar(days: int = 14) -> list[dict]:
    """Economic events for the next `days` (Yahoo returns 12 rows a call unless asked for more)."""
    import yfinance as yf  # noqa: PLC0415

    # Yahoo returns 12 rows unless asked and caps a page at 100; page through the whole range.
    start = date.today()
    end = start + timedelta(days=days)
    cal = yf.Calendars(start=start.isoformat(), end=end.isoformat())
    frames = []
    for off in range(0, 600, 100):
        try:
            df = cal.get_economic_events_calendar(start=start.isoformat(), end=end.isoformat(), limit=100, offset=off, force=True)
        except Exception as e:  # noqa: BLE001
            log.warning("calendar page %d failed: %s", off, e)
            break
        if df is None or df.empty:
            break
        frames.append(df.reset_index())
        if len(df) < 100:
            break
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": _clean(r.get("Event Time")),
            "country": r.get("Region"),
            "event": r.get("Event"),
            "reference_period": r.get("For"),
            "actual": _clean(r.get("Actual")),
            "consensus": _clean(r.get("Expected")),
            "previous": _clean(r.get("Last")),
        })
    rows.sort(key=lambda x: str(x["date"] or ""))
    return rows


@cached(ttl=3600)
def company_events(symbols: tuple[str, ...]) -> list[dict]:
    df = _df(obb().yfinance.company_calendar(symbol=",".join(symbols)))
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "symbol": r.get("symbol"),
                "earnings_date": _clean(r.get("earnings_date")),
                "ex_dividend_date": _clean(r.get("ex_dividend_date")),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #

UST_COLS = {
    "month_1": 1 / 12,
    "month_3": 0.25,
    "month_6": 0.5,
    "year_1": 1,
    "year_2": 2,
    "year_3": 3,
    "year_5": 5,
    "year_7": 7,
    "year_10": 10,
    "year_20": 20,
    "year_30": 30,
}


@cached(ttl=3 * 3600)
def ust_history(years: float = 3) -> pd.DataFrame:
    """Daily Treasury par yields in %, columns as years-to-maturity floats."""
    start = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
    df = _df(obb().federal_reserve.treasury_rates(start_date=start))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    out = pd.DataFrame({yrs: to_percent(df[c]) for c, yrs in UST_COLS.items() if c in df})
    return out.dropna(how="all")


def _maturity_years(m) -> float | None:
    if m is None:
        return None
    s = str(m)
    if "_" in s:
        parts = s.split("_")
        try:
            months = sum(int(parts[i + 1]) * (12 if parts[i] == "year" else 1) for i in range(0, len(parts), 2))
            return months / 12
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


@cached(ttl=3 * 3600)
def ecb_curve(on: str | None = None) -> dict[float, float]:
    """Euro area AAA par-yield curve (proxy for Bunds) in %."""
    kw = {"rating": "aaa", "yield_curve_type": "par_yield"}
    if on:
        kw["date"] = on
    df = _df(obb().ecb.yield_curve(**kw))
    if df.empty:
        return {}
    if "date" in df:
        df = df[df["date"] == df["date"].max()]
    yrs = df["maturity"].map(_maturity_years) if "maturity" in df else None
    if yrs is None and "maturity_years" in df:
        yrs = df["maturity_years"]
    rates = to_percent(df["rate"])
    return {float(y): float(r) for y, r in zip(yrs, rates) if y is not None and pd.notna(r)}


MOF_CURRENT = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
MOF_ALL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"


def parse_mof_csv(text: str) -> pd.DataFrame:
    """Parse the MOF JGB benchmark CSV into a date-indexed frame, columns = years.

    The file starts with a title line, then a header like
    'Date,1Y,2Y,...,40Y'. Missing points are '-'. Dates are Gregorian
    in the English file (2026/10/1); older files may use era dates (R8.10.1).
    """
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.lower().startswith("date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), na_values=["-", ""])
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = df["date"].map(_parse_jp_date)
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    cols = {}
    for c in df.columns:
        cs = str(c).strip().upper()
        if cs.endswith("Y"):
            try:
                cols[c] = float(cs[:-1])
            except ValueError:
                pass
    df = df[list(cols)].rename(columns=cols).apply(pd.to_numeric, errors="coerce")
    return df


_ERAS = {"R": 2018, "H": 1988, "S": 1925}


def _parse_jp_date(s) -> pd.Timestamp | None:
    s = str(s).strip()
    if not s:
        return None
    if s[0] in _ERAS and "." in s:
        try:
            y, m, d = s[1:].split(".")
            return pd.Timestamp(_ERAS[s[0]] + int(y), int(m), int(d))
        except ValueError:
            return None
    try:
        return pd.Timestamp(s.replace(".", "/"))
    except (ValueError, TypeError):
        return None


@cached(ttl=6 * 3600)
def jgb_history() -> pd.DataFrame:
    frames = []
    with httpx.Client(timeout=30, headers={"User-Agent": "keel/1.0"}) as c:
        for url in (MOF_ALL, MOF_CURRENT):
            try:
                r = c.get(url)
                r.raise_for_status()
                frames.append(parse_mof_csv(r.content.decode("utf-8", errors="replace")))
            except Exception as e:  # noqa: BLE001
                log.warning("MOF %s failed: %s", url, e)
    if not frames:
        raise ValueError("JGB data unavailable")
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------- #
# Options (implied vol)
# --------------------------------------------------------------------------- #


@cached(ttl=1800)
def implied_vol(symbol: str, target_days: int = 365, rate: float = 0.04, div_yield: float = 0.0) -> dict:
    """ATM implied vol and 90% put skew near target_days, computed from option prices.

    Yahoo's own impliedVolatility field is junk outside US trading hours (it is
    derived from zero bids), so vols are solved here with Black-Scholes from the
    mid-quote when there is a live market, else the last traded price. ATM is the
    median over 95-105% strikes; the next-closest expiries are tried if the nearest
    has too few prices. rate and div_yield are decimals. Coverage is good for US
    names and thin for most Japanese single stocks.
    """
    import yfinance as yf  # noqa: PLC0415

    from .analytics import implied_vol_from_price  # noqa: PLC0415

    t = yf.Ticker(symbol)
    exps = list(t.options or [])
    if not exps:
        return {}
    try:
        spot = float(t.fast_info["last_price"])
    except Exception:  # noqa: BLE001
        return {}
    today = date.today()
    exps = [e for e in exps if (date.fromisoformat(e) - today).days >= 20]
    exps.sort(key=lambda e: abs((date.fromisoformat(e) - today).days - target_days))

    def vols(df: pd.DataFrame, kind: str, tyears: float, lo: float, hi: float) -> list[float]:
        out = []
        for _, r in df[(df["strike"] >= spot * lo) & (df["strike"] <= spot * hi)].iterrows():
            bid, ask, last = r.get("bid") or 0, r.get("ask") or 0, r.get("lastPrice") or 0
            px = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            v = implied_vol_from_price(kind, float(px), spot, float(r["strike"]), tyears, rate, div_yield)
            if v is not None and 0.05 < v < 3:
                out.append(v * 100)
        return out

    for exp in exps[:4]:
        try:
            ch = t.option_chain(exp)
        except Exception:  # noqa: BLE001
            continue
        tyears = (date.fromisoformat(exp) - today).days / 365.0
        near = vols(ch.calls, "call", tyears, 0.95, 1.05) + vols(ch.puts, "put", tyears, 0.95, 1.05)
        if len(near) < 2:
            continue
        atm = float(np.median(near))
        p90 = vols(ch.puts, "put", tyears, 0.87, 0.93)
        put90 = float(np.median(p90)) if p90 else None
        return {
            "expiry": exp,
            "days": (date.fromisoformat(exp) - today).days,
            "atm_iv": atm,
            "put90_iv": put90,
            "skew_90": (put90 - atm) if put90 is not None else None,
        }
    return {}


# --------------------------------------------------------------------------- #
# Macro
# --------------------------------------------------------------------------- #


OECD_CPI_URL = ("https://sdmx.oecd.org/public/rest/v2/data/dataflow/OECD.SDD.TPS/DSD_PRICES%40DF_PRICES_ALL/1.0/"
                "{areas}.*.N.CPI.IX._T.N._Z?dimensionAtObservation=TIME_PERIOD&detail=full&c[TIME_PERIOD]=ge:{start}")


def parse_oecd_cpi_csv(text: str) -> pd.DataFrame:
    """SDMX-CSV from the OECD prices dataflow -> monthly CPI index, one column per ISO3 area.

    Monthly series are preferred; areas that only publish quarterly (Australia,
    New Zealand) are spread to month-ends by carrying each quarter forward.
    """
    df = pd.read_csv(io.StringIO(text))
    df = df[["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE"]].dropna(subset=["OBS_VALUE"])
    out = {}
    for area, g in df.groupby("REF_AREA"):
        m = g[g["FREQ"] == "M"]
        if not m.empty:
            idx = pd.PeriodIndex(m["TIME_PERIOD"], freq="M").to_timestamp(how="end").normalize()
            out[area] = pd.Series(m["OBS_VALUE"].astype(float).values, index=idx).sort_index()
            continue
        q = g[g["FREQ"] == "Q"]
        if not q.empty:
            idx = pd.PeriodIndex(q["TIME_PERIOD"].str.replace("-Q", "Q"), freq="Q").to_timestamp(how="end").normalize()
            ser = pd.Series(q["OBS_VALUE"].astype(float).values, index=idx).sort_index()
            out[area] = ser.resample("ME").ffill()
    return pd.DataFrame(out).sort_index()


@cached(ttl=12 * 3600)
def cpi_index(areas: tuple[str, ...], start: str = "2005-01") -> pd.DataFrame:
    """Monthly CPI index levels from the OECD (ISO3 area codes; euro area = EA20).

    Called directly rather than through OpenBB's OECD extension, whose metadata
    loader needs more memory than Render's free tier has.
    """
    url = OECD_CPI_URL.format(areas="+".join(areas), start=start)
    headers = {"Accept": "application/vnd.sdmx.data+csv; version=2.0.0", "User-Agent": "keel/1.0"}
    with httpx.Client(timeout=60, headers=headers) as c:
        r = c.get(url)
        r.raise_for_status()
    return parse_oecd_cpi_csv(r.text)
