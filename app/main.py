"""Keel: a small market-data API on top of OpenBB's Open Data Platform.

Serves the Carry, Brief, Barrier and Arcade artifacts, the live-data hooks in
Frontier / Parity / Cadence, and (through `openbb-mcp --app app/main.py`) an
MCP server so Claude can query the same numbers.

Only public market data passes through here. No client data is sent or stored.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import analytics as A
from . import data as D

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("keel")

TOKEN = os.environ.get("KEEL_TOKEN", "").strip()


def check_token(request: Request):
    """Optional shared secret. Set KEEL_TOKEN on Render to switch it on."""
    if not TOKEN:
        return
    got = request.headers.get("x-keel-token") or request.query_params.get("token")
    if got != TOKEN:
        raise HTTPException(status_code=401, detail="Missing or wrong token")


app = FastAPI(
    title="Keel",
    version="1.0.0",
    description="Market data for the UGM tools, built on the OpenBB Open Data Platform.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _syms(s: str, cap: int = 25) -> list[str]:
    out = [x.strip() for x in (s or "").split(",") if x.strip()]
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    if not seen:
        raise HTTPException(400, "Give at least one symbol")
    return seen[:cap]


def _short_name(n):
    """'Nintendo Co., Ltd.' -> 'Nintendo' for news search."""
    if not n:
        return None
    import re  # noqa: PLC0415

    n = re.sub(r"[,.]?\s*(Co\.?|Ltd\.?|Inc\.?|Corporation|Corp\.?|Holdings?|Group|plc|N\.V\.|S\.A\.)\b.*$", "", n, flags=re.I)
    return n.strip() or None


def _sig(v, n: int = 6):
    """Round to n significant figures (prices from 0.5 to 50,000 keep their meaningful digits)."""
    v = float(v)
    if not math.isfinite(v) or v == 0:
        return v if math.isfinite(v) else None
    return round(v, max(0, n - 1 - int(math.floor(math.log10(abs(v))))))


def _safe(v):
    """JSON-safe floats (NaN/inf -> None), rounded."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_safe(x) for x in v]
    return v


# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "service": "keel", "token_required": bool(TOKEN), "date": date.today().isoformat()}


@app.get("/api/yields", operation_id="hedged_yields", dependencies=[Depends(check_token)])
def yields(basis_usd_bp: float = 0.0, basis_eur_bp: float = 0.0):
    """Government curves (UST, EUR AAA, JGB) and yen-hedged yields for a Japanese investor.

    Hedge cost = 1Y foreign rate - 1Y JGB yield (+ optional cross-currency basis in bp).
    All rates in percent.
    """
    errors: list[str] = []
    tenors = [1, 2, 3, 5, 7, 10, 20, 30]

    us_curve: dict[float, float] = {}
    us_hist = pd.DataFrame()
    try:
        us_hist = D.ust_history(3)
        last = us_hist.dropna(how="all").iloc[-1]
        us_curve = {float(k): float(v) for k, v in last.items() if pd.notna(v)}
        us_asof = str(us_hist.dropna(how="all").index[-1].date())
    except Exception as e:  # noqa: BLE001
        errors.append(f"UST: {e}")
        us_asof = None

    jp_curve: dict[float, float] = {}
    jp_hist = pd.DataFrame()
    try:
        jp_hist = D.jgb_history()
        last = jp_hist.dropna(how="all").iloc[-1]
        jp_curve = {float(k): float(v) for k, v in last.items() if pd.notna(v)}
        jp_asof = str(jp_hist.dropna(how="all").index[-1].date())
    except Exception as e:  # noqa: BLE001
        errors.append(f"JGB: {e}")
        jp_asof = None

    eu_curve: dict[float, float] = {}
    try:
        eu_curve = D.ecb_curve()
    except Exception as e:  # noqa: BLE001
        errors.append(f"EUR: {e}")

    fx = {}
    for pair in ("USDJPY=X", "EURJPY=X"):
        try:
            s = D.closes(pair, 1)
            fx[pair[:6]] = {"spot": float(s.iloc[-1]), "vol_1y": A.realized_vol(s, 252)}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{pair}: {e}")

    r_us = A.interp_curve(us_curve, 1)
    r_eu = A.interp_curve(eu_curve, 1)
    r_jp = A.interp_curve(jp_curve, 1)

    rows = []
    for t in tenors:
        us, eu, jp = A.interp_curve(us_curve, t), A.interp_curve(eu_curve, t), A.interp_curve(jp_curve, t)
        rows.append(
            {
                "tenor": t,
                "us": us,
                "eu": eu,
                "jp": jp,
                "us_hedged": A.hedged_yield(us, r_us, r_jp, basis_usd_bp) if None not in (us, r_us, r_jp) else None,
                "eu_hedged": A.hedged_yield(eu, r_eu, r_jp, basis_eur_bp) if None not in (eu, r_eu, r_jp) else None,
            }
        )

    history = []
    if not us_hist.empty and not jp_hist.empty:
        m_us = us_hist.resample("ME").last()
        m_jp = jp_hist.resample("ME").last()
        joined = pd.DataFrame(
            {
                "us10": m_us.get(10.0),
                "us1": m_us.get(1.0),
                "jp10": m_jp.get(10.0),
                "jp1": m_jp.get(1.0),
            }
        ).dropna()
        for d, r in joined.iterrows():
            hc = A.hedge_cost(r.us1, r.jp1, basis_usd_bp)
            history.append(
                {
                    "date": str(d.date()),
                    "us10": r.us10,
                    "jp10": r.jp10,
                    "usd_hedge_cost": hc,
                    "us10_hedged": r.us10 - hc,
                }
            )

    return _safe(
        {
            "asof": {"us": us_asof, "jp": jp_asof},
            "curves": {
                "us": [{"t": k, "y": v} for k, v in sorted(us_curve.items())],
                "eu": [{"t": k, "y": v} for k, v in sorted(eu_curve.items())],
                "jp": [{"t": k, "y": v} for k, v in sorted(jp_curve.items())],
            },
            "short": {"usd_1y": r_us, "eur_1y": r_eu, "jpy_1y": r_jp},
            "hedge_cost": {
                "usd": A.hedge_cost(r_us, r_jp, basis_usd_bp) if None not in (r_us, r_jp) else None,
                "eur": A.hedge_cost(r_eu, r_jp, basis_eur_bp) if None not in (r_eu, r_jp) else None,
            },
            "fx": fx,
            "table": rows,
            "history": history,
            "errors": errors,
            "method": "Hedge cost = 1Y foreign government yield minus 1Y JGB yield plus basis (covered interest parity). "
            "Sources: Federal Reserve H.15 and ECB AAA curve via OpenBB; JGBs from Japan MOF.",
        }
    )


@app.get("/api/snapshot", operation_id="market_snapshot", dependencies=[Depends(check_token)])
def snapshot(symbols: str = Query(..., description="Comma-separated Yahoo symbols, e.g. USDJPY=X,^N225,7974.T")):
    """Price, returns (1D to 1Y, YTD), realised vol and 52-week range per symbol."""
    syms = _syms(symbols)
    prices = D.many_closes(syms, 2)
    meta = {}
    try:
        meta = D.metrics(tuple(s for s in syms if "=" not in s and not s.startswith("^")))
    except Exception:  # noqa: BLE001
        pass
    items = []
    for s in syms:
        c = prices.get(s)
        if c is None:
            items.append({"symbol": s, "error": "no data"})
            continue
        m = meta.get(s, {})
        items.append(
            {
                "symbol": s,
                "name": m.get("name"),
                "currency": m.get("currency"),
                **A.snapshot_stats(c),
                "spark": A.sparkline(c, 130),
            }
        )
    return _safe({"items": items})


@app.get("/api/brief", operation_id="meeting_brief", dependencies=[Depends(check_token)])
def brief(
    symbols: str = Query(..., description="Comma-separated Yahoo symbols"),
    days: int = Query(14, ge=1, le=45, description="Calendar look-ahead in days"),
    countries: str = Query("US,JP,EU,EZ,DE,GB,CN", description="Country codes kept from the economic calendar"),
):
    """Everything for a pre-meeting one-pager: snapshot, news, company events, economic calendar."""
    syms = _syms(symbols, 12)
    snap = snapshot(",".join(syms))
    errors = []

    news_items: list[dict] = []
    names = {it["symbol"]: _short_name(it.get("name")) for it in snap["items"]}
    got = D._pool(lambda s: D.news(s, 5, names.get(s)), syms)
    if not any(got.values()):
        errors.append("news: no headlines returned")
    for s in syms:
        news_items.extend(got.get(s) or [])
    news_items.sort(key=lambda n: n.get("date") or "", reverse=True)
    seen, dedup = set(), []
    for n in news_items:
        if n["title"] and n["title"] not in seen:
            seen.add(n["title"])
            dedup.append(n)

    events = []
    eq = tuple(s for s in syms if "=" not in s and not s.startswith("^"))
    if eq:
        try:
            events = D.company_events(eq)
        except Exception as e:  # noqa: BLE001
            errors.append(f"company calendar: {e}")

    cal = []
    try:
        keep = {D.country_key(c) or c.strip().upper() for c in countries.split(",") if c.strip()}
        allev = D.econ_calendar(days)
        for ev in allev:
            k = D.country_key(ev.get("country"))
            if not keep or (k and k in keep):
                cal.append({**ev, "country": k or ev.get("country")})
        if not allev:
            errors.append("economic calendar: no events returned")
        if allev and not cal:
            errors.append(f"economic calendar: {len(allev)} events, none for {sorted(keep)}")
    except Exception as e:  # noqa: BLE001
        errors.append(f"economic calendar: {e}")

    return _safe(
        {
            "generated": date.today().isoformat(),
            "items": snap["items"],
            "news": dedup[:14],
            "events": events,
            "calendar": cal[:60],
            "errors": errors,
        }
    )


# Government yields served by /api/history next to Yahoo symbols: name -> (source, tenor in years).
# Values are yields in percent, so callers should take differences (bp), not returns.
YIELD_SYMS = {
    "UST2Y": ("ust", 2.0), "UST5Y": ("ust", 5.0), "UST10Y": ("ust", 10.0), "UST30Y": ("ust", 30.0),
    "JGB2Y": ("jgb", 2.0), "JGB5Y": ("jgb", 5.0), "JGB10Y": ("jgb", 10.0), "JGB30Y": ("jgb", 30.0),
}


def _yield_series(sym: str, years: float) -> pd.Series | None:
    src, tenor = YIELD_SYMS[sym]
    df = D.ust_history(years) if src == "ust" else D.jgb_history()
    if tenor not in df:
        return None
    s = pd.to_numeric(df[tenor], errors="coerce").dropna()
    return s if not s.empty else None


@app.get("/api/history", operation_id="price_history", dependencies=[Depends(check_token)])
def history(
    symbols: str = Query(..., description="Comma-separated Yahoo symbols, or government yields "
                         "UST2Y, UST5Y, UST10Y, UST30Y, JGB2Y, JGB5Y, JGB10Y, JGB30Y (in %)"),
    start: str | None = Query(None, description="YYYY-MM-DD; default 2 years ago"),
    interval: str = Query("M", pattern="^(D|W|M)$", description="D, W or M (period-end closes)"),
    total: bool = Query(False, description="Adjust for dividends and distributions (total return); "
                        "ignored for FX, indices and government yields"),
    compact: bool = Query(False, description="Columns instead of rows per symbol ({d: [...], v: [...]}), "
                          "about half the size; for long histories"),
):
    """Closing prices per symbol, resampled to day, week or month end.
    With `total=true`, funds and stocks are adjusted for distributions (a total-return index).

    UST*/JGB* names return daily government yields in percent (Fed H.15 via OpenBB; Japan MOF);
    `units` marks them 'pct' so callers use changes in bp rather than returns."""
    syms = _syms(symbols, 12)
    st = pd.Timestamp(start) if start else pd.Timestamp(date.today() - timedelta(days=730))
    years = max(1.0, (pd.Timestamp(date.today()) - st).days / 365.25 + 0.1)
    ysyms = [s for s in syms if s.upper() in YIELD_SYMS]
    psyms = [s for s in syms if s not in ysyms]
    got = D.many_closes(psyms, D.year_bucket(years), dividends=total) if psyms else {}
    for s in ysyms:
        try:
            ys = _yield_series(s.upper(), years)
        except Exception as e:  # noqa: BLE001
            log.warning("yield %s failed: %s", s, e)
            ys = None
        if ys is not None:
            got[s] = ys
    out = {}
    for s, c in got.items():
        c = c[c.index >= st]
        if interval == "W":
            c = c.resample("W-FRI").last()
        elif interval == "M":
            c = c.resample("ME").last()
        c = c.dropna()
        if compact:
            out[s] = {"d": [str(i.date()) for i in c.index], "v": [_sig(v) for v in c.values]}
        else:
            out[s] = [{"d": str(i.date()), "v": float(v)} for i, v in c.items()]
    units = {s: ("pct" if s in ysyms else "price") for s in out}
    return _safe({"series": out, "units": units, "missing": [s for s in syms if s not in out],
                  "total_return": bool(total), **({"compact": True} if compact else {})})


@app.get("/api/ohlc", operation_id="price_bars", dependencies=[Depends(check_token)])
def ohlc_bars(
    symbols: str = Query(..., description="Comma-separated Yahoo symbols, or UST2Y..UST30Y / JGB2Y..JGB30Y "
                         "(yields: close only, so open = high = low = close)"),
    start: str | None = Query(None, description="YYYY-MM-DD; default 2 years ago"),
    interval: str = Query("D", pattern="^(D|W|M)$", description="D, W (Friday) or M (month-end) bars"),
    compact: bool = Query(False, description="Columns instead of rows per symbol ({d, o, h, l, c, v} arrays), "
                          "about half the size; for long histories"),
):
    """Open, high, low, close and volume bars per symbol, for bar-pattern studies (e.g. DeMark).
    Prices are split-adjusted but not dividend-adjusted. `close_only` lists symbols whose
    high/low are not available (government yields)."""
    syms = _syms(symbols, 12)
    st = pd.Timestamp(start) if start else pd.Timestamp(date.today() - timedelta(days=730))
    years = max(1.0, (pd.Timestamp(date.today()) - st).days / 365.25 + 0.1)
    ysyms = [s for s in syms if s.upper() in YIELD_SYMS]
    psyms = [s for s in syms if s not in ysyms]
    yb = D.year_bucket(years)
    got = D._pool(lambda s: D.ohlc(s, yb), psyms, workers=3) if psyms else {}
    got = {k: v for k, v in got.items() if v is not None and len(v)}
    for s in ysyms:
        try:
            ys = _yield_series(s.upper(), years)
        except Exception as e:  # noqa: BLE001
            log.warning("yield %s failed: %s", s, e)
            ys = None
        if ys is not None:
            got[s] = pd.DataFrame({"open": ys, "high": ys, "low": ys, "close": ys, "volume": np.nan})
    out = {}
    for s, df in got.items():
        df = D.resample_ohlc(df[df.index >= st], interval)
        if compact:
            out[s] = {"d": [str(i.date()) for i in df.index], **{k[0]: [_sig(x) for x in df[k].values] for k in ("open", "high", "low", "close")},
                      "v": [float(x) if pd.notna(x) and x > 0 else None for x in df["volume"].values]}
            continue
        out[s] = [{"d": str(i.date()), "o": float(r.open), "h": float(r.high), "l": float(r.low), "c": float(r.close),
                   **({"v": float(r.volume)} if pd.notna(r.volume) and r.volume > 0 else {})}
                  for i, r in df.iterrows()]
    return _safe({"series": out, "close_only": [s for s in ysyms if s in out],
                  "missing": [s for s in syms if s not in out], "interval": interval, **({"compact": True} if compact else {})})


@app.get("/api/sp/screen", operation_id="structured_product_screen", dependencies=[Depends(check_token)])
def sp_screen(
    symbols: str = Query(..., description="Comma-separated underlyings"),
    tenor_m: int = Query(12, ge=3, le=36, description="Note tenor in months"),
    ki: float = Query(60, ge=30, le=95, description="Knock-in barrier, % of initial"),
    autocall: float = Query(100, ge=80, le=120, description="Autocall level, % of initial"),
    obs_m: int = Query(3, ge=1, le=12, description="Autocall observation every N months"),
    years: int = Query(10, ge=3, le=20, description="History used for the back-test"),
    iv: bool = Query(True, description="Fetch option implied vols (slower)"),
):
    """Rank underlyings for autocalls / FCNs: vol, implied vol and skew, dividend yield,
    correlation, and a historical back-test of autocall, knock-in and loss rates.
    Also back-tests the worst-of basket of all symbols given."""
    syms = _syms(symbols, 8)
    td, ob = int(tenor_m * 21), int(obs_m * 21)
    prices = D.many_closes(syms, years)
    meta = {}
    try:
        meta = D.metrics(tuple(s for s in syms if not s.startswith("^")))
    except Exception:  # noqa: BLE001
        pass
    usd_rate = 0.04
    try:
        h = D.ust_history(1).dropna(how="all")
        usd_rate = float(h[1.0].dropna().iloc[-1]) / 100
    except Exception:  # noqa: BLE001
        pass
    jpy_rate = None
    try:
        jp = D.jgb_history().dropna(how="all").iloc[-1]
        jpy_rate = A.interp_curve({float(k): float(v) for k, v in jp.items() if pd.notna(v)}, tenor_m / 12) / 100
    except Exception:  # noqa: BLE001
        pass

    def _iv(s):
        dy = (meta.get(s, {}).get("dividend_yield") or 0) / 100  # Yahoo gives percent
        return D.implied_vol(s, tenor_m * 30, usd_rate, dy)

    ivs = D._pool(_iv, syms) if iv else {}

    rows = []
    for s in syms:
        c = prices.get(s)
        if c is None:
            rows.append({"symbol": s, "error": "no data"})
            continue
        m = meta.get(s, {})
        v = ivs.get(s) or {}
        bt = A.barrier_backtest({s: c}, td, ki / 100, autocall / 100, ob)
        rows.append(
            {
                "symbol": s,
                "name": m.get("name"),
                "last": float(c.iloc[-1]),
                "vol_60d": A.realized_vol(c, 60),
                "vol_1y": A.realized_vol(c, 252),
                "atm_iv": v.get("atm_iv"),
                "skew_90": v.get("skew_90"),
                "iv_expiry": v.get("expiry"),
                "div_yield": m.get("dividend_yield"),  # percent, as Yahoo reports it
                "max_dd_1y": A.max_drawdown(c.iloc[-252:]),
                "worst_tenor_drop": A.min_distance_path(c, td),
                "backtest": bt,
                "windows": A.tenor_windows(c, td),  # [final, lowest] per rolling window
            }
        )
    basket = A.barrier_backtest(prices, td, ki / 100, autocall / 100, ob) if len(prices) >= 2 else None
    return _safe(
        {
            "params": {"tenor_m": tenor_m, "ki": ki, "autocall": autocall, "obs_m": obs_m, "years": years},
            "rates": {"usd": usd_rate * 100, "jpy": jpy_rate * 100 if jpy_rate is not None else None},
            "rows": rows,
            "basket": basket,
            "corr": A.correlation_matrix(prices),
            "note": "Back-test rolls the note daily through history using closing prices; it is a description of the past, not a probability.",
        }
    )


@app.get("/api/comps", operation_id="peer_comps", dependencies=[Depends(check_token)])
def comps(symbols: str = Query(..., description="Comma-separated tickers, e.g. 7974.T,6758.T,9697.T")):
    """Peer comparison: valuation, margins, growth, returns, vol and 1-year price path."""
    syms = _syms(symbols, 15)
    prices = D.many_closes(syms, 2)
    meta = D.metrics(tuple(syms))
    rows = []
    for s in syms:
        m = meta.get(s, {})
        c = prices.get(s)
        st = A.snapshot_stats(c) if c is not None else {}
        rows.append(
            {
                "symbol": s,
                "name": m.get("name"),
                "currency": m.get("currency"),
                "market_cap": m.get("market_cap"),
                "pe": m.get("pe_ratio"),
                "fwd_pe": m.get("forward_pe"),
                "ev_ebitda": m.get("enterprise_to_ebitda"),
                "op_margin": m.get("operating_margin"),
                "rev_growth": m.get("revenue_growth"),
                "roe": m.get("return_on_equity"),
                "div_yield": m.get("dividend_yield"),  # percent, as Yahoo reports it
                **st,
                "spark": A.sparkline(c, 260) if c is not None else [],
            }
        )
    jpy_1y = None
    try:
        jp = D.jgb_history().dropna(how="all").iloc[-1]
        jpy_1y = A.interp_curve({float(k): float(v) for k, v in jp.items() if pd.notna(v)}, 1)
    except Exception:  # noqa: BLE001
        pass
    return _safe({"rows": rows, "corr": A.correlation_matrix(prices), "jpy_1y": jpy_1y})


@app.get("/api/returns", operation_id="return_statistics", dependencies=[Depends(check_token)])
def returns(
    symbols: str = Query(..., description="Comma-separated Yahoo symbols"),
    years: int = Query(10, ge=3, le=25),
):
    """Annualised mean return, volatility, covariance and correlation from month-end prices."""
    syms = _syms(symbols, 15)
    prices = D.many_closes(syms, years)
    stats = A.monthly_return_stats(prices)
    stats["missing"] = [s for s in syms if s not in prices]
    return _safe(stats)


# Parity: Parity's ISO3 code -> (OECD area code, currency)
PPP_MAP = {
    "AUS": ("AUS", "AUD"), "BRA": ("BRA", "BRL"), "CAN": ("CAN", "CAD"), "CHE": ("CHE", "CHF"),
    "CHL": ("CHL", "CLP"), "CHN": ("CHN", "CNY"), "COL": ("COL", "COP"), "CRI": ("CRI", "CRC"),
    "CZE": ("CZE", "CZK"), "DNK": ("DNK", "DKK"), "EUZ": ("EA20", "EUR"),
    "GBR": ("GBR", "GBP"), "HUN": ("HUN", "HUF"), "IDN": ("IDN", "IDR"), "IND": ("IND", "INR"),
    "ISR": ("ISR", "ILS"), "JPN": ("JPN", "JPY"), "KOR": ("KOR", "KRW"), "MEX": ("MEX", "MXN"),
    "NOR": ("NOR", "NOK"), "NZL": ("NZL", "NZD"), "POL": ("POL", "PLN"), "SAU": ("SAU", "SAR"),
    "SWE": ("SWE", "SEK"), "TUR": ("TUR", "TRY"), "ZAF": ("ZAF", "ZAR"),
}


@app.get("/api/ppp", operation_id="relative_ppp", dependencies=[Depends(check_token)])
def ppp(base_year: int = Query(2005, ge=2000, le=2020)):
    """Relative purchasing-power parity against the dollar from OECD CPI (OECD SDMX API).

    Fair rate today = average market rate in the base year x (local CPI growth / US CPI growth).
    Gap < 0 means the currency trades cheaper than inflation differentials imply.
    """
    areas = tuple(sorted({v[0] for v in PPP_MAP.values()} | {"USA"}))
    try:
        cpi = D.cpi_index(areas, f"{base_year}-01")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"OECD CPI unavailable: {e}") from e
    if cpi.empty or "USA" not in cpi:
        raise HTTPException(503, "US CPI unavailable from the OECD right now; try again in a minute")
    fx = {}
    try:
        fx = D.monthly_closes(tuple(f"{c}=X" for _, c in PPP_MAP.values()), date.today().year - base_year + 1)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"FX history unavailable: {e}") from e
    base_mask = cpi.index.year == base_year
    us = cpi["USA"].dropna()
    us_base = cpi.loc[base_mask, "USA"].mean()
    stale_cut = pd.Timestamp(date.today()) - pd.DateOffset(months=24)
    rows, skipped = [], []
    for iso, (area, ccy) in PPP_MAP.items():
        s = fx.get(f"{ccy}=X")
        if area not in cpi or s is None:
            skipped.append(iso)
            continue
        loc = cpi[area].dropna()
        b = loc[loc.index.year == base_year]
        fb = s[s.index.year == base_year]
        if b.empty or fb.empty or loc.empty:
            skipped.append(iso)
            continue
        asof = loc.index[-1]
        if asof < stale_cut:  # the OECD index for this country stopped being updated
            skipped.append(iso)
            continue
        # Compare both price levels at the same date, so a lagging series is not set against newer US data.
        us_then = us[us.index <= asof]
        if us_then.empty:
            skipped.append(iso)
            continue
        ratio = loc.iloc[-1] / b.mean()
        us_ratio = us_then.iloc[-1] / us_base
        fair = float(fb.mean()) * ratio / us_ratio
        spot = float(s.iloc[-1])
        rows.append(
            {
                "iso": iso,
                "ccy": ccy,
                "spot": spot,
                "fx_base": float(fb.mean()),
                "cpi_local": float(ratio),
                "cpi_us": float(us_ratio),
                "cpi_asof": str(asof.date()),
                "fair": fair,
                "gap": fair / spot - 1,
                "stale": bool(asof < pd.Timestamp(date.today()) - pd.DateOffset(months=9)),
            }
        )
    return _safe({"base_year": base_year, "rows": rows, "missing": skipped,
                  "note": "Prices compared at each country's latest CPI month; series more than 24 months old are left out."})



# --------------------------------------------------------------------------- #
# FX carry back-test data
# --------------------------------------------------------------------------- #

# Deliverable currencies only. (OECD area for the 3M interbank rate, BIS area for a policy-rate proxy)
FX_CARRY = {
    "USD": ("USA", "US"), "EUR": ("EA20", "XM"), "JPY": ("JPN", "JP"), "GBP": ("GBR", "GB"),
    "CAD": ("CAN", "CA"), "AUD": ("AUS", "AU"), "CNH": ("CHN", "CN"), "MXN": ("MEX", "MX"),
    "ZAR": ("ZAF", "ZA"), "TRY": ("TUR", "TR"), "SAR": (None, "SA"), "PLN": ("POL", "PL"),
    "HUF": ("HUN", "HU"), "CZK": ("CZE", "CZ"), "SGD": (None, "SG"), "HKD": (None, "HK"),
}
FX_EURO_ALT = ("EA19", "EA")  # older OECD area codes for the euro area
# Benchmarks: key -> (Yahoo symbol, quote currency, total return via dividend adjustment)
FX_BENCH = {
    "GOLD": ("GC=F", "USD", False), "SILVER": ("SI=F", "USD", False), "OIL": ("CL=F", "USD", False),
    "SPX": ("^SP500TR", "USD", False), "NKY": ("^N225", "JPY", False), "SX5E": ("^STOXX50E", "EUR", False),
    "AGG": ("AGG", "USD", True),
}


@app.get("/api/fxcarry", operation_id="fx_carry_data", dependencies=[Depends(check_token)])
def fx_carry(years: int = Query(22, ge=5, le=25)):
    """Month-end spot rates (units per USD) and 3-month rates (% p.a.) for deliverable G20 and EM
    currencies, aligned on one month grid, for spot and spot-plus-carry back-tests.

    Rates: OECD 3-month interbank where it exists; otherwise (or after the OECD series stops)
    the BIS central-bank policy rate as a proxy. `rate_src` says which, per currency.
    """
    start = f"{date.today().year - years}-01"
    errors = []
    # Spot: Yahoo "XXX=X" quotes are units of XXX per USD. CNH history is short; CNY fills in before it.
    syms = tuple(sorted({f"{c}=X" for c in FX_CARRY if c not in ("USD", "CNH")} | {"CNH=X", "CNY=X"}))
    try:
        fx = D.monthly_closes(syms, years + 1)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"FX history unavailable: {e}") from e
    spot = {}
    for c in FX_CARRY:
        if c == "USD":
            continue
        s = fx.get(f"{c}=X")
        if c == "CNH":
            cny = fx.get("CNY=X")
            if s is None or s.empty:
                s = cny
            elif cny is not None and not cny.empty:
                s = pd.concat([cny[cny.index < s.index[0]], s]).sort_index()
        if s is None or s.empty:
            errors.append(f"{c} spot")
            continue
        s = D.despike(s.resample("ME").last())
        spot[c] = s
    if not spot:
        raise HTTPException(503, "Yahoo returned no FX history; try again in a minute")

    # Rates
    oecd_areas = tuple(sorted({a for a, _ in FX_CARRY.values() if a} | {FX_EURO_ALT[0]}))
    bis_areas = tuple(sorted({b for _, b in FX_CARRY.values() if b} | {FX_EURO_ALT[1]}))
    try:
        oecd = D.oecd_3m_rates(oecd_areas, start)
    except Exception as e:  # noqa: BLE001
        oecd = pd.DataFrame()
        errors.append(f"OECD rates: {e}")
    try:
        bis = D.bis_policy_rates(bis_areas, start)
    except Exception as e:  # noqa: BLE001
        bis = pd.DataFrame()
        errors.append(f"BIS rates: {e}")
    if oecd.empty and bis.empty:
        errors.append("no 3M rate source answered; carry is unavailable")

    def col(df, *names):
        for n in names:
            if n and n in df and df[n].notna().any():
                return df[n].dropna()
        return None

    stale = pd.Timestamp(date.today()) - pd.DateOffset(months=4)
    rate, src, rate_asof = {}, {}, {}
    for c, (oa, ba) in FX_CARRY.items():
        o = col(oecd, oa, FX_EURO_ALT[0] if c == "EUR" else None)
        b = col(bis, ba, FX_EURO_ALT[1] if c == "EUR" else None)
        if o is not None and b is not None and o.index[-1] < stale:
            r, how = pd.concat([o, b[b.index > o.index[-1]]]).sort_index(), "oecd+bis"
        elif o is not None:
            r, how = o, "oecd"
        elif b is not None:
            r, how = b, "bis"
        else:
            r, how = None, None
        if r is None:
            errors.append(f"{c} rate")
            continue
        rate_asof[c] = str(r.index[-1].date())
        rate[c], src[c] = r, how

    # One month grid: from the start to the latest spot month; rates carried forward at most 6 months.
    end = max(s.index[-1] for s in spot.values())
    grid = pd.date_range(pd.Timestamp(start + "-01") + pd.offsets.MonthEnd(0), end, freq="ME")

    def aligned(s, ffill):
        x = s.reindex(s.index.union(grid)).sort_index()
        x = x.ffill(limit=ffill) if ffill else x
        return [None if pd.isna(v) else float(v) for v in x.reindex(grid).values]

    out_spot = {"USD": [1.0] * len(grid)}
    out_spot.update({c: aligned(s, 1) for c, s in spot.items()})
    out_rate = {c: aligned(r, 6) for c, r in rate.items()}

    # Benchmarks (best effort: a failure here never blocks the FX data)
    bench = {}
    for div in (False, True):
        keys = [k for k, (_, _, d) in FX_BENCH.items() if d == div]
        try:
            got = D.monthly_closes(tuple(FX_BENCH[k][0] for k in keys), years + 1, dividends=True) if div else \
                D.monthly_closes(tuple(FX_BENCH[k][0] for k in keys), years + 1)
        except Exception as e:  # noqa: BLE001
            errors.append(f"benchmarks: {e}")
            continue
        for k in keys:
            sym, ccy, tr = FX_BENCH[k]
            s = got.get(sym)
            if s is None or s.empty:
                errors.append(f"{k} benchmark")
                continue
            s = D.despike(s[s > 0].resample("ME").last(), 0.5)
            bench[k] = {"sym": sym, "ccy": ccy, "tr": tr, "v": aligned(s, 1)}
    return _safe(
        {
            "asof": str(grid[-1].date()),
            "months": [str(d.date()) for d in grid],
            "ccys": [c for c in FX_CARRY if c in out_spot],
            "spot": out_spot,
            "rate": out_rate,
            "rate_src": src,
            "rate_asof": rate_asof,
            "bench": bench,
            "errors": errors,
            "note": "Spot: Yahoo month-end closes, units per USD (CNH before 2011 = onshore CNY). "
            "Rates: OECD 3-month interbank, % p.a.; 'bis' = BIS central-bank policy rate used as a proxy. "
            "Benchmarks: month-end prices in their own currency (gold, silver, WTI = front-month futures; "
            "S&P 500 total return; Nikkei 225 and Euro Stoxx 50 price indices; AGG with distributions reinvested).",
        }
    )


# --------------------------------------------------------------------------- #
# Rates carry: receive the 10-year, pay the 3-month, per currency
# --------------------------------------------------------------------------- #

# Currency -> (OECD area for the 3-month rate, OECD area for the 10-year yield, BIS area for a policy-rate proxy)
RATES_CARRY = {
    "USD": ("USA", "USA", "US"), "EUR": ("EA20", "DEU", "XM"), "JPY": ("JPN", "JPN", "JP"),
    "GBP": ("GBR", "GBR", "GB"), "CHF": ("CHE", "CHE", "CH"), "CAD": ("CAN", "CAN", "CA"),
    "AUD": ("AUS", "AUS", "AU"), "SEK": ("SWE", "SWE", "SE"), "NOK": ("NOR", "NOR", "NO"),
    "NZD": ("NZL", "NZL", "NZ"), "PLN": ("POL", "POL", "PL"), "HUF": ("HUN", "HUN", "HU"),
    "CZK": ("CZE", "CZE", "CZ"), "MXN": ("MEX", "MEX", "MX"), "ZAR": ("ZAF", "ZAF", "ZA"),
}
RATES_GROUPS = {"G5": ["USD", "EUR", "JPY", "GBP", "CHF"],
                "G10": ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "SEK", "NOK", "NZD"],
                "EM": ["PLN", "HUF", "CZK", "MXN", "ZAR"]}


def _trend_filter(y: pd.Series, trend: int = 160, avg: int = 5) -> pd.DataFrame:
    """Daily 10-year yield against its moving average: 'gap' = average of the last `avg` days
    minus the `trend`-day average, in bp. A positive gap means yields are trending up."""
    y = y.dropna()
    gap = (y.rolling(avg).mean() - y.rolling(trend).mean()) * 100
    return pd.DataFrame({"y": y, "ma": y.rolling(trend).mean(), "gap": gap}).dropna()


@app.get("/api/ratescarry", operation_id="rates_carry_data", dependencies=[Depends(check_token)])
def rates_carry(years: int = Query(25, ge=5, le=35)):
    """Month-end 3-month rates and 10-year government yields (% p.a.) for G10 and main EM currencies,
    for rates carry strategies (receive the 10-year, pay the 3-month, FX hedged). Also the US 10-year
    trend filter: the 5-day average yield minus its 160-day average, in bp, at each month end and today.

    Sources: OECD 3-month interbank (BIS policy rate as a proxy where missing or stale) and OECD
    long-term interest rates (10-year government bonds; EUR uses Germany). US filter: Fed H.15 daily."""
    start = f"{date.today().year - years}-01"
    errors: list[str] = []
    s_areas = tuple(sorted({a for a, _, _ in RATES_CARRY.values()} | {FX_EURO_ALT[0]}))
    l_areas = tuple(sorted({b for _, b, _ in RATES_CARRY.values()}))
    b_areas = tuple(sorted({c for _, _, c in RATES_CARRY.values()}))
    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
            return pd.DataFrame()
    st3, lt, bis = safe(D.oecd_3m_rates, s_areas, start), safe(D.oecd_lt_rates, l_areas, start), safe(D.bis_policy_rates, b_areas, start)
    if lt.empty:
        raise HTTPException(503, "OECD long-term rates did not answer; try again in a minute")
    stale = pd.Timestamp(date.today()) - pd.DateOffset(months=4)
    short, long_, src, asof = {}, {}, {}, {}
    for c, (sa, la, ba) in RATES_CARRY.items():
        o = st3[sa].dropna() if sa in st3 and st3[sa].notna().any() else (
            st3[FX_EURO_ALT[0]].dropna() if c == "EUR" and FX_EURO_ALT[0] in st3 else None)
        b = bis[ba].dropna() if ba in bis and bis[ba].notna().any() else None
        if o is not None and len(o) and b is not None and o.index[-1] < stale:
            r, how = pd.concat([o, b[b.index > o.index[-1]]]).sort_index(), "oecd+bis"
        elif o is not None and len(o):
            r, how = o, "oecd"
        elif b is not None:
            r, how = b, "bis"
        else:
            r, how = None, None
        y = lt[la].dropna() if la in lt and lt[la].notna().any() else None
        if r is None or y is None:
            errors.append(f"{c}: {'3M' if r is None else '10Y'} rate missing")
            continue
        short[c], long_[c], src[c] = r, y, how
        asof[c] = str(min(r.index[-1], y.index[-1]).date())
    if not long_:
        raise HTTPException(503, "No rates came back; try again in a minute")
    end = max(s.index[-1] for s in long_.values())
    grid = pd.date_range(pd.Timestamp(start + "-01") + pd.offsets.MonthEnd(0), end, freq="ME")

    def aligned(s, ffill):
        x = s.reindex(s.index.union(grid)).sort_index()
        x = x.ffill(limit=ffill) if ffill else x
        return [None if pd.isna(v) else round(float(v), 4) for v in x.reindex(grid).values]

    filt, today = None, None
    try:
        f = _trend_filter(_yield_series("UST10Y", years + 1))
        fm = f["gap"].resample("ME").last()
        filt = aligned(fm, 0)
        last = f.iloc[-1]
        today = {"date": str(f.index[-1].date()), "y10": round(float(last.y), 3), "ma160": round(float(last.ma), 3),
                 "gap_bp": round(float(last.gap), 1),
                 "recent": [{"d": str(i.date()), "y": round(float(r.y), 3), "ma": round(float(r.ma), 3)}
                            for i, r in f.iloc[-260:].iterrows()]}
    except Exception as e:  # noqa: BLE001
        errors.append(f"US 10Y trend filter: {e}")
    return _safe({
        "asof": str(grid[-1].date()), "months": [str(d.date()) for d in grid],
        "ccys": [c for c in RATES_CARRY if c in long_], "groups": RATES_GROUPS,
        "short": {c: aligned(s, 3) for c, s in short.items()}, "long": {c: aligned(s, 1) for c, s in long_.items()},
        "short_src": src, "asof_by_ccy": asof, "usd_filter_gap_bp": filt, "usd_filter_today": today, "errors": errors,
        "note": "3M: OECD 3-month interbank (bis = BIS policy rate proxy). 10Y: OECD long-term government bond yields, "
                "monthly averages (EUR = Germany). Filter: Fed H.15 daily 10-year; gap = 5-day average minus 160-day average, bp.",
    })


# --------------------------------------------------------------------------- #
# FX factors: carry, value, momentum, reversals, quality across G10 and EM
# --------------------------------------------------------------------------- #

# ccy -> (OECD area for 3M rate / CPI / CLI / confidence, OECD area for 10Y, BIS area, local equity index)
FXF = {
    "USD": ("USA", "USA", "US", "^GSPC"), "EUR": ("EA20", "DEU", "XM", "^STOXX50E"), "JPY": ("JPN", "JPN", "JP", "^N225"),
    "GBP": ("GBR", "GBR", "GB", "^FTSE"), "CHF": ("CHE", "CHE", "CH", "^SSMI"), "CAD": ("CAN", "CAN", "CA", "^GSPTSE"),
    "AUD": ("AUS", "AUS", "AU", "^AXJO"), "NZD": ("NZL", "NZL", "NZ", "^NZ50"), "SEK": ("SWE", "SWE", "SE", "^OMX"),
    "NOK": ("NOR", "NOR", "NO", "OBX.OL"), "BRL": ("BRA", None, "BR", "^BVSP"), "CNH": ("CHN", None, "CN", "000001.SS"),
    "CZK": ("CZE", "CZE", "CZ", None), "HUF": ("HUN", "HUN", "HU", "^BUX.BD"), "IDR": ("IDN", None, "ID", "^JKSE"),
    "ILS": ("ISR", "ISR", "IL", "^TA125.TA"), "INR": ("IND", None, "IN", "^BSESN"), "KRW": ("KOR", "KOR", "KR", "^KS11"),
    "MXN": ("MEX", "MEX", "MX", "^MXX"), "PLN": ("POL", "POL", "PL", "WIG20.WA"), "SGD": (None, None, "SG", "^STI"),
    "THB": (None, None, "TH", "^SET.BK"), "TRY": ("TUR", "TUR", "TR", "XU100.IS"), "ZAR": ("ZAF", "ZAF", "ZA", None),
}
FXF_G10 = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]
# One-way transaction cost per trade, bp of notional (G10 1bp; EM 1.5 to 9bp)
FXF_COST = {"EUR": 1, "JPY": 1, "GBP": 1, "CHF": 1, "CAD": 1, "AUD": 1, "NZD": 1, "SEK": 1, "NOK": 1, "BRL": 3.5,
            "CNH": 2, "CZK": 1.5, "HUF": 2, "IDR": 7, "ILS": 2, "INR": 3.5, "KRW": 2.5, "MXN": 1.5, "PLN": 1.5,
            "SGD": 2, "THB": 9, "TRY": 4, "ZAR": 2.5, "USD": 0}


@app.get("/api/fxfactors", operation_id="fx_factor_data", dependencies=[Depends(check_token)])
def fx_factors(years: int = Query(24, ge=8, le=26)):
    """Month-end panel for currency factor investing across G10 and EM (24 currencies against the dollar):
    spot (units per USD; the last row is the latest close), 3-month rates and 10-year yields (% p.a.),
    CPI index (for PPP value), local equity index, OECD leading indicator and consumer confidence.
    Plus one-way trading costs in bp. Front ends compute carry, value, momentum, reversal and quality signals.

    Sources: Yahoo month-end closes; OECD 3-month and long-term rates (BIS policy rate where missing or
    stale), OECD CPI, CLI and consumer confidence (published with a one to two month lag; `asof` per series)."""
    start = f"{date.today().year - years}-01"
    errors: list[str] = []

    def safe(name, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            return pd.DataFrame() if name != "yahoo" else {}

    fx_syms = tuple(sorted({f"{c}=X" for c in FXF if c not in ("USD", "CNH")} | {"CNH=X", "CNY=X"}))
    eq_syms = tuple(sorted({v[3] for v in FXF.values() if v[3]}))
    fx = safe("yahoo", D.monthly_closes, fx_syms, years + 1)
    if not fx:
        raise HTTPException(503, "Yahoo returned no FX history; try again in a minute")
    eq = safe("yahoo", D.monthly_closes, eq_syms, years + 1)
    areas = tuple(sorted({v[0] for v in FXF.values() if v[0]} | {FX_EURO_ALT[0]}))
    lt_areas = tuple(sorted({v[1] for v in FXF.values() if v[1]}))
    bis_areas = tuple(sorted({v[2] for v in FXF.values() if v[2]}))
    st3 = safe("OECD 3M", D.oecd_3m_rates, areas, start)
    lt = safe("OECD 10Y", D.oecd_lt_rates, lt_areas, start)
    bis = safe("BIS policy", D.bis_policy_rates, bis_areas, start)
    cpi = safe("OECD CPI", D.cpi_index, areas, start)
    cli = safe("OECD CLI", D.oecd_cli, areas, start)
    cci = safe("OECD consumer confidence", D.oecd_confidence, areas, "CCICP", start)

    spot = {}
    for c in FXF:
        if c == "USD":
            continue
        s = fx.get(f"{c}=X")
        if c == "CNH":
            cny = fx.get("CNY=X")
            s = cny if s is None or s.empty else (pd.concat([cny[cny.index < s.index[0]], s]).sort_index()
                                                   if cny is not None and not cny.empty else s)
        if s is None or s.empty:
            errors.append(f"{c} spot")
            continue
        spot[c] = D.despike(s.resample("ME").last())
    end = max(s.index[-1] for s in spot.values())
    grid = pd.date_range(pd.Timestamp(start + "-01") + pd.offsets.MonthEnd(0), end, freq="ME")
    stale = pd.Timestamp(date.today()) - pd.DateOffset(months=4)

    def col(df, a):
        if df is None or df.empty or a is None:
            return None
        if a in df and df[a].notna().any():
            return df[a].dropna()
        if a == "EA20" and FX_EURO_ALT[0] in df and df[FX_EURO_ALT[0]].notna().any():
            return df[FX_EURO_ALT[0]].dropna()
        return None

    def aligned(s, ffill):
        if s is None or s.empty:
            return None
        s = s.copy()
        s.index = s.index + pd.offsets.MonthEnd(0)
        s = s[~s.index.duplicated(keep="last")]
        x = s.reindex(s.index.union(grid)).sort_index()
        x = x.ffill(limit=ffill) if ffill else x
        v = x.reindex(grid)
        return None if v.isna().all() else [None if pd.isna(z) else round(float(z), 6) for z in v.values]

    out = {k: {} for k in ("spot", "rate", "y10", "cpi", "equity", "cli", "cci")}
    src, asof = {}, {}
    for c, (a, la, ba, ei) in FXF.items():
        if c != "USD":
            out["spot"][c] = aligned(spot.get(c), 0)
        o, b = col(st3, a), col(bis, ba)
        if o is not None and b is not None and o.index[-1] < stale:
            r, how = pd.concat([o, b[b.index > o.index[-1]]]).sort_index(), "oecd+bis"
        elif o is not None:
            r, how = o, "oecd"
        elif b is not None:
            r, how = b, "bis"
        else:
            r, how = None, None
        out["rate"][c], src[c] = aligned(r, 3), how
        out["y10"][c] = aligned(col(lt, la), 1)
        out["cpi"][c] = aligned(col(cpi, a), 3)
        out["equity"][c] = aligned(eq.get(ei).resample("ME").last() if ei and eq.get(ei) is not None else None, 0)
        out["cli"][c] = aligned(col(cli, a), 0)
        out["cci"][c] = aligned(col(cci, a), 0)
        asof[c] = {k: (str(col(df, a).index[-1].date()) if col(df, a) is not None else None)
                   for k, df in (("cpi", cpi), ("cli", cli), ("cci", cci))}
    for k in out:
        out[k] = {c: v for c, v in out[k].items() if v is not None}
    return _safe({
        "asof": str(grid[-1].date()), "spot_asof": str(end.date()), "months": [str(d.date()) for d in grid],
        "ccys": list(FXF), "g10": FXF_G10, "cost_bp": FXF_COST, **out, "rate_src": src, "macro_asof": asof,
        "errors": errors,
        "note": "Month-end values; the last month is the latest close so far. Spot in units per USD. Rates % p.a.: "
                "3M = OECD interbank (bis = BIS policy rate), 10Y = OECD long-term government yield (EUR = Germany). "
                "CPI index, CLI (100 = trend) and consumer confidence (100 = average) from the OECD, with a 1-2 month lag.",
    })


# --------------------------------------------------------------------------- #
# Option premium: short 1-month straddles off volatility indices, hedging rules, regimes, seasonality
# --------------------------------------------------------------------------- #

from . import volprem as VP  # noqa: E402

# key -> (underlying, vol index, model, half bid/ask in vol units, hedge cost, name)
VP_MKTS = {
    "SPX": ("^GSPC", "^VIX", "ln", 0.003, 1e-4, "S&P 500"),
    "NDX": ("^NDX", "^VXN", "ln", 0.004, 1e-4, "Nasdaq 100"),
    "OIL": ("USO", "^OVX", "ln", 0.008, 2e-4, "Oil (USO)"),
    "GOLD": ("GLD", "^GVZ", "ln", 0.005, 1e-4, "Gold (GLD)"),
    "UST": ("UST10Y", "^MOVE", "n", 2.0, 0.25, "US Treasuries"),
}
# VIX-style indices price a variance swap, which sits above at-the-money vol because of the skew.
# Straddles are priced at index / ratio (typical index-to-ATM ratios); MOVE is already an ATM normal vol.
VP_ATM_RATIO = {"SPX": 1.10, "NDX": 1.08, "OIL": 1.05, "GOLD": 1.04, "UST": 1.0}
VP_UST_DUR = 8.0          # P&L per bp of yield, as a fraction of bond notional: duration / 10,000
VP_FX = {"EURUSD": "EURUSD=X", "USDJPY": "JPY=X", "GBPUSD": "GBPUSD=X", "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
         "USDCHF": "CHF=X", "NZDUSD": "NZDUSD=X", "USDNOK": "NOK=X", "USDSEK": "SEK=X"}
VP_FX_ETF = {"EURUSD": "FXE", "USDJPY": "FXY", "GBPUSD": "FXB", "AUDUSD": "FXA", "USDCAD": "FXC", "USDCHF": "FXF"}
VP_MODES = ("none", "1", "5", "10", "budget")


def _r4(v):
    return None if v is None or not np.isfinite(v) else float(f"{float(v):.4g}")


def _regimes(x: pd.Series, train_end: str = "2012-12-31") -> tuple[pd.Series, dict]:
    """Four volatility regimes from a Gaussian HMM on weekly log vol, fitted on the early sample and run
    forward (filtered, no look-ahead). Returns the regime (0 low .. 3 distressed) at each week and the model."""
    w = np.log(x.resample("W-FRI").last().dropna().clip(lower=1e-6))
    tr = w[w.index <= train_end]
    if len(tr) < 260:
        tr = w.iloc[: max(260, len(w) // 2)]
    m = VP.hmm_fit(tr.values, 4, iters=40)
    pr = VP.hmm_filter(w.values, m)
    return pd.Series(pr.argmax(1), index=w.index), {"levels": [round(float(math.exp(v)), 4) for v in m["mu"]],
                                                     "today": [round(float(v), 3) for v in pr[-1]],
                                                     "stay": [round(float(m["A"][i, i]), 3) for i in range(4)],
                                                     "train_end": str(tr.index[-1].date())}


def _seasonal(x: pd.Series) -> list[dict]:
    """Average month-on-month change in vol (in vol points) by calendar month, and how often it rose."""
    me = x.resample("ME").last().dropna()
    ch = me.diff().dropna()
    out = []
    for mth in range(1, 13):
        c = ch[ch.index.month == mth]
        out.append({"m": mth, "avg": _r4(c.mean()), "up": _r4((c > 0).mean()) if len(c) else None, "n": int(len(c))})
    return out


@D.cached(ttl=6 * 3600)
def _volpremium(years: float):
    errors: list[str] = []
    syms = [v[0] for v in VP_MKTS.values() if v[0] != "UST10Y"] + [v[1] for v in VP_MKTS.values()] + \
           ["^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW"] + list(VP_FX.values())
    got = D.many_closes(syms, years)
    try:
        ust = _yield_series("UST10Y", years)
        if ust is not None:
            got["UST10Y"] = ust * 100            # bp
    except Exception as e:  # noqa: BLE001
        errors.append(f"UST10Y: {e}")
    mk = {}
    for key, (u, vi, model, oc, hc, name) in VP_MKTS.items():
        s, v = got.get(u), got.get(vi)
        if s is None or v is None:
            errors.append(f"{key}: {'price' if s is None else 'vol index'} missing")
            continue
        df = pd.concat([s.rename("s"), v.rename("v")], axis=1).dropna()
        df = df[(df.s > 0) & (df.v > 0)]
        if len(df) < 600:
            errors.append(f"{key}: short history")
            continue
        iv = ((df.v / 100.0 if model == "ln" else df.v) / VP_ATM_RATIO.get(key, 1.0)).values   # MOVE: bp a year
        sv = df.s.values
        opt, dlt, prem = VP.option_book(sv, iv, model, oc)
        scale = VP_UST_DUR / 1e4 if model == "n" else 1.0
        daily = {}
        for h in VP_MODES:
            hp, done = VP.hedge_pnl(sv, iv, dlt, model, h, hc)
            daily[h] = pd.Series((opt + hp) * scale, index=df.index)
            if h == "budget":
                bdone = done
        rv = VP.realised(df.s, model)
        hv = VP.horizon_vol(df.s, model)
        ivs = pd.Series(iv, index=df.index)
        fwd = rv.shift(-VP.TENOR)
        reg, rmodel = _regimes(ivs)
        regm = reg.resample("ME").last()
        unit = 100.0 if model == "ln" else 1.0                        # vol points (%) or bp
        mo = {h: daily[h].resample("ME").sum() for h in VP_MODES}
        pm = pd.Series(prem * scale, index=df.index).resample("ME").sum()
        idx = mo["none"].index
        bs = VP.budget_state(sv, iv, bdone, model, 10.0)
        last = df.index[-260:]
        mk[key] = {
            "name": name, "model": model, "unit": "bp" if model == "n" else "%",
            "start": str(df.index[0].date()),
            "months": [str(d.date()) for d in idx],
            "iv": [_r4(x * unit) for x in ivs.resample("ME").last().reindex(idx).values],
            "rv_fwd": [_r4(x * unit) for x in fwd.resample("ME").last().reindex(idx).values],
            "hv": [_r4(x * unit) for x in hv.resample("ME").last().reindex(idx).values],
            "ret": {h: [_r4(x) for x in mo[h].values] for h in VP_MODES},
            "prem": [_r4(x) for x in pm.reindex(idx).values],
            "regime": [None if pd.isna(regm.get(d, np.nan)) else int(regm.get(d)) for d in idx],
            "reg_model": rmodel,
            "season": _seasonal(ivs * unit),
            "today": {"date": str(df.index[-1].date()), "spot": _r4(sv[-1]), "iv": _r4(iv[-1] * unit),
                      "rv1m": _r4(rv.iloc[-1] * unit), "rv3m": _r4(VP.realised(df.s, model, 63).iloc[-1] * unit),
                      "hv": _r4(hv.iloc[-1] * unit),
                      "iv_pct": _r4((ivs.iloc[-VP.DAYS * 10:] <= ivs.iloc[-1]).mean()),
                      "budget": {k: _r4(v) if isinstance(v, float) else v for k, v in bs.items()},
                      "hedges_per_month": _r4(bdone[-VP.DAYS:].sum() / 12)},
            "daily": {"d": [str(d.date()) for d in last], "iv": [_r4(x * unit) for x in ivs[last].values],
                      "rv": [_r4(x * unit) for x in rv[last].values]},
        }
    # FX: realised only (no free implied history), plus today's implied from currency ETF options
    fx = {}
    for pair, sym in VP_FX.items():
        s = got.get(sym)
        if s is None or len(s) < 600:
            errors.append(f"{pair} spot")
            continue
        rv = VP.realised(s, "ln").dropna() * 100
        reg, rmodel = _regimes(rv / 100)
        regm = reg.resample("ME").last()
        me = rv.resample("ME").last()
        fx[pair] = {"months": [str(d.date()) for d in me.index], "rv": [_r4(x) for x in me.values],
                    "regime": [None if pd.isna(regm.get(d, np.nan)) else int(regm.get(d)) for d in me.index],
                    "reg_model": rmodel, "season": _seasonal(rv),
                    "today": {"rv1m": _r4(rv.iloc[-1]), "rv3m": _r4(VP.realised(s, "ln", 63).iloc[-1] * 100),
                              "date": str(rv.index[-1].date())}}
    spy = {}   # option chains are slow and unreliable from the server; skew comes from CBOE SKEW instead
    term = {k: _r4(got[k].iloc[-1]) if got.get(k) is not None else None for k in ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW")}
    v, v3 = got.get("^VIX"), got.get("^VIX3M")
    ratio = []
    if v is not None and v3 is not None:
        r = (v / v3).dropna().resample("ME").last()
        ratio = [{"d": str(d.date()), "v": _r4(x)} for d, x in r.items()]
    return {"asof": str(date.today()), "markets": mk, "fx": fx, "term": term, "vix_vix3m": ratio,
            "spy_skew": {k: _r4(spy.get(k)) if isinstance(spy.get(k), (int, float)) else spy.get(k) for k in ("atm_iv", "put90_iv", "skew_90", "days")},
            "modes": list(VP_MODES), "errors": errors,
            "atm_ratio": VP_ATM_RATIO,
            "note": "Short 1-month ATM straddles sold weekly (1/4 notional each), priced Black-Scholes off the vol index "
                    "divided by a typical index-to-ATM ratio (atm_ratio) "
                    "(Bachelier off MOVE for the 10-year yield, P&L x duration 8), marked daily, costs included. "
                    "Returns are fractions of notional per month. Regimes: 4-state HMM on weekly log vol, fitted on the "
                    "early sample, filtered forward. FX: realised vol (Yahoo), today's implied from currency ETF options."}


import threading  # noqa: E402

VP_LOCK = threading.Lock()


def _warm_volpremium():
    """Compute the default option-premium payload in the background at start-up, so the first page
    load after the free server wakes finds it cached instead of timing out."""
    import time  # noqa: PLC0415
    time.sleep(8)
    try:
        with VP_LOCK:
            _volpremium(float(D.year_bucket(22)))
    except Exception as e:  # noqa: BLE001
        log.warning("volpremium warm-up failed: %s", e)
    try:
        _dimensions(20.0)
    except Exception as e:  # noqa: BLE001
        log.warning("dimensions warm-up failed: %s", e)


# Off by default: on the 512 MB free instance a start-up warm-up can run out of memory and restart the service.
if os.environ.get("KEEL_WARM", "0") == "1" and not os.environ.get("PYTEST_CURRENT_TEST"):
    threading.Thread(target=_warm_volpremium, daemon=True).start()


@app.get("/api/volpremium", operation_id="option_premium", dependencies=[Depends(check_token)])
def vol_premium(years: int = Query(22, ge=8, le=26)):
    """Option premium across equities, oil, gold, Treasuries and FX: implied vs realised vol at the option's
    horizon, back-tests of selling 1-month straddles (unhedged, hedged every 1/5/10 days, or on a variance
    budget), four volatility regimes (HMM), seasonality, the VIX term structure, and today's state
    (implied vol percentile, hedge budget used). Heavy: cached for six hours."""
    try:
        with VP_LOCK:
            return _safe(_volpremium(float(D.year_bucket(years))))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Option premium data did not load: {e}") from e


# --------------------------------------------------------------------------- #
# Dimensions: monthly returns of assets and systematic strategies, for diversification counting
# --------------------------------------------------------------------------- #

DIM_ASSETS = {  # key -> (Yahoo symbol, name, group)
    "SPY": ("SPY", "US equities", "Equities"), "EFA": ("EFA", "Developed ex-US equities", "Equities"),
    "EEM": ("EEM", "EM equities", "Equities"), "EWJ": ("EWJ", "Japan equities (USD)", "Equities"),
    "IEF": ("IEF", "US Treasuries 7-10Y", "Bonds"), "TLT": ("TLT", "US Treasuries 20Y+", "Bonds"),
    "LQD": ("LQD", "US IG credit", "Bonds"), "HYG": ("HYG", "US high yield", "Bonds"),
    "EMB": ("EMB", "EM USD bonds", "Bonds"), "GLD": ("GLD", "Gold", "Real assets"),
    "DBC": ("DBC", "Commodities", "Real assets"), "VNQ": ("VNQ", "US REITs", "Real assets"),
    "BTC": ("BTC-USD", "Bitcoin", "Real assets"), "JPY": ("JPY=X", "Yen vs USD", "FX"),
    "EUR": ("EUR=X", "Euro vs USD", "FX"),
}
DIM_TREND = ("SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "JPY", "EUR")


def _trend_returns(r: pd.DataFrame) -> pd.Series:
    """Multi-asset time-series momentum: each market long or short by the sign of its 12-month return,
    sized to equal volatility (trailing 12 months), equal risk across markets, monthly."""
    cols = [c for c in DIM_TREND if c in r]
    x = r[cols]
    sig = np.sign((1 + x).rolling(12).apply(np.prod, raw=True) - 1)
    vol = x.rolling(12, min_periods=10).std() * math.sqrt(12)
    pos = (sig * (0.10 / vol)).clip(-3, 3)
    out = (pos.shift(1) * x).mean(axis=1, skipna=True)
    out[pos.shift(1).notna().sum(axis=1) < 4] = np.nan
    return out


def _fx_carry_returns(d: dict) -> pd.Series:
    """Long the three highest and short the three lowest 3-month rates among USD and ten liquid currencies,
    funded in dollars (spot move plus rate gap), monthly."""
    univ = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "MXN", "ZAR", "PLN", "HUF", "CZK"]
    idx = pd.to_datetime(d["months"])
    sp = pd.DataFrame({c: d["spot"].get(c) for c in univ if c in d["spot"]}, index=idx).astype(float)
    rt = pd.DataFrame({c: d["rate"].get(c) for c in univ if c in d["rate"]}, index=idx).astype(float)
    xr = (sp.shift(1) / sp) * (1 + rt.shift(1) / 1200)
    xr = xr.sub(1 + rt["USD"].shift(1) / 1200, axis=0)
    xr["USD"] = 0.0
    out = pd.Series(np.nan, index=idx)
    for i in range(1, len(idx)):
        rr = rt.iloc[i - 1].dropna()
        ok = [c for c in rr.index if pd.notna(xr.iloc[i].get(c))]
        if len(ok) < 8:
            continue
        o = rr[ok].sort_values()
        out.iloc[i] = xr.iloc[i][o.index[-3:]].mean() - xr.iloc[i][o.index[:3]].mean()
    return out


def _rates_carry_returns(d: dict) -> pd.Series:
    """Receive the 10-year and pay the 3-month in G10 markets, equal weight, FX hedged, standing aside
    for the month when the US 10-year is above its 160-day trend (5-day average) at the month end."""
    idx = pd.to_datetime(d["months"])
    g10 = [c for c in d["groups"]["G10"] if c in d["long"]]
    rets = []
    for c in g10:
        y = pd.Series(d["long"][c], index=idx, dtype=float)
        s3 = pd.Series(d["short"][c], index=idx, dtype=float)
        dur = (1 - (1 + y.clip(lower=0.05) / 100) ** -10) / (y.clip(lower=0.05) / 100)
        rets.append(((y.shift(1) - s3.shift(1)) / 1200 - dur.shift(1) * y.diff() / 100).rename(c))
    r = pd.concat(rets, axis=1).mean(axis=1, skipna=True)
    gap = pd.Series(d.get("usd_filter_gap_bp") or [None] * len(idx), index=idx, dtype=float)
    on = (gap.shift(1) <= 0) | gap.shift(1).isna()
    return r.where(on, 0.0)


@D.cached(ttl=12 * 3600)
def _spx_shortvol(years: float) -> pd.Series:
    """Monthly returns of selling S&P 1-month straddles weekly, hedged weekly (as in Premium), on its own so
    this route does not need the whole option-premium computation."""
    got = D.many_closes(["^GSPC", "^VIX"], years + 1)
    df = pd.concat([got["^GSPC"].rename("s"), got["^VIX"].rename("v")], axis=1).dropna()
    iv = (df.v / 100.0 / VP_ATM_RATIO["SPX"]).values
    opt, dlt, _ = VP.option_book(df.s.values, iv, "ln", 0.003)
    hp, _ = VP.hedge_pnl(df.s.values, iv, dlt, "ln", "5", 1e-4)
    return pd.Series(opt + hp, index=df.index).resample("ME").sum()


@D.cached(ttl=6 * 3600)
def _dimensions(years: float):
    errors: list[str] = []
    px = D.monthly_closes(tuple(v[0] for v in DIM_ASSETS.values()), years + 1, True)
    r = {}
    for k, (sym, _, _) in DIM_ASSETS.items():
        s = px.get(sym)
        if s is None or len(s) < 24:
            errors.append(f"{k} prices")
            continue
        s = s.resample("ME").last()
        r[k] = (s.shift(1) / s - 1) if k in ("JPY", "EUR") else s.pct_change()   # FX: units per USD -> long the currency
    R = pd.DataFrame(r)
    strat, names = {}, {}
    try:
        strat["TREND"] = _trend_returns(R)
        names["TREND"] = "Trend (12-month, multi-asset)"
    except Exception as e:  # noqa: BLE001
        errors.append(f"trend: {e}")
    for key, name, fn in (("FXCARRY", "FX carry (top 3 vs bottom 3)", lambda: _fx_carry_returns(fx_carry(22))),
                          ("RATESCARRY", "Rates carry (G10, trend-filtered)", lambda: _rates_carry_returns(rates_carry(25))),
                          ("SHORTVOL", "Short vol (S&P straddles, weekly hedge)", None)):
        try:
            if key == "SHORTVOL":
                ser = _spx_shortvol(years)
            else:
                ser = fn()
            strat[key] = ser
            names[key] = name
        except Exception as e:  # noqa: BLE001
            errors.append(f"{key}: {e}")
    all_ = pd.concat([R] + [s.rename(k) for k, s in strat.items()], axis=1)
    all_.index = all_.index + pd.offsets.MonthEnd(0)
    all_ = all_.groupby(level=0).last()
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=int(years))
    all_ = all_[all_.index >= start]
    series = {}
    for k in all_.columns:
        meta = DIM_ASSETS.get(k)
        series[k] = {"name": meta[1] if meta else names.get(k, k), "group": meta[2] if meta else "Strategies",
                     "vals": [None if pd.isna(v) else round(float(v), 5) for v in all_[k].values]}
    return {"asof": str(all_.index[-1].date()), "months": [str(d.date()) for d in all_.index],
            "series": series, "errors": errors,
            "note": "Monthly total returns in USD (ETFs, dividends included); FX = holding the currency against USD, "
                    "spot only. Strategies: trend = sign of 12-month return, equal vol; FX carry = top 3 minus bottom 3 "
                    "3-month rates among 11 currencies; rates carry = G10 receivers, US trend filter; short vol = S&P "
                    "1-month straddles, hedged weekly (as in Premium). The last month is month to date."}


@app.get("/api/dimensions", operation_id="dimensions_returns", dependencies=[Depends(check_token)])
def dimensions(years: int = Query(20, ge=5, le=22)):
    """Monthly returns of a multi-asset universe (equities, bonds, credit, gold, commodities, REITs, bitcoin,
    yen, euro) and four systematic strategies (trend, FX carry, rates carry, short volatility), aligned on one
    grid, for counting diversification: effective number of independent drivers in a universe or portfolio."""
    try:
        return _safe(_dimensions(float(years)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Dimensions data did not load: {e}") from e


# --------------------------------------------------------------------------- #
# Macro regimes (growth x inflation)
# --------------------------------------------------------------------------- #

# Short code -> (OECD CLI area codes to try, OECD CPI area code)
MACRO_AREAS = {
    "US": (("USA",), "USA"), "JP": (("JPN",), "JPN"), "EA": (("EA20", "EA19"), "EA20"),
    "UK": (("GBR",), "GBR"), "DE": (("DEU",), "DEU"), "CN": (("CHN",), "CHN"),
}


@app.get("/api/macro", operation_id="macro_regime_data", dependencies=[Depends(check_token)])
def macro(
    areas: str = Query("US,JP", description="Comma-separated: US, JP, EA, UK, DE, CN"),
    start: str = Query("2000-01", pattern=r"^\d{4}-\d{2}$", description="First month, YYYY-MM"),
):
    """Monthly OECD composite leading indicator (amplitude adjusted, 100 = long-term trend),
    business and consumer confidence (`bci`, `cci`; amplitude adjusted, 100 = long-term average)
    and CPI inflation (% year on year) per area, for growth x inflation regime work.

    Both series are published with a lag of about one to two months; `asof` gives each
    series' latest month so callers can lag their signals accordingly."""
    want = [a for a in _syms(areas.upper(), 6) if a in MACRO_AREAS]
    if not want:
        raise HTTPException(400, f"Areas must be among {', '.join(MACRO_AREAS)}")
    errors: list[str] = []
    cli_codes = tuple(sorted({c for a in want for c in MACRO_AREAS[a][0]}))
    try:
        cli = D.oecd_cli(cli_codes, start)
    except Exception as e:  # noqa: BLE001
        cli = pd.DataFrame()
        errors.append(f"OECD CLI: {e}")
    try:
        cpi = D.cpi_yoy(tuple(sorted({MACRO_AREAS[a][1] for a in want})), start)
    except Exception as e:  # noqa: BLE001
        cpi = pd.DataFrame()
        errors.append(f"OECD CPI: {e}")
    conf = {}
    for key, measure in (("bci", "BCICP"), ("cci", "CCICP")):
        try:
            conf[key] = D.oecd_confidence(cli_codes, measure, start)
        except Exception as e:  # noqa: BLE001
            conf[key] = pd.DataFrame()
            errors.append(f"OECD {measure}: {e}")
    if cli.empty and cpi.empty:
        raise HTTPException(503, "OECD macro data unavailable right now; " + "; ".join(errors or ["empty answer"]))

    def pts(s: pd.Series | None) -> list[dict]:
        if s is None:
            return []
        s = s.dropna()
        return [{"d": str(i.date()), "v": float(v)} for i, v in s.items()]

    series, asof, missing = {}, {}, []
    for a in want:
        cli_s = next((cli[c] for c in MACRO_AREAS[a][0] if c in cli and cli[c].notna().any()), None)
        cpi_s = cpi[MACRO_AREAS[a][1]] if MACRO_AREAS[a][1] in cpi else None
        if cli_s is None and cpi_s is None:
            missing.append(a)
            continue
        series[a] = {"cli": pts(cli_s), "cpi_yoy": pts(cpi_s)}
        for key, df in conf.items():
            series[a][key] = pts(next((df[c] for c in MACRO_AREAS[a][0] if c in df and df[c].notna().any()), None))
        asof[a] = {k: (v[-1]["d"] if v else None) for k, v in series[a].items()}
    return _safe({
        "series": series, "asof": asof, "missing": missing, "errors": errors,
        "note": "OECD composite leading indicator and business/consumer confidence (amplitude adjusted, long-term average = 100) and "
                "CPI all items, % change on a year earlier. Month-end dates. Both are revised and published "
                "with a lag of about 1-2 months.",
    })


# --------------------------------------------------------------------------- #
# News-based measures (Bybee, Kelly, Manela and Xiu 2024): app/news.py
# --------------------------------------------------------------------------- #
from . import news as N  # noqa: E402


@app.get("/api/newsmood", operation_id="news_sentiment", dependencies=[Depends(check_token)])
def newsmood(start: str = Query("2000-01-01", description="YYYY-MM-DD"), daily_days: int = Query(730, ge=30, le=3700)):
    """San Francisco Fed Daily News Sentiment Index: tone of economics articles in 24 US newspapers,
    daily since 1980, updated about weekly. Returns month averages from `start` and the last
    `daily_days` of daily values. Positive = more upbeat economic news than average."""
    try:
        s = N.news_sentiment()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"SF Fed news sentiment unavailable: {e}") from e
    s = s[s.index >= pd.Timestamp(start)]
    m = s.resample("ME").mean().dropna()
    d = s[s.index >= s.index[-1] - pd.Timedelta(days=daily_days)]
    return _safe({
        "asof": str(s.index[-1].date()),
        "monthly": [{"d": str(i.date()), "v": float(v)} for i, v in m.items()],
        "daily": [{"d": str(i.date()), "v": float(v)} for i, v in d.items()],
        "source": "Federal Reserve Bank of San Francisco, Daily News Sentiment Index (Shapiro, Sudhof and Wilson)",
        "note": "Month = average of daily values; the current month is partial.",
    })


@app.get("/api/fxnews", operation_id="fx_intervention_news", dependencies=[Depends(check_token)])
def fxnews(days: int = Query(90, ge=14, le=180, description="Days of daily coverage counts"),
           ladder_days: int = Query(30, ge=7, le=60, description="Days of headlines read for the verbal ladder")):
    """Yen-intervention news from Google News RSS: daily count of Japanese headlines mentioning
    為替介入 (coverage), and recent JA/EN headlines classified by how far official language has
    escalated (2 urgency, 3 excessive/speculative, 4 decisive/all options, 5 rate check or
    intervention reported). Headlines link to the articles."""
    errors = []
    cov = None
    try:
        cov = N.coverage("為替介入", "ja", days)
        if cov["failed_windows"]:
            errors.append(f"coverage: {cov['failed_windows']} of {cov['windows']} windows failed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"coverage: {e}")
    lad = N.ladder_headlines(ladder_days)
    errors += lad["errors"]
    if cov is None and not lad["items"]:
        raise HTTPException(503, "Google News did not answer: " + "; ".join(errors))
    return _safe({
        "asof": datetime_now_jst(),
        "coverage": cov,
        "headlines": lad["items"][:120],
        "ladder": [{"rung": r, "score": s, "pattern": p} for r, s, p in N.LADDER],
        "errors": errors,
        "note": "Headline counts depend on what Google News indexes; compare a day with its own history, not with other sources.",
    })


def datetime_now_jst() -> str:
    from datetime import datetime  # noqa: PLC0415

    return datetime.now(N.JST).isoformat(timespec="minutes")


from . import sentiment as S  # noqa: E402


@app.get("/api/cot", operation_id="cot_positioning", dependencies=[Depends(check_token)])
def cot_positioning(
    markets: str = Query("JPY,EUR,AUD,SPX,TY,GC", description="Comma-separated keys: " + ",".join(S.COT_MARKETS)),
    start: str = Query("2006-01-01", description="YYYY-MM-DD"),
):
    """CFTC Commitments of Traders (legacy, futures only), weekly as of Tuesday, released Friday.
    Per market: open interest, large-speculator (non-commercial) longs, shorts, net, and net as %
    of open interest (`pct`), plus commercials' net (`cnet`). Positive = speculators net long the
    contract (for JPY: long yen, i.e. short USD/JPY)."""
    keys = [k.strip().upper() for k in markets.split(",") if k.strip().upper() in S.COT_MARKETS][:20]
    if not keys:
        raise HTTPException(400, "No known markets. Choose from " + ",".join(S.COT_MARKETS))
    try:
        got = S.cot(tuple(keys), start[:10])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"CFTC unavailable: {e}") from e
    return _safe({
        "series": got,
        "names": {k: {"en": S.COT_MARKETS[k][1], "ja": S.COT_MARKETS[k][2]} for k in keys},
        "missing": [k for k in keys if k not in got],
        "asof": max((v[-1]["d"] for v in got.values() if v), default=None),
        "source": "CFTC Commitments of Traders, legacy futures-only report (publicreporting.cftc.gov)",
    })


@app.get("/api/jpflows", operation_id="japan_flows", dependencies=[Depends(check_token)])
def japan_flows(start: str = Query("2005-01-01", description="YYYY-MM-DD")):
    """Japan equity flows and positioning, weekly.
    `mof`: non-residents' net purchases of Japanese equities (fe) and long-term bonds (fb), and
    residents' net purchases of foreign equities (re), 100 million yen (MOF, since 2005).
    `margin`: margin shorts and longs, Tokyo + Nagoya (shares in thousands, value in million yen);
    longs / shorts in shares is the 信用倍率 (JPX, since 2002).
    `investors`: last few weeks of TSE Prime net buying by foreigners, individuals, trust banks
    and investment trusts, 100 million yen (JPX)."""
    errors, out = [], {}
    for k, fn in (("mof", S.mof_week), ("margin", S.jpx_margin), ("investors", S.jpx_investors)):
        try:
            out[k] = [r for r in fn() if r["d"] >= start[:10]]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{k}: {e}")
            out[k] = []
    if not any(out.values()):
        raise HTTPException(503, "MOF and JPX did not answer: " + "; ".join(errors))
    return _safe({**out, "errors": errors,
                  "asof": {k: (v[-1]["d"] if v else None) for k, v in out.items()},
                  "source": "Ministry of Finance (対外及び対内証券売買契約等の状況, weekly); JPX (信用取引現在高; 投資部門別売買状況)"})


@app.get("/api/headlines", operation_id="news_headlines", dependencies=[Depends(check_token)])
def news_headlines(
    q: str = Query(..., description="Google News search, e.g. 'stock market' or '日経平均'"),
    lang: str = Query("en", pattern="^(en|ja)$"),
    days: int = Query(3, ge=1, le=14),
    limit: int = Query(40, ge=5, le=100),
):
    """Recent Google News headlines for a query (title, source, JST date, link), newest first.
    For scoring tone yourself; Keel does not score them."""
    try:
        items = N._gnews(f"{q} when:{days}d", lang)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, str(e)) from e
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["d"], reverse=True):
        if it["title"] in seen:
            continue
        seen.add(it["title"])
        out.append(it)
    return _safe({"q": q, "lang": lang, "asof": datetime_now_jst(), "items": out[:limit]})


@app.get("/api/social", operation_id="social_sentiment", dependencies=[Depends(check_token)])
def social_sentiment(
    symbols: str = Query("SPY,QQQ,IWM,NIKKEI,EWJ,USDJPY", description="StockTwits symbols, e.g. " + ",".join(S.ST_SYMBOLS)),
    pages: int = Query(2, ge=1, le=5, description="Pages of 30 messages per symbol"),
):
    """StockTwits: the latest messages per symbol and how many their authors tagged Bullish or Bearish.
    `bull_share` = bullish / (bullish + bearish). Busy symbols (SPY, QQQ) cover a few hours, quiet ones
    (EWJ, NIKKEI) weeks, so compare a symbol with itself."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    try:
        got = S.stocktwits(tuple(syms), pages)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"StockTwits unavailable: {e}") from e
    out = {}
    for s, ms in got.items():
        b, r = sum(m["s"] == 1 for m in ms), sum(m["s"] == -1 for m in ms)
        out[s] = {"n": len(ms), "bull": b, "bear": r, "bull_share": round(b / (b + r), 3) if b + r else None,
                  "from": ms[-1]["t"] if ms else None, "to": ms[0]["t"] if ms else None, "items": ms[:40]}
    return _safe({"series": out, "missing": [s for s in syms if not got.get(s)], "asof": datetime_now_jst(),
                  "source": "StockTwits public API; tags are chosen by each message's author"})


@app.get("/api/reddit", operation_id="reddit_posts", dependencies=[Depends(check_token)])
def reddit_posts(
    subs: str = Query(",".join(S.REDDIT_SUBS), description="Subreddits, comma-separated"),
    sort: str = Query("hot", pattern="^(hot|new|top)$"),
    limit: int = Query(50, ge=5, le=100),
):
    """Recent Reddit post titles from investing subreddits (title, score, comments, flair, link), via
    Reddit's official Data API. Needs REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (a free 'script' app at
    reddit.com/prefs/apps) on the server. Keel does not score tone."""
    if not S.reddit_configured():
        raise HTTPException(503, "Reddit is not configured: set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET on the Keel server")
    names = [x.strip().removeprefix("r/") for x in subs.split(",") if x.strip()][:8]
    try:
        got = S.reddit(tuple(names), sort, limit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Reddit unavailable: {e}") from e
    return _safe({"posts": got, "missing": [n for n in names if not got.get(n)], "sort": sort, "asof": datetime_now_jst(),
                  "source": "Reddit Data API"})
