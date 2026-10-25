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
    got = D.many_closes(psyms, years, dividends=total) if psyms else {}
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
        out[s] = [{"d": str(i.date()), "v": float(v)} for i, v in c.dropna().items()]
    units = {s: ("pct" if s in ysyms else "price") for s in out}
    return _safe({"series": out, "units": units, "missing": [s for s in syms if s not in out],
                  "total_return": bool(total)})


@app.get("/api/ohlc", operation_id="price_bars", dependencies=[Depends(check_token)])
def ohlc_bars(
    symbols: str = Query(..., description="Comma-separated Yahoo symbols, or UST2Y..UST30Y / JGB2Y..JGB30Y "
                         "(yields: close only, so open = high = low = close)"),
    start: str | None = Query(None, description="YYYY-MM-DD; default 2 years ago"),
    interval: str = Query("D", pattern="^(D|W|M)$", description="D, W (Friday) or M (month-end) bars"),
):
    """Open, high, low, close and volume bars per symbol, for bar-pattern studies (e.g. DeMark).
    Prices are split-adjusted but not dividend-adjusted. `close_only` lists symbols whose
    high/low are not available (government yields)."""
    syms = _syms(symbols, 12)
    st = pd.Timestamp(start) if start else pd.Timestamp(date.today() - timedelta(days=730))
    years = max(1.0, (pd.Timestamp(date.today()) - st).days / 365.25 + 0.1)
    ysyms = [s for s in syms if s.upper() in YIELD_SYMS]
    psyms = [s for s in syms if s not in ysyms]
    got = D._pool(lambda s: D.ohlc(s, years), psyms, workers=3) if psyms else {}
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
        out[s] = [{"d": str(i.date()), "o": float(r.open), "h": float(r.high), "l": float(r.low), "c": float(r.close),
                   **({"v": float(r.volume)} if pd.notna(r.volume) and r.volume > 0 else {})}
                  for i, r in df.iterrows()]
    return _safe({"series": out, "close_only": [s for s in ysyms if s in out],
                  "missing": [s for s in syms if s not in out], "interval": interval})


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
