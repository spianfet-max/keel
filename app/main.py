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


@app.get("/api/history", operation_id="price_history", dependencies=[Depends(check_token)])
def history(
    symbols: str = Query(..., description="Comma-separated Yahoo symbols"),
    start: str | None = Query(None, description="YYYY-MM-DD; default 2 years ago"),
    interval: str = Query("M", pattern="^(D|W|M)$", description="D, W or M (period-end closes)"),
):
    """Closing prices per symbol, resampled to day, week or month end."""
    syms = _syms(symbols, 12)
    st = pd.Timestamp(start) if start else pd.Timestamp(date.today() - timedelta(days=730))
    years = max(1.0, (pd.Timestamp(date.today()) - st).days / 365.25 + 0.1)
    out = {}
    for s, c in D.many_closes(syms, years).items():
        c = c[c.index >= st]
        if interval == "W":
            c = c.resample("W-FRI").last()
        elif interval == "M":
            c = c.resample("ME").last()
        out[s] = [{"d": str(i.date()), "v": float(v)} for i, v in c.dropna().items()]
    return _safe({"series": out, "missing": [s for s in syms if s not in out]})


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

