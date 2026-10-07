"""Pure calculations used by the Keel API.

Everything here takes plain pandas / numpy inputs so it can be unit-tested
without any network access. Rates are in percent (4.25 == 4.25%).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #


def to_percent(values: pd.Series) -> pd.Series:
    """Normalise a rate series to percent.

    Providers disagree: ECB/Fed return decimals (0.042), MOF returns percent (4.2).
    A median absolute value below 0.25 is treated as decimal.
    """
    v = pd.to_numeric(values, errors="coerce")
    med = v.dropna().abs().median()
    if med is not None and not math.isnan(med) and med < 0.25:
        return v * 100
    return v


def hedge_cost(r_foreign: float, r_jpy: float, basis_bp: float = 0.0) -> float:
    """Annualised cost (in %) for a JPY investor to hedge a foreign-currency asset.

    Covered interest parity: forward points ~ rate differential. A negative
    cross-currency basis (typical for USD/JPY) raises the cost for a yen investor,
    so the basis is entered as the extra cost in bp.
    """
    return (r_foreign - r_jpy) + basis_bp / 100.0


def hedged_yield(y_foreign: float, r_foreign: float, r_jpy: float, basis_bp: float = 0.0) -> float:
    """Yen-hedged yield (in %) of a foreign bond."""
    return y_foreign - hedge_cost(r_foreign, r_jpy, basis_bp)


def interp_curve(curve: dict[float, float], tenor: float) -> float | None:
    """Linear interpolation on a {years: rate} curve; flat beyond the ends."""
    pts = sorted((k, v) for k, v in curve.items() if v is not None and not math.isnan(v))
    if not pts:
        return None
    if tenor <= pts[0][0]:
        return pts[0][1]
    if tenor >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= tenor <= x1:
            return y0 + (y1 - y0) * (tenor - x0) / (x1 - x0)
    return None


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #


SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 20, 25, 50, 100)


def fix_splits(closes: pd.Series, tol: float = 0.06) -> pd.Series:
    """Undo unadjusted stock splits: a one-day move close to 1:n or n:1 rescales the earlier history.

    Yahoo occasionally misses splits for Japanese ETFs and small caps, which shows up
    as a single 90% 'crash'. Real one-day moves of that size essentially never happen
    in the instruments these tools cover.
    """
    c = pd.to_numeric(closes, errors="coerce").dropna().astype(float).copy()
    if len(c) < 3:
        return c
    vals = c.values
    for i in range(1, len(vals)):
        r = vals[i] / vals[i - 1]
        if 0.4 < r < 2.5:
            continue
        for n in SPLIT_RATIOS:
            for target in (1 / n, float(n)):
                if abs(r / target - 1) < tol:
                    vals[:i] = vals[:i] * target
                    break
            else:
                continue
            break
    return pd.Series(vals, index=c.index, name=closes.name)


def log_returns(closes: pd.Series) -> pd.Series:
    c = pd.to_numeric(closes, errors="coerce").dropna()
    c = c[c > 0]
    return np.log(c / c.shift(1)).dropna()


def realized_vol(closes: pd.Series, window: int | None = None) -> float | None:
    """Annualised close-to-close volatility in %."""
    r = log_returns(closes)
    if window:
        r = r.iloc[-window:]
    if len(r) < 5:
        return None
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS) * 100)


def pct_change_since(closes: pd.Series, days_back: int) -> float | None:
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if len(c) <= days_back:
        return None
    return float((c.iloc[-1] / c.iloc[-1 - days_back] - 1) * 100)


def abs_change_since(closes: pd.Series, days_back: int) -> float | None:
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if len(c) <= days_back:
        return None
    return float(c.iloc[-1] - c.iloc[-1 - days_back])


def ytd_base(closes: pd.Series) -> float | None:
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if c.empty:
        return None
    idx = pd.to_datetime(c.index)
    prior = c[idx.year < idx[-1].year]
    return float(prior.iloc[-1] if not prior.empty else c[idx.year == idx[-1].year].iloc[0])


def ytd_change(closes: pd.Series) -> float | None:
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if c.empty:
        return None
    idx = pd.to_datetime(c.index)
    year = idx[-1].year
    prior = c[idx.year < year]
    base = prior.iloc[-1] if not prior.empty else c[idx.year == year].iloc[0]
    return float((c.iloc[-1] / base - 1) * 100)


def max_drawdown(closes: pd.Series) -> float | None:
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if c.empty:
        return None
    dd = c / c.cummax() - 1
    return float(dd.min() * 100)


def snapshot_stats(closes: pd.Series) -> dict:
    """Headline numbers for one instrument."""
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if c.empty:
        return {}
    last = float(c.iloc[-1])
    tail = c.iloc[-252:]
    return {
        "last": last,
        "asof": str(pd.to_datetime(c.index[-1]).date()),
        "chg_1d": pct_change_since(c, 1),
        "chg_1w": pct_change_since(c, 5),
        "chg_1m": pct_change_since(c, 21),
        "chg_3m": pct_change_since(c, 63),
        "chg_ytd": ytd_change(c),
        "chg_1y": pct_change_since(c, 252),
        # level changes: use these for yields (in percentage points) where % changes mislead
        "abs_1d": abs_change_since(c, 1),
        "abs_1w": abs_change_since(c, 5),
        "abs_1m": abs_change_since(c, 21),
        "abs_ytd": (last - ytd_base(c)) if ytd_base(c) is not None else None,
        "abs_1y": abs_change_since(c, 252),
        "vol_20d": realized_vol(c, 20),
        "vol_60d": realized_vol(c, 60),
        "vol_1y": realized_vol(c, 252),
        "high_52w": float(tail.max()),
        "low_52w": float(tail.min()),
        "pos_52w": float((last - tail.min()) / (tail.max() - tail.min()) * 100)
        if tail.max() > tail.min()
        else None,
        "max_dd_1y": max_drawdown(tail),
    }


def sparkline(closes: pd.Series, points: int = 90) -> list[dict]:
    c = pd.to_numeric(closes, errors="coerce").dropna().iloc[-points:]
    return [{"d": str(pd.to_datetime(i).date()), "v": round(float(v), 6)} for i, v in c.items()]


def correlation_matrix(prices: dict[str, pd.Series], window: int = 252) -> dict:
    rets = pd.DataFrame({k: log_returns(v) for k, v in prices.items()}).dropna()
    rets = rets.iloc[-window:]
    if rets.shape[0] < 20 or rets.shape[1] < 2:
        return {"symbols": list(prices), "matrix": None}
    m = rets.corr()
    return {
        "symbols": list(m.columns),
        "matrix": [[round(float(x), 3) for x in row] for row in m.values],
    }


# --------------------------------------------------------------------------- #
# Structured products: historical barrier back-test
# --------------------------------------------------------------------------- #


def barrier_backtest(
    prices: dict[str, pd.Series],
    tenor_days: int = 252,
    ki: float = 0.6,
    autocall: float = 1.0,
    obs_every: int = 63,
    strike: float = 1.0,
) -> dict:
    """Roll a worst-of autocall / FCN through history, one start date per day.

    For each start date the basket is rebased to 1. The note:
      * autocalls on the first observation date where the worst performer >= autocall,
      * otherwise is knocked in if the worst performer ever closes < ki,
      * and loses (1 - worst final) if knocked in and worst final < strike.
    Daily closes approximate a continuous barrier.
    """
    df = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in prices.items()}).dropna()
    n = len(df)
    if n <= tenor_days + 5:
        return {"windows": 0}

    arr = df.values
    starts = range(0, n - tenor_days)
    ki_hits = autocalls = losses = 0
    loss_sizes: list[float] = []
    call_times: list[int] = []
    worst_finals: list[float] = []

    for s in starts:
        path = arr[s : s + tenor_days + 1] / arr[s]
        worst = path.min(axis=1)
        called_at = None
        for t in range(obs_every, tenor_days + 1, obs_every):
            if worst[t] >= autocall:
                called_at = t
                break
        if called_at is not None:
            autocalls += 1
            call_times.append(called_at)
            continue
        final = worst[-1]
        worst_finals.append(float(final))
        if worst.min() < ki:
            ki_hits += 1
            if final < strike:
                losses += 1
                loss_sizes.append(float((1 - final / strike) * 100))

    total = len(starts)
    return {
        "windows": total,
        "from": str(pd.to_datetime(df.index[0]).date()),
        "to": str(pd.to_datetime(df.index[-1]).date()),
        "autocall_rate": round(autocalls / total * 100, 1),
        "avg_call_months": round(float(np.mean(call_times)) / 21, 1) if call_times else None,
        "ki_rate": round(ki_hits / total * 100, 1),
        "loss_rate": round(losses / total * 100, 1),
        "avg_loss": round(float(np.mean(loss_sizes)), 1) if loss_sizes else None,
        "worst_loss": round(float(np.max(loss_sizes)), 1) if loss_sizes else None,
    }


def min_distance_path(closes: pd.Series, tenor_days: int = 252) -> float | None:
    """Worst rolling-tenor drawdown from start, in % (how close history came to a KI)."""
    c = pd.to_numeric(closes, errors="coerce").dropna().values
    if len(c) <= tenor_days:
        return None
    worst = 1.0
    for s in range(0, len(c) - tenor_days):
        m = c[s : s + tenor_days + 1].min() / c[s]
        worst = min(worst, m)
    return round((worst - 1) * 100, 1)


# --------------------------------------------------------------------------- #
# Portfolio (Frontier)
# --------------------------------------------------------------------------- #


def monthly_return_stats(prices: dict[str, pd.Series]) -> dict:
    """Annualised mean, vol and covariance from month-end returns."""
    df = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in prices.items()})
    df.index = pd.to_datetime(df.index)
    m = df.resample("ME").last().dropna(how="any")
    r = m.pct_change().dropna()
    if len(r) < 12:
        return {"symbols": list(prices), "months": len(r)}
    mu = r.mean() * 12
    cov = r.cov() * 12
    return {
        "symbols": list(r.columns),
        "months": int(len(r)),
        "from": str(r.index[0].date()),
        "to": str(r.index[-1].date()),
        "mean": {k: round(float(v) * 100, 3) for k, v in mu.items()},
        "vol": {k: round(math.sqrt(float(cov.loc[k, k])) * 100, 3) for k in r.columns},
        "cov": [[round(float(x) * 10000, 4) for x in row] for row in cov.values],
        "corr": [[round(float(x), 3) for x in row] for row in r.corr().values],
    }


# --------------------------------------------------------------------------- #
# Options (collar / concentrated stock)
# --------------------------------------------------------------------------- #


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(kind: str, s: float, k: float, t: float, r: float, q: float, vol: float) -> float:
    """Black-Scholes with continuous dividend yield. r, q, vol as decimals."""
    if t <= 0 or vol <= 0:
        return max(0.0, (s - k) if kind == "call" else (k - s))
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "call":
        return s * math.exp(-q * t) * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
    return k * math.exp(-r * t) * _ncdf(-d2) - s * math.exp(-q * t) * _ncdf(-d1)


def zero_cost_call_strike(
    s: float, put_k: float, t: float, r: float, q: float, vol: float, skew: float = 0.0
) -> float | None:
    """Call strike that finances a put at put_k (bisection). skew: call vol = vol - skew."""
    target = bs_price("put", s, put_k, t, r, q, vol + skew)
    lo, hi = s * 1.0001, s * 4
    cv = max(0.01, vol - skew)
    if bs_price("call", s, lo, t, r, q, cv) < target:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_price("call", s, mid, t, r, q, cv) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
