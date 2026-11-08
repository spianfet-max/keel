"""Option premium engine: short 1-month straddles priced off volatility indices, with
different hedging rules, volatility regimes and seasonality.

Pure numpy/pandas, no network: main.py feeds it daily prices and implied vols.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TENOR = 21           # trading days in a 1-month option
STAGGER = 5          # sell a new straddle every week, 1/4 notional each (four live at once)
DAYS = 252
SQ2PI = math.sqrt(2 * math.pi)


def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQ2PI


def straddle(model: str, s: float, k: float, vol: float, tau: float) -> tuple[float, float, float]:
    """Price, delta and vega of a straddle struck at k (r = q = 0).
    model 'ln': Black-Scholes on a price; 'n': Bachelier on a level (vol in level units a year)."""
    if tau <= 0:
        return abs(s - k), (1.0 if s > k else -1.0 if s < k else 0.0), 0.0
    sd = max(vol * math.sqrt(tau), 1e-12)
    if model == "n":
        d = (s - k) / sd
        return (s - k) * (2 * _N(d) - 1) + 2 * sd * _n(d), 2 * _N(d) - 1, 2 * math.sqrt(tau) * _n(d)
    d1 = (math.log(s / k) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return s * (2 * _N(d1) - 1) - k * (2 * _N(d2) - 1), 2 * _N(d1) - 1, 2 * s * math.sqrt(tau) * _n(d1)


def option_book(s: np.ndarray, iv: np.ndarray, model: str = "ln", opt_cost: float = 0.003):
    """Sell a 1-month straddle every week, a quarter of the notional each, held to expiry and marked at
    each day's implied vol. Returns the daily option P&L (entry costs included) and the book's delta at
    each close, both per unit of notional ('ln': each straddle on 1/K units, so P&L is a fraction of notional;
    'n': level units). `iv`: annual vol (decimal for 'ln', level units for 'n');
    opt_cost: half bid/ask in vol units charged on vega at entry."""
    n = len(s)
    pnl, delta, prem = np.zeros(n), np.zeros(n), np.zeros(n)
    book: list[list[float]] = []      # [strike, expiry index, last price, quantity]
    for t in range(n):
        if t > 0:
            for o in book:
                p = straddle(model, s[t], o[0], iv[t], (o[1] - t) / DAYS)[0]
                pnl[t] -= o[3] * (p - o[2])
                o[2] = p
            book = [o for o in book if o[1] > t]
        if t % STAGGER == 0 and iv[t] > 0:
            p, _, vg = straddle(model, s[t], s[t], iv[t], TENOR / DAYS)
            q = 0.25 / (s[t] if model == "ln" else 1.0)
            book.append([s[t], t + TENOR, p, q])
            pnl[t] -= q * vg * opt_cost
            prem[t] = q * p
        delta[t] = sum(-o[3] * straddle(model, s[t], o[0], iv[t], (o[1] - t) / DAYS)[1] for o in book)
    return pnl, delta, prem


def hedge_pnl(s: np.ndarray, iv: np.ndarray, delta: np.ndarray, model: str, hedge: str,
              hedge_cost: float = 1e-4, budget_days: float = 10.0):
    """P&L of the delta hedge under a rule: 'none', every N days ('1', '5', '10'), or 'budget' (re-hedge once
    the squared moves since the last hedge add up to `budget_days` days of implied variance: rarely when
    markets are calm, often when they are stressed). Returns daily P&L and the days a hedge was done."""
    n = len(s)
    out, done = np.zeros(n), np.zeros(n, dtype=bool)
    if hedge == "none":
        return out, done
    every = None if hedge == "budget" else int(hedge)
    h, last, acc = 0.0, 0, 0.0
    for t in range(n):
        if t > 0:
            ds = s[t] - s[t - 1]
            out[t] += h * ds
            acc += (math.log(s[t] / s[t - 1]) ** 2) if model == "ln" else ds * ds
        unit_var = iv[t] ** 2 / DAYS
        due = t == 0 or (every is not None and t - last >= every) or \
            (every is None and acc >= budget_days * unit_var)
        if due:
            new_h = -delta[t]
            out[t] -= abs(new_h - h) * (s[t] if model == "ln" else 1.0) * hedge_cost
            h, last, acc = new_h, t, 0.0
            done[t] = True
    return out, done


def budget_state(s: np.ndarray, iv: np.ndarray, hedged_on: np.ndarray, model: str, budget_days: float) -> dict:
    """How much of the variance budget has been used since the last hedge, as of the last day."""
    idx = np.where(hedged_on)[0]
    last = int(idx[-1]) if len(idx) else 0
    if model == "ln":
        r = np.diff(np.log(s[last:]))
    else:
        r = np.diff(s[last:])
    used = float(np.sum(r ** 2))
    unit = float(iv[-1] ** 2 / DAYS)
    return {"days_since_hedge": int(len(s) - 1 - last), "budget_used": used / (budget_days * unit) if unit > 0 else None}


# --------------------------------------------------------------------------- #
# Hidden Markov model for volatility regimes (Gaussian on log vol)
# --------------------------------------------------------------------------- #


def hmm_fit(x: np.ndarray, k: int = 4, iters: int = 60, seed: int = 0) -> dict:
    """Baum-Welch for a k-state Gaussian HMM on a 1-D series; states sorted by mean."""
    x = np.asarray(x, dtype=float)
    qs = np.quantile(x, np.linspace(0.1, 0.9, k))
    mu, sig = qs.copy(), np.full(k, x.std() / k + 1e-6)
    A = np.full((k, k), 0.02 / (k - 1))
    np.fill_diagonal(A, 0.98)
    pi = np.full(k, 1 / k)
    for _ in range(iters):
        B = np.exp(-0.5 * ((x[:, None] - mu) / sig) ** 2) / (sig * SQ2PI) + 1e-300
        T = len(x)
        al, c = np.zeros((T, k)), np.zeros(T)
        al[0] = pi * B[0]
        c[0] = al[0].sum()
        al[0] /= c[0]
        for t in range(1, T):
            al[t] = (al[t - 1] @ A) * B[t]
            c[t] = al[t].sum()
            al[t] /= c[t]
        be = np.ones((T, k))
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        g /= g.sum(1, keepdims=True)
        xi = np.einsum("ti,ij,tj->ij", al[:-1], A, (B[1:] * be[1:]) / c[1:, None])
        A = xi / xi.sum(1, keepdims=True)
        pi = g[0]
        w = g.sum(0) + 1e-9
        mu = (g * x[:, None]).sum(0) / w
        sig = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / w) + 1e-4
    o = np.argsort(mu)
    return {"mu": mu[o], "sig": sig[o], "A": A[np.ix_(o, o)], "pi": pi[o]}


def hmm_filter(x: np.ndarray, m: dict) -> np.ndarray:
    """Filtered state probabilities (each row uses data up to that date only)."""
    mu, sig, A = m["mu"], m["sig"], m["A"]
    k = len(mu)
    out = np.zeros((len(x), k))
    p = m["pi"].copy()
    for t, v in enumerate(x):
        b = np.exp(-0.5 * ((v - mu) / sig) ** 2) / (sig * SQ2PI) + 1e-300
        p = (p @ A if t else p) * b
        p /= p.sum()
        out[t] = p
    return out


# --------------------------------------------------------------------------- #
# Volatility measures
# --------------------------------------------------------------------------- #


def realised(s: pd.Series, model: str, window: int = TENOR) -> pd.Series:
    """Annualised realised vol from daily moves (decimal for 'ln', level units for 'n')."""
    r = np.log(s).diff() if model == "ln" else s.diff()
    return r.rolling(window).std() * math.sqrt(DAYS)


def horizon_vol(s: pd.Series, model: str, months: int = 24) -> pd.Series:
    """Vol measured at the option's own horizon: monthly moves over the last `months` months, adjusted for
    the volatility of volatility (a vol that swings around is riskier than its average suggests).
    Daily index; recomputed at each month end and carried forward."""
    me = s.resample("ME").last().dropna()
    r = np.log(me).diff() if model == "ln" else me.diff()
    base = np.sqrt((r ** 2).rolling(months, min_periods=months // 2).mean() * 12)
    dv = realised(s, model).resample("ME").last()
    vov = np.log(dv).diff().rolling(months, min_periods=months // 2).std()
    adj = base * np.sqrt(1 + vov.reindex(base.index) ** 2)
    return adj.reindex(s.index, method="ffill")
