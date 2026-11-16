"""Pitch: turn a view on one asset into the evidence for and against it.

For each asset this module scores up to ten pillars (trend, carry, rates driver, value, positioning, flows,
macro, seasonality, earnings revisions, risk) from the point of view of a LONG position, on a scale from -1
(the evidence argues against being long) to +1 (it supports being long). The page flips the signs for a
short view, weights the pillars, and builds the headline, scenarios and trade ideas.

It also returns the quant pieces that do not depend on direction: a fair-value fit (FX: spot on the rate
spread), historical analogues (past weeks whose signals looked like today's and what followed), and the
statistics the page needs for the fan chart.

Pure pandas/numpy: `build(asset, src)` takes a small data adapter (see main.py) so tests can feed fixtures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

H = (4, 13, 26, 52)  # forward horizons, weeks


@dataclass
class Asset:
    key: str
    name: str
    ja: str
    cls: str                       # fx | rates | equity | commodity | crypto
    sym: str                       # Yahoo symbol, or UST10Y-style yield key
    unit: str = "price"            # price | yield
    base: str | None = None        # FX: base and quote currencies
    quote: str | None = None
    cot: list = field(default_factory=list)   # [(COT key, +1 if spec long = long this asset)]
    areas: list = field(default_factory=list)  # macro areas
    rev: str | None = None         # earnings-revisions group
    flows: str | None = None       # MOF series that supports a long: re | fe | fb
    jpy_cross: bool = False        # long = selling yen (intervention risk applies)
    decimals: int = 2


A = Asset
ASSETS: dict[str, Asset] = {a.key: a for a in [
    A("USDJPY", "USD/JPY", "ドル円", "fx", "JPY=X", base="USD", quote="JPY", cot=[("JPY", -1)], areas=["US", "JP"], flows="re", jpy_cross=True),
    A("EURUSD", "EUR/USD", "ユーロドル", "fx", "EURUSD=X", base="EUR", quote="USD", cot=[("EUR", 1)], areas=["EA", "US"], decimals=4),
    A("EURJPY", "EUR/JPY", "ユーロ円", "fx", "EURJPY=X", base="EUR", quote="JPY", cot=[("EUR", 1), ("JPY", -1)], areas=["EA", "JP"], flows="re", jpy_cross=True),
    A("GBPUSD", "GBP/USD", "ポンドドル", "fx", "GBPUSD=X", base="GBP", quote="USD", cot=[("GBP", 1)], areas=["UK", "US"], decimals=4),
    A("GBPJPY", "GBP/JPY", "ポンド円", "fx", "GBPJPY=X", base="GBP", quote="JPY", cot=[("GBP", 1), ("JPY", -1)], areas=["UK", "JP"], flows="re", jpy_cross=True),
    A("AUDUSD", "AUD/USD", "豪ドル米ドル", "fx", "AUDUSD=X", base="AUD", quote="USD", cot=[("AUD", 1)], areas=["CN", "US"], decimals=4),
    A("AUDJPY", "AUD/JPY", "豪ドル円", "fx", "AUDJPY=X", base="AUD", quote="JPY", cot=[("AUD", 1), ("JPY", -1)], areas=["CN", "JP"], flows="re", jpy_cross=True),
    A("USDCNY", "USD/CNY", "ドル人民元", "fx", "CNY=X", base="USD", quote="CNY", areas=["US", "CN"], decimals=4),
    A("MXNJPY", "MXN/JPY", "メキシコペソ円", "fx", "MXNJPY=X", base="MXN", quote="JPY", cot=[("MXN", 1), ("JPY", -1)], areas=["US", "JP"], flows="re", jpy_cross=True, decimals=3),
    A("UST10", "US 10-year Treasury", "米10年国債", "rates", "UST10Y", unit="yield", cot=[("TY", 1)], areas=["US"]),
    A("UST2", "US 2-year Treasury", "米2年国債", "rates", "UST2Y", unit="yield", cot=[("TU", 1)], areas=["US"]),
    A("JGB10", "JGB 10-year", "日本10年国債", "rates", "JGB10Y", unit="yield", areas=["JP"], flows="fb", decimals=3),
    A("SPX", "S&P 500", "S&P500", "equity", "^GSPC", cot=[("SPX", 1)], areas=["US"], rev="reg_US", decimals=0),
    A("NDX", "Nasdaq 100", "ナスダック100", "equity", "^NDX", cot=[("NDX", 1)], areas=["US"], rev="technology", decimals=0),
    A("N225", "Nikkei 225", "日経平均", "equity", "^N225", areas=["JP"], rev="reg_JP", flows="fe", decimals=0),
    A("SX5E", "Euro Stoxx 50", "ユーロ・ストックス50", "equity", "^STOXX50E", areas=["EA"], rev="reg_EU", decimals=0),
    A("HSI", "Hang Seng", "ハンセン指数", "equity", "^HSI", areas=["CN"], rev="reg_AS", decimals=0),
    A("EEM", "Emerging markets (EEM)", "新興国株（EEM）", "equity", "EEM", areas=["CN", "US"], rev="reg_AS"),
    A("GOLD", "Gold", "金", "commodity", "GC=F", cot=[("GC", 1)], areas=["US"], decimals=0),
    A("WTI", "WTI crude oil", "WTI原油", "commodity", "CL=F", cot=[("CL", 1)], areas=["US", "CN"]),
    A("COPPER", "Copper", "銅", "commodity", "HG=F", cot=[("HG", 1)], areas=["CN", "US"], decimals=3),
    A("BTC", "Bitcoin", "ビットコイン", "crypto", "BTC-USD", cot=[("BTC", 1)], areas=["US"], decimals=0),
]}
RISK_BENCH = "^GSPC"


# --------------------------------------------------------------------------- helpers
def weekly(s: pd.Series) -> pd.Series:
    return s.dropna().resample("W-FRI").last().dropna()


def moves(x: pd.Series, k: int, unit: str) -> pd.Series:
    """Change over k weeks in the position's own terms: % (log) for prices, bp for a long bond (yield down = +)."""
    if unit == "yield":
        return -(x - x.shift(k)) * 100
    return np.log(x / x.shift(k)) * 100


def ann_vol(x: pd.Series, unit: str, weeks: int) -> float | None:
    m = moves(x, 1, unit).dropna().iloc[-weeks:]
    return float(m.std() * math.sqrt(52)) if len(m) > weeks // 2 else None


def pct_rank(s: pd.Series, v: float) -> float:
    s = s.dropna()
    return float((s < v).mean()) if len(s) else 0.5


def clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def th(v, scale):
    return float(math.tanh(v / scale)) if v is not None and scale else 0.0


def pts(s: pd.Series, last: int | None = None, nd: int = 4) -> dict:
    s = s.dropna()
    if last:
        s = s.iloc[-last:]
    return {"d": [str(i.date()) for i in s.index], "v": [round(float(v), nd) for v in s.values]}


def zscore_last(s: pd.Series, window: int | None = None) -> float | None:
    s = s.dropna()
    if window:
        s = s.iloc[-window:]
    if len(s) < 20 or s.std() == 0:
        return None
    return float((s.iloc[-1] - s.mean()) / s.std())


def trend_signals(x: pd.Series, unit: str) -> pd.DataFrame:
    """Five +/-1 trend signals per week: up over 4, 13, 26, 52 weeks, and above the 40-week average."""
    sig = pd.DataFrame(index=x.index)
    for k in (4, 13, 26, 52):
        sig[f"m{k}"] = np.sign(moves(x, k, unit))
    ma = x.rolling(40).mean()
    sig["ma40"] = np.sign(ma - x) if unit == "yield" else np.sign(x - ma)
    return sig


# --------------------------------------------------------------------------- pillars
def p_trend(x, a):
    sig = trend_signals(x, a.unit)
    last = sig.iloc[-1]
    if last.isna().any():
        return None
    score = float(last.mean())
    ma = x.rolling(40).mean()
    flips = {"ma40": float(ma.iloc[-1]), "m13": float(x.iloc[-14]), "m26": float(x.iloc[-27]), "m52": float(x.iloc[-53])}
    n_up = int((last > 0).sum())
    return {"score": score, "metrics": {"signals": {k: int(v) for k, v in last.items()}, "up": n_up, "flips": flips},
            "chart": {"kind": "price_ma", "px": pts(x, 156), "ma": pts(ma, 156)},
            "en": f"{n_up} of 5 trend signals point up (4, 13, 26, 52 weeks and the 40-week average).",
            "ja": f"5つのトレンド・シグナルのうち{n_up}つが上向き（4・13・26・52週、40週平均）。"}


def p_carry(x, a, src, vol52):
    if a.cls == "fx":
        rb, rq = src.rate(a.base), src.rate(a.quote)
        if rb is None or rq is None:
            return None
        diff = rb.iloc[-1] - rq.iloc[-1]
        hist = (rb - rq).dropna()
        ratio = diff / vol52 if vol52 else 0
        return {"score": th(ratio, 0.35),
                "metrics": {"carry_pct": round(float(diff), 3), "carry_vol": round(float(ratio), 3),
                            "r_base": round(float(rb.iloc[-1]), 3), "r_quote": round(float(rq.iloc[-1]), 3),
                            "asof": str(rb.index[-1].date())},
                "chart": {"kind": "line", "unit": "%", "series": [{"name": f"{a.base}−{a.quote} 3M rate", "ja": f"{a.base}−{a.quote} 3か月金利差", **pts(hist.iloc[-120:])}]},
                "en": f"Long {a.base} earns {diff:+.2f}% a year over {a.quote}; carry/vol {ratio:.2f}.",
                "ja": f"{a.base}買いは{a.quote}に対し年{diff:+.2f}%のキャリー。キャリー÷ボラ {ratio:.2f}。"}
    if a.cls == "rates":
        ccy = "JPY" if a.key.startswith("JGB") else "USD"
        r3 = src.rate(ccy)
        curve = src.curve(ccy)
        if r3 is None or curve is None:
            return None
        ten = 10.0 if a.key.endswith("10") else 2.0
        y = curve.get(ten)
        short = 7.0 if ten == 10 else 1.0
        ys = curve.get(short)
        if y is None:
            return None
        carry = (y - float(r3.iloc[-1])) * 100
        roll = ((y - ys) / (ten - short) * 100) if ys is not None else 0.0
        tot = carry + roll
        ratio = tot / vol52 if vol52 else 0
        return {"score": th(ratio, 0.35),
                "metrics": {"carry_bp": round(carry, 1), "roll_bp": round(roll, 1), "total_bp": round(tot, 1),
                            "carry_vol": round(ratio, 3)},
                "chart": None,
                "en": f"Holding the bond earns about {tot:+.0f}bp a year ({carry:+.0f}bp over cash, {roll:+.0f}bp roll-down); carry/vol {ratio:.2f}.",
                "ja": f"保有で年約{tot:+.0f}bp（対短期金利{carry:+.0f}bp、ロールダウン{roll:+.0f}bp）。キャリー÷ボラ {ratio:.2f}。"}
    if a.key in ("GOLD", "BTC"):
        r = src.rate("USD")
        if r is None:
            return None
        diff = -float(r.iloc[-1])
        ratio = diff / vol52 if vol52 else 0
        return {"score": th(ratio, 0.35) * 0.6, "metrics": {"carry_pct": round(diff, 3), "carry_vol": round(ratio, 3)},
                "chart": None,
                "en": f"No yield: holding it costs the dollar cash rate, {diff:+.2f}% a year.",
                "ja": f"利回りなし：保有コストはドル短期金利、年{diff:+.2f}%。"}
    return None


def p_driver(x, a, src):
    """FX: the rate spread and a fair-value fit. Bonds: the front end."""
    if a.cls == "fx":
        if a.key == "USDJPY":
            us, jp = src.yld("UST2Y"), src.yld("JGB2Y")
            if us is None or jp is None:
                return None, None
            spread = weekly(us - jp.reindex(us.index, method="ffill")).dropna()
            label, ja = "US−Japan 2-year yield spread", "日米2年金利差"
        else:
            rb, rq = src.rate(a.base), src.rate(a.quote)
            if rb is None or rq is None:
                return None, None
            m = (rb - rq).dropna()
            spread = m.reindex(m.index.union(x.index)).interpolate().reindex(x.index).dropna()
            label, ja = f"{a.base}−{a.quote} 3M rate spread", f"{a.base}−{a.quote} 3か月金利差"
        j = pd.concat([np.log(x), spread], axis=1, keys=["lx", "s"]).dropna()
        if len(j) < 60:
            return None, None
        d13 = float((j["s"].iloc[-1] - j["s"].iloc[-14]) * 100)
        ch = j.diff().dropna().iloc[-104:]
        corr = float(ch["lx"].corr(ch["s"])) if len(ch) > 30 else 0.0
        score = th(d13, 30) * (0.4 + 0.6 * max(0.0, corr))
        fit = j.iloc[-156:]
        b, c0 = np.polyfit(fit["s"], fit["lx"], 1)
        resid = fit["lx"] - (c0 + b * fit["s"])
        sd = float(resid.std())
        fair = float(math.exp(c0 + b * j["s"].iloc[-1]))
        z = float(resid.iloc[-1] / sd) if sd else 0.0
        r2 = float(1 - resid.var() / fit["lx"].var()) if fit["lx"].var() else 0.0
        fv = {"driver": label, "ja": ja, "fair": fair, "spot": float(x.iloc[-1]), "z": z, "beta_per_100bp": float(b),
              "r2": r2, "spread": float(j["s"].iloc[-1]), "weeks": len(fit),
              "spot_series": pts(np.exp(fit["lx"]), nd=a.decimals + 2),
              "fair_series": pts(np.exp(c0 + b * fit["s"]), nd=a.decimals + 2),
              "spread_series": pts(fit["s"])}
        return ({"score": score, "metrics": {"spread": round(float(j['s'].iloc[-1]), 3), "chg13_bp": round(d13, 1), "corr2y": round(corr, 2)},
                 "chart": {"kind": "dual", "px": pts(np.exp(j["lx"].iloc[-156:]), nd=a.decimals + 2), "drv": pts(j["s"].iloc[-156:]), "drv_name": label, "drv_ja": ja},
                 "en": f"{label} {'widened' if d13 > 0 else 'narrowed'} {abs(d13):.0f}bp in 13 weeks; weekly correlation with spot {corr:.2f}.",
                 "ja": f"{ja}は13週で{abs(d13):.0f}bp{'拡大' if d13 > 0 else '縮小'}。スポットとの週次相関{corr:.2f}。"}, fv)
    if a.cls == "rates":
        front = src.yld("JGB2Y" if a.key.startswith("JGB") else "UST2Y")
        if front is None:
            return None, None
        f = weekly(front)
        if a.key == "UST2":
            r = src.rate("USD")
            if r is None:
                return None, None
            f = r
            d = float((f.iloc[-1] - f.iloc[-4]) * 100)  # monthly series: 3 months
            name, ja = "US 3-month rate", "米3か月金利"
        else:
            d = float((f.iloc[-1] - f.iloc[-14]) * 100)
            name, ja = ("JGB 2-year yield", "日本2年金利") if a.key.startswith("JGB") else ("US 2-year yield", "米2年金利")
        return ({"score": -th(d, 30), "metrics": {"chg_bp": round(d, 1)},
                 "chart": {"kind": "line", "unit": "%", "series": [{"name": name, "ja": ja, **pts(f, 156)}]},
                 "en": f"The front end leads: {name} {'up' if d > 0 else 'down'} {abs(d):.0f}bp over three months.",
                 "ja": f"短期金利が先行：{ja}は3か月で{abs(d):.0f}bp{'上昇' if d > 0 else '低下'}。"}, None)
    return None, None


def p_value(x, a, src):
    if a.cls == "fx":
        gb, gq = src.ppp_gap(a.base), src.ppp_gap(a.quote)
        if gb is not None and gq is not None:
            # gap > 0: that currency is rich against the dollar. Long base/quote is cheap when base is cheap vs quote.
            rel = (1 + gq) / (1 + gb) - 1
            return {"score": th(rel, 0.25),
                    "metrics": {"ppp_gap_base": round(gb, 4), "ppp_gap_quote": round(gq, 4), "rel": round(rel, 4),
                                "fair": float(x.iloc[-1] * (1 + gb) / (1 + gq)) if (1 + gq) else None},
                    "chart": None,
                    "en": (f"On relative PPP {a.quote if a.base == 'USD' else a.base} is {abs(gq if a.base == 'USD' else gb) * 100:.0f}% "
                           f"{'rich' if (gq if a.base == 'USD' else gb) > 0 else 'cheap'} against the dollar"
                           + ("" if "USD" in (a.base, a.quote) else f" and {a.quote} {abs(gq) * 100:.0f}% {'rich' if gq > 0 else 'cheap'}")
                           + f", so the pair is {abs(rel) * 100:.0f}% {'cheap' if rel > 0 else 'rich'}."),
                    "ja": (f"相対PPPでは{a.quote if a.base == 'USD' else a.base}は対ドル{abs(gq if a.base == 'USD' else gb) * 100:.0f}%"
                           f"{'割高' if (gq if a.base == 'USD' else gb) > 0 else '割安'}"
                           + ("" if "USD" in (a.base, a.quote) else f"、{a.quote}は{abs(gq) * 100:.0f}%{'割高' if gq > 0 else '割安'}")
                           + f"。この通貨ペアは{abs(rel) * 100:.0f}%{'割安' if rel > 0 else '割高'}。")}
    # Stretch against the 5-year average (prices: log gap; yields: bp gap), z-scored on its own history.
    if a.unit == "yield":
        gap = (x - x.rolling(260, min_periods=150).mean()) * 100
        z = zscore_last(gap)
        if z is None:
            return None
        score = th(z, 1.5) * 0.8  # high yields: bonds cheap
        word = "above" if gap.iloc[-1] > 0 else "below"
        return {"score": score, "metrics": {"gap_bp": round(float(gap.iloc[-1]), 1), "z": round(z, 2)},
                "chart": {"kind": "line", "unit": "bp", "series": [{"name": "Yield minus 5-year average", "ja": "利回り−5年平均", **pts(gap, 520)}]},
                "en": f"The yield is {abs(gap.iloc[-1]):.0f}bp {word} its 5-year average ({z:+.1f} sd).",
                "ja": f"利回りは5年平均より{abs(gap.iloc[-1]):.0f}bp{'高い' if gap.iloc[-1] > 0 else '低い'}（{z:+.1f}sd）。"}
    gap = np.log(x / x.rolling(260, min_periods=150).mean()) * 100
    z = zscore_last(gap)
    if z is None:
        return None
    return {"score": -th(z, 1.5) * 0.8, "metrics": {"gap_pct": round(float(gap.iloc[-1]), 1), "z": round(z, 2)},
            "chart": {"kind": "line", "unit": "%", "series": [{"name": "Price vs 5-year average", "ja": "価格の5年平均比", **pts(gap, 520)}]},
            "en": f"Price is {abs(gap.iloc[-1]):.0f}% {'above' if gap.iloc[-1] > 0 else 'below'} its 5-year average ({z:+.1f} sd against its own history).",
            "ja": f"価格は5年平均より{abs(gap.iloc[-1]):.0f}%{'上' if gap.iloc[-1] > 0 else '下'}（自らの過去比{z:+.1f}sd）。"}


def p_positioning(a, src):
    if not a.cot:
        return None
    parts = []
    for key, sign in a.cot:
        rows = src.cot(key)
        if not rows:
            continue
        s = pd.Series({pd.Timestamp(r["d"]): r["pct"] * sign for r in rows if r.get("pct") is not None}).sort_index()
        if len(s) > 60:
            parts.append(s)
    if not parts:
        return None
    s = pd.concat(parts, axis=1).mean(axis=1).dropna()
    win = s.iloc[-156:]
    p = pct_rank(win, float(s.iloc[-1]))
    if p > 0.85:
        score = -(p - 0.85) / 0.15
    elif p < 0.15:
        score = (0.15 - p) / 0.15
    else:
        score = 0.25 * th(float(s.iloc[-1] - s.iloc[-14]), 10)
    crowd = "crowded long" if p > 0.85 else "crowded short" if p < 0.15 else "not stretched"
    crowd_ja = "買いが過密" if p > 0.85 else "売りが過密" if p < 0.15 else "偏りなし"
    return {"score": clip(score), "metrics": {"net_pct_oi": round(float(s.iloc[-1]), 1), "pctile_3y": round(p, 3), "asof": str(s.index[-1].date())},
            "chart": {"kind": "line", "unit": "% OI", "series": [{"name": "Speculators net, in the direction of a long", "ja": "投機筋ネット（買い方向）", **pts(s, 260)}]},
            "en": f"Speculators are {s.iloc[-1]:+.0f}% of open interest in the direction of a long, {p * 100:.0f}th percentile of 3 years: {crowd}.",
            "ja": f"投機筋は買い方向に建玉の{s.iloc[-1]:+.0f}%、3年の{p * 100:.0f}パーセンタイル：{crowd_ja}。"}


def p_flows(a, src):
    if not a.flows:
        return None
    rows = src.mof()
    if not rows:
        return None
    s = pd.Series({pd.Timestamp(r["d"]): r.get(a.flows) for r in rows}).dropna().sort_index()
    if len(s) < 80:
        return None
    s13 = s.rolling(13).sum()
    z = zscore_last(s13, 156)
    if z is None:
        return None
    what = {"re": ("Japanese investors' net buying of foreign equities (yen selling)", "国内投資家の対外株式投資（円売り）"),
            "fe": ("Foreign investors' net buying of Japanese equities", "海外投資家の日本株買い越し"),
            "fb": ("Foreign investors' net buying of Japanese bonds", "海外投資家の日本債券買い越し")}[a.flows]
    return {"score": th(z, 1.5), "metrics": {"sum13_oku": round(float(s13.iloc[-1]), 0), "z3y": round(z, 2), "asof": str(s.index[-1].date())},
            "chart": {"kind": "bars", "unit": "¥100m", "series": [{"name": what[0], "ja": what[1], **pts(s, 104, 0)}]},
            "en": f"{what[0]}: ¥{s13.iloc[-1] / 10000:+.1f}tn over 13 weeks, {z:+.1f} sd against 3 years.",
            "ja": f"{what[1]}：13週で{s13.iloc[-1] / 10000:+.1f}兆円、3年比{z:+.1f}sd。"}


def _macro_mom(src, area):
    cli, cpi = src.cli(area), src.cpi(area)
    dc = float(cli.iloc[-1] - cli.iloc[-4]) if cli is not None and len(cli) > 4 else None
    di = float(cpi.iloc[-1] - cpi.iloc[-7]) if cpi is not None and len(cpi) > 7 else None
    return dc, di, cli, cpi


def p_macro(x, a, src):
    if not a.areas:
        return None
    m = {ar: _macro_mom(src, ar) for ar in a.areas}
    if a.cls == "fx":
        b, q = m.get(a.areas[0]), m.get(a.areas[1])
        if not b or not q or b[0] is None or q[0] is None:
            return None
        rel = b[0] - q[0]
        score = th(rel, 0.5)
        en = f"Leading indicator over 3 months: {a.areas[0]} {b[0]:+.2f}, {a.areas[1]} {q[0]:+.2f}; growth momentum favours {a.areas[0] if rel > 0 else a.areas[1]}."
        ja = f"景気先行指数の3か月変化：{a.areas[0]} {b[0]:+.2f}、{a.areas[1]} {q[0]:+.2f}。成長モメンタムは{a.areas[0] if rel > 0 else a.areas[1]}優位。"
    else:
        dcs = [v[0] for v in m.values() if v[0] is not None]
        dis = [v[1] for v in m.values() if v[1] is not None]
        if not dcs:
            return None
        dc, di = float(np.mean(dcs)), float(np.mean(dis)) if dis else 0.0
        if a.cls == "rates":
            score = -0.5 * th(dc, 0.4) - 0.5 * th(di, 0.5)
        elif a.key == "GOLD":
            score = 0.3 * th(-dc, 0.6) + 0.4 * th(di, 0.5)
        else:
            score = th(dc, 0.4)
        en = f"Leading indicator {dc:+.2f} over 3 months and inflation {di:+.1f}pt over 6 months ({', '.join(a.areas)})."
        ja = f"景気先行指数は3か月で{dc:+.2f}、インフレ率は6か月で{di:+.1f}pt（{', '.join(a.areas)}）。"
    # Regime table: the asset's average monthly move in each growth x inflation quadrant, macro lagged 2 months.
    area = a.areas[0]
    _, _, cli, cpi = m[area]
    reg = None
    if cli is not None and cpi is not None:
        mx = x.resample("ME").last()
        mv = moves(mx, 1, a.unit)
        g = (cli - cli.shift(3)).shift(2)
        i = (cpi - cpi.shift(6)).shift(2)
        j = pd.concat([mv, g.reindex(mv.index, method="ffill"), i.reindex(mv.index, method="ffill")], axis=1, keys=["r", "g", "i"]).dropna()
        if len(j) > 60:
            q = np.where(j["g"] > 0, np.where(j["i"] > 0, "reflation", "goldilocks"), np.where(j["i"] > 0, "stagflation", "deflation"))
            j["q"] = q
            reg = {k: {"mean_ann": round(float(v["r"].mean() * 12), 2), "n": int(len(v))} for k, v in j.groupby("q")}
            gn, inn = (cli.iloc[-1] - cli.iloc[-4]) > 0, (cpi.iloc[-1] - cpi.iloc[-7]) > 0 if len(cpi) > 7 else False
            reg["now"] = ("reflation" if inn else "goldilocks") if gn else ("stagflation" if inn else "deflation")
            reg["area"] = area
    return {"score": clip(score), "metrics": {"regimes": reg},
            "chart": {"kind": "line", "unit": "", "series": [{"name": f"Leading indicator {ar}", "ja": f"景気先行指数 {ar}", **pts(v[2], 60)} for ar, v in m.items() if v[2] is not None]},
            "en": en, "ja": ja}


def p_seasonality(x, a):
    mx = x.resample("ME").last()
    mv = moves(mx, 1, a.unit).dropna()
    if len(mv) < 60:
        return None
    by = mv.groupby(mv.index.month)
    stats = {int(k): {"mean": round(float(v.mean()), 3), "hit": round(float((v > 0).mean()), 3), "n": int(len(v))} for k, v in by}
    nxt = [(x.index[-1].month % 12) + i for i in range(1, 4)]
    nxt = [((n - 1) % 12) + 1 for n in nxt]
    s3 = sum(stats[n]["mean"] for n in nxt if n in stats)
    sd = float(mv.std() * math.sqrt(3))
    score = th(s3 / sd if sd else 0, 0.5) * 0.7
    u = "bp" if a.unit == "yield" else "%"
    return {"score": score, "metrics": {"months": stats, "next3": nxt, "next3_mean": round(s3, 2)},
            "chart": {"kind": "months", "unit": u},
            "en": f"The next three calendar months have averaged {s3:+.1f}{u} for a long since {mx.index[0].year}.",
            "ja": f"今後3か月（暦月）は{mx.index[0].year}年以降、買いで平均{s3:+.1f}{u}。"}


def p_revisions(a, src):
    if not a.rev:
        return None
    g = src.revisions(a.rev)
    if not g:
        return None
    net = g["up_share"] - g["dn_share"]
    score = 0.5 * th(net, 0.3) + 0.5 * th(g["rev2m"], 0.02)
    return {"score": clip(score), "metrics": {k: g.get(k) for k in ("n", "rev2m", "rev1m", "up_share", "dn_share", "net_analysts", "direction", "weeks_since_turn")},
            "chart": None,
            "en": f"Earnings estimates ({g.get('name')}): {g['up_share'] * 100:.0f}% of companies revised up, {g['dn_share'] * 100:.0f}% down over two months; median {g['rev2m'] * 100:+.1f}%.",
            "ja": f"業績予想（{g.get('ja')}）：2か月で上方修正{g['up_share'] * 100:.0f}%、下方{g['dn_share'] * 100:.0f}%、中央値{g['rev2m'] * 100:+.1f}%。"}


def p_risk(x, a, src):
    v13 = moves(x, 1, a.unit).rolling(13).std()
    if v13.dropna().empty:
        return None
    pv = pct_rank(v13.iloc[-520:], float(v13.iloc[-1]))
    bench = src.px(RISK_BENCH)
    beta = 0.0
    if bench is not None and a.sym != RISK_BENCH:
        b = moves(weekly(bench), 1, "price")
        r = moves(x, 1, a.unit)
        j = pd.concat([r, b], axis=1).dropna().iloc[-156:]
        if len(j) > 50 and j.iloc[:, 1].var():
            beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    elif a.sym == RISK_BENCH:
        beta = 1.0
    rb = clip(beta)
    score = -(pv - 0.5) * 2 * rb * 0.6
    notes_en, notes_ja = [f"realised volatility at the {pv * 100:.0f}th percentile of 10 years", ], [f"実現ボラは10年の{pv * 100:.0f}パーセンタイル"]
    interv = None
    if a.jpy_cross:
        usdjpy = src.px("JPY=X")
        if usdjpy is not None:
            u = weekly(usdjpy)
            lvl = float(u.iloc[-1])
            m4 = float(np.log(u.iloc[-1] / u.iloc[-5]))
            sd4 = float(np.log(u / u.shift(1)).iloc[-156:].std() * 2)
            sev = 0.5 * clip((lvl - 150) / 10, 0, 1) + 0.5 * clip((m4 / sd4) / 2.5 if sd4 else 0, 0, 1)
            interv = {"usdjpy": lvl, "move4w_sd": round(m4 / sd4, 2) if sd4 else None, "severity": round(sev, 2)}
            score -= sev
            notes_en.append(f"intervention risk {sev * 100:.0f}/100 (USD/JPY {lvl:.1f})")
            notes_ja.append(f"介入リスク{sev * 100:.0f}/100（ドル円{lvl:.1f}）")
    return {"score": clip(score), "metrics": {"vol_pctile": round(pv, 3), "beta_spx": round(beta, 2), "intervention": interv},
            "chart": {"kind": "line", "unit": "% a.y." if a.unit == "price" else "bp a.y.", "series": [{"name": "13-week realised volatility", "ja": "13週実現ボラ", **pts(v13 * math.sqrt(52), 520)}]},
            "en": "Risk: " + "; ".join(notes_en) + f"; beta to the S&P 500 {beta:+.2f}.",
            "ja": "リスク：" + "、".join(notes_ja) + f"。S&P500へのベータ{beta:+.2f}。"}


PILLAR_META = {
    "trend": ("Trend", "トレンド"), "carry": ("Carry", "キャリー"), "driver": ("Rates", "金利"),
    "value": ("Value", "バリュー"), "positioning": ("Positioning", "ポジション"), "flows": ("Flows", "フロー"),
    "macro": ("Macro", "マクロ"), "seasonality": ("Seasonality", "季節性"), "revisions": ("Earnings revisions", "業績修正"),
    "risk": ("Risks", "リスク"),
}
DEFAULT_W = {"trend": 1.5, "carry": 1.2, "driver": 1.2, "value": 0.8, "positioning": 0.8, "flows": 0.8,
             "macro": 1.0, "seasonality": 0.4, "revisions": 1.2, "risk": 1.0}


# --------------------------------------------------------------------------- analogues
def analogues(x: pd.Series, a: Asset, k: int = 40) -> dict | None:
    """Past weeks whose trend, momentum, stretch and volatility looked like this week, and what followed."""
    unit = a.unit
    feat = pd.DataFrame(index=x.index)
    feat["trend"] = trend_signals(x, unit).mean(axis=1)
    m1 = moves(x, 1, unit)
    sd = m1.rolling(156, min_periods=52).std()
    feat["mom13"] = moves(x, 13, unit) / (sd * math.sqrt(13))
    if unit == "yield":
        feat["stretch"] = (x - x.rolling(260, min_periods=150).mean()) * -100 / (sd * math.sqrt(52))
    else:
        feat["stretch"] = np.log(x / x.rolling(260, min_periods=150).mean()) * 100 / (sd * math.sqrt(52))
    v13 = m1.rolling(13).std()
    feat["vol"] = v13 / sd
    feat = feat.dropna()
    if len(feat) < 200:
        return None
    z = (feat - feat.mean()) / feat.std()
    now = z.iloc[-1]
    fwd = {h: moves(x, h, unit).shift(-h) for h in H}
    cand = z.iloc[:-max(H)]
    d = ((cand - now) ** 2).sum(axis=1).sort_values()
    picked: list[pd.Timestamp] = []
    for t in d.index:
        if all(abs((t - p).days) > 28 for p in picked):
            picked.append(t)
        if len(picked) >= k:
            break
    out = {"n": len(picked), "dates": [str(t.date()) for t in picked[:12]], "dist": [round(float(d[t]), 3) for t in picked[:12]],
           "features_now": {c: round(float(feat[c].iloc[-1]), 3) for c in feat.columns}, "fwd": {}, "all": {}}
    for h in H:
        f = fwd[h]
        vals = [float(f[t]) for t in picked if pd.notna(f.get(t))]
        out["fwd"][h] = [round(v, 3) for v in vals]
        allv = f.dropna()
        out["all"][h] = {"mean": round(float(allv.mean()), 3), "hit": round(float((allv > 0).mean()), 3),
                         "q": [round(float(q), 3) for q in allv.quantile([0.1, 0.25, 0.5, 0.75, 0.9])]}
    # The analogues' paths over 26 weeks, for a spaghetti chart.
    paths = []
    for t in picked[:20]:
        i = x.index.get_loc(t)
        seg = x.iloc[i:i + 27]
        if len(seg) == 27:
            paths.append([round(float(-(seg.iloc[j] - seg.iloc[0]) * 100 if unit == "yield"
                                        else math.log(seg.iloc[j] / seg.iloc[0]) * 100), 3) for j in range(27)])
    out["paths"] = paths
    return out


# --------------------------------------------------------------------------- build
def build(key: str, src) -> dict:
    a = ASSETS[key]
    raw = src.px(a.sym) if a.unit == "price" else src.yld(a.sym)
    if raw is None or len(raw) < 300:
        raise ValueError(f"no history for {a.name}")
    x = weekly(raw)
    if a.unit == "price":
        x = x[x > 0]
    last = float(raw.dropna().iloc[-1])
    asof = str(raw.dropna().index[-1].date())
    vol52, vol13 = ann_vol(x, a.unit, 52), ann_vol(x, a.unit, 13)
    vol5y = ann_vol(x, a.unit, 260)
    yr = x[x.index.year == x.index[-1].year]
    prev_end = x[x.index.year < x.index[-1].year]
    stats = {"spot": last, "asof": asof, "vol52": vol52, "vol13": vol13, "vol5y": vol5y,
             "chg": {lab: (float(moves(x, k, a.unit).iloc[-1]) if len(x) > k else None) for lab, k in (("1w", 1), ("1m", 4), ("3m", 13), ("1y", 52))},
             "ytd": float(moves(pd.concat([prev_end.iloc[-1:], yr.iloc[-1:]]), 1, a.unit).iloc[-1]) if len(prev_end) and len(yr) else None,
             "hi52": float(x.iloc[-52:].max()), "lo52": float(x.iloc[-52:].min())}
    pillars: dict = {}
    pillars["trend"] = p_trend(x, a)
    pillars["carry"] = p_carry(x, a, src, vol52)
    drv, fv = p_driver(x, a, src)
    pillars["driver"] = drv
    pillars["value"] = p_value(x, a, src)
    pillars["positioning"] = p_positioning(a, src)
    pillars["flows"] = p_flows(a, src)
    pillars["macro"] = p_macro(x, a, src)
    pillars["seasonality"] = p_seasonality(x, a)
    pillars["revisions"] = p_revisions(a, src)
    pillars["risk"] = p_risk(x, a, src)
    out_p = []
    for k, p in pillars.items():
        if p is None:
            continue
        en, ja = PILLAR_META[k]
        out_p.append({"key": k, "name": en, "ja": ja, "w": DEFAULT_W[k], "score": round(float(p["score"]), 3),
                      "text": p["en"], "text_ja": p["ja"], "metrics": p.get("metrics"), "chart": p.get("chart")})
    ccy = {"N225": "JPY", "JGB10": "JPY", "SX5E": "EUR"}.get(key, "USD")
    cash = {}
    for cc in ([a.base, a.quote] if a.cls == "fx" else [ccy]):
        r = src.rate(cc)
        if r is not None:
            cash[cc] = round(float(r.iloc[-1]), 3)
    carry_ann = None
    c = pillars.get("carry")
    if c:
        m = c["metrics"]
        carry_ann = m.get("carry_pct", m.get("total_bp"))
    return {
        "key": key, "name": a.name, "ja": a.ja, "cls": a.cls, "unit": a.unit, "decimals": a.decimals,
        "base": a.base, "quote": a.quote, "sym": a.sym,
        "stats": stats, "carry_ann": carry_ann, "ccy": a.quote if a.cls == "fx" else ccy, "cash": cash,
        "px": pts(x, nd=a.decimals + 3),
        "pillars": out_p, "fair_value": fv, "analogues": analogues(x, a),
        "missing": [k for k, p in pillars.items() if p is None],
    }


def universe() -> list[dict]:
    return [{"key": a.key, "name": a.name, "ja": a.ja, "cls": a.cls, "unit": a.unit} for a in ASSETS.values()]
