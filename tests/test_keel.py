"""Offline tests: the data layer is replaced by synthetic series."""

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import analytics as A
from app import data as D
from app import main as M

# --------------------------------------------------------------------------- #
# analytics
# --------------------------------------------------------------------------- #


def gbm(n=2600, mu=0.05, vol=0.2, s0=100, seed=1, start="2016-01-01"):
    rng = np.random.default_rng(seed)
    r = rng.normal((mu - vol**2 / 2) / 252, vol / math.sqrt(252), n)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(s0 * np.exp(np.cumsum(r)), index=idx)


def test_to_percent():
    assert A.to_percent(pd.Series([0.042, 0.043])).iloc[0] == pytest.approx(4.2)
    assert A.to_percent(pd.Series([4.2, 4.3])).iloc[0] == pytest.approx(4.2)
    # negative JGB era yields stay in percent
    assert A.to_percent(pd.Series([-0.1, 0.3, 0.5, 1.2])).iloc[0] == pytest.approx(-0.1)


def test_hedged_yield():
    # 4.3% UST, 4.0% USD 1Y vs 0.8% JGB 1Y -> hedge 3.2% -> 1.1%
    assert A.hedged_yield(4.3, 4.0, 0.8) == pytest.approx(1.1)
    assert A.hedge_cost(4.0, 0.8, basis_bp=20) == pytest.approx(3.4)


def test_interp_curve():
    c = {1: 1.0, 2: 2.0, 10: 4.0}
    assert A.interp_curve(c, 1.5) == pytest.approx(1.5)
    assert A.interp_curve(c, 0.5) == 1.0
    assert A.interp_curve(c, 30) == 4.0
    assert A.interp_curve({}, 5) is None


def test_realized_vol_close_to_truth():
    s = gbm(vol=0.25, n=5000)
    assert A.realized_vol(s) == pytest.approx(25, abs=1.5)


def test_snapshot_stats_keys():
    st = A.snapshot_stats(gbm(n=400))
    for k in ("last", "chg_1d", "chg_ytd", "vol_20d", "pos_52w", "max_dd_1y"):
        assert k in st
    assert 0 <= st["pos_52w"] <= 100
    assert st["max_dd_1y"] <= 0
    assert st["abs_1d"] is not None and st["abs_ytd"] is not None


def test_barrier_backtest_bounds_and_logic():
    up = pd.Series(np.linspace(100, 300, 1500), index=pd.bdate_range("2018-01-01", periods=1500))
    bt = A.barrier_backtest({"UP": up}, 252, 0.6, 1.0, 63)
    assert bt["autocall_rate"] == 100.0 and bt["ki_rate"] == 0.0

    down = pd.Series(np.linspace(300, 50, 1500), index=up.index)
    bt = A.barrier_backtest({"DN": down}, 252, 0.6, 1.0, 63)
    assert bt["autocall_rate"] == 0.0
    assert bt["loss_rate"] > 0 and bt["loss_rate"] <= bt["ki_rate"] <= 100

    # worst-of basket can never autocall more than its worst member
    a, b = gbm(seed=2), gbm(seed=3, vol=0.35)
    single = A.barrier_backtest({"b": b}, 252, 0.6, 1.0, 63)
    basket = A.barrier_backtest({"a": a, "b": b}, 252, 0.6, 1.0, 63)
    assert basket["autocall_rate"] <= single["autocall_rate"] + 1e-9
    assert basket["ki_rate"] >= single["ki_rate"] - 1e-9


def test_fix_splits():
    idx = pd.bdate_range("2024-01-01", periods=6)
    raw = pd.Series([3000, 3010, 3020, 302, 303, 304], index=idx, dtype=float)  # unadjusted 10:1 split
    fixed = A.fix_splits(raw)
    assert fixed.iloc[0] == pytest.approx(300) and fixed.iloc[-1] == 304
    assert A.realized_vol(fixed) < 20
    crash = pd.Series([100, 100, 55, 56, 57, 58], index=idx, dtype=float)  # real -45% move left alone
    assert A.fix_splits(crash).iloc[0] == 100


def test_monthly_return_stats():
    st = A.monthly_return_stats({"A": gbm(seed=4), "B": gbm(seed=5, vol=0.1)})
    assert st["months"] > 100
    assert st["vol"]["A"] == pytest.approx(20, abs=4)
    assert st["corr"][0][0] == pytest.approx(1)


def test_collar_zero_cost():
    k = A.zero_cost_call_strike(100, 90, 1, 0.005, 0.02, 0.3)
    assert k is not None and 100 < k < 200
    put = A.bs_price("put", 100, 90, 1, 0.005, 0.02, 0.3)
    call = A.bs_price("call", 100, k, 1, 0.005, 0.02, 0.3)
    assert call == pytest.approx(put, rel=1e-4)


def test_implied_vol_roundtrip():
    px = A.bs_price("put", 100, 90, 1, 0.04, 0.01, 0.31)
    assert A.implied_vol_from_price("put", px, 100, 90, 1, 0.04, 0.01) == pytest.approx(0.31, abs=1e-4)
    assert A.implied_vol_from_price("call", 0.0, 100, 90, 1, 0.04, 0.01) is None


def test_bs_put_call_parity():
    s, k, t, r, q, v = 100, 105, 0.75, 0.01, 0.02, 0.25
    c, p = A.bs_price("call", s, k, t, r, q, v), A.bs_price("put", s, k, t, r, q, v)
    assert c - p == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-9)


# --------------------------------------------------------------------------- #
# MOF parser
# --------------------------------------------------------------------------- #

MOF_SAMPLE = """Interest Rate (Japanese Government Bonds),,,,,,,,,,,,,,,
Date,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y,25Y,30Y,40Y
2026/10/1,0.812,0.95,1.02,1.1,1.21,1.3,1.38,1.45,1.52,1.6,2.0,2.4,2.6,2.8,3.0
2026/10/2,0.815,0.951,1.021,1.1,1.22,1.31,1.39,1.46,1.53,1.61,2.01,2.41,-,2.81,3.01
R8.10.5,0.82,0.96,1.03,1.11,1.23,1.32,1.4,1.47,1.54,1.62,2.02,2.42,2.62,2.82,3.02
"""


def test_parse_mof():
    df = D.parse_mof_csv(MOF_SAMPLE)
    assert list(df.columns)[:3] == [1.0, 2.0, 3.0]
    assert df.index[-1] == pd.Timestamp("2026-10-05")
    assert math.isnan(df.loc["2026-10-02", 25.0])
    assert df.loc["2026-10-01", 10.0] == pytest.approx(1.6)


# --------------------------------------------------------------------------- #
# API with fake data
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(monkeypatch):
    def fake_closes(sym, years=3):
        seed = abs(hash(sym)) % 1000
        if sym.endswith("=X"):
            return gbm(n=5900, seed=seed, vol=0.1, s0=150 if "JPY" in sym else 1.1, start="2004-01-01")
        return gbm(seed=seed, s0=5000)

    monkeypatch.setattr(D, "closes", fake_closes)
    monkeypatch.setattr(
        D, "metrics",
        lambda syms: {s: {"name": f"Name {s}", "dividend_yield": 0.02, "pe_ratio": 20.0, "currency": "JPY"} for s in syms},
    )
    idx = pd.bdate_range("2023-10-01", "2026-10-02")
    ust = pd.DataFrame({k: 4.0 + 0.02 * k for k in D.UST_COLS.values()}, index=idx)
    jgb = pd.DataFrame({k: 0.8 + 0.07 * k for k in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]}, index=idx)
    monkeypatch.setattr(D, "ust_history", lambda years=3: ust)
    monkeypatch.setattr(D, "jgb_history", lambda: jgb)
    monkeypatch.setattr(D, "ecb_curve", lambda on=None: {1.0: 2.0, 10.0: 2.6, 30.0: 2.9})
    monkeypatch.setattr(D, "news", lambda s, limit=6, query=None: [{"title": f"{s} headline", "date": "2026-10-05", "url": "https://x", "source": "Y", "symbol": s}])
    monkeypatch.setattr(D, "econ_calendar", lambda days=14: [{"date": "2026-10-10", "country": "US", "event": "CPI"}, {"date": "2026-10-11", "country": "BR", "event": "IPCA"}, {"date": "2026-10-12", "country": "Japan", "event": "PPI"}])
    monkeypatch.setattr(D, "company_events", lambda syms: [{"symbol": syms[0], "earnings_date": "2026-11-05"}])
    monkeypatch.setattr(D, "implied_vol", lambda s, d=365, r=0.04, q=0.0: {"atm_iv": 22.0, "skew_90": 3.0, "expiry": "2027-09-17"})
    cpi_idx = pd.date_range("2005-01-31", "2026-08-31", freq="ME")
    cpi = pd.DataFrame({n: np.linspace(100, 160, len(cpi_idx)) for n, _ in M.PPP_MAP.values()}, index=cpi_idx)
    cpi["japan"] = np.linspace(100, 112, len(cpi_idx))
    cpi["united_states"] = np.linspace(100, 170, len(cpi_idx))
    monkeypatch.setattr(D, "cpi_index", lambda names, start="2000-01-01": cpi)
    return TestClient(M.app)


def test_health(client):
    assert client.get("/health").json()["ok"] is True


def test_yields(client):
    j = client.get("/api/yields").json()
    assert j["errors"] == []
    t10 = next(r for r in j["table"] if r["tenor"] == 10)
    # us10 4.2, us1 4.02, jp1 0.87 -> hedged = 4.2 - 3.15 = 1.05
    assert t10["us_hedged"] == pytest.approx(4.2 - (4.02 - 0.87), abs=1e-3)
    assert j["history"] and "us10_hedged" in j["history"][-1]


def test_country_key():
    assert D.country_key("US") == "US" and D.country_key("Japan") == "JP"
    assert D.country_key("EMU") == "EU" and D.country_key("BR") is None


def test_brief_filters_calendar(client):
    j = client.get("/api/brief", params={"symbols": "USDJPY=X,7974.T"}).json()
    assert len(j["items"]) == 2 and j["items"][0]["spark"]
    assert [e["country"] for e in j["calendar"]] == ["US", "JP"]
    assert j["news"] and j["events"]


def test_sp_screen(client):
    j = client.get("/api/sp/screen", params={"symbols": "^N225,7203.T", "tenor_m": 12, "ki": 60}).json()
    assert len(j["rows"]) == 2 and j["basket"]["windows"] > 0
    assert j["rows"][0]["atm_iv"] == 22.0
    assert len(j["corr"]["matrix"]) == 2


def test_comps_and_returns(client):
    j = client.get("/api/comps", params={"symbols": "7974.T,6758.T"}).json()
    assert j["rows"][0]["name"] == "Name 7974.T" and j["jpy_1y"] == pytest.approx(0.87)
    r = client.get("/api/returns", params={"symbols": "IEF,SPY", "years": 10}).json()
    assert r["months"] > 100 and len(r["cov"]) == 2


def test_history(client):
    j = client.get("/api/history", params={"symbols": "USDJPY=X", "start": "2025-01-01"}).json()
    assert len(j["series"]["USDJPY=X"]) >= 12


def test_ppp_sign(client):
    j = client.get("/api/ppp").json()
    jp = next(r for r in j["rows"] if r["iso"] == "JPY" or r["iso"] == "JPN")
    # Japan's prices rose less than US prices: fair USDJPY falls; gap sign follows fair/spot - 1
    assert jp["fair"] < jp["fx_base"]
    assert jp["gap"] == pytest.approx(jp["fair"] / jp["spot"] - 1, abs=1e-3)


def test_token(monkeypatch, client):
    monkeypatch.setattr(M, "TOKEN", "s3cret")
    assert client.get("/api/yields").status_code == 401
    assert client.get("/api/yields", headers={"x-keel-token": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_bad_input(client):
    assert client.get("/api/snapshot", params={"symbols": " , "}).status_code == 400
