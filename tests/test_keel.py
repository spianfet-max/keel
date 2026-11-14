"""Offline tests: the data layer is replaced by synthetic series."""

import math
import os

os.environ["KEEL_WARM"] = "0"   # no background warm-up thread during tests

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


def test_tenor_windows():
    up = pd.Series(np.linspace(100, 200, 600), index=pd.bdate_range("2020-01-01", periods=600))
    w = A.tenor_windows(up, 252, 100)
    assert 90 <= len(w) <= 400 and all(f > 1 and m == 1.0 for f, m in w)
    assert A.tenor_windows(up.iloc[:100], 252) == []


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


OECD_SAMPLE = """DATAFLOW,REF_AREA,FREQ,METHODOLOGY,MEASURE,UNIT_MEASURE,EXPENDITURE,ADJUSTMENT,TRANSFORMATION,TIME_PERIOD,OBS_VALUE
OECD.SDD.TPS:DSD_PRICES@DF_PRICES_ALL(1.0),JPN,M,N,CPI,IX,_T,N,_Z,2026-07,111.2
OECD.SDD.TPS:DSD_PRICES@DF_PRICES_ALL(1.0),JPN,M,N,CPI,IX,_T,N,_Z,2026-08,111.5
OECD.SDD.TPS:DSD_PRICES@DF_PRICES_ALL(1.0),AUS,Q,N,CPI,IX,_T,N,_Z,2026-Q2,140.0
OECD.SDD.TPS:DSD_PRICES@DF_PRICES_ALL(1.0),EA20,M,N,CPI,IX,_T,N,_Z,2026-08,
"""


def test_parse_oecd_cpi():
    df = D.parse_oecd_cpi_csv(OECD_SAMPLE)
    assert df.loc["2026-08-31", "JPN"] == pytest.approx(111.5)
    assert df.loc["2026-06-30", "AUS"] == pytest.approx(140.0)
    assert "EA20" not in df


def test_splice():
    o = pd.Series([100.0, 101, 102], index=pd.date_range("2021-04-30", periods=3, freq="ME"))
    n = pd.Series([50.0, 51, 52, 53], index=pd.date_range("2021-06-30", periods=4, freq="ME"))
    sp = D.splice(o, n)
    assert sp.index[-1] == pd.Timestamp("2021-09-30") and sp.iloc[-1] == pytest.approx(102 * 53 / 50)
    assert len(sp) == 6


def test_extend_with_yoy():
    idx = pd.date_range("2020-01-31", "2021-06-30", freq="ME")
    level = pd.Series(np.linspace(100, 101.7, len(idx)), index=idx)
    yoy = pd.Series(3.0, index=pd.date_range("2021-07-31", "2022-12-31", freq="ME"))
    ext = A_ext = D.extend_with_yoy(level, yoy)
    assert ext.index[-1] == pd.Timestamp("2022-12-31")
    assert ext.loc["2022-06-30"] == pytest.approx(level.loc["2021-06-30"] * 1.03)
    assert ext.loc["2022-12-31"] == pytest.approx(ext.loc["2021-12-31"] * 1.03)


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
    def fake_closes(sym, years=3, dividends=False):
        seed = abs(hash(sym)) % 1000
        if sym.endswith("=X"):
            return gbm(n=5900, seed=seed, vol=0.1, s0=150 if "JPY" in sym else 1.1, start="2004-01-01")
        s = gbm(seed=seed, s0=5000)
        return s * np.exp(np.arange(len(s)) * 0.0002) if dividends else s

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
    cpi["JPN"] = np.linspace(100, 112, len(cpi_idx))
    cpi["USA"] = np.linspace(100, 170, len(cpi_idx))
    monkeypatch.setattr(D, "cpi_index", lambda names, start="2000-01-01": cpi)
    monkeypatch.setattr(D, "monthly_closes", lambda syms, years=22, dividends=False: {s: fake_closes(s).resample("ME").last() for s in syms})
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
    assert j["rows"][0]["windows"] and len(j["rows"][0]["windows"][0]) == 2
    assert len(j["corr"]["matrix"]) == 2


def test_comps_and_returns(client):
    j = client.get("/api/comps", params={"symbols": "7974.T,6758.T"}).json()
    assert j["rows"][0]["name"] == "Name 7974.T" and j["jpy_1y"] == pytest.approx(0.87)
    r = client.get("/api/returns", params={"symbols": "IEF,SPY", "years": 10}).json()
    assert r["months"] > 100 and len(r["cov"]) == 2


def test_history(client):
    j = client.get("/api/history", params={"symbols": "USDJPY=X", "start": "2025-01-01"}).json()
    assert len(j["series"]["USDJPY=X"]) >= 12


def test_ohlc(client, monkeypatch):
    def fake_ohlc(sym, years=3):
        c = gbm(seed=3, s0=150, start="2017-01-02")
        return pd.DataFrame({"open": c * 0.999, "high": c * 1.004, "low": c * 0.995, "close": c, "volume": 1000.0})
    monkeypatch.setattr(D, "ohlc", fake_ohlc)
    j = client.get("/api/ohlc", params={"symbols": "USDJPY=X,UST10Y", "start": "2025-01-01", "interval": "W"}).json()
    b = j["series"]["USDJPY=X"]
    assert all(x["l"] <= min(x["o"], x["c"]) and x["h"] >= max(x["o"], x["c"]) for x in b)
    assert pd.Timestamp(b[0]["d"]).dayofweek == 4 and b[0]["v"] > 1000
    assert j["close_only"] == ["UST10Y"] and j["series"]["UST10Y"][-1]["h"] == j["series"]["UST10Y"][-1]["c"]
    d = client.get("/api/ohlc", params={"symbols": "USDJPY=X", "start": "2026-09-01"}).json()["series"]["USDJPY=X"]
    m = client.get("/api/ohlc", params={"symbols": "USDJPY=X", "start": "2026-09-01", "interval": "M"}).json()["series"]["USDJPY=X"]
    sep = [x for x in d if x["d"].startswith("2026-09")]
    assert m[0]["h"] == max(x["h"] for x in sep) and m[0]["c"] == sep[-1]["c"] and m[0]["o"] == sep[0]["o"]


def test_history_total_return(client):
    p = client.get("/api/history", params={"symbols": "TLT", "start": "2025-01-01", "interval": "W"}).json()
    t = client.get("/api/history", params={"symbols": "TLT", "start": "2025-01-01", "interval": "W", "total": "true"}).json()
    assert t["total_return"] is True and p["total_return"] is False
    assert t["series"]["TLT"][-1]["v"] > p["series"]["TLT"][-1]["v"]


def test_ohlc_from_provider_frame(monkeypatch):
    """D.ohlc on the frame shape OpenBB returns (plain row index, a 'date' column)."""
    idx = pd.bdate_range("2026-09-01", periods=30)
    frame = pd.DataFrame({"date": idx, "open": np.linspace(150, 152, 30), "high": np.linspace(151, 153, 30),
                          "low": np.linspace(149, 151, 30), "close": np.linspace(150.5, 152.5, 30), "volume": 0.0})

    class Res:
        def to_dataframe(self):
            return frame.copy()

    class O:
        class yfinance:
            class equity:
                class price:
                    @staticmethod
                    def historical(**kw):
                        return Res()

    monkeypatch.setattr(D, "obb", lambda: O)
    D.ohlc.cache_clear()
    out = D.ohlc("USDJPY=X", 1)
    assert len(out) == 30 and out["close"].notna().all() and out.index[0] == idx[0]
    assert out["close"].iloc[-1] == pytest.approx(152.5)


def test_history_yields(client):
    j = client.get("/api/history", params={"symbols": "USDJPY=X,UST2Y,JGB10Y", "start": "2025-01-01", "interval": "D"}).json()
    assert j["units"] == {"USDJPY=X": "price", "UST2Y": "pct", "JGB10Y": "pct"}
    assert j["series"]["UST2Y"][-1]["v"] == pytest.approx(4.04)
    assert j["series"]["JGB10Y"][-1]["v"] == pytest.approx(1.5)
    assert j["series"]["JGB10Y"][0]["d"] >= "2025-01-01" and not j["missing"]


def test_ppp_matches_dates_and_drops_stale(client, monkeypatch):
    idx = pd.date_range("2005-01-31", "2026-08-31", freq="ME")
    cpi = pd.DataFrame({"USA": np.linspace(100, 170, len(idx)), "JPN": np.linspace(100, 112, len(idx)),
                        "MEX": np.linspace(100, 200, len(idx))}, index=idx)
    cpi.loc[cpi.index > "2021-06-30", "JPN"] = np.nan  # Japan's series stops in 2021
    cpi.loc[cpi.index > "2025-12-31", "MEX"] = np.nan  # Mexico lags 9 months
    monkeypatch.setattr(D, "cpi_index", lambda names, start="2000-01-01": cpi)
    j = client.get("/api/ppp").json()
    isos = {r["iso"]: r for r in j["rows"]}
    assert "JPN" not in isos and "JPN" in j["missing"]
    mex = isos["MEX"]
    us_at = cpi.loc["2025-12-31", "USA"] / cpi.loc[cpi.index.year == 2005, "USA"].mean()
    assert mex["cpi_us"] == pytest.approx(us_at, rel=1e-3) and mex["stale"] is True


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


# --------------------------------------------------------------------------- #
# FX carry
# --------------------------------------------------------------------------- #

RATE_CSV = """DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE
X,JPN,M,IR3TIB,PA,2026-07,0.6
X,JPN,M,IR3TIB,PA,2026-08,0.65
X,USA,M,IR3TIB,PA,2026-08,4.1
X,USA,Q,IR3TIB,PA,2026-Q3,4.0
X,USA,M,IRLT,PA,2026-08,4.4
"""


def test_parse_rate_csv():
    df = D._parse_rate_csv(RATE_CSV)
    assert df.loc["2026-08-31", "JPN"] == pytest.approx(0.65)
    assert df.loc["2026-08-31", "USA"] == pytest.approx(4.1)
    assert len(df) == 2


def test_despike():
    s = pd.Series([100, 101, 300, 102, 103.0], index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    assert list(D.despike(s).values) == [100, 101, 102, 103]


def test_fx_carry(client, monkeypatch):
    idx = pd.date_range("2004-01-31", "2026-03-31", freq="ME")
    oecd = pd.DataFrame({a: 2.0 for a, _ in M.FX_CARRY.values() if a}, index=idx)
    oecd["JPN"] = 0.1
    bis = pd.DataFrame({"SA": 5.0, "HK": 4.0, "US": 3.0}, index=pd.date_range("2004-01-31", "2026-09-30", freq="ME"))
    monkeypatch.setattr(D, "oecd_3m_rates", lambda areas, start="2000-01": oecd)
    monkeypatch.setattr(D, "bis_policy_rates", lambda areas, start="2000-01": bis)
    j = client.get("/api/fxcarry").json()
    n = len(j["months"])
    assert j["ccys"][0] == "USD" and "MXN" in j["ccys"]
    assert all(len(v) == n for v in j["spot"].values())
    assert all(len(v) == n for v in j["rate"].values())
    assert j["rate_src"]["SAR"] == "bis" and j["rate_src"]["JPY"] == "oecd"
    assert j["rate_src"]["USD"] == "oecd+bis"
    assert "SGD rate" in j["errors"]
    assert j["spot"]["USD"][0] == 1.0
    assert set(j["bench"]) == set(M.FX_BENCH)
    assert len(j["bench"]["GOLD"]["v"]) == n and j["bench"]["AGG"]["tr"] is True


# --------------------------------------------------------------------------- #
# Macro regimes
# --------------------------------------------------------------------------- #

CLI_CSV = """DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,ADJUSTMENT,TRANSFORMATION,TIME_HORIZ,METHODOLOGY,TIME_PERIOD,OBS_VALUE
X,USA,M,LI,IX,_Z,AA,IX,_Z,H,2026-07,100.4
X,USA,M,LI,IX,_Z,AA,IX,_Z,H,2026-08,100.6
X,USA,M,LI,IX,_Z,NOR,IX,_Z,H,2026-08,99.0
X,JPN,M,LI,IX,_Z,AA,IX,_Z,H,2026-08,99.1
X,JPN,M,BCICP,IX,_Z,AA,IX,_Z,H,2026-08,50.0
"""


def test_parse_cli_csv():
    df = D.parse_cli_csv(CLI_CSV)
    assert df.loc["2026-08-31", "USA"] == pytest.approx(100.6)
    assert df.loc["2026-08-31", "JPN"] == pytest.approx(99.1)
    assert len(df) == 2


def test_parse_confidence_csv():
    text = CLI_CSV.replace(",LI,", ",BCICP,")
    df = D.parse_cli_csv(text, "BCICP")
    assert df.loc["2026-08-31", "USA"] == pytest.approx(100.6)


def test_macro(client, monkeypatch):
    idx = pd.date_range("2000-01-31", "2026-08-31", freq="ME")
    cli = pd.DataFrame({"USA": 100 + np.sin(np.arange(len(idx)) / 6), "EA19": 100.0}, index=idx)
    cpi = pd.DataFrame({"USA": 2.5, "JPN": 1.0, "EA20": 2.0}, index=idx)
    monkeypatch.setattr(D, "oecd_cli", lambda areas, start="2000-01": cli)
    monkeypatch.setattr(D, "cpi_yoy", lambda areas, start="2000-01": cpi)
    conf = {"BCICP": pd.DataFrame({"USA": 99.5, "JPN": 100.2}, index=idx), "CCICP": pd.DataFrame()}
    monkeypatch.setattr(D, "oecd_confidence", lambda areas, measure, start="2000-01": conf[measure])
    j = client.get("/api/macro", params={"areas": "US,JP,EA"}).json()
    assert j["series"]["US"]["bci"][-1]["v"] == 99.5 and j["series"]["JP"]["bci"] and j["series"]["US"]["cci"] == []
    assert len(j["series"]["US"]["cli"]) == len(idx) and j["series"]["US"]["cpi_yoy"][-1]["v"] == 2.5
    assert j["series"]["JP"]["cli"] == [] and j["series"]["JP"]["cpi_yoy"]
    assert j["series"]["EA"]["cli"][0]["v"] == 100.0  # EA19 fallback
    assert j["asof"]["US"]["cli"] == "2026-08-31"
    assert client.get("/api/macro", params={"areas": "XX"}).status_code == 400
    monkeypatch.setattr(D, "oecd_cli", lambda areas, start="2000-01": pd.DataFrame())
    monkeypatch.setattr(D, "cpi_yoy", lambda areas, start="2000-01": (_ for _ in ()).throw(ValueError("down")))
    assert client.get("/api/macro").status_code == 503


# --------------------------------------------------------------------------- #
# news (Bybee et al. 2024 follow-ups): parsers and routes, no network
# --------------------------------------------------------------------------- #
from app import news as N  # noqa: E402

RSS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><rss version="2.0"><channel><title>t</title>
<item><title>Yen jumps as BOJ rate check raises intervention risk - Yahoo Finance</title><link>https://news.google.com/rss/articles/A1?oc=5</link><pubDate>Fri, 18 Sep 2026 16:03:53 GMT</pubDate><source url="https://finance.yahoo.com">Yahoo Finance</source></item>
<item><title>日銀が「レートチェック」を実施 円安けん制か - NHKニュース</title><link>https://news.google.com/rss/articles/A2?oc=5</link><pubDate>Fri, 18 Sep 2026 07:00:00 GMT</pubDate><source url="https://www3.nhk.or.jp">NHKニュース</source></item>
<item><title>財務相、投機的な動きに「断固たる措置」 為替 - 日本経済新聞</title><link>https://news.google.com/rss/articles/A3?oc=5</link><pubDate>Wed, 14 Oct 2026 23:30:00 GMT</pubDate><source url="https://www.nikkei.com">日本経済新聞</source></item>
<item><title>日経平均が反落 - 日本経済新聞</title><link>https://news.google.com/rss/articles/A4?oc=5</link><pubDate>Wed, 14 Oct 2026 03:00:00 GMT</pubDate><source url="https://www.nikkei.com">日本経済新聞</source></item>
</channel></rss>"""


def test_parse_rss_and_ladder():
    items = N.parse_rss(RSS)
    assert len(items) == 4
    assert items[0]["title"] == "Yen jumps as BOJ rate check raises intervention risk" and items[0]["source"] == "Yahoo Finance"
    assert items[0]["d"] == "2026-09-19"  # 16:03 GMT is the next day in Tokyo
    assert items[2]["d"] == "2026-10-15"
    assert N.classify(items[0]["title"]) == (5, 100)
    assert N.classify(items[1]["title"]) == (5, 100)
    assert N.classify(items[2]["title"]) == (4, 80)  # 断固たる beats 投機的
    assert N.classify(items[3]["title"]) is None  # not about FX
    assert N.classify("Yen weakens; ministry watching markets with a sense of urgency") == (2, 30)
    assert N.classify("日銀が「レートチェック」") == (5, 100)  # no FX word, but a rate check is FX by nature
    assert N.classify("介入もレートチェックもいまのところ伝わらず＝ＮＹ為替") is None  # reports that nothing happened
    assert N.classify("英財務相、ＥＵ産業政策で英国排除しないよう要請へ") is None  # not about the yen
    assert N.classify("日米利上げ観測、円高行き過ぎに修正余地") is None  # yen strength


def test_parse_sentiment_csv():
    s = N.parse_sentiment_csv("date,Daily news sentiment\n01/01/1980,-0.04\n01/02/1980,-0.11\n10/11/2026,0.06\n")
    assert len(s) == 3 and s.index[-1] == pd.Timestamp("2026-10-11") and s.iloc[0] == pytest.approx(-0.04)


def test_coverage_dedupes_and_counts(monkeypatch):
    from datetime import datetime, timedelta  # noqa: PLC0415

    today = datetime.now(N.JST).date()
    d1 = (today - timedelta(days=2)).isoformat()
    calls = []

    def fake(query, lang):
        calls.append(query)
        return [{"d": d1, "title": "a", "source": "x", "link": "L1"}, {"d": d1, "title": "b", "source": "y", "link": "L2"}]

    monkeypatch.setattr(N, "_gnews", fake)
    from app import data as Dm  # noqa: PLC0415

    Dm._cache.clear()
    c = N.coverage("為替介入", "ja", 14)
    assert c["series"][-1]["d"] == today.isoformat()
    assert sum(x["n"] for x in c["series"]) == 2  # same links in every window counted once
    assert next(x["n"] for x in c["series"] if x["d"] == d1) == 2
    assert all("after:" in q and "before:" in q for q in calls)


def test_news_routes(client, monkeypatch):
    idx = pd.date_range("2024-01-01", "2026-10-11", freq="D")
    monkeypatch.setattr(N, "news_sentiment", lambda: pd.Series(0.05, index=idx))
    j = client.get("/api/newsmood", params={"start": "2025-01-01"}).json()
    assert j["asof"] == "2026-10-11" and j["monthly"][0]["d"] == "2025-01-31" and j["daily"]
    monkeypatch.setattr(N, "coverage", lambda q, lang, days: {"series": [{"d": "2026-10-14", "n": 11}], "windows": 30, "failed_windows": 0, "capped_windows": []})
    monkeypatch.setattr(N, "ladder_headlines", lambda days=30: {"items": [{"d": "2026-10-15", "title": "x", "rung": 4, "score": 80}], "errors": []})
    j = client.get("/api/fxnews").json()
    assert j["coverage"]["series"][0]["n"] == 11 and j["headlines"][0]["rung"] == 4 and len(j["ladder"]) == 4


# --------------------------------------------------------------------------- #
# Sentiment sources
# --------------------------------------------------------------------------- #
from app import sentiment as S  # noqa: E402


def test_parse_cot():
    rows = [
        {"report_date_as_yyyy_mm_dd": "2026-10-13T00:00:00.000", "cftc_contract_market_code": "097741",
         "open_interest_all": "200000", "noncomm_positions_long_all": "30000", "noncomm_positions_short_all": "70000",
         "comm_positions_long_all": "100000", "comm_positions_short_all": "50000"},
        {"report_date_as_yyyy_mm_dd": "2026-10-06T00:00:00.000", "cftc_contract_market_code": "097741",
         "open_interest_all": "0", "noncomm_positions_long_all": "1", "noncomm_positions_short_all": "1"},
        {"report_date_as_yyyy_mm_dd": "2026-10-13T00:00:00.000", "cftc_contract_market_code": "999999",
         "open_interest_all": "1", "noncomm_positions_long_all": "1", "noncomm_positions_short_all": "1"},
    ]
    c = S.parse_cot(rows)
    assert list(c) == ["JPY"] and [r["d"] for r in c["JPY"]] == ["2026-10-06", "2026-10-13"]
    assert c["JPY"][1]["net"] == -40000 and c["JPY"][1]["pct"] == -20.0 and c["JPY"][1]["cnet"] == 50000
    assert c["JPY"][0]["pct"] is None


def test_mof_week_parse():
    txt = ('x,y\n,1. Portfolio\n'
           '2005．1．2〜 1．8,"1,689 ","1,419 ",270 ,"23,929 ","15,051 ","8,878 ","9,149 ","1,662 ","1,001 ",662 ,"9,811 ",'
           '"17,853 ","15,575 ","2,278 ","15,867 ","11,675 ","4,192 ","6,470 ","7,885 ","3,991 ","3,894 ","10,363 "\n'
           '2025．12．28〜 1．3,1,1,-5 ,1,1,1,1,1,1,1,1,1,1,"-1,234 ",1,1,-7 ,1,1,1,1,1\n'
           '   (Note 1),Transactions,,,,,,,,,,,,,,,,,,,,,\n')
    r = S.parse_mof_week(txt)
    assert r == [{"d": "2005-01-08", "fe": 2278.0, "fb": 4192.0, "re": 270.0},
                 {"d": "2026-01-03", "fe": -1234.0, "fb": -7.0, "re": -5.0}]


def test_margin_and_investor_parsers():
    hist = pd.DataFrame([["月日", None, None, None, None], ["[東京・名古屋", None, None, None, None],
                         [pd.Timestamp("2026-09-18"), 481794, 884462, 3497919, 6455988],
                         ["[東京", None, None, None, None], [pd.Timestamp("2026-09-18"), 1, 1, 1, 1]])
    assert S.parse_margin_hist(hist) == [{"d": "2026-09-18", "ss": 481794.0, "sv": 884462.0, "ls": 3497919.0, "lv": 6455988.0}]
    wk = pd.DataFrame([[None] * 8 for _ in range(16)])
    wk.iloc[0, 1] = pd.Timestamp("2026-10-16")
    wk.iloc[13, 1] = "信用取引残高合計\nTotal"
    wk.iloc[13, 4], wk.iloc[13, 6], wk.iloc[14, 4], wk.iloc[14, 6] = 366263, 3734445, 667053, 9.0e6
    assert S.parse_margin_week(wk) == {"d": "2026-10-16", "ss": 366263.0, "ls": 3734445.0, "sv": 667053.0, "lv": 9.0e6}
    iv = pd.DataFrame([[None] * 59 for _ in range(9)])
    iv.iloc[7, 1], iv.iloc[8, 2] = "東証プライム\nTSE Prime", "金額 Value"
    iv.iloc[8, 21], iv.iloc[8, 25], iv.iloc[8, 13], iv.iloc[8, 17] = -352159410, 8684813, 29421973, 183980265
    iv.iloc[8, 53], iv.iloc[8, 33] = -76396056, 138108954
    r = S.parse_investor_week(iv, "x/stock_1_w_20261013_20261016.xlsx")
    assert r == {"d": "2026-10-16", "foreign": -3434.7, "indiv": 2134.0, "trust_bk": -764.0, "inv_trust": 1381.1}


def test_sentiment_routes(client, monkeypatch):
    monkeypatch.setattr(S, "cot", lambda keys, start: {"JPY": [{"d": "2026-10-13", "pct": -20.0}]})
    j = client.get("/api/cot", params={"markets": "JPY,SPX,nope"}).json()
    assert j["series"]["JPY"][0]["pct"] == -20.0 and j["missing"] == ["SPX"] and j["asof"] == "2026-10-13"
    assert client.get("/api/cot", params={"markets": "nope"}).status_code == 400
    monkeypatch.setattr(S, "mof_week", lambda: [{"d": "2004-12-31", "fe": 1}, {"d": "2026-10-17", "fe": -3000}])
    monkeypatch.setattr(S, "jpx_margin", lambda: (_ for _ in ()).throw(ValueError("down")))
    monkeypatch.setattr(S, "jpx_investors", lambda: [])
    j = client.get("/api/jpflows").json()
    assert j["mof"] == [{"d": "2026-10-17", "fe": -3000}] and j["margin"] == [] and "margin: down" in j["errors"][0]
    monkeypatch.setattr(N, "_gnews", lambda q, lang: [{"d": "2026-10-20", "title": "a", "source": "s", "link": "1"},
                                                      {"d": "2026-10-21", "title": "a", "source": "t", "link": "2"},
                                                      {"d": "2026-10-21", "title": "b", "source": "t", "link": "3"}])
    j = client.get("/api/headlines", params={"q": "stocks"}).json()
    assert [i["title"] for i in j["items"]] == ["a", "b"] and j["items"][0]["d"] == "2026-10-21"


def test_stocktwits_parse_and_route(client, monkeypatch):
    j = {"messages": [{"id": 3, "created_at": "2026-10-25T08:00:00Z", "body": "it&#39;s to the  moon\n$SPY", "entities": {"sentiment": {"basic": "Bullish"}}, "likes": {"total": 4}},
                      {"id": 2, "created_at": "2026-10-25T07:00:00Z", "body": "x", "entities": {"sentiment": None}},
                      {"id": 1, "created_at": "2026-10-25T06:00:00Z", "body": "y", "entities": {"sentiment": {"basic": "Bearish"}}}]}
    p = S.parse_stocktwits(j)
    assert [m["s"] for m in p] == [1, 0, -1] and p[0]["body"] == "it's to the moon $SPY" and p[0]["likes"] == 4 and p[1]["likes"] == 0
    monkeypatch.setattr(S, "stocktwits", lambda syms, pages: {"SPY": p, "NOPE": []})
    r = client.get("/api/social", params={"symbols": "SPY,NOPE"}).json()
    assert r["series"]["SPY"]["bull_share"] == 0.5 and r["series"]["SPY"]["to"] == "2026-10-25T08:00:00Z" and r["missing"] == ["NOPE"]


def test_reddit_parse_and_route(client, monkeypatch):
    j = {"data": {"children": [{"data": {"stickied": True, "title": "Daily thread"}},
                               {"data": {"created_utc": 1761379200, "title": "SPY  puts\\nprinting", "score": 42, "num_comments": 7,
                                         "link_flair_text": "YOLO", "permalink": "/r/wallstreetbets/comments/x/"}}]}}
    p = S.parse_reddit(j)
    assert len(p) == 1 and p[0]["t"] == "2025-10-25T08:00:00Z" and p[0]["score"] == 42 and p[0]["link"].endswith("/comments/x/")
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    assert client.get("/api/reddit").status_code == 503
    monkeypatch.setenv("REDDIT_CLIENT_ID", "a"); monkeypatch.setenv("REDDIT_CLIENT_SECRET", "b")
    monkeypatch.setattr(S, "reddit", lambda subs, sort, limit: {"stocks": p, "investing": []})
    r = client.get("/api/reddit", params={"subs": "r/stocks,investing"}).json()
    assert r["posts"]["stocks"][0]["flair"] == "YOLO" and r["missing"] == ["investing"]


def test_fx_close_fix():
    from app.analytics import fx_close_fix  # noqa: PLC0415

    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    rng = np.random.default_rng(1)
    o = 150 + np.cumsum(rng.normal(0, .5, 60))
    stale = pd.DataFrame({"open": o, "high": o + 1, "low": o - 1, "close": o + .01}, index=idx)
    stale.iloc[10] = [stale.iloc[10, 0]] * 4  # a flat placeholder bar
    f = fx_close_fix(stale)
    assert len(f) == 59 and f["close"].iloc[0] == f["open"].iloc[1] and f["close"].iloc[-1] == stale["close"].iloc[-1]
    assert (f["high"] >= f["close"]).all() and (f["low"] <= f["close"]).all()
    clean = pd.DataFrame({"open": o, "high": o + 1, "low": o - 1, "close": o + .6}, index=idx)
    assert fx_close_fix(clean).equals(clean)  # a healthy series is left alone


def test_history_compact(client):
    rows = client.get("/api/history", params={"symbols": "USDJPY=X,^N225", "start": "2025-01-01", "interval": "D"}).json()
    col = client.get("/api/history", params={"symbols": "USDJPY=X,^N225", "start": "2025-01-01", "interval": "D", "compact": "true"}).json()
    assert col["compact"] is True and "compact" not in rows
    for s in ("USDJPY=X", "^N225"):
        a, b = rows["series"][s], col["series"][s]
        assert [x["d"] for x in a] == b["d"]
        assert all(abs(x["v"] - y) <= 1e-5 * abs(x["v"]) for x, y in zip(a, b["v"]))


def test_ohlc_compact(client, monkeypatch):
    def fake_ohlc(sym, years=3):
        c = gbm(seed=3, s0=150, start="2017-01-02")
        return pd.DataFrame({"open": c * 0.999, "high": c * 1.004, "low": c * 0.995, "close": c, "volume": 1000.0})
    monkeypatch.setattr(D, "ohlc", fake_ohlc)
    j = client.get("/api/ohlc", params={"symbols": "USDJPY=X", "start": "2025-01-01", "compact": "true"}).json()
    b = j["series"]["USDJPY=X"]
    assert j["compact"] is True and len(b["d"]) == len(b["o"]) == len(b["h"]) == len(b["l"]) == len(b["c"]) > 100
    assert all(l <= c <= h for l, c, h in zip(b["l"], b["c"], b["h"]))


def test_year_bucket():
    assert D.year_bucket(2.1) == 3 and D.year_bucket(10.1) == 17 and D.year_bucket(16.9) == 17 and D.year_bucket(30) == 30


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(D, "CACHE_MAX_ITEMS", 5)
    D._cache.clear(); D._sizes.clear()

    @D.cached(ttl=60)
    def f(i):
        return pd.Series(np.arange(10.0) + i)

    for i in range(12):
        f(i)
    assert len(D._cache) == 5
    assert ("f", (11,), ()) in D._cache and ("f", (0,), ()) not in D._cache


def test_ratescarry(client, monkeypatch):
    idx = pd.date_range("2005-01-31", "2026-09-30", freq="ME")
    lt = pd.DataFrame({b: np.linspace(4.0, 3.0, len(idx)) + i * 0.1 for i, (_, b, _) in enumerate(M.RATES_CARRY.values())}, index=idx)
    st = pd.DataFrame({a: np.linspace(3.0, 1.0, len(idx)) for a, _, _ in M.RATES_CARRY.values()}, index=idx)
    monkeypatch.setattr(D, "oecd_lt_rates", lambda areas, start="2000-01": lt)
    monkeypatch.setattr(D, "oecd_3m_rates", lambda areas, start="2000-01": st)
    monkeypatch.setattr(D, "bis_policy_rates", lambda areas, start="2000-01": pd.DataFrame())
    j = client.get("/api/ratescarry", params={"years": 20}).json()
    assert "USD" in j["ccys"] and "EUR" in j["ccys"] and len(j["months"]) == len(j["long"]["USD"])
    assert j["long"]["USD"][-1] > j["short"]["USD"][-1]
    assert j["usd_filter_today"] is not None and "gap_bp" in j["usd_filter_today"]


def test_trend_filter_sign():
    up = pd.Series(np.linspace(3, 5, 400), index=pd.bdate_range("2024-01-01", periods=400))
    f = M._trend_filter(up)
    assert (f["gap"] > 0).all()


def test_fxfactors(client, monkeypatch):
    idx = pd.date_range("2004-01-31", "2026-10-31", freq="ME")
    n = len(idx)
    fxs = {f"{c}=X": pd.Series(np.linspace(1, 1.2, n) * (i + 1), index=idx) for i, c in enumerate(M.FXF) if c != "USD"}
    fxs["CNY=X"] = pd.Series(np.linspace(6, 7, n), index=idx)
    eqs = {v[3]: pd.Series(np.linspace(100, 200, n), index=idx) for v in M.FXF.values() if v[3]}
    monkeypatch.setattr(D, "monthly_closes", lambda syms, years=22, dividends=False: {s: (fxs.get(s) if s in fxs else eqs.get(s)) for s in syms if s in fxs or s in eqs})
    areas = [v[0] for v in M.FXF.values() if v[0]]
    df = pd.DataFrame({a: np.linspace(2, 4, n) for a in areas}, index=idx)
    for name in ("oecd_3m_rates", "oecd_lt_rates", "cpi_index", "oecd_cli"):
        monkeypatch.setattr(D, name, lambda a, start="2000-01", _df=df: _df)
    monkeypatch.setattr(D, "oecd_confidence", lambda a, m, start="2000-01": df)
    monkeypatch.setattr(D, "bis_policy_rates", lambda a, start="2000-01": pd.DataFrame())
    j = client.get("/api/fxfactors", params={"years": 20}).json()
    assert "JPY" in j["spot"] and "USD" not in j["spot"] and len(j["spot"]["JPY"]) == len(j["months"])
    assert j["rate"]["USD"][-1] is not None and "USD" in j["equity"]
    assert j["cost_bp"]["THB"] == 9


def test_volprem_engine():
    from app import volprem as VP
    rng = np.random.default_rng(1)
    s = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
    iv = np.full(1500, 0.25)          # implied well above realised (~16%): sellers should win
    opt, dlt, prem = VP.option_book(s, iv, "ln", 0.003)
    for h in ("none", "1", "budget"):
        hp, _ = VP.hedge_pnl(s, iv, dlt, "ln", h)
        assert (opt + hp).sum() > 0
    hp1, d1 = VP.hedge_pnl(s, iv, dlt, "ln", "1")
    hpb, db = VP.hedge_pnl(s, iv, dlt, "ln", "budget")
    assert d1.sum() > db.sum() > 0
    assert np.std(opt + hp1) < np.std(opt)          # hedging daily cuts the noise
    y = 400 + np.cumsum(rng.normal(0, 5, 800))      # a yield in bp, Bachelier
    o2, d2, _ = VP.option_book(y, np.full(800, 120.0), "n", 2.0)
    assert np.isfinite(o2).all()


def test_hmm_regimes_sorted():
    from app import volprem as VP
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(m, 0.05, 150) for m in (2.5, 3.0, 3.4, 3.9, 2.5)])
    m = VP.hmm_fit(x, 4, iters=30)
    assert list(m["mu"]) == sorted(m["mu"])
    p = VP.hmm_filter(x, m)
    assert p.shape == (len(x), 4) and p[-1].argmax() == 0 and p[500].argmax() == 3


def test_volpremium_route(client, monkeypatch):
    idx = pd.bdate_range("2012-01-02", periods=1800)
    rng = np.random.default_rng(3)
    def fake(syms, years=3, dividends=False):
        out = {}
        for sym in syms:
            if sym.startswith("^") and sym not in ("^GSPC", "^NDX", "^RUT"):
                out[sym] = pd.Series(18 + 4 * np.sin(np.arange(1800) / 90) + rng.normal(0, .3, 1800), index=idx)
            else:
                out[sym] = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .009, 1800))), index=idx)
        return out
    monkeypatch.setattr(D, "many_closes", fake)
    monkeypatch.setattr(M, "_yield_series", lambda sym, years: pd.Series(4 + np.cumsum(rng.normal(0, .05, 1800)), index=idx))
    monkeypatch.setattr(D, "implied_vol", lambda sym, days=365, **k: {"atm_iv": 9.5, "days": 30})
    j = client.get("/api/volpremium", params={"years": 8}).json()
    assert "SPX" in j["markets"] and "UST" in j["markets"] and "EURUSD" in j["fx"]
    m = j["markets"]["SPX"]
    assert len(m["ret"]["budget"]) == len(m["months"]) and len(m["season"]) == 12
    assert "rv1m" in j["fx"]["EURUSD"]["today"]


def test_dimensions_route(client, monkeypatch):
    idx = pd.date_range("2005-01-31", "2026-10-31", freq="ME")
    rng = np.random.default_rng(4)
    px = {sym: pd.Series(100 * np.exp(np.cumsum(rng.normal(0.004, 0.04, len(idx)))), index=idx)
          for sym, _, _ in M.DIM_ASSETS.values()}
    monkeypatch.setattr(D, "monthly_closes", lambda syms, years=22, dividends=False: {s: px[s] for s in syms})
    months = [str(d.date()) for d in idx]
    ccys = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "MXN", "ZAR", "PLN", "HUF", "CZK"]
    fx = {"months": months, "spot": {c: list(np.linspace(1, 1.1, len(idx)) * (i + 1)) for i, c in enumerate(ccys) if c != "USD"},
          "rate": {c: [float(i)] * len(idx) for i, c in enumerate(ccys)}}
    fx["spot"]["USD"] = [1.0] * len(idx)
    rc = {"months": months, "groups": {"G10": ["USD", "EUR"]}, "long": {"USD": [4.0] * len(idx), "EUR": [3.0] * len(idx)},
          "short": {"USD": [2.0] * len(idx), "EUR": [1.0] * len(idx)}, "usd_filter_gap_bp": [-5.0] * len(idx)}
    monkeypatch.setattr(M, "fx_carry", lambda years=22: fx)
    monkeypatch.setattr(M, "rates_carry", lambda years=25: rc)
    monkeypatch.setattr(M, "_spx_shortvol", lambda years: pd.Series(0.01, index=idx))
    j = client.get("/api/dimensions", params={"years": 15}).json()
    assert {"SPY", "TREND", "FXCARRY", "RATESCARRY", "SHORTVOL"} <= set(j["series"])
    assert j["series"]["TREND"]["group"] == "Strategies"
    rc_vals = [v for v in j["series"]["RATESCARRY"]["vals"] if v is not None]
    assert abs(rc_vals[-1] - 0.00167) < 1e-4     # flat curve, no yield change: (4-2 and 3-1)/12 = 0.17%
    assert all(len(s["vals"]) == len(j["months"]) for s in j["series"].values())


def test_revisions_parse_and_groups():
    from app import revisions as R

    def item(period, pts, up=2, dn=1):
        keys = ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"]
        return {"period": period, "endDate": "2027-03-31",
                "epsTrend": {k: {"raw": v} for k, v in zip(keys, pts)},
                "epsRevisions": {"upLast30days": {"raw": up}, "downLast30days": {"raw": dn}},
                "earningsEstimate": {"numberOfAnalysts": {"raw": 10}}}

    # rising for a month after two months of cuts: the turn is about 4 weeks ago
    rec = R.parse_trend("XOM", [item("0y", [1.10, 1.08, 1.00, 1.03, 1.06]), item("+1y", [1.32, 1.30, 1.20, 1.24, 1.27])])
    assert rec["fy0"][0] == 1.10 and rec["fy1_up"] == 2 and rec["fy0_end"] == "2027-03-31"
    recs = {t: R.parse_trend(t, rec and [item("0y", [1.10, 1.08, 1.00, 1.03, 1.06]), item("+1y", [1.32, 1.30, 1.20, 1.24, 1.27])])
            for t in R.GROUPS["energy"][3]}
    d = R.build(recs)
    g = d["groups"]["energy"]
    assert g["n"] == 10 and g["direction"] == 1 and g["weeks_since_turn"] == 4
    assert g["up_share"] == 1.0 and g["net_analysts"] == pytest.approx(1 / 3)
    assert "reg_US" in d["groups"] and "world" in d["groups"]


def test_revisions_route_preparing_then_ok(client, monkeypatch):
    from app import revisions as R

    monkeypatch.setattr(R, "_STATE", {"data": None, "at": 0.0, "running": False, "error": None})
    monkeypatch.setattr(R, "fetch_trend", lambda t: R.parse_trend(t, [{"period": "+1y", "epsTrend": {
        "current": {"raw": 2.0}, "7daysAgo": {"raw": 2.0}, "30daysAgo": {"raw": 1.9}, "60daysAgo": {"raw": 1.8},
        "90daysAgo": {"raw": 1.8}}}]))
    R._run(fetch=R.fetch_trend, workers=2)
    j = client.get("/api/revisions").json()
    assert j["status"] == "ok" and j["groups"]["gaming"]["n"] == 10
    assert j["groups"]["world"]["rev2m"] > 0
