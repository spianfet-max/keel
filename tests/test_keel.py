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
