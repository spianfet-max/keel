# Keel

A small market-data service on top of the [OpenBB Open Data Platform](https://github.com/OpenBB-finance/OpenBB). One backend feeds:

| Tool | What it does | Keel call |
|------|--------------|-----------|
| **Carry** (artifact) | UST / EUR AAA / JGB curves and yen-hedged yields, hedge-cost history | `api_yields` |
| **Meeting Brief** (artifact) | One-page market pack for a client meeting: moves, talking points, calendar, news | `api_brief` |
| **Barrier** (artifact) | Historical autocall / knock-in / loss rates for worst-of notes, vol, skew, correlation | `api_sp_screen` |
| **Arcade** (artifact) | Japanese games-maker peer board and a zero-cost collar planner | `api_comps` |
| **Claude via MCP** | Ask Claude market questions answered from these tools plus the wider OpenBB catalogue | all of the above |
| **Frontier** | "Calibrate from history": vol and correlations from 10 years of ETF proxies | `api_returns` |
| **Parity** | "Second opinion: inflation": CPI-based relative PPP next to the Big Mac gap | `api_ppp` |
| **Cadence** | "Themes against the market": meeting themes vs USD/JPY, US 10Y, Nikkei | `api_history` |

Only public market data passes through Keel. Pages send ticker symbols and parameters; no client data.

## Deploy (Render, free)

1. Create an empty GitHub repo called `keel` and push this folder to it.
2. On Render: **New + → Blueprint**, pick the repo. `render.yaml` creates two services in Singapore:
   - `keel-api` — the REST API (`/health`, `/docs`, `/api/...`); deployed at `https://keel-api-tqpz.onrender.com`
   - `keel-mcp` — the same routes as MCP tools at `/mcp`
3. Optional: set `KEEL_TOKEN` on `keel-api` to require an `x-keel-token` header (or `?token=`). Leave it empty if the REST API should stay open; it only serves public data.

Free instances sleep after 15 minutes; the first call after that takes about a minute. The pages say so and retry once.

## Connect Claude (and the four artifacts)

claude.ai → Settings → Connectors → **Add custom connector**

- Name: **`Keel`** (exactly — the artifacts look for this name)
- URL: the `keel-mcp` service address shown on Render + `/mcp` (Render adds a suffix, e.g. `keel-mcp-xxxx.onrender.com`)

Then open any of the artifacts and allow Keel when asked. Until then they show clearly labelled sample figures.

In chat, Claude gets the Keel tools (`api_yields`, `api_brief`, `api_sp_screen`, `api_comps`, `api_returns`, `api_history`, `api_ppp`, `api_snapshot`) and OpenBB's own `openbb_list_commands` / `openbb_dispatch` for everything else the installed providers offer. `mcp_prompt.txt` tells it about symbols and conventions.

## Data sources

| Data | Source | Key needed |
|------|--------|-----------|
| Prices, quotes, fundamentals, news, earnings dates, economic calendar | OpenBB `yfinance` extension | no |
| US Treasury curve | OpenBB `federal_reserve` (H.15) | no |
| Euro-area AAA curve | OpenBB `ecb` | no |
| CPI | OECD SDMX API, called directly (OpenBB's `oecd` extension needs more memory than the free tier) | no |
| JGB curve | Japan Ministry of Finance CSV (not in OpenBB) | no |
| Option implied vols | `yfinance` library bundled with OpenBB's extension; thin for Japanese single stocks | no |

Free sources can be delayed or wrong. Nothing here is investment advice.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload          # http://127.0.0.1:8000/docs
pytest -q                              # offline tests, synthetic data
openbb-mcp --app mcp_app.py --name app # MCP on http://127.0.0.1:8001/mcp (needs requirements-mcp.txt)
```

`requirements.txt` installs only the OpenBB pieces Keel uses (~300 MB RAM). `pip install openbb` pulls every extension and needs about 1 GB, more than Render's free tier.

## Layout

```
app/analytics.py   pure calculations (hedged yield, vol, barrier back-test, return stats, Black-Scholes)
app/data.py        every outside call, with an in-memory cache
app/main.py        FastAPI routes; each operation_id becomes an MCP tool name
mcp_app.py         entry point for openbb-mcp
mcp_prompt.txt     system prompt for the MCP server
frontends/         sources for the four artifacts (core.css/js + one page each); build.py inlines them
tests/             pytest, no network
```

## Notes on the numbers

- **Hedge cost** = foreign 1Y (or 3M) government yield − 1Y JGB + cross-currency basis you set. Covered interest parity; real forward points also carry the basis and dealer spread.
- **Barrier back-test** starts a note every trading day, uses closing prices for the barrier, and ignores coupons and funding. It describes the past.
- **Collar** uses Black–Scholes with dividend yield and the 1Y JGB as the yen rate. Japanese single-stock collars trade OTC; dealer quotes will differ.
- **Relative PPP** takes the base-year average exchange rate and moves it by CPI growth relative to the US since then. The base year matters.
