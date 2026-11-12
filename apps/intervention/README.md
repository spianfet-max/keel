# Intervention

USD/JPY intervention risk meter: one gauge (0–100) built from market measures of how close the yen is to a Ministry of Finance intervention, plus a manual read of official language.

- **Measures:** speed of the move, distance from past MOF intervention levels, stretch from the 200-day average, volatility, the move against the US–Japan 2-year rate gap, and official language (verbal escalation ladder from Keel's news feed)
- **Sections:** score breakdown, escalation ladder and press coverage of "為替介入", a five-day scenario slider, score history with MOF episodes marked, USD/JPY against past intervention levels, and a table of past interventions with the set-up and what followed
- **Data:** Keel `api_history` (USDJPY=X, UST2Y, JGB2Y daily) and `api_fxnews`; inside claude.ai through the Keel connector, elsewhere through the REST API at keel-api-tqpz.onrender.com. A built-in sample (FRED H.10 daily rates) is used when Keel cannot be reached.
- **Languages:** English and Japanese; light and dark themes

`index.html` is the published page as a single self-contained file: open it in a browser, or publish it as an artifact with the Keel connector (tools `api_history`, `api_fxnews`).

Not investment advice; free public data can be delayed or wrong.
