# SEO Counterfactual Forecast

Forecasts 12 months of non-brand Google Search Console clicks and estimates the extra clicks if rankings improve by +3 positions. It outputs a monthly table for three scenarios (low, base, high), charts and an Excel file.

## What it does

1. **Loads daily clicks** and fills missing days.
2. **Cleans spikes** (STL decomposition + IQR rule) by replacing them with expected values.
3. **Builds the baseline**, meaning clicks with no changes. It takes the same weekday 364 days earlier and scales it by the year-on-year (YoY) ratio of the last 28 days. Days that already happened keep their actual clicks.
4. **Builds a CTR-by-position curve** from your own page × query data (log-log fit on the median CTR per position).
5. **Applies the +3 gain.** For each page-query it compares CTR at the current position with CTR three positions higher, then averages the ratios weighted by clicks.
6. **Backtests the uplift** on page-queries that really moved up, with a control group and a bootstrap confidence interval.
7. **Applies a share of the uplift** (realisation) with a linear ramp for each scenario.
8. **Exports** the monthly table, charts and an Excel file.

Prophet and SARIMAX are not used. With under two years of history they cannot separate seasonality from trend, so the baseline is seasonal-naive.

## Requirements

**Python 3.9-3.12**

```text
pandas>=1.5
numpy>=1.23
scipy>=1.9
statsmodels>=0.14
plotly>=5.15
matplotlib>=3.6
openpyxl>=3.1
tabulate>=0.9
```

```bash
pip install -r requirements.txt
```

Run it in Jupyter, Colab or VS Code. `fig.show()` needs a notebook or a browser.

### Input files

| File | Level | Columns |
|---|---|---|
| Daily clicks | Date, non-brand, one site section | `date`, `clicks` |
| Page × query | Date × page × query, non-brand, same section | `date`, `page`, `query`, `clicks`, `impressions`, `position` |

- Both files must be **non-brand**. Brand clicks don't respond to ranking changes.
- Daily file: at least one full year of history, so every forecast day has a same-weekday reference.
- Page × query file: needs data **after** the Search Console impression inflation window (`CURVE_FROM`). Only a few months of data gives a sparse curve.
- Update the file paths and sheet names at the top of the script.

## Settings

| Variable | Meaning |
|---|---|
| `FORECAST_START` | First day of the 12-month forecast |
| `YOY_WINDOW` | Days used for the YoY factor (28) |
| `POSITION_GAIN` | Positions gained (3) |
| `start`, `end` | Search Console impression inflation window, excluded from the CTR curve |
| `MIN_IMPR` | Minimum impressions per page-query-month for the curve (50) |
| `SHIP_DATE` | Day the uplift starts (day after the last actual) |
| `SCENARIOS` | Realisation and ramp length for each scenario |

| Scenario | Share of predicted lift | Ramp |
|---|---|---|
| Low | 15% | 150 days |
| Base | 40% | 90 days |
| High | 70% | 60 days |

Low comes from the backtest, Base from the top of its confidence interval, and High is a judgment call.

## Output

`forecast_scenarios.xlsx`:

| Sheet | Content |
|---|---|
| `month` | Baseline, Low, Base, High, incremental clicks and uplift % per month, plus a total row |
| `assumptions` | Realisation, ramp and uplift for each scenario |
| `daily` | Daily baseline and scenario clicks |
| `ctr_curves` | The CTR-by-position curve |

The script also shows a forecast line chart and a "CTR now vs after +3" chart.

## Limitations

- The uplift comes from a CTR curve, which is correlation. The backtest is the only check, and a small sample (n < 30) is not reliable.
- Impressions are held constant, so extra impressions from ranking higher are not counted.
- The same +3 is applied to every page-query. Cannibalisation, competitors and AI Overview changes are not modelled.
- The curve embeds today's AI Overview coverage, so the gain shrinks if coverage grows.
- Hand-set parameters (the 1.0-3.0 uplift cap, ramps, percentiles and the 364-day lag) are assumptions to document.
