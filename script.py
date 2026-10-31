
# SETUP

!pip install plotly prophet

# data viz
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# anomaly detection
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# dataset manipulation
import pandas as pd
import numpy as np

# timeseries forecasting
from prophet import Prophet

#river cruise brand
df = pd.read_excel('/content/Untitled spreadsheet (9).xlsx')
df.columns = map(str.lower, df.columns)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# Ensure daily frequency (important for decomposition)
df = df.asfreq('D')

# Handle missing values
df['clicks'] = df['clicks'].interpolate()

fig = px.line(df.reset_index(), x='date', y='clicks', title='river cruise brand')
fig.update_layout(xaxis_title='Date', yaxis_title='Clicks', hovermode='x unified')
fig.show()


# STATIONARITY
# ============================================================
result = adfuller(df['clicks'].dropna())
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p-value: {result[1]}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}, {value}')
# p > 0.05 => not stationary; seasonality likely plays a role.

# TIME SERIES DECOMPOSITION (weekly)
# ============================================================
result_weekly = seasonal_decompose(df['clicks'], model='additive', period=7)
result_weekly.plot()
plt.title('Weekly Seasonal Decomposition')
plt.show()

# STL ANOMALY DETECTION
# ============================================================
stl = STL(df['clicks'], period=7, robust=True)
result = stl.fit()

resid = result.resid
Q1, Q3 = resid.quantile(0.25), resid.quantile(0.75)
IQR = Q3 - Q1
anomalies = df[(resid < Q1 - 1.5 * IQR) | (resid > Q3 + 1.5 * IQR)]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df['clicks'], label='Clicks', color='steelblue')
ax.scatter(anomalies.index, anomalies['clicks'], color='red', label='Anomalies', zorder=5)
ax.set_title('Click Anomalies (STL + IQR)')
ax.legend()
plt.tight_layout()
plt.show()

# REPLACE ANOMALIES (don't remove them)
# ============================================================
df['trend'] = result.trend
df['seasonal'] = result.seasonal
df['resid'] = result.resid

Q1, Q3 = df['resid'].quantile(0.25), df['resid'].quantile(0.75)
IQR = Q3 - Q1
df['anomaly'] = (df['resid'] < Q1 - 1.5 * IQR) | (df['resid'] > Q3 + 1.5 * IQR)

df['clean_clicks'] = df['clicks'].copy()
df.loc[df['anomaly'], 'clean_clicks'] = df['trend'] + df['seasonal']

# Validate cleaned series
result_clean = seasonal_decompose(df['clean_clicks'], model='additive', period=7)
result_clean.plot()
plt.title('Weekly Seasonal Decomposition (Cleaned Data)')
plt.show()

FORECAST_START = pd.Timestamp('2026-09-01')
last = df.index.max()
idx = pd.date_range(FORECAST_START, FORECAST_START + pd.DateOffset(months=12) - pd.Timedelta(days=1))
HORIZON_DAYS = len(idx)

# YoY from the last 28 days (current level), not the 90-day average that overstated it
YOY_WINDOW = 28
r_now = df['clean_clicks'].iloc[-YOY_WINDOW:].sum()
r_ly  = df['clean_clicks'].reindex(df.index[-YOY_WINDOW:] - pd.Timedelta(days=364)).sum()
yoy = r_now / r_ly
print(f'YoY (last {YOY_WINDOW}d): {yoy:.3f}')

ref = idx - pd.Timedelta(days=364)
ly = df['clean_clicks'].reindex(ref).ffill().values
SETTLED_FROM = df.index[-YOY_WINDOW]                    # reference days already at today's level: don't rescale
fc = pd.Series(ly * np.where(ref < SETTLED_FROM, yoy, 1.0), index=idx)

# days that already happened use actual clicks, so September = 1,452 and October = real clicks + forecast days
snaive = fc.where(idx > last, df['clicks'].reindex(idx)).rename('seasonal_naive')

print(snaive.resample('MS').sum().round(0).to_markdown())

 The fact that the shortest horizon has the highest error means that the model is tuned for the long-term predictions.

# COUNTERFACTUAL CONFIG
# ============================================================
# GSC impression inflation window (no longer a Prophet regressor, still used for the CTR curve and backtest)
start = pd.to_datetime('2025-05-13')
end   = pd.to_datetime('2026-04-13')

POSITION_GAIN = 3
SHIP_DATE = last + pd.Timedelta(days=1)
MIN_IMPR = 50                                        # per page-query-month
CURVE_FROM = end + pd.offsets.MonthBegin(1)          # post-inflation only: inflated impressions depress CTR

# CTR-BY-POSITION CURVE + BLENDED UPLIFT

q = pd.read_excel('/content/second one.xlsx', sheet_name="river cruise non brand")
q.columns = map(str.lower, q.columns)
q['date'] = pd.to_datetime(q['date'])
q['month'] = q['date'].dt.to_period('M').dt.to_timestamp()
q['pos_x_impr'] = q['position'] * q['impressions']

# page x query x month, position impression-weighted (as GSC does)
m = (q.groupby(['page', 'query', 'month'], as_index=False)
       .agg(clicks=('clicks', 'sum'), impressions=('impressions', 'sum'), pos_x_impr=('pos_x_impr', 'sum')))
m['position'] = m['pos_x_impr'] / m['impressions']
m['ctr'] = m['clicks'] / m['impressions']
m = m[m['impressions'] >= MIN_IMPR]

curve_src = m[m['month'] >= CURVE_FROM]

def ctr_curve(d, pctl, min_n=30):
    d = d[(d['position'] <= 20.5) & (d['ctr'] > 0)].copy()
    d['bucket'] = d['position'].round().clip(1, 20)
    g = d.groupby('bucket')['ctr']
    b = g.quantile(pctl)[g.size() >= min_n]
    # weighted log-log fit: smooth, strictly decreasing, no plateaus or interpolated steps
    slope, icpt = np.polyfit(np.log(b.index.values), np.log(b.values), 1,
                             w=np.sqrt(g.size()[b.index].values))
    pos = np.arange(1, 21)
    return pd.Series(np.exp(icpt) * pos ** slope, index=pos)

# current state = last 3 months, page x query
cur = (m[m['month'] >= m['month'].max() - pd.DateOffset(months=2)]
         .groupby(['page', 'query'], as_index=False)
         .agg(clicks=('clicks', 'sum'), impressions=('impressions', 'sum'), pos_x_impr=('pos_x_impr', 'sum')))
cur['position'] = cur['pos_x_impr'] / cur['impressions']

def blended_uplift(curve, gain):
    x, y = curve.index.values, curve.values
    new_pos = np.maximum(1, cur['position'] - gain)      # floor at position 1

    denom = np.interp(cur['position'], x, y)
    num = np.interp(new_pos, x, y)

    f = np.where(denom > 0, num / denom, 1.0)
    f = np.clip(f, 1.0, 3.0)      # a +3 position gain rarely triples clicks; tune the cap from the backtest
    return np.average(f, weights=cur['clicks'])          # click-weighted, applied to URL's own CTR

base_curve = ctr_curve(curve_src, 0.5)

# BACKTEST
# ============================================================
MIN_IMPR_BT = 20          # backtest only; the CTR curve still uses MIN_IMPR
GAIN_LO, GAIN_HI = 1.5, 5 # "about +3 positions"

# rebuild monthly page x query without the MIN_IMPR filter
mb = (q.groupby(['page', 'query', 'month'], as_index=False)
        .agg(clicks=('clicks', 'sum'), impressions=('impressions', 'sum'), pos_x_impr=('pos_x_impr', 'sum')))
mb = mb[mb['impressions'] >= MIN_IMPR_BT].copy()
mb['position'] = mb['pos_x_impr'] / mb['impressions']
mb['ctr'] = mb['clicks'] / mb['impressions']
mb['regime'] = np.select(
    [mb['month'] < start.to_period('M').to_timestamp(), mb['month'] <= end.to_period('M').to_timestamp()],
    ['pre', 'infl'], 'post')                              # boundary months approximate
mb = mb[mb['clicks'] > 0][['page', 'query', 'month', 'position', 'ctr', 'regime']]

# pair month m with month m+1
nxt = mb.copy()
nxt['month'] = nxt['month'] - pd.DateOffset(months=1)
pairs = mb.merge(nxt, on=['page', 'query', 'month'], suffixes=('_before', '_after'))
pairs = pairs[pairs['regime_before'] == pairs['regime_after']]   # never straddle the inflation window
pairs['gain'] = pairs['position_before'] - pairs['position_after']

# control: page-queries that didn't move -> CTR drift unrelated to ranking
ctrl = pairs[pairs['gain'].abs() <= 0.5]
drift = (ctrl['ctr_after'] / ctrl['ctr_before']).median() if len(ctrl) else 1.0
drift = 1.0 if not np.isfinite(drift) else drift

# treated: ~+3 positions gained
bt = pairs[pairs['gain'].between(GAIN_LO, GAIN_HI)].copy()
x, y = base_curve.index.values, base_curve.values
bt['pred_ratio'] = np.interp(bt['position_after'], x, y) / np.interp(bt['position_before'], x, y)
bt['actual_ratio'] = (bt['ctr_after'] / bt['ctr_before']) / drift

def realisation(d):
    return (d['actual_ratio'].median() - 1) / (d['pred_ratio'].median() - 1)

if len(bt) >= 2:
    rng = np.random.default_rng(0)
    boot = [realisation(bt.iloc[rng.integers(0, len(bt), len(bt))]) for _ in range(500)]
    obs = realisation(bt)
    print(f"treated n={len(bt)} | control n={len(ctrl)}, drift {drift-1:+.1%} | "
          f"predicted {bt['pred_ratio'].median()-1:+.1%} | actual {bt['actual_ratio'].median()-1:+.1%} | "
          f"realisation {obs:.0%} (90% CI {np.percentile(boot, 5):.0%} to {np.percentile(boot, 95):.0%})")
    if len(bt) < 30:
        print("WARNING: n < 30, do not apply this realisation to SCENARIOS")
else:
    print(f"Only {len(bt)} treated pairs; backtest not usable. Control n={len(ctrl)}")

# SCENARIOS, RAMP, MONTHLY TABLE

SCENARIOS = {
    'low':  dict(realisation=0.15, ramp_days=150),   # backtest point estimate
    'base': dict(realisation=0.40, ramp_days=90),    # backtest upper CI bound
    'high': dict(realisation=0.70, ramp_days=60),    # judgment: beyond what the backtest supports
}

last = df.index.max()
scen = pd.DataFrame({'baseline': snaive.clip(lower=0)})

full_uplift = blended_uplift(ctr_curve(curve_src, 0.5), POSITION_GAIN)   # curve-implied lift for the full +3

assump = []
for name, s in SCENARIOS.items():
    uplift = 1 + s['realisation'] * (full_uplift - 1)                    # scale the lift, not the positions
    ramp = np.clip((scen.index - SHIP_DATE).days / s['ramp_days'], 0, 1) # linear ramp
    scen[name] = scen['baseline'] * (1 + ramp * (uplift - 1))
    assump.append(dict(scenario=name, positions_gained=POSITION_GAIN, curve_full_uplift=round(full_uplift, 3),
                       realisation=s['realisation'], ramp_days=s['ramp_days'],
                       blended_uplift=round(uplift, 3), ship_date=SHIP_DATE.date()))
assump = pd.DataFrame(assump)

monthly = scen.resample('MS').sum()
monthly['incremental'] = monthly['base'] - monthly['baseline']
monthly = monthly[['baseline', 'base', 'incremental', 'low', 'high']].round(0)
monthly.columns = ['Baseline clicks', 'Counterfactual (base)', 'Incremental', 'Low', 'High']
monthly.index = monthly.index.strftime('%Y-%m')
print(monthly.to_markdown())

# BASELINE BACKTEST + ERROR MARGIN PER SCENARIO
# ============================================================
BLOCK = 28            # compare 28-day totals
MAX_BLOCKS = 12       # horizons tried (months ahead)
ORIGIN_STEP = 7       # one forecast origin per week

def clean_to(origin):
    """Production cleaning (STL + IQR) using data up to `origin` only."""
    s = df['clicks'].loc[:origin]
    r = STL(s, period=7, robust=True).fit()
    q1, q3 = r.resid.quantile(0.25), r.resid.quantile(0.75)
    iqr = q3 - q1
    bad = (r.resid < q1 - 1.5 * iqr) | (r.resid > q3 + 1.5 * iqr)
    out = s.copy()
    out[bad] = (r.trend + r.seasonal)[bad]
    return out

first_origin = df.index.min() + pd.Timedelta(days=364 + YOY_WINDOW - 1)   # needs a full year + YoY window behind it
rows = []
for origin in pd.date_range(first_origin, last - pd.Timedelta(days=BLOCK), freq=f'{ORIGIN_STEP}D'):
    hist = clean_to(origin)
    r_now = hist.iloc[-YOY_WINDOW:].sum()
    r_ly  = hist.reindex(hist.index[-YOY_WINDOW:] - pd.Timedelta(days=364)).sum()
    yoy_o, settled = r_now / r_ly, hist.index[-YOY_WINDOW]
    for h in range(1, MAX_BLOCKS + 1):
        win = pd.date_range(origin + pd.Timedelta(days=(h - 1) * BLOCK + 1), periods=BLOCK)
        if win[-1] > last:
            break
        ref = win - pd.Timedelta(days=364)
        ly = hist.reindex(ref)
        if ly.isna().any():
            continue
        pred = (ly.values * np.where(ref < settled, yoy_o, 1.0)).sum()
        act = df['clicks'].reindex(win).sum()
        if act > 0:
            rows.append(dict(horizon_months=h, origin=origin, forecast=pred, actual=act))

bt_fc = pd.DataFrame(rows)
assert len(bt_fc), "Not enough history for a backtest"
bt_fc['bias'] = (bt_fc['forecast'] - bt_fc['actual']) / bt_fc['actual']     # + = forecast too high
bt_fc['ape'] = bt_fc['bias'].abs()

res = bt_fc.groupby('horizon_months').agg(
    MAPE=('ape', 'mean'), bias=('bias', 'mean'),
    overshoot=('bias', lambda b: b.clip(lower=0).mean()),
    undershoot=('bias', lambda b: (-b).clip(lower=0).mean()),
    n_origins=('ape', 'size'))
H_MAX = int(res.index.max())
res = res.reindex(range(1, H_MAX + 1)).ffill()
res['down'] = res['overshoot'].cummax()            # error should not shrink as the horizon grows
res['up']   = res['undershoot'].cummax()
print(res.round(3).to_markdown())
if res['n_origins'].min() < 8:
    print('WARNING: fewer than 8 origins at some horizons, treat the MAPE as indicative only')

# band per day: no margin on days already observed
days_ahead = pd.Series((scen.index - last).days, index=scen.index)
h_day = np.ceil(days_ahead.clip(lower=1) / BLOCK).astype(int).clip(upper=H_MAX)
live = days_ahead > 0
down = np.where(live, h_day.map(res['down']), 0.0)
up   = np.where(live, np.minimum(h_day.map(res['up']), 0.9), 0.0)

band = pd.DataFrame(index=scen.index)
for s in ['baseline', 'low', 'base', 'high']:
    band[f'{s}_lo'] = scen[s] / (1 + down)         # forecast too high by e  ->  actual = forecast / (1 + e)
    band[f'{s}_hi'] = scen[s] / (1 - up)           # forecast too low by u   ->  actual = forecast / (1 - u)

mon = scen.resample('MS').sum()
mb  = band.resample('MS').sum()
margin_tbl = pd.DataFrame(index=mon.index.strftime('%Y-%m'))
for s in ['low', 'base', 'high']:
    margin_tbl[s.title()] = mon[s].values
    margin_tbl[f'{s.title()} lower'] = mb[f'{s}_lo'].values
    margin_tbl[f'{s.title()} upper'] = mb[f'{s}_hi'].values
margin_tbl['Downside %'] = (mb['base_lo'] / mon['base'] - 1).values
margin_tbl['Upside %']   = (mb['base_hi'] / mon['base'] - 1).values
max_h = days_ahead.where(live).resample('MS').max().apply(lambda d: np.ceil(d / BLOCK) if d == d else 0)
margin_tbl['Basis'] = np.where(max_h.values == 0, 'actuals', np.where(max_h.values <= H_MAX, 'measured', 'extrapolated'))
margin_tbl.index.name = 'Month'
print(margin_tbl.round(3).to_markdown())

# DATA VIZ
# ============================================================
SHIP_DATE = '2026-09-01'  # Set forecast date flag

plot_df = (pd.concat([
              df['clean_clicks'].tail(365).rename('Actual (cleaned)'),
              scen.rename(columns={'baseline': 'Baseline forecast', 'low': 'Low', 'base': 'Base', 'high': 'High'})
          ], axis=1)
          .rename_axis('date').reset_index()
          .melt('date', var_name='series', value_name='clicks').dropna())

fig = px.line(plot_df, x='date', y='clicks', color='series',
              title=f'River Cruises Clicks 12 months Forecast if rankings improved +{POSITION_GAIN}')
fig.update_traces(selector=dict(name='Baseline forecast'), line_dash='dash')
fig.add_shape(type='line', x0=SHIP_DATE, x1=SHIP_DATE, y0=0, y1=1, yref='paper', line=dict(dash='dot'))
fig.add_annotation(x=SHIP_DATE, y=1, yref='paper', text='Forecast date', showarrow=False, xanchor='left')
fig.update_layout(xaxis_title='Date', yaxis_title='Clicks', hovermode='x unified')
fig.show()


# CURRENT CTR vs CTR AFTER +3 POSITIONS
# ============================================================
from plotly.subplots import make_subplots

MAX_POS = 10   # page 1 only, where the curve is most reliable

curve = ctr_curve(curve_src, 0.5)
pos = np.arange(1, MAX_POS + 1)
ctr_now = curve.reindex(pos).values
ctr_new = curve.reindex(np.maximum(1, pos - POSITION_GAIN)).values
mult = ctr_new / ctr_now

# overall CTR across your page-queries: now vs after (impressions held constant)
overall_now = cur['clicks'].sum() / cur['impressions'].sum()
scen_ctr = {'Now': overall_now}
for _, r in assump.iterrows():
    scen_ctr[f"{r['scenario'].title()} (+{r['realisation']:.0%} of lift)"] = overall_now * r['blended_uplift']

fig3 = make_subplots(
    rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.1,
    subplot_titles=(f'CTR at each position: now vs after +{POSITION_GAIN} positions (full curve effect)',
                    'Overall non-brand CTR: now vs scenarios'))

fig3.add_trace(go.Bar(x=pos, y=ctr_now, name='Current CTR', marker_color='#9aa5b1',
                      hovertemplate='Position %{x}<br>Current CTR %{y:.1%}<extra></extra>'), row=1, col=1)
fig3.add_trace(go.Bar(x=pos, y=ctr_new, name=f'CTR after +{POSITION_GAIN}', marker_color='#2ca02c',
                      text=[f'x{v:.2f}' for v in mult], textposition='outside',
                      hovertemplate='From position %{x}<br>CTR after %{y:.1%}<extra></extra>'), row=1, col=1)

names = list(scen_ctr.keys())
fig3.add_trace(go.Bar(x=names, y=list(scen_ctr.values()), showlegend=False,
                      marker_color=['#9aa5b1', '#a5d6a7', '#66bb6a', '#2e7d32'][:len(names)],
                      text=[f'{v:.2%}' for v in scen_ctr.values()], textposition='outside',
                      hovertemplate='%{x}<br>CTR %{y:.2%}<extra></extra>'), row=1, col=2)

fig3.update_xaxes(title_text='Current position', dtick=1, row=1, col=1)
fig3.update_yaxes(title_text='CTR', tickformat='.0%', row=1, col=1)
fig3.update_yaxes(title_text='CTR', tickformat='.1%', row=1, col=2)
fig3.update_layout(title=f'River Cruises: CTR gained from improving rankings by +{POSITION_GAIN}',
                   barmode='group', height=480, legend=dict(orientation='h', y=-0.2))
fig3.show()

ctr_tbl = pd.DataFrame({'position': pos, 'current_ctr': ctr_now,
                        f'ctr_after_+{POSITION_GAIN}': ctr_new, 'multiplier': mult.round(2)})

mon = scen.resample('MS').sum().round(0)
mon.index = mon.index.strftime('%Y-%m')

month_tbl = pd.DataFrame({
    'Baseline': mon['baseline'], 'Low': mon['low'], 'Base': mon['base'], 'High': mon['high'],
    'Low incremental':  mon['low']  - mon['baseline'],
    'Base incremental': mon['base'] - mon['baseline'],
    'High incremental': mon['high'] - mon['baseline'],
    'Low uplift %':  mon['low']  / mon['baseline'] - 1,
    'Base uplift %': mon['base'] / mon['baseline'] - 1,
    'High uplift %': mon['high'] / mon['baseline'] - 1,
}).replace([np.inf, -np.inf], np.nan)
month_tbl.index.name = 'Month'

# Error margin from the backtest cell (measured for months 1-3, extrapolated after)
bounds = margin_tbl[['Low lower', 'Low upper', 'Base lower', 'Base upper', 'High lower', 'High upper']].round(0)
month_tbl = month_tbl.join(bounds)
month_tbl['Downside %'] = month_tbl['Base lower'] / month_tbl['Base'] - 1
month_tbl['Upside %']   = month_tbl['Base upper'] / month_tbl['Base'] - 1
month_tbl['Basis']      = margin_tbl['Basis']

# Total row (percentages recomputed, not summed)
sum_cols = ['Baseline', 'Low', 'Base', 'High', 'Low incremental', 'Base incremental', 'High incremental',
            'Low lower', 'Low upper', 'Base lower', 'Base upper', 'High lower', 'High upper']
totals = month_tbl[sum_cols].sum()
for s in ['Low', 'Base', 'High']:
    totals[f'{s} uplift %'] = totals[s] / totals['Baseline'] - 1 if totals['Baseline'] else np.nan
totals['Downside %'] = totals['Base lower'] / totals['Base'] - 1
totals['Upside %']   = totals['Base upper'] / totals['Base'] - 1
totals['Basis'] = 'mixed'
month_tbl.loc['Total'] = totals

print(month_tbl.round(3).to_markdown())

EXPORT = True
if EXPORT:
    curves = pd.DataFrame({'median': ctr_curve(curve_src, 0.5)})
    curves.index.name = 'position'

    with pd.ExcelWriter('forecast_scenarios.xlsx') as xw:
        month_tbl.to_excel(xw, sheet_name='month')
        assump.drop(columns='ship_date').to_excel(xw, sheet_name='assumptions', index=False)
        res.to_excel(xw, sheet_name='backtest_mape')
        scen.round(1).rename_axis('date').to_excel(xw, sheet_name='daily')
        curves.to_excel(xw, sheet_name='ctr_curves')

        ws = xw.sheets['month']
        for cell in ws[1]:                                   # format every '%' column by header name
            if isinstance(cell.value, str) and '%' in cell.value:
                for c in ws.iter_cols(min_row=2, min_col=cell.column, max_col=cell.column):
                    for x in c:
                        x.number_format = '0.0%'
        for i in range(1, ws.max_column + 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = 18
