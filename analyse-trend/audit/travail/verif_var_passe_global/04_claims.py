"""Narrative claims: (a) per-market cost contributions of v5 at L2 (is ER4 the largest item?);
(b) 2020-2021 gap: drop CUA1 only vs SCO1 only (v5, 84-1 markets, lag 1, bp costs); (c) margin-share proxy for
v2 vs asset-mix margin (couts/margin_estimate.csv)."""
import importlib.util
import sys

import numpy as np
import pandas as pd

S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/verif_var_passe_global/'
spec = importlib.util.spec_from_file_location('r1', OUT + '01_runs.py')
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
trend = r1.trend

# (a) v5 L2 positions -> per-market cost
cols = r1.C82
r = r1.rets[cols]
g = r1.risk[cols]
zero = {c: 0.0 for c in cols}
base = trend.run_portfolio(r, g, zero, zero, forecast_fn=trend.forecast_regime)
o = r1.lagged_pnl(base['positions'], r, r1.REAL[cols], r1.REAL_ROLL[cols])
scale = (0.20 / (o['fut'].ewm(span=32, min_periods=32).std() * 16)).clip(0.5, 2.0).shift(1)
pos = trend.run_portfolio(r, g, zero, zero, forecast_fn=trend.forecast_regime, scale=scale)['positions']
held = 0.5 * pos.shift(1).fillna(0) + 0.5 * pos.shift(2).fillna(0)
tr = pos.diff().abs().fillna(pos.abs())
yrs = lambda ix: (ix[-1] - ix[0]).days / 365.25 + 1 / 261
out = {}
for start in ('1990-01-01', '2010-01-01'):
    h, t = held.loc[start:], tr.loc[start:]
    y = yrs(h.index)
    c = (t * r1.REAL[cols]).sum() / y + (h.abs() * r1.REAL_ROLL[cols] / 252).sum() / y
    out[start[:4]] = c
cm = pd.DataFrame(out) * 100
cm['mean_abs_pos_1990'] = held.loc['1990':].abs().mean()
cm = cm.sort_values('1990', ascending=False)
print('v5 L2 cost by market, %/yr (top 10):')
print(cm.head(10).round(4).to_string())
print('total', cm['1990'].sum().round(4), cm['2010'].sum().round(4))
cm.to_csv(OUT + 'v5_L2_cost_by_market.csv', float_format='%.5f')

# (b) 2020/2021 decomposition: v5 at lag 1, bp costs, drop CUA1 only / SCO1 only
bp, bproll = pd.Series(r1.costs), pd.Series(r1.roll)
res = {}
for lab, drop in (('84', []), ('-CUA1', ['CUA1 Comdty']), ('-SCO1', ['SCO1 Comdty']), ('-both', r1.DROP)):
    cc = [c for c in r1.rets.columns if c not in drop]
    x = r1.l2('v5', cols=cc, lag=1.0, cst=bp, rol=bproll)
    tb = pd.read_pickle(OUT + 'series_verif.pkl')[('v5', 'L1')]['cash']
    tot = (x['fut'].loc['1990':] + tb)
    yy = (1 + tot).groupby(tot.index.year).prod() - 1
    s10 = x['fut'].loc['2010':]
    res[lab] = pd.concat([yy.loc[2019:2025], pd.Series({'sharpe2010': s10.mean() / s10.std() * 16,
                                                        'cagr_total2010': (1 + tot.loc['2010':]).prod() ** (1 / yrs(tot.loc['2010':].index)) - 1})])
print('\nv5 lag 1 bp costs, calendar years and 2010+ by universe:')
print(pd.DataFrame(res).round(4).to_string())
pd.DataFrame(res).to_csv(OUT + 'v5_drop_one_by_one.csv', float_format='%.5f')
