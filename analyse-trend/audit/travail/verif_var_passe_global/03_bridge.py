"""Bridge L1 -> L2 for v2 and v5 from the independent runs, compared with decomposition.csv; narrative checks."""
import pickle

import numpy as np
import pandas as pd

S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/verif_var_passe_global/'
PREV = S + 'audit/var_passe_global/'
exec(open(OUT + '02_metrics.py').read().split('rows = []')[0])   # R, TB, IBR, G5, met, build, PER, VN

dec = pd.read_csv(PREV + 'decomposition.csv')
P = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-11'}


def with_tb(df):
    d = df.loc['1990-01-01':].copy()
    d['cash'] = TB
    d['m'] = 0.0
    d['total'] = d['fut'] + TB
    d['excess'] = d['fut']
    return d


steps = {}
for v in ('v2', 'v5'):
    steps[(v, 'S0 L1 publié')] = with_tb(R[(v, 'L1')])
    steps[(v, 'S1 univers 82')] = with_tb(R[(v, 'S1')])
    steps[(v, 'S2 + exécution 0,5 j')] = with_tb(R[(v, 'S2')])
    steps[(v, 'S3 + coûts IBKR réels')] = with_tb(R[(v, 'L2')])
    steps[(v, 'S4 + cash IBKR (= L2)')] = build(v, 'L2')
rows = []
for (v, st), df in steps.items():
    for p, s in P.items():
        m = met(df, s)
        for k in ('sharpe', 'cagr_total', 'cost_pa'):
            ref = dec[(dec.variant == VN[v]) & (dec.step == st) & (dec.period == p) & (dec.metric == k)]['value']
            rows.append(dict(variant=v, step=st, period=p, metric=k, mine=m[k], prev=float(ref.iloc[0])))
b = pd.DataFrame(rows)
b['diff'] = b.mine - b.prev
print('bridge max |diff|:', b['diff'].abs().max())
pd.set_option('display.width', 250)
print((b[b.metric == 'cagr_total'].pivot_table(index=['variant', 'step'], columns='period', values='mine', sort=False) * 100).round(2))
print((b[b.metric == 'sharpe'].pivot_table(index=['variant', 'step'], columns='period', values='mine', sort=False)).round(3))
b.to_csv(OUT + 'compare_bridge.csv', index=False, float_format='%.6f')

# calendar years 2020/2021 for v5 by step (claim: L1 gap 2020-21 mostly from CUA1)
yrs_ = {}
for st in ['S0 L1 publié', 'S1 univers 82', 'S2 + exécution 0,5 j', 'S3 + coûts IBKR réels', 'S4 + cash IBKR (= L2)']:
    t = steps[('v5', st)]['total']
    yrs_[st] = ((1 + t).groupby(t.index.year).prod() - 1).loc[2019:2025]
print('\nv5 calendar years by step (%):')
print((pd.DataFrame(yrs_).T * 100).round(1).to_string())

# ranking claims at L2 (sharpe and cagr, all periods)
d = pd.read_csv(OUT + 'compare_var_passe_global.csv')
for lvl in ('L0 brut', 'L1 backtest', 'L2 réaliste'):
    for m in ('cagr_total', 'sharpe'):
        t = d[(d.level == lvl) & (d.metric == m)].pivot_table(index='variant', columns='period', values='mine')
        print('\n', lvl, m)
        print(t.round(4).to_string())
