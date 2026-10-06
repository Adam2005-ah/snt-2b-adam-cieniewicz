"""(6) spot-check remaining claims: published-84 CAGR (T-bill) since 2000/2010 for all 5 variants with my engine,
gold contribution for v5_100k 2023-07..2026-07, exact values behind rounded table cells, ER4-patched audit 1M Sharpe."""
import pickle
import numpy as np, pandas as pd
import indep as I
ALL = list(I.RETS.columns)
TBf = I.TB  # on IDX (1999+)
for v in I.SPEC:
    n = I.system(v, ALL)['net'].reindex(I.IDX)
    out = []
    for a in ('2000-01-01', '2010-01-01'):
        x = (n + TBf).loc[a:'2026-07-10']; out.append(round(I.cagr(x) * 100, 2))
    print('published84', v, 'CAGR total T-bill since 2000 / 2010:', out)
MY = pickle.load(open(I.HERE + 'subsets_mine.pkl', 'rb'))
d = I.simulate('v5', MY[('v5', 100e3)], 100e3, 1.5)
held = (0.5 * d['frac'].shift(1) + 0.5 * d['frac'].shift(2)).fillna(0)
pnl = (held * I.RETS.reindex(I.IDX)[MY[('v5', 100e3)]].fillna(0)).loc['2023-07-10':'2026-07-10']
yrs = (pd.Timestamp('2026-07-10') - pd.Timestamp('2023-07-10')).days / 365.25
print('v5_100k gross pnl %/yr 2023-07..2026-07, top:', (pnl.sum() / yrs * 100).sort_values().round(2).tail(3).to_dict(), 'total', round(pnl.sum().sum() / yrs * 100, 2))
W = pd.read_csv(I.A + 'var_passe_retail/var_passe_retail_wide.csv')
q = W[(W.execution == 'half_day_lag') & (W.key.isin(['v3_250k', 'v4_250k'])) & (W.period == '2000-2026')]
print(q[['key', 'cagr_total_net_all_fees', 'ref_published84_cagr_total_tbill']].to_string())
e = pd.read_csv(I.A + 'verif_capital/er4_spread_sensitivity.csv')
print(e[(e.universe == 'subset_ge1c')][['case', 'capital_usd', 'period', 'sharpe_net_real']].to_string())
