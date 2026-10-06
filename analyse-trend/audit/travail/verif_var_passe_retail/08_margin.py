"""(8) Margin: flat-k model on the 84-market fractional positions of each variant vs the task's shortcut
(17 % x gross exposure ratio to v5), and effect of the gap on the cash leg."""
import pickle
import numpy as np, pandas as pd
import indep as I
ALL = list(I.RETS.columns)
idx = I.RETS.loc['2000-01-01':'2026-07-10'].index
e = I.RETS.ewm(span=32, min_periods=32).std() * 16; y = I.RETS.rolling(252, min_periods=126).std() * 16
sig = np.maximum(e, y).ffill()
rows = {}
for v in I.SPEC:
    p = I.system(v, ALL)['pos']
    rows[v] = dict(gross=p.loc[idx].abs().sum(axis=1).mean(), margin84=(p.abs() * 0.25 * sig).sum(axis=1).loc[idx].mean())
R = pd.DataFrame(rows).T
R['shortcut_17pct_x_gross'] = 0.17 * R.gross / R.loc['v5', 'gross']
W = pd.read_csv(I.A + 'var_passe_retail/var_passe_retail_wide.csv')
m = W[(W.execution == 'half_day_lag') & (W.period == '2000-2026')].pivot_table(index='variant', columns='capital_usd', values='mean_margin_pct') / 100
R = R.join(m)
print(R.round(3).to_string())
# effect on cash leg of using the shortcut instead of the computed margin: (margin gap) x mean IB rate 2000-2026
ibr = (I.TBR - 0.005).clip(lower=0).loc['2000':].mean()
print('mean IB rate 2000-2026 (pct):', round(ibr * 100, 2), ' cash-leg effect of a 9-pt margin gap (pct/yr):', round(0.09 * ibr * 100, 3))
