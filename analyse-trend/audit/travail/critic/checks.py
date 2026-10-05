"""Completeness-critic checks on variant 5 (read-only on project files)."""
import sys
import numpy as np
import pandas as pd

A = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
OUT = A + 'critic/'
pos = pd.read_pickle(A + 'operations/v5_positions.pkl')
rets = pd.read_pickle(A + 'operations/rets.pkl')
net = pd.read_pickle(A + 'operations/v5_net.pkl')
groups = pd.read_pickle(A + 'operations/groups.pkl')
AN = 261


def sr(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(AN)


periods = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-10'}

# 1) stale-series scan since 2010
rows = []
for c in rets.columns:
    s = rets[c].loc['2010':].dropna()
    if len(s) < 500:
        s = rets[c].dropna()
    rows.append({'mkt': c, 'group': groups[c], 'n': len(s), 'zero_share': (s == 0).mean(),
                 'ac1': s.autocorr(1), 'ac1_abs_t': abs(s.autocorr(1)) * np.sqrt(len(s))})
stale = pd.DataFrame(rows).sort_values('ac1', ascending=False)
stale.to_csv(OUT + 'stale_scan.csv', index=False)
print('STALE SCAN (top by lag-1 autocorr, since 2010)')
print(stale.head(12).to_string(index=False))
print(stale.sort_values('zero_share', ascending=False).head(8).to_string(index=False))

# 2) P&L by class and by market (gross, positions decided t held t+1)
pnl = pos.shift(1) * rets.fillna(0)
cls = pnl.T.groupby(groups).sum().T
out = {}
for k, d in periods.items():
    out[k] = cls.loc[d:].mean() * AN * 100
bycls = pd.DataFrame(out)
bycls.loc['TOTAL gross'] = bycls.sum()
bycls.loc['costs'] = [-(net[['trading', 'roll']].sum(axis=1).loc[d:].mean() * AN * 100) for d in periods.values()]
print('\nGROSS P&L BY CLASS, % of capital per year')
print(bycls.round(2).to_string())
bycls.to_csv(OUT + 'pnl_by_class.csv')
mk = pd.DataFrame({k: pnl.loc[d:].mean() * AN * 100 for k, d in periods.items()})
mk.to_csv(OUT + 'pnl_by_market.csv')
print('\nTop/bottom markets since 2010 (gross %/yr):')
print(mk['2010'].sort_values().head(6).round(2).to_string())
print(mk['2010'].sort_values().tail(8).round(2).to_string())

# 3) execution-delay sensitivity (one extra day of lag; costs unchanged)
cost = net[['trading', 'roll']].sum(axis=1)
lag2 = (pos.shift(2) * rets.fillna(0)).sum(axis=1) - cost
base = net['net']
res = []
for k, d in periods.items():
    res.append({'from': k, 'sr_lag1': sr(base.loc[d:]), 'sr_lag2': sr(lag2.loc[d:]),
                'mean_lag1': base.loc[d:].mean() * AN * 100, 'mean_lag2': lag2.loc[d:].mean() * AN * 100})
lagdf = pd.DataFrame(res)
print('\nEXECUTION LAG (+1 day)')
print(lagdf.round(3).to_string(index=False))
lagdf.to_csv(OUT + 'lag_sensitivity.csv', index=False)

# 4) long-horizon vol: variance ratios and geometric drag
x = base.loc['1990':]
vr = {}
for h in [21, 63, 126, 252]:
    agg = x.rolling(h).sum().iloc[::h]
    vr[h] = agg.var() / (x.var() * h)
ann = (1 + x).groupby(x.index.year).prod() - 1
print('\nVARIANCE RATIOS v5 net excess since 1990:', {k: round(v, 2) for k, v in vr.items()})
print('daily-annualised vol %.3f ; vol of calendar-year returns %.3f' % (x.std() * np.sqrt(AN), ann.std()))
for k, d in periods.items():
    y = base.loc[d:]
    yrs = (y.index[-1] - y.index[0]).days / 365.25
    cagr = (1 + y).prod() ** (1 / yrs) - 1
    print(k, 'arith %.2f%%  CAGR %.2f%%  drag %.2f pts' % (y.mean() * AN * 100, cagr * 100, (y.mean() * AN - cagr) * 100))
