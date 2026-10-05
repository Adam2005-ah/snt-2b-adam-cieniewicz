"""Tasks 1, 2, 5: v5 Sharpe without artefact markets, with execution lags; leave-one-out ranking."""
import numpy as np, pandas as pd, core
ALL = list(core.rets.columns)
TIER_B = ['LH1 Comdty', 'FC1 Comdty', 'CT1 Comdty', 'RS1 Comdty', 'SFI5 Comdty', 'IK1 Comdty', 'MWE1 Comdty']
SETS = {
    'all84': [],
    'ex_CUA1': ['CUA1 Comdty'],
    'ex_CUA1_SCO1': ['CUA1 Comdty', 'SCO1 Comdty'],
    'ex_CUA1_tierB': ['CUA1 Comdty'] + TIER_B,
    'ex_CUA1_SCO1_tierB': ['CUA1 Comdty', 'SCO1 Comdty'] + TIER_B,
}
LAGS = [1.0, 1.25, 1.5, 2.0]
rows = []
nets = {}
for lab, drop in SETS.items():
    cols = [c for c in ALL if c not in drop]
    for L in LAGS:
        res = core.v5(cols, lag=L)
        nets[(lab, L)] = res['net']
        r = dict(universe=lab, n=len(cols), lag=L)
        r.update({f'SR_{k}': v for k, v in core.sharpes(res['net']).items()})
        r.update({f'SR16_{k}': v for k, v in core.sharpes(res['net'], an=256).items()})
        r['mean2010_pct'] = res['net'].loc['2010':].mean() * 261 * 100
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(core.OUT + 'sharpe_by_universe_and_lag.csv', index=False, float_format='%.4f')
pd.to_pickle(nets, core.OUT + 'nets_universe_lag.pkl')
pd.set_option('display.width', 250)
print(df[['universe', 'n', 'lag'] + [c for c in df.columns if c.startswith('SR_')] + ['mean2010_pct']].round(3).to_string(index=False))

# critic-style lag test (pos.shift(2), same costs, scale from lag-1) for reference
b = core.v5(ALL)
crit = (b['positions'].shift(2) * core.rets.fillna(0)).sum(axis=1) - b['cost']
print('critic-style lag2 (no scale re-estimation):', {k: round(v, 3) for k, v in core.sharpes(crit).items()})

# leave-one-out since 2010 / 2000 / 1990
base = core.sharpes(b['net'])
loo = []
for c in ALL:
    n = core.v5([x for x in ALL if x != c])['net']
    s = core.sharpes(n)
    loo.append(dict(mkt=c, group=core.groups[c], **{f'dSR_{k}': s[k] - base[k] for k in ['1990', '2000', '2010', '2015', '2023-07']}))
loo = pd.DataFrame(loo).sort_values('dSR_2010')
loo.to_csv(core.OUT + 'leave_one_out.csv', index=False, float_format='%.4f')
print('LEAVE-ONE-OUT: change in Sharpe when the market is removed (most negative = biggest contributor)')
print(loo.head(12).round(3).to_string(index=False))
print(loo.tail(6).round(3).to_string(index=False))
print('dSR_2010 distribution: median %.3f, 5th pct %.3f, std %.3f' % (loo.dSR_2010.median(), loo.dSR_2010.quantile(.05), loo.dSR_2010.std()))
