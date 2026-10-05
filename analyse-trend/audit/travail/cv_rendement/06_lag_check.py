"""Independent check of the critic's execution-lag claim (one extra day: positions.shift(2) instead of shift(1))."""
import sys, pickle, numpy as np, pandas as pd
S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
sys.path.insert(0, S + 'audit/simulation')
import engine as E
subs = pickle.load(open(S + 'audit/simulation/subsets.pkl', 'rb'))
rows = []
for name, cols in (('v5_84', list(E.RETS.columns)), ('sub_100k', subs[('sub', 100000.0, 1)]), ('sub_1000k', subs[('sub', 1000000.0, 1)])):
    o = E.system(cols)
    pos, r = o['pos'], E.RETS[cols].fillna(0)
    cost = o['gross'] - o['net']
    for lag in (1, 2):
        net = (pos.shift(lag).fillna(0) * r).sum(axis=1) - cost
        for a in ('1990', '2000', '2010', '2015'):
            x = net.loc[a:'2026-07-10']
            rows.append(dict(series=name, lag=lag, start=a, sharpe=round(x.mean() / x.std() * np.sqrt(261), 3)))
T = pd.DataFrame(rows).pivot_table(index=['series', 'start'], columns='lag', values='sharpe')
T['delta'] = T[2] - T[1]
print(T.round(3)); T.to_csv('lag_check.csv')
