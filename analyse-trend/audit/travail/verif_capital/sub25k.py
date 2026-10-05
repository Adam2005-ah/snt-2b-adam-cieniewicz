import sys, pandas as pd, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation')
import engine as E
S = E.S
sp = pd.read_csv(S + 'audit/capital/specs.csv', index_col=0)
T = pd.read_csv(S + 'audit/capital/tiers.csv').set_index('capital_usd')
liq = sp['liquid_notional_usd']
def sh(x, a): x = x.loc[a:'2026-07-10']; x = x[x.index >= x.ne(0).idxmax()]; return round(x.mean() / x.std() * 16, 2)
for m in (1, 4):
    cols = T.loc[25e3, f'strict_subset_ge{m}c_markets'].split(' | '); dropped = []
    while True:
        s = E.system(cols)
        nc = s['pos'].loc['2023-07-10':'2026-07-10', cols].abs().median() * 25e3 / liq[cols]
        if (nc >= m).all() or len(cols) <= 1: break
        w = nc.idxmin(); dropped.append(w); cols = [c for c in cols if c != w]
    n = s['net']; cum = (1 + n.loc['1990':]).cumprod()
    print(m, len(cols), cols, 'dropped', dropped, 'contracts', nc.round(2).to_dict(), 'sharpe 1990/2000/2010', sh(n, '1990'), sh(n, '2000'), sh(n, '2010'), 'maxdd', round((cum / cum.cummax() - 1).min(), 3))
