import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, trend, pickle
S='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
d=pickle.load(open('mine.pkl','rb')); rets=d['rets']; groups=d['groups']; risk=d['risk']
subs=pickle.load(open(S+'audit/simulation/subsets.pkl','rb'))
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
def sh(x,a): x=x.loc[a:'2026-07-10']; x=x[x.index>=x.ne(0).idxmax()]; return round(x.mean()/x.std()*16,2)
rows=[]
for (k,C,m),cols in sorted(subs.items(), key=lambda z:(z[0][2],z[0][1])):
    if C>1e6: continue
    v=trend.run_vol_targeted(rets[cols], risk[cols], {c:costs[c] for c in cols}, {c:roll[c] for c in cols}, forecast_fn=trend.forecast_regime)
    n=v['net']; cum=(1+n.loc['1990':]).cumprod()
    rows.append(dict(capital=C, rule=f'>={m}c', n=len(cols), sharpe_1990=sh(n,'1990'), sharpe_2000=sh(n,'2000'), sharpe_2010=sh(n,'2010'), maxdd_1990=round((cum/cum.cummax()-1).min(),3)))
print(pd.DataFrame(rows).to_string())
