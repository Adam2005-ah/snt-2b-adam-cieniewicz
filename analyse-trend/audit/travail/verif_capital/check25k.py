import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, trend, pickle
S='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
d=pickle.load(open('mine.pkl','rb')); rets=d['rets']; groups=d['groups']; risk=d['risk']
sp=pd.read_csv(S+'audit/capital/specs.csv',index_col=0)
t=pd.read_csv(S+'audit/capital/tiers.csv').set_index('capital_usd')
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
liq=sp['liquid_notional_usd']
for C in [25e3,50e3,100e3]:
    for m in [1,4]:
        cols=t.loc[C,f'strict_subset_ge{m}c_markets'].split(' | ')
        v=trend.run_vol_targeted(rets[cols], risk[cols], {c:costs[c] for c in cols}, {c:roll[c] for c in cols}, forecast_fn=trend.forecast_regime)
        med=v['positions'].loc['2023-07-10':'2026-07-10'].abs().median()
        nc=(med*C/liq[cols]).round(2)
        n=v['net']; s90=n.loc['1990':].mean()/n.loc['1990':].std()*16; s10=n.loc['2010':].mean()/n.loc['2010':].std()*16
        fails=nc[nc<m]
        print(int(C),m,len(cols),'sharpe90 %.2f sharpe10 %.2f'%(s90,s10),'FAIL:',fails.to_dict())
