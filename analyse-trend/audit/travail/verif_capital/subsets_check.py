import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, trend, pickle
S='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
d=pickle.load(open('mine.pkl','rb')); rets=d['rets']; groups=d['groups']; risk=d['risk']
sp=pd.read_csv(S+'audit/capital/specs.csv',index_col=0)
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
def run(cols):
    cols=list(cols)
    v=trend.run_vol_targeted(rets[cols], risk[cols], {c:costs[c] for c in cols}, {c:roll[c] for c in cols}, forecast_fn=trend.forecast_regime)
    return v
def sh(x,a,b='2026-07-10'):
    x=x.loc[a:b]; return round(x.mean()/x.std()*16,3)
strict=list(sp.index[sp['liquid_access'].eq('OK')])
res={}
for lab,cols in [('strict66',strict),('strict66+CUA1+SCO1',strict+['CUA1 Comdty','SCO1 Comdty']),('all84',list(rets.columns))]:
    v=run(cols); n=v['net']; res[lab]=v
    print(lab,len(cols),'1990',sh(n,'1990'),'2000',sh(n,'2000'),'2010',sh(n,'2010'))
v=res['all84']; p=v['positions']; g=(p.shift(1)*rets.fillna(0))
for t in ['CUA1 Comdty','SCO1 Comdty']:
    print(t,'gross contrib %/yr since 2010', round(g[t].loc['2010':].mean()*252*100,2), 'first valid', rets[t].first_valid_index())
# leave-out excluded markets other than these
excl=[c for c in rets.columns if c not in strict]
print('excluded 18:', excl)
print('gross contrib since 2010 of excluded (%/yr):', (g[excl].loc['2010':].mean()*252*100).round(2).sort_values().to_dict())
