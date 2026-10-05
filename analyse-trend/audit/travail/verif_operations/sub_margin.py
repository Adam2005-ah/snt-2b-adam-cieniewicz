import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, trend
rets=pd.read_pickle('rets.pkl'); groups=pd.read_pickle('groups.pkl')
SUB=['ES1 Index','VG1 Index','NO1 Index','HI1 Index','TU1 Comdty','TY1 Comdty','RX1 Comdty','DU1 Comdty','SFR5 Comdty','EC1 Curncy','JY1 Curncy','BP1 Curncy','AD1 Curncy','CL1 Comdty','NG1 Comdty','GC1 Comdty','HG1 Comdty','C 1 Comdty','S 1 Comdty','LC1 Comdty']
r=rets[SUB]; g=groups[SUB]; risk=g.replace({'Bonds':'Taux','STIR':'Taux'})
v=trend.run_vol_targeted(r,risk,{c: rb.COSTS[x] for c,x in g.items()},{c: 2*rb.COSTS[x]*rb.ROLLS_PER_YEAR[x] for c,x in g.items()},forecast_fn=trend.forecast_regime)
ewm=r.ewm(span=32,min_periods=32).std()*16; y1=r.rolling(252,min_periods=126).std()*16; sig=np.maximum(ewm,y1).ffill()
kA={'Equities':0.40,'Bonds':0.37,'STIR':0.20,'FX':0.27,'Energy':0.21,'Metals':0.34,'Agriculture':0.23}
m=(v['positions'].abs()*sig*g.map(kA)).sum(axis=1).loc['2000':]
print('20-mkt margin mean',round(m.mean(),3),'p95',round(m.quantile(.95),3),'max',round(m.max(),3),m.idxmax().date(),'last',round(m.iloc[-1],3))
