import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, trend
rets=pd.read_pickle('rets.pkl'); groups=pd.read_pickle('groups.pkl')
SUB=['ES1 Index','VG1 Index','NO1 Index','HI1 Index','TU1 Comdty','TY1 Comdty','RX1 Comdty','DU1 Comdty','SFR5 Comdty',
     'EC1 Curncy','JY1 Curncy','BP1 Curncy','AD1 Curncy','CL1 Comdty','NG1 Comdty','GC1 Comdty','HG1 Comdty','C 1 Comdty','S 1 Comdty','LC1 Comdty']
r=rets[SUB]; g=groups[SUB]; risk=g.replace({'Bonds':'Taux','STIR':'Taux'})
costs={c: rb.COSTS[x] for c,x in g.items()}; roll={c: 2*rb.COSTS[x]*rb.ROLLS_PER_YEAR[x] for c,x in g.items()}
v=trend.run_vol_targeted(r,risk,costs,roll,forecast_fn=trend.forecast_regime)
n=v['net']; full=pd.read_pickle('net.pkl')
for s in ['1990','2000','2010','2023']:
    for lab,x in [('20',n),('84',full)]:
        d=x.loc[s:]; print(lab,s,round(d.mean()/d.std()*16,3))
tb=pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/us-etf/alm0421_macro/DTB3.csv',parse_dates=['date']).set_index('date')['value']/100/252
tb=tb.reindex(pd.date_range('1954-01-01','2026-12-31',freq='B')).ffill()
for lab,x in [('20',n),('84',full)]:
    c=(1+x+tb.reindex(x.index).fillna(0)).cumprod(); print(lab,'total 2022-09-30->end',round(c.iloc[-1]/c.loc[:'2022-09-30'].iloc[-1]-1,4))
    c=(1+x).cumprod(); dd=c/c.cummax()-1; print(lab,'excess maxDD since 2000',round(dd.loc['2000':].min(),3))
# weekly
risk84=groups.replace({'Bonds':'Taux','STIR':'Taux'}); costs84={c: rb.COSTS[x] for c,x in groups.items()}; roll84={c: 2*rb.COSTS[x]*rb.ROLLS_PER_YEAR[x] for c,x in groups.items()}
w=trend.run_vol_targeted(rets,risk84,costs84,roll84,freq='W',forecast_fn=trend.forecast_regime)
for lab,x in [('weekly',w),('daily',None)]:
    if x is None: d=full.loc['1990':]
    else: d=x['net'].loc['1990':]
    print(lab,'sharpe 1990-',round(d.mean()/d.std()*16,3))
print('weekly costs/yr', round(((w['trading_costs']+w['roll_costs']).loc['1990':].mean()*252),4))
