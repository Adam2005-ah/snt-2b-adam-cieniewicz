import pandas as pd, numpy as np
OUT='./'
net=pd.read_pickle('v5_net.pkl')['net'].loc['1990-01-01':]
tb=pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/us-etf/alm0421_macro/DTB3.csv',parse_dates=['date']).set_index('date')['value']/100/252
tb=tb.reindex(pd.date_range('1954-01-01','2026-12-31',freq='B')).ffill().reindex(net.index).fillna(0)
tot=net+tb
def dd_table(r,label,thr=-0.2):
    c=(1+r).cumprod(); peak=c.cummax(); dd=c/peak-1
    rows=[]; i=0; idx=c.index
    in_dd=False
    for t in range(len(c)):
        if not in_dd and dd.iloc[t]<0:
            in_dd=True; start=t-1 if t>0 else 0
        if in_dd and dd.iloc[t]>=0:
            seg=dd.iloc[start:t+1]; trough=seg.idxmin()
            rows.append(dict(series=label,peak=idx[start].date(),trough=trough.date(),recovery=idx[t].date(),depth=seg.min(),
                             months_to_trough=(trough-idx[start]).days/30.44,months_underwater=(idx[t]-idx[start]).days/30.44,recovered=True))
            in_dd=False
    if in_dd:
        seg=dd.iloc[start:]; trough=seg.idxmin()
        rows.append(dict(series=label,peak=idx[start].date(),trough=trough.date(),recovery=None,depth=seg.min(),
                         months_to_trough=(trough-idx[start]).days/30.44,months_underwater=(idx[-1]-idx[start]).days/30.44,recovered=False))
    df=pd.DataFrame(rows)
    return df, dd
ex, ddx = dd_table(net,'excess of T-bill')
tt, ddt = dd_table(tot,'total return (incl. T-bill)')
big=pd.concat([ex[ex.depth<=-0.2], tt[tt.depth<=-0.2]])
pd.set_option('display.width',200)
print(big.round(3).to_string())
big.to_csv(OUT+'drawdowns_over20_v5.csv',index=False)
# longest underwater spells (any depth)
lon=pd.concat([ex.nlargest(6,'months_underwater'),tt.nlargest(6,'months_underwater')])
print(lon.round(3).to_string()); lon.to_csv(OUT+'longest_underwater_v5.csv',index=False)
# yearly
yr=pd.DataFrame({'excess':(1+net).groupby(net.index.year).prod()-1,'total':(1+tot).groupby(tot.index.year).prod()-1})
print(yr.round(3).T.to_string())
print('negative years excess',(yr.excess<0).sum(),'of',len(yr),' total',(yr.total<0).sum())
yr.to_csv(OUT+'years_v5.csv')
# current dd
print('current dd excess',ddx.iloc[-1],'total',ddt.iloc[-1], 'peak date excess', ((1+net).cumprod()).idxmax(), 'peak total',((1+tot).cumprod()).idxmax())
# since sept 2022
c=(1+net).cumprod(); print('from 2022-09-30 to end excess', c.iloc[-1]/c.loc[:'2022-09-30'].iloc[-1]-1)
c2=(1+tot).cumprod(); print('total', c2.iloc[-1]/c2.loc[:'2022-09-30'].iloc[-1]-1)
# rolling 3y and 5y returns: share negative
for yrs in [1,3,5,10]:
    w=int(252*yrs); roll=(c2/c2.shift(w))**(1/yrs)-1; rx=(c/c.shift(w))**(1/yrs)-1
    print(yrs,'y rolling: share total<0',round((roll<0).mean(),3),'excess<0',round((rx<0).mean(),3),'min total',round(roll.min(),3),roll.idxmin().date())
# fraction of time in dd > 10%/20%
print('time in dd>10%',(ddt<-0.1).mean(),' >20%',(ddt<-0.2).mean(),' >30%',(ddt<-0.3).mean())
# months
m=(1+tot).resample('ME').prod()-1; print('worst month',m.min(),m.idxmin(),'pct months negative',(m<0).mean())
