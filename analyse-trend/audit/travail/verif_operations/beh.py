import pandas as pd, numpy as np
net=pd.read_pickle('net.pkl').loc['1990-01-01':]
tb=pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/us-etf/alm0421_macro/DTB3.csv',parse_dates=['date']).set_index('date')['value'].dropna()
# daily accrual by calendar days between net dates (alternative to /252)
tbr=tb.reindex(tb.index.union(net.index)).ffill().reindex(net.index)
days=pd.Series(net.index,index=net.index).diff().dt.days.fillna(1)
tbd=tbr/100*days/360  # T-bill discount-ish, act/360
tot=net+tbd
for lab,r in [('excess',net),('total',tot)]:
    c=(1+r).cumprod(); dd=c/c.cummax()-1
    print(lab,'maxDD',round(dd.min(),4),dd.idxmin().date(),'current dd',round(dd.iloc[-1],4),'peak',c.idxmax().date(), 'CAGR', round(c.iloc[-1]**(365.25/(r.index[-1]-r.index[0]).days)-1,4))
    print('  share time dd< -10/-20/-30%:',[round((dd<x).mean(),3) for x in (-0.1,-0.2,-0.3)])
    y=(1+r).groupby(r.index.year).prod()-1
    print('  neg years',(y<0).sum(),'of',len(y), y.loc[2016:].round(3).to_dict())
    m=(1+r).resample('ME').prod()-1; print('  worst month',round(m.min(),4),m.idxmin().date(),'neg months',round((m<0).mean(),3))
    for yrs in [1,3,5,10]:
        w=int(round(252*yrs)); 
        # use calendar offset instead
        cc=c.copy(); prev=cc.reindex(cc.index-pd.DateOffset(years=yrs),method='ffill').values
        rr=pd.Series((cc.values/prev)**(1/yrs)-1,index=cc.index)[cc.index>=cc.index[0]+pd.DateOffset(years=yrs)]
        print('  rolling',yrs,'y share<0',round((rr<0).mean(),3),'min',round(rr.min(),4),rr.idxmin().date())
    c0=c.loc[:'2022-09-30'].iloc[-1]; print('  2022-09-30->end',round(c.iloc[-1]/c0-1,4))
# vol by year
print((net.groupby(net.index.year).std()*16).round(3).to_dict())
print('since 2023 sharpe',net.loc['2023':].mean()/net.loc['2023':].std()*16, 'since 2010 CAGR', (1+tot.loc['2010':]).prod()**(365.25/(tot.index[-1]-pd.Timestamp('2010-01-01')).days)-1)
print('worst days',net.nsmallest(6).round(4).to_dict())
c=(1+net).cumprod()
w5=c/c.shift(5)-1; print('worst 5d',w5.nsmallest(3).round(4).to_dict())
w21=c/c.shift(21)-1; print('worst 21d',w21.min(),w21.idxmin())
