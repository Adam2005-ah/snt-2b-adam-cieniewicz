import pandas as pd, numpy as np
B='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/'
tb=pd.read_csv(B+'us-etf/alm0421_macro/DTB3.csv',parse_dates=['date']).set_index('date')['value'].dropna()
S={'SG Trend':(B+'alternatives/pofo_indices/SG_Trend_Index_daily.csv','close'),'SG CTA':(B+'alternatives/pofo_indices/SG_CTA_Index_daily.csv','close'),
   'DBMF':(B+'us-etf/alm0421/DBMF.csv','adj_close'),'KMLM':(B+'us-etf/alm0421/KMLM.csv','adj_close'),'AQMIX':(B+'us-etf/alm0421/AQMIX.csv','adj_close')}
rows=[]
for n,(f,c) in S.items():
    s=pd.read_csv(f,parse_dates=[0],index_col=0)[c].dropna().sort_index()
    s=s.loc[:'2026-07-10']
    d=s.pct_change().dropna()
    days=pd.Series(d.index,index=d.index).diff().dt.days.fillna(1)
    rf=(tb.reindex(tb.index.union(d.index)).ffill().reindex(d.index)/100*days/365)
    ex=d-rf
    yrs=(s.index[-1]-s.index[0]).days/365.25
    a=s.loc[:'2022-09-30'].iloc[-1]
    rows.append(dict(series=n,start=s.index[0].date(),end=s.index[-1].date(),cagr=(s.iloc[-1]/s.iloc[0])**(1/yrs)-1,vol=d.std()*np.sqrt(252),
        sharpe=ex.mean()/ex.std()*np.sqrt(252),maxdd=(s/s.cummax()-1).min(),ret_20220930_20260710=s.iloc[-1]/a-1,
        sharpe_2000=(lambda e: e.mean()/e.std()*np.sqrt(252))(ex.loc['2000':]), excess_cagr=(1+ex).prod()**(1/yrs)-1))
out=pd.DataFrame(rows); pd.set_option('display.width',250); print(out.round(4).to_string()); out.to_csv('funds_recheck.csv',index=False)
