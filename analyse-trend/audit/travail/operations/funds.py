import pandas as pd, numpy as np
B='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/'
tb=pd.read_csv(B+'us-etf/alm0421_macro/DTB3.csv',parse_dates=['date']).set_index('date')['value']/100/252
tb=tb.reindex(pd.date_range('1954-01-01','2026-12-31',freq='B')).ffill()
S={}
for n,f,c in [('SG Trend',B+'alternatives/pofo_indices/SG_Trend_Index_daily.csv','close'),('SG CTA',B+'alternatives/pofo_indices/SG_CTA_Index_daily.csv','close'),
              ('DBMF (US ETF)',B+'us-etf/alm0421/DBMF.csv','adj_close'),('KMLM (US ETF)',B+'us-etf/alm0421/KMLM.csv','adj_close'),('AQMIX',B+'us-etf/alm0421/AQMIX.csv','adj_close')]:
    s=pd.read_csv(f,parse_dates=[0]).set_index(pd.read_csv(f).columns[0]) if False else pd.read_csv(f,parse_dates=[0],index_col=0)[c].dropna()
    S[n]=s
net=pd.read_pickle('v5_net.pkl')['net']; tot=(net+tb.reindex(net.index).fillna(0))
S['Variant 5 backtest (gross of tax, net of modelled costs)']=(1+tot).cumprod()
sub=pd.read_pickle('v5sub_net.pkl'); S['Variant 5, 20 markets']=(1+sub+tb.reindex(sub.index).fillna(0)).cumprod()
rows=[]
for n,s in S.items():
    s=s.dropna()
    d=s.pct_change().dropna()
    def per(a,b):
        x=s.loc[a:b]; 
        if len(x)<20 or x.index[0]>pd.Timestamp(a)+pd.Timedelta(days=40): return np.nan
        yrs=(x.index[-1]-x.index[0]).days/365.25; return (x.iloc[-1]/x.iloc[0])**(1/yrs)-1
    ex=d-tb.reindex(d.index).ffill().fillna(0)
    rows.append(dict(series=n,start=s.index[0].date(),end=s.index[-1].date(),
        cagr_full=(s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1,
        sharpe_full=ex.mean()/ex.std()*np.sqrt(252), vol=d.std()*np.sqrt(252),
        maxdd=(s/s.cummax()-1).min(),
        cagr_2000_2026=per('2000-01-01','2026-07-10'), cagr_2010_2026=per('2010-01-01','2026-07-10'),
        cagr_2020_2026=per('2020-01-01','2026-07-10'),
        ret_2022_09_30_to_2026_07_10=(s.loc[:'2026-07-10'].iloc[-1]/s.loc[:'2022-09-30'].iloc[-1]-1) if s.index[0]<pd.Timestamp('2022-09-30') else np.nan,
        **{f'y{y}':(s.loc[str(y)].iloc[-1]/s.loc[:str(y-1)].iloc[-1]-1) if s.index[0]<pd.Timestamp(f'{y}-01-05') and len(s.loc[str(y)])>0 else np.nan for y in range(2019,2027)}))
df=pd.DataFrame(rows); pd.set_option('display.width',250)
print(df.round(3).to_string()); df.to_csv('funds_vs_v5_track_records.csv',index=False)
