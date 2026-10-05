import pandas as pd, numpy as np
r=pd.read_pickle('rets.pkl'); P=pd.read_pickle('pos.pkl'); g=pd.read_pickle('groups.pkl'); net=pd.read_pickle('net.pkl')
px=pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv',index_col=0,parse_dates=True)
ewm=r.ewm(span=32,min_periods=32).std()*16; y1=r.rolling(252,min_periods=126).std()*16
sig=np.maximum(ewm,y1).ffill()
# FX / bond k check from CME data (latest maintenance x1.1)
for t,m,mult in [('EC1 Curncy',2400*1.1,125000),('JY1 Curncy',2600*1.1,12.5e6),('TY1 Comdty',1875*1.1,1000)]:
    p=px[t].iloc[-1]; 
    notional = mult*p if t!='JY1 Curncy' else (mult*p if p<1 else mult*p/1e4)
    s=sig[t].loc['2025-07-10':'2026-07-10'].mean()
    print(t,'price',p,'notional',round(notional),'pct',round(m/notional,4),'sig',round(s,4),'k',round(m/notional/s,3))
kA={'Equities':0.40,'Bonds':0.37,'STIR':0.20,'FX':0.27,'Energy':0.21,'Metals':0.34,'Agriculture':0.23}   # previous agent
kC={'Equities':0.40,'Bonds':0.37,'STIR':0.20,'FX':0.27,'Energy':0.27,'Metals':0.24,'Agriculture':0.26}   # CME-corrected commodity k
kL={c:0.25 for c in kA}  # flat 0.25 sensitivity
gross=P.abs().sum(axis=1)
res={}
for lab,kc in [('B_prev_agent',kA),('B_corrected',kC),('B_flat_k0.25',kL)]:
    m=(P.abs()*sig*g.map(kc)).sum(axis=1).loc['2000':]
    res[lab]=m
D=pd.DataFrame(res); D['gross']=gross.loc['2000':]
def s(x): return pd.Series({'mean':x.mean(),'median':x.median(),'p95':x.quantile(.95),'p99':x.quantile(.99),'max':x.max(),'date_max':x.idxmax().date(),'share>0.3':(x>0.3).mean(),'share>0.4':(x>0.4).mean(),'share>0.5':(x>0.5).mean(),'2026-07-10':x.iloc[-1]})
S=D.apply(s).T; pd.set_option('display.width',250); print(S.to_string())
S.to_csv('margin_recomputed_summary.csv')
dates=['2009-09-16','2012-05-23','2014-09-03','2019-04-23','2020-03-16','2020-07-21','2023-03-10','2023-03-13','2025-09-22','2026-07-10']
pt=D.loc[[pd.Timestamp(x) for x in dates]].round(3); print(pt.to_string()); pt.to_csv('margin_spot_dates.csv')
# class breakdown at 2009-09-16 and 2026-07-10 under prev agent k
for dd in ['2009-09-16','2026-07-10']:
    row=(P.loc[dd].abs()*sig.loc[dd]*g.map(kA)); print(dd, row.groupby(g).sum().round(3).to_dict(), 'gross by class', P.loc[dd].abs().groupby(g).sum().round(2).to_dict())
# stress: 2x margin + realised next-5d loss
cum=(1+net).cumprod(); fwd5=(cum.shift(-5)/cum-1)
for lab in res:
    st=(2*D[lab]/(1+fwd5.reindex(D.index).clip(upper=0))).dropna()
    print(lab,'stress max',round(st.max(),3),st.idxmax().date(),'p99',round(st.quantile(.99),3), 'svb 2023-03-10',round(st.loc['2023-03-10'],3))
D.to_pickle('margin_recomputed.pkl')
