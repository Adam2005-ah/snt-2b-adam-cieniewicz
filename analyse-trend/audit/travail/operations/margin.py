import pandas as pd, numpy as np
r=pd.read_pickle('rets.pkl'); P=pd.read_pickle('v5_positions.pkl'); g=pd.read_pickle('groups.pkl')
net=pd.read_pickle('v5_net.pkl')['net']
ewm=r.ewm(span=32,min_periods=32).std()*16
y1=r.rolling(252,min_periods=126).std()*16
sig=np.maximum(ewm,y1).ffill()
kc={'Equities':0.40,'Bonds':0.37,'STIR':0.20,'FX':0.27,'Energy':0.21,'Metals':0.34,'Agriculture':0.23}
k=g.map(kc)
# approach A: fixed % notional (2025-26 levels)
fixed={'Equities':0.07,'STIR':0.002,'FX':0.025,'Energy':0.10,'Metals':0.09,'Agriculture':0.05}
bond={'TU1 Comdty':0.008,'DU1 Comdty':0.008,'FV1 Comdty':0.013,'OE1 Comdty':0.013,'US1 Comdty':0.045,'WN1 Comdty':0.06,'UB1 Comdty':0.06}
mA=pd.Series({c:(bond.get(c,0.02) if g[c]=='Bonds' else fixed[g[c]]) for c in P.columns})
held=P  # positions decided at t, held overnight t->t+1 : margin requirement at close of t
mB=(held.abs()*sig*k).sum(axis=1)
mBc=(held.abs()*sig).mul(k).groupby(g,axis=1).sum() if False else (held.abs()*sig*k).T.groupby(g).sum().T
mAser=(held.abs()*mA).sum(axis=1)
gross=held.abs().sum(axis=1)
# equity relative: margin/equity uses fraction of capital; capital updated daily in backtest (positions as fraction of current capital)
df=pd.DataFrame({'gross_exposure':gross,'margin_B_volscaled':mB,'margin_A_fixed_pct':mAser})
df=df.join(mBc.add_prefix('B_'))
df=df.loc['2000-01-01':]
df.to_csv('margin_daily_v5.csv')
def summ(s):
    return pd.Series({'mean':s.mean(),'median':s.median(),'p95':s.quantile(.95),'p99':s.quantile(.99),'max':s.max(),'date_max':s.idxmax().date()})
out=pd.DataFrame({c:summ(df[c]) for c in ['gross_exposure','margin_B_volscaled','margin_A_fixed_pct']}).T
print(out)
# top peaks (distinct episodes)
s=df['margin_B_volscaled']
peaks=[]
tmp=s.copy()
for i in range(8):
    d=tmp.idxmax(); peaks.append((d.date(),round(tmp[d],3), round(df.loc[d,'margin_A_fixed_pct'],3), round(df.loc[d,'gross_exposure'],1)))
    tmp[(tmp.index>d-pd.Timedelta(days=120))&(tmp.index<d+pd.Timedelta(days=120))]=np.nan
print('peaks B (date, B, A, gross):',peaks)
pk=pd.DataFrame(peaks,columns=['date','margin_B','margin_A','gross_exposure'])
# class breakdown at peak and on average
print((mBc.loc['2000':].mean()).round(3)); print(mBc.loc[pd.Timestamp(peaks[0][0])].round(3))
# yearly means
yr=df[['margin_B_volscaled','margin_A_fixed_pct','gross_exposure']].groupby(df.index.year).agg(['mean','max']).round(3)
print(yr)
yr.to_csv('margin_by_year_v5.csv')
out.to_csv('margin_summary_v5.csv'); pk.to_csv('margin_peaks_v5.csv',index=False)
# losses vs cushion
n=net.loc['2000':]
cum=(1+n).cumprod()
l1=n.nsmallest(5); l5=(cum/cum.shift(5)-1).nsmallest(15)
print('worst 1d',l1.round(4).to_dict())
# distinct 5d
w5=(cum/cum.shift(5)-1); t=w5.copy(); res=[]
for i in range(5):
    d=t.idxmin(); res.append((d.date(),round(t[d],4))); t[(t.index>d-pd.Timedelta(days=30))&(t.index<d+pd.Timedelta(days=30))]=np.nan
print('worst 5d',res)
w20=(cum/cum.shift(21)-1); print('worst 21d',w20.min(),w20.idxmin())
# stress: margin x2 (exchange hikes) + worst subsequent 5-day loss; equity after loss E=1+L; maintenance ~ initial/1.1
fwd5=(cum.shift(-5)/cum-1)
stress=pd.DataFrame({'mB':s,'fwd5':fwd5}).dropna()
stress['ratio_after']=2*stress.mB/(1+stress.fwd5.clip(upper=0))
print('stressed ratio (2x margin + realised next-5d loss) max',stress.ratio_after.max(),stress.ratio_after.idxmax(), 'p99',stress.ratio_after.quantile(.99))
print('share of days margin_B>0.5:',(s>0.5).mean(),' >0.4:',(s>0.4).mean(),' >0.3',(s>0.3).mean())
pd.DataFrame({'worst_1d':[str(l1.round(4).to_dict())],'worst_5d':[str(res)],'worst_21d':[w20.min()],'worst_21d_date':[w20.idxmin().date()],'stress_ratio_max':[stress.ratio_after.max()],'stress_ratio_date':[stress.ratio_after.idxmax().date()]}).to_csv('loss_vs_cushion_v5.csv',index=False)
