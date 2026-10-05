import pandas as pd, numpy as np
D='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/alternatives/futures_sepp/'
px=pd.read_csv(D+'futures84_backadjusted_prices_daily.csv',index_col=0,parse_dates=True)
r=pd.read_csv(D+'futures84_usd_returns_daily.csv',index_col=0,parse_dates=True)
print(px.shape, r.shape, px.index[0], px.index[-1], (px.columns==r.columns).all())
px=px.loc['1985':]; r=r.reindex(px.index)
dp=px.diff()
# implied actual level: rolling regression of dP on r (no intercept), 63d
num=(dp*r).rolling(126,min_periods=40).sum(); den=(r*r).rolling(126,min_periods=40).sum()
L=num/den
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
rows=[]
for c in px.columns:
    p=px[c].dropna()
    if len(p)==0: continue
    rows.append(dict(t=c, start=p.index[0].date(), last=p.iloc[-1], L_last=L[c].dropna().iloc[-1] if L[c].notna().any() else np.nan,
       p2000=px[c].asof(pd.Timestamp('2000-01-03')), L2000=L[c].asof(pd.Timestamp('2000-01-03')),
       p2010=px[c].asof(pd.Timestamp('2010-01-04')), L2010=L[c].asof(pd.Timestamp('2010-01-04')),
       minp2000=px[c].loc['2000':].min(), minp1990=px[c].loc['1990':].min()))
print(pd.DataFrame(rows).round(3).to_string())
