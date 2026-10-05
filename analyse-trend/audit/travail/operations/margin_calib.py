import pandas as pd, numpy as np
r=pd.read_pickle('rets.pkl')
ewm=r.ewm(span=32,min_periods=32).std()*16
y1=r.rolling(252,min_periods=126).std()*16
sig=np.maximum(ewm,y1)
# observed 2025-26 exchange initial margins (approx) as % notional at 2026-07 prices
obs={'ES1 Index':(0.07,'ES ~$22-28k on ~$380k (TradeZero/Clarus)'),
     'TY1 Comdty':(2062/109031,'ZN $2,062 (barchart)'),
     'RX1 Comdty':(2500/125550,'Bund EUR 2,120-2,890 (deltavalue)'),
     'SFR5 Comdty':(501/239700,'SR3 months 9-12 $501 (CME via search)'),
     'EC1 Curncy':(2970/143075,'6E $2,970 (deltavalue/ironbeam)'),
     'CL1 Comdty':(6750/71410,'CL ~$6,750 (benzinga; 2026 level uncertain)'),
     'GC1 Comdty':(0.09,'GC 9% of notional Feb 2026 (CME/investinglive)'),
     'C 1 Comdty':(1072/21975,'ZC $990-1,072 (barchart)'),
     'S 1 Comdty':(2200/59588,'ZS $2,200 (barchart)'),
     'LC1 Comdty':(3630/94080,'LE $3,630 (barchart)'),
     'KC1 Comdty':(10609/125343,'KC $10,609 (ICE notice 2025-09-17)')}
rows=[]
for t,(m,src) in obs.items():
    s=sig[t].loc['2025-07-10':'2026-07-10'].mean()
    rows.append(dict(ticker=t,margin_pct_notional=m,vol_ann_max_ewm32_1y=s,k=m/s,source=src))
df=pd.DataFrame(rows); print(df.to_string()); print('median k',df.k.median())
df.to_csv('margin_calibration.csv',index=False)
