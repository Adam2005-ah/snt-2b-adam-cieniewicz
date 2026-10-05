import pandas as pd, numpy as np
E='ext/'
d=pd.read_csv(E+'margin_history.csv',parse_dates=['effective_date'])
print(d.groupby('product').tier_code.unique())
r=pd.read_pickle('rets.pkl')
ewm=r.ewm(span=32,min_periods=32).std()*16
y1=r.rolling(252,min_periods=126).std()*16
sig=np.maximum(ewm,y1)
spec={'GC':('GC1 Comdty','GC_F',100,1),'SI':('SI1 Comdty','SI_F',5000,1),'HG':('HG1 Comdty','HG_F',25000,1),
      'CL':('CL1 Comdty','CL_F',1000,1),'ZC':('C 1 Comdty','ZC_F',5000,0.01),'ZS':('S 1 Comdty','ZS_F',5000,0.01)}
rows=[]; ts={}
for p,(t,f,mult,sc) in spec.items():
    x=d[d['product']==p].sort_values('effective_date').drop_duplicates('effective_date',keep='last').set_index('effective_date')['maintenance_margin']
    px=pd.read_parquet(E+f+'.parquet')['close']*sc
    idx=sig.loc['2020-07-01':'2026-06-30'].index
    m=x.reindex(idx.union(x.index)).ffill().reindex(idx)*1.1   # initial = 110% maintenance (CME spec accounts)
    pr=px.reindex(idx.union(px.index)).ffill().reindex(idx)
    pct=m/(pr*mult); k=pct/sig[t].reindex(idx)
    ts[p]=k
    last=idx[-1]
    rows.append(dict(product=p,init_margin_2026_06_30=m.iloc[-1],price=pr.iloc[-1],pct_notional_2026_06=pct.iloc[-1],
        sig_2026_06=sig[t].iloc[sig.index.get_loc(last)],k_2026_06=k.iloc[-1],
        k_mean_2025_07_2026_07=k.loc['2025-07-10':].mean(),k_median_2020_2026=k.median(),k_p10=k.quantile(.1),k_p90=k.quantile(.9),
        pct_notional_max=pct.max(),pct_max_date=pct.idxmax().date(),pct_min=pct.min()))
out=pd.DataFrame(rows); pd.set_option('display.width',250); print(out.round(4).to_string())
out.to_csv('margin_k_calibration_cme.csv',index=False)
K=pd.DataFrame(ts); print(K.resample('YE').median().round(3))
# 2020 march: ratio of margin max in Mar-Apr 2020 to Feb 2020
for p in spec:
    x=d[d['product']==p].sort_values('effective_date').set_index('effective_date')['maintenance_margin']
    a=x.loc[:'2020-02-28']; 
