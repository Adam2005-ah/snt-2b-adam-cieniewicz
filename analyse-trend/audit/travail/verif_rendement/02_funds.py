import pandas as pd, numpy as np
P='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt'
tb=pd.read_csv(f'{P}/us-etf/alm0421_macro/DTB3.csv',parse_dates=['date']).set_index('date')['value']
tb=pd.to_numeric(tb,errors='coerce').dropna()
# convert discount yield to bond-equivalent-ish simple annual rate: r = d/(1-d*91/360) on 360 basis -> 365
d=tb/100; bey = 365*d/(360-91*d)
cal=pd.date_range('1954-01-01','2026-12-31',freq='D')
rate=bey.reindex(cal).ffill()
daily_cash = (1+rate)**(1/365)-1  # per calendar day
cashidx=(1+daily_cash).cumprod()
def load(f):
    return pd.read_csv(f'{P}/alternatives/pofo_indices/{f}',parse_dates=['date'],comment='#').set_index('date')['close']
sg=load('SG_Trend_Index_daily.csv'); cta=load('SG_CTA_Index_daily.csv'); btop=load('BTOP50_monthly.csv'); aqr=load('AQR_TSMOM_excess_index_monthly.csv')
# prepend base 1000 on 1999-12-31
sg=pd.concat([pd.Series({pd.Timestamp('1999-12-31'):1000.0}),sg]); cta=pd.concat([pd.Series({pd.Timestamp('1999-12-31'):1000.0}),cta])
def excess_from_level(lvl):
    r=lvl.pct_change().dropna()
    c=(cashidx.reindex(lvl.index).pct_change()).reindex(r.index)
    return (r-c)
def mexcess(lvl):
    m=lvl.resample('ME').last().dropna()
    cm=cashidx.reindex(cal).resample('ME').last()
    r=m.pct_change().dropna(); c=cm.pct_change().reindex(r.index)
    return r-c, r, c
sgx=excess_from_level(sg); sgm,sgm_tot,cm=mexcess(sg)
ctam,_,_=mexcess(cta)
btm,btm_tot,_=mexcess(btop)
aqm=aqr.resample('ME').last().pct_change().dropna()
def srd(x,ppy=252): return x.mean()/x.std()*np.sqrt(ppy)
END='2026-07-10'
rows=[]
for w,(a,b) in {'since 2000':('2000','2026-07'),'since 2010':('2010','2026-07'),'since 2015':('2015','2026-07'),'since 2023':('2023','2026-07'),'2000-2009':('2000','2009'),'full available':('1980','2026-12')}.items():
    x=sgx.loc[a:b]; m=sgm.loc[a:b]
    rows.append(dict(window=w,
      SG_daily_SR_252=srd(x), SG_daily_SR_rows=srd(x, len(x)/((x.index[-1]-x.index[0]).days/365.25)),
      SG_monthly_SR=srd(m,12), SG_arith_excess=m.mean()*12, SG_geo_excess=(1+m).prod()**(12/len(m))-1, SG_vol_m=m.std()*np.sqrt(12),
      CTA_monthly_SR=srd(ctam.loc[a:b],12), BTOP_monthly_SR=srd(btm.loc[a:b],12), AQR_monthly_SR=srd(aqm.loc[a:b],12)))
R=pd.DataFrame(rows).set_index('window'); pd.set_option('display.width',250); print(R.round(3).T)
print('BTOP start', btm.index[0], 'AQR start', aqm.index[0], 'BTOP full SR', srd(btm,12), 'AQR full', srd(aqm,12))
print('rows/yr SG', len(sgx)/((sgx.index[-1]-sgx.index[0]).days/365.25))
pd.to_pickle(dict(sgm=sgm, sgm_tot=sgm_tot, cm=cm, btm=btm, btm_tot=btm_tot, ctam=ctam, aqm=aqm, sgx=sgx, cashidx=cashidx), 'funds.pkl')
