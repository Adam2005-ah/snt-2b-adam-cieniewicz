import pandas as pd, numpy as np
D = pd.read_pickle('daily.pkl')
P = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt'
tb = pd.read_csv(f'{P}/us-etf/alm0421_macro/DTB3.csv', parse_dates=['date']).set_index('date')['value']
tb = pd.to_numeric(tb, errors='coerce')
bd = pd.date_range('1954-01-01','2026-12-31',freq='B')
tb_d = tb.reindex(bd).ffill()/100/252
def load(f):
    s = pd.read_csv(f'{P}/alternatives/pofo_indices/{f}', parse_dates=['date'], comment='#').set_index('date')['close']
    return s
sg = load('SG_Trend_Index_daily.csv'); cta = load('SG_CTA_Index_daily.csv')
btop = load('BTOP50_monthly.csv'); aqr = load('AQR_TSMOM_excess_index_monthly.csv')
# daily excess for SG indices
def daily_excess(lvl):
    r = lvl.pct_change().dropna()
    # cash per day accrued over calendar gap: use T-bill annual rate * days/360? use business-day accrual
    rate = tb.reindex(pd.date_range(lvl.index.min(), lvl.index.max())).ffill()/100
    gap = pd.Series(lvl.index, index=lvl.index).diff().dt.days.reindex(r.index)
    c = rate.reindex(r.index).values * gap.values / 365.0
    return r - c
sg_x = daily_excess(sg); cta_x = daily_excess(cta)
# monthly
tb_m = (1 + tb_d).resample('ME').prod() - 1
btop_r = btop.pct_change().dropna(); btop_r.index = btop_r.index.to_period('M').to_timestamp('M')
btop_x = btop_r - tb_m.reindex(btop_r.index)
aqr_r = aqr.pct_change().dropna(); aqr_r.index = aqr_r.index.to_period('M').to_timestamp('M')
aqr_x = aqr_r
def monthly(dx):  # daily excess -> monthly excess (compounded, approx)
    return (1+dx).resample('ME').prod()-1
v5m = monthly(D['v5_net'].loc['1990':]); v2m = monthly(D['v2_net'].loc['1990':])
def sr_d(x): x=x.dropna(); return x.mean()/x.std()*np.sqrt(252) if len(x)>50 else np.nan
def sr_m(x): x=x.dropna(); return x.mean()/x.std()*np.sqrt(12) if len(x)>12 else np.nan
def vol_d(x): x=x.dropna(); return x.std()*np.sqrt(252)
def vol_m(x): x=x.dropna(); return x.std()*np.sqrt(12)
END='2026-07-10'
windows = {'1990-1999':('1990','1999'),'2000-2009':('2000','2009'),'2010-2019':('2010','2019'),'2020-2026':('2020',END),
           '1990-2026 (full)':('1990',END),'since 2000':('2000',END),'since 2010':('2010',END),'since 2015':('2015',END),
           'since 2020':('2020',END),'since 2023':('2023',END),
           '1987-1989':('1987','1989'), '1985-1989':('1985','1989')}
rows=[]
for w,(a,b) in windows.items():
    r={'window':w}
    r['v5 SR (daily)']=sr_d(D['v5_net'].loc[a:b]); r['v2 SR (daily)']=sr_d(D['v2_net'].loc[a:b])
    r['v5 SR (monthly)']=sr_m(v5m.loc[a:b]); r['v2 SR (monthly)']=sr_m(v2m.loc[a:b])
    r['v5 ann excess']=D['v5_net'].loc[a:b].mean()*252; r['v5 vol']=vol_d(D['v5_net'].loc[a:b])
    r['v2 ann excess']=D['v2_net'].loc[a:b].mean()*252; r['v2 vol']=vol_d(D['v2_net'].loc[a:b])
    r['SG Trend SR (daily, net fees)']=sr_d(sg_x.loc[a:b]); r['SG Trend excess ann']=sg_x.loc[a:b].mean()*252 if len(sg_x.loc[a:b])>50 else np.nan
    r['SG Trend vol']=vol_d(sg_x.loc[a:b]) if len(sg_x.loc[a:b])>50 else np.nan
    r['SG CTA SR (daily, net fees)']=sr_d(cta_x.loc[a:b])
    r['BTOP50 SR (monthly, net fees)']=sr_m(btop_x.loc[a:b]); r['BTOP50 excess ann']=btop_x.loc[a:b].mean()*12; r['BTOP50 vol']=vol_m(btop_x.loc[a:b])
    r['AQR TSMOM SR (monthly, gross, no costs)']=sr_m(aqr_x.loc[a:b]); r['AQR TSMOM vol']=vol_m(aqr_x.loc[a:b])
    rows.append(r)
T = pd.DataFrame(rows).set_index('window')
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(T.round(3).T)
T.to_csv('sharpe_par_periode.csv')
# full history of each index
print('BTOP50 full', btop_x.index.min(), sr_m(btop_x), btop_x.mean()*12, vol_m(btop_x))
print('AQR full', aqr_x.index.min(), sr_m(aqr_x), aqr_x.mean()*12, vol_m(aqr_x))
print('SG trend full', sr_d(sg_x), 'SG CTA full', sr_d(cta_x))
# rolling 5y Sharpe (monthly evaluated, daily data)
def roll5(x):
    x=x.dropna(); m=x.rolling(1260).mean(); s=x.rolling(1260).std(); return (m/s*np.sqrt(252)).dropna()
res={}
for k,x in {'v5':D['v5_net'].loc['1990':], 'v2':D['v2_net'].loc['1990':], 'SG Trend (net)':sg_x}.items():
    r=roll5(x); r=r.resample('ME').last()
    res[k]={'n months':len(r),'min':r.min(),'p5':r.quantile(.05),'p10':r.quantile(.1),'p25':r.quantile(.25),'median':r.median(),
            'p75':r.quantile(.75),'p90':r.quantile(.9),'max':r.max(),'share<0':(r<0).mean(),'share<0.3':(r<0.3).mean(),
            'last':r.iloc[-1],'last date':r.index[-1].date(),
            'median since 2005':r.loc['2005':].median(),'median since 2015':r.loc['2015':].median(),'share<0 since 2010':(r.loc['2010':]<0).mean()}
R=pd.DataFrame(res); print(R); R.to_csv('sharpe_glissant_5ans.csv')
# yearly rolling for print
r5=roll5(D['v5_net'].loc['1990':]).resample('YE').last(); r2=roll5(D['v2_net'].loc['1990':]).resample('YE').last(); rs=roll5(sg_x).resample('YE').last()
Y=pd.DataFrame({'v5 5y SR':r5,'v2 5y SR':r2,'SG Trend 5y SR':rs}); Y.index=Y.index.year; print(Y.round(2))
Y.to_csv('sharpe_glissant_5ans_fin_annee.csv')
pd.to_pickle({'sg_x':sg_x,'cta_x':cta_x,'btop_x':btop_x,'aqr_x':aqr_x,'tb_d':tb_d,'tb_m':tb_m,'sg':sg,'cta':cta,'btop_r':btop_r}, 'indices.pkl')
