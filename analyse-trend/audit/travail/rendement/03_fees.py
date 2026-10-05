import pandas as pd, numpy as np
D = pd.read_pickle('daily.pkl'); I = pd.read_pickle('indices.pkl')
sg, cta, btop_r, tb_m = I['sg'], I['cta'], I['btop_r'], I['tb_m']
END='2026-07-10'
def gross_up_monthly(net_tot, cash_m, mgmt, perf):
    """Invert fees month by month: mgmt accrues monthly on NAV, incentive accrued monthly on gains above HWM
    (crystallised annually in reality; monthly accrual with running HWM is the usual NAV practice).
    net_tot: monthly total (funded) net return. Returns monthly gross total return series."""
    nav_net = 1.0; hwm = 1.0; gross=[]
    for t, rn in net_tot.items():
        # gross assets before fees: A1 = nav_net*(1+g); after mgmt: A1 - nav_net*mgmt/12
        # incentive: perf*max(0, A1 - mgmt - hwm) ; net nav = A1 - mgmt - inc = nav_net*(1+rn)
        target = nav_net*(1+rn); m = nav_net*mgmt/12
        # solve for A1: if target > hwm => A1 - m - perf*(A1-m-hwm) = target => A1 = (target - perf*hwm)/(1-perf) + m
        if target > hwm:
            A1 = (target - perf*hwm)/(1-perf) + m
            hwm = target
        else:
            A1 = target + m
        gross.append(A1/nav_net - 1); nav_net = target
    return pd.Series(gross, index=net_tot.index)
def msr(x): x=x.dropna(); return x.mean()/x.std()*np.sqrt(12)
rows=[]
sg_m = sg.resample('ME').last().pct_change().dropna()
cta_m = cta.resample('ME').last().pct_change().dropna()
v5m = (1+D['v5_net']).resample('ME').prod()-1; v2m=(1+D['v2_net']).resample('ME').prod()-1
for name, tot in [('SG Trend', sg_m), ('SG CTA', cta_m), ('BTOP50', btop_r)]:
    for (mg, pf) in [(0,0),(1.0,0.2),(1.5,0.2),(2.0,0.2)]:
        g = gross_up_monthly(tot, tb_m, mg/100, pf)
        gx = g - tb_m.reindex(g.index)
        for w,(a,b) in {'since 2000':('2000-01',END),'since 2010':('2010-01',END),'since 2015':('2015-01',END),'1990-1999':('1990','1999'),'full':('1987',END)}.items():
            x = gx.loc[a:b]
            if len(x)<24: continue
            rows.append({'index':name,'fees added back':f'{mg}/{int(pf*100)}','window':w,'excess ann':x.mean()*12,'vol':x.std()*np.sqrt(12),'SR gross':msr(x),
                         'fee drag %/yr': (g.loc[a:b].mean()-tot.loc[a:b].mean())*12,
                         'v2 SR (monthly)':msr(v2m.loc[a:b]),'v5 SR (monthly)':msr(v5m.loc[a:b]),
                         'ratio SR/v2':msr(x)/msr(v2m.loc[a:b]),'ratio SR/v5':msr(x)/msr(v5m.loc[a:b])})
F=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(F.round(3).to_string()); F.to_csv('fonds_reels_frais_reintegres.csv', index=False)
# correlations
sgx = I['sg_x']
w = pd.DataFrame({'v5':D['v5_net'],'v2':D['v2_net'],'SG Trend':sgx,'SG CTA':I['cta_x'],'ES1':D['ES1 Index']}).loc['2000':END]
wk = (1+w.fillna(0)).resample('W-FRI').prod()-1
print('weekly corr 2000-2026\n', wk.corr().round(2))
mo = (1+w.fillna(0)).resample('ME').prod()-1
print('monthly corr\n', mo.corr().round(2))
# beta of v5 on SG Trend
import numpy.linalg as la
X = wk[['SG Trend']].assign(c=1).values; y = wk['v5'].values
b = la.lstsq(X,y,rcond=None)[0]; resid = y - X@b
print('v5 = %.4f + %.2f*SG weekly; alpha ann %.3f, resid vol %.3f, alpha IR %.2f' % (b[1], b[0], b[1]*52, resid.std()*np.sqrt(52), b[1]*52/(resid.std()*np.sqrt(52))))
for per in [('2000','2009'),('2010',END),('2015',END),('2020',END)]:
    s=wk.loc[per[0]:per[1]]; X=s[['SG Trend']].assign(c=1).values; y=s['v5'].values; b=la.lstsq(X,y,rcond=None)[0]; r=y-X@b
    print(per, 'beta %.2f alpha ann %.3f IR %.2f corr %.2f' % (b[0], b[1]*52, b[1]*52/(r.std()*np.sqrt(52)), s['v5'].corr(s['SG Trend'])))
corr = pd.concat([wk.corr()], keys=['weekly 2000-2026']); corr.to_csv('correlations_hebdo.csv')
# corr v5 vs ES1 since 1990 daily / weekly / monthly
z = pd.DataFrame({'v5':D['v5_net'],'v2':D['v2_net'],'ES1':D['ES1 Index']}).loc['1990':END].dropna()
print('daily corr since 1990', z.corr().round(3).to_dict())
zw=(1+z).resample('W-FRI').prod()-1; zm=(1+z).resample('ME').prod()-1
print('weekly', zw.corr().round(3).to_dict()); print('monthly', zm.corr().round(3).to_dict())
