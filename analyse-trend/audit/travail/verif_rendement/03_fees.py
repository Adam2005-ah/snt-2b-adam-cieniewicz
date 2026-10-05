import pandas as pd, numpy as np
F=pd.read_pickle('funds.pkl'); M=pd.read_pickle('mine.pkl')
sgm_tot=F['sgm_tot']; cm=F['cm']
def gross_up(net, mgmt, perf, crys='A'):
    """Simulate a fund: gross assets G evolve with gross return g; mgmt accrued monthly on gross pre-fee assets;
    incentive accrued on (NAV_before_inc - HWM) and crystallised at period end (A=annual, M=monthly). Solve g each month s.t. NAV net matches."""
    nav=1.0; hwm=1.0; accrued_base=None; out=[]
    # with accrual: NAV_t = A_t - accrued_inc_t, accrued_inc_t = perf*max(0, A_t - hwm) where A_t = assets after mgmt
    A=1.0  # assets after mgmt fees, before incentive accrual
    for t, rn in net.items():
        target = nav*(1+rn)
        # A_new = A*(1+g) - A*mgmt/12 ; nav_new = A_new - perf*max(0, A_new - hwm)
        if target > hwm: A_new = (target - perf*hwm)/(1-perf)
        else: A_new = target
        g = (A_new + A*mgmt/12)/A - 1
        out.append(g)
        A = A_new; nav = target
        if (crys=='M') or (crys=='A' and t.month==12):
            # crystallise: pay accrued incentive, assets drop to nav, hwm resets
            if nav > hwm: hwm = nav
            A = nav
    return pd.Series(out, index=net.index)
def sr(x): return x.mean()/x.std()*np.sqrt(12)
v5m=(1+M['v5']).resample('ME').prod()-1; v2m=(1+M['v2']).resample('ME').prod()-1
rows=[]
for fee in [(0,0),(1,.15),(1,.2),(1.5,.2),(2,.2)]:
    for crys in ['M','A']:
        g=gross_up(sgm_tot.loc[:'2026-07'], fee[0]/100, fee[1], crys)
        gx=g-cm.reindex(g.index)
        for w,a in {'since 2000':'2000','since 2010':'2010','since 2015':'2015'}.items():
            x=gx.loc[a:'2026-07']
            rows.append(dict(fee=f'{fee[0]}/{int(fee[1]*100)}',crys=crys,window=w,SR_gross=sr(x),drag=(g.loc[a:'2026-07'].mean()-sgm_tot.loc[a:'2026-07'].mean())*12,
                             v5=sr(v5m.loc[a:'2026-07']), v2=sr(v2m.loc[a:'2026-07']), ratio_v5=sr(x)/sr(v5m.loc[a:'2026-07']), ratio_v2=sr(x)/sr(v2m.loc[a:'2026-07'])))
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(3).to_string())
R.to_csv('fees_addback_check.csv',index=False)
