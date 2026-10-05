import pandas as pd, numpy as np
from scipy import stats as st
R='/home/user/snt-2b-adam-cieniewicz/analyse-trend'
D = pd.read_pickle('daily.pkl'); END='2026-07-10'
# --- collect trials ---
tr=[]
a=pd.read_csv(f'{R}/resultats/ameliorations.csv',index_col=0); tr+= [('ameliorations',i,v) for i,v in a['sharpe 1990-2026'].items()]
b=pd.read_csv(f'{R}/experiences/taille_selon_volatilite.csv'); tr+= [('taille_vol',r['name'],r['sharpe 1990-2026']) for _,r in b.iterrows()]
c=pd.read_csv(f'{R}/experiences/taille_selon_performance.csv'); tr+= [('taille_perf',r['variant'],r['SR 1990-2026']) for _,r in c.iterrows()]
d=pd.read_csv(f'{R}/resultats/portefeuille_resume.csv',index_col=0); d=d[d['début'].astype(str).str.startswith('1990') & d['fin'].astype(str).str.startswith('2026')]; tr+= [('portefeuille',i,v) for i,v in d['sharpe'].items()]
e=pd.read_csv(f'{R}/resultats/sous_ensembles_resume.csv',index_col=0); tr+= [('sous_ensembles',i,v) for i,v in e['sharpe'].items()]
f=pd.read_csv(f'{R}/experiences/carry_resultats.csv'); tr+= [('carry (Carver data 1990-2024)',r['name'],r['sharpe 1990-2024']) for _,r in f.iterrows()]
T=pd.DataFrame(tr,columns=['file','variant','SR_ann']).dropna()
T.to_csv('essais_comptes.csv',index=False)
print(T.groupby('file')['SR_ann'].agg(['count','mean','std','min','max']).round(3))
print('total trials', len(T), 'SR std all', T.SR_ann.std().round(3))
trend_only = T[~T.file.str.startswith('carry') & ~T.file.str.startswith('sous')]
print('trend-only same-data trials', len(trend_only), 'std', trend_only.SR_ann.std().round(3), 'max', trend_only.SR_ann.max())
EG = 0.5772156649
def sr0(N, var_sr):
    return np.sqrt(var_sr)*((1-EG)*st.norm.ppf(1-1/N) + EG*st.norm.ppf(1-1/(N*np.e)))
def dsr(x, N, sd_ann):
    x=x.dropna(); n=len(x); sr=x.mean()/x.std(); g3=st.skew(x); g4=st.kurtosis(x, fisher=False)
    s0 = sr0(N, (sd_ann/np.sqrt(252))**2)
    z = (sr - s0)*np.sqrt(n-1)/np.sqrt(1 - g3*sr + (g4-1)/4*sr**2)
    psr0 = st.norm.cdf(sr*np.sqrt(n-1)/np.sqrt(1 - g3*sr + (g4-1)/4*sr**2))
    # min track record length to reject SR<=s0 at 95%
    mintrl = 1 + (1 - g3*sr + (g4-1)/4*sr**2)*(st.norm.ppf(.95)/(sr-s0))**2 if sr>s0 else np.inf
    return dict(T_days=n, SR_ann=sr*np.sqrt(252), skew_daily=g3, kurt_daily=g4, SR0_ann=s0*np.sqrt(252), PSR_vs0=psr0, DSR=st.norm.cdf(z),
                deflated_SR_ann=(sr-s0)*np.sqrt(252), MinTRL_years=mintrl/252)
rows=[]
sd_obs = trend_only.SR_ann.std()
for var in ['v5','v2']:
    for w,(a_,b_) in {'1990-2026':('1990',END),'since 2000':('2000',END),'since 2010':('2010',END),'since 2015':('2015',END)}.items():
        for N in [20,50,74]:
            for sd in [sd_obs, 0.25]:
                r=dsr(D[f'{var}_net'].loc[a_:b_], N, sd); r.update(variant=var, window=w, N=N, sd_SR_trials=sd); rows.append(r)
Q=pd.DataFrame(rows)[['variant','window','N','sd_SR_trials','T_days','SR_ann','skew_daily','kurt_daily','SR0_ann','deflated_SR_ann','PSR_vs0','DSR','MinTRL_years']]
pd.set_option('display.width',250); print(Q.round(3).to_string()); Q.to_csv('deflated_sharpe.csv',index=False)
# --- Harvey-Liu style haircut (Bonferroni / Sidak, monthly t-stat as HL) + HLZ t>3 hurdle
rows=[]
for var in ['v5','v2']:
    m=(1+D[f'{var}_net']).resample('ME').prod()-1
    for w,(a_,b_) in {'1990-2026':('1990',END),'since 2000':('2000',END),'since 2010':('2010',END),'since 2015':('2015',END)}.items():
        x=m.loc[a_:b_]; n=len(x); sr=x.mean()/x.std()*np.sqrt(12); t=x.mean()/x.std()*np.sqrt(n); p=2*(1-st.norm.cdf(t))
        for N in [20,50,74]:
            for meth in ['Bonferroni','Sidak']:
                pa = min(1,p*N) if meth=='Bonferroni' else 1-(1-p)**N
                ta = st.norm.ppf(1-pa/2) if pa<1 else 0.0
                hsr = ta/np.sqrt(n/12)
                rows.append(dict(variant=var,window=w,years=n/12,SR_ann_monthly=sr,t_stat=t,p=p,N=N,method=meth,p_adj=pa,haircut_SR=hsr,haircut_pct=1-hsr/sr))
        # HLZ hurdle t=3
        rows.append(dict(variant=var,window=w,years=n/12,SR_ann_monthly=sr,t_stat=t,p=p,N=np.nan,method='HLZ t>3 hurdle: SR needed',haircut_SR=3/np.sqrt(n/12)))
H=pd.DataFrame(rows); print(H.round(3).to_string()); H.to_csv('haircut_harvey_liu.csv',index=False)
# --- Test of the improvement v5 vs v2 (v5 scaled to v2 vol), NW t-stat, and expected max t under null with N tries
import statsmodels.api as sm
out=[]
for w,(a_,b_) in {'1990-2026':('1990',END),'since 2000':('2000',END),'since 2010':('2010',END),'since 2015':('2015',END),'since 2020':('2020',END)}.items():
    x5=D['v5_net'].loc[a_:b_]; x2=D['v2_net'].loc[a_:b_]
    diff = x5*(x2.std()/x5.std()) - x2
    mod=sm.OLS(diff.values, np.ones(len(diff))).fit(cov_type='HAC',cov_kwds={'maxlags':10})
    out.append(dict(window=w, SR_v5=x5.mean()/x5.std()*np.sqrt(252), SR_v2=x2.mean()/x2.std()*np.sqrt(252), excess_at_v2_vol=diff.mean()*252, t_NW10=mod.tvalues[0]))
I=pd.DataFrame(out)
for N in [7,20,47]:
    # expected max of N iid N(0,1) (upper bound, trials are correlated) 
    I[f'E[max t] N={N}'] = (1-EG)*st.norm.ppf(1-1/N)+EG*st.norm.ppf(1-1/(N*np.e))
print(I.round(3).to_string()); I.to_csv('test_amelioration_v5_vs_v2.csv',index=False)
