import pandas as pd, numpy as np
from scipy import stats as st
M=pd.read_pickle('mine.pkl').loc['1990':'2026-07-10']
EG=0.5772156649015329
def emax(N): return (1-EG)*st.norm.ppf(1-1/N)+EG*st.norm.ppf(1-1/(N*np.e))
def psr(x, sr_star_per_period, lo_adj=None):
    x=np.asarray(x); T=len(x); sr=x.mean()/x.std(ddof=1); g3=st.skew(x); g4=st.kurtosis(x,fisher=False)
    se=np.sqrt((1-g3*sr+(g4-1)/4*sr**2)/(T-1))
    if lo_adj is not None: se*=np.sqrt(lo_adj)
    return st.norm.cdf((sr-sr_star_per_period)/se), sr, se
rows=[]
for freq,ppy in [('D',261),('ME',12)]:
    for w,a in {'1990-2026':'1990','since 2000':'2000','since 2010':'2010','since 2015':'2015'}.items():
        s=M.v5.loc[a:]
        x = s.values if freq=='D' else ((1+s).resample('ME').prod()-1).values
        for N in [20,74]:
            for sd in [0.123,0.25]:
                sr0=sd*emax(N)/np.sqrt(ppy)
                d,sr,se=psr(x,sr0)
                # autocorrelation adjustment: use variance ratio at 252 days relative to sampling frequency
                d2,_,se2=psr(x,sr0,lo_adj=1.5 if freq=='D' else 1.5/1.18)
                rows.append(dict(freq=freq,window=w,N=N,sd=sd,SR_ann=sr*np.sqrt(ppy),SE_ann=se*np.sqrt(ppy),SR0_ann=sr0*np.sqrt(ppy),DSR=d,DSR_autocorr_adj=d2))
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(3).to_string()); R.to_csv('dsr_check.csv',index=False)
print('E[max] N=20,47,74:', emax(20), emax(47), emax(74))
print('95% quantile of max of N iid normals N=47:', st.norm.ppf(0.95**(1/47)))
