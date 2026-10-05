import pandas as pd, numpy as np
# --- analytic Kelly table (Gaussian, continuous rebalancing): g = SR*s - s^2/2
rows=[]
for sr in [0.15,0.25,0.35,0.45,0.55]:
    for s in [0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.214,0.25,0.30,0.35,0.40]:
        rows.append(dict(SR=sr, vol=s, arith_excess=sr*s, geo_excess=sr*s-s*s/2, kelly_fraction=s/sr, share_of_max_growth=(sr*s-s*s/2)/(sr*sr/2)))
K=pd.DataFrame(rows); K.to_csv('kelly_analytique.csv',index=False)
print(K[K.vol.isin([0.10,0.175,0.214,0.30])].round(3).to_string())
# --- bootstrap with fat tails + parameter uncertainty
D = pd.read_pickle('daily.pkl').loc['1990-01-01':'2026-07-10']
DPY=261; NP=6000; H=10*DPY; rng=np.random.default_rng(7)
x=D['v5_net'].values; n=len(x); z=(x-x.mean())/x.std()/np.sqrt(DPY)   # unit annual vol, zero mean
p=1/60; idx=np.empty((NP,H),dtype=np.int32); idx[:,0]=rng.integers(0,n,NP)
new=rng.random((NP,H))<p; st=rng.integers(0,n,(NP,H))
for t in range(1,H): idx[:,t]=np.where(new[:,t],st[:,t],(idx[:,t-1]+1)%n)
Zp=z[idx]
SRdraw = rng.choice([0.15,0.35,0.55], size=NP, p=[0.25,0.5,0.25])
SRwide = np.clip(rng.normal(0.35,0.20,NP), -0.3, 1.0)
CASH=0.03
out=[]
for label, srv in [('SR fixed 0.35',np.full(NP,0.35)),('SR mixture 0.15/0.35/0.55 (25/50/25)',SRdraw),('SR ~ N(0.35, 0.20)',SRwide),('SR fixed 0.15',np.full(NP,0.15))]:
    for s in [0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.214,0.25,0.30,0.35,0.40,0.50]:
        r = Zp*s + (srv*s)[:,None]/DPY + CASH/DPY
        r = np.maximum(r, -0.99)
        W = np.cumprod(1+r,axis=1); WT=W[:,-1]
        dd = (W/np.maximum.accumulate(W,axis=1)-1).min(axis=1)
        logW=np.log(WT)
        row=dict(prior=label, vol=s, leverage_vs_v5=s/0.214, median_total_CAGR=np.median(WT)**0.1-1, p5_total_CAGR=np.percentile(WT,5)**0.1-1,
                 mean_log_growth=logW.mean()/10, P_maxDD_worse_30=np.mean(dd<-0.3), P_maxDD_worse_50=np.mean(dd<-0.5), median_maxDD=np.median(dd))
        for g in [2,3,5]:
            row[f'CE growth gamma={g}'] = (np.mean(WT**(1-g)))**(1/(1-g))**1 ** 1
            row[f'CE growth gamma={g}'] = np.mean(WT**(1-g))**(1/((1-g)*10)) - 1
        out.append(row)
O=pd.DataFrame(out); O.to_csv('levier_optimal_bootstrap.csv',index=False)
pd.set_option('display.width',250)
print(O.round(3).to_string())
best=[]
for lab,g in O.groupby('prior'):
    b={'prior':lab,'argmax E[log] vol':g.loc[g.mean_log_growth.idxmax(),'vol'],'argmax median CAGR vol':g.loc[g.median_total_CAGR.idxmax(),'vol']}
    for gg in [2,3,5]: b[f'argmax CE gamma={gg} vol']=g.loc[g[f'CE growth gamma={gg}'].idxmax(),'vol']
    ok=g[g.P_maxDD_worse_30<=0.25]; b['max median CAGR s.t. P(DD<-30%)<=25% : vol']=ok.loc[ok.median_total_CAGR.idxmax(),'vol'] if len(ok) else np.nan
    best.append(b)
B=pd.DataFrame(best); print(B.to_string()); B.to_csv('levier_optimal_resume.csv',index=False)
