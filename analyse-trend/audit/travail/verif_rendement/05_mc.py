import pandas as pd, numpy as np, time
M=pd.read_pickle('mine.pkl').loc['1990':'2026-07-10']
DPY=261; H=10*DPY; NP=4000; CASH=0.03
x5=M.v5.values; n=len(x5)
def sb_idx(n,NP,H,block,rng):
    # Politis-Romano stationary bootstrap, circular
    idx=np.empty((NP,H),dtype=np.int64); idx[:,0]=rng.integers(0,n,NP)
    jump=rng.random((NP,H))<1/block; newst=rng.integers(0,n,(NP,H))
    for t in range(1,H): idx[:,t]=np.where(jump[:,t],newst[:,t],(idx[:,t-1]+1)%n)
    return idx
def stats(r, h_years=10, cash=CASH):
    h=h_years*DPY; r=r[:,:h]; tot=r+cash/DPY
    W=np.cumprod(1+tot,axis=1); cagr=W[:,-1]**(1/h_years)-1
    dd=(W/np.maximum.accumulate(W,axis=1)-1).min(axis=1)
    Wex=np.prod(1+r,axis=1)
    # annual vol of non-overlapping 1y sums
    ys=r.reshape(r.shape[0],h_years,DPY).sum(axis=2); 
    return dict(med_cagr=np.median(cagr),p5=np.percentile(cagr,5),P_loss=np.mean(W[:,-1]<1),P_below_cash=np.mean(Wex<1),
                medDD=np.median(dd),p5DD=np.percentile(dd,5),P_DD50=np.mean(dd<-0.5), ann_vol_of_1y_sums=ys.std(), daily_vol_ann=r.std()*np.sqrt(DPY))
z=(x5-x5.mean())/x5.std()*0.214/np.sqrt(DPY)
rows=[]
for block in [1,20,60,250]:
    rng=np.random.default_rng(1)
    I=sb_idx(n,NP,H,block,rng)
    for sr in [1.17,0.35,0.15]:
        r=z[I]+sr*0.214/DPY
        s=stats(r); s.update(block=block,SR=sr); rows.append(s)
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(3).to_string()); R.to_csv('mc_block_sensitivity.csv',index=False)
# historical: rolling 10y windows of actual v5 total maxDD and 1y vol
c=0.03/DPY
W=(1+M.v5.values+0)  # excess
dds=[]
for st_ in range(0,n-H,21):
    w=np.cumprod(1+M.v5.values[st_:st_+H]+c); dds.append((w/np.maximum.accumulate(w)-1).min())
print('historical rolling 10y maxDD (excess+3pct cash) median %.3f min %.3f'%(np.median(dds),np.min(dds)))
