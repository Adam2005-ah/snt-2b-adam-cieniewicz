import numpy as np, pandas as pd
exec(open('05_mc.py').read().split('z=(x5')[0])
z=(x5-x5.mean())/x5.std()*0.214/np.sqrt(DPY)
for seed in [2,3,20261004]:
    rng=np.random.default_rng(seed); I=sb_idx(n,6000,H,60,rng)
    r=z[I]+0.35*0.214/DPY; s=stats(r); print(seed, {k:round(v,3) for k,v in s.items()})
# historical rolling 10y
c=0.03/DPY; dds=[]
for st_ in range(0,n-H,21):
    w=np.cumprod(1+M.v5.values[st_:st_+H]+c); dds.append((w/np.maximum.accumulate(w)-1).min())
print('historical rolling 10y maxDD median', np.median(dds), 'min', np.min(dds))
