import numpy as np, pandas as pd, core
R = core.rets
mk = ['CUA1 Comdty','SCO1 Comdty','LH1 Comdty','FC1 Comdty','CT1 Comdty','IK1 Comdty','SFI5 Comdty','RS1 Comdty','CL1 Comdty','GC1 Comdty','C 1 Comdty']
def by_year(f):
    out = {}
    for c in mk:
        s = R[c].dropna(); out[c.split()[0]] = s.groupby(s.index.year).apply(f)
    return pd.concat(out, axis=1).sort_index()
pd.set_option('display.width', 250)
print('AC1 by year'); print(by_year(lambda x: x.autocorr()).loc[2005:].round(2).to_string())
print('zero share by year'); print(by_year(lambda x: (x==0).mean()).loc[2005:].round(2).to_string())
print('CL1 zero share by year 1983-2000'); s=R['CL1 Comdty'].dropna(); print(s.groupby(s.index.year).apply(lambda x:(x==0).mean()).loc[:2000].round(2).to_dict())
