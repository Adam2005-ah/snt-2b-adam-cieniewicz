"""(1) my engine vs project trend.run_portfolio / run_vol_targeted on all 84 (v2, v5) and on the previous step's 250k
subsets (v2, v5) -- passing a column subset to trend recomputes weights/IDM for that subset (task statement)."""
import time, pickle
import numpy as np, pandas as pd
from indep import *
t0 = time.time()
rets, risk = RETS, RISK
costs, roll = BPC.to_dict(), ROLLBP.to_dict()
SUBS = pickle.load(open(A + 'var_passe_retail/subsets_variantes.pkl', 'rb'))
out = {}
for v in ['v2', 'v5']:
    for lab, cols in [('84', list(rets.columns)), ('250k', SUBS[(v, 250e3)])]:
        r, k = rets[cols], risk[cols]
        cc = {c: costs[c] for c in cols}; rr = {c: roll[c] for c in cols}
        if v == 'v2':
            ref = trend.run_portfolio(r, k, cc, rr)
        else:
            ref = trend.run_vol_targeted(r, k, cc, rr, forecast_fn=trend.forecast_regime)
        mine = system(v, cols)
        dp = (mine['pos'] - ref['positions']).abs().max().max()
        dn = (mine['net'] - ref['net']).abs().max()
        x = ref['net'].loc['1990':'2026-07-10']
        print(v, lab, len(cols), 'max|dpos|', dp, 'max|dnet|', dn, 'SR90', round(sr(x), 3), 'vol', round(x.std() * 16, 4),
              'gross exp', round(ref['positions'].loc['1990':].abs().sum(axis=1).mean(), 2), round(time.time() - t0), flush=True)
        out[(v, lab)] = ref['positions']
pickle.dump(out, open(HERE + 'ref_positions_v2_v5.pkl', 'wb'))
