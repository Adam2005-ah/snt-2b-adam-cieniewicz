"""Fresh seeds 101..404 (4 x 10,000 paths): DIY central 10 y at fixed USD 150 / 400 / 1,000 / 2,740 (400 + time), no tax."""
import os, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import indep as I
ESTR = 0.024
C = [('21% 100k', 100e3, 'sub_100k', 0.214, 0.20, (0.40, 0.17)), ('21% 250k', 250e3, 'sub_250k', 0.214, 0.24, (0.40, 0.17)),
     ('21% 1M', 1e6, 'sub_1000k', 0.214, 0.27, (0.40, 0.17)), ('12% 100k', 100e3, 'sub_50k', 0.12, 0.11, (0.25, 0.095)),
     ('12% 250k', 250e3, 'sub_100k', 0.12, 0.20, (0.25, 0.095)), ('12% 1M', 1e6, 'sub_500k', 0.12, 0.27, (0.25, 0.095))]
rows = []; t0 = time.time()
for seed in (101, 202, 303, 404):
    SH = I.Shocks('fresh', seed=seed, npath=10000)
    for name, K0, sn, vol, sr, (c, m) in C:
        z = SH.z(sn)
        for fx in (150., 400., 1000., 2740.):
            r = I.diy(z, sr, vol, K0, ESTR, fx, c, m, years=(10,))[10]
            rows.append(dict(seed=seed, case=name, fixed=fx, **r))
    print('seed', seed, round(time.time() - t0, 1), flush=True)
    del SH
R = pd.DataFrame(rows); R.to_csv('fresh_fixed_costs.csv', index=False)
A = (R.groupby(['case', 'fixed'], sort=False)[['med', 'P_loss', 'P_below_cash']].mean() * 100).round(2)
print(A.unstack('fixed').to_string())
print('done', round(time.time() - t0, 1))
