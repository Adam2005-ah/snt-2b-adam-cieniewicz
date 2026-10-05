"""Fresh seeds (4 x 10,000 paths, block 126) for the central cases, plus an i.i.d. Gaussian benchmark with the same
drift/vol (plausibility of probabilities and drawdowns). Equity mu is the audit's calibrated arithmetic mu (held fixed),
so the equity median is free to deviate from 6 % under other seeds. Fixed cost USD 400."""
import os, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import indep as I

ESTR = 0.024
MU = {'sp': 0.07377, 'nq': 0.08183}
VOL_E = {'sp': (0.195, 'eq_sp_eur'), 'nq': (0.25, 'eq_nq_eur')}
CASES = [('DIY 21% 100k', '100k', 0.214, 0.20, 100e3, 'sub_100k', (0.40, 0.17)),
         ('DIY 21% 250k', '250k', 0.214, 0.24, 250e3, 'sub_250k', (0.40, 0.17)),
         ('DIY 21% 1M', '1M', 0.214, 0.27, 1e6, 'sub_1000k', (0.40, 0.17)),
         ('DIY 12% 100k', '100k', 0.12, 0.11, 100e3, 'sub_50k', (0.25, 0.095)),
         ('DIY 12% 250k', '250k', 0.12, 0.20, 250e3, 'sub_100k', (0.25, 0.095)),
         ('DIY 12% 1M', '1M', 0.12, 0.27, 1e6, 'sub_500k', (0.25, 0.095))]
rows = []
t0 = time.time()
for seed in (101, 202, 303, 404):
    SH = I.Shocks('fresh', seed=seed, npath=10000)
    for name, tier, vol, sr, K0, sn, (c, m) in CASES:
        r = I.diy(SH.z(sn), sr, vol, K0, ESTR, 400., c, m, years=(10,))[10]
        rows.append(dict(seed=seed, case=name, **r))
    zt = SH.z('dbi_rep_ex')
    r, _ = I.fund(zt, ESTR + 0.25 * 0.12, 0.12, ESTR, years=(10,)); rows.append(dict(seed=seed, case='ETF', **r[10]))
    for k, (ve, zn) in VOL_E.items():
        ze = SH.z(zn)
        r, _ = I.fund(ze, MU[k], ve, ESTR, years=(10,)); rows.append(dict(seed=seed, case='100% ' + k, **r[10]))
        r = I.mix(ze, zt, MU[k], ve, ESTR + 0.25 * 0.12, 0.12, ESTR, years=(10,)); rows.append(dict(seed=seed, case='70/30 ' + k, **r[10]))
    print('seed', seed, round(time.time() - t0, 1), flush=True)
    del SH

# Gaussian i.i.d. benchmark (seed 7, 10,000 paths): same drift and vol, normal shocks, equity/trend correlation from data
rng = np.random.default_rng(7)
n = 10 * I.AN
for name, tier, vol, sr, K0, sn, (c, m) in CASES:
    z = rng.standard_normal((10000, n))
    r = I.diy(z, sr, vol, K0, ESTR, 400., c, m, years=(10,))[10]
    rows.append(dict(seed='gauss', case=name, **r))
zt = rng.standard_normal((10000, n))
r, _ = I.fund(zt, ESTR + 0.25 * 0.12, 0.12, ESTR, years=(10,)); rows.append(dict(seed='gauss', case='ETF', **r[10]))
for k, (ve, zn) in VOL_E.items():
    rho = np.corrcoef(I.J00[zn], I.J00['dbi_rep_ex'])[0, 1]
    ze = rho * zt + np.sqrt(1 - rho ** 2) * rng.standard_normal((10000, n))
    r, _ = I.fund(ze, MU[k], ve, ESTR, years=(10,)); rows.append(dict(seed='gauss', case='100% ' + k, **r[10]))
    r = I.mix(ze, zt, MU[k], ve, ESTR + 0.25 * 0.12, 0.12, ESTR, years=(10,)); rows.append(dict(seed='gauss', case='70/30 ' + k, rho=rho, **r[10]))
R = pd.DataFrame(rows)
R.to_csv('fresh_seeds_and_gauss.csv', index=False)
pd.set_option('display.width', 250)
cols = ['med', 'p10', 'p90', 'P_loss', 'P_below_cash', 'dd_med', 'dd_p10', 'P_touch60', 'mean_log']
fr = R[R.seed != 'gauss']
agg = fr.groupby('case', sort=False)[cols].mean()
agg['med_sd_across_seeds'] = fr.groupby('case', sort=False)['med'].std()
print('FRESH 4x10k mean:\n', (agg * 100).round(2).to_string())
print('GAUSS:\n', (R[R.seed == 'gauss'].set_index('case')[cols] * 100).round(2).to_string())
agg.to_csv('fresh_seeds_summary.csv')
print('done', round(time.time() - t0, 1))
