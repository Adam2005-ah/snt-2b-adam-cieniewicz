"""Fresh seeds 101..404 (4 x 10,000 paths): full pess/central/opt grid at 5 and 10 y, USD 400 fixed, no tax.
Equity arithmetic mu held at the audit's calibrated values."""
import os, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import indep as I
ESTR = 0.024
TIERS = {'100k': (100e3, 'sub_100k', (0.00, 0.20, 0.40), 'sub_50k', (-0.09, 0.11, 0.31)),
         '250k': (250e3, 'sub_250k', (0.03, 0.24, 0.43), 'sub_100k', (0.00, 0.20, 0.40)),
         '1M': (1e6, 'sub_1000k', (0.05, 0.27, 0.47), 'sub_500k', (0.05, 0.27, 0.47))}
POL = {0.214: (0.40, 0.17), 0.12: (0.25, 0.095)}
LAB = ('pess', 'central', 'opt')
MU = {('sp', 'pess'): 0.0499, ('sp', 'central'): 0.07377, ('sp', 'opt'): 0.09246,
      ('nq', 'pess'): 0.05796, ('nq', 'central'): 0.08183, ('nq', 'opt'): 0.10052}
VOL_E = {'sp': (0.195, 'eq_sp_eur'), 'nq': (0.25, 'eq_nq_eur')}
TR = {'pess': 0.10, 'central': 0.25, 'opt': 0.40}
rows = []; t0 = time.time()
for seed in (101, 202, 303, 404):
    SH = I.Shocks('fresh', seed=seed, npath=10000)
    for tier, (K0, s21, g21, s12, g12) in TIERS.items():
        for vol, sn, grid in ((0.214, s21, g21), (0.12, s12, g12)):
            z = SH.z(sn)
            for lab, sr in zip(LAB, grid):
                for y, r in I.diy(z, sr, vol, K0, ESTR, 400., *POL[vol]).items():
                    rows.append(dict(seed=seed, case='DIY %d%% %s' % (round(vol * 100), tier), scen=lab, horizon=y, **r))
    zt = SH.z('dbi_rep_ex')
    for lab in LAB:
        res, _ = I.fund(zt, ESTR + TR[lab] * 0.12, 0.12, ESTR)
        for y, r in res.items(): rows.append(dict(seed=seed, case='ETF', scen=lab, horizon=y, **r))
    for k, (ve, zn) in VOL_E.items():
        ze = SH.z(zn)
        for lab in LAB:
            res, _ = I.fund(ze, MU[(k, lab)], ve, ESTR)
            for y, r in res.items(): rows.append(dict(seed=seed, case='100% ' + k, scen=lab, horizon=y, **r))
            for y, r in I.mix(ze, zt, MU[(k, lab)], ve, ESTR + TR[lab] * 0.12, 0.12, ESTR).items():
                rows.append(dict(seed=seed, case='70/30 ' + k, scen=lab, horizon=y, **r))
    print('seed', seed, round(time.time() - t0, 1), flush=True)
    del SH
R = pd.DataFrame(rows); R.to_csv('fresh_full_grid.csv', index=False)
cols = ['med', 'p10', 'p90', 'P_loss', 'P_below_cash', 'dd_med', 'dd_p10', 'P_touch60']
A = R.groupby(['horizon', 'case', 'scen'], sort=False)[cols].mean() * 100
A['med_sd_seeds'] = R.groupby(['horizon', 'case', 'scen'], sort=False)['med'].std() * 100
A.round(3).to_csv('fresh_full_grid_summary.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(A.round(2).to_string())
print('done', round(time.time() - t0, 1))
