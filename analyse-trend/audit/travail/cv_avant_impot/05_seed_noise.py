"""MC noise of the pre-tax medians: re-run the central 10-y cases with 6 fresh seeds (5000 paths, block 126).
Also an independent (engine-free) check of the untaxed 70/30 annual-rebalance mix on the main seed's shocks."""
import os, sys, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import mc_engine as M
AN, NP, NY, ESTR, INFL = M.AN, 5000, 10, 0.024, 0.02
S = pd.read_pickle('series.pkl')
D90 = S.loc['1990-01-01':'2026-07-10']
J00 = S.loc['2000-03-29':'2026-07-10', ['dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur']].dropna()
CASES = {'100k': (100e3, 'sub_100k', 0.20, 'sub_50k', 0.11), '250k': (250e3, 'sub_250k', 0.24, 'sub_100k', 0.20),
         '1M': (1e6, 'sub_1000k', 0.27, 'sub_500k', 0.27)}
POL = {0.214: (0.40, 0.17), 0.12: (0.25, 0.095)}
F = pd.read_csv('final_table_avant_impot.csv')
MU_sp = float(F[(F.case == 'e 100% equities (sp)') & (F.scen == 'central')].assumptions.iloc[0].split('arith ')[1].split('%')[0]) / 100
rows = []
t0 = time.time()
for seed in range(1, 7):
    rng = np.random.default_rng(seed)
    i90 = M.sb_idx(len(D90), NP, NY * AN, 126, rng)
    i00 = M.sb_idx(len(J00), NP, NY * AN, 126, rng)
    for tier, (K0, s21, sr21, s12, sr12) in CASES.items():
        for vol, sn, sr in ((0.214, s21, sr21), (0.12, s12, sr12)):
            z = M.standardise(D90[sn].values)[i90]
            r = M.sim_diy(z, sr, vol, K0, ESTR, years=(10,), fixed_usd=400.0, c=POL[vol][0], m=POL[vol][1], pfu=0.0, cash_tax=0.0)[10]
            rows.append(dict(seed=seed, case='DIY %s %d%%' % (tier, round(vol * 100)), med=r['at_med'], p10=r['at_p10'], p90=r['at_p90'],
                             P_nom_loss=r['P_nom_loss'], P_below_cash=r['P_below_cash'], dd_med=r['dd_med'], P_touch_60pct=r['P_touch_60pct']))
    zt = M.standardise(J00['dbi_rep_ex'].values)[i00]
    r = M.sim_etf(zt, 0.25, 0.12, ESTR, years=(10,), tax=0.0, cash_tax=0.0)[10]
    rows.append(dict(seed=seed, case='UCITS ETF', med=r['at_med'], p10=r['at_p10'], p90=r['at_p90'], P_nom_loss=r['P_nom_loss'],
                     P_below_cash=r['P_below_cash'], dd_med=r['dd_med']))
    ze = M.standardise(J00['eq_sp_eur'].values)[i00]
    r = M.sim_mix(ze, zt, MU_sp, 0.195, 0.25, 0.12, ESTR, years=(10,), tax_e=0, tax_t=0, reb_tax=0, cash_tax=0)[10]
    rows.append(dict(seed=seed, case='70/30 sp', med=r['at_med'], p10=r['at_p10'], p90=r['at_p90'], P_nom_loss=r['P_nom_loss'],
                     P_below_cash=r['P_below_cash'], dd_med=r['dd_med']))
    print('seed', seed, round(time.time() - t0, 1), flush=True)
N = pd.DataFrame(rows); N.to_csv('mc_seed_noise_avant_impot.csv', index=False)
agg = N.groupby('case').agg(med_mean=('med', 'mean'), med_sd=('med', 'std'), med_min=('med', 'min'), med_max=('med', 'max'),
                            Ploss_sd=('P_nom_loss', 'std'), Pcash_sd=('P_below_cash', 'std'))
main = F[(F.horizon == 10) & ((F.fixed_set == 'ibg400') | F.fixed_set.isna())]
def mainv(c):
    if c.startswith('DIY'):
        _, tier, v = c.split(); v = 0.214 if v == '21%' else 0.12
        return main[(main.case == 'a/b DIY v5 subset') & (main.tier == tier) & np.isclose(main.vol, v) & (main.scen == 'central')].at_med.iloc[0]
    if c == 'UCITS ETF':
        return main[(main.case == 'c UCITS trend ETF') & (main.scen == 'central')].at_med.iloc[0]
    return main[(main.case == 'e 70/30 annual rebalance (sp)') & (main.scen == 'eq central / trend central')].at_med.iloc[0]
agg['main_seed_med'] = [mainv(c) for c in agg.index]
agg['main_minus_mean_pt'] = (agg.main_seed_med - agg.med_mean) * 100
agg.to_csv('mc_seed_noise_summary.csv')
pd.set_option('display.width', 250)
print(agg.round(4).to_string())  # NB: 06-Oct fix: file post-processed to percent units
