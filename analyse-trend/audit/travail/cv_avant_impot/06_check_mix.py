"""Engine-free check of the untaxed 70/30 annual-rebalance mix and of the untaxed ETF on the main-seed shocks
(rebuilds the same Z arrays as 04_final_avant_impot.py by replaying the same RNG sequence)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import mc_engine as M
AN, NP, NY, ESTR = M.AN, 5000, 10, 0.024
S = pd.read_pickle('series.pkl')
D90 = S.loc['1990-01-01':'2026-07-10']
J00 = S.loc['2000-03-29':'2026-07-10', ['dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur']].dropna()
rng = np.random.default_rng(20261005)
for b in (63, 126, 252):
    M.sb_idx(len(D90), NP, NY * AN, b, rng)        # burn the D90 draws in the same order
I00 = M.sb_idx(len(J00), NP, NY * AN, 63, rng); I00 = M.sb_idx(len(J00), NP, NY * AN, 126, rng)
F = pd.read_csv('final_table_avant_impot.csv')
out = []
for k, ve, zn in (('sp', 0.195, 'eq_sp_eur'), ('nq', 0.25, 'eq_nq_eur')):
    mu = float(F[(F.case == 'e 100%% equities (%s)' % k) & (F.scen == 'central')].assumptions.iloc[0].split('arith ')[1].split('%')[0]) / 100
    ze = M.standardise(J00[zn].values)[I00].astype(float); zt = M.standardise(J00['dbi_rep_ex'].values)[I00].astype(float)
    re = mu / AN + ze * ve / np.sqrt(AN); rt = (ESTR + 0.25 * 0.12) / AN + zt * 0.12 / np.sqrt(AN)
    W = np.ones(NP)
    for yr in range(NY):
        sl = slice(yr * AN, (yr + 1) * AN)
        W = W * (0.7 * np.prod(1 + re[:, sl], axis=1) + 0.3 * np.prod(1 + rt[:, sl], axis=1))   # rebalanced each Dec, no tax
        if yr + 1 in (5, 10):
            cg = W ** (1 / (yr + 1)) - 1
            eng = F[(F.case == 'e 70/30 annual rebalance (%s)' % k) & (F.scen == 'eq central / trend central') & (F.horizon == yr + 1)].iloc[0]
            out.append(dict(check='70/30 %s %dy' % (k, yr + 1), direct_med=np.median(cg), engine_med=eng.at_med,
                            direct_p10=np.percentile(cg, 10), engine_p10=eng.at_p10, direct_Ploss=np.mean(W < 1), engine_Ploss=eng.P_nom_loss))
    Wt = np.prod(1 + rt, axis=1); cg = Wt ** 0.1 - 1
    eng = F[(F.case == 'c UCITS trend ETF') & (F.scen == 'central') & (F.horizon == 10)].iloc[0]
    if k == 'sp':
        out.append(dict(check='ETF 10y', direct_med=np.median(cg), engine_med=eng.at_med, direct_p10=np.percentile(cg, 10),
                        engine_p10=eng.at_p10, direct_Ploss=np.mean(Wt < 1), engine_Ploss=eng.P_nom_loss))
O = pd.DataFrame(out); O.to_csv('check_mix_direct.csv', index=False)
print(O.to_string())
