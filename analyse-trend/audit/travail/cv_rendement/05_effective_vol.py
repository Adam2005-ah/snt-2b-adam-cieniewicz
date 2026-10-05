"""Effective (long-horizon) volatility of the bootstrapped paths vs the daily-annualised input vol."""
import numpy as np, pandas as pd
import mc_engine as M
AN = M.AN; NP = 3000
S = pd.read_pickle('series.pkl')
D90 = S.loc['1990-01-01':'2026-07-10']
J00 = S.loc['2000-03-29':'2026-07-10', ['dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur']].dropna()
rng = np.random.default_rng(3)
rows = []
for b in (126, 252):
    i90 = M.sb_idx(len(D90), NP, 10 * AN, b, rng); i00 = M.sb_idx(len(J00), NP, 10 * AN, b, rng)
    for name, src, idx, vol in (('v5_84', D90, i90, 0.214), ('sub_100k', D90, i90, 0.214), ('sub_1000k', D90, i90, 0.214), ('sub_50k', D90, i90, 0.12),
                                ('dbi_rep_ex', J00, i00, 0.12), ('sgtrend_ex', J00, i00, 0.12), ('eq_sp_eur', J00, i00, 0.18),
                                ('eq_sp_eur', J00, i00, 0.203), ('eq_nq_eur', J00, i00, 0.25)):
        z = M.standardise(src[name].values)[idx].astype(float) * vol / np.sqrt(AN)
        lr = np.log1p(z)
        y1 = lr[:, :AN].sum(1); y10 = lr.sum(1)
        rows.append(dict(series=name, block=b, vol_input=vol, eff_vol_1y=y1.std(), eff_vol_10y=y10.std() / np.sqrt(10)))
    if name == 'eq_nq_eur':
        z = None
T = pd.DataFrame(rows).round(4); print(T.to_string(index=False)); T.to_csv('effective_vol.csv', index=False)
# MC noise of the median after-tax CAGR: 8 seeds, DIY 100k central and ETF central
res = []
for seed in range(8):
    r = np.random.default_rng(100 + seed)
    idx = M.sb_idx(len(D90), 5000, 10 * AN, 126, r)
    z = M.standardise(D90['sub_100k'].values)[idx]
    a = M.sim_diy(z, 0.20, 0.214, 100e3, 0.024, years=(10,), fixed_usd=500.0, c=0.40, m=0.17)[10]
    res.append(dict(seed=seed, diy100k_med=a['at_med'], diy100k_Ploss=a['P_nom_loss'], diy100k_dd=a['dd_med']))
R = pd.DataFrame(res); print(R.describe().loc[['mean', 'std']].round(4)); R.to_csv('mc_seed_noise.csv', index=False)
