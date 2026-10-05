"""Alternative central forward Sharpe from my own anchors (subset lag cost only -0.02/-0.03, see lag_check.csv):
100k 0.25, 250k 0.28, 1M 0.31 (21.4 % vol). Same engine/seed family as 04_final.py."""
import numpy as np, pandas as pd
import mc_engine as M
AN = M.AN
S = pd.read_pickle('series.pkl'); D90 = S.loc['1990-01-01':'2026-07-10']
rng = np.random.default_rng(20261005)
idx = M.sb_idx(len(D90), 5000, 10 * AN, 126, rng)
rows = []
for tier, K0, fx, sh, sr in (('100k', 100e3, 500.0, 'sub_100k', 0.25), ('250k', 250e3, 625.0, 'sub_250k', 0.28), ('1M', 1e6, 1000.0, 'sub_1000k', 0.31)):
    z = M.standardise(D90[sh].values)[idx]
    for y, r in M.sim_diy(z, sr, 0.214, K0, 0.024, years=(5, 10), fixed_usd=fx, c=0.40, m=0.17).items():
        rows.append(dict(tier=tier, sr=sr, horizon=y, **{k: r[k] for k in ('at_med', 'at_p10', 'at_p90', 'real_med', 'P_nom_loss', 'P_below_cash', 'dd_med', 'dd_p10')}))
T = pd.DataFrame(rows); print((T.set_index(['tier', 'sr', 'horizon']) * 100).round(1)); T.to_csv('alt_central_sr.csv', index=False)
