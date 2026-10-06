"""(7) Is series_variantes.pkl usable by the forward MC (cv_avant_impot/mc_engine.py)? Quick 500-path smoke test per
key; also shows the trap of aligning it on the 1990-start series.pkl (NaN shocks)."""
import sys, numpy as np, pandas as pd
A = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
sys.path.insert(0, A + 'cv_avant_impot')
import mc_engine as M
S = pd.read_pickle(A + 'var_passe_retail/series_variantes.pkl')
ST = pd.read_csv(A + 'var_passe_retail/series_variantes_stats.csv', index_col=0)
old = pd.read_pickle(A + 'cv_avant_impot/series.pkl')
rng = np.random.default_rng(1)
idx = M.sb_idx(len(S), 500, 10 * M.AN, 126, rng)
rows = []
for k in S.columns:
    z = M.standardise(S[k].values)[idx]
    C = {'100k': 100e3, '250k': 250e3, '1M': 1e6}[k.split('_')[1]]
    r = M.sim_diy(z, 0.24, float(ST.loc[k, 'vol_realised_2000']), C, 0.024, years=(10,), mode='ibkr', fixed_usd=400.0,
                  c=0.40, m=float(ST.loc[k, 'mean_margin_2000']), pfu=0.0, cash_tax=0.0)[10]
    rows.append(dict(key=k, z_nan=int(np.isnan(z).sum()), vol=ST.loc[k, 'vol_realised_2000'], m=ST.loc[k, 'mean_margin_2000'],
                     med_10y_at_SR024=round(r['at_med'] * 100, 2)))
print(pd.DataFrame(rows).to_string(index=False))
J = old.join(S, how='left').loc['1990-01-01':'2026-07-10']
print('joined on series.pkl 1990+ index: NaN rows in v5_250k =', int(J['v5_250k'].isna().sum()),
      '-> standardise gives NaN:', bool(np.isnan(M.standardise(J['v5_250k'].values)).all()))
print('index overlap: series.pkl rows 1999-01-04..2026-07-10', len(old.loc['1999-01-04':'2026-07-10']), ' series_variantes rows', len(S),
      ' same dates:', old.loc['1999-01-04':'2026-07-10'].index.equals(S.index))
