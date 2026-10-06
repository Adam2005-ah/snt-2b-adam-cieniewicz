"""Forward Monte Carlo, 5 variants x 3 tiers, 4 seeds x 10,000 paths, 5 and 10 years, net of all fees.
Engine: mc_engine.sim_diy (copy of cv_avant_impot/mc_engine.py), called with pfu=0 and cash_tax=0 exactly as the verified
variant-5 run; mode 'ibkr' (margin unpaid, IBKR rate ESTR-0.5 % above USD 10k x min(NAV/100k,1), rest of NAV in a
money-market fund at ESTR-0.12 %), fixed costs USD 400/yr, ESTR 2.4 %, inflation 2 %.
Shocks: each key's own daily series (series_variantes.pkl, 1999-01-04..2026-07-10, 7,180 days) standardised, circular
stationary bootstrap, mean block 126 d. Within a seed every key uses the SAME index matrix (common random numbers), so
differences between variants/tiers are not seed noise.
Outputs: raw_runs.csv (one row per seed x key x case x scenario x horizon)."""
import os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_engine as M

AUD = os.path.dirname(HERE)
SER = os.path.join(AUD, 'var_passe_retail', 'series_variantes.pkl')
NP, NY, BLOCK = 10000, 10, 126
ESTR, INFL, FIXED = 0.024, 0.02, 400.0
SEEDS = (101, 202, 303, 404)
DSR = 0.20
P = pd.read_csv(os.path.join(HERE, 'params.csv'), index_col=0)


def run_seed(seed):
    t0 = time.time()
    S = pd.read_pickle(SER)
    rng = np.random.default_rng(seed)
    idx = M.sb_idx(len(S), NP, NY * M.AN, BLOCK, rng)
    rows = []

    def go(z, p, case, scen, sr, m, c, cash_model):
        res = M.sim_diy(z, sr, p.vol, p.K0, ESTR, years=(5, 10), mode='ibkr', fixed_usd=FIXED, c=c, m=m,
                        infl=INFL, pfu=0.0, cash_tax=0.0)
        for y, r in res.items():
            d = dict(seed=seed, key=key, variant=p.variant, tier=p.tier, case=case, scen=scen, sr=sr, vol=p.vol,
                     m=m, c=c, cash_model=cash_model, horizon=y)
            d.update(r)
            rows.append(d)

    for key, p in P.iterrows():
        z = M.standardise(S[key].values)[idx]
        for scen, d in (('pess', -DSR), ('central', 0.0), ('opt', DSR)):
            go(z, p, 'commune', scen, p.sr_central_case1 + d, p.m, p.c, 'expo_brute')
            if p.variant in ('v1', 'v2'):
                go(z, p, 'v1v2_sans_selection', scen, p.sr_central_case2 + d, p.m, p.c, 'expo_brute')
        go(z, p, 'commune', 'central', p.sr_central_case1, p.m_contracts, p.c_contracts, 'marge_contrats')
        del z
        print('seed', seed, key, round(time.time() - t0, 1), flush=True)
    return rows


if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(run_seed, SEEDS)
    R = pd.DataFrame([r for rows in out for r in rows])
    R = R.rename(columns={'at_med': 'cagr_med', 'at_p10': 'cagr_p10', 'at_p90': 'cagr_p90',
                          'at_mean_wealth': 'cagr_mean_wealth', 'cash_at': 'cash_cagr'})
    R.to_csv(os.path.join(HERE, 'raw_runs.csv'), index=False)
    print('done', R.shape, round(time.time() - t0, 1))
