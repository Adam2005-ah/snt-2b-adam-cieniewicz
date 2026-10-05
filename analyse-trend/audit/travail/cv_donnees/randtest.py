"""Task 3/4: composition-luck test - random subsets with identical risk-class counts, drawn from the 'strict' liquid
universe (same pool the capital subsets were drawn from) and from all 84."""
import numpy as np, pandas as pd, core
rng = np.random.default_rng(20261005)
SP = pd.read_csv(core.S + 'audit/capital/specs.csv', index_col=0)
STRICT = list(SP.index[SP['liquid_access'].eq('OK')])
ALL = list(core.rets.columns)
U = pd.read_pickle(core.S + 'audit/simulation/subsets.pkl')
OPS20 = ['ES1 Index','VG1 Index','NO1 Index','HI1 Index','TU1 Comdty','TY1 Comdty','RX1 Comdty','DU1 Comdty','SFR5 Comdty',
         'EC1 Curncy','JY1 Curncy','BP1 Curncy','AD1 Curncy','CL1 Comdty','NG1 Comdty','GC1 Comdty','HG1 Comdty','C 1 Comdty','S 1 Comdty','LC1 Comdty']
TARGETS = {'sub100k_14': U[('sub', 100000.0, 1)], 'sub250k_24': U[('sub', 250000.0, 1)], 'sub1M_40': U[('sub', 1000000.0, 1)], 'ops20': OPS20}
PER = {'2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-10'}
N = 400

def stats(cols):
    n = core.v5(cols, start='2007-01-01')['net']
    return {k: core.sr(n.loc[d:]) for k, d in PER.items()}

rows, draws_all = [], []
for lab, cols in TARGETS.items():
    act = stats(cols)
    cnt = core.risk[cols].value_counts()
    for pool_lab, pool in [('strict66', STRICT), ('all84', ALL)]:
        pool_cls = core.risk[pool]
        sims = []
        for i in range(N):
            pick = []
            for cl, k in cnt.items():
                cand = list(pool_cls.index[pool_cls == cl])
                pick += list(rng.choice(cand, size=min(k, len(cand)), replace=False))
            s = stats(pick); s['draw'] = i; s['picks'] = '|'.join(sorted(pick))
            sims.append(s)
        sims = pd.DataFrame(sims); sims['target'] = lab; sims['pool'] = pool_lab
        draws_all.append(sims)
        r = dict(target=lab, pool=pool_lab, n=len(cols))
        for k in PER:
            r[f'actual_{k}'] = act[k]; r[f'rand_med_{k}'] = sims[k].median(); r[f'rand_p05_{k}'] = sims[k].quantile(.05)
            r[f'rand_p95_{k}'] = sims[k].quantile(.95); r[f'pctile_{k}'] = (sims[k] < act[k]).mean()
        rows.append(r)
        print(lab, pool_lab, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
out = pd.DataFrame(rows); out.to_csv(core.OUT + 'composition_luck.csv', index=False, float_format='%.4f')
pd.concat(draws_all).to_csv(core.OUT + 'composition_luck_draws.csv', index=False, float_format='%.4f')
