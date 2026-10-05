import pandas as pd, core
U = pd.read_pickle(core.S + 'audit/simulation/subsets.pkl')
rows = []
for C in [100000.0, 250000.0, 1000000.0]:
    cols = U[('sub', C, 1)]
    for L in [1.0, 1.25, 1.5, 2.0]:
        n = core.v5(cols, lag=L)['net']
        rows.append(dict(capital=C, n=len(cols), lag=L, has_CUA1_or_SCO1=bool({'CUA1 Comdty', 'SCO1 Comdty'} & set(cols)),
                         **{k: core.sr(n.loc[d:]) for k, d in core.PERIODS.items()}))
df = pd.DataFrame(rows); df.to_csv(core.OUT + 'subsets_lag.csv', index=False, float_format='%.4f')
print(df.round(3).to_string(index=False))
