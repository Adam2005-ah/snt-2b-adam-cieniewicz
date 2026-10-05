"""Task 5: tradable-universe restatement of the v5 headline."""
import numpy as np, pandas as pd, core
ALL = list(core.rets.columns)
TIER_B = ['LH1 Comdty', 'FC1 Comdty', 'CT1 Comdty', 'RS1 Comdty', 'SFI5 Comdty', 'IK1 Comdty', 'MWE1 Comdty']
UNI = {'A_backtest_84': [], 'B_ex_CUA1': ['CUA1 Comdty'], 'C_ex_CUA1_SCO1': ['CUA1 Comdty', 'SCO1 Comdty'],
       'D_ex_CUA1_SCO1_tierB': ['CUA1 Comdty', 'SCO1 Comdty'] + TIER_B}
LAGS = {'same_close(1.0)': 1.0, 'auto_0.25d(1.25)': 1.25, 'manual_0.5d(1.5)': 1.5, 'full_day(2.0)': 2.0}
PER = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-10'}
tb = core.tbill_daily(core.rets.index)
rows = []
for u, drop in UNI.items():
    cols = [c for c in ALL if c not in drop]
    for ll, L in LAGS.items():
        res = core.v5(cols, lag=L)
        for k, d in PER.items():
            rows.append(dict(universe=u, n=len(cols), execution=ll, start=k, **core.perf(res['net'], d, tb),
                             sharpe16=core.sr(res['net'].loc[d:], 256)))
df = pd.DataFrame(rows); df.to_csv(core.OUT + 'restatement_grid.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
piv = df.pivot_table(index=['universe', 'execution'], columns='start', values='sharpe', sort=False)
print('SHARPE (excess, sqrt(261))'); print(piv.round(3).to_string())
for v in ['cagr_excess', 'cagr_total', 'maxdd_excess', 'vol']:
    print(v); print(df[df.start.isin(['1990', '2000', '2010'])].pivot_table(index=['universe', 'execution'], columns='start', values=v, sort=False).round(3).to_string())
tbm = {k: tb.loc[d:].mean() * 261 for k, d in PER.items()}
print('average T-bill accrual %/yr by start:', {k: round(v * 100, 2) for k, v in tbm.items()})
