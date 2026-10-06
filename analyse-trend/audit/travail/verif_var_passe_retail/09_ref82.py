"""(9) Fairer reference for the retail-vs-published comparison: each variant on the 82 tradable markets (no CUA1/SCO1),
half-day lag, fractional, project bp costs (the audit's 'tradable restatement', so far only for v5)."""
import numpy as np, pandas as pd
import indep as I
ALL = list(I.RETS.columns)
C82 = [c for c in ALL if c not in ('CUA1 Comdty', 'SCO1 Comdty')]
idx = I.RETS.loc['1990-01-01':'2026-07-10'].index
tb = I.rb.load_close(I.SCR + 'mkt/us-etf/alm0421_macro/DTB3.csv', 'value') / 100
r = tb.reindex(idx.union(tb.index)).ffill().reindex(idx).shift(1).bfill()
TB = r * pd.Series(idx.to_series().diff().dt.days.fillna(1).values, index=idx) / 365
P = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-10'}
rows = []
for v in I.SPEC:
    for lab, cols, lag in [('published84_same_close', ALL, 1.0), ('tradable82_half_day_lag', C82, 1.5)]:
        n = I.system(v, cols, lag)['net'].reindex(idx)
        for p, a in P.items():
            x = n.loc[a:'2026-07-10']
            rows.append(dict(variant=v, ref=lab, period=p, sharpe16=I.sr(x), vol=x.std() * 16, cagr_excess=I.cagr(x),
                             cagr_total_tbill=I.cagr(x + TB.loc[a:'2026-07-10'])))
R = pd.DataFrame(rows)
R.to_csv('reference_84_vs_82lag.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
print(R.pivot_table(index=['variant', 'ref'], columns='period', values='sharpe16').round(3).to_string())
print(R.pivot_table(index=['variant', 'ref'], columns='period', values='cagr_total_tbill').mul(100).round(1).to_string())
