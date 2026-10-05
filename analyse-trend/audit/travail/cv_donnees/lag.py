"""Task 2: execution lag - by class, significance, and on the cleaned universe."""
import numpy as np, pandas as pd, core
ALL = list(core.rets.columns)
CLEAN = [c for c in ALL if c != 'CUA1 Comdty']
out = []
for lab, cols in [('all84', ALL), ('ex_CUA1', CLEAN)]:
    r1 = core.v5(cols, lag=1.0); r2 = core.v5(cols, lag=2.0)
    for per, d in core.PERIODS.items():
        dd = (r1['net'] - r2['net']).loc[d:]
        g1 = r1['inst_gross'].loc[d:]; g2 = r2['inst_gross'].loc[d:]
        diff_cls = (g1 - g2).T.groupby(core.groups[cols]).sum().T.mean() * 261 * 100
        row = dict(universe=lab, period=per, lag_cost_pct=dd.mean() * 261 * 100, t_stat=dd.mean() / dd.std() * np.sqrt(len(dd)),
                   dSR=core.sr(r2['net'].loc[d:]) - core.sr(r1['net'].loc[d:]))
        row.update({f'cost_{k}': v for k, v in diff_cls.items()})
        out.append(row)
df = pd.DataFrame(out)
df.to_csv(core.OUT + 'lag_by_class.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
print(df.round(3).to_string(index=False))
# relation market AC1 vs lag cost (since 2010) and share of lag cost from top AC markets
st = pd.read_csv(core.OUT + 'stale_scan_full.csv', index_col=0)
print('cross-market corr(ac1_2010, lag_cost_2010_pct) = %.2f; corr(ac1_2010 x avg_abs_pos, lag cost) = %.2f' % (
    st['ac1_2010'].corr(st['lag_cost_2010_pct']), (st['ac1_2010'] * st['avg_abs_pos_2010']).corr(st['lag_cost_2010_pct'])))
print('lag cost 2010 %/yr: total', round(st.lag_cost_2010_pct.sum(), 3), '; markets with ac1>0.05:', round(st.loc[st.ac1_2010 > 0.05, 'lag_cost_2010_pct'].sum(), 3),
      '; top 10 markets:', st.lag_cost_2010_pct.sort_values(ascending=False).head(10).round(3).to_dict())
