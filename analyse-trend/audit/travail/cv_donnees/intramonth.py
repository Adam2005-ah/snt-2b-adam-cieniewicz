import numpy as np, pandas as pd, core
R = core.rets
rows = []
for c in R.columns:
    s = R[c].dropna().loc['2010':]
    if len(s) < 500: s = R[c].dropna()
    df = pd.DataFrame({'r': s}); ym = df.index.to_period('M')
    bfe = df.groupby(ym).cumcount(ascending=False); bfs = df.groupby(ym).cumcount()
    rows.append(dict(mkt=c, group=core.groups[c], std_last1to4_over_first5=df.r[bfe.between(1, 4)].std() / df.r[bfs.between(0, 4)].std()))
im = pd.DataFrame(rows).set_index('mkt').sort_values('std_last1to4_over_first5')
im.to_csv(core.OUT + 'intramonth_vol_ratio.csv', float_format='%.3f')
print(im.head(8).round(2).to_string()); print('median all:', round(im.iloc[:, 1].median(), 2))
st = pd.read_csv(core.OUT + 'stale_scan_full.csv', index_col=0)
LIQ = ['CL1 Comdty', 'CO1 Comdty', 'NG1 Comdty', 'HO1 Comdty', 'GC1 Comdty', 'SI1 Comdty', 'HG1 Comdty', 'C 1 Comdty', 'S 1 Comdty', 'W 1 Comdty', 'SB1 Comdty', 'KC1 Comdty']
cols = ['zero_share_2010', 'zero_share_specific_2010', 'longest_zero_run_listing', 'ac1_2010', 'ac1_t_2010', 'vr5_2010', 'vr20_2010', 'gross_pnl_2010_pct', 'gross_pnl_2010_sr', 'share_of_total_gross_2010', 'lag_cost_2010_pct', 'gross_pnl_2023_pct']
tab = st.loc[['CUA1 Comdty', 'SCO1 Comdty'] + LIQ, cols].copy()
tab.loc['median liquid (12)'] = st.loc[LIQ, cols].median()
tab.to_csv(core.OUT + 'cua1_sco1_vs_liquid.csv', float_format='%.4f')
pd.set_option('display.width', 250); print(tab.round(3).to_string())
