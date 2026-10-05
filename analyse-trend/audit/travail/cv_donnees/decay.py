"""P&L decay with extra execution delay, per market group: markets whose backtest P&L relies on
short-horizon autocorrelation (stale/limit/smoothed settlement) lose P&L much faster when delayed."""
import numpy as np, pandas as pd, core
b = core.v5()
pos, R = b['positions'], core.rets.fillna(0)
TIER_B = ['LH1 Comdty', 'FC1 Comdty', 'CT1 Comdty', 'RS1 Comdty', 'SFI5 Comdty', 'IK1 Comdty', 'MWE1 Comdty']
LIQ_COM = ['CL1 Comdty', 'CO1 Comdty', 'NG1 Comdty', 'HO1 Comdty', 'XB1 Comdty', 'GC1 Comdty', 'SI1 Comdty', 'HG1 Comdty',
           'C 1 Comdty', 'S 1 Comdty', 'W 1 Comdty', 'SB1 Comdty', 'KC1 Comdty', 'LC1 Comdty', 'SM1 Comdty', 'BO1 Comdty']
groups = {'CUA1': ['CUA1 Comdty'], 'SCO1': ['SCO1 Comdty'], 'tierB(7)': TIER_B, 'liquid_commod(16)': LIQ_COM,
          'other(59)': [c for c in R.columns if c not in TIER_B + LIQ_COM + ['CUA1 Comdty', 'SCO1 Comdty']]}
rows = []
for start in ['1990-01-01', '2010-01-01']:
    for g, cols in groups.items():
        r = dict(start=start[:4], group=g)
        base = None
        for k in [1, 2, 3, 6, 11]:
            p = (pos[cols].shift(k) * R[cols]).sum(axis=1).loc[start:].mean() * 261 * 100
            if k == 1: base = p
            r[f'gross_lag{k}'] = p
        for k in [2, 3, 6, 11]: r[f'kept_lag{k}'] = r[f'gross_lag{k}'] / base
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(core.OUT + 'pnl_decay_by_group.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250); print(df.round(3).to_string(index=False))
st = pd.read_csv(core.OUT + 'stale_scan_full.csv', index_col=0)
print(st.loc[TIER_B + ['CUA1 Comdty', 'SCO1 Comdty'], ['vr5_2010', 'vr20_2010', 'ac1_2010', 'ac2_2010', 'gross_pnl_2010_pct', 'gross_pnl_1990_pct']].round(3).to_string())
print('median VR5 liquid commodities %.2f, others %.2f' % (st.loc[LIQ_COM, 'vr5_2010'].median(), st.loc[groups['other(59)'], 'vr5_2010'].median()))
