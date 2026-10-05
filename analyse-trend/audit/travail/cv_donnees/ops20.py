"""Task 4: the operations agent's hard-coded 20-market subset - recompute and stress the choice."""
import numpy as np, pandas as pd, core
OPS20 = ['ES1 Index','VG1 Index','NO1 Index','HI1 Index','TU1 Comdty','TY1 Comdty','RX1 Comdty','DU1 Comdty','SFR5 Comdty',
         'EC1 Curncy','JY1 Curncy','BP1 Curncy','AD1 Curncy','CL1 Comdty','NG1 Comdty','GC1 Comdty','HG1 Comdty','C 1 Comdty','S 1 Comdty','LC1 Comdty']
tb = core.tbill_daily(core.rets.index)
PER4 = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01', '2023-01': '2023-01-01', '2023-07': '2023-07-10'}
rows = []
ref = pd.read_pickle(core.S + 'audit/operations/v5sub_net.pkl')
for L in [1.0, 1.5, 2.0]:
    for cm in [1.0, 1.5]:
        res = core.v5(OPS20, lag=L, cost_mult=cm)
        if L == 1.0 and cm == 1.0:
            print('max diff vs operations v5sub_net.pkl:', (res['net'] - ref).abs().max())
            print('ops convention (x16):', {k: round(core.sr(res['net'].loc[d:], 256), 3) for k, d in PER4.items()})
            pd.to_pickle(res['net'], core.OUT + 'ops20_net.pkl')
        for k, d in PER4.items():
            p = core.perf(res['net'], d, tb)
            rows.append(dict(lag=L, cost_mult=cm, start=k, **p, costs_pct=res['cost'].loc[d:].mean() * 261 * 100))
df = pd.DataFrame(rows); df.to_csv(core.OUT + 'ops20_perf.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
print(df[(df.cost_mult == 1.0)].round(3).to_string(index=False))
# 84-market comparison with T-bill for the same periods
res84 = core.v5()
print('all84:', pd.DataFrame({k: core.perf(res84['net'], d, tb) for k, d in PER4.items()}).round(3).to_string())
# drawdown episodes
n = df  # noqa
r = core.v5(OPS20)['net'].loc['1990':]
c = (1 + r).cumprod(); dd = c / c.cummax() - 1
print('ops20 worst DD since 1990: %.3f at %s (peak %s)' % (dd.min(), dd.idxmin().date(), c.loc[:dd.idxmin()].idxmax().date()))
print('ops20 DD on 2026-07-10: %.3f' % dd.iloc[-1])
# swap sensitivity: replace hand-picked members by the next most-traded peer
SWAPS = {'LC1->W 1': ('LC1 Comdty', 'W 1 Comdty'), 'LC1->SB1': ('LC1 Comdty', 'SB1 Comdty'), 'LC1->SM1': ('LC1 Comdty', 'SM1 Comdty'),
         'NG1->CO1': ('NG1 Comdty', 'CO1 Comdty'), 'HG1->SI1': ('HG1 Comdty', 'SI1 Comdty'), 'HI1->NQ1': ('HI1 Index', 'NQ1 Index'),
         'HI1->GX1': ('HI1 Index', 'GX1 Index'), 'DU1->FV1': ('DU1 Comdty', 'FV1 Comdty'), 'TU1->US1': ('TU1 Comdty', 'US1 Comdty'),
         'AD1->CD1': ('AD1 Curncy', 'CD1 Curncy'), 'GC1->SI1': ('GC1 Comdty', 'SI1 Comdty')}
srow = []
base = core.v5(OPS20)['net']
srow.append(dict(variant='ops20 as built', **{k: core.sr(base.loc[d:]) for k, d in PER4.items()}))
for lab, (a, b) in SWAPS.items():
    cols = [b if x == a else x for x in OPS20]
    n = core.v5(cols)['net']
    srow.append(dict(variant=lab, **{k: core.sr(n.loc[d:]) for k, d in PER4.items()}))
# a documented ex-ante volume rule (FIA-style most-traded per class, same class counts)
VOL20 = ['ES1 Index', 'NQ1 Index', 'VG1 Index', 'NO1 Index', 'SFR5 Comdty', 'TY1 Comdty', 'FV1 Comdty', 'TU1 Comdty', 'RX1 Comdty',
         'EC1 Curncy', 'JY1 Curncy', 'BP1 Curncy', 'AD1 Curncy', 'CL1 Comdty', 'CO1 Comdty', 'GC1 Comdty', 'SI1 Comdty',
         'C 1 Comdty', 'S 1 Comdty', 'SB1 Comdty']
n = core.v5(VOL20)['net']
srow.append(dict(variant='volume-ranked 20 (same class counts)', **{k: core.sr(n.loc[d:]) for k, d in PER4.items()}))
sw = pd.DataFrame(srow); sw.to_csv(core.OUT + 'ops20_swaps.csv', index=False, float_format='%.4f')
print(sw.round(3).to_string(index=False))
