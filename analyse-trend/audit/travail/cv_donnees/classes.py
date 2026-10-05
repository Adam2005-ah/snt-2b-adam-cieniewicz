"""Task 3: gross P&L by asset class (v5 and subsets) and composition-luck tests."""
import numpy as np, pandas as pd, core
ALL = list(core.rets.columns)
SUBS = pd.read_pickle(core.S + 'audit/simulation/subsets.pkl')
OPS20 = ['ES1 Index','VG1 Index','NO1 Index','HI1 Index','TU1 Comdty','TY1 Comdty','RX1 Comdty','DU1 Comdty','SFR5 Comdty',
         'EC1 Curncy','JY1 Curncy','BP1 Curncy','AD1 Curncy','CL1 Comdty','NG1 Comdty','GC1 Comdty','HG1 Comdty','C 1 Comdty','S 1 Comdty','LC1 Comdty']
U = {'all84': ALL, 'sub100k_14': SUBS[('sub', 100000.0, 1)], 'sub250k_24': SUBS[('sub', 250000.0, 1)],
     'sub1M_40': SUBS[('sub', 1000000.0, 1)], 'ops20': OPS20}
P = {'2010': '2010-01-01', '2023-07': '2023-07-10', '2010_to_2023-06': ('2010-01-01', '2023-07-07')}
rows, srs = [], []
res_cache = {}
for lab, cols in U.items():
    res = core.v5(cols); res_cache[lab] = res
    for per, d in P.items():
        sl = slice(*d) if isinstance(d, tuple) else slice(d, None)
        g = res['inst_gross'].loc[sl]
        cls = g.T.groupby(core.groups[cols]).sum().T.mean() * 261 * 100
        r = dict(universe=lab, period=per, **cls.to_dict())
        r['TOTAL_gross'] = cls.sum(); r['costs'] = -res['cost'].loc[sl].mean() * 261 * 100; r['net'] = res['net'].loc[sl].mean() * 261 * 100
        r['SR_net'] = core.sr(res['net'].loc[sl]); r['SR_gross'] = core.sr(res['gross'].loc[sl])
        rows.append(r)
    comp = core.groups[cols].value_counts().to_dict()
    srs.append(dict(universe=lab, n=len(cols), comp=comp, **{f'SR_{k}': v for k, v in core.sharpes(res['net']).items()}))
df = pd.DataFrame(rows)
df.to_csv(core.OUT + 'pnl_by_class_subsets.csv', index=False, float_format='%.3f')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
print(df.round(2).to_string(index=False))
print(pd.DataFrame(srs).round(3).to_string(index=False))
pd.to_pickle(res_cache, core.OUT + 'subset_runs.pkl')
