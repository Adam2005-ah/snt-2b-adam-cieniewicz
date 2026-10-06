"""P&L contribution by market (whole contracts, half-day lag, gross of costs) for v5 and v2 tiers over 2023-07 -> 2026-07
and 2010 -> 2026, to explain tier-to-tier dispersion (composition luck)."""
import sys, pickle
import numpy as np, pandas as pd
HERE = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/var_passe_retail/'
sys.path.insert(0, HERE)
import veng as V
sys.path.insert(0, V.SIMDIR)
import sim
sim.CC.loc['ER4 Comdty', 'hs'] = 0.0025
tab, N, HS = sim.contract_table('liquid_mult')
N = pd.DataFrame(np.tile(N.iloc[-1].values, (len(N), 1)), index=N.index, columns=N.columns)
EX = pickle.load(open(HERE + 'series_variantes_extra.pkl', 'rb'))
SUBS = pickle.load(open(HERE + 'subsets_variantes.pkl', 'rb'))
R = V.RETS.reindex(sim.IDX)
rows = []
for k in ['v5_100k', 'v5_250k', 'v5_1M', 'v2_100k', 'v2_250k', 'v2_1M']:
    v, t = k.split('_'); C = {'100k': 100e3, '250k': 250e3, '1M': 1e6}[t]
    cols = SUBS[(v, C)]
    s = V.system(v, cols, lag=1.5)
    n = sim.integer_positions(s, cols, C, N)
    frac = n * N[cols] / C
    held = (0.5 * frac.shift(1) + 0.5 * frac.shift(2)).fillna(0)
    pnl = held * R[cols].fillna(0)
    for per, (a, b) in {'2010-2026': ('2010-01-01', '2026-07-10'), '2023-07_2026-07': ('2023-07-10', '2026-07-10')}.items():
        yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        c = pnl.loc[a:b].sum() / yrs * 100
        cls = c.groupby(V.RISK[cols]).sum()
        top = c.sort_values()
        rows.append(dict(key=k, period=per, gross_pct_yr=round(c.sum(), 2), **{f'cls_{x}': round(y, 2) for x, y in cls.items()},
                         best3=' | '.join(f'{i}:{top[i]:.1f}' for i in top.index[::-1][:3]),
                         worst3=' | '.join(f'{i}:{top[i]:.1f}' for i in top.index[:3])))
df = pd.DataFrame(rows); pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 80)
print(df.to_string(index=False))
df.to_csv(HERE + 'contrib_par_marche.csv', index=False)
