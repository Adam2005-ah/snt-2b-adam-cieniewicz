import sys, pickle, time
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation')
import numpy as np, pandas as pd
import sim
E = sim.E
CAPS = [50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6, 20e6]
TIERS = pd.read_csv(sim.S + 'audit/capital/tiers.csv').set_index('capital_usd')
STRICT = list(sim.SP.index[sim.SP['liquid_access'].eq('OK')])
ALL = list(E.RETS.columns)
NLIQ_TODAY = sim.TAB['liquid'][1].iloc[-1]

def verified_subset(C, m):
    """Greedy round-robin subset from the capital audit (tiers.csv), then re-run variant 5 on it and drop, one at a
    time, any market whose median |position| (2023-07-10 -> 2026-07-10) is below m contracts at today's notional."""
    cols = TIERS.loc[C, f'strict_subset_ge{m}c_markets'].split(' | ')
    dropped = []
    while True:
        s = E.system(cols)
        med = s['pos'].loc['2023-07-10':'2026-07-10', cols].abs().median()
        nc = med * C / NLIQ_TODAY[cols]
        if (nc >= m).all() or len(cols) <= 1:
            return cols, s, nc, dropped
        worst = nc.idxmin(); dropped.append(worst); cols = [c for c in cols if c != worst]

rows, subsets, sysc = [], [], {}
t0 = time.time()
for C in CAPS:
    for mode in ['hist', 'today']:
        sfx = '' if mode == 'hist' else '_today'
        r, _, _ = sim.evaluate('a_all84_smallest', ALL, C, sim.IDEAL, 'small' + sfx, extra=dict(notional=mode))
        rows += r
        key = ('strict66',)
        if key not in sysc: sysc[key] = E.system(STRICT)
        r, _, _ = sim.evaluate('strict66_liquid', STRICT, C, sysc[key], 'liquid' + sfx, extra=dict(notional=mode))
        rows += r
    for m in (1, 4):
        cols, s, nc, dropped = verified_subset(C, m)
        sysc[('sub', C, m)] = (cols, s)
        subsets.append(dict(capital_usd=C, rule=f'>={m} contract(s) at median', n=len(cols),
                            classes=', '.join(f'{k}:{v}' for k, v in E.RISK[cols].value_counts().items()),
                            markets=' | '.join(cols), median_contracts=' | '.join(f'{c}:{nc[c]:.1f}' for c in cols),
                            dropped_after_rerun=' | '.join(dropped)))
        for mode in ['hist', 'today']:
            sfx = '' if mode == 'hist' else '_today'
            r, _, _ = sim.evaluate(f'b_strict_subset_ge{m}c', cols, C, s, 'liquid' + sfx, extra=dict(notional=mode))
            rows += r
    print(int(C), round(time.time() - t0), flush=True)

# weekly rebalancing, best subsets at 100k and 250k (both subset rules, both notional modes)
for C in [100e3, 250e3]:
    for m in (1, 4):
        cols, sd = sysc[('sub', C, m)]
        sw = E.system(cols, freq='W')
        for mode in ['hist', 'today']:
            sfx = '' if mode == 'hist' else '_today'
            r, _, _ = sim.evaluate(f'b_strict_subset_ge{m}c', cols, C, sw, 'liquid' + sfx, freq='W', extra=dict(notional=mode))
            rows += r
# weekly for the full 84 too (reference) at 1M
sw84 = E.system(ALL, freq='W')
for C in [250e3, 1e6]:
    for mode in ['hist', 'today']:
        sfx = '' if mode == 'hist' else '_today'
        r, _, _ = sim.evaluate('a_all84_smallest', ALL, C, sw84, 'small' + sfx, freq='W', extra=dict(notional=mode))
        rows += r

res = pd.DataFrame(rows)
first = ['universe', 'notional', 'capital_usd', 'freq', 'period', 'n_markets']
res = res[first + [c for c in res.columns if c not in first]]
res.to_csv(sim.OUT + 'results.csv', index=False)
pd.DataFrame(subsets).to_csv(sim.OUT + 'subsets.csv', index=False)
pickle.dump({k: v[0] for k, v in sysc.items() if k[0] == 'sub'}, open(sim.OUT + 'subsets.pkl', 'wb'))
# contract specs actually used
for k in ['small', 'liquid']:
    tab, N, HS = sim.TAB[k]
    t = tab.copy(); t['notional_usd_1999'] = N.iloc[0].round(0); t['notional_usd_2010'] = N.loc['2010-01-04'].round(0)
    t['notional_usd_today'] = N.iloc[-1].round(0); t['halfspread_usd_today'] = HS.iloc[-1].round(2)
    t.to_csv(sim.OUT + f'contracts_{k}.csv')
print('done', round(time.time() - t0))
