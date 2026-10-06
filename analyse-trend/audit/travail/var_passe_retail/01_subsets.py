"""Per-variant feasible market subsets at USD 100k / 250k / 1M, with the audit rule:
1. capital/tiers.py greedy: strict retail-accessible universe (specs.csv liquid_access == 'OK', 66 markets, already
   without CUA1 / SCO1), smallest liquid contract, k = liquid notional x weight(84) x 2.5 / median |position| of THIS
   variant in the 84-market run over 2023-07-10 -> 2026-07-10; round-robin across the 6 risk classes (classes ordered
   by their cheapest market, cheapest first within a class); a market is added only if every market still holds
   >= 1 contract with weights and IDM recomputed for the enlarged set.
2. simulation/run_all.py verified_subset: re-run the variant on the subset, compute each market's median |position|
   (2023-07-10 -> 2026-07-10) x C / today's liquid-contract notional, drop the worst market until all are >= 1.
For v5 this must reproduce capital/tiers.csv and simulation/subsets.csv (14 / 24 / 40 markets)."""
import sys, pickle
import numpy as np, pandas as pd
HERE = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/var_passe_retail/'
sys.path.insert(0, HERE)
import veng as V
sys.path.insert(0, V.SIMDIR)
import sim  # noqa: E402
import trend  # noqa: E402

AUD = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
sp = pd.read_csv(AUD + 'capital/specs.csv', index_col=0)
risk = sp['group'].replace({'Bonds': 'Taux', 'STIR': 'Taux'})
strict = sp['liquid_access'].eq('OK')
ref = pickle.load(open(HERE + 'ref84.pkl', 'rb'))
NLIQ_TODAY = sim.TAB['liquid'][1].iloc[-1]
TIERS = [100e3, 250e3, 1e6]
W0, W1 = '2023-07-10', '2026-07-10'


def greedy(universe, k, C, m=1):
    u = pd.DataFrame({'k': k[universe], 'cls': risk[universe]})
    order_in = {c: list(g.sort_values('k').index) for c, g in u.groupby('cls')}
    cls_order = sorted(order_in, key=lambda c: u.loc[order_in[c][0], 'k'])
    seq = []
    while any(order_in.values()):
        for c in cls_order:
            if order_in[c]:
                seq.append(order_in[c].pop(0))
    chosen = []
    for t in seq:
        trial = chosen + [t]
        cl = risk[trial]; ncl = cl.nunique(); cnt = cl.value_counts()
        w = pd.Series({x: 1 / ncl / cnt[cl[x]] for x in trial})
        contracts = C * w * trend.idm_for(len(trial)) / k[trial]
        if (contracts >= m).all():
            chosen = trial
    return chosen


def verified(v, cols, C, m=1):
    dropped = []
    while True:
        s = V.system(v, cols)
        med = s['pos'].loc[W0:W1, cols].abs().median()
        nc = med * C / NLIQ_TODAY[cols]
        if (nc >= m).all() or len(cols) <= 1:
            return cols, nc, dropped
        worst = nc.idxmin(); dropped.append(worst); cols = [c for c in cols if c != worst]


short = {'Equities': 'Eq', 'Taux': 'Rates', 'FX': 'FX', 'Energy': 'Ener', 'Metals': 'Met', 'Agriculture': 'Ag'}
tiers_audit = pd.read_csv(AUD + 'capital/tiers.csv').set_index('capital_usd')
subs_audit = pd.read_csv(AUD + 'simulation/subsets.csv')
rows, SUBS = [], {}
for v in V.VARIANTS:
    med84 = ref[v]['pos'].loc[W0:W1].abs().median().reindex(sp.index)
    k = sp['liquid_notional_usd'] * sp['weight'] * 2.5 / med84
    for C in TIERS:
        g = greedy(strict, k, C)
        cols, nc, dropped = verified(v, list(g), C)
        # same rule checked on the half-day-lag system actually simulated (v4/v5 scale measured on lagged P&L)
        sl = V.system(v, cols, lag=1.5)
        nc_lag = sl['pos'].loc[W0:W1, cols].abs().median() * C / NLIQ_TODAY[cols]
        SUBS[(v, C)] = cols
        tag = f'{v}_{int(C / 1e3)}k' if C < 1e6 else f'{v}_1M'
        rows.append(dict(key=tag, variant=v, variant_name=V.VARIANTS[v]['name'], capital_usd=int(C), n_markets=len(cols),
                         n_greedy=len(g), classes=', '.join(f'{short[a]}:{b}' for a, b in risk[cols].value_counts().items()),
                         markets=' | '.join(cols), median_contracts=' | '.join(f'{c}:{nc[c]:.1f}' for c in cols),
                         min_median_contracts=round(nc.min(), 2), min_median_contracts_lag05=round(nc_lag.min(), 2),
                         dropped_after_rerun=' | '.join(dropped),
                         med_abs_pos84_vs_v5=round((med84 / ref['v5']['pos'].loc[W0:W1].abs().median().reindex(sp.index)).median(), 3)))
        print(tag, len(g), '->', len(cols), 'dropped', dropped, 'min nc', round(nc.min(), 2), 'lag', round(nc_lag.min(), 2), flush=True)
        if v == 'v5':
            ga = tiers_audit.loc[C, 'strict_subset_ge1c_markets'].split(' | ')
            va = subs_audit[(subs_audit.capital_usd == C) & subs_audit.rule.str.startswith('>=1')].markets.iloc[0].split(' | ')
            print('   v5 audit check: greedy identical', ga == list(g), '; verified identical', va == cols)
df = pd.DataFrame(rows)
df.to_csv(HERE + 'subsets_variantes.csv', index=False)
pickle.dump(SUBS, open(HERE + 'subsets_variantes.pkl', 'wb'))
# overlap table
for C in TIERS:
    base = set(SUBS[('v5', C)])
    print(int(C), {v: (len(SUBS[(v, C)]), len(set(SUBS[(v, C)]) & base)) for v in V.VARIANTS})
