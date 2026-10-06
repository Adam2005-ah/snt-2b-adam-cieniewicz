"""(2) independent subset selection for all 5 variants x 3 tiers: greedy round-robin (strict 66, liquid contract,
k from each variant's own 84-market median |position| 2023-07-10..2026-07-10) + re-run/drop loop with my engine, then
the >=1-contract rule checked with each variant's OWN subset positions, same-close and half-day lag."""
import time, pickle
import numpy as np, pandas as pd
from indep import *
t0 = time.time()
W0, W1 = '2023-07-10', '2026-07-10'
ALL = list(RETS.columns)
assert 'CUA1 Comdty' not in STRICT and 'SCO1 Comdty' not in STRICT and len(STRICT) == 66
# 84-system weight at the end and IDM for 84
live_end = RETS.loc[:W1].notna().cumsum().iloc[-1] > trend.WARMUP_DAYS
w84 = pd.Series({c: 1 / RISK.nunique() / (RISK == RISK[c]).sum() for c in ALL})
print('all 84 live at end:', live_end.all(), ' idm84', trend.idm_for(84), ' w84 vs specs max diff', (w84 - SP['weight']).abs().max())
prev = pd.read_csv(A + 'var_passe_retail/subsets_variantes.csv').set_index('key')


def greedy(k, C):
    u = pd.DataFrame({'k': k[STRICT], 'cls': RISK[STRICT]})
    queues = {g: list(d.sort_values('k').index) for g, d in u.groupby('cls')}
    order = sorted(queues, key=lambda g: u.loc[queues[g][0], 'k'])
    seq = []
    while any(queues.values()):
        for g in order:
            if queues[g]:
                seq.append(queues[g].pop(0))
    chosen = []
    for m in seq:
        trial = chosen + [m]
        cl = RISK[trial]
        w = np.array([1 / cl.nunique() / (cl == cl[x]).sum() for x in trial])
        if (C * w * trend.idm_for(len(trial)) / k[trial].values >= 1).all():
            chosen = trial
    return chosen


def tag(v, C):
    return f'{v}_{int(C / 1e3)}k' if C < 1e6 else f'{v}_1M'


rows, MY = [], {}
for v in SPEC:
    s84 = system(v, ALL)
    med84 = s84['pos'].loc[W0:W1].abs().median()
    k = NOTIONAL.reindex(ALL) * w84 * trend.idm_for(84) / med84
    for C in (100e3, 250e3, 1e6):
        g = greedy(k, C)
        cols, dropped = list(g), []
        while True:
            s = system(v, cols)
            nc = s['pos'].loc[W0:W1, cols].abs().median() * C / NOTIONAL[cols]
            if (nc >= 1).all():
                break
            worst = nc.idxmin(); dropped.append(worst); cols.remove(worst)
        sl = system(v, cols, lag=1.5)
        ncl = sl['pos'].loc[W0:W1, cols].abs().median() * C / NOTIONAL[cols]
        # contract-level check: median |whole contracts| held over the window (half-day-lag system)
        n_int = contracts(sl, cols, C)
        med_int = n_int.loc[W0:W1].abs().median()
        key = tag(v, C)
        pm = prev.loc[key, 'markets'].split(' | ')
        MY[(v, C)] = cols
        rows.append(dict(key=key, n_greedy=len(g), n_verified=len(cols), dropped=' | '.join(dropped),
                         min_med_contracts_same_close=round(nc.min(), 3), min_med_contracts_lag=round(ncl.min(), 3),
                         n_markets_median_whole_contracts_zero=int((med_int < 1).sum()),
                         markets_median_whole_contracts_zero=' | '.join(med_int.index[med_int < 1]),
                         same_set_as_prev=set(pm) == set(cols), same_order_as_prev=pm == cols,
                         prev_n=len(pm), only_mine=' | '.join(sorted(set(cols) - set(pm))), only_prev=' | '.join(sorted(set(pm) - set(cols))),
                         prev_min_med=prev.loc[key, 'min_median_contracts'], prev_min_med_lag=prev.loc[key, 'min_median_contracts_lag05'],
                         markets=' | '.join(cols)))
        print(rows[-1]['key'], len(g), '->', len(cols), dropped, 'min nc', round(nc.min(), 3), round(ncl.min(), 3),
              'int-med0', rows[-1]['n_markets_median_whole_contracts_zero'], 'same as prev', rows[-1]['same_set_as_prev'], round(time.time() - t0), flush=True)
R = pd.DataFrame(rows)
R.to_csv(HERE + 'subsets_check.csv', index=False)
pickle.dump(MY, open(HERE + 'subsets_mine.pkl', 'wb'))
# v5 vs audit (capital/tiers.csv greedy and simulation/subsets.csv verified)
ta = pd.read_csv(A + 'capital/tiers.csv').set_index('capital_usd')
sa = pd.read_csv(A + 'simulation/subsets.csv')
for C in (100e3, 250e3, 1e6):
    va = sa[(sa.capital_usd == C) & sa.rule.str.startswith('>=1')].markets.iloc[0].split(' | ')
    print('v5', int(C), 'audit verified n', len(va), 'mine', len(MY[('v5', C)]), 'identical', va == MY[('v5', C)])
# overlap and v3 within v5
for C in (100e3, 250e3, 1e6):
    print(int(C), 'v3 subset of v5:', set(MY[('v3', C)]) <= set(MY[('v5', C)]),
          ' v2 vs v5: only v2', sorted(set(MY[('v2', C)]) - set(MY[('v5', C)])), ' only v5', sorted(set(MY[('v5', C)]) - set(MY[('v2', C)])))
# median-position ratio v2 / v5 (84)
m2 = system('v2', ALL)['pos'].loc[W0:W1].abs().median(); m5 = system('v5', ALL)['pos'].loc[W0:W1].abs().median()
print('median of med|pos| ratio v2/v5:', round((m2 / m5).median(), 3))
