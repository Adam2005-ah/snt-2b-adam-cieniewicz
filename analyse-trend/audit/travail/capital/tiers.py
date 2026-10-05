import pandas as pd, numpy as np, sys
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import trend
sp = pd.read_csv('specs.csv', index_col=0)
TIERS = [25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6, 20e6]
risk = sp['group'].replace({'Bonds': 'Taux', 'STIR': 'Taux'})
# 'accessible' (optimistic): smallest contract, excluding NO/NO?/ILLIQ and very-illiquid smallest contracts; KID? kept
acc = ~sp['access'].isin(['NO', 'NO?', 'ILLIQ']) & (sp['small_liquidity'] != 'VL')
# 'strict' (realistic): smallest contract that is at least moderately liquid AND on an exchange with confirmed EU-retail KIDs
strict = sp['liquid_access'].eq('OK')
short = {'Equities': 'Eq', 'Taux': 'Rates', 'FX': 'FX', 'Energy': 'Ener', 'Metals': 'Met', 'Agriculture': 'Ag'}
def classes(mask):
    c = risk[mask].value_counts()
    return ', '.join(f"{short[k]}:{v}" for k, v in c.items())
# k: capital needed for 1 contract at the median position if the market had (weight x IDM) = 1
k_small = sp['small_notional_usd'] * sp['weight'] * 2.5 / sp['v5_med_abs_pos']
k_liq = sp['liquid_notional_usd'] * sp['weight'] * 2.5 / sp['v5_med_abs_pos']
def greedy(universe, k, C, m):
    """Round-robin over the 6 risk classes (cheapest first within each class); a market is added only if every
    market in the set still holds >= m contracts at its median position once weights (equal per class, equal
    within class) and IDM (Carver table) are recomputed for the enlarged set."""
    u = pd.DataFrame({'k': k[universe], 'cls': risk[universe]})
    order_in = {c: list(g.sort_values('k').index) for c, g in u.groupby('cls')}
    cls_order = sorted(order_in, key=lambda c: u.loc[order_in[c][0], 'k'])
    seq = []
    while any(order_in.values()):
        for c in cls_order:
            if order_in[c]: seq.append(order_in[c].pop(0))
    chosen = []
    for t in seq:
        trial = chosen + [t]
        cl = risk[trial]; ncl = cl.nunique(); cnt = cl.value_counts()
        w = pd.Series({x: 1 / ncl / cnt[cl[x]] for x in trial})
        contracts = C * w * trend.idm_for(len(trial)) / k[trial]
        if (contracts >= m).all():
            chosen = trial
    return chosen
rows = []
for C in TIERS:
    r = {'capital_usd': C}
    for lab, mask, col in [('all84', pd.Series(True, index=sp.index), 'small'), ('accessible', acc, 'small'), ('strict', strict, 'liquid')]:
        ok1 = mask & (sp[f'mincap_1c_{col}'] <= C); ok4 = mask & (sp[f'mincap_4c_{col}'] <= C)
        r[f'{lab}_n_ge1c'] = int(ok1.sum()); r[f'{lab}_n_ge4c'] = int(ok4.sum())
        if lab != 'all84':
            r[f'{lab}_classes_ge1c'] = classes(ok1); r[f'{lab}_classes_ge4c'] = classes(ok4)
            r[f'{lab}_markets_ge1c'] = ' | '.join(sp.index[ok1])
    r['all84_n_ge1c_std_contract'] = int((sp['mincap_1c_std'] <= C).sum())
    r['all84_n_ge4c_carver_formula'] = int((sp['carver_mincap_4c_small'] <= C).sum())
    for lab, mask, k in [('acc', acc, k_small), ('strict', strict, k_liq)]:
        for m in (1, 4):
            ch = greedy(mask, k, C, m)
            r[f'{lab}_subset_ge{m}c_n'] = len(ch); r[f'{lab}_subset_ge{m}c_classes'] = classes(sp.index.isin(ch))
            r[f'{lab}_subset_ge{m}c_markets'] = ' | '.join(ch)
    rows.append(r)
tiers = pd.DataFrame(rows)
tiers.to_csv('tiers.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 400)
print(tiers[['capital_usd','all84_n_ge1c','all84_n_ge4c','accessible_n_ge1c','accessible_n_ge4c','strict_n_ge1c','strict_n_ge4c','all84_n_ge1c_std_contract','all84_n_ge4c_carver_formula','acc_subset_ge1c_n','acc_subset_ge4c_n','strict_subset_ge1c_n','strict_subset_ge4c_n']].to_string())
for _, r in tiers.iterrows():
    print(int(r.capital_usd), '| strict >=1c:', r.strict_classes_ge1c, '| strict >=4c:', r.strict_classes_ge4c)
    print('   strict markets>=1c:', r.strict_markets_ge1c)
    print('   STRICT subset4:', r.strict_subset_ge4c_n, r.strict_subset_ge4c_classes, '::', r.strict_subset_ge4c_markets)
    print('   STRICT subset1:', r.strict_subset_ge1c_n, r.strict_subset_ge1c_classes)
