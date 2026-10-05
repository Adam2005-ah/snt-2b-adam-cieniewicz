import sys, pickle, pandas as pd, numpy as np
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import trend, run_backtests as rb
sp = pd.read_csv('specs.csv', index_col=0); tiers = pd.read_csv('tiers.csv').set_index('capital_usd')
d = pickle.load(open('v5.pkl', 'rb')); rets, groups = d['rets'], d['groups']
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}; roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
rows = []
for C in [25e3, 50e3, 100e3, 250e3, 500e3, 1e6]:
    sub = tiers.loc[C, 'strict_subset_ge4c_markets'].split(' | ')
    res = trend.run_vol_targeted(rets[sub], risk[sub], {k: costs[k] for k in sub}, {k: roll[k] for k in sub}, forecast_fn=trend.forecast_regime)
    a = res['positions'].loc['2023-07-10':'2026-07-10'].abs()
    for t in sub:
        med = a[t].median()
        rows.append({'capital_usd': C, 'ticker': t, 'name': sp.loc[t, 'name'], 'class': risk[t], 'contract': sp.loc[t, 'liquid_contract'],
                     'contract_notional_usd': round(sp.loc[t, 'liquid_notional_usd']), 'median_abs_pos_frac': round(med, 3),
                     'median_contracts': round(med * C / sp.loc[t, 'liquid_notional_usd'], 1),
                     'p90_contracts': round(a[t].quantile(.9) * C / sp.loc[t, 'liquid_notional_usd'], 1)})
df = pd.DataFrame(rows); df.to_csv('retail_portfolios.csv', index=False)
pd.set_option('display.width', 250); print(df.to_string())
