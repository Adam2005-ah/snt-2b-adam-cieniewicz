"""Sensitivity: wider contract buffer for small accounts, B = max(10 % of the forecast-10 position, b contracts)."""
import sys, pickle
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation')
import pandas as pd, sim
E = sim.E
subs = pickle.load(open(sim.OUT + 'subsets.pkl', 'rb'))
STRICT = list(sim.SP.index[sim.SP['liquid_access'].eq('OK')])
s66 = E.system(STRICT)
rows = []
for C in [100e3, 250e3, 1e6]:
    cols = subs[('sub', C, 1)]; s = E.system(cols)
    for b in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for lab, cc, ss in [('b_strict_subset_ge1c', cols, s), ('strict66_liquid', STRICT, s66)]:
            r, _, _ = sim.evaluate(lab, cc, C, ss, 'liquid_today', extra=dict(notional='today', min_buffer_contracts=b), min_buffer=b)
            rows += r
res = pd.DataFrame(rows)
res.to_csv(sim.OUT + 'buffer_sensitivity.csv', index=False)
pd.set_option('display.width', 250)
print(res[['universe', 'capital_usd', 'min_buffer_contracts', 'period', 'sharpe_net_real', 'sharpe_gross', 'cagr_excess_net_real', 'vol', 'te_vs_own_fractional',
           'trade_orders_per_yr', 'trade_contract_sides_per_yr', 'commission_flat_pct_cap', 'fee_plus_halfspread_pct_cap']].to_string())
