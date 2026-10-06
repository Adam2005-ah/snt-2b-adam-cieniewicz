"""Independent contract table check: today's notional and cost per side for the liquid contract, built from the
cost audit file's own columns (cost_side_usd / cost_side_small_usd), compared with what var_passe_retail used (sim)."""
import sys, numpy as np, pandas as pd
A = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
sp = pd.read_csv(A + 'capital/specs.csv', index_col=0)
cc = pd.read_csv(A + 'couts/contract_costs_per_market.csv', index_col=0)
strict = sp.index[sp.liquid_access.eq('OK')]
rows = []
for t in strict:
    lm = sp.loc[t, 'liquid_mult']; c = cc.loc[t]
    if abs(lm / c['mult'] - 1) < 0.05:
        kind = 'std'; cost = c['cost_side_usd']
        if t == 'ER4 Comdty':
            cost = c['fee_usd'] + 0.0025 * c['mult'] * c['fx']
    elif lm == c['mult_s']:
        kind = 'small'; cost = c['cost_side_small_usd']
    else:
        kind = 'micro_assumed'; cost = 1.0 + 1.5 * c['hs'] * lm * c['fx']
    rows.append(dict(ticker=t, kind=kind, liquid_mult=lm, notional_specs=sp.loc[t, 'liquid_notional_usd'],
                     notional_cc=(c['notional_usd'] if kind == 'std' else c['notional_small_usd'] if kind == 'small' else np.nan),
                     cost_side_usd=cost, cost_bp=cost / sp.loc[t, 'liquid_notional_usd'] * 1e4, group=sp.loc[t, 'group']))
T = pd.DataFrame(rows).set_index('ticker')
sys.path.insert(0, A + 'simulation')
import sim
sim.CC.loc['ER4 Comdty', 'hs'] = 0.0025
tab, N, HS = sim.contract_table('liquid_mult')
T['notional_sim'] = N.iloc[-1].reindex(T.index)
T['cost_sim'] = (tab['fee_spec'] + HS.iloc[-1]).reindex(T.index)
T['d_notional'] = T.notional_sim / T.notional_specs - 1
T['d_cost'] = T.cost_sim / T.cost_side_usd - 1
pd.set_option('display.width', 250)
print(T.round(4).to_string())
print('max |d_notional|', T.d_notional.abs().max(), ' max |d_cost|', T.d_cost.abs().max())
print(T.kind.value_counts())
T.to_csv('contract_table_check.csv')
