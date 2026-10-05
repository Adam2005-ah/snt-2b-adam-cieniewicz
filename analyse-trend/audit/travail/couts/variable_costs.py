import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts')
import run_backtests as rb, specs
OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts/'
D = pickle.load(open(OUT + 'v5_cache.pkl', 'rb'))
groups = D['groups']
P = pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
T = specs.table()
assert set(T.index) == set(groups.index), set(groups.index) ^ set(T.index)
fx = specs.fx_frame(P)
last = P.ffill().iloc[-1]
fx_last = fx.iloc[-1]
T['group'] = groups
T['price_last'] = last
T['fx'] = [fx_last[c] for c in T.ccy]
T['notional_usd'] = T.price_last * T.mult * T.fx
T['fee_usd'] = T.fee * T.fx
T['spread_usd'] = T.hs * T.mult * T.fx
T['cost_side_usd'] = T.fee_usd + T.spread_usd
T['real_bp'] = T.cost_side_usd / T.notional_usd * 1e4
T['bt_bp'] = [rb.COSTS[g] * 1e4 for g in T.group]
T['notional_small_usd'] = T.price_last * T.mult_s * T.fx
T['cost_side_small_usd'] = (T.fee_s + T.hs_s * T.mult_s) * T.fx
T['real_bp_small'] = T.cost_side_small_usd / T.notional_small_usd * 1e4
T['rolls_per_year'] = [rb.ROLLS_PER_YEAR[g] for g in T.group]

# --- 1. no-rounding cost of v5 with realistic per-market bp ---
res = {}
for v in ['v5', 'v2']:
    pos = D[v]['positions']
    dpos = pos.diff().abs().fillna(pos.abs())
    held = pos.shift(1).fillna(0).abs()
    for label, bp in [('backtest', T.bt_bp), ('realistic', T.real_bp)]:
        bpv = bp.reindex(pos.columns) / 1e4
        trade = (dpos * bpv).sum(axis=1)
        rollc = (held * bpv * 2 * T.rolls_per_year.reindex(pos.columns) / 252).sum(axis=1)
        for per, start in [('1990-2026', '1990-01-01'), ('2010-2026', '2010-01-01'), ('2023-07_2026-07', '2023-07-10')]:
            res[(v, label, per)] = {'trading_%': trade.loc[start:].mean() * 252 * 100,
                                    'roll_%': rollc.loc[start:].mean() * 252 * 100,
                                    'total_%': (trade + rollc).loc[start:].mean() * 252 * 100}
R1 = pd.DataFrame(res).T.round(2)
print(R1)
# check backtest internal cost equals recomputed
print('backtest-reported v5 cost 1990+: %.2f%%' % (((D['v5']['trading_costs'] + D['v5']['roll_costs']).loc['1990':].mean()) * 252 * 100))

# turnover per class
pos = D['v5']['positions']
to = pos.diff().abs().loc['1990':].mean() * 252
gross = pos.abs().loc['1990':].mean()
T['v5_turnover_x_per_yr'] = to
T['v5_mean_abs_pos'] = gross
T['v5_cost_contrib_real_%'] = (to * T.real_bp / 1e4 + gross * T.real_bp / 1e4 * 2 * T.rolls_per_year) * 100
T['v5_cost_contrib_bt_%'] = (to * T.bt_bp / 1e4 + gross * T.bt_bp / 1e4 * 2 * T.rolls_per_year) * 100
T.to_csv(OUT + 'contract_costs_per_market.csv')
print(T[['group', 'notional_usd', 'cost_side_usd', 'real_bp', 'bt_bp', 'notional_small_usd', 'real_bp_small', 'v5_turnover_x_per_yr', 'v5_mean_abs_pos']].round(2).to_string())
print(T.groupby('group')[['v5_cost_contrib_real_%', 'v5_cost_contrib_bt_%']].sum().round(3))
R1.to_csv(OUT + 'variable_costs_norounding.csv')
