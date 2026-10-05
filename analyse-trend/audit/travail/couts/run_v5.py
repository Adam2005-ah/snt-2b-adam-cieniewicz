import sys, pickle
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, run_backtests as rb, listings, trend
OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts/'
full, groups, names = rb.load_futures('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
v2 = trend.run_portfolio(rets, risk, costs, roll)
keep = lambda d: {k: d[k] for k in ['net', 'gross', 'positions', 'trading_costs', 'roll_costs']}
pickle.dump({'v5': keep(v5), 'v2': keep(v2), 'groups': groups, 'rets': rets}, open(OUT + 'v5_cache.pkl', 'wb'))
print(rb.START, v5['net'].loc[rb.START:].mean()*252, (v5['trading_costs']+v5['roll_costs']).loc[rb.START:].mean()*252)
