import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, run_backtests as rb, listings, trend, pickle
full, groups, names = rb.load_futures('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
v2 = trend.run_portfolio(rets, risk, costs, roll)
vol = rets.apply(trend.annual_vol)
out = {'v5_pos': v5['positions'], 'v5_fc': v5['forecasts'], 'v5_net': v5['net'], 'v5_live': v5['live'],
       'v2_pos': v2['positions'], 'v2_net': v2['net'], 'vol': vol, 'groups': groups, 'risk': risk, 'names': names, 'rets': rets}
pickle.dump(out, open('v5.pkl', 'wb'))
print(groups.value_counts()); print(v5['positions'].index[-1])
