import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, run_backtests as rb, listings, trend
OUT='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/operations/'
full, groups, names = rb.load_futures('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
v5['positions'].to_pickle(OUT+'v5_positions.pkl')
v5['forecasts'].to_pickle(OUT+'v5_forecasts.pkl')
pd.DataFrame({'net':v5['net'],'gross':v5['gross'],'trading':v5['trading_costs'],'roll':v5['roll_costs']}).to_pickle(OUT+'v5_net.pkl')
v5['live'].to_pickle(OUT+'v5_live.pkl')
groups.to_pickle(OUT+'groups.pkl'); names.to_pickle(OUT+'names.pkl')
rets.to_pickle(OUT+'rets.pkl')
