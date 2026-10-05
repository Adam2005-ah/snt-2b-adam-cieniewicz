import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, listings, trend
MKT='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt'
full, groups, names = rb.load_futures(MKT)
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
v2 = trend.run_portfolio(rets, risk, costs, roll)
out = pd.DataFrame({'v5': v5['net'], 'v2': v2['net'], 'v5_gross': v5['gross'], 'v2_gross': v2['gross'],
    'v5_cost': v5['trading_costs']+v5['roll_costs'], 'v5_lev': v5['positions'].abs().sum(axis=1), 'ES': full['ES1 Index']})
out.to_pickle('mine.pkl')
D=pd.read_pickle('../rendement/daily.pkl')
print('max abs diff v5', (out['v5']-D['v5_net']).abs().max(), 'v2', (out['v2']-D['v2_net']).abs().max())
print(out.index.min(), out.index.max())
o=out.loc['1990':]
for v in ['v5','v2']:
    s=o[v]; print(v, 'mean', s.mean()*252, 'vol', s.std()*np.sqrt(252), 'SR', s.mean()/s.std()*np.sqrt(252), 'first', s.first_valid_index())
