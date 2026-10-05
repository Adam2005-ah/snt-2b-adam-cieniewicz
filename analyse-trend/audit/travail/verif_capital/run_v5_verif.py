import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, listings, trend, pickle
S='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
full, groups, names = rb.load_futures(S+'mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
v2 = trend.run_portfolio(rets, risk, costs, roll)
n=v5['net']; print('sharpe v5', n.mean()/n.std()*16, 'v2', v2['net'].mean()/v2['net'].std()*16)
pickle.dump({'v5':v5['positions'],'v2':v2['positions'],'net5':v5['net'],'rets':rets,'groups':groups,'risk':risk}, open('mine.pkl','wb'))
old=pickle.load(open(S+'audit/capital/v5.pkl','rb'))
print('max diff pos vs capital pkl', (old['v5_pos']-v5['positions']).abs().max().max())
