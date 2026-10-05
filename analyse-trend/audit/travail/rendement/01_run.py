import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, run_backtests as rb, listings, trend
MKT='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt'
full, groups, names = rb.load_futures(MKT)
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
v2 = trend.run_portfolio(rets, risk, costs, roll)
out = pd.DataFrame({'v5_net': v5['net'], 'v5_gross': v5['gross'], 'v5_costs': v5['trading_costs']+v5['roll_costs'],
                    'v5_lev': v5['positions'].abs().sum(axis=1),
                    'v2_net': v2['net'], 'v2_gross': v2['gross'], 'v2_costs': v2['trading_costs']+v2['roll_costs'],
                    'v2_lev': v2['positions'].abs().sum(axis=1),
                    'n_live': v5['live'].sum(axis=1)})
# also ES1 (S&P 500 futures excess return) for equity benchmark
print([c for c in full.columns if 'ES' in c or 'SP' in c][:10])
for c in ['ES1 Index','SP1 Index']:
    if c in full.columns: out[c] = full[c]
out.to_pickle('daily.pkl')
o = out.loc['1990':]
for v in ['v5','v2']:
    s=o[v+'_net']; print(v, s.mean()*252, s.std()*16, s.mean()/s.std()*16, o[v+'_costs'].mean()*252, o[v+'_lev'].mean())
