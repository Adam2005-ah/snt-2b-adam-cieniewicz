"""Check veng.system(v, all 84, lag=1) against trend.run_portfolio / run_vol_targeted for the 5 variants, and
store the full-84 published reference (fractional, bp-cost model) per variant."""
import sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/var_passe_retail')
import veng as V
import trend, run_backtests as rb  # noqa

rets, risk = V.RETS, V.RISK
costs = V.COSTS.to_dict(); roll = V.ROLLC.to_dict()
t0 = time.time()
ref = {
    'v1': trend.run_portfolio(rets, risk, costs, roll, sizing_vol=trend.fixed_vol(rets)),
    'v2': trend.run_portfolio(rets, risk, costs, roll),
    'v3': trend.run_portfolio(rets, risk, costs, roll, forecast_fn=trend.forecast_regime),
    'v4': trend.run_vol_targeted(rets, risk, costs, roll),
    'v5': trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime),
}
print('trend runs', round(time.time() - t0))
ALL = list(rets.columns)
out = {}
for v in V.VARIANTS:
    s = V.system(v, ALL)
    dp = (s['pos'] - ref[v]['positions']).abs().max().max()
    dn = (s['net'] - ref[v]['net']).abs().max()
    x = ref[v]['net'].loc['1990':'2026-07-10']
    tb = rb.load_close('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/us-etf/alm0421_macro/DTB3.csv', 'value') / 100
    print(v, 'max|pos diff|', dp, 'max|net diff|', dn, 'Sharpe90', round(x.mean() / x.std() * 16, 3),
          'vol', round(x.std() * 16, 3), 'gross exp', round(ref[v]['positions'].loc['1990':].abs().sum(axis=1).mean(), 2))
    out[v] = dict(net=ref[v]['net'], pos=ref[v]['positions'])
pickle.dump(out, open(V.OUT + 'ref84.pkl', 'wb'))
