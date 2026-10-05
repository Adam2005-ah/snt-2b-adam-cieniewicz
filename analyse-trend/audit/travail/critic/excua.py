import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import numpy as np, pandas as pd, run_backtests as rb, listings, trend
full, groups, names = rb.load_futures('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
def run(drop):
    r = rets.drop(columns=drop); g = groups.drop(drop)
    risk = g.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
    costs = {c: rb.COSTS[x] for c, x in g.items()}
    roll = {c: 2 * rb.COSTS[x] * rb.ROLLS_PER_YEAR[x] for c, x in g.items()}
    return trend.run_vol_targeted(r, risk, costs, roll, forecast_fn=trend.forecast_regime)['net']
def sr(x): x = x.dropna(); return x.mean()/x.std()*np.sqrt(261)
out = {}
for lab, drop in [('all84', []), ('ex_CUA1', ['CUA1 Comdty']), ('ex_CUA1_SCO1', ['CUA1 Comdty', 'SCO1 Comdty'])]:
    n = run(drop)
    out[lab] = {k: round(sr(n.loc[d:]), 3) for k, d in [('1990','1990'),('2000','2000'),('2010','2010'),('2015','2015'),('2023-07','2023-07-10')]}
    out[lab]['mean2010'] = round(n.loc['2010':].mean()*261*100, 2)
print(pd.DataFrame(out).T.to_string())
pd.DataFrame(out).T.to_csv('excua_sharpes.csv')
