"""Check 1: engine at lag 1 + bp costs reproduces trend.run_* exactly for the 5 variants (84 markets).
Check 2: engine on the 82-market tradable universe, lag 1.5, bp costs reproduces cv_donnees restatement_grid for v5.
Saves the published runs (L0/L1) to runs_published.pkl."""
import pickle
import numpy as np
import pandas as pd
import engine as E
import trend

rets, risk = E.rets, E.risk
costs, roll = E.BP_COSTS.to_dict(), E.BP_ROLL.to_dict()
pub = {
    '1. Montant fixe': trend.run_portfolio(rets, risk, costs, roll, sizing_vol=trend.fixed_vol(rets)),
    '2. Système actuel': trend.run_portfolio(rets, risk, costs, roll),
    '3. + stratégie 13': trend.run_portfolio(rets, risk, costs, roll, forecast_fn=trend.forecast_regime),
    '4. + pilotage du risque': trend.run_vol_targeted(rets, risk, costs, roll),
    '5. Les deux': trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime),
}
keep = ['net', 'gross', 'trading_costs', 'roll_costs', 'positions']
pickle.dump({k: {x: v[x] for x in keep} for k, v in pub.items()}, open(E.OUT + 'runs_published.pkl', 'wb'))

for name, p in pub.items():
    mine = E.variant(name)
    d = {x: float((mine[x] - p[x]).abs().max()) for x in ['net', 'gross', 'trading_costs', 'roll_costs']}
    d['positions'] = float((mine['positions'] - p['positions']).abs().max().max())
    print(name, {k: f'{v:.2e}' for k, v in d.items()})

cols = [c for c in rets.columns if c not in ('CUA1 Comdty', 'SCO1 Comdty')]
r = E.variant('5. Les deux', cols, lag=1.5)
for start, ref in [('1990-01-01', 1.074), ('2000-01-01', 0.724), ('2010-01-01', 0.375)]:
    x = r['net'].loc[start:]
    print('v5 tradable, lag 1.5, bp costs, Sharpe16 from', start[:4], round(x.mean() / x.std() * 16, 4), 'grid', ref)
