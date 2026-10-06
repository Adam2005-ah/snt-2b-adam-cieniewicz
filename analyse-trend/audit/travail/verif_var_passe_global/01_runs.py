"""Adversarial verifier, independent re-run of L1 and L2 for the 5 variants.

Independence: does NOT import var_passe_global/engine.py nor read its caches. Positions come straight from the
project functions trend.run_portfolio / trend.run_vol_targeted (L1) or trend.run_portfolio on the 82-market
subset (L2, with the vol-targeting scale rebuilt here from the lagged real-cost P&L). Lag, real costs and
cash are re-implemented from the raw inputs (couts/contract_costs_per_market.csv, FRED DTB3).
Output: runs.pkl = {(variant, level): DataFrame(fut, trading, roll, gross_exp)} on the full futures calendar.
"""
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import listings  # noqa: E402
import run_backtests as rb  # noqa: E402
import trend  # noqa: E402

S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/verif_var_passe_global/'
full, groups, names = rb.load_futures(S + 'mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}

# ---- real IBKR cost per unit notional, standard contract, rebuilt from the raw columns
cc = pd.read_csv(S + 'audit/couts/contract_costs_per_market.csv', index_col=0)
cc.loc['ER4 Comdty', 'hs'] = 0.0025
notional = cc['mult'] * cc['price_last'] * cc['fx']
REAL = ((cc['fee'] + cc['hs'] * cc['mult']) * cc['fx'] / notional).reindex(rets.columns)
RPY = pd.Series({c: rb.ROLLS_PER_YEAR[g] for c, g in groups.items()})
REAL_ROLL = 2 * REAL * RPY
DROP = ['CUA1 Comdty', 'SCO1 Comdty']
C82 = [c for c in rets.columns if c not in DROP]

SPEC = {  # name: (forecast_fn, fixed sizing?, vol targeted?)
    'v1': (trend.forecast, True, False),
    'v2': (trend.forecast, False, False),
    'v3': (trend.forecast_regime, False, False),
    'v4': (trend.forecast, False, True),
    'v5': (trend.forecast_regime, False, True),
}


def l1(v):
    fn, fixed, vt = SPEC[v]
    if vt:
        r = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=fn)
    else:
        r = trend.run_portfolio(rets, risk, costs, roll, forecast_fn=fn,
                                sizing_vol=trend.fixed_vol(rets) if fixed else None)
    held = r['positions'].shift(1).fillna(0)
    df = pd.DataFrame({'fut': r['net'], 'gross': r['gross'], 'trading': r['trading_costs'], 'roll': r['roll_costs'],
                       'gross_exp': held.abs().sum(axis=1), 'pos_exp': r['positions'].abs().sum(axis=1)})
    return df


def lagged_pnl(pos, r, cst, rol, lag=1.5):
    """held_t = (1-f) P_{t-1} + f P_{t-2}; trades booked when executed (same split)."""
    f = lag - 1.0
    held = (1 - f) * pos.shift(1).fillna(0) + f * pos.shift(2).fillna(0)
    tr = pos.diff().abs()
    tr.iloc[0] = pos.iloc[0].abs()
    tr_exec = (1 - f) * tr + f * tr.shift(1).fillna(0)
    gross = (held * r.fillna(0)).sum(axis=1)
    trading = (tr_exec * cst).sum(axis=1)
    rolling = (held.abs() * rol / 252).sum(axis=1)
    return pd.DataFrame({'fut': gross - trading - rolling, 'gross': gross, 'trading': trading, 'roll': rolling,
                         'gross_exp': held.abs().sum(axis=1), 'pos_exp': pos.abs().sum(axis=1)})


def l2(v, cols=C82, lag=1.5, cst=None, rol=None):
    cst = REAL[cols] if cst is None else cst[cols]
    rol = REAL_ROLL[cols] if rol is None else rol[cols]
    fn, fixed, vt = SPEC[v]
    r = rets[cols]
    g = risk[cols]
    zero = {c: 0.0 for c in cols}
    sv = trend.fixed_vol(r) if fixed else None
    base = trend.run_portfolio(r, g, zero, zero, forecast_fn=fn, sizing_vol=sv)
    out = lagged_pnl(base['positions'], r, cst, rol, lag)
    if vt:
        realised = out['fut'].ewm(span=32, min_periods=32).std() * 16
        scale = (0.20 / realised).clip(0.5, 2.0).shift(1)
        res = trend.run_portfolio(r, g, zero, zero, forecast_fn=fn, scale=scale)
        out = lagged_pnl(res['positions'], r, cst, rol, lag)
        out['scale'] = scale
    return out


def job(arg):
    v, lvl = arg
    if lvl == 'L1':
        return arg, l1(v)
    if lvl == 'L2':
        return arg, l2(v)
    if lvl == 'S1':   # 82 markets, lag 1, bp costs
        return arg, l2(v, lag=1.0, cst=pd.Series(costs), rol=pd.Series(roll))
    if lvl == 'S2':   # 82 markets, lag 1.5, bp costs
        return arg, l2(v, lag=1.5, cst=pd.Series(costs), rol=pd.Series(roll))


if __name__ == '__main__':
    print('ER4 real bp:', REAL['ER4 Comdty'] * 1e4, ' check vs table real_bp (non-ER4) max diff:',
          (REAL * 1e4 - cc['real_bp'].reindex(rets.columns)).abs().drop('ER4 Comdty').max())
    tasks = [(v, l) for v in SPEC for l in ('L1', 'L2')] + [(v, l) for v in ('v2', 'v5') for l in ('S1', 'S2')]
    with Pool(4) as p:
        res = dict(p.map(job, tasks))
    pickle.dump({'runs': res, 'REAL': REAL, 'REAL_ROLL': REAL_ROLL}, open(OUT + 'runs.pkl', 'wb'))
    print('done')
