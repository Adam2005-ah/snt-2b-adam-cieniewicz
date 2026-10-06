"""Generic re-implementation of the 5 project variants (trend.run_portfolio / run_vol_targeted) for the
historical net-of-fees restatement (var_passe_global).

Per-market vol, forecasts (standard and strategy-13 'regime'), frozen 'fixed' vol and live flags depend only on
each market's own returns -> computed once for the 84 markets and cached. Portfolio construction (class weights,
IDM, buffer, vol targeting) is redone for any column subset, exactly as trend.run_portfolio does.

Options on top of the project code:
  lag      : 1 = project convention (decided at close t, earns the return of t+1)
             1 + f (0<f<1) = fractional lag as in cv_donnees/core.py:
                 held = (1-f) * pos.shift(1) + f * pos.shift(2)   ('manual_0.5d' = 1.5)
             trades are booked when executed (same split), so total trade volume is unchanged.
  costs    : per-market cost per unit of notional traded (fraction), and annual roll cost per unit held.
The vol-targeting scale (variants 4, 5) is measured on the base portfolio's own net P&L under the same
lag / costs (what a real trader would observe), then shifted one day, exactly as trend.run_vol_targeted.
Read-only on project files.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import listings  # noqa: E402
import run_backtests as rb  # noqa: E402
import trend  # noqa: E402

S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
MKT = S + 'mkt'
OUT = S + 'audit/var_passe_global/'
CACHE = OUT + 'cache_signals.pkl'

full, groups, names = rb.load_futures(MKT)
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
BP_COSTS = pd.Series({c: rb.COSTS[g] for c, g in groups.items()})
BP_ROLL = pd.Series({c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()})

if os.path.exists(CACHE):
    VOL, FC_STD, FC_REG, FIXED = pd.read_pickle(CACHE)
else:
    VOL = rets.apply(trend.annual_vol)
    FC_STD = rets.apply(lambda s: trend.forecast(s, VOL[s.name]))
    FC_REG = rets.apply(lambda s: trend.forecast_regime(s, VOL[s.name]))
    FIXED = trend.fixed_vol(rets)
    pd.to_pickle((VOL, FC_STD, FC_REG, FIXED), CACHE)

VARIANTS = {  # name: (forecast, sizing, vol_targeted)
    '1. Montant fixe': ('std', 'fixed', False),
    '2. Système actuel': ('std', 'vol', False),
    '3. + stratégie 13': ('reg', 'vol', False),
    '4. + pilotage du risque': ('std', 'vol', True),
    '5. Les deux': ('reg', 'vol', True),
}


def _buffer(target, unit):
    """Vectorised over markets, identical to trend.buffered_positions with a daily mask."""
    T = target.values
    W = (trend.BUFFER * unit.fillna(0)).values
    out = np.zeros_like(T)
    cur = np.zeros(T.shape[1])
    for i in range(T.shape[0]):
        t = T[i]
        nan = np.isnan(t)
        lo = np.where(nan, 0.0, t - W[i])
        hi = np.where(nan, 0.0, t + W[i])
        cur = np.where(nan, 0.0, np.minimum(np.maximum(cur, lo), hi))
        out[i] = cur
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def portfolio(cols, fc_kind='std', sizing='vol', scale=None, lag=1.0, costs=None, roll=None, risk_target=0.20):
    cols = list(cols)
    costs = BP_COSTS if costs is None else costs
    roll = BP_ROLL if roll is None else roll
    r = rets[cols]
    vol = VOL[cols]
    fc = (FC_STD if fc_kind == 'std' else FC_REG)[cols]
    live = (r.notna().cumsum() > trend.WARMUP_DAYS) & vol.notna() & fc.notna()
    g = risk[cols]
    class_count = live.T.groupby(g).transform('sum').T
    n_classes = live.T.groupby(g).any().T.sum(axis=1)
    weights = (live / class_count.where(class_count > 0)).div(n_classes, axis=0)
    idm = live.sum(axis=1).map(trend.idm_for)
    size_vol = vol if sizing == 'vol' else FIXED[cols]
    unit = (risk_target * weights.mul(idm, axis=0) / size_vol).where(live)
    if scale is not None:
        unit = unit.mul(scale.reindex(unit.index).fillna(1.0), axis=0)
    target = fc / 10 * unit
    pos = _buffer(target, unit)

    f = lag - 1.0
    k = int(np.floor(f))
    fr = f - k
    held = (1 - fr) * pos.shift(1 + k).fillna(0) + fr * pos.shift(2 + k).fillna(0)
    trades = pos.diff().abs().fillna(pos.abs())
    tr_exec = (1 - fr) * trades.shift(k).fillna(0) + fr * trades.shift(k + 1).fillna(0)
    gross = (held * r.fillna(0)).sum(axis=1)
    trading = (tr_exec * costs[cols]).sum(axis=1)
    rolling = (held.abs() * roll[cols] / 252).sum(axis=1)
    return dict(net=gross - trading - rolling, gross=gross, trading_costs=trading, roll_costs=rolling,
                positions=pos, held=held, live=live)


def variant(name, cols=None, lag=1.0, costs=None, roll=None, span=32, bounds=(0.5, 2.0), risk_target=0.20):
    cols = list(rets.columns) if cols is None else list(cols)
    fc_kind, sizing, vt = VARIANTS[name]
    base = portfolio(cols, fc_kind, sizing, None, lag, costs, roll, risk_target)
    if not vt:
        return base
    realised = base['net'].ewm(span=span, min_periods=span).std() * 16
    scale = (risk_target / realised).clip(*bounds).shift(1)
    out = portfolio(cols, fc_kind, sizing, scale, lag, costs, roll, risk_target)
    out['scale'] = scale
    return out
