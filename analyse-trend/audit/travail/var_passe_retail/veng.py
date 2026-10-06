"""Generic engine for the 5 variants (same maths as trend.run_portfolio / trend.run_vol_targeted), with
per-market vol, forecasts (plain and regime) and the frozen 'fixed' vol computed once for the 84 markets
(they depend only on each market's own returns, so are valid for any subset); weights, IDM, buffer and
portfolio vol-targeting are recomputed on the subset.

Execution lag option (as audit/cv_donnees/core.py):
    lag = 1.0 : project convention (decided at close t, earns return of t+1)
    lag = 1.5 : half-day lag, P&L = 0.5 * pos.shift(1) * r + 0.5 * pos.shift(2) * r ; trading costs booked half on
                t, half on t+1; for v4/v5 the portfolio vol-targeting scale is measured on the lagged P&L.
Read-only on project files.
"""
import os, sys, pickle
import numpy as np, pandas as pd
SIMDIR = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation'
sys.path.insert(0, SIMDIR)
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import engine as E   # noqa: E402  (RETS, GROUPS, VOL, FC regime, COSTS, ROLLC)
import trend         # noqa: E402

OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/var_passe_retail/'
CACHE = OUT + 'cache_fc_def_fixvol.pkl'
RETS, VOL, RISK = E.RETS, E.VOL, E.RISK
COSTS, ROLLC = E.COSTS, E.ROLLC
FC_REG = E.FC
if os.path.exists(CACHE):
    FC_DEF, FIXV = pd.read_pickle(CACHE)
else:
    FC_DEF = RETS.apply(lambda s: trend.forecast(s, VOL[s.name]))
    FIXV = trend.fixed_vol(RETS)
    pd.to_pickle((FC_DEF, FIXV), CACHE)

VARIANTS = {
    'v1': dict(name='1. Montant fixe', fc='def', fixed=True, vt=False),
    'v2': dict(name='2. Systeme actuel', fc='def', fixed=False, vt=False),
    'v3': dict(name='3. + strategie 13', fc='reg', fixed=False, vt=False),
    'v4': dict(name='4. + pilotage du risque', fc='def', fixed=False, vt=True),
    'v5': dict(name='5. Les deux', fc='reg', fixed=False, vt=True),
}


def _buffer(target, unit):
    """trend.buffered_positions, vectorised across markets, daily rebalancing."""
    T = target.values
    W = (trend.BUFFER * unit.fillna(0)).values
    out = np.zeros_like(T)
    cur = np.zeros(T.shape[1])
    for i in range(T.shape[0]):
        t = T[i]
        nan = np.isnan(t)
        cur = np.where(nan, 0.0, np.minimum(np.maximum(cur, np.where(nan, 0, t - W[i])), np.where(nan, 0, t + W[i])))
        out[i] = cur
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def _portfolio(cols, fc_all, size_vol_all, scale=None, lag=1.0, risk_target=0.20):
    r, vol, fc, groups = RETS[cols], VOL[cols], fc_all[cols], RISK[cols]
    age = r.notna().cumsum()
    live = (age > trend.WARMUP_DAYS) & vol.notna() & fc.notna()
    class_count = live.T.groupby(groups).transform('sum').T
    n_classes = live.T.groupby(groups).any().T.sum(axis=1)
    weights = (live / class_count.where(class_count > 0)).div(n_classes, axis=0)
    idm = live.sum(axis=1).map(trend.idm_for)
    svol = vol if size_vol_all is None else size_vol_all[cols]
    unit = (risk_target * weights.mul(idm, axis=0) / svol).where(live)
    if scale is not None:
        unit = unit.mul(scale.reindex(unit.index).fillna(1.0), axis=0)
    target = fc / 10 * unit
    pos = _buffer(target, unit)
    fr = lag - 1.0
    assert 0 <= fr <= 1
    held = ((1 - fr) * pos.shift(1) + fr * pos.shift(2)).fillna(0)
    trades = pos.diff().abs().fillna(pos.abs())
    tr_exec = (1 - fr) * trades + fr * trades.shift(1).fillna(0)
    gross = (held * r.fillna(0)).sum(axis=1)
    trading = (tr_exec * COSTS[cols]).sum(axis=1)
    rolling = (held.abs() * ROLLC[cols] / 252).sum(axis=1)
    mask = pd.Series(True, index=r.index)
    return dict(net=gross - trading - rolling, gross=gross, pos=pos, target=target, unit=unit, mask=mask, live=live)


def system(v, cols, lag=1.0, span=32, bounds=(0.5, 2.0), risk_target=0.20):
    spec = VARIANTS[v]
    cols = list(cols)
    fc = FC_REG if spec['fc'] == 'reg' else FC_DEF
    sv = FIXV if spec['fixed'] else None
    if not spec['vt']:
        out = _portfolio(cols, fc, sv, None, lag, risk_target)
        out['scale'] = None
        return out
    base = _portfolio(cols, fc, sv, None, lag, risk_target)
    realised = base['net'].ewm(span=span, min_periods=span).std() * 16
    scale = (risk_target / realised).clip(*bounds).shift(1)
    out = _portfolio(cols, fc, sv, scale, lag, risk_target)
    out['scale'] = scale
    return out
