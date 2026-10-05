"""Fast re-implementation of variant 5 (trend.run_vol_targeted + forecast_regime) for audit work.

Per-market vol, forecasts and 'live' flags depend only on each market's own returns, so they are computed
once for the 84 markets and cached; portfolio construction (weights, IDM, buffer, vol targeting) is redone
for any subset. Adds an execution-lag option:
    lag = 1      : project convention (position decided on close t earns return of t+1)
    lag = 2      : one full extra day (trade at close t+1 on the signal of close t)
    lag = 1 + f  : fractional, P&L = (1-f) * pos.shift(1) * r + f * pos.shift(2) * r
                   (approximates executing a fraction f of the way through day t+1 in 'return time')
Costs: identical trades, so identical totals; trading costs are booked when the trade is executed.
The vol-targeting scale is measured on the lagged P&L (what a real trader would observe).
Read-only on project files.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import run_backtests as rb  # noqa: E402
import listings  # noqa: E402
import trend  # noqa: E402

S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
MKT = S + 'mkt'
OUT = S + 'audit/cv_donnees/'
CACHE = OUT + 'cache_fc.pkl'
AN = 261  # weekdays per year in this data set (critic convention); project trend.stats uses 16 = sqrt(256)

full, groups, names = rb.load_futures(MKT)
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
COSTS = pd.Series({c: rb.COSTS[g] for c, g in groups.items()})
ROLL = pd.Series({c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()})

if os.path.exists(CACHE):
    VOL, FC = pd.read_pickle(CACHE)
else:
    VOL = rets.apply(trend.annual_vol)
    FC = rets.apply(lambda s: trend.forecast_regime(s, VOL[s.name]))
    pd.to_pickle((VOL, FC), CACHE)
LIVE = (rets.notna().cumsum() > trend.WARMUP_DAYS) & VOL.notna() & FC.notna()


def _buffer(target, unit):
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


def _portfolio(cols, scale=None, lag=1.0, risk_target=0.20, cost_mult=1.0, r=None, start=None):
    r = rets[cols] if r is None else r[cols]
    live = LIVE[cols]
    vol_, fc_ = VOL[cols], FC[cols]
    if start is not None:
        r, live, vol_, fc_ = r.loc[start:], live.loc[start:], vol_.loc[start:], fc_.loc[start:]
    g = risk[cols]
    class_count = live.T.groupby(g).transform('sum').T
    n_classes = live.T.groupby(g).any().T.sum(axis=1)
    weights = (live / class_count.where(class_count > 0)).div(n_classes, axis=0)
    idm = live.sum(axis=1).map(trend.idm_for)
    unit = (risk_target * weights.mul(idm, axis=0) / vol_).where(live)
    if scale is not None:
        unit = unit.mul(scale.reindex(unit.index).fillna(1.0), axis=0)
    target = fc_ / 10 * unit
    pos = _buffer(target, unit)
    f = lag - 1.0
    k = int(np.floor(f))
    fr = f - k
    p1 = pos.shift(1 + k).fillna(0)
    p2 = pos.shift(2 + k).fillna(0)
    held = (1 - fr) * p1 + fr * p2
    rr = r.fillna(0)
    inst_gross = held * rr
    trades = pos.diff().abs().fillna(pos.abs())
    tr_exec = (1 - fr) * trades.shift(k).fillna(0) + fr * trades.shift(k + 1).fillna(0)
    inst_cost = tr_exec * COSTS[cols] * cost_mult + held.abs() * ROLL[cols] * cost_mult / 252
    gross = inst_gross.sum(axis=1)
    cost = inst_cost.sum(axis=1)
    return dict(net=gross - cost, gross=gross, cost=cost, positions=pos, held=held,
                inst_gross=inst_gross, inst_cost=inst_cost, unit=unit)


def v5(cols=None, lag=1.0, risk_target=0.20, cost_mult=1.0, span=32, bounds=(0.5, 2.0), start=None):
    """start: optional date to truncate the simulation window (signals still use full history)."""
    cols = list(rets.columns) if cols is None else list(cols)
    base = _portfolio(cols, None, lag, risk_target, cost_mult, start=start)
    realised = base['net'].ewm(span=span, min_periods=span).std() * 16
    scale = (risk_target / realised).clip(*bounds).shift(1)
    out = _portfolio(cols, scale, lag, risk_target, cost_mult, start=start)
    out['scale'] = scale
    return out


PERIODS = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01',
           '2023-07': '2023-07-10'}


def sr(x, an=AN):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(an)


def sharpes(net, an=AN, periods=PERIODS):
    return {k: sr(net.loc[d:], an) for k, d in periods.items()}


def tbill_daily(index):
    """DTB3 accrued on actual calendar days between observations of `index` (act/365)."""
    tb = rb.load_close(f'{MKT}/us-etf/alm0421_macro/DTB3.csv', 'value') / 100
    tb = tb.reindex(index.union(tb.index)).ffill().reindex(index)
    days = pd.Series(index, index=index).diff().dt.days.fillna(1)
    return tb.shift(1).fillna(tb) * days / 365


def perf(net, start, tb=None):
    d = net.loc[start:].dropna()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    curve = (1 + d).cumprod()
    res = dict(sharpe=sr(d), vol=d.std() * np.sqrt(AN), cagr_excess=curve.iloc[-1] ** (1 / yrs) - 1,
               maxdd_excess=(curve / curve.cummax() - 1).min(), arith_excess=d.mean() * AN)
    if tb is not None:
        t = (d + tb.reindex(d.index).fillna(0))
        ct = (1 + t).cumprod()
        res.update(cagr_total=ct.iloc[-1] ** (1 / yrs) - 1, maxdd_total=(ct / ct.cummax() - 1).min())
    return res
