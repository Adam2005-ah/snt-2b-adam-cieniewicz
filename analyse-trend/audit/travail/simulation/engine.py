"""Variant 5 ('Les deux') engine: same maths as trend.run_vol_targeted(..., forecast_fn=trend.forecast_regime), but
vol and forecasts are computed once for the 84 markets (they are per-market, so valid for any subset) and the
function also returns the unbuffered target and the 'unit' (position for forecast 10), needed for Carver's
contract-level buffering."""
import sys, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import trend, run_backtests as rb, listings
S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/simulation/'
CACHE = OUT + 'base_cache.pkl'

def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    full, groups, names = rb.load_futures(S + 'mkt')
    rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
    vol = rets.apply(trend.annual_vol)
    fc = rets.apply(lambda s: trend.forecast_regime(s, vol[s.name]))
    d = dict(rets=rets, groups=groups, names=names, vol=vol, fc=fc)
    pickle.dump(d, open(CACHE, 'wb'))
    return d

D = load()
RETS, GROUPS, VOL, FC = D['rets'], D['groups'], D['vol'], D['fc']
RISK = GROUPS.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
COSTS = pd.Series({c: rb.COSTS[g] for c, g in GROUPS.items()})
ROLLC = pd.Series({c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in GROUPS.items()})
ROLLS_YR = pd.Series({c: rb.ROLLS_PER_YEAR[g] for c, g in GROUPS.items()})

def _portfolio(cols, freq, scale=None, risk_target=0.20):
    r, vol, fc, groups = RETS[cols], VOL[cols], FC[cols], RISK[cols]
    age = r.notna().cumsum()
    live = (age > trend.WARMUP_DAYS) & vol.notna() & fc.notna()
    class_count = live.T.groupby(groups).transform('sum').T
    n_classes = live.T.groupby(groups).any().T.sum(axis=1)
    weights = (live / class_count.where(class_count > 0)).div(n_classes, axis=0)
    idm = live.sum(axis=1).map(trend.idm_for)
    unit = (risk_target * weights.mul(idm, axis=0) / vol).where(live)
    if scale is not None:
        unit = unit.mul(scale.reindex(unit.index).fillna(1.0), axis=0)
    target = fc / 10 * unit
    mask = trend.rebalance_mask(r.index, freq)
    pos = pd.DataFrame({c: trend.buffered_positions(target[c], unit[c].fillna(0), mask) for c in cols})
    held = pos.shift(1).fillna(0)
    gross = (held * r.fillna(0)).sum(axis=1)
    trading = (pos.diff().abs().fillna(pos.abs()) * COSTS[cols]).sum(axis=1)
    rolling = (held.abs() * ROLLC[cols] / 252).sum(axis=1)
    return dict(net=gross - trading - rolling, gross=gross, pos=pos, target=target, unit=unit, mask=mask, live=live)

def system(cols, freq='D', span=32, bounds=(0.5, 2.0), risk_target=0.20):
    cols = list(cols)
    base = _portfolio(cols, freq, None, risk_target)
    realised = base['net'].ewm(span=span, min_periods=span).std() * 16
    scale = (risk_target / realised).clip(*bounds).shift(1)
    out = _portfolio(cols, freq, scale, risk_target)
    out['scale'] = scale
    return out

if __name__ == '__main__':
    import time; t0 = time.time()
    v = system(list(RETS.columns))
    print('time', round(time.time() - t0, 1))
    ref = pickle.load(open(S + 'audit/capital/v5.pkl', 'rb'))
    diff = (v['pos'] - ref['v5_pos']).abs().max().max()
    x = v['net'].loc['1990':'2026-07-10']
    print('max |pos diff| vs trend.run_vol_targeted:', diff, ' Sharpe 1990+:', round(x.mean() / x.std() * 16, 3),
          ' net diff:', (v['net'] - ref['v5_net']).abs().max())
