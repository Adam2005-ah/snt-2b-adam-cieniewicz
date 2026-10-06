"""Independent re-implementation (verifier) of the retail historical simulation of the 5 variants.
Uses only the project code (trend.py: annual_vol, forecast, fixed_vol, idm_for, buffered_positions, run_portfolio,
run_vol_targeted as reference), the raw data (recipe of the task), specs.csv and the cost-audit csv. It does NOT import
var_passe_retail/veng.py, simulation/sim.py or simulation/engine.py.
Read-only on project files; writes only into this folder."""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import run_backtests as rb, listings, trend  # noqa: E402

SCR = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
A = SCR + 'audit/'
HERE = A + 'verif_var_passe_retail/'
CACHE = HERE + 'cache_indep.pkl'

if os.path.exists(CACHE):
    D = pickle.load(open(CACHE, 'rb'))
else:
    full, groups, names = rb.load_futures(SCR + 'mkt')
    rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
    vol = rets.apply(trend.annual_vol)
    D = dict(rets=rets, groups=groups,
             vol=vol,
             fc_def=rets.apply(lambda s: trend.forecast(s, vol[s.name])),
             fc_reg=rets.apply(lambda s: trend.forecast_regime(s, vol[s.name])),
             fixv=trend.fixed_vol(rets))
    pickle.dump(D, open(CACHE, 'wb'))

RETS, GROUPS, VOL = D['rets'], D['groups'], D['vol']
RISK = GROUPS.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
BPC = pd.Series({c: rb.COSTS[g] for c, g in GROUPS.items()})
ROLLBP = pd.Series({c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in GROUPS.items()})
ROLLS_YR = pd.Series({c: rb.ROLLS_PER_YEAR[g] for c, g in GROUPS.items()})
SPEC = {'v1': ('def', True, False), 'v2': ('def', False, False), 'v3': ('reg', False, False),
        'v4': ('def', False, True), 'v5': ('reg', False, True)}


def frac_portfolio(cols, fc, svol, scale, lag):
    """Fractional positions (fraction of capital) for a subset; weights, IDM recomputed on the subset."""
    r, v, f = RETS[cols], VOL[cols], fc[cols]
    live = (r.notna().cumsum() > trend.WARMUP_DAYS) & v.notna() & f.notna()
    cls = RISK[cols]
    n_in = pd.DataFrame({c: live[cls.index[cls == cls[c]]].sum(axis=1) for c in cols})
    n_cls = pd.concat([live[cls.index[cls == g]].any(axis=1) for g in cls.unique()], axis=1).sum(axis=1)
    w = live.astype(float) / n_in.where(n_in > 0) / n_cls.replace(0, np.nan).values[:, None]
    idm = live.sum(axis=1).map(trend.idm_for)
    sv = v if svol is None else svol[cols]
    unit = (0.20 * w.mul(idm, axis=0) / sv).where(live)
    if scale is not None:
        unit = unit.mul(scale.reindex(unit.index).fillna(1.0), axis=0)
    target = f / 10 * unit
    mask = pd.Series(True, index=r.index)
    pos = pd.DataFrame({c: trend.buffered_positions(target[c], unit[c].fillna(0), mask) for c in cols})
    a = lag - 1.0
    held = ((1 - a) * pos.shift(1) + a * pos.shift(2)).fillna(0)
    tr = pos.diff().abs().fillna(pos.abs())
    tr = (1 - a) * tr + a * tr.shift(1).fillna(0)
    net = (held * r.fillna(0)).sum(axis=1) - (tr * BPC[cols]).sum(axis=1) - (held.abs() * ROLLBP[cols] / 252).sum(axis=1)
    return dict(pos=pos, target=target, unit=unit, net=net)


def system(v, cols, lag=1.0):
    fcn, fixed, vt = SPEC[v]
    fc = D['fc_reg'] if fcn == 'reg' else D['fc_def']
    sv = D['fixv'] if fixed else None
    base = frac_portfolio(list(cols), fc, sv, None, lag)
    if not vt:
        return base
    realised = base['net'].ewm(span=32, min_periods=32).std() * 16
    scale = (0.20 / realised).clip(0.5, 2.0).shift(1)
    out = frac_portfolio(list(cols), fc, sv, scale, lag)
    out['scale'] = scale
    return out


# ---------------- contract table (today's liquid contract), independent of sim.py ----------------
SP = pd.read_csv(A + 'capital/specs.csv', index_col=0)
CC = pd.read_csv(A + 'couts/contract_costs_per_market.csv', index_col=0)
_t = pd.read_csv(HERE + 'contract_table_check.csv', index_col=0)
NOTIONAL = _t['notional_specs']          # USD notional of one liquid contract, 2026-07-10
COST_SIDE = _t['cost_side_usd']          # USD fee + half-spread per contract side (ER4 at 0.0025)
STRICT = list(SP.index[SP.liquid_access.eq('OK')])

START, END = pd.Timestamp('1999-01-04'), pd.Timestamp('2026-07-10')
IDX = RETS.loc[START:END].index
tb = rb.load_close(SCR + 'mkt/us-etf/alm0421_macro/DTB3.csv', 'value') / 100
TBR = tb.reindex(IDX.union(tb.index)).ffill().reindex(IDX).shift(1).bfill()
DAYS = pd.Series(IDX.to_series().diff().dt.days.fillna(1).values, index=IDX)
TB = TBR * DAYS / 365
_e = RETS.ewm(span=32, min_periods=32).std() * 16
_y = RETS.rolling(252, min_periods=126).std() * 16
SIG = np.maximum(_e, _y).ffill().reindex(IDX)


def contracts(s, cols, C):
    X = (s['target'].reindex(IDX)[cols] * C / NOTIONAL[cols]).values
    B = (0.1 * s['unit'].reindex(IDX)[cols].abs() * C / NOTIONAL[cols]).values
    n = np.zeros_like(X)
    cur = np.zeros(X.shape[1])
    for i in range(len(X)):
        x = X[i]
        lo, hi = np.round(x - B[i]), np.round(x + B[i])
        cur = np.where(np.isnan(x), 0.0, np.clip(cur, lo, hi))
        n[i] = cur
    return pd.DataFrame(n, index=IDX, columns=cols)


def simulate(v, cols, C, lag=1.5, fixed_usd=400.0):
    cols = list(cols)
    s = system(v, cols, lag)
    n = contracts(s, cols, C)
    frac = n * NOTIONAL[cols] / C
    a = lag - 1.0
    held = ((1 - a) * frac.shift(1) + a * frac.shift(2)).fillna(0)
    r = RETS.reindex(IDX)[cols].fillna(0)
    gross = (held * r).sum(axis=1)
    dn = n.diff().abs().fillna(n.abs())
    dn_ex = (1 - a) * dn + a * dn.shift(1).fillna(0)
    roll_sides = n.abs() * 2 * ROLLS_YR[cols] / 252
    cost = ((dn_ex + roll_sides) * COST_SIDE[cols]).sum(axis=1) / C
    net = gross - cost
    margin = (frac.abs() * 0.25 * SIG[cols]).sum(axis=1)
    bal = (C * (1 - margin.shift(1).bfill()) - 10e3).clip(lower=0) * min(C / 100e3, 1.0)
    cash = bal * (TBR - 0.005).clip(lower=0) * DAYS / 365 / C
    fixed = fixed_usd / C * DAYS / 365
    return dict(s=s, n=n, frac=frac, net=net, gross=gross, cost=cost, margin=margin, cash=cash, fixed=fixed,
                total=net + cash - fixed, gross_exp=frac.abs().sum(axis=1))


def cagr(x):
    x = x.dropna()
    return (1 + x).prod() ** (365.25 / (x.index[-1] - x.index[0]).days) - 1


def sr(x):
    x = x.dropna()
    return x.mean() / x.std() * 16


def mdd(x):
    c = (1 + x.dropna()).cumprod()
    return (c / c.cummax() - 1).min()


PERIODS = {'2000-2026': ('2000-01-01', '2026-07-10'), '2010-2026': ('2010-01-01', '2026-07-10'),
           '2015-2026': ('2015-01-01', '2026-07-10'), '2023-07_2026-07': ('2023-07-10', '2026-07-10')}
