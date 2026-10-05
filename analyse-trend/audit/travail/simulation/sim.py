"""Variant 5 traded in whole contracts, capital held fixed, P&L as % of that capital.
Carver contract buffering (AFTS strategy 8): B = 0.1 x (position at forecast 10, in contracts); on a rebalancing day
lower = round(target - B), upper = round(target + B); trade to the nearest edge only if the current position is
outside [lower, upper]. Decided at close t, held from t+1 (as in trend.py). Micro contracts assumed to have existed
throughout (they mostly date from 2019-2024)."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation')
import engine as E
S, OUT = E.S, E.OUT
SP = pd.read_csv(S + 'audit/capital/specs.csv', index_col=0)
CC = pd.read_csv(S + 'audit/couts/contract_costs_per_market.csv', index_col=0)
L = pd.read_pickle(OUT + 'levels.pkl'); FX = pd.read_pickle(OUT + 'fx.pkl')
PX = pd.read_csv(S + 'mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
SIM_START, END = pd.Timestamp('1999-01-04'), pd.Timestamp('2026-07-10')
IDX = E.RETS.loc[SIM_START:END].index
TODAY = END
L = L.reindex(IDX).ffill()
L = L.fillna(PX.reindex(IDX).ffill())       # fallback: ratio path where a proxy has a hole
# coffee before the first Yahoo print (2000-01-03): hold that level (ratio path is ~6x too high there)
L.loc[:'2000-01-02', 'KC1 Comdty'] = L.loc['2000-01-03', 'KC1 Comdty']
L = L.bfill()
FX = FX.reindex(IDX).ffill().bfill()

def contract_table(multcol):
    mult = SP[multcol]
    micro = mult < SP['std_mult']
    fee_spec, hs_px, note = {}, {}, {}
    for t in SP.index:
        m = mult[t]
        if np.isnan(m):
            continue
        c = CC.loc[t]
        if abs(m / c['mult'] - 1) < 0.05:
            fee_spec[t] = c['fee'] * c['fx']; hs_px[t] = c['hs']; note[t] = 'std (pst fee+spread)'
        elif m == c['mult_s']:
            fee_spec[t] = c['fee_s'] * c['fx']; hs_px[t] = c['hs_s']; note[t] = 'small (fee+spread from cost audit)'
        else:   # micro not in the cost table: USD 1.0 fee, half-spread = 1.5 x the standard contract's (price units)
            fee_spec[t] = 1.0; hs_px[t] = 1.5 * c['hs']; note[t] = 'micro (assumed fee 1.0, 1.5x std half-spread)'
    tab = pd.DataFrame({'mult': mult, 'micro': micro, 'fee_flat': np.where(micro, 1.0, 2.5),
                        'fee_spec': pd.Series(fee_spec), 'hs_px': pd.Series(hs_px), 'note': pd.Series(note)}).dropna(subset=['mult'])
    N = pd.DataFrame({t: tab.loc[t, 'mult'] * L[t] * FX[t] for t in tab.index})        # USD notional per contract
    HS = pd.DataFrame({t: tab.loc[t, 'hs_px'] * tab.loc[t, 'mult'] * FX[t] for t in tab.index})  # USD half-spread
    return tab, N, HS

TAB = {'small': contract_table('small_mult'), 'liquid': contract_table('liquid_mult')}
# forward-looking variant: today's (2026-07-10) contract notional and spread held constant through history
for k in ['small', 'liquid']:
    tab, N, HS = TAB[k]
    TAB[k + '_today'] = (tab, pd.DataFrame(np.tile(N.iloc[-1].values, (len(N), 1)), index=N.index, columns=N.columns),
                         pd.DataFrame(np.tile(HS.iloc[-1].values, (len(HS), 1)), index=HS.index, columns=HS.columns))

def integer_positions(sysout, cols, C, N, min_buffer=0.0):
    tgt = sysout['target'].reindex(IDX)[cols]; unit = sysout['unit'].reindex(IDX)[cols]
    mask = sysout['mask'].reindex(IDX).values
    Nn = N[cols].values
    X = (tgt.values * C / Nn); B = np.maximum(0.1 * np.abs(unit.values) * C / Nn, min_buffer)
    n = np.zeros_like(X); cur = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        x = X[i]; ok = ~np.isnan(x)
        if mask[i]:
            lo = np.round(x - B[i]); hi = np.round(x + B[i])
            cur = np.where(ok, np.minimum(np.maximum(cur, lo), hi), 0.0)
        else:
            cur = np.where(ok, cur, 0.0)
        n[i] = cur
    return pd.DataFrame(n, index=IDX, columns=cols)

def daily_series(n, C, N, HS, tab, cols):
    r = E.RETS.reindex(IDX)[cols].fillna(0)
    frac = n * N[cols] / C
    gross = (frac.shift(1).fillna(0) * r).sum(axis=1)
    dn = n.diff().abs().fillna(n.abs())
    rollsides = n.abs() * 2 * E.ROLLS_YR[cols] / 252                       # expected roll sides per day
    bp_cost = (dn * N[cols] / C * E.COSTS[cols]).sum(axis=1) + (frac.shift(1).fillna(0).abs() * E.ROLLC[cols] / 252).sum(axis=1)
    sides = dn + rollsides
    real_cost = (sides * (tab.loc[cols, 'fee_spec'] + HS[cols])).sum(axis=1) / C
    flat_comm = (sides * tab.loc[cols, 'fee_flat']).sum(axis=1) / C
    spread_cost = (sides * HS[cols]).sum(axis=1) / C
    return dict(gross=gross, net_bp=gross - bp_cost, net_real=gross - real_cost, bp_cost=bp_cost, real_cost=real_cost,
                flat_comm=flat_comm, spread_cost=spread_cost, trade_sides=dn.sum(axis=1), roll_sides=rollsides.sum(axis=1),
                orders=(dn > 0).sum(axis=1), roll_orders=((n.abs() > 0) * E.ROLLS_YR[cols] / 252).sum(axis=1),
                open_contracts=n.abs().sum(axis=1), markets_held=(n != 0).sum(axis=1), frac=frac)

def perf(x):
    x = x.dropna()
    cum = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return dict(sharpe=x.mean() / x.std() * 16, mean_ann=x.mean() * 252, cagr=cum.iloc[-1] ** (1 / yrs) - 1,
                vol=x.std() * 16, maxdd=(cum / cum.cummax() - 1).min())

IDEAL = E.system(list(E.RETS.columns))
IDEAL_NET = IDEAL['net']
PERIODS = {'2000-2026': ('2000-01-01', '2026-07-10'), '2010-2026': ('2010-01-01', '2026-07-10')}

def evaluate(label, cols, C, sysout, contract, freq='D', extra=None, min_buffer=0.0):
    tab, N, HS = TAB[contract]
    n = integer_positions(sysout, cols, C, N, min_buffer)
    d = daily_series(n, C, N, HS, tab, cols)
    own = sysout['net'].reindex(IDX)
    tgt = sysout['pos'].reindex(IDX)[cols]
    rows = []
    for pname, (a, b) in PERIODS.items():
        sl = slice(a, b)
        yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        nr, nb, g = d['net_real'].loc[sl], d['net_bp'].loc[sl], d['gross'].loc[sl]
        ideal, ownp = IDEAL_NET.loc[sl], own.loc[sl]
        pr, pb, po, pi = perf(nr), perf(nb), perf(ownp), perf(ideal)
        rows.append(dict(universe=label, capital_usd=C, freq=freq, period=pname, n_markets=len(cols),
            sharpe_net_real=round(pr['sharpe'], 3), sharpe_net_bpmodel=round(pb['sharpe'], 3), sharpe_gross=round(perf(g)['sharpe'], 3),
            mean_excess_ann_net_real=round(pr['mean_ann'], 4), cagr_excess_net_real=round(pr['cagr'], 4),
            vol=round(pr['vol'], 4), maxdd_excess_net_real=round(pr['maxdd'], 4),
            corr_vs_ideal84=round(nb.corr(ideal), 3), te_vs_ideal84=round((nb - ideal).std() * 16, 4),
            corr_vs_own_fractional=round(nb.corr(ownp), 3), te_vs_own_fractional=round((nb - ownp).std() * 16, 4),
            own_fractional_sharpe=round(po['sharpe'], 3), own_fractional_cagr=round(po['cagr'], 4), own_fractional_vol=round(po['vol'], 4),
            ideal84_sharpe=round(pi['sharpe'], 3), ideal84_cagr=round(pi['cagr'], 4),
            share_days_mkts_target_nonzero_but_0_contracts=round(((d['frac'].loc[sl] == 0) & (tgt.loc[sl].abs() > 0)).values.sum()
                                                                  / max((tgt.loc[sl].abs() > 0).values.sum(), 1), 3),
            avg_open_contracts=round(d['open_contracts'].loc[sl].mean(), 1), avg_markets_held=round(d['markets_held'].loc[sl].mean(), 1),
            trade_contract_sides_per_yr=round(d['trade_sides'].loc[sl].sum() / yrs), roll_contract_sides_per_yr=round(d['roll_sides'].loc[sl].sum() / yrs),
            trade_orders_per_yr=round(d['orders'].loc[sl].sum() / yrs), roll_orders_per_yr=round(d['roll_orders'].loc[sl].sum() / yrs),
            commission_flat_usd_per_yr=round(d['flat_comm'].loc[sl].sum() * C / yrs), commission_flat_pct_cap=round(d['flat_comm'].loc[sl].sum() / yrs * 100, 3),
            fee_plus_halfspread_pct_cap=round(d['real_cost'].loc[sl].sum() / yrs * 100, 3), halfspread_pct_cap=round(d['spread_cost'].loc[sl].sum() / yrs * 100, 3),
            bpmodel_cost_pct_cap=round(d['bp_cost'].loc[sl].sum() / yrs * 100, 3),
            gross_mean_ann=round(g.mean() * 252, 4), markets=' | '.join(cols) if len(cols) < 84 else 'all 84', **(extra or {})))
    return rows, n, d
