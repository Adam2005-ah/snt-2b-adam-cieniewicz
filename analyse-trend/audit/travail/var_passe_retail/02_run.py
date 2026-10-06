"""Historical returns of each variant as a retail investor could have run it at IBKR (USD 100k / 250k / 1M).

Per variant x tier:
  * own feasible subset (01_subsets.py), system re-run on it (weights, IDM, vol-targeting recomputed);
  * whole contracts with Carver contract buffering (simulation/sim.integer_positions), smallest LIQUID contract,
    'today' notional mode (2026-07-10 notional and spread held constant through history, as the audit's forward-
    looking mode), capital held fixed at C, P&L as % of C;
  * real per-contract costs at IBKR: fee (commission + exchange) + half-spread, on trades and on rolls (2 sides per
    roll), ER4 half-spread 0.0025 (ICE) instead of 0.0072;
  * execution: half-day lag (P&L = 0.5 pos(t-1) r + 0.5 pos(t-2) r, trading costs booked half on t, half on t+1,
    vol-target scale measured on the lagged P&L) = headline; same-close run kept for reconciliation with the audit;
  * IBKR cash leg: margin (flat-k model of the audit, 0.25 x sigma x |exposure|; ~17 % of capital for v5-84) earns 0;
    the rest earns max(T-bill - 0.5 %, 0), nothing on the first USD 10k, pro-rata below 100k NAV;
  * fixed costs: IB Gateway standard budget USD 400 / yr.
Conventions: Sharpe and vol x16 (project), CAGR in calendar time, US T-bill = DTB3 accrued act/365 on calendar days.
"""
import sys, pickle, time
import numpy as np, pandas as pd
HERE = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/var_passe_retail/'
sys.path.insert(0, HERE)
import veng as V
sys.path.insert(0, V.SIMDIR)
import sim  # noqa: E402
import run_backtests as rb  # noqa: E402

t0 = time.time()
# ---- contract table: liquid contract, today's notional / spread, ER4 corrected ----
sim.CC.loc['ER4 Comdty', 'hs'] = 0.0025
tab, N, HS = sim.contract_table('liquid_mult')
N = pd.DataFrame(np.tile(N.iloc[-1].values, (len(N), 1)), index=N.index, columns=N.columns)
HS = pd.DataFrame(np.tile(HS.iloc[-1].values, (len(HS), 1)), index=HS.index, columns=HS.columns)
IDX = sim.IDX
RETS = V.RETS.reindex(IDX)
ROLLS_YR = V.E.ROLLS_YR
# margin sigma (audit verif_operations/marg.py): max(EWM32 vol, 1y vol), flat k = 0.25
_ewm = V.RETS.ewm(span=32, min_periods=32).std() * 16
_y1 = V.RETS.rolling(252, min_periods=126).std() * 16
SIG = np.maximum(_ewm, _y1).ffill().reindex(IDX)
K_MARGIN = 0.25
# T-bill
tb_rate = rb.load_close(V.E.S + 'mkt/us-etf/alm0421_macro/DTB3.csv', 'value') / 100
tb_rate = tb_rate.reindex(IDX.union(tb_rate.index)).ffill().reindex(IDX).shift(1).bfill()
DAYS = pd.Series(IDX, index=IDX).diff().dt.days.fillna(1)
TB = tb_rate * DAYS / 365
IB_RATE = (tb_rate - 0.005).clip(lower=0)
FIXED_USD = 400.0

SUBS = pickle.load(open(HERE + 'subsets_variantes.pkl', 'rb'))
REF = pickle.load(open(HERE + 'ref84.pkl', 'rb'))
TIERS = [100e3, 250e3, 1e6]
PERIODS = {'2000-2026': ('2000-01-01', '2026-07-10'), '2010-2026': ('2010-01-01', '2026-07-10'),
           '2015-2026': ('2015-01-01', '2026-07-10'), '2023-07_2026-07': ('2023-07-10', '2026-07-10')}


def tag(v, C):
    return f'{v}_{int(C / 1e3)}k' if C < 1e6 else f'{v}_1M'


def contracts_run(s, cols, C, lag):
    n = sim.integer_positions(s, cols, C, N)
    r = RETS[cols].fillna(0)
    frac = n * N[cols] / C
    fr = lag - 1.0
    held = ((1 - fr) * frac.shift(1) + fr * frac.shift(2)).fillna(0)
    gross = (held * r).sum(axis=1)
    dn = n.diff().abs().fillna(n.abs())
    dn_exec = (1 - fr) * dn + fr * dn.shift(1).fillna(0)
    rollsides = n.abs() * 2 * ROLLS_YR[cols] / 252
    sides = dn_exec + rollsides
    fee = (sides * tab.loc[cols, 'fee_spec']).sum(axis=1) / C
    spread = (sides * HS[cols]).sum(axis=1) / C
    roll_cost = (rollsides * (tab.loc[cols, 'fee_spec'] + HS[cols])).sum(axis=1) / C
    net = gross - fee - spread
    margin = (frac.abs() * SIG[cols] * K_MARGIN).sum(axis=1)
    cash_bal = (C * (1 - margin.shift(1).fillna(margin.iloc[0])) - 10e3).clip(lower=0) * min(C / 100e3, 1.0)
    cash_leg = cash_bal * IB_RATE * DAYS / 365 / C
    fixed = FIXED_USD / C * DAYS / 365
    return dict(n=n, net=net, gross=gross, fee=fee, spread=spread, roll_cost=roll_cost, margin=margin, cash_leg=cash_leg,
                fixed=fixed, total_net=net + cash_leg - fixed, total_tbill=net + TB,
                gross_exp=frac.abs().sum(axis=1), gross_exp_ex_stir=frac.loc[:, V.E.GROUPS[cols].ne('STIR').values].abs().sum(axis=1),
                orders=(dn > 0).sum(axis=1).astype(float),
                roll_orders=((n.abs() > 0) * ROLLS_YR[cols] / 252).sum(axis=1),
                contract_sides=dn.sum(axis=1) + rollsides.sum(axis=1),
                markets_held=(n != 0).sum(axis=1).astype(float))


def cagr(x):
    x = x.dropna(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (1 + x).prod() ** (1 / yrs) - 1


def mdd(x):
    c = (1 + x.dropna()).cumprod(); return (c / c.cummax() - 1).min()


def sr(x):
    x = x.dropna(); return x.mean() / x.std() * 16


rows, SER, EXTRA = [], {}, {}
for v in V.VARIANTS:
    ref84 = REF[v]['net'].reindex(IDX)
    for C in TIERS:
        cols = SUBS[(v, C)]
        k = tag(v, C)
        res = {}
        for lag in (1.0, 1.5):
            s = V.system(v, cols, lag=lag)
            res[lag] = (s, contracts_run(s, cols, C, lag))
        own_frac = res[1.0][0]['net'].reindex(IDX)          # own subset, fractional, bp model, same close
        own_frac_lag = res[1.5][0]['net'].reindex(IDX)
        for lag, (s, d) in res.items():
            for pname, (a, b) in PERIODS.items():
                sl = slice(a, b)
                yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
                ex, tn, tt, tb = d['net'].loc[sl], d['total_net'].loc[sl], d['total_tbill'].loc[sl], TB.loc[sl]
                row = dict(key=k, variant=v, variant_name=V.VARIANTS[v]['name'], capital_usd=int(C),
                           execution='half_day_lag' if lag == 1.5 else 'same_close', period=pname, n_markets=len(cols),
                           cagr_excess=cagr(ex), cagr_total_tbill=cagr(tt), cagr_total_net_all_fees=cagr(tn),
                           mean_excess_ann=ex.mean() * 256, vol=ex.std() * 16, sharpe=sr(ex),
                           sharpe_all_in=sr(tn - tb), maxdd_excess=mdd(ex), maxdd_total_net=mdd(tn),
                           cost_fee_pct=d['fee'].loc[sl].sum() / yrs * 100, cost_halfspread_pct=d['spread'].loc[sl].sum() / yrs * 100,
                           cost_fee_plus_halfspread_pct=(d['fee'] + d['spread']).loc[sl].sum() / yrs * 100,
                           of_which_rolls_pct=d['roll_cost'].loc[sl].sum() / yrs * 100,
                           cost_fixed_pct=d['fixed'].loc[sl].sum() / yrs * 100,
                           cost_cash_shortfall_pct=(tb - d['cash_leg'].loc[sl]).sum() / yrs * 100,
                           tbill_avg_pct=tb.sum() / yrs * 100, ib_cash_avg_pct=d['cash_leg'].loc[sl].sum() / yrs * 100,
                           orders_per_yr=(d['orders'].loc[sl].sum() + d['roll_orders'].loc[sl].sum()) / yrs,
                           trade_orders_per_yr=d['orders'].loc[sl].sum() / yrs, roll_orders_per_yr=d['roll_orders'].loc[sl].sum() / yrs,
                           contract_sides_per_yr=d['contract_sides'].loc[sl].sum() / yrs,
                           mean_gross_exposure=d['gross_exp'].loc[sl].mean(), mean_gross_exposure_ex_stir=d['gross_exp_ex_stir'].loc[sl].mean(),
                           mean_margin_pct=d['margin'].loc[sl].mean() * 100,
                           p95_margin_pct=d['margin'].loc[sl].quantile(0.95) * 100,
                           avg_markets_held=d['markets_held'].loc[sl].mean(),
                           ref_own_subset_fractional_sharpe=sr((own_frac_lag if lag == 1.5 else own_frac).loc[sl]),
                           ref_own_subset_fractional_cagr_excess=cagr((own_frac_lag if lag == 1.5 else own_frac).loc[sl]),
                           ref_published84_sharpe=sr(ref84.loc[sl]), ref_published84_cagr_excess=cagr(ref84.loc[sl]),
                           ref_published84_cagr_total_tbill=cagr((ref84 + TB).loc[sl]), ref_published84_vol=ref84.loc[sl].std() * 16,
                           corr_vs_own_fractional=d['net'].loc[sl].corr(own_frac.loc[sl]))
                row['cost_total_drag_pct'] = row['cost_fee_plus_halfspread_pct'] + row['cost_fixed_pct'] + row['cost_cash_shortfall_pct']
                rows.append(row)
        dl = res[1.5][1]; d1 = res[1.0][1]
        SER[k] = dl['net']
        EXTRA[k] = dict(net_same_close=d1['net'], cash_leg=dl['cash_leg'], fixed=dl['fixed'], margin=dl['margin'],
                        gross_exp=dl['gross_exp'], gross_exp_ex_stir=dl['gross_exp_ex_stir'], total_net=dl['total_net'], n=dl['n'])
        r10 = [x for x in rows if x['key'] == k and x['period'] == '2010-2026']
        print(k, len(cols), 'SR10 same/lag', [round(x['sharpe'], 3) for x in r10], 'all-in', [round(x['sharpe_all_in'], 3) for x in r10],
              'cagr net', [round(x['cagr_total_net_all_fees'] * 100, 2) for x in r10], round(time.time() - t0), flush=True)

res = pd.DataFrame(rows)
res.to_csv(HERE + 'var_passe_retail_wide.csv', index=False, float_format='%.5g')
idcols = ['key', 'variant', 'variant_name', 'capital_usd', 'execution', 'period', 'n_markets']
long = res.melt(id_vars=idcols, var_name='metric', value_name='value')
long.to_csv(HERE + 'var_passe_retail.csv', index=False, float_format='%.5g')

# ---- series for the forward Monte Carlo ----
S = pd.DataFrame(SER).loc['1999-01-04':'2026-07-10']
stats = {}
for k in S.columns:
    x = S[k]; e = EXTRA[k]
    stats[k] = dict(vol_realised_1999=x.std() * 16, vol_realised_2000=x.loc['2000':].std() * 16, vol_realised_2010=x.loc['2010':].std() * 16,
                    sharpe_2000=sr(x.loc['2000':]), sharpe_2010=sr(x.loc['2010':]), sharpe_2015=sr(x.loc['2015':]),
                    mean_gross_exposure_2000=e['gross_exp'].loc['2000':].mean(), mean_gross_exposure_2010=e['gross_exp'].loc['2010':].mean(),
                    mean_gross_exposure_ex_stir_2000=e['gross_exp_ex_stir'].loc['2000':].mean(),
                    mean_margin_2000=e['margin'].loc['2000':].mean(), mean_margin_2010=e['margin'].loc['2010':].mean(),
                    n_markets=len(SUBS[(k.split('_')[0], {'100k': 100e3, '250k': 250e3, '1M': 1e6}[k.split('_')[1]])]))
ST = pd.DataFrame(stats).T
S.attrs['description'] = ('daily net EXCESS return over cash, whole contracts, real IBKR fee + half-spread (ER4 0.0025), '
                          'half-day execution lag, capital fixed, before cash leg and fixed costs; 1999-01-04 includes the initial trade-in')
S.attrs['stats'] = ST.to_dict()
S.to_pickle(HERE + 'series_variantes.pkl')
ST.to_csv(HERE + 'series_variantes_stats.csv', float_format='%.5g')
pickle.dump({k: {kk: vv for kk, vv in e.items() if kk != 'n'} for k, e in EXTRA.items()}, open(HERE + 'series_variantes_extra.pkl', 'wb'))
print(ST.round(3).to_string())
# ---- compact summary (headline = half-day lag) ----
keep = ['key', 'variant', 'variant_name', 'capital_usd', 'period', 'n_markets', 'cagr_excess', 'cagr_total_tbill', 'cagr_total_net_all_fees',
        'vol', 'sharpe', 'sharpe_all_in', 'maxdd_excess', 'maxdd_total_net', 'cost_fee_plus_halfspread_pct', 'cost_fixed_pct',
        'cost_cash_shortfall_pct', 'cost_total_drag_pct', 'orders_per_yr', 'mean_margin_pct',
        'ref_published84_sharpe', 'ref_published84_cagr_total_tbill', 'ref_own_subset_fractional_sharpe']
H = res[res.execution == 'half_day_lag'][keep].copy()
sc = res[res.execution == 'same_close'].set_index(['key', 'period'])['sharpe']
H['sharpe_same_close'] = [sc[(k, p)] for k, p in zip(H.key, H.period)]
H.to_csv(HERE + 'var_passe_retail_resume.csv', index=False, float_format='%.4g')
print('done', round(time.time() - t0))
