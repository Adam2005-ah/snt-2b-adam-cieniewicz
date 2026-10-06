"""Historical returns of the 5 variants at three levels of 'net' (no tax anywhere).

L0 gross    : published positions, before any trading / roll cost ('gross' series) + full US T-bill.
L1 backtest : as published (bp cost model, 84 markets, same-close execution) + full US T-bill.
L2 realistic: tradable universe (84 - CUA1 - SCO1 = 82 markets, weights/IDM recomputed),
              half-day execution lag ('manual_0.5d', lag 1.5, cv_donnees method),
              real IBKR per-contract costs (commission + exchange fees + half-spread) of the STANDARD contract,
              expressed per unit of notional traded (real_bp of couts/contract_costs_per_market.csv, ER4 half-spread
              corrected 0.0072 -> 0.0025), rolls = 2 x cost x rolls/yr,
              IBKR cash rule on the cash leg: margin share m_t = 0.17 x G_t / 8.7837 (G_t gross exposure held,
              8.7837 = v5 published mean) earns 0, the rest earns max(T-bill - 0.5 %, 0).
Conventions (project, rapport_pdf.py): cash = DTB3 of the previous day x calendar days / 365 on the futures calendar
starting 1990-01-01; CAGR in calendar time with years = days/365.25 + 1/261; vol = std(daily total) x 16;
Sharpe = mean/std of daily excess over T-bill x 16; costs/yr = sum of daily costs / years.
"""
import pickle
import numpy as np
import pandas as pd
import engine as E
import run_backtests as rb

OUT = E.OUT
START = '1990-01-01'
END = E.rets.index[-1]
PERIODS = {'1990-2026': '1990-01-01', '2000-2026': '2000-01-01', '2010-2026': '2010-01-01',
           '2015-2026': '2015-01-01', '2023-07->2026-07': '2023-07-11'}  # last: 3 years from close 10/07/2023
DROP = ['CUA1 Comdty', 'SCO1 Comdty']
LAG = 1.5
MARGIN_V5, G_V5 = 0.17, 8.7837
IB_HAIRCUT = 0.005

# ------------------------------------------------------------------ cash legs
idx = E.rets.loc[START:].index
rate = rb.load_close(f'{E.MKT}/us-etf/alm0421_macro/DTB3.csv', 'value') / 100
rate = rate.reindex(rate.index.union(idx)).ffill().reindex(idx)
days = pd.Series(idx, index=idx).diff().dt.days.fillna(1)
prev = rate.shift(1).fillna(rate)
TBILL = prev * days / 365
IB_RATE = (prev - IB_HAIRCUT).clip(lower=0) * days / 365   # what non-margin cash earns at IBKR

# ------------------------------------------------------------------ real costs
cc = pd.read_csv(OUT + '../couts/contract_costs_per_market.csv', index_col=0)
cc.loc['ER4 Comdty', 'hs'] = 0.0025
cc['cost_side_usd_corr'] = cc['fee'] * cc['fx'] + cc['hs'] * cc['mult'] * cc['fx']
cc['real_frac'] = cc['cost_side_usd_corr'] / cc['notional_usd']
chk = (cc['real_frac'] * 1e4 - cc['real_bp']).abs().drop('ER4 Comdty').max()
assert chk < 1e-9, chk
REAL = cc['real_frac'].reindex(E.rets.columns)
REAL_ROLL = 2 * REAL * pd.Series({c: rb.ROLLS_PER_YEAR[g] for c, g in E.groups.items()})
pd.DataFrame({'group': E.groups, 'bp_model_bp': E.BP_COSTS * 1e4, 'real_bp_corrected': REAL * 1e4,
              'roll_bp_model_pa_bp': E.BP_ROLL * 1e4, 'roll_real_pa_bp': REAL_ROLL * 1e4,
              'tradable': ~E.rets.columns.isin(DROP)}).to_csv(OUT + 'cost_vectors.csv', float_format='%.4f')
print('ER4 real bp corrected:', round(REAL['ER4 Comdty'] * 1e4, 4))

# ------------------------------------------------------------------ runs
pub = pickle.load(open(OUT + 'runs_published.pkl', 'rb'))
cols = [c for c in E.rets.columns if c not in DROP]
L2 = {v: E.variant(v, cols, lag=LAG, costs=REAL, roll=REAL_ROLL) for v in E.VARIANTS}

series = {}   # (variant, level) -> DataFrame of daily components from 1990
for v in E.VARIANTS:
    p = pub[v]
    held_pub = p['positions'].shift(1).fillna(0)
    for lvl, net, trad, rol, gexp in [
        ('L0 brut', p['gross'], 0 * p['gross'], 0 * p['gross'], held_pub.abs().sum(axis=1)),
        ('L1 backtest', p['net'], p['trading_costs'], p['roll_costs'], held_pub.abs().sum(axis=1)),
        ('L2 réaliste', L2[v]['net'], L2[v]['trading_costs'], L2[v]['roll_costs'], L2[v]['held'].abs().sum(axis=1)),
    ]:
        df = pd.DataFrame({'fut': net, 'trading': trad, 'roll': rol, 'gross_exp': gexp}).loc[START:]
        if lvl.startswith('L2'):
            m = (MARGIN_V5 * df['gross_exp'] / G_V5).clip(upper=1)
            cash = (1 - m) * IB_RATE
        else:
            m = 0 * df['fut']
            cash = TBILL.copy()
        df['margin_share'] = m
        df['cash'] = cash
        df['shortfall'] = TBILL - cash
        df['total'] = df['fut'] + cash
        df['excess'] = df['total'] - TBILL
        series[(v, lvl)] = df
pickle.dump({'series': series, 'tbill': TBILL, 'ib_rate': IB_RATE}, open(OUT + 'series.pkl', 'wb'))


# ------------------------------------------------------------------ metrics
def years(ix):
    return (ix[-1] - ix[0]).days / 365.25 + 1 / 261


def cagr(x):
    return (1 + x).prod() ** (1 / years(x.index)) - 1


def maxdd(x):
    c = (1 + x).cumprod()
    return (c / c.cummax() - 1).min()


rows = []
for (v, lvl), df in series.items():
    for pname, start in PERIODS.items():
        d = df.loc[start:]
        y = years(d.index)
        met = {
            'cagr_excess': cagr(d['excess']),
            'cagr_total': cagr(d['total']),
            'vol': d['total'].std() * 16,
            'sharpe': d['excess'].mean() / d['excess'].std() * 16,
            'maxdd_total': maxdd(d['total']),
            'cost_pa': (d['trading'] + d['roll']).sum() / y,
            'trading_cost_pa': d['trading'].sum() / y,
            'roll_cost_pa': d['roll'].sum() / y,
            'cash_shortfall_pa': d['shortfall'].sum() / y,
            'tbill_pa': TBILL.loc[start:].sum() / y,
            'cagr_futures_excess': cagr(d['fut']),          # before the cash shortfall (= excess at L0/L1)
            'sharpe_futures_excess': d['fut'].mean() / d['fut'].std() * 16,
            'maxdd_excess': maxdd(d['excess']),
            'gross_exposure_mean': d['gross_exp'].mean(),
            'margin_share_mean': d['margin_share'].mean(),
            'years': y,
        }
        for k, val in met.items():
            rows.append(dict(variant=v, level=lvl, period=pname, metric=k, value=val))
res = pd.DataFrame(rows)
res.to_csv(OUT + 'var_passe_global.csv', index=False, float_format='%.6f')

# ------------------------------------------------------------------ calendar years
yr = {}
for (v, lvl), df in series.items():
    t = df['total'].loc['2010':]
    yr[(v, lvl)] = (1 + t).groupby(t.index.year).prod() - 1
yr[('Monétaire (T-bill US)', 'référence')] = (1 + TBILL.loc['2010':]).groupby(TBILL.loc['2010':].index.year).prod() - 1
ibc = series[('5. Les deux', 'L2 réaliste')]['cash'].loc['2010':]
yr[('Cash IBKR (v5, marge non rémunérée)', 'référence')] = (1 + ibc).groupby(ibc.index.year).prod() - 1
ann = pd.DataFrame(yr).T
ann.columns = [str(c) if c < 2026 else '2026 (au 10/07)' for c in ann.columns]
ann.index.names = ['variant', 'level']
ann.to_csv(OUT + 'annees.csv', float_format='%.4f')

# ------------------------------------------------------------------ printout
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
for metric in ['cagr_total', 'cagr_excess', 'sharpe', 'vol', 'maxdd_total', 'cost_pa', 'cash_shortfall_pa',
               'gross_exposure_mean']:
    t = res[res.metric == metric].pivot_table(index=['variant', 'level'], columns='period', values='value', sort=False)
    print('\n', metric)
    print((t * (1 if metric in ('sharpe', 'gross_exposure_mean') else 100)).round(2).to_string())
print('\nAnnées (rendement total, %)')
print((ann * 100).round(1).to_string())
