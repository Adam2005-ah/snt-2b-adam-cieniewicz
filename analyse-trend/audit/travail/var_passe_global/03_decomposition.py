"""Step-by-step bridge L1 -> L2 for each variant, plus sensitivities of L2.

Steps (cumulative):
  S0 L1 publié            : 84 markets, same-close, bp model, full T-bill
  S1 univers 82           : drop CUA1 + SCO1 (weights / IDM recomputed)
  S2 + exécution 0,5 j    : lag 1.5
  S3 + coûts IBKR réels   : real_bp of the standard contract (2026 prices), rolls 2 x cost x rolls/yr
  S4 + cash IBKR (= L2)   : margin share earns 0, rest earns max(T-bill - 0.5 %, 0)
Sensitivities on L2:
  X1 coûts fixes en USD/contrat : same USD (local-ccy) cost per contract as today, divided by the HISTORICAL notional
     (mult x actual price level, simulation/levels.pkl) -> higher bp costs when prices were lower. Still optimistic for
     the pit era (1990s commissions and tick spreads were wider).
  X2 coûts réels x 2
  X3 marge 23 % (couts/margin_estimate.csv exchange initial margin for v5) instead of 17 %
  X4 exécution 1 jour plein (lag 2.0)
"""
import pickle
import numpy as np
import pandas as pd
import engine as E

exec(open(E.OUT + '02_main.py').read().split('# ------------------------------------------------------------------ runs')[0])

cols82 = [c for c in E.rets.columns if c not in DROP]
cc = pd.read_csv(E.OUT + '../couts/contract_costs_per_market.csv', index_col=0)
cc.loc['ER4 Comdty', 'hs'] = 0.0025
lev = pd.read_pickle(E.OUT + '../simulation/levels.pkl').reindex(E.rets.index).ffill().bfill()
# cost per unit notional = (fee + hs x mult) / (mult x level_t), both in the contract's currency (FX cancels)
usd_const = ((cc['fee'] / cc['mult']).reindex(lev.columns) + cc['hs'].reindex(lev.columns)) / lev
chk = (usd_const.iloc[-1] - REAL).abs().max()
print('X1 cost at 2026-07-10 equals real cost:', f'{chk:.2e}')
rpy = pd.Series({c: rb.ROLLS_PER_YEAR[g] for c, g in E.groups.items()})
usd_const_roll = 2 * usd_const * rpy
print('X1 mean cost bp 1990s vs 2026 (v-weighted later); median across markets 1995:',
      round((usd_const.loc['1995-06-30', cols82] * 1e4).median(), 2), ' 2026:', round((REAL[cols82] * 1e4).median(), 2))

PER = {'1990': '1990-01-01', '2000': '2000-01-01', '2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-11'}


def years(ix):
    return (ix[-1] - ix[0]).days / 365.25 + 1 / 261


def summarize(v, step, run, margin=None):
    held = run['held'] if 'held' in run else run['positions'].shift(1).fillna(0)
    df = pd.DataFrame({'fut': run['net'], 'cost': run['trading_costs'] + run['roll_costs'],
                       'g': held.abs().sum(axis=1)}).loc[START:]
    if margin is None:
        cash = TBILL
    else:
        cash = (1 - (margin * df['g'] / G_V5).clip(upper=1)) * IB_RATE
    total = df['fut'] + cash
    exc = total - TBILL
    out = []
    for k, s in PER.items():
        t, e = total.loc[s:], exc.loc[s:]
        y = years(t.index)
        out += [dict(variant=v, step=step, period=k, metric='sharpe', value=e.mean() / e.std() * 16),
                dict(variant=v, step=step, period=k, metric='cagr_total', value=(1 + t).prod() ** (1 / y) - 1),
                dict(variant=v, step=step, period=k, metric='cost_pa', value=df['cost'].loc[s:].sum() / y)]
    return out


pub = pickle.load(open(E.OUT + 'runs_published.pkl', 'rb'))
rows = []
for v in E.VARIANTS:
    rows += summarize(v, 'S0 L1 publié', pub[v])
    rows += summarize(v, 'S1 univers 82', E.variant(v, cols82))
    rows += summarize(v, 'S2 + exécution 0,5 j', E.variant(v, cols82, lag=1.5))
    l2 = E.variant(v, cols82, lag=1.5, costs=REAL, roll=REAL_ROLL)
    rows += summarize(v, 'S3 + coûts IBKR réels', l2)
    rows += summarize(v, 'S4 + cash IBKR (= L2)', l2, margin=MARGIN_V5)
    rows += summarize(v, 'X1 L2, coûts fixes en USD/contrat', E.variant(v, cols82, lag=1.5, costs=usd_const[cols82],
                                                                        roll=usd_const_roll[cols82]), margin=MARGIN_V5)
    rows += summarize(v, 'X2 L2, coûts réels x2', E.variant(v, cols82, lag=1.5, costs=2 * REAL, roll=2 * REAL_ROLL),
                      margin=MARGIN_V5)
    rows += summarize(v, 'X3 L2, marge 23 %', l2, margin=0.23)
    rows += summarize(v, 'X4 L2, exécution 1 jour', E.variant(v, cols82, lag=2.0, costs=REAL, roll=REAL_ROLL),
                      margin=MARGIN_V5)
    print('done', v, flush=True)
df = pd.DataFrame(rows)
df.to_csv(E.OUT + 'decomposition.csv', index=False, float_format='%.6f')
pd.set_option('display.width', 250)
for m in ['sharpe', 'cagr_total', 'cost_pa']:
    t = df[df.metric == m].pivot_table(index=['variant', 'step'], columns='period', values='value', sort=False)
    print('\n', m)
    print((t * (1 if m == 'sharpe' else 100)).round(2).to_string())
