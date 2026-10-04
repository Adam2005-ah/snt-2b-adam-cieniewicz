"""Pourquoi les années 1990 ont été si bonnes : contributions par classe, par marché et par mois,
force des tendances par décennie, corrélation entre marchés, coûts, prix figés, et comparaison avec
les vrais fonds (BTOP50) et la réplication académique AQR TSMOM.

Usage : python annees_1990.py DOSSIER_SCRATCHPAD (qui contient mkt/)
"""
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import run_backtests as rb, trend, listings
SP = sys.argv[1]
full, groups, names = rb.load_futures(f'{SP}/mkt')
rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, '1900-01-01'))))
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
costs = {c: rb.COSTS[g] for c, g in groups.items()}
roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
res = trend.run_portfolio(rets, risk, costs, roll, freq='D')
net = res['net']
held = res['positions'].shift(1).fillna(0)
inst = held * rets.fillna(0) - res['positions'].diff().abs().fillna(0) * pd.Series(costs) - held.abs() * pd.Series(roll) / 252
cls = inst.T.groupby(groups).sum().T
print('live instruments on 1990-06-29:', int(res['live'].loc['1990-06-29'].sum()), ' 1995:', int(res['live'].loc['1995-06-30'].sum()), ' 2005:', int(res['live'].loc['2005-06-30'].sum()), ' 2020:', int(res['live'].loc['2020-06-30'].sum()))
yrs = (1 + net.loc['1990':'1999']).groupby(net.loc['1990':'1999'].index.year).prod() - 1
A = cls.loc['1990':'1999'].groupby(cls.loc['1990':'1999'].index.year).sum()
A['TOTAL net'] = yrs
print(A.round(3).to_string())
dec = {}
for a, b in [('1990', '1999'), ('2000', '2009'), ('2010', '2019'), ('2020', '2026')]:
    c = cls.loc[a:b]; dec[f'{a}s'] = c.mean() * 252
print('\navg annual contribution by class per decade:'); print(pd.DataFrame(dec).round(3).to_string())
I = inst.loc['1990':'1999'].sum().sort_values()
print('\nTop 12 instruments 1990s (cumulative arithmetic, fraction of capital):')
print(pd.DataFrame({'pnl': I.tail(12)[::-1].round(3), 'name': names.reindex(I.tail(12)[::-1].index)}).to_string())
print('Worst 5:'); print(pd.DataFrame({'pnl': I.head(5).round(3), 'name': names.reindex(I.head(5).index)}).to_string())
# standalone per-instrument trend Sharpe median by decade (instrument P&L sharpe)
med = {}
for a, b in [('1990', '1999'), ('2000', '2009'), ('2010', '2019'), ('2020', '2026')]:
    x = inst.loc[a:b]
    x = x.loc[:, (x != 0).sum() > 500]
    sh = x.mean() / x.std() * 16
    med[f'{a}s'] = {'median instrument Sharpe': sh.median(), 'n': x.shape[1],
                    'avg pairwise corr of instrument P&L': (x.corr().values[np.triu_indices(x.shape[1], 1)]).mean(),
                    'portfolio Sharpe': net.loc[a:b].mean() / net.loc[a:b].std() * 16}
print('\n', pd.DataFrame(med).round(3).to_string())
# best months 1990s
m = (1 + net.loc['1990':'1999']).resample('ME').prod() - 1
clm = cls.loc['1990':'1999'].resample('ME').sum()
print('\nBest 10 months 1990s:')
for d in m.sort_values(ascending=False).head(10).index:
    top = inst.loc[d - pd.offsets.MonthBegin(1):d].sum().sort_values(ascending=False).head(3)
    print(d.strftime('%Y-%m'), round(m[d], 3), '|', {names.get(k, k): round(v, 3) for k, v in top.items()})
# costs x3 in 1990s
r3 = trend.run_portfolio(rets, risk, {c: 3*v for c, v in costs.items()}, {c: 3*v for c, v in roll.items()}, freq='D')['net']
r5 = trend.run_portfolio(rets, risk, {c: 5*v for c, v in costs.items()}, {c: 5*v for c, v in roll.items()}, freq='D')['net']
sh = lambda d: d.mean() / d.std() * 16
print('\n1990s Sharpe costs x1/x3/x5:', round(sh(net.loc['1990':'1999']), 2), round(sh(r3.loc['1990':'1999']), 2), round(sh(r5.loc['1990':'1999']), 2))
# stale price check: lag-1 autocorr of daily returns by decade, median across instruments
ac = {}
for a, b in [('1990', '1999'), ('2010', '2019')]:
    x = rets.loc[a:b]; x = x.loc[:, x.notna().sum() > 500]
    ac[f'{a}s'] = x.apply(lambda s: s.dropna().autocorr()).describe()[['mean', '50%', 'max']]
print('\nlag-1 autocorr of daily returns:'); print(pd.DataFrame(ac).round(3).to_string())
x = rets.loc['1990':'1999']; x = x.loc[:, x.notna().sum() > 500]
a1 = x.apply(lambda s: s.dropna().autocorr()).sort_values(ascending=False).head(6)
print('highest autocorr 1990s:', {names.get(k, k): round(v, 2) for k, v in a1.items()})
# real CTA indices
def mstats(path, col='close', excess=True):
    s = pd.read_csv(path, parse_dates=['date']).set_index('date')[col]
    r = s.pct_change().dropna()
    tb = rb.load_close(f'{SP}/mkt/us-etf/alm0421_macro/DTB3.csv', 'value') / 100 / 12
    tbm = tb.resample('ME').mean().reindex(r.index, method='nearest')
    x = r - tbm if excess else r
    out = {}
    for a, b in [('1990', '1999'), ('2000', '2009'), ('2010', '2019'), ('2020', '2026')]:
        y = x.loc[a:b]; out[f'{a}s'] = f'{y.mean()*12:.3f} / SR {y.mean()/y.std()*np.sqrt(12):.2f}'
    return out
print('\nBTOP50 (real CTAs, net of fees, excess of T-bill):', mstats(f'{SP}/mkt/alternatives/pofo_indices/BTOP50_monthly.csv'))
print('AQR TSMOM (academic replication, excess index):', mstats(f'{SP}/mkt/alternatives/pofo_indices/AQR_TSMOM_excess_index_monthly.csv', excess=False))
nm = (1 + net).resample('ME').prod() - 1
print('This system (monthly):', {f'{a}s': f'{nm.loc[a:b].mean()*12:.3f} / SR {nm.loc[a:b].mean()/nm.loc[a:b].std()*np.sqrt(12):.2f}' for a, b in [('1990','1999'),('2000','2009'),('2010','2019'),('2020','2026')]})
