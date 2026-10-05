"""Task 3: was the capital-tier subset selection look-ahead? Re-run the capital audit's greedy rule
(capital/tiers.py + simulation/run_all.py) with (a) the original 2023-07..2026-07 median |position| window
(hindsight, reproduces the audit) and (b) the 3 years BEFORE 2023-07-10 with notionals rolled back to 2023-07."""
import numpy as np, pandas as pd, core, trend
SP = pd.read_csv(core.S + 'audit/capital/specs.csv', index_col=0)
STRICT = SP['liquid_access'].eq('OK')
risk = core.risk
px = pd.read_csv(core.MKT + '/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
full = core.v5()
U = pd.read_pickle(core.S + 'audit/simulation/subsets.pkl')

def greedy(universe, k, C, m):
    u = pd.DataFrame({'k': k[universe], 'cls': risk[universe]}).dropna()
    order_in = {c: list(g.sort_values('k').index) for c, g in u.groupby('cls')}
    cls_order = sorted(order_in, key=lambda c: u.loc[order_in[c][0], 'k'])
    seq = []
    while any(order_in.values()):
        for c in cls_order:
            if order_in[c]: seq.append(order_in[c].pop(0))
    chosen = []
    for t in seq:
        trial = chosen + [t]; cl = risk[trial]; ncl = cl.nunique(); cnt = cl.value_counts()
        w = pd.Series({x: 1 / ncl / cnt[cl[x]] for x in trial})
        if ((C * w * trend.idm_for(len(trial)) / k[trial]) >= m).all(): chosen = trial
    return chosen

def select(C, m, w0, w1, notional):
    med = full['positions'].loc[w0:w1].abs().median()
    k = notional * SP['weight'] * 2.5 / med
    cols = greedy(list(SP.index[STRICT]), k, C, m)
    dropped = []
    while True:
        s = core.v5(cols, start='2007-01-01')
        nc = s['positions'].loc[w0:w1, cols].abs().median() * C / notional[cols]
        if (nc >= m).all() or len(cols) <= 1: return cols, dropped
        worst = nc.idxmin(); dropped.append(worst); cols = [c for c in cols if c != worst]

ratio = (px.loc[:'2023-07-07'].ffill().iloc[-1] / px.ffill().iloc[-1]).reindex(SP.index)
N_TODAY = SP['liquid_notional_usd']
N_2023 = N_TODAY * ratio
PER = {'2010': '2010-01-01', '2015': '2015-01-01', '2023-07': '2023-07-10'}
rows = []
for C in [100e3, 250e3, 1e6]:
    hind, _ = select(C, 1, '2023-07-10', '2026-07-10', N_TODAY)
    exa, drop = select(C, 1, '2020-07-10', '2023-07-07', N_2023)
    orig = U[('sub', C, 1)]
    for lab, cols in [('audit_subset', orig), ('reproduced_hindsight', hind), ('ex_ante_2023', exa)]:
        n = core.v5(cols, start='2007-01-01')['net']
        r = dict(capital=C, selection=lab, n=len(cols), **{f'SR_{k}': core.sr(n.loc[d:]) for k, d in PER.items()},
                 same_as_audit=set(cols) == set(orig), markets='|'.join(cols))
        rows.append(r)
    print(int(C), 'hindsight==audit:', set(hind) == set(orig), '| ex-ante adds:', sorted(set(exa) - set(orig)), '| ex-ante removes:', sorted(set(orig) - set(exa)), flush=True)
df = pd.DataFrame(rows); df.to_csv(core.OUT + 'subset_selection_exante.csv', index=False, float_format='%.4f')
print(df.drop(columns='markets').round(3).to_string(index=False))
# cross-sectional: does the selection score (median |forecast| 2023-26) predict 2023-26 P&L?
fc = core.FC.loc['2023-07-10':].abs().median()
g = full['inst_gross'].loc['2023-07-10':]
srm = g.mean() / g.std() * np.sqrt(261)
fc_pre = core.FC.loc['2020-07-10':'2023-07-07'].abs().median()
print('corr across markets: median|forecast| 2023-26 vs per-market P&L Sharpe 2023-26 = %.2f (Spearman %.2f); pre-window median|forecast| vs 2023-26 Sharpe = %.2f' % (
    fc.corr(srm), fc.corr(srm, method='spearman'), fc_pre.corr(srm, method='spearman')))
