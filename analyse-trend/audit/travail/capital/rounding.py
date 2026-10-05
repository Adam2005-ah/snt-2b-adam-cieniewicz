"""(1) Integer-contract tracking of variant 5 over 2023-07-10 -> 2026-07-10 at each capital tier (full 84 markets, and
the 'strict' retail subsets from tiers.py). (2) Unrounded back-tests (1990-2026) of those subsets. (3) Margin estimate."""
import sys, pickle, pandas as pd, numpy as np
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import trend, run_backtests as rb
D = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
sp = pd.read_csv('specs.csv', index_col=0); tiers = pd.read_csv('tiers.csv')
d = pickle.load(open('v5.pkl', 'rb')); rets, groups, vol = d['rets'], d['groups'], d['vol']
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
px = pd.read_csv(D+'mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True).ffill()
fxcol = {'EUR': ('EC1 Curncy', 1), 'GBP': ('BP1 Curncy', 100), 'CHF': ('SF1 Curncy', 100), 'CAD': ('CD1 Curncy', 100),
         'JPY': ('JY1 Curncy', 1e4), 'AUD': ('AD1 Curncy', 100), 'SEK': ('SE1 Curncy', 100)}
def fx_series(ccy):
    if ccy == 'USD': return pd.Series(1.0, index=px.index)
    if ccy == 'HKD': return pd.Series(1/7.8, index=px.index)
    c, k = fxcol[ccy]; return px[c] / k
ROLLS = {'Equities': 4, 'Bonds': 4, 'STIR': 4, 'FX': 4, 'Energy': 12, 'Metals': 6, 'Agriculture': 5}
START, END = '2023-07-10', '2026-07-10'
def rounded_track(pos, rets_, C, multcol, h=0.25):
    pos = pos.loc[START:END]; r = rets_.reindex(pos.index).fillna(0)
    N = pd.DataFrame({t: float(sp.loc[t, multcol]) * px[t].reindex(pos.index) * fx_series(sp.loc[t, 'ccy']).reindex(pos.index)
                      for t in pos.columns})
    x = (pos * C / N).values; n = np.zeros_like(x); prev = np.zeros(x.shape[1])
    for i in range(x.shape[0]):           # hysteresis: trade only if ideal is > 0.75 contract away, then round
        move = np.abs(x[i] - prev) > 0.5 + h
        prev = np.where(move, np.round(x[i]), prev); n[i] = prev
    n = pd.DataFrame(n, index=pos.index, columns=pos.columns)
    frac = n * N / C
    ideal = (pos.shift(1) * r).sum(axis=1).iloc[1:]; real = (frac.shift(1) * r).sum(axis=1).iloc[1:]
    te = (real - ideal).std() * 16; vol_i = ideal.std() * 16
    yrs = len(pos) / 252
    roll_sides = (n.abs().mean() * pd.Series({t: ROLLS[groups[t]] for t in pos.columns}) * 2).sum()
    return dict(corr=round(ideal.corr(real), 3), te_ann=round(te, 4), vol_ideal=round(vol_i, 4), vol_rounded=round(real.std()*16, 4),
                te_over_vol=round(te / vol_i, 3), share_nonzero_ideal_rounded_to_zero=round(((n == 0) & (pos.abs() > 0)).values.mean(), 3),
                avg_open_contracts=round(n.abs().sum(axis=1).mean(), 1),
                contract_sides_traded_per_year=round(n.diff().abs().sum().sum() / yrs),
                roll_contract_sides_per_year=round(roll_sides),
                gross_excess_ret_ideal=round(ideal.mean()*252, 4), gross_excess_ret_rounded=round(real.mean()*252, 4))
def margin_est(pos, k=0.35):
    """Rough initial margin / equity ~ k x sum |position| x annual vol (SPAN ~ 3-4 daily sigma; k=0.35 calibrated loosely
    on ES ~5 % and ZN ~1.5 % of notional)."""
    m = (pos.abs() * vol[pos.columns].shift(0)).sum(axis=1) * k
    m = m.loc['1990':]
    return round(m.median(), 3), round(m.quantile(0.95), 3), round(m.max(), 3), round(m.loc[START:].median(), 3)
out, bt = [], []
pos5 = d['v5_pos']
for C in tiers['capital_usd']:
    for mc, lab in [('small_mult', 'full84_smallest_contract'), ('std_mult', 'full84_standard_contract')]:
        r = {'capital_usd': C, 'portfolio': lab}; r.update(rounded_track(pos5, rets, C, mc)); out.append(r)
costs = {c: rb.COSTS[g] for c, g in groups.items()}; roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
def stats(net):
    s = {}
    for a in ['1990-01-01', '2010-01-01', '2023-07-10']:
        x = net.loc[a:'2026-07-10']; x = x[x.index >= x.ne(0).idxmax()]
        s[f'sharpe_since_{a[:4]}'] = round(x.mean() / x.std() * 16, 2)
        if a == '1990-01-01':
            s['first_day'] = x.index[0].date(); s['vol'] = round(x.std() * 16, 3); s['excess_ret_ann'] = round(x.mean() * 252, 3)
            cum = (1 + x).cumprod(); s['maxdd_excess'] = round((cum / cum.cummax() - 1).min(), 3)
    return s
mm = margin_est(pos5)
bt.append({'portfolio': 'full84 (variant 5)', 'n': 84, **stats(d['v5_net']),
           'mean_gross_exposure': round(pos5.abs().sum(axis=1).loc['1990':].mean(), 2),
           'margin_med': mm[0], 'margin_p95': mm[1], 'margin_max': mm[2], 'margin_med_3y': mm[3]})
seen = {}
for _, t in tiers.iterrows():
    if t.capital_usd > 5e6: continue
    for m in (4, 1):
        mk = tuple(t[f'strict_subset_ge{m}c_markets'].split(' | '))
        if mk not in seen:
            sub = list(mk)
            res = trend.run_vol_targeted(rets[sub], risk[sub], {k: costs[k] for k in sub}, {k: roll[k] for k in sub},
                                         forecast_fn=trend.forecast_regime)
            seen[mk] = res
            mm = margin_est(res['positions'])
            bt.append({'portfolio': f'strict_ge{m}c @ {int(t.capital_usd/1e3)}k', 'n': len(sub), **stats(res['net']),
                       'mean_gross_exposure': round(res['positions'].abs().sum(axis=1).loc['1990':].mean(), 2),
                       'margin_med': mm[0], 'margin_p95': mm[1], 'margin_max': mm[2], 'margin_med_3y': mm[3],
                       'markets': ' | '.join(sub)})
        r = {'capital_usd': t.capital_usd, 'portfolio': f'strict_ge{m}c ({len(mk)} mkts)'}
        r.update(rounded_track(seen[mk]['positions'], rets[list(mk)], t.capital_usd, 'liquid_mult')); out.append(r)
rt = pd.DataFrame(out); bt = pd.DataFrame(bt)
rt.to_csv('rounding_tracking.csv', index=False); bt.to_csv('subset_backtests.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
print(rt.to_string()); print(bt.drop(columns=['markets']).to_string())
