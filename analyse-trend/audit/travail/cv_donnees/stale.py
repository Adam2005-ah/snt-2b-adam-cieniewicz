"""Task 1: staleness / artefact diagnostics for all 84 series + v5 P&L contribution and lag sensitivity."""
import numpy as np, pandas as pd
import core
R = core.rets
px = pd.read_csv(core.MKT + '/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
px = px.reindex(columns=R.columns)

# global non-trading days: >= 50% of markets with data show exactly 0
has = R.notna(); zero = (R == 0)
glob_hol = zero.sum(axis=1) >= 0.5 * has.sum(axis=1)

def longest_run(mask):
    m = mask.values.astype(int)
    best = cur = 0; end = None
    for i, v in enumerate(m):
        cur = cur + 1 if v else 0
        if cur > best: best, end = cur, i
    return best, (mask.index[end].date() if end is not None else None)

def ac(x, k=1):
    return x.autocorr(k)

def vr(x, h):
    s = x.rolling(h).sum().iloc[::h].dropna()
    return s.var() / (x.var() * h)

base = core.v5(lag=1.0); lag2 = core.v5(lag=2.0)
rows = []
for c in R.columns:
    s_all = R[c].dropna()
    for lab, s in [('listing', s_all), ('2010', s_all.loc['2010':])]:
        pass
    s10 = s_all.loc['2010':]
    if len(s10) < 500: s10 = s_all
    spec = s10[~glob_hol.reindex(s10.index).fillna(False)]
    p = px[c].dropna(); p = p[p.index >= s_all.index[0]] if len(s_all) else p
    w = s10.clip(s10.quantile(0.01), s10.quantile(0.99))
    run_r, run_r_end = longest_run((s_all == 0))
    run_r10, run_r10_end = longest_run((s10 == 0))
    run_p, run_p_end = longest_run((p.diff() == 0)) if len(p) else (np.nan, None)
    nz = s10[s10 != 0]
    rep = ((s10 == s10.shift(1)) & (s10 != 0)).sum()
    sd = s10.std()
    spikes = ((s10.abs() > 6 * sd) & ((s10 + s10.shift(-1)).abs() < 0.3 * s10.abs())).sum()
    g1 = base['inst_gross'][c].loc['2010':]; g2 = lag2['inst_gross'][c].loc['2010':]
    g1_23 = base['inst_gross'][c].loc['2023-07-10':]; g2_23 = lag2['inst_gross'][c].loc['2023-07-10':]
    rows.append(dict(mkt=c, name=core.names[c], group=core.groups[c], start=s_all.index[0].date() if len(s_all) else None,
        n_2010=len(s10), zero_share_2010=(s10 == 0).mean(), zero_share_specific_2010=(spec == 0).mean(),
        zero_share_listing=(s_all == 0).mean(),
        longest_zero_run_listing=run_r, longest_zero_run_end=run_r_end, longest_zero_run_2010=run_r10, longest_zero_run_2010_end=run_r10_end,
        longest_flat_price_run=run_p, flat_price_end=run_p_end,
        repeated_nonzero_returns_2010=int(rep),
        ac1_2010=ac(s10), ac1_t_2010=ac(s10) * np.sqrt(len(s10)), ac1_winsor_2010=ac(w), ac1_nonzero_2010=ac(nz),
        ac2_2010=ac(s10, 2), ac1_listing=ac(s_all), ac1_2015=ac(s_all.loc['2015':]), ac1_2023=ac(s_all.loc['2023-07-10':]),
        vr5_2010=vr(s10, 5), vr20_2010=vr(s10, 20), kurt_2010=s10.kurt(), spike_reversals_2010=int(spikes),
        vol_2010=sd * np.sqrt(261),
        gross_pnl_2010_pct=g1.mean() * 261 * 100, gross_pnl_2010_sr=g1.mean() / g1.std() * np.sqrt(261) if g1.std() > 0 else np.nan,
        gross_pnl_2010_lag2_pct=g2.mean() * 261 * 100, lag_cost_2010_pct=(g1.mean() - g2.mean()) * 261 * 100,
        gross_pnl_2023_pct=g1_23.mean() * 261 * 100, lag_cost_2023_pct=(g1_23.mean() - g2_23.mean()) * 261 * 100,
        gross_pnl_1990_pct=base['inst_gross'][c].loc['1990':].mean() * 261 * 100,
        lag_cost_1990_pct=(base['inst_gross'][c].loc['1990':].mean() - lag2['inst_gross'][c].loc['1990':].mean()) * 261 * 100,
        avg_abs_pos_2010=base['held'][c].loc['2010':].abs().mean()))
df = pd.DataFrame(rows).set_index('mkt')
tot = base['gross'].loc['2010':].mean() * 261 * 100
df['share_of_total_gross_2010'] = df['gross_pnl_2010_pct'] / tot
df.to_csv(core.OUT + 'stale_scan_full.csv', float_format='%.5g')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
cols = ['group', 'zero_share_2010', 'zero_share_specific_2010', 'longest_zero_run_listing', 'longest_flat_price_run', 'repeated_nonzero_returns_2010',
        'ac1_2010', 'ac1_t_2010', 'ac1_winsor_2010', 'ac1_listing', 'vr5_2010', 'gross_pnl_2010_pct', 'gross_pnl_2010_sr', 'lag_cost_2010_pct']
print('total gross since 2010 %.2f %%/yr' % tot)
print(df.sort_values('ac1_2010', ascending=False)[cols].round(3).to_string())
