"""Path-dependent check: same integer engine as audit/simulation/sim.py (Carver buffer, today's notional, realistic
fee+half-spread), but contracts are sized on CURRENT equity (start capital C0, P&L reinvested, losses shrink size),
instead of a fixed capital. Compares with the fixed-capital result and its compounded 'CAGR'."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation')
import sim
E = sim.E
subs = pickle.load(open(sim.OUT + 'subsets.pkl', 'rb'))
tab, N, HS = sim.TAB['liquid_today']
def run_equity(sysout, cols, C0, start='2000-01-01', end='2026-07-10'):
    idx = sim.IDX[(sim.IDX >= pd.Timestamp(start)) & (sim.IDX <= pd.Timestamp(end))]
    tgt = sysout['target'].reindex(idx)[cols].values; unit = sysout['unit'].reindex(idx)[cols].values
    mask = sysout['mask'].reindex(idx).values
    r = E.RETS.reindex(idx)[cols].fillna(0).values
    Nn = N.reindex(idx)[cols].values; hs = HS.reindex(idx)[cols].values
    fee = tab.loc[cols, 'fee_spec'].values; rolls = E.ROLLS_YR[cols].values
    eq = C0; cur = np.zeros(len(cols)); eqs = []; held0 = 0; tot = 0
    for i in range(len(idx)):
        # P&L of positions held from previous day
        pnl = np.nansum(cur * Nn[i] * r[i])
        eq += pnl
        # decide new positions at close i on current equity
        x = tgt[i] * eq / Nn[i]; ok = ~np.isnan(x)
        B = 0.1 * np.abs(unit[i]) * eq / Nn[i]
        new = cur.copy()
        if mask[i]:
            lo = np.round(x - B); hi = np.round(x + B)
            new = np.where(ok, np.minimum(np.maximum(cur, lo), hi), 0.0)
        else:
            new = np.where(ok, cur, 0.0)
        sides = np.abs(new - cur) + np.abs(new) * 2 * rolls / 252
        eq -= np.nansum(sides * (fee + hs[i]))
        tot += np.isnan(x).sum() < len(cols) and (new[ok] == 0).sum()
        held0 += ((new == 0) & ok & (np.abs(np.nan_to_num(tgt[i])) > 0)).sum()
        cur = new; eqs.append(eq)
        if eq <= 0: break
    e = pd.Series(eqs, index=idx[:len(eqs)])
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    ret = e.pct_change().dropna()
    return dict(end_equity=round(e.iloc[-1]), cagr=round((e.iloc[-1] / C0) ** (1 / yrs) - 1, 4), maxdd=round((e / e.cummax() - 1).min(), 3),
                min_equity=round(e.min()), sharpe=round(ret.mean() / ret.std() * 16, 3))
rows = []
STRICT = list(sim.SP.index[sim.SP['liquid_access'].eq('OK')])
for C, lab in [(50e3, 'sub1'), (100e3, 'sub1'), (250e3, 'sub1'), (1e6, 'sub1')]:
    cols = subs[('sub', C, 1)]; s = E.system(cols)
    fixed, _, _ = sim.evaluate('fixed', cols, C, s, 'liquid_today')
    f = [x for x in fixed if x['period'] == '2000-2026'][0]
    comp = run_equity(s, cols, C)
    rows.append(dict(capital=C, universe=f'subset>=1c ({len(cols)})', fixed_sharpe=f['sharpe_net_real'], fixed_cagr_compounded=f['cagr_excess_net_real'],
                     fixed_maxdd=f['maxdd_excess_net_real'], **{f'equity_{k}': v for k, v in comp.items()}))
s66 = E.system(STRICT)
for C in [100e3, 1e6]:
    fixed, _, _ = sim.evaluate('fixed', STRICT, C, s66, 'liquid_today')
    f = [x for x in fixed if x['period'] == '2000-2026'][0]
    comp = run_equity(s66, STRICT, C)
    rows.append(dict(capital=C, universe='strict66', fixed_sharpe=f['sharpe_net_real'], fixed_cagr_compounded=f['cagr_excess_net_real'],
                     fixed_maxdd=f['maxdd_excess_net_real'], **{f'equity_{k}': v for k, v in comp.items()}))
out = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(out.to_string()); out.to_csv('compounding_check.csv', index=False)
