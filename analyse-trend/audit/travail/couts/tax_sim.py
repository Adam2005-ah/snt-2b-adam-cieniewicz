"""After-tax simulation of variant 5 for a French resident (PFU 31.4%), calendar-year taxation of futures P&L
(realised at each roll/close -> assume all P&L of the year realised), losses carried forward 10 years against
same-nature gains; interest taxed separately at 31.4% (cannot absorb futures losses). Tax paid at year end."""
import pickle, pandas as pd, numpy as np
OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts/'
D = pickle.load(open(OUT + 'v5_cache.pkl', 'rb'))
tb = pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/us-etf/alm0421_macro/DTB3.csv', index_col=0, parse_dates=True).iloc[:, 0]
tb = pd.to_numeric(tb, errors='coerce').ffill()
for p in [('1990', '2026'), ('2000', '2026'), ('2010', '2026'), ('2020', '2026')]:
    print('T-bill avg %s-%s: %.2f%%' % (p[0], p[1], tb.loc[p[0]:p[1]].mean()))
def run(excess, cash_rate, extra_drag=0.0, tax=0.314, years=None):
    """excess: daily excess return (after trading costs); cash_rate: annual % series; extra_drag: annual fraction
    (fixed costs + cash shortfall + extra trading costs) deducted daily."""
    idx = excess.index
    cr = (cash_rate.reindex(cash_rate.index.union(idx)).ffill().reindex(idx) / 100 / 252)
    wealth_pre, wealth_post, carry = 1.0, 1.0, []  # carry: list of [year, amount] losses
    out = []
    for y, ex in excess.groupby(excess.index.year):
        c = cr.loc[ex.index]
        # pre-tax compounding
        g_pre = np.prod(1 + ex.values + c.values - extra_drag / 252)
        # post-tax: P&L amounts within the year on post-tax wealth
        w = wealth_post; fut_pnl = 0.0; int_pnl = 0.0
        for e, ci in zip(ex.values, c.values):
            fut = w * (e - extra_drag / 252); it = w * ci
            fut_pnl += fut; int_pnl += it; w += fut + it
        # loss carry-forward (10 years)
        carry = [[yy, a] for yy, a in carry if y - yy <= 10]
        taxable = fut_pnl
        if taxable > 0:
            for item in carry:
                use = min(item[1], taxable); item[1] -= use; taxable -= use
            carry = [i for i in carry if i[1] > 1e-12]
            t_fut = tax * taxable
        else:
            carry.append([y, -taxable]); t_fut = 0.0
        t_int = tax * max(int_pnl, 0)
        w -= t_fut + t_int
        out.append({'year': y, 'pre_tax_ret_%': (g_pre - 1) * 100, 'post_tax_ret_%': (w / wealth_post - 1) * 100,
                    'tax_paid_%start': (t_fut + t_int) / wealth_post * 100, 'loss_cf_%': sum(a for _, a in carry) / w * 100})
        wealth_pre *= g_pre; wealth_post = w
    df = pd.DataFrame(out).set_index('year')
    n = len(idx) / 252
    credit = tax * sum(a for _, a in carry)
    run.last_credit = credit / wealth_post
    run.post_with_credit = (wealth_post + credit) ** (1 / n) - 1
    return df, wealth_pre ** (1 / n) - 1, wealth_post ** (1 / n) - 1
ex = D['v5']['net'].loc['1990':'2026-07-10'].fillna(0)
rows = []
for start in ['1990', '2000', '2010', '2016']:
    for drag_name, drag in [('backtest costs only', 0.0), ('+0.5%/yr extra (large acct: extra trading 0.4 + cash 0.1)', 0.005), ('+1.5%/yr extra (250k acct)', 0.015)]:
        df, cpre, cpost = run(ex.loc[start:], tb, drag)
        rows.append({'start': start, 'scenario': drag_name, 'pre_tax_CAGR_%': cpre * 100, 'after_tax_CAGR_%': cpost * 100,
                     'tax_drag_pts': (cpre - cpost) * 100, 'after_tax_CAGR_if_unused_losses_credited_%': run.post_with_credit * 100, 'unused_loss_cf_%wealth': run.last_credit/0.314*100, 'ratio_after/pre': cpost / cpre if cpre > 0 else np.nan})
R = pd.DataFrame(rows); print(R.round(2).to_string())
R.to_csv(OUT + 'after_tax_backtest.csv', index=False)
df, a, b = run(ex.loc['2010':], tb, 0.005); print(df.round(1).tail(17))
# Generic: scaled strategy with lower expected Sharpe (future): excess scaled so that mean excess = target, same vol
rows = []
vol = ex.std() * 16
for target_ex in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
    shifted = ex - ex.mean() + target_ex / 252
    for cash_name, cash in [('EUR cash 2.4%', pd.Series(2.4, index=ex.index)), ('USD T-bill 4.0%', pd.Series(4.0, index=ex.index))]:
        df, cpre, cpost = run(shifted, cash, 0.0)
        rows.append({'mean_excess_after_costs_%': target_ex * 100, 'cash': cash_name, 'vol_%': vol * 100,
                     'pre_tax_CAGR_%': cpre * 100, 'after_tax_CAGR_%': cpost * 100, 'after_tax_if_losses_credited_%': run.post_with_credit * 100})
G = pd.DataFrame(rows); print(G.round(2).to_string()); G.to_csv(OUT + 'after_tax_generic.csv', index=False)
