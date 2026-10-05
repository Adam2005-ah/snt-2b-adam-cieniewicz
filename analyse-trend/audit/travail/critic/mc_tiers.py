"""10-year Monte Carlo of a retail implementation of variant 5 for a French (EUR) investor.

Daily excess-return shapes come from the v5 net series 1990-2026 (de-meaned, unit-variance),
resampled with a stationary bootstrap (mean block 126 days) to keep part of the medium-horizon
autocorrelation. Each tier gets: forward Sharpe after variable costs and execution lag (scenario),
vol target, fixed costs and cash shortfall (% capital/yr), EUR cash rate. Tax: PFU 31.4 % on the
year's net gain (futures P&L + interest treated as realised each year), losses carried forward
10 years against later gains.
"""
import numpy as np
import pandas as pd

A = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
net = pd.read_pickle(A + 'operations/v5_net.pkl')['net'].loc['1990':].dropna().values
z = (net - net.mean()) / net.std()
AN, YEARS, NPATH, BLOCK = 261, 10, 4000, 126
PFU, ESTR = 0.314, 0.023
rng = np.random.default_rng(7)
N = AN * YEARS
L = len(z)

# shared bootstrap index paths (same shocks for every tier/scenario -> comparable)
idx = np.empty((NPATH, N), dtype=np.int64)
for p in range(NPATH):
    i = rng.integers(L)
    for t in range(N):
        if t > 0 and rng.random() < 1 / BLOCK:
            i = rng.integers(L)
        else:
            i = (i + 1) % L if t > 0 else i
        idx[p, t] = i
shocks = z[idx]


def simulate(sr, vol, fixed, shortfall, cash=ESTR):
    mu_excess = sr * vol - fixed - shortfall          # arithmetic excess over EUR cash, after frictions
    daily = shocks * vol / np.sqrt(AN) + mu_excess / AN + cash / AN
    pre = np.cumprod(1 + daily, axis=1)
    # after-tax path, yearly tax with 10y carry-forward
    W = np.ones(NPATH)
    carry = np.zeros((NPATH, YEARS))  # losses by vintage
    wealth_daily = []
    for y in range(YEARS):
        g = np.cumprod(1 + daily[:, y * AN:(y + 1) * AN], axis=1)
        start = W.copy()
        path = start[:, None] * g
        wealth_daily.append(path)
        end = path[:, -1]
        gain = end - start
        taxable = np.where(gain > 0, gain, 0.0)
        # use carried losses oldest first (all within 10 y here)
        for v in range(y):
            use = np.minimum(carry[:, v], taxable)
            carry[:, v] -= use
            taxable -= use
        carry[:, y] = np.where(gain < 0, -gain, 0.0)
        W = end - PFU * taxable
    after = np.concatenate(wealth_daily, axis=1)
    cagr_pre = pre[:, -1] ** (1 / YEARS) - 1
    cagr_post = W ** (1 / YEARS) - 1
    mdd = (pre / np.maximum.accumulate(pre, axis=1) - 1).min(axis=1)
    cash_after_tax = cash * (1 - PFU)
    return {
        'sr': sr, 'vol': vol, 'mu_excess_arith': round(mu_excess * 100, 2),
        'pre_tax_median': round(np.median(cagr_pre) * 100, 2),
        'pre_p10': round(np.percentile(cagr_pre, 10) * 100, 2), 'pre_p90': round(np.percentile(cagr_pre, 90) * 100, 2),
        'after_tax_median': round(np.median(cagr_post) * 100, 2),
        'after_p10': round(np.percentile(cagr_post, 10) * 100, 2), 'after_p90': round(np.percentile(cagr_post, 90) * 100, 2),
        'P_nominal_loss_10y': round((W < 1).mean() * 100, 1),
        'P_below_cash_after_tax': round((cagr_post < cash_after_tax).mean() * 100, 1),
        'maxDD_median': round(np.median(mdd) * 100, 1), 'maxDD_p10': round(np.percentile(mdd, 10) * 100, 1),
    }


# tier inputs: forward SR (after variable costs & execution lag) pess/central/opt;
# fixed costs (% cap/yr, lean-minimal data, VPS, Norgate); EUR cash shortfall (verif_couts, 50 % XEON)
tiers = {
    '100k (14-mkt micro subset)': dict(srs=(0.0, 0.20, 0.40), fixed=0.005, short=0.0093),
    '250k (24-mkt subset)': dict(srs=(0.03, 0.24, 0.43), fixed=0.0025, short=0.0079),
    '1M (40-mkt subset / strict66)': dict(srs=(0.05, 0.27, 0.47), fixed=0.001, short=0.0072),
}
rows = []
for name, t in tiers.items():
    for lab, s in zip(('pess', 'central', 'opt'), t['srs']):
        for vol in (0.214, 0.12):
            r = simulate(s, vol, t['fixed'], t['short'])
            r.update({'tier': name, 'scenario': lab})
            rows.append(r)
# reference: UCITS trend ETF (iMGP DBi EUR-hedged), TER inside SR, vol 12 %, accumulating -> tax at exit
df = pd.DataFrame(rows)
cols = ['tier', 'scenario', 'vol', 'sr', 'mu_excess_arith', 'pre_tax_median', 'pre_p10', 'pre_p90',
        'after_tax_median', 'after_p10', 'after_p90', 'P_nominal_loss_10y', 'P_below_cash_after_tax',
        'maxDD_median', 'maxDD_p10']
df = df[cols]
pd.set_option('display.width', 250)
print(df.to_string(index=False))
df.to_csv(A + 'critic/mc_tiers.csv', index=False)

# ETF reference with deferral: pre-tax path, PFU only on final gain
for sr in (0.10, 0.25, 0.40):
    vol = 0.12
    daily = shocks * vol / np.sqrt(AN) + (sr * vol) / AN + ESTR / AN
    W = np.cumprod(1 + daily, axis=1)[:, -1]
    post = np.where(W > 1, 1 + (W - 1) * (1 - PFU), W)
    print('ETF SR %.2f vol 12%%: pre-tax median %.2f%%, after-tax (exit) median %.2f%%, p10 %.2f%%' % (
        sr, (np.median(W) ** 0.1 - 1) * 100, (np.median(post) ** 0.1 - 1) * 100, (np.percentile(post, 10) ** 0.1 - 1) * 100))
