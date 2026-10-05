"""Sensitivity of critic/mc_tiers.py (central scenarios, 21.4% vol) to the externally verified facts:
 - EUR cash: ECB DFR 2.50% since 16 Sep 2026 -> EuroSTR ~2.43% (critic used 2.30%); 2.68% if the ECB hikes on 29 Oct 2026.
 - Tax regime: PFU 31.4% with 10-y carry (art 150 ter, occasional) vs BNC requalification (art 92-2-5 CGI):
   progressive scale + social levies, losses only vs same-nature BNC profits for 6 years (art 156-I-2 CGI).
   BNC effective marginal rate = TMI + 18.6% - 6.8% x TMI (deductible CSG); TMI 30% -> 46.6%, TMI 41% -> 56.8%.
 - ETF route: accumulating UCITS (iMGP DBi R EUR HP, TER 0.75% inside SR), PFU only at exit.
Same bootstrap (seed 7, block 126, 4000 paths) as the critic, so the deltas are comparable.
"""
import numpy as np, pandas as pd
A = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
net = pd.read_pickle(A + 'operations/v5_net.pkl')['net'].loc['1990':].dropna().values
z = (net - net.mean()) / net.std()
AN, YEARS, NPATH, BLOCK = 261, 10, 4000, 126
rng = np.random.default_rng(7)
N = AN * YEARS; L = len(z)
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

def simulate(sr, vol, fixed, shortfall, cash, tax, carry_years):
    mu = sr * vol - fixed - shortfall
    daily = shocks * vol / np.sqrt(AN) + mu / AN + cash / AN
    pre = np.cumprod(1 + daily, axis=1)
    W = np.ones(NPATH); carry = np.zeros((NPATH, YEARS))
    for y in range(YEARS):
        g = np.cumprod(1 + daily[:, y * AN:(y + 1) * AN], axis=1)
        start = W.copy(); end = start * g[:, -1]
        gain = end - start
        taxable = np.where(gain > 0, gain, 0.0)
        for v in range(max(0, y - carry_years), y):
            use = np.minimum(carry[:, v], taxable); carry[:, v] -= use; taxable -= use
        carry[:, y] = np.where(gain < 0, -gain, 0.0)
        W = end - tax * taxable
    cpre = pre[:, -1] ** (1 / YEARS) - 1; cpost = W ** (1 / YEARS) - 1
    return round(np.median(cpre) * 100, 2), round(np.median(cpost) * 100, 2), round((W < 1).mean() * 100, 1)

tiers = {'100k': (0.20, 0.005, 0.0093), '250k': (0.24, 0.0025, 0.0079), '1M': (0.27, 0.001, 0.0072)}
regimes = {'PFU31.4_carry10': (0.314, 10), 'BNC_TMI30_carry6': (0.466, 6), 'BNC_TMI41_carry6': (0.568, 6)}
rows = []
for cash in (0.023, 0.0243, 0.0268):
    for tier, (sr, fx, sh) in tiers.items():
        for reg, (tax, cy) in regimes.items():
            pre, post, ploss = simulate(sr, 0.214, fx, sh, cash, tax, cy)
            rows.append(dict(cash_eur=cash, tier=tier, regime=reg, sr=sr, pre_tax_median=pre,
                             after_tax_median=post, P_nominal_loss_10y=ploss))
    for sr in (0.10, 0.25, 0.40):
        daily = shocks * 0.12 / np.sqrt(AN) + (sr * 0.12) / AN + cash / AN
        Wt = np.cumprod(1 + daily, axis=1)[:, -1]
        post = np.where(Wt > 1, 1 + (Wt - 1) * (1 - 0.314), Wt)
        rows.append(dict(cash_eur=cash, tier='UCITS_ETF_12vol', regime='PFU_at_exit', sr=sr,
                         pre_tax_median=round((np.median(Wt) ** .1 - 1) * 100, 2),
                         after_tax_median=round((np.median(post) ** .1 - 1) * 100, 2),
                         P_nominal_loss_10y=round((Wt < 1).mean() * 100, 1)))
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
df.to_csv(A + 'cv_fonds_fisc/mc_rate_tax_sensitivity.csv', index=False)
