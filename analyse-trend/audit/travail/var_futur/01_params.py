"""Forward-Sharpe rule and per-key simulation parameters for the 5 variants x 3 capital tiers.

Anchor A(v, tier) = realistic retail Sharpe since 2010 (half-day lag, whole contracts, real IBKR fee + half-spread,
excess over cash, before cash leg and fixed costs, x16 convention) = series_variantes_stats.csv 'sharpe_2010'.
Common haircut h(tier) = v5_central(tier) / A(v5, tier), with v5_central = 0.20 / 0.24 / 0.27 (audit).
  Case 1 'commune'       : SR_central = h * A for every variant.
  Case 2 'v1v2_sans_selection': v1 and v2 (not chosen after seeing the results) only get the decay part of the haircut,
                           decay = 75 % of the variant's own haircut in Sharpe units, (1-h)*A
                           -> SR_central = A - 0.75 (1-h) A = A (0.25 + 0.75 h); v3..v5 unchanged.
  pess / opt = central -/+ 0.20.
Vol = series' realised vol since 2000 (x16).
Cash at IBKR: margin share m scaled by the variant's published (84-market, 1990+) mean gross exposure, v5 = 17 %;
  cash kept at IBKR c = m + 0.23 * vol / vol(v5, tier)  (v5: 40 % / 17 %, the audit's policy; the 23 % buffer over
  margin scales with the account's volatility). Sensitivity: m = mean margin computed on each key's contracts 2000-2026.
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
ST = pd.read_csv(os.path.join(AUD, 'var_passe_retail', 'series_variantes_stats.csv'), index_col=0)

TIERS = {'100k': 100e3, '250k': 250e3, '1M': 1e6}
V5C = {'100k': 0.20, '250k': 0.24, '1M': 0.27}
GE84 = {'v1': 5.2556, 'v2': 5.5956, 'v3': 6.7733, 'v4': 7.7788, 'v5': 8.7837}   # published 84 mkts, 1990+ (ref84.pkl)
NAMES = {'v1': '1. Montant fixe', 'v2': '2. Système actuel', 'v3': '3. + stratégie 13', 'v4': '4. + pilotage du risque',
         'v5': '5. Les deux'}
DECAY_SHARE = 0.75
M_V5, C_V5 = 0.17, 0.40
rows = []
for tier, K0 in TIERS.items():
    a5 = ST.loc['v5_' + tier, 'sharpe_2010']
    h = V5C[tier] / a5
    vol5 = ST.loc['v5_' + tier, 'vol_realised_2000']
    for v in ('v1', 'v2', 'v3', 'v4', 'v5'):
        key = '%s_%s' % (v, tier)
        A = ST.loc[key, 'sharpe_2010']
        vol = ST.loc[key, 'vol_realised_2000']
        m = M_V5 * GE84[v] / GE84['v5']
        buf = (C_V5 - M_V5) * vol / vol5
        m_ct = ST.loc[key, 'mean_margin_2000']
        sr1 = h * A
        sr2 = A * (1 - DECAY_SHARE * (1 - h)) if v in ('v1', 'v2') else sr1
        rows.append(dict(key=key, variant=v, variant_name=NAMES[v], tier=tier, K0=K0, A_sharpe_2010=A, h_tier=h,
                         sr_central_case1=sr1, sr_central_case2=sr2, vol=vol, m=m, c=min(m + buf, 1.0),
                         m_contracts=m_ct, c_contracts=min(m_ct + buf, 1.0),
                         sharpe_2000=ST.loc[key, 'sharpe_2000'], n_markets=int(ST.loc[key, 'n_markets'])))
P = pd.DataFrame(rows).set_index('key')
P.to_csv(os.path.join(HERE, 'params.csv'))
if __name__ == '__main__':
    pd.set_option('display.width', 250)
    print(P.drop(columns=['variant_name']).round(4).to_string())
