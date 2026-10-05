"""Independent recomputation of the ib_gateway budget (verifier). Before tax; no tax content."""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
E, J, VAT = 1.1446, 0.006216, 1.20
cx23 = (5.49 + .50) * VAT * 12 * E; cx33 = (8.49 + .50) * VAT * 12 * E
cpx32 = (35.49 + .50) * VAT * 12 * E           # fallback if Hetzner CX stays unavailable
eurex_lo, eurex_hi = 8.00 * 12 * E, 10.75 * 12 * E
euronext = 3 * 12 * E; ose_lo, ose_hi = 0.0, 200 * J * 12
ice_amp = {'US': 148, 'EU_fin': 139, 'EU_com': 161}; ice_dir = {'US': 132, 'EU_fin': 123, 'EU_com': 161}
VAR = {1e5: 3.303, 2.5e5: 3.650, 1e6: 2.870}; CASH = {1e5: 0.79, 2.5e5: 0.67, 1e6: 0.61}
rows = []
for cap in VAR:
    miss_lo, miss_hi = (10, 60) if cap == 1e5 else (0, 0)
    std = round(cx33) + miss_lo + round(eurex_hi) + round(euronext) + round(ose_hi) + 50
    std_robust = std + (270 if cap >= 2.5e5 else 0)
    ice_keys = [] if cap < 2.5e5 else (['US', 'EU_fin'] if cap < 1e6 else ['US', 'EU_fin', 'EU_com'])
    ice_hi = sum(ice_amp[k] for k in ice_keys) * 12; ice_lo = sum(ice_dir[k] for k in ice_keys) * 12
    comf_hi = round(cx33) + round(cx23) + 2 * miss_lo + round(eurex_hi) + round(euronext) + round(ose_hi) + ice_hi + 270 + 100
    comf_lo = comf_hi - ice_hi + ice_lo
    rows.append(dict(capital=int(cap), minimal=round(cx23) + miss_lo + 25, standard_central=std, standard_robust_norgate=std_robust,
                     comfortable_lo=comf_lo, comfortable_hi=comf_hi,
                     std_pct=round(std / cap * 100, 3), std_robust_pct=round(std_robust / cap * 100, 3),
                     total_std_pct=round(VAR[cap] + std / cap * 100 + CASH[cap], 2),
                     total_robust_pct=round(VAR[cap] + std_robust / cap * 100 + CASH[cap], 2),
                     not_in_sharpe_std=round(std / cap * 100 + CASH[cap], 2),
                     not_in_sharpe_robust=round(std_robust / cap * 100 + CASH[cap], 2)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, 'recomputed_budget.csv'), index=False)
print(out.to_string(index=False))
print('server lines USD/yr: CX23 %.0f, CX33 %.0f, CPX32 fallback %.0f; Eurex %.0f-%.0f; Euronext %.0f; OSE %.0f-%.0f' % (cx23, cx33, cpx32, eurex_lo, eurex_hi, euronext, ose_lo, ose_hi))
