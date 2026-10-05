"""Assemble costs.csv: fixed + variable + cash-interest shortfall per capital tier, plus tax overlay."""
import pandas as pd, numpy as np
OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts/'
TIERS = [25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6]
EURUSD = 1.17   # approx Oct 2026 (assumption; back-adjusted EC1 = 1.1446 on 2026-07-10)
# ---------- rates (Oct 2026) ----------
TBILL = 4.00          # FRED DTB3 2026-10-01
EFFR = 3.86           # assumption: FOMC 16 Sep 2026 range 3.75-4.00
ESTR = 2.42           # assumption: ECB DFR 2.50 from Sep 2026, ESTR ~ DFR - 8bp
IB_USD = EFFR - 0.5   # IBKR: BM - 0.5% on balances > 10k, scaled by NAV/100k (USD) if NAV < 100k
IB_EUR = ESTR - 0.5
def ib_interest(cash, nav, rate):
    return max(cash - 10e3, 0) * rate / 100 * min(nav / 100e3, 1)
# ---------- fixed costs (USD/yr) ----------
FIX = {
 'data_minimal': 0,            # IBKR delayed (15-20 min) data free via API; signals from IBKR daily bars / own EOD data
 'data_lean': 12 * (10 + 9.4 + 1.3 + 7 + 5 + 10 + 6.5),  # US bundle $10 (waived if >= $30 comm./month), Eurex EUR 8, OSE JPY 200, HKFE ~HKD 56, SGX ~5, ASX24 ~10 (asm), MX CAD 9; ICE & Euronext delayed
 'data_full': 12 * (10 + 9.4 + 1.3 + 7 + 5 + 10 + 6.5 + 132 + 254 + 65 + 15),  # + ICE US softs ~$132, ICE Europe comm.+fin. $254, Euronext ~$65 (AMP price), Nasdaq Nordic ~$15 (asm)
 'eod_data_norgate': 270,      # optional, Norgate futures 12 months
 'vps': 12 * 6.0 * EURUSD,     # Hetzner CX23 ~EUR 6/month incl. IPv4 (or home PC)
 'software_pysystemtrade': 0,
 'fx_conversions': 60,         # ~2-3 conversions/month at min USD 2
}
FIX['fixed_lean'] = FIX['data_lean'] + FIX['vps'] + FIX['fx_conversions'] + FIX['eod_data_norgate']
FIX['fixed_full'] = FIX['data_full'] + FIX['vps'] + FIX['fx_conversions'] + FIX['eod_data_norgate']
FIX['fixed_minimal'] = FIX['data_minimal'] + FIX['vps'] + FIX['fx_conversions']
var = pd.read_csv(OUT + 'tier_variable_costs_2023.csv')
var7 = pd.read_csv(OUT + 'tier_variable_costs_2019.csv')
rows = []
for K in TIERS:
    r = {'capital_usd': K, 'capital_eur_approx': round(K / EURUSD)}
    for k in ['fixed_minimal', 'fixed_lean', 'fixed_full']:
        r[k + '_usd_yr'] = round(FIX[k]); r[k + '_%cap'] = FIX[k] / K * 100
    for small, tag in [(False, 'std'), (True, 'micro')]:
        name = 'micro/mini where available' if small else 'standard'
        s = var[(var.capital_usd == K) & (var.contracts == name) & (var.hysteresis_ct == 0.75)].iloc[0]
        s7 = var7[(var7.capital_usd == K) & (var7.contracts == name) & (var7.hysteresis_ct == 0.75)].iloc[0]
        r[f'var_cost_%cap_{tag}_2023-26'] = s['variable_total_%cap']; r[f'var_cost_%cap_{tag}_2019-26'] = s7['variable_total_%cap']
        r[f'fees_%cap_{tag}'] = s['fees_%cap']; r[f'spread_%cap_{tag}'] = s['spread_%cap']
        r[f'contracts_per_month_{tag}'] = s['contracts_per_month']
        r[f'markets_held_avg_{tag}'] = s['mkts_nonzero_avg']; r[f'tracking_error_vs_ideal_%_{tag}'] = s['tracking_error_%']
    # interest: USD investor, all capital as cash at IBKR
    i_all = ib_interest(K, K, IB_USD)
    r['usd_interest_allcash_%'] = i_all / K * 100
    r['usd_shortfall_vs_tbill_allcash_%'] = TBILL - i_all / K * 100
    # optimized: 50% cash at IBKR (margin p95 ~40% + buffer), 50% T-bills (cost ~2bp/yr)
    i_opt = ib_interest(0.5 * K, K, IB_USD) + 0.5 * K * (TBILL - 0.02) / 100
    r['usd_shortfall_vs_tbill_50pct_tbills_%'] = TBILL - i_opt / K * 100
    # EUR investor vs ESTR
    Ke = K / EURUSD
    j_all = ib_interest(Ke, K, IB_EUR)  # NAV test in USD
    r['eur_shortfall_vs_estr_allcash_%'] = ESTR - j_all / Ke * 100
    j_opt = ib_interest(0.5 * Ke, K, IB_EUR) + 0.5 * Ke * (ESTR - 0.02) / 100   # XEON ~ ESTR+8.5bp-10bp TER
    r['eur_shortfall_vs_estr_50pct_xeon_%'] = ESTR - j_opt / Ke * 100
    r['eur_cash_rate_gap_vs_us_tbill_%'] = TBILL - ESTR
    rows.append(r)
C = pd.DataFrame(rows)
BT_COST = 2.31   # backtest model cost for v5 over 2023-07..2026-07 (2.38 over 1990-2026)
for tag in ['std', 'micro']:
    C[f'extra_var_vs_backtest_%_{tag}'] = C[f'var_cost_%cap_{tag}_2023-26'] - BT_COST
EXTRA_VAR_FULL = 0.5   # extra variable cost vs backtest model once the strategy is fully replicated (5M std: +0.34 in 2023-26, +0.65 in 2019-26)
C['extra_var_at_full_replication_%'] = EXTRA_VAR_FULL
C['total_drag_vs_backtest_%_lean_data_50pct_tbills'] = C['fixed_lean_%cap'] + EXTRA_VAR_FULL + C['usd_shortfall_vs_tbill_50pct_tbills_%']
C['total_drag_vs_backtest_%_full_data_allcash'] = C['fixed_full_%cap'] + EXTRA_VAR_FULL + C['usd_shortfall_vs_tbill_allcash_%']
C['replication'] = np.where(C['tracking_error_vs_ideal_%_micro'] > 10, 'NO (tracking error >10%: not the backtested strategy)',
                    np.where(C['tracking_error_vs_ideal_%_micro'] > 4, 'partial (TE 4-10%)', 'good (TE <4%)'))
# tax overlay: PFU 31.4%, annual realisation; after-tax/pre-tax ratio from tax_sim (0.63-0.69 long run)
C['after_tax_factor_long_run'] = 0.66
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(C.round(2).T.to_string())
C.round(4).to_csv(OUT + 'costs.csv', index=False)
pd.Series(FIX).round(0).to_csv(OUT + 'fixed_cost_items_usd.csv', header=['usd_per_year'])
print(pd.Series(FIX).round(0))
print('IB_USD %.2f IB_EUR %.2f' % (IB_USD, IB_EUR))
