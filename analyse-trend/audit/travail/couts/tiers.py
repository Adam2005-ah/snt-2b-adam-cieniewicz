import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts')
import run_backtests as rb, specs
OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts/'
D = pickle.load(open(OUT + 'v5_cache.pkl', 'rb')); g = D['groups']
P = pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True).ffill()
T = specs.table(); fx = specs.fx_frame(P)
START = sys.argv[1] if len(sys.argv) > 1 else '2023-07-10'
pos = D['v5']['positions'].loc[START:]
idx = pos.index
P = P.reindex(idx).ffill(); fx = fx.reindex(idx).ffill()
print('min price in window:', P.min()[P.min() <= 0])
fxm = pd.DataFrame({c: fx[T.loc[c, 'ccy']] for c in pos.columns})
rolls = pd.Series({c: rb.ROLLS_PER_YEAR[g[c]] for c in pos.columns})
years = (idx[-1] - idx[0]).days / 365.25
TIERS = [25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6]

def simulate(K, small, hyst):
    mult = pd.Series({c: (T.loc[c, 'mult_s'] if small and pd.notna(T.loc[c, 'mult_s']) else T.loc[c, 'mult']) for c in pos.columns})
    fee = pd.Series({c: (T.loc[c, 'fee_s'] if small and pd.notna(T.loc[c, 'mult_s']) else T.loc[c, 'fee']) for c in pos.columns})
    hs = pd.Series({c: (T.loc[c, 'hs_s'] if small and pd.notna(T.loc[c, 'mult_s']) else T.loc[c, 'hs']) for c in pos.columns})
    notional = P[pos.columns] * mult * fxm                     # USD per contract
    cost_side = (fee + hs * mult) * fxm                         # USD per contract per side
    target = (pos * K / notional).fillna(0)                     # contracts, fractional
    n = np.zeros(len(pos.columns)); held = []
    tv = target.values
    for t in range(len(idx)):
        dev = tv[t] - n
        move = np.abs(dev) > hyst
        n = np.where(move, np.round(tv[t]), n)
        held.append(n.copy())
    N = pd.DataFrame(held, index=idx, columns=pos.columns)
    trades = N.diff().abs().fillna(N.abs())
    roll_ct = N.abs() * rolls * 2 / 252
    fees = ((trades + roll_ct) * fee * fxm).sum().sum() / years
    spread = ((trades + roll_ct) * hs * mult * fxm).sum().sum() / years
    ct_year = (trades + roll_ct).sum().sum() / years
    # tracking: realised exposure vs ideal
    real = N * notional / K
    err = (real - pos).fillna(0)
    rets = D['rets'].reindex(idx)[pos.columns].fillna(0)
    pnl_ideal = (pos.shift(1) * rets).sum(axis=1); pnl_real = (real.shift(1) * rets).sum(axis=1)
    te = (pnl_real - pnl_ideal).std() * 16
    corr = pnl_real.corr(pnl_ideal)
    nz = (N != 0).sum(axis=1).mean()
    live = (pos.abs() > 0).sum(axis=1).mean()
    return {'capital_usd': K, 'contracts': 'micro/mini where available' if small else 'standard', 'hysteresis_ct': hyst,
            'contracts_traded_per_year': ct_year, 'contracts_per_month': ct_year / 12,
            'fees_usd_per_year': fees, 'spread_usd_per_year': spread,
            'fees_%cap': fees / K * 100, 'spread_%cap': spread / K * 100, 'variable_total_%cap': (fees + spread) / K * 100,
            'mkts_nonzero_avg': nz, 'mkts_live_avg': live, 'tracking_error_%': te * 100, 'corr_with_ideal': corr,
            'ideal_ann_return_%': pnl_ideal.mean() * 252 * 100, 'rounded_ann_return_%': pnl_real.mean() * 252 * 100}
rows = []
for K in TIERS:
    for small in [False, True]:
        for h in [0.5, 0.75]:
            rows.append(simulate(K, small, h))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('window', idx[0].date(), idx[-1].date(), 'years %.2f' % years)
print(R.round(2).to_string())
R.to_csv(OUT + f'tier_variable_costs_{START[:4]}.csv', index=False)
