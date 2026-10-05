import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend'); sys.path.insert(0, '.')
import run_backtests as rb, specs
D = pickle.load(open('v5_cache.pkl', 'rb')); g = D['groups']
P = pd.read_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True).ffill()
T = specs.table(); fx = specs.fx_frame(P)
pos = D['v5']['positions'].loc['2023-07-10':]; idx = pos.index
P = P.reindex(idx); fx = fx.reindex(idx)
fxm = pd.DataFrame({c: fx[T.loc[c, 'ccy']] for c in pos.columns})
notional = P[pos.columns] * T.mult * fxm
bp = (T.fee + T.hs * T.mult) * fxm / notional            # cost per side as fraction of notional, time-varying
rolls = pd.Series({c: rb.ROLLS_PER_YEAR[g[c]] for c in pos.columns})
dpos = pos.diff().abs(); held = pos.abs()
real = (dpos * bp).mean() * 252 * 100 + (held * bp * 2 * rolls / 252).mean() * 252 * 100
btbp = pd.Series({c: rb.COSTS[g[c]] for c in pos.columns})
bt = (dpos * btbp).mean() * 252 * 100 + (held * btbp * 2 * rolls / 252).mean() * 252 * 100
df = pd.DataFrame({'group': g, 'real_%': real, 'bt_%': bt, 'turnover': dpos.mean()*252, 'abs_pos': held.mean(), 'bp_avg': bp.mean()*1e4})
print(df.sort_values('real_%', ascending=False).head(20).round(3))
print(df.groupby('group')[['real_%', 'bt_%']].sum().round(3)); print(df[['real_%', 'bt_%']].sum().round(3))
