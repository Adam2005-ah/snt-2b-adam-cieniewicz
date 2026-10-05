"""Build the daily return series used by the cross-check Monte Carlo.
- v5 full 84 (operations/v5_net.pkl, = trend.run_vol_targeted, verified bit-identical by two agents)
- v5 'own fractional' subsets for the >=1-contract retail tiers (simulation/subsets.pkl) via simulation/engine.system
- ES1, NQ1 (USD excess), EC1 (EURUSD futures) -> EUR-investor unhedged equity shocks
- SG CTA DBi replica (iMGP DBi proxy) and SG Trend index, converted to excess of the 3m T-bill
"""
import sys, pickle, numpy as np, pandas as pd
S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/cv_rendement/'
sys.path.insert(0, S + 'audit/simulation')
import engine as E

v5 = pd.read_pickle(S + 'audit/operations/v5_net.pkl')['net']
full = E.system(list(E.RETS.columns))['net']
print('check engine vs operations pickle, max abs diff:', (full - v5).abs().max())
subs = pickle.load(open(S + 'audit/simulation/subsets.pkl', 'rb'))
ser = {'v5_84': v5}
for cap in [50000.0, 100000.0, 250000.0, 500000.0, 1000000.0]:
    cols = subs[('sub', cap, 1)]
    ser['sub_%dk' % int(cap / 1000)] = E.system(cols)['net']
    print(cap, len(cols))
# ex-ethanol / iron ore full system
cols = [c for c in E.RETS.columns if c not in ('CUA1 Comdty', 'SCO1 Comdty')]
ser['v5_82_exCUA_SCO'] = E.system(cols)['net']
R = E.RETS
for c in ['ES1 Index', 'NQ1 Index', 'EC1 Curncy']:
    ser[c] = R[c]
tb = pd.read_csv(S + 'mkt/us-etf/alm0421_macro/DTB3.csv', index_col=0, parse_dates=True)['value'] / 100
for f, k in [('SG_CTA_DBi_replica_daily', 'dbi_rep'), ('SG_Trend_Index_daily', 'sgtrend'), ('SG_CTA_Index_daily', 'sgcta')]:
    p = pd.read_csv(S + 'mkt/alternatives/pofo_indices/%s.csv' % f, index_col=0, parse_dates=True, comment='#')['close']
    r = p.pct_change().dropna()
    rf = tb.reindex(r.index).ffill()
    dt = r.index.to_series().diff().dt.days.fillna(1)
    ser[k + '_ex'] = r - rf * dt / 365
D = pd.DataFrame(ser)
D.to_pickle(OUT + 'series.pkl')
def st(x, a, b=None):
    x = x.loc[a:b].dropna(); return round(x.mean() / x.std() * np.sqrt(261), 3) if len(x) > 200 else np.nan
rows = []
for k in D.columns:
    x = D[k]
    rows.append(dict(series=k, start=x.dropna().index[0].date(), sr_1990=st(x, '1990'), sr_2000=st(x, '2000'), sr_2010=st(x, '2010'),
                     sr_2015=st(x, '2015'), sr_2020=st(x, '2020'), sr_2023_07=st(x, '2023-07'),
                     vol_2000=round(x.loc['2000':].std() * np.sqrt(261), 4)))
T = pd.DataFrame(rows); print(T.to_string()); T.to_csv(OUT + 'series_sharpes.csv', index=False)
