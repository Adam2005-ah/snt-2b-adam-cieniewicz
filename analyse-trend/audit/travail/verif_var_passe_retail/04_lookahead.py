"""(4) look-ahead test: recompute everything (vol, forecasts, regime rank, fixed vol, weights, IDM, vol-target scale,
whole contracts, margin sigma) on data truncated at T and compare with the full-sample run up to T.
v2_250k, v5_250k, v1_100k, v4_1M; T in {2008-09-30, 2015-06-30, 2022-12-30}."""
import pickle
import numpy as np, pandas as pd
import indep as I
MY = pickle.load(open(I.HERE + 'subsets_mine.pkl', 'rb'))
FULL = {k: v for k, v in I.D.items()}
FR, FV = I.RETS, I.VOL
cases = [('v2', 250e3), ('v5', 250e3), ('v1', 100e3), ('v4', 1e6)]
full_runs = {(v, C): I.simulate(v, MY[(v, C)], C, 1.5) for v, C in cases}
for T in ['2008-09-30', '2015-06-30', '2022-12-30']:
    r = FR.loc[:T]
    vol = r.apply(I.trend.annual_vol)
    I.D = dict(rets=r, groups=FULL['groups'], vol=vol,
               fc_def=r.apply(lambda s: I.trend.forecast(s, vol[s.name])),
               fc_reg=r.apply(lambda s: I.trend.forecast_regime(s, vol[s.name])),
               fixv=I.trend.fixed_vol(r))
    I.RETS, I.VOL = r, vol
    for v, C in cases:
        f = full_runs[(v, C)]
        t = I.simulate(v, MY[(v, C)], C, 1.5)      # IDX/TB/SIG are global; SIG is ewm/rolling (recomputed below)
        idx = t['n'].index[t['n'].index <= T]
        dn = (t['n'].loc[idx] - f['n'].loc[idx]).abs().max().max()
        dnet = (t['net'].loc[idx] - f['net'].loc[idx]).abs().max()
        dpos = (t['s']['pos'].loc[:T] - f['s']['pos'].loc[:T]).abs().max().max()
        print(T, v, int(C), 'max|d contracts|', dn, 'max|d frac pos|', float(dpos), 'max|d net|', float(dnet), flush=True)
    I.D, I.RETS, I.VOL = FULL, FR, FV
# margin sigma: backward-looking by construction? recompute on truncated data
T = '2015-06-30'
e = FR.loc[:T].ewm(span=32, min_periods=32).std() * 16; y = FR.loc[:T].rolling(252, min_periods=126).std() * 16
sig_t = np.maximum(e, y).ffill().reindex(I.IDX[I.IDX <= T])
print('margin sigma truncated vs full max diff', float((sig_t - I.SIG.loc[:T]).abs().max().max()))
