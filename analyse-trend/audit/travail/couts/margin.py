import pickle, pandas as pd, numpy as np
OUT = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/couts/'
D = pickle.load(open(OUT + 'v5_cache.pkl', 'rb')); g = D['groups']
# approximate exchange initial margin as % of notional (calibrated on ES 7.0%, ZN 1.9%, SR3 0.1-0.3%, CL 10.6%, GC 4.0%)
base = {'Equities': 0.07, 'Bonds': 0.025, 'STIR': 0.0025, 'FX': 0.025, 'Energy': 0.10, 'Metals': 0.05, 'Agriculture': 0.06}
spec = {'TU1 Comdty': .006, 'FV1 Comdty': .011, 'TY1 Comdty': .019, 'UXY1 Comdty': .026, 'US1 Comdty': .035, 'WN1 Comdty': .05,
        'DU1 Comdty': .007, 'OE1 Comdty': .013, 'RX1 Comdty': .023, 'UB1 Comdty': .05, 'G 1 Comdty': .03, 'JB1 Comdty': .015,
        'CN1 Comdty': .02, 'XM1 Comdty': .015, 'PE1 Curncy': .05, 'PA1 Comdty': .10, 'NG1 Comdty': .15, 'LC1 Comdty': .035,
        'LH1 Comdty': .05, 'FC1 Comdty': .035, 'CC1 Comdty': .15, 'QC1 Comdty': .15, 'KC1 Comdty': .10}
m = pd.Series({c: spec.get(c, base[g[c]]) for c in g.index})
out = {}
for v in ['v5', 'v2']:
    pos = D[v]['positions'].loc['1990':]
    mte = (pos.abs() * m).sum(axis=1)
    gross = pos.abs().sum(axis=1)
    out[v] = {'mean_margin_to_equity_%': mte.mean()*100, 'median_%': mte.median()*100, 'p95_%': mte.quantile(.95)*100,
              'max_%': mte.max()*100, 'last3y_mean_%': mte.loc['2023-07-10':].mean()*100, 'last_%': mte.iloc[-1]*100,
              'gross_exposure_mean_x': gross.mean(), 'gross_last3y_x': gross.loc['2023-07-10':].mean()}
print(pd.DataFrame(out).round(2))
pd.DataFrame(out).to_csv(OUT + 'margin_estimate.csv')
