"""(3) independent whole-contract simulation, all 15 keys (v2_250k and v5_250k are the primary targets), half-day lag
and same close, compared metric by metric with var_passe_retail_wide.csv and series by series with series_variantes.pkl."""
import time, pickle
import numpy as np, pandas as pd
from indep import *
t0 = time.time()
MY = pickle.load(open(HERE + 'subsets_mine.pkl', 'rb'))
W = pd.read_csv(A + 'var_passe_retail/var_passe_retail_wide.csv')
SER = pd.read_pickle(A + 'var_passe_retail/series_variantes.pkl')
EXTRA = pickle.load(open(A + 'var_passe_retail/series_variantes_extra.pkl', 'rb'))
rows, MYSER = [], {}
for v in SPEC:
    for C in (100e3, 250e3, 1e6):
        key = f'{v}_{int(C / 1e3)}k' if C < 1e6 else f'{v}_1M'
        for lag in (1.5, 1.0):
            d = simulate(v, MY[(v, C)], C, lag)
            ex = 'half_day_lag' if lag == 1.5 else 'same_close'
            if lag == 1.5:
                MYSER[key] = d
                a = d['net'].loc['1999-01-04':'2026-07-10']; b = SER[key]
                e = EXTRA[key]
                print(key, 'series max|diff| net', float((a - b).abs().max()), 'total', float((d['total'] - e['total_net']).abs().max()),
                      'cash', float((d['cash'] - e['cash_leg']).abs().max()), 'margin', float((d['margin'] - e['margin']).abs().max()), flush=True)
            for p, (s0, s1) in PERIODS.items():
                sl = slice(s0, s1); yrs = (pd.Timestamp(s1) - pd.Timestamp(s0)).days / 365.25
                ex_, tn, tb = d['net'].loc[sl], d['total'].loc[sl], TB.loc[sl]
                mine = dict(cagr_excess=cagr(ex_), cagr_total_tbill=cagr(ex_ + tb), cagr_total_net_all_fees=cagr(tn),
                            vol=ex_.std() * 16, sharpe=sr(ex_), sharpe_all_in=sr(tn - tb), maxdd_excess=mdd(ex_),
                            cost_fee_plus_halfspread_pct=d['cost'].loc[sl].sum() / yrs * 100,
                            cost_fixed_pct=d['fixed'].loc[sl].sum() / yrs * 100,
                            cost_cash_shortfall_pct=(tb - d['cash'].loc[sl]).sum() / yrs * 100,
                            mean_margin_pct=d['margin'].loc[sl].mean() * 100, mean_gross_exposure=d['gross_exp'].loc[sl].mean(),
                            tbill_avg_pct=tb.sum() / yrs * 100)
                mine['cost_total_drag_pct'] = mine['cost_fee_plus_halfspread_pct'] + mine['cost_fixed_pct'] + mine['cost_cash_shortfall_pct']
                ref = W[(W.key == key) & (W.execution == ex) & (W.period == p)].iloc[0]
                for m, x in mine.items():
                    rows.append(dict(key=key, execution=ex, period=p, metric=m, mine=x, prev=ref[m], diff=x - ref[m]))
        print(key, round(time.time() - t0), flush=True)
R = pd.DataFrame(rows)
R.to_csv(HERE + 'metrics_compare.csv', index=False)
print('max |diff| by metric:')
print(R.groupby('metric')['diff'].apply(lambda s: s.abs().max()).to_string())
pickle.dump({k: {kk: d[kk] for kk in ('net', 'total', 'cash', 'fixed', 'margin', 'gross_exp', 'cost')} for k, d in MYSER.items()},
            open(HERE + 'my_series.pkl', 'wb'))
