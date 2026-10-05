import numpy as np, pandas as pd
D = pd.read_pickle('series.pkl')
D['eq_sp_eur'] = (1 + D['ES1 Index']) / (1 + D['EC1 Curncy']) - 1
D['eq_nq_eur'] = (1 + D['NQ1 Index']) / (1 + D['EC1 Curncy']) - 1
def vr(x, k):
    x = x.dropna().values; x = x - x.mean()
    s = np.convolve(x, np.ones(k), 'valid')
    return s.var() / (k * x.var())
rows = []
for k in ['v5_84', 'v5_82_exCUA_SCO', 'sub_50k', 'sub_100k', 'sub_250k', 'sub_500k', 'sub_1000k', 'dbi_rep_ex', 'sgtrend_ex', 'sgcta_ex', 'ES1 Index', 'eq_sp_eur', 'eq_nq_eur']:
    x = D[k].loc['1990':'2026-07-10'].dropna()
    if k in ('dbi_rep_ex', 'sgtrend_ex', 'sgcta_ex', 'eq_sp_eur', 'eq_nq_eur'):
        x = D[k].loc['2000-03-29':'2026-07-10'].dropna()
    yr = (1 + x).groupby(x.index.year).prod() - 1
    rows.append(dict(series=k, start=x.index[0].date(), n=len(x), vol_daily_ann=round(x.std() * np.sqrt(261), 4), skew=round(x.skew(), 2), kurt=round(x.kurt(), 1),
                     ac1=round(x.autocorr(1), 3), vr21=round(vr(x, 21), 2), vr63=round(vr(x, 63), 2), vr126=round(vr(x, 126), 2), vr252=round(vr(x, 252), 2),
                     sd_calendar_year=round(yr.iloc[1:-1].std(), 3)))
T = pd.DataFrame(rows); print(T.to_string()); T.to_csv('shock_properties.csv', index=False)
s = D.loc['2000-03-29':'2026-07-10']
print(s[['v5_84', 'sub_100k', 'sub_1000k', 'dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur', 'ES1 Index']].corr().round(2))
m = (1 + s[['v5_84', 'dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'ES1 Index']]).resample('ME').prod() - 1
print('monthly corr\n', m.corr().round(2))
D.to_pickle('series.pkl')
