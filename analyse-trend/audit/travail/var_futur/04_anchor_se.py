"""How precise are the anchors? Sharpe since 2010 of each variant vs v5 at the same tier, correlation, and the standard
error of the Sharpe difference (Jobson-Korkie / Memmel, i.i.d. approximation, x16 convention). -> anchor_se.csv"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_pickle(os.path.join(os.path.dirname(HERE), 'var_passe_retail', 'series_variantes.pkl')).loc['2010':]
T = len(S) / 261
rows = []
for t in ('100k', '250k', '1M'):
    b = S['v5_' + t]; sb = b.mean() / b.std() * 16
    for v in ('v1', 'v2', 'v3', 'v4', 'v5'):
        a = S[v + '_' + t]; sa = a.mean() / a.std() * 16; r = a.corr(b)
        se_d = np.sqrt((2 - 2 * r + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * r ** 2)) / T) if v != 'v5' else np.nan
        rows.append(dict(key=v + '_' + t, sharpe_2010=sa, se_sharpe=np.sqrt((1 + 0.5 * sa ** 2) / T), corr_with_v5=r,
                         v5_minus_variant=sb - sa if v != 'v5' else np.nan, se_difference=se_d,
                         t_stat=(sb - sa) / se_d if v != 'v5' else np.nan))
A = pd.DataFrame(rows); A.to_csv(os.path.join(HERE, 'anchor_se.csv'), index=False)
print('years', round(T, 2)); print(A.round(3).to_string())
