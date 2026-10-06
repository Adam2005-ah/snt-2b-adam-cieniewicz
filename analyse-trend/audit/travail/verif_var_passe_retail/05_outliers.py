"""(5) largest daily moves of the retail series: which market drives them (data artefact or genuine)?"""
import pickle
import numpy as np, pandas as pd
import indep as I
MY = pickle.load(open(I.HERE + 'subsets_mine.pkl', 'rb'))
S = pd.read_pickle(I.A + 'var_passe_retail/series_variantes.pkl')
rows = []
for key in ['v1_100k', 'v1_250k', 'v1_1M', 'v3_100k', 'v5_250k', 'v2_250k']:
    v, t = key.split('_'); C = {'100k': 100e3, '250k': 250e3, '1M': 1e6}[t]
    d = I.simulate(v, MY[(v, C)], C, 1.5)
    held = (0.5 * d['frac'].shift(1) + 0.5 * d['frac'].shift(2)).fillna(0)
    pnl = held * I.RETS.reindex(I.IDX)[MY[(v, C)]].fillna(0)
    for day in S[key].abs().nlargest(4).index:
        c = pnl.loc[day]; top = c.abs().nlargest(2).index
        rows.append(dict(key=key, day=day.date(), day_ret=S.loc[day, key], top=' | '.join(f'{m}: pnl {c[m]*100:.1f}% ret {I.RETS.loc[day, m]*100:.1f}% held {held.loc[day, m]:.2f}' for m in top)))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
print(pd.DataFrame(rows).to_string(index=False))
