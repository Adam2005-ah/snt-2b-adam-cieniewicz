"""Metrics from the independent runs (runs.pkl) and comparison with var_passe_global.csv / annees.csv /
decomposition.csv and the published project files (resultats/rapport_indicateurs.csv, rapport_annees.csv,
rapport_cagr_periodes.csv)."""
import pickle

import numpy as np
import pandas as pd

S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/verif_var_passe_global/'
PREV = S + 'audit/var_passe_global/'
PROJ = '/home/user/snt-2b-adam-cieniewicz/analyse-trend/resultats/'
D = pickle.load(open(OUT + 'runs.pkl', 'rb'))
R = D['runs']
VN = {'v1': '1. Montant fixe', 'v2': '2. Système actuel', 'v3': '3. + stratégie 13', 'v4': '4. + pilotage du risque',
      'v5': '5. Les deux'}

# ---- cash, rebuilt from the raw FRED file
dtb3 = pd.read_csv(S + 'mkt/us-etf/alm0421_macro/DTB3.csv', parse_dates=['date']).set_index('date')['value'].dropna() / 100
idx = R[('v2', 'L1')].loc['1990-01-01':].index
on_cal = dtb3.reindex(dtb3.index.union(idx)).ffill().reindex(idx)
prev_rate = on_cal.shift(1)
prev_rate.iloc[0] = on_cal.iloc[0]
ndays = np.r_[1, np.diff(idx.values).astype('timedelta64[D]').astype(int)]
TB = prev_rate * ndays / 365
IBR = (prev_rate - 0.005).clip(lower=0) * ndays / 365
G5 = R[('v5', 'L1')]['pos_exp'].loc['1990-01-01':].mean()
print('v5 published mean gross exposure from 1990 (positions):', round(G5, 4))

PER = {'1990-2026': '1990-01-01', '2000-2026': '2000-01-01', '2010-2026': '2010-01-01', '2015-2026': '2015-01-01',
       '2023-07->2026-07': '2023-07-11'}


def yrs(ix):
    return (ix[-1] - ix[0]).days / 365.25 + 1 / 261


def build(v, lvl, margin=0.17):
    if lvl != 'L0':
        df = R[(v, lvl)].loc['1990-01-01':].copy()
    if lvl == 'L0':
        df = R[(v, 'L1')].loc['1990-01-01':].copy()
        df['fut'] = df['gross']
        df['trading'] = 0.0
        df['roll'] = 0.0
    if lvl == 'L2':
        m = (margin * df['gross_exp'] / G5).clip(upper=1)
        cash = (1 - m) * IBR
    else:
        m = 0 * df['fut']
        cash = TB
    df['m'] = m
    df['cash'] = cash
    df['total'] = df['fut'] + cash
    df['excess'] = df['total'] - TB
    return df


def met(df, start):
    d = df.loc[start:]
    y = yrs(d.index)
    curve = (1 + d['total']).cumprod()
    return {'cagr_total': curve.iloc[-1] ** (1 / y) - 1,
            'cagr_excess': (1 + d['excess']).prod() ** (1 / y) - 1,
            'sharpe': d['excess'].mean() / d['excess'].std() * 16,
            'vol': d['total'].std() * 16,
            'maxdd_total': (curve / curve.cummax() - 1).min(),
            'cost_pa': (d['trading'] + d['roll']).sum() / y,
            'cash_shortfall_pa': (TB.loc[start:] - d['cash']).sum() / y,
            'margin_share_mean': d['m'].mean(),
            'gross_exposure_mean': d['gross_exp'].mean(),
            'years': y}


rows = []
SER = {}
for v in VN:
    for lvl in ('L0', 'L1', 'L2'):
        df = build(v, lvl)
        SER[(v, lvl)] = df
        for p, s in PER.items():
            for k, x in met(df, s).items():
                rows.append(dict(variant=VN[v], level={'L0': 'L0 brut', 'L1': 'L1 backtest', 'L2': 'L2 réaliste'}[lvl],
                                 period=p, metric=k, mine=x))
mine = pd.DataFrame(rows)
prev = pd.read_csv(PREV + 'var_passe_global.csv')
cmp_ = mine.merge(prev, on=['variant', 'level', 'period', 'metric'], how='left')
cmp_['diff'] = cmp_['mine'] - cmp_['value']
cmp_.to_csv(OUT + 'compare_var_passe_global.csv', index=False, float_format='%.6f')
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
print('\nmax |diff| by metric x level:')
print(cmp_.groupby(['metric', 'level'])['diff'].apply(lambda x: x.abs().max()).unstack().to_string())

# ---- published figures (project resultats)
ind = pd.read_csv(PROJ + 'rapport_indicateurs.csv', index_col=0)
per = pd.read_csv(PROJ + 'rapport_cagr_periodes.csv', index_col=0)
print('\nL1 vs published rapport_indicateurs (1990-2026):')
for v, name in VN.items():
    m = met(SER[(v, 'L1')], '1990-01-01')
    print(name, 'CAGR %.4f/%.4f  Sharpe %.4f/%.4f  vol %.4f/%.4f  DD %.4f/%.4f  costs %.4f/%.4f  expo %.4f/%.4f' % (
        m['cagr_total'], ind.loc['CAGR (rendement total)', name], m['sharpe'], ind.loc['Sharpe', name],
        m['vol'], ind.loc['Volatilité annualisée (écart-type quotidien)', name], m['maxdd_total'],
        ind.loc['Pire baisse (max drawdown)', name], m['cost_pa'], ind.loc['Coûts par an (transactions + roulement)', name],
        SER[(v, 'L1')]['pos_exp'].loc['1990':].mean(), ind.loc['Exposition moyenne (notionnel / capital)', name]))
    print('   periods 2000/2010:', round(met(SER[(v, 'L1')], '2000-01-01')['cagr_total'], 4), per.loc[name, '2000-2026'],
          round(met(SER[(v, 'L1')], '2010-01-01')['cagr_total'], 4), per.loc[name, '2010-2026'])

# ---- calendar years
ann_pub = pd.read_csv(PROJ + 'rapport_annees.csv', index_col=0)
ann_prev = pd.read_csv(PREV + 'annees.csv', index_col=[0, 1])
mx = 0
mx_prev = {}
for v, name in VN.items():
    for lvl, ln in (('L0', 'L0 brut'), ('L1', 'L1 backtest'), ('L2', 'L2 réaliste')):
        t = SER[(v, lvl)]['total']
        y = (1 + t).groupby(t.index.year).prod() - 1
        if lvl == 'L1':
            common = [i for i in y.index if str(i) in ann_pub.index.astype(str)]
            pub = ann_pub.copy()
            pub.index = pub.index.astype(str)
            d = (y.loc[common].values - pub.loc[[str(i) for i in common], name].values)
            mx = max(mx, np.abs(d).max())
        yy = y.loc[2010:]
        pv = ann_prev.loc[(name, ln)].values.astype(float)
        mx_prev[(v, lvl)] = np.abs(yy.values - pv).max()
print('\nL1 calendar years vs rapport_annees.csv (all years in file): max |diff| =', mx, '(file rounded to 4 dp)')
print('calendar years vs var_passe_global/annees.csv max |diff|:', {k: round(x, 5) for k, x in mx_prev.items()})
pickle.dump(SER, open(OUT + 'series_verif.pkl', 'wb'))
