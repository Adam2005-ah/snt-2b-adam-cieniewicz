"""Aggregate raw_runs.csv (4 seeds x 10,000 paths) -> var_futur.csv (long), var_futur_wide.csv,
var_futur_tableau_10ans.csv (compact report table, central, 10 y), checks.csv."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
R = pd.read_csv(os.path.join(HERE, 'raw_runs.csv'))
P = pd.read_csv(os.path.join(HERE, 'params.csv'), index_col=0)
PAST = pd.read_csv(os.path.join(AUD, 'var_passe_retail', 'var_passe_retail_wide.csv'))
PAST = PAST[PAST.execution == 'half_day_lag']
REF = pd.read_csv('/home/user/snt-2b-adam-cieniewicz/analyse-trend/audit/rendement_attendu_10ans_ib_gateway.csv')

MET = ['cagr_med', 'cagr_p10', 'cagr_p90', 'cagr_mean_wealth', 'real_med', 'real_p10', 'real_p90', 'P_nom_loss',
       'P_real_loss', 'P_below_cash', 'cash_cagr', 'dd_med', 'dd_p10', 'ddr_med', 'ddr_p10', 'P_touch_60pct']
G = ['key', 'variant', 'tier', 'case', 'scen', 'horizon', 'cash_model']
mean = R.groupby(G, sort=False)[MET + ['sr', 'vol', 'm', 'c']].mean()
sd = R.groupby(G, sort=False)[MET].std()
ns = R.groupby(G, sort=False).seed.nunique()
W = mean.copy()
for k in MET:
    W[k + '_seed_sd'] = sd[k]
W['n_seeds'] = ns
W = W.reset_index()
W['variant_name'] = W.key.map(P.variant_name)
W['capital_usd'] = W.key.map(P.K0).astype(int)
W['n_markets'] = W.key.map(P.n_markets)
W['A_sharpe_2010'] = W.key.map(P.A_sharpe_2010)
W['h_tier'] = W.key.map(P.h_tier)
W['central_retenu'] = (W.cash_model == 'expo_brute') & (
    ((W.variant.isin(['v1', 'v2'])) & (W.case == 'v1v2_sans_selection')) |
    ((W.variant.isin(['v3', 'v4', 'v5'])) & (W.case == 'commune')))
# v3..v5: case 2 is identical to case 1 -> duplicate the rows so case 2 is a complete grid
dup = W[(W.case == 'commune') & W.variant.isin(['v3', 'v4', 'v5']) & (W.cash_model == 'expo_brute')].copy()
dup['case'] = 'v1v2_sans_selection'; dup['central_retenu'] = False
W = pd.concat([W, dup], ignore_index=True)
order = {'v1': 1, 'v2': 2, 'v3': 3, 'v4': 4, 'v5': 5}; tord = {'100k': 1, '250k': 2, '1M': 3}
sord = {'pess': 1, 'central': 2, 'opt': 3}
W = W.sort_values(['cash_model', 'case', 'horizon', 'variant', 'tier', 'scen'],
                  key=lambda s: s.map(order) if s.name == 'variant' else s.map(tord) if s.name == 'tier'
                  else s.map(sord) if s.name == 'scen' else s).reset_index(drop=True)
W['n_paths_total'] = W.n_seeds * 10000
lead = ['key', 'variant', 'variant_name', 'tier', 'capital_usd', 'n_markets', 'case', 'scen', 'horizon', 'cash_model',
        'central_retenu', 'A_sharpe_2010', 'h_tier', 'sr', 'vol', 'm', 'c', 'n_seeds', 'n_paths_total']
W = W[lead + MET + [k + '_seed_sd' for k in MET]]
W.to_csv(os.path.join(HERE, 'var_futur_wide.csv'), index=False)

L = W.melt(id_vars=lead, value_vars=MET, var_name='metric', value_name='value')
L['seed_sd'] = W.melt(id_vars=lead, value_vars=[k + '_seed_sd' for k in MET]).value.values
L = L.rename(columns={'scen': 'scenario', 'horizon': 'horizon_years', 'sr': 'sharpe_forward', 'm': 'margin_share_nav',
                      'c': 'cash_at_ibkr_share_nav'})
L.to_csv(os.path.join(HERE, 'var_futur.csv'), index=False)

# ---------------- compact table: central, 10 y ----------------
def pick(case, h=10, scen='central', cm='expo_brute'):
    x = W[(W.case == case) & (W.horizon == h) & (W.scen == scen) & (W.cash_model == cm)]
    return x.set_index('key')


rows = []
for key, p in P.iterrows():
    c2 = 'v1v2_sans_selection' if p.variant in ('v1', 'v2') else 'commune'
    c = pick(c2).loc[key]; c1 = pick('commune').loc[key]
    pe = pick(c2, scen='pess').loc[key]; op = pick(c2, scen='opt').loc[key]
    c5 = pick(c2, h=5).loc[key]
    mc = pick('commune', cm='marge_contrats').loc[key]
    past = PAST[PAST.key == key].set_index('period')
    rows.append({
        'Variante': p.variant_name, 'Capital': {'100k': '100 k$', '250k': '250 k$', '1M': '1 M$'}[p.tier],
        'key': key, 'Marchés': int(p.n_markets),
        'Sharpe réel 2010-26 (ancre)': round(p.A_sharpe_2010, 2),
        'Sharpe futur central': round(c.sr, 3), 'Sharpe futur cas 1 (décote commune)': round(p.sr_central_case1, 3),
        'Volatilité': round(c.vol, 3),
        'Médiane 10 ans': c.cagr_med, '10e centile': c.cagr_p10, '90e centile': c.cagr_p90,
        'Médiane réelle (2 % infl.)': c.real_med, 'P(perte)': c.P_nom_loss, 'P(< monétaire)': c.P_below_cash,
        'Baisse max médiane': c.dd_med, 'Baisse max pire décile': c.dd_p10, 'P(toucher 60 % du capital)': c.P_touch_60pct,
        'Médiane pessimiste': pe.cagr_med, 'Médiane optimiste': op.cagr_med,
        'Médiane cas 1 (décote commune)': c1.cagr_med, 'Médiane 5 ans': c5.cagr_med,
        'P(perte) 5 ans': c5.P_nom_loss, 'Écart médiane si marge calculée sur contrats (vs cas 1)': mc.cagr_med - c1.cagr_med,
        'Pente médiane par +0,1 de Sharpe': (op.cagr_med - pe.cagr_med) / 4,
        'Passé 2000-26 net de tous frais (USD)': past.loc['2000-2026', 'cagr_total_net_all_fees'],
        'Passé 2010-26 net de tous frais (USD)': past.loc['2010-2026', 'cagr_total_net_all_fees'],
        'Passé 2015-26 net de tous frais (USD)': past.loc['2015-2026', 'cagr_total_net_all_fees'],
        'Passé 2023-07..2026-07 net de tous frais (USD)': past.loc['2023-07_2026-07', 'cagr_total_net_all_fees'],
        'Écart-type médiane entre graines': c.cagr_med_seed_sd,
    })
T = pd.DataFrame(rows)
T['_o'] = T.key.str[1].astype(int) * 10 + T.key.map(lambda k: tord[k.split('_')[1]])
T = T.sort_values('_o').drop(columns='_o')
ref_rows = []
for _, r in REF.iterrows():
    if 'Variante 5' in r.cas:
        continue
    ref_rows.append({'Variante': 'Référence (audit vérifié) : ' + r.cas, 'Capital': '–', 'key': 'ref',
                     'Médiane 10 ans': r['median'], '10e centile': r.p10, '90e centile': r.p90,
                     'Médiane réelle (2 % infl.)': (1 + r['median']) / 1.02 - 1, 'P(perte)': r.perte,
                     'P(< monétaire)': r.sous_cash, 'Baisse max médiane': r.dd, 'Baisse max pire décile': r.dd10,
                     'Médiane pessimiste': r.pess, 'Médiane optimiste': r.opt})
T = pd.concat([T, pd.DataFrame(ref_rows)], ignore_index=True)
T.to_csv(os.path.join(HERE, 'var_futur_tableau_10ans.csv'), index=False)

# ---------------- checks ----------------
chk = []
AUD5 = {'100k': 0.0328, '250k': 0.0461, '1M': 0.0541}   # audit (verif_avant_impot fresh 4x10k, USD 400)
for t, a in AUD5.items():
    x = pick('commune').loc['v5_' + t]
    chk.append(dict(check='v5 central 10y median vs audit', tier=t, new=x.cagr_med, audit=a, diff_pt=(x.cagr_med - a) * 100,
                    ok=abs(x.cagr_med - a) <= 0.0015, seed_sd_pt=x.cagr_med_seed_sd * 100))
AUDF = {'100k': (-0.0505, 0.1240, 0.3109, 0.4429, -0.4630, -0.6532),
        '250k': (-0.0448, 0.1427, 0.2595, 0.3728, -0.4666, -0.6616),
        '1M': (-0.0396, 0.1547, 0.2331, 0.3382, -0.4605, -0.6535)}
for t, v in AUDF.items():
    x = pick('commune').loc['v5_' + t]
    for nm, a in zip(('cagr_p10', 'cagr_p90', 'P_nom_loss', 'P_below_cash', 'dd_med', 'dd_p10'), v):
        chk.append(dict(check='v5 central 10y %s vs audit' % nm, tier=t, new=x[nm], audit=a, diff_pt=(x[nm] - a) * 100))
C = pd.DataFrame(chk); C.to_csv(os.path.join(HERE, 'checks.csv'), index=False)

if __name__ == '__main__':
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
    print(C.round(4).to_string())
    pct = [c for c in T.columns if c not in ('Variante', 'Capital', 'key', 'Marchés', 'Sharpe réel 2010-26 (ancre)',
                                              'Sharpe futur central', 'Sharpe futur cas 1 (décote commune)', 'Volatilité')]
    Tp = T.copy(); Tp[pct] = (Tp[pct].astype(float) * 100).round(2)
    print(Tp.drop(columns=['Variante']).to_string())
