"""Clean pre-tax tables from final_table_avant_impot.csv:
 - final_summary_avant_impot.csv  (percent units, every case/scenario/horizon, DIY for each fixed-cost budget)
 - rendement_attendu_10ans_avant_impot.csv (same columns as audit/rendement_attendu_10ans.csv, 10 y, central,
   DIY at the USD 400/yr IB Gateway budget + medians at USD 150 and 1,000)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
F = pd.read_csv('final_table_avant_impot.csv')

# ---------- full summary in percent ----------
rows = []
order = []
for fset in ('ibg150', 'ibg400', 'ibg1000', 'prev_500_625_1000'):
    for tier in ('100k', '250k', '1M'):
        for vol in (0.214, 0.12):
            for sc in ('pess', 'central', 'opt'):
                f = F[(F.case == 'a/b DIY v5 subset') & (F.fixed_set == fset) & (F.tier == tier) & np.isclose(F.vol, vol) & (F.scen == sc)]
                order.append(('DIY v5 %s @%d%% vol, fixed $%s/yr' % (tier, round(vol * 100), int(f.fixed_usd.iloc[0])), fset, f))
for sc in ('estr-0.5', 'central', 'estr+0.5'):
    order.append(('EUR money-market fund (ESTR-0.10%)', '', F[(F.case == 'd EUR cash (XEON-type MMF, acc.)') & (F.scen == sc)]))
for sc in ('pess', 'central', 'opt'):
    order.append(('UCITS trend ETF iMGP DBi (12%)', '', F[(F.case == 'c UCITS trend ETF') & (F.scen == sc)]))
for k, nm in (('sp', 'S&P 500-type'), ('nq', 'Nasdaq-100-type')):
    for sc in ('pess', 'central', 'opt'):
        order.append(('100%% %s equities (EUR)' % nm, '', F[(F.case == 'e 100%% equities (%s)' % k) & (F.scen == sc)]))
    for sc in ('eq pess / trend pess', 'eq central / trend central', 'eq opt / trend opt', 'eq pess / trend central',
               'eq central / trend pess', 'eq central / trend opt'):
        order.append(('70/30 %s / trend ETF, annual rebal.' % nm, '', F[(F.case == 'e 70/30 annual rebalance (%s)' % k) & (F.scen == sc)]))
for name, fset, f in order:
    for _, r in f.iterrows():
        rows.append({'case': name, 'fixed_set': fset, 'scenario': r['scen'], 'fwd_SR': r['sr'], 'horizon_y': r['horizon'],
                     'pre_tax_med': r['at_med'], 'pre_tax_p10': r['at_p10'], 'pre_tax_p90': r['at_p90'],
                     'real_med': r['real_med'], 'real_p10': r.get('real_p10'), 'real_p90': r.get('real_p90'),
                     'P_nominal_loss': r['P_nom_loss'], 'P_real_loss': r.get('P_real_loss'), 'P_below_cash': r.get('P_below_cash'),
                     'cash_cagr': r.get('cash_at'),
                     'maxDD_med_nominal': r.get('dd_med'), 'maxDD_worst_decile_nominal': r.get('dd_p10'),
                     'maxDD_med_real': r.get('ddr_med'), 'maxDD_worst_decile_real': r.get('ddr_p10'),
                     'P_account_below_60pct_of_start': r.get('P_touch_60pct')})
T = pd.DataFrame(rows)
num = T.columns[5:]
T[num] = (T[num].astype(float) * 100).round(2)
T.to_csv('final_summary_avant_impot.csv', index=False)

# ---------- report-ready 10-y central table (fractions, like audit/rendement_attendu_10ans.csv) ----------
t = F[F.horizon == 10]


def diy(tier, vol, sc, fset='ibg400'):
    return t[(t.case == 'a/b DIY v5 subset') & (t.fixed_set == fset) & (t.tier == tier) & np.isclose(t.vol, vol) & (t.scen == sc)].iloc[0]


def one(case, sc):
    return t[(t.case == case) & (t.scen == sc)].iloc[0]


out = []
r = one('d EUR cash (XEON-type MMF, acc.)', 'central')
out.append(dict(cas='Monétaire en euros (fonds type XEON)', median=r.at_med, p10=r.at_p10, p90=r.at_p90, reel=r.real_med, perte=0.0,
                sous_cash=np.nan, dd=0.0, dd10=np.nan, pess=one('d EUR cash (XEON-type MMF, acc.)', 'estr-0.5').at_med,
                opt=one('d EUR cash (XEON-type MMF, acc.)', 'estr+0.5').at_med))
for vol, vtxt in ((0.214, ''), (0.12, ' à 12 % de risque')):
    for tier, ttxt in (('100k', '100 k$'), ('250k', '250 k$'), ('1M', '1 M$')):
        r = diy(tier, vol, 'central')
        out.append(dict(cas='Variante 5 soi-même%s, %s' % (vtxt, ttxt), median=r.at_med, p10=r.at_p10, p90=r.at_p90, reel=r.real_med,
                        perte=r.P_nom_loss, sous_cash=r.P_below_cash, dd=r.dd_med, dd10=r.dd_p10,
                        pess=diy(tier, vol, 'pess').at_med, opt=diy(tier, vol, 'opt').at_med, p_60pct=r.P_touch_60pct,
                        median_fixe150=diy(tier, vol, 'central', 'ibg150').at_med, median_fixe1000=diy(tier, vol, 'central', 'ibg1000').at_med,
                        median_fixe_ancien=diy(tier, vol, 'central', 'prev_500_625_1000').at_med))
r = one('c UCITS trend ETF', 'central')
out.append(dict(cas='ETF trend UCITS (iMGP DBi, couvert €)', median=r.at_med, p10=r.at_p10, p90=r.at_p90, reel=r.real_med, perte=r.P_nom_loss,
                sous_cash=r.P_below_cash, dd=r.dd_med, dd10=r.dd_p10, pess=one('c UCITS trend ETF', 'pess').at_med,
                opt=one('c UCITS trend ETF', 'opt').at_med))
for k, nm in (('sp', 'S&P 500'), ('nq', 'Nasdaq-100, type QQQ')):
    c = 'e 100%% equities (%s)' % k
    r = one(c, 'central')
    out.append(dict(cas='100 %% %s' % nm, median=r.at_med, p10=r.at_p10, p90=r.at_p90, reel=r.real_med, perte=r.P_nom_loss,
                    sous_cash=r.P_below_cash, dd=r.dd_med, dd10=r.dd_p10, pess=one(c, 'pess').at_med, opt=one(c, 'opt').at_med))
    c = 'e 70/30 annual rebalance (%s)' % k
    r = one(c, 'eq central / trend central')
    out.append(dict(cas='70 %% %s + 30 %% ETF trend' % nm.split(',')[0], median=r.at_med, p10=r.at_p10, p90=r.at_p90, reel=r.real_med,
                    perte=r.P_nom_loss, sous_cash=r.P_below_cash, dd=r.dd_med, dd10=r.dd_p10,
                    pess=one(c, 'eq pess / trend pess').at_med, opt=one(c, 'eq opt / trend opt').at_med))
O = pd.DataFrame(out)
O.to_csv('rendement_attendu_10ans_avant_impot.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30)
print(O.set_index('cas').mul(100).round(2).to_string())
