"""Run the independent implementation on the audit's own shocks (mode 'same') and compare row by row with
cv_avant_impot/final_table_avant_impot.csv and with the previous cv_rendement/final_table.csv pre_* columns."""
import os, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import indep as I

t0 = time.time()
ESTR = 0.024
SH = I.Shocks('same')
print('shocks built', round(time.time() - t0, 1))
NEW = pd.read_csv('../cv_avant_impot/final_table_avant_impot.csv')
OLD = pd.read_csv('../cv_rendement/final_table.csv')

TIERS = {'100k': (100e3, 500., 'sub_100k', (0.00, 0.20, 0.40), 'sub_50k', (-0.09, 0.11, 0.31)),
         '250k': (250e3, 625., 'sub_250k', (0.03, 0.24, 0.43), 'sub_100k', (0.00, 0.20, 0.40)),
         '1M': (1e6, 1000., 'sub_1000k', (0.05, 0.27, 0.47), 'sub_500k', (0.05, 0.27, 0.47))}
POL = {0.214: (0.40, 0.17), 0.12: (0.25, 0.095)}
LAB = ('pess', 'central', 'opt')
rows = []
for tier, (K0, prevfx, s21, g21, s12, g12) in TIERS.items():
    for vol, grid, sn in ((0.214, g21, s21), (0.12, g12, s12)):
        z = SH.z(sn)
        for fx_set, fx in (('ibg150', 150.), ('ibg400', 400.), ('ibg1000', 1000.), ('prev_500_625_1000', prevfx)):
            for lab, sr in zip(LAB, grid):
                if fx_set != 'ibg400' and lab != 'central' and fx_set != 'prev_500_625_1000':
                    continue
                res = I.diy(z, sr, vol, K0, ESTR, fx, *POL[vol])
                for y, r in res.items():
                    rows.append(dict(case='a/b DIY v5 subset', tier=tier, vol=vol, scen=lab, fixed_set=fx_set, horizon=y, **r))
        print(tier, vol, round(time.time() - t0, 1))

zt = SH.z('dbi_rep_ex')
for lab, sr in zip(LAB, (0.10, 0.25, 0.40)):
    res, _ = I.fund(zt, ESTR + sr * 0.12, 0.12, ESTR)
    for y, r in res.items():
        rows.append(dict(case='c UCITS trend ETF', tier='any', scen=lab, horizon=y, **r))

# equity calibration, done independently: bisection on mu so that median 10y CAGR = target
EQ = {'pess': 0.035, 'central': 0.060, 'opt': 0.080}
VOL_E = {'sp': (0.195, 'eq_sp_eur'), 'nq': (0.25, 'eq_nq_eur')}
MU = {}
for k, (ve, zn) in VOL_E.items():
    ze = SH.z(zn)
    for lab, g in EQ.items():
        lo, hi = g, g + 0.08
        for _ in range(40):
            mid = (lo + hi) / 2
            r, _ = I.fund(ze, mid, ve, ESTR, years=(10,))
            lo, hi = (mid, hi) if r[10]['med'] < g else (lo, mid)
        MU[(k, lab)] = (lo + hi) / 2
        res, _ = I.fund(ze, MU[(k, lab)], ve, ESTR)
        for y, r in res.items():
            rows.append(dict(case='e 100%% equities (%s)' % k, tier='any', scen=lab, horizon=y, mu=MU[(k, lab)], **r))
    TR = {'pess': 0.10, 'central': 0.25, 'opt': 0.40}
    for el, tl in [('pess', 'pess'), ('central', 'central'), ('opt', 'opt'), ('pess', 'central'), ('central', 'pess'), ('central', 'opt')]:
        res = I.mix(ze, zt, MU[(k, el)], ve, ESTR + TR[tl] * 0.12, 0.12, ESTR)
        for y, r in res.items():
            rows.append(dict(case='e 70/30 annual rebalance (%s)' % k, tier='any', scen='eq %s / trend %s' % (el, tl), horizon=y, **r))
print('equity mu (indep calibration):', {k: round(v, 5) for k, v in MU.items()})
R = pd.DataFrame(rows)
R.to_csv('indep_same_shocks.csv', index=False)

# ---------- compare ----------
cmp = []
for _, r in R.iterrows():
    q = NEW[(NEW.case == r.case) & (NEW.scen == r.scen) & (NEW.horizon == r.horizon)]
    if r.case.startswith('a/b'):
        q = q[(q.tier == r.tier) & np.isclose(q.vol, r.vol) & (q.fixed_set == r.fixed_set)]
    q = q.iloc[0]
    d = dict(case=r.case, tier=r.tier, vol=r.get('vol'), scen=r.scen, fixed_set=r.get('fixed_set'), horizon=r.horizon,
             mine_med=r.med, theirs_med=q.at_med, d_med_pt=(r.med - q.at_med) * 100, d_p10_pt=(r.p10 - q.at_p10) * 100,
             d_p90_pt=(r.p90 - q.at_p90) * 100, d_Ploss=r.P_loss - q.P_nom_loss, d_Pcash=r.P_below_cash - q.P_below_cash,
             d_dd=r.dd_med - q.dd_med, d_dd10=r.dd_p10 - q.dd_p10, d_real=r.real_med - q.real_med,
             d_touch=(r.get('P_touch60', np.nan) - q.P_touch_60pct) if r.case.startswith('a/b') else np.nan)
    # previous table pre_med
    oc = {'e 100% equities (sp)': 'e 100% equities CTO (sp)', 'e 100% equities (nq)': 'e 100% equities CTO (nq)'}.get(r.case, r.case)
    o = OLD[(OLD.case == oc) & (OLD.scen == r.scen) & (OLD.horizon == r.horizon)]
    if r.case.startswith('a/b'):
        o = o[(o.tier == r.tier) & np.isclose(o.vol, r.vol)] if r.fixed_set == 'prev_500_625_1000' else o.iloc[0:0]
    if len(o) and 'pre_med' in o and not np.isnan(o.iloc[0].pre_med):
        d['old_pre_med'] = o.iloc[0].pre_med; d['d_vs_old_pre_pt'] = (r.med - o.iloc[0].pre_med) * 100
    cmp.append(d)
C = pd.DataFrame(cmp); C.to_csv('compare_same_shocks.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
for c in ['d_med_pt', 'd_p10_pt', 'd_p90_pt', 'd_Ploss', 'd_Pcash', 'd_dd', 'd_dd10', 'd_real', 'd_touch', 'd_vs_old_pre_pt']:
    print(c, 'max abs', np.nanmax(np.abs(C[c].values.astype(float))))
print(C[np.abs(C.d_med_pt) > 1e-6].to_string())
print('done', round(time.time() - t0, 1))
