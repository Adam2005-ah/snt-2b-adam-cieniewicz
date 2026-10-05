"""PRE-TAX (avant impot) re-run of cv_rendement/04_final.py. Every assumption of the verified final run is kept
(same series.pkl, same seed 20261005 and same index-generation order -> identical bootstrap shocks, stationary
bootstrap mean block 126 d, 5000 paths, 5 and 10 y, ESTR 2.4 %, IBKR cash rule with margin unpaid, MMF at ESTR-0.12 %,
same forward-Sharpe grids, same equity calibration), but ALL taxes are zero:
  DIY: pfu=0 (nothing taxed: futures pool, MMF, IBKR interest), cash benchmark untaxed (cash_tax=0);
  ETF / equities: tax=0, cash_tax=0;   mixes: tax_e=tax_t=0, reb_tax=0 (rebalancing untaxed), cash_tax=0;
  cash: MMF at ESTR - 0.10 % TER, untaxed.
Fixed costs (DIY, IB Gateway route): USD 150 / 400 / 1,000 per year at every tier + previous 500/625/1,000 for comparison.
Outputs: final_table_avant_impot.csv, final_summary_avant_impot.csv (percent), sensitivity_avant_impot.csv,
         rendement_attendu_10ans_avant_impot.csv (report-ready central table), sanity_check_pre_med.csv.
Column convention: at_* columns keep their old names for compatibility with rapport_audit.py but are PRE-TAX here
(pre_* = at_*)."""
import os, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
import mc_engine as M

t0 = time.time()
NP, NY, BLOCK = 5000, 10, 126
YEARS = (5, 10)
ESTR = 0.024
INFL = 0.02
AN = M.AN
EURUSD = 1.17
TIME_COST_USD = 80 * 25 * EURUSD          # 80 h x EUR 25 = EUR 2,000 ~ USD 2,340 / yr
S = pd.read_pickle('series.pkl')

# ---------------- samples & bootstrap indices: IDENTICAL generation order to cv_rendement/04_final.py ----------------
D90 = S.loc['1990-01-01':'2026-07-10']
J00 = S.loc['2000-03-29':'2026-07-10', ['dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur']].dropna()
rng = np.random.default_rng(20261005)
IDX90 = {b: M.sb_idx(len(D90), NP, NY * AN, b, rng) for b in (63, 126, 252)}
IDX00 = {b: M.sb_idx(len(J00), NP, NY * AN, b, rng) for b in (63, 126, 252)}
print('indices', round(time.time() - t0, 1), 'J00 rows', len(J00), 'D90 rows', len(D90))
_cache = {}


def Z(name, block=126):
    key = (name, block)
    if key not in _cache:
        if name in J00.columns:
            _cache[key] = M.standardise(J00[name].values)[IDX00[block]]
        else:
            _cache[key] = M.standardise(D90[name].values)[IDX90[block]]
    return _cache[key]


rows = []


def add(res, **kw):
    for y, r in res.items():
        d = dict(kw); d['horizon'] = y; d.update(r)
        d.setdefault('pre_med', r['at_med']); d.setdefault('pre_p10', r['at_p10']); d.setdefault('pre_p90', r['at_p90'])
        rows.append(d)


# ---------------- (a)/(b) DIY tiers ----------------
TIERS = {
    #           K0     prev fixed$  shocks21    SR grid 21 %              shocks12    SR grid 12 % (granularity penalty)
    '100k': (100e3, 500.0, 'sub_100k', (0.00, 0.20, 0.40), 'sub_50k', (-0.09, 0.11, 0.31)),
    '250k': (250e3, 625.0, 'sub_250k', (0.03, 0.24, 0.43), 'sub_100k', (0.00, 0.20, 0.40)),
    '1M': (1e6, 1000.0, 'sub_1000k', (0.05, 0.27, 0.47), 'sub_500k', (0.05, 0.27, 0.47)),
}
POLICY = {0.214: dict(c=0.40, m=0.17), 0.12: dict(c=0.25, m=0.095)}   # cash kept at IBKR / mean margin, % NAV
LAB = ('pess', 'central', 'opt')
FIXED_SETS = {'ibg150': lambda t: 150.0, 'ibg400': lambda t: 400.0, 'ibg1000': lambda t: 1000.0,
              'prev_500_625_1000': lambda t: TIERS[t][1]}


def diy(tier, vol, sr, fixed, estr=ESTR, block=126, years=YEARS):
    K0, _, s21, _, s12, _ = TIERS[tier]
    pol = POLICY[vol]
    z = Z(s21 if vol > 0.2 else s12, block)
    return M.sim_diy(z, sr, vol, K0, estr, years=years, mode='ibkr', fixed_usd=fixed, c=pol['c'], m=pol['m'],
                     tax_mode='split', carry_value=0.0, infl=INFL, pfu=0.0, cash_tax=0.0)


for fset, ffun in FIXED_SETS.items():
    for tier, (K0, _, s21, g21, s12, g12) in TIERS.items():
        fx = ffun(tier)
        for vol, grid in ((0.214, g21), (0.12, g12)):
            for lab, sr in zip(LAB, grid):
                add(diy(tier, vol, sr, fx), case='a/b DIY v5 subset', tier=tier, vol=vol, scen=lab, sr=sr, fixed_set=fset,
                    fixed_usd=fx, assumptions='PRE-TAX; shocks=%s, fixed $%d/yr, IBKR cash %d%%/margin %.1f%% unpaid, rest MMF ESTR-0.12%%'
                    % (s21 if vol > 0.2 else s12, fx, POLICY[vol]['c'] * 100, POLICY[vol]['m'] * 100))
    print('DIY', fset, round(time.time() - t0, 1))

# ---------------- (c) UCITS trend ETF (iMGP DBi R EUR HP, TER 0.75 % inside SR) ----------------
for lab, sr in zip(LAB, (0.10, 0.25, 0.40)):
    add(M.sim_etf(Z('dbi_rep_ex'), sr, 0.12, ESTR, years=YEARS, tax=0.0, cash_tax=0.0, infl=INFL), case='c UCITS trend ETF',
        tier='any', vol=0.12, scen=lab, sr=sr, assumptions='PRE-TAX; DBi-replica shocks 2000-26, EUR-hedged, accumulating')

# ---------------- (d) cash: money-market fund at ESTR - 0.10 %, untaxed ----------------
for y in YEARS:
    for lab, e in (('estr-0.5', ESTR - 0.005), ('central', ESTR), ('estr+0.5', ESTR + 0.005)):
        w = M.cash_benchmark(e, y, tax=0.0); cagr = w ** (1 / y) - 1
        rows.append(dict(case='d EUR cash (XEON-type MMF, acc.)', tier='any', vol=0.0, scen=lab, sr=np.nan, horizon=y,
                         at_med=cagr, at_p10=cagr, at_p90=cagr, at_mean_wealth=cagr, real_med=(1 + cagr) / (1 + INFL) - 1,
                         real_p10=(1 + cagr) / (1 + INFL) - 1, real_p90=(1 + cagr) / (1 + INFL) - 1,
                         P_nom_loss=0.0, P_real_loss=float(cagr < INFL), P_below_cash=np.nan, cash_at=cagr,
                         dd_med=0.0, dd_p10=0.0, pre_med=cagr, pre_p10=cagr, pre_p90=cagr,
                         assumptions='PRE-TAX; ESTR %.2f%%, TER 0.10%%' % (e * 100)))

# ---------------- (e) equities, 70/30 mix (annual rebalance) ----------------
EQ = {'pess': 0.035, 'central': 0.060, 'opt': 0.080}   # target median pre-tax CAGR in EUR, net of ETF fee
VOL_E = {'sp': (0.195, 'eq_sp_eur'), 'nq': (0.25, 'eq_nq_eur')}


def calib(zname, vol, target):
    mu = target + vol ** 2 / 2
    for _ in range(3):
        r = M.sim_etf(Z(zname), 0.0, vol, 0.0, years=(10,), tax=0.0, total_mu=mu)[10]['pre_med']
        mu += target - r
    return mu


MU = {(k, lab): calib(VOL_E[k][1], VOL_E[k][0], g) for k in VOL_E for lab, g in EQ.items()}
print('equity calibration (arith mu):', {k: round(v, 5) for k, v in MU.items()})
TR = {'pess': 0.10, 'central': 0.25, 'opt': 0.40}
for k, (ve, zn) in VOL_E.items():
    for lab, g in EQ.items():
        add(M.sim_etf(Z(zn), 0.0, ve, ESTR, years=YEARS, total_mu=MU[(k, lab)], tax=0.0, cash_tax=0.0, infl=INFL),
            case='e 100%% equities (%s)' % k, tier='any', vol=ve, scen=lab, sr=np.nan,
            assumptions='PRE-TAX; median geo target %.1f%%, arith %.2f%%, EUR unhedged' % (g * 100, MU[(k, lab)] * 100))
    combos = [('pess', 'pess'), ('central', 'central'), ('opt', 'opt'), ('pess', 'central'), ('central', 'pess'), ('central', 'opt')]
    for el, tl in combos:
        add(M.sim_mix(Z(zn), Z('dbi_rep_ex'), MU[(k, el)], ve, TR[tl], 0.12, ESTR, w_e=0.7, years=YEARS, rebalance=True,
                      tax_e=0.0, tax_t=0.0, reb_tax=0.0, cash_tax=0.0, infl=INFL),
            case='e 70/30 annual rebalance (%s)' % k, tier='any', vol=np.nan, scen='eq %s / trend %s' % (el, tl), sr=TR[tl],
            assumptions='PRE-TAX; equity geo %.1f%%, trend ETF SR %.2f @12%%' % (EQ[el] * 100, TR[tl]))
print('mix done', round(time.time() - t0, 1))

F = pd.DataFrame(rows)
prev_cols = list(pd.read_csv('prev_final_table.csv', nrows=1).columns)
cols = [c for c in prev_cols if c in F.columns] + ['fixed_set', 'fixed_usd']
F = F[cols]
F.to_csv('final_table_avant_impot.csv', index=False)
print('table saved', F.shape, round(time.time() - t0, 1))

# ---------------- sanity check vs previous pre_med ----------------
P = pd.read_csv('prev_final_table.csv')
chk = []
pd_ = P[P.case == 'a/b DIY v5 subset']
nd = F[(F.case == 'a/b DIY v5 subset') & (F.fixed_set == 'prev_500_625_1000')]
for _, r in pd_.iterrows():
    n = nd[(nd.tier == r.tier) & np.isclose(nd.vol, r.vol) & (nd.scen == r.scen) & (nd.horizon == r.horizon)].iloc[0]
    chk.append(dict(case='DIY', tier=r.tier, vol=r.vol, scen=r.scen, horizon=r.horizon, prev_pre_med=r.pre_med, new_med=n.at_med,
                    diff_pt=(n.at_med - r.pre_med) * 100, prev_pre_p10=r.pre_p10, new_p10=n.at_p10, prev_pre_p90=r.pre_p90, new_p90=n.at_p90,
                    prev_after_tax_med=r.at_med, tax_drag_pt=(n.at_med - r.at_med) * 100))
for oc, nc in (('c UCITS trend ETF', 'c UCITS trend ETF'), ('e 100% equities CTO (sp)', 'e 100% equities (sp)'),
               ('e 100% equities CTO (nq)', 'e 100% equities (nq)')):
    for _, r in P[P.case == oc].iterrows():
        n = F[(F.case == nc) & (F.scen == r.scen) & (F.horizon == r.horizon)].iloc[0]
        chk.append(dict(case=nc, tier='any', vol=r.vol, scen=r.scen, horizon=r.horizon, prev_pre_med=r.pre_med, new_med=n.at_med,
                        diff_pt=(n.at_med - r.pre_med) * 100, prev_pre_p10=r.pre_p10, new_p10=n.at_p10, prev_pre_p90=r.pre_p90,
                        new_p90=n.at_p90, prev_after_tax_med=r.at_med, tax_drag_pt=(n.at_med - r.at_med) * 100))
C = pd.DataFrame(chk); C.to_csv('sanity_check_pre_med.csv', index=False)
print('sanity: max |diff| pt (median) =', C.diff_pt.abs().max(), ' p10', ((C.new_p10 - C.prev_pre_p10).abs().max() * 100),
      ' p90', ((C.new_p90 - C.prev_pre_p90).abs().max() * 100))

# ---------------- sensitivities (10 y) ----------------
sen = []


def s_add(case, tier, what, r, fixed=np.nan):
    r = r[10]
    sen.append(dict(case=case, tier=tier, sensitivity=what, fixed_usd=fixed, horizon=10, med=r['at_med'], p10=r['at_p10'], p90=r['at_p90'],
                    real_med=r['real_med'], P_nom_loss=r['P_nom_loss'], P_below_cash=r['P_below_cash'], dd_med=r['dd_med'],
                    dd_p10=r['dd_p10'], P_touch_60pct=r.get('P_touch_60pct', np.nan)))


BASE_FIX = 400.0
for tier, (K0, prevfx, s21, g21, s12, g12) in TIERS.items():
    for vol, grid in ((0.214, g21), (0.12, g12)):
        cname = 'DIY %d%% vol' % round(vol * 100)
        sr = grid[1]
        s_add(cname, tier, 'base (central SR %.2f, ESTR 2.4 %%, fixed $400)' % sr, diy(tier, vol, sr, BASE_FIX, years=(10,)), BASE_FIX)
        s_add(cname, tier, 'SR -0.10', diy(tier, vol, sr - 0.1, BASE_FIX, years=(10,)), BASE_FIX)
        s_add(cname, tier, 'SR +0.10', diy(tier, vol, sr + 0.1, BASE_FIX, years=(10,)), BASE_FIX)
        s_add(cname, tier, 'ESTR -0.5 (1.9 %)', diy(tier, vol, sr, BASE_FIX, estr=ESTR - 0.005, years=(10,)), BASE_FIX)
        s_add(cname, tier, 'ESTR +0.5 (2.9 %)', diy(tier, vol, sr, BASE_FIX, estr=ESTR + 0.005, years=(10,)), BASE_FIX)
        s_add(cname, tier, 'fixed $150/yr', diy(tier, vol, sr, 150.0, years=(10,)), 150.0)
        s_add(cname, tier, 'fixed $1,000/yr', diy(tier, vol, sr, 1000.0, years=(10,)), 1000.0)
        s_add(cname, tier, 'fixed previous ($%d/yr)' % prevfx, diy(tier, vol, sr, prevfx, years=(10,)), prevfx)
        s_add(cname, tier, 'time 80 h x EUR 25 (~$%d/yr) on top of $400' % TIME_COST_USD,
              diy(tier, vol, sr, BASE_FIX + TIME_COST_USD, years=(10,)), BASE_FIX + TIME_COST_USD)
    print('sens', tier, round(time.time() - t0, 1))
for what, kw in (('base (SR 0.25)', {}), ('SR -0.10', dict(sr=0.15)), ('SR +0.10', dict(sr=0.35)),
                 ('ESTR -0.5 (1.9 %)', dict(estr=ESTR - 0.005)), ('ESTR +0.5 (2.9 %)', dict(estr=ESTR + 0.005))):
    s_add('UCITS trend ETF', 'any', what, M.sim_etf(Z('dbi_rep_ex'), kw.get('sr', 0.25), 0.12, kw.get('estr', ESTR), years=(10,),
                                                     tax=0.0, cash_tax=0.0, infl=INFL))
for k, (ve, zn) in VOL_E.items():
    for what, kw in (('base (eq 6 %, trend SR 0.25)', {}), ('trend SR -0.10', dict(sr=0.15)), ('trend SR +0.10', dict(sr=0.35)),
                     ('ESTR -0.5 (1.9 %)', dict(estr=ESTR - 0.005)), ('ESTR +0.5 (2.9 %)', dict(estr=ESTR + 0.005))):
        s_add('70/30 annual rebalance (%s)' % k, 'any', what,
              M.sim_mix(Z(zn), Z('dbi_rep_ex'), MU[(k, 'central')], ve, kw.get('sr', 0.25), 0.12, kw.get('estr', ESTR), w_e=0.7,
                        years=(10,), rebalance=True, tax_e=0.0, tax_t=0.0, reb_tax=0.0, cash_tax=0.0, infl=INFL))
G = pd.DataFrame(sen); G.to_csv('sensitivity_avant_impot.csv', index=False)
print('sensitivity done', round(time.time() - t0, 1))
