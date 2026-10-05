"""Final cross-check Monte Carlo: DIY v5 tiers, UCITS trend ETF, cash, equities and 70/30 mix, French investor (EUR).
Outputs: final_table.csv (all cases, 5 y and 10 y), reconcile.csv (critic -> corrected waterfall), sensitivity.csv."""
import time, numpy as np, pandas as pd
import mc_engine as M

t0 = time.time()
NP, NY, BLOCK = 5000, 10, 126
YEARS = (5, 10)
ESTR = 0.024          # EUR cash: ECB DFR 2.50 % since 16-09-2026 -> ESTR ~2.43 % spot; 10y OIS ~2.9 % (incl. term premium)
INFL = 0.02
AN = M.AN
S = pd.read_pickle('series.pkl')

# ---------------- samples & bootstrap indices (common random numbers inside each sample) ----------------
D90 = S.loc['1990-01-01':'2026-07-10']
J00 = S.loc['2000-03-29':'2026-07-10', ['dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur']].dropna()
rng = np.random.default_rng(20261005)
IDX90 = {b: M.sb_idx(len(D90), NP, NY * AN, b, rng) for b in (63, 126, 252)}
IDX00 = {b: M.sb_idx(len(J00), NP, NY * AN, b, rng) for b in (63, 126, 252)}
print('indices', round(time.time() - t0, 1), 'J00 rows', len(J00))
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
        d = dict(kw); d['horizon'] = y; d.update(r); rows.append(d)


# ---------------- (a)/(b) DIY tiers ----------------
TIERS = {
    #           K0     fixed$  shocks21    SR grid 21 %              shocks12    SR grid 12 % (granularity: tier at K0*12/21.4)
    '100k': (100e3, 500.0, 'sub_100k', (0.00, 0.20, 0.40), 'sub_50k', (-0.09, 0.11, 0.31)),
    '250k': (250e3, 625.0, 'sub_250k', (0.03, 0.24, 0.43), 'sub_100k', (0.00, 0.20, 0.40)),
    '1M': (1e6, 1000.0, 'sub_1000k', (0.05, 0.27, 0.47), 'sub_500k', (0.05, 0.27, 0.47)),
}
POLICY = {0.214: dict(c=0.40, m=0.17), 0.12: dict(c=0.25, m=0.095)}   # cash kept at IBKR / mean margin, % NAV
LAB = ('pess', 'central', 'opt')


def diy(tier, vol, sr, estr=ESTR, block=126, shocks=None, c=None, m=None, fixed=None, carry_value=0.0, years=YEARS):
    K0, fx, s21, _, s12, _ = TIERS[tier]
    pol = POLICY[vol]
    z = Z(shocks or (s21 if vol > 0.2 else s12), block)
    return M.sim_diy(z, sr, vol, K0, estr, years=years, mode='ibkr', fixed_usd=fx if fixed is None else fixed,
                     c=pol['c'] if c is None else c, m=pol['m'] if m is None else m, tax_mode='split', carry_value=carry_value,
                     infl=INFL)


for tier, (K0, fx, s21, g21, s12, g12) in TIERS.items():
    for vol, grid in ((0.214, g21), (0.12, g12)):
        for lab, sr in zip(LAB, grid):
            res = diy(tier, vol, sr)
            pre = M.sim_diy(Z(s21 if vol > 0.2 else s12), sr, vol, K0, ESTR, years=YEARS, fixed_usd=fx, tax_mode='split',
                            pfu=0.0, **POLICY[vol])
            for y in YEARS:
                res[y]['pre_med'] = pre[y]['at_med']; res[y]['pre_p10'] = pre[y]['at_p10']; res[y]['pre_p90'] = pre[y]['at_p90']
            add(res, case='a/b DIY v5 subset', tier=tier, vol=vol, scen=lab, sr=sr,
                assumptions='shocks=%s, fixed $%d/yr, IBKR cash %d%%/margin %.1f%%' % (s21 if vol > 0.2 else s12, fx, POLICY[vol]['c'] * 100, POLICY[vol]['m'] * 100))
print('DIY done', round(time.time() - t0, 1))

# ---------------- (c) UCITS trend ETF (iMGP DBi R EUR HP, LU3359622902, TER 0.75 % inside SR) ----------------
for lab, sr in zip(LAB, (0.10, 0.25, 0.40)):
    add(M.sim_etf(Z('dbi_rep_ex'), sr, 0.12, ESTR, years=YEARS, infl=INFL), case='c UCITS trend ETF', tier='any', vol=0.12,
        scen=lab, sr=sr, assumptions='DBi-replica shocks 2000-26, EUR-hedged, accumulating, PFU at exit')

# ---------------- (d) cash ----------------
for y in YEARS:
    for lab, e in (('estr-0.5', ESTR - 0.005), ('central', ESTR), ('estr+0.5', ESTR + 0.005)):
        w = M.cash_benchmark(e, y); cagr = w ** (1 / y) - 1
        rows.append(dict(case='d EUR cash (XEON-type MMF, acc., PFU at exit)', tier='any', vol=0.0, scen=lab, sr=np.nan, horizon=y,
                         at_med=cagr, at_p10=cagr, at_p90=cagr, real_med=(1 + cagr) / (1 + INFL) - 1, P_nom_loss=0.0, P_below_cash=np.nan,
                         dd_med=0.0, assumptions='ESTR %.2f%%, TER 0.10%%' % (e * 100)))
        ann = (e - 0.001) * (1 - M.PFU)
        rows.append(dict(case='d EUR cash taxed yearly (interest, RCM)', tier='any', vol=0.0, scen=lab, sr=np.nan, horizon=y,
                         at_med=ann, real_med=(1 + ann) / (1 + INFL) - 1, P_nom_loss=0.0, assumptions='ESTR %.2f%% x 0.686' % (e * 100)))
    rows.append(dict(case='d Livret A (tax-free, capped EUR 22,950)', tier='any', vol=0.0, scen='1.7% since 2026-08-01', horizon=y,
                     at_med=0.017, real_med=1.017 / 1.02 - 1, P_nom_loss=0.0, assumptions='info only'))

# ---------------- (e) equities, 70/30 mix ----------------
EQ = {'pess': 0.035, 'central': 0.060, 'opt': 0.080}   # target median pre-tax CAGR in EUR, net of ETF fee
VOL_E = {'sp': (0.195, 'eq_sp_eur'), 'nq': (0.25, 'eq_nq_eur')}   # daily-annualised input; effective 1-10y vol ~16.5-17 % (sp), ~23-25 % (nq)


def calib(zname, vol, target):
    mu = target + vol ** 2 / 2
    for _ in range(3):
        r = M.sim_etf(Z(zname), 0.0, vol, 0.0, years=(10,), tax=0.0, total_mu=mu)[10]['pre_med']
        mu += target - r
    return mu


MU = {(k, lab): calib(VOL_E[k][1], VOL_E[k][0], g) for k in VOL_E for lab, g in EQ.items()}
print('equity calibration (arith mu):', {k: round(v, 4) for k, v in MU.items()})
TR = {'pess': 0.10, 'central': 0.25, 'opt': 0.40}
for k, (ve, zn) in VOL_E.items():
    for lab, g in EQ.items():
        add(M.sim_etf(Z(zn), 0.0, ve, ESTR, years=YEARS, total_mu=MU[(k, lab)], infl=INFL), case='e 100%% equities CTO (%s)' % k,
            tier='any', vol=ve, scen=lab, sr=np.nan, assumptions='geo target %.1f%%, arith %.2f%%' % (g * 100, MU[(k, lab)] * 100))
        add(M.sim_etf(Z(zn), 0.0, ve, ESTR, years=YEARS, total_mu=MU[(k, lab)], tax=0.186, infl=INFL), case='e 100%% equities PEA (%s)' % k,
            tier='any', vol=ve, scen=lab, sr=np.nan, assumptions='PEA: 18.6 %% PS only after 5 y')
    combos = [('pess', 'pess'), ('central', 'central'), ('opt', 'opt'), ('pess', 'central'), ('central', 'pess'), ('central', 'opt')]
    for el, tl in combos:
        for reb, te, name in ((True, M.PFU, '70/30 CTO, annual rebalance'), (False, M.PFU, '70/30 CTO, no rebalance'),
                              (False, 0.186, '70/30 equities in PEA + ETF in CTO, no rebalance')):
            add(M.sim_mix(Z(zn), Z('dbi_rep_ex'), MU[(k, el)], ve, TR[tl], 0.12, ESTR, w_e=0.7, years=YEARS, rebalance=reb,
                          tax_e=te, infl=INFL), case='e %s (%s)' % (name, k), tier='any', vol=np.nan, scen='eq %s / trend %s' % (el, tl),
                sr=TR[tl], assumptions='equity geo %.1f%%, trend ETF SR %.2f @12%%' % (EQ[el] * 100, TR[tl]))
print('mix done', round(time.time() - t0, 1))

F = pd.DataFrame(rows)
F.to_csv('final_table.csv', index=False)

# ---------------- reconciliation waterfall (10 y, 21.4 % vol, central) ----------------
rec = []
CR = {'100k': (0.005, 0.0093), '250k': (0.0025, 0.0079), '1M': (0.001, 0.0072)}
for tier, (K0, fx, s21, g21, _, _) in TIERS.items():
    sr = g21[1]
    steps = [
        ('0 critic replication (v5_84 shocks, ESTR 2.3, const drag, one tax pool)',
         M.sim_diy(Z('v5_84'), sr, 0.214, 1.0, 0.023, years=(10,), mode='const', const_drag=sum(CR[tier]), tax_mode='pool')),
        ('1 ESTR 2.3 -> 2.4', M.sim_diy(Z('v5_84'), sr, 0.214, 1.0, 0.024, years=(10,), mode='const', const_drag=sum(CR[tier]), tax_mode='pool')),
        ('2 own-subset shocks instead of v5_84', M.sim_diy(Z(s21), sr, 0.214, 1.0, 0.024, years=(10,), mode='const', const_drag=sum(CR[tier]), tax_mode='pool')),
        ('3 IBKR rule path-dependent (NAV pro-rata, 10k), fixed $ not deductible, interest taxed w/o offset; critic cash policy 50%/23%',
         diy(tier, 0.214, sr, c=0.50, m=0.23, years=(10,))),
        ('4 subset cash policy 40% at IBKR / margin 17% (=final)', diy(tier, 0.214, sr, years=(10,))),
    ]
    for name, r in steps:
        r = r[10]
        rec.append(dict(tier=tier, step=name, at_med=r['at_med'], at_p10=r['at_p10'], at_p90=r['at_p90'], P_nom_loss=r['P_nom_loss'],
                        P_below_cash=r['P_below_cash'], dd_med=r['dd_med'], dd_p10=r['dd_p10']))
R = pd.DataFrame(rec); R.to_csv('reconcile.csv', index=False)
print('reconcile done', round(time.time() - t0, 1))

# ---------------- sensitivities (10 y) ----------------
sen = []


def s_add(case, tier, what, r):
    r = r[10]
    sen.append(dict(case=case, tier=tier, sensitivity=what, at_med=r['at_med'], at_p10=r['at_p10'], at_p90=r['at_p90'],
                    real_med=r['real_med'], P_nom_loss=r['P_nom_loss'], P_below_cash=r['P_below_cash'], dd_med=r['dd_med'], dd_p10=r['dd_p10']))


for tier, (K0, fx, s21, g21, _, _) in TIERS.items():
    sr = g21[1]
    s_add('DIY 21%', tier, 'base', diy(tier, 0.214, sr, years=(10,)))
    s_add('DIY 21%', tier, 'SR -0.10', diy(tier, 0.214, sr - 0.1, years=(10,)))
    s_add('DIY 21%', tier, 'SR +0.10', diy(tier, 0.214, sr + 0.1, years=(10,)))
    s_add('DIY 21%', tier, 'ESTR -0.5 (1.9 %)', diy(tier, 0.214, sr, estr=ESTR - 0.005, years=(10,)))
    s_add('DIY 21%', tier, 'ESTR +0.5 (2.9 %)', diy(tier, 0.214, sr, estr=ESTR + 0.005, years=(10,)))
    s_add('DIY 21%', tier, 'block 63', diy(tier, 0.214, sr, block=63, years=(10,)))
    s_add('DIY 21%', tier, 'block 252', diy(tier, 0.214, sr, block=252, years=(10,)))
    s_add('DIY 21%', tier, 'v5_84 shocks (critic)', diy(tier, 0.214, sr, shocks='v5_84', years=(10,)))
    s_add('DIY 21%', tier, 'v5 ex CUA1/SCO1 shocks', diy(tier, 0.214, sr, shocks='v5_82_exCUA_SCO', years=(10,)))
    s_add('DIY 21%', tier, 'loss carry-forward used vs other gains (QQQ)', diy(tier, 0.214, sr, carry_value=1.0, years=(10,)))
    s_add('DIY 21%', tier, 'fixed costs minimal $144', diy(tier, 0.214, sr, fixed=144.0, years=(10,)))
    s_add('DIY 21%', tier, 'fixed costs lean $1005', diy(tier, 0.214, sr, fixed=1005.0, years=(10,)))
    s_add('DIY 21%', tier, 'time cost 80 h x EUR 25 (~$2,340/yr) added', diy(tier, 0.214, sr, fixed=fx + 2340.0, years=(10,)))
    s_add('DIY 21%', tier, 'lag: SR -0.05 more', diy(tier, 0.214, sr - 0.05, years=(10,)))
for what, kw in (('base', {}), ('SR -0.10', dict(sr=0.15)), ('SR +0.10', dict(sr=0.35)), ('ESTR -0.5', dict(estr=ESTR - 0.005)),
                 ('ESTR +0.5', dict(estr=ESTR + 0.005)), ('SG Trend shocks', dict(z='sgtrend_ex')), ('vol 9 % (SR 0.25)', dict(vol=0.09)),
                 ('block 252', dict(block=252))):
    s_add('UCITS ETF', 'any', what, M.sim_etf(Z(kw.get('z', 'dbi_rep_ex'), kw.get('block', 126)), kw.get('sr', 0.25), kw.get('vol', 0.12),
                                               kw.get('estr', ESTR), years=(10,), infl=INFL))
for what, kw in (('base', {}), ('ESTR -0.5', dict(estr=ESTR - 0.005)), ('ESTR +0.5', dict(estr=ESTR + 0.005)),
                 ('trend SR -0.10', dict(sr=0.15)), ('trend SR +0.10', dict(sr=0.35)), ('block 252', dict(block=252))):
    s_add('70/30 CTO rebal (sp)', 'any', what, M.sim_mix(Z('eq_sp_eur', kw.get('block', 126)), Z('dbi_rep_ex', kw.get('block', 126)),
                                                          MU[('sp', 'central')], 0.195, kw.get('sr', 0.25), 0.12, kw.get('estr', ESTR), years=(10,)))
s_add('100% equities CTO (sp)', 'any', 'block 252', M.sim_etf(Z('eq_sp_eur', 252), 0.0, 0.195, ESTR, years=(10,), total_mu=MU[('sp', 'central')]))
for ve in (0.18, 0.215):
    mu = calib('eq_sp_eur', ve, 0.06)
    s_add('100% equities CTO (sp)', 'any', 'equity input vol %.1f%%' % (ve * 100), M.sim_etf(Z('eq_sp_eur'), 0.0, ve, ESTR, years=(10,), total_mu=mu))
    s_add('70/30 CTO rebal (sp)', 'any', 'equity input vol %.1f%%' % (ve * 100), M.sim_mix(Z('eq_sp_eur'), Z('dbi_rep_ex'), mu, ve, 0.25, 0.12, ESTR, years=(10,)))
for w in (0.8, 0.6, 0.5):
    s_add('%d/%d CTO rebal (sp)' % (w * 100, 100 - w * 100), 'any', 'weight', M.sim_mix(Z('eq_sp_eur'), Z('dbi_rep_ex'), MU[('sp', 'central')], 0.195, 0.25, 0.12, ESTR, w_e=w, years=(10,)))
G = pd.DataFrame(sen); G.to_csv('sensitivity.csv', index=False)
print('sensitivity done', round(time.time() - t0, 1))
