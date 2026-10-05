"""Clean FINAL table (central/pess/opt) from final_table.csv -> final_summary.csv (percent units)."""
import pandas as pd
F = pd.read_csv('final_table.csv')
keep = []
def pick(case_sub, scen=None, tier=None, vol=None):
    f = F[F.case.str.contains(case_sub, regex=False)]
    if scen is not None: f = f[f.scen == scen]
    if tier is not None: f = f[f.tier == tier]
    if vol is not None: f = f[(f.vol - vol).abs() < 1e-9]
    return f
lab = []
for tier in ('100k', '250k', '1M'):
    for vol in (0.214, 0.12):
        for sc in ('pess', 'central', 'opt'):
            lab.append(('DIY v5 %s @%d%% vol' % (tier, round(vol * 100)), pick('DIY', sc, tier, vol)))
for sc in ('pess', 'central', 'opt'):
    lab.append(('UCITS trend ETF iMGP DBi (12%)', pick('c UCITS', sc)))
for sc in ('estr-0.5', 'central', 'estr+0.5'):
    lab.append(('EUR cash MMF acc. (after tax)', pick('d EUR cash (XEON', sc)))
for sc in ('pess', 'central', 'opt'):
    lab.append(('100% US equities CTO', pick('e 100% equities CTO (sp)', sc)))
    lab.append(('100% US equities PEA', pick('e 100% equities PEA (sp)', sc)))
for sc in ('eq pess / trend pess', 'eq central / trend central', 'eq opt / trend opt', 'eq pess / trend central', 'eq central / trend pess'):
    lab.append(('70/30 eq/trend ETF CTO rebal.', pick('e 70/30 CTO, annual rebalance (sp)', sc)))
    lab.append(('70/30 eq PEA + trend ETF CTO', pick('e 70/30 equities in PEA + ETF in CTO, no rebalance (sp)', sc)))
for sc in ('central',):
    lab.append(('100% Nasdaq-type CTO (25% vol)', pick('e 100% equities CTO (nq)', sc)))
    lab.append(('70/30 Nasdaq-type/trend CTO rebal.', pick('e 70/30 CTO, annual rebalance (nq)', 'eq central / trend central')))
rows = []
for name, f in lab:
    for _, r in f.iterrows():
        rows.append({'case': name, 'scenario': r['scen'], 'fwd_SR': r['sr'], 'horizon_y': r['horizon'],
                     'pre_tax_med': r.get('pre_med'), 'after_tax_med': r['at_med'], 'after_tax_p10': r.get('at_p10'), 'after_tax_p90': r.get('at_p90'),
                     'real_after_tax_med': r['real_med'], 'real_p10': r.get('real_p10'), 'real_p90': r.get('real_p90'),
                     'P_nominal_loss': r['P_nom_loss'], 'P_below_after_tax_cash': r.get('P_below_cash'),
                     'maxDD_med_nominal': r.get('dd_med'), 'maxDD_worst_decile_nominal': r.get('dd_p10'),
                     'maxDD_med_real': r.get('ddr_med'), 'maxDD_worst_decile_real': r.get('ddr_p10'), 'P_account_below_60pct_of_start': r.get('P_touch_60pct')})
T = pd.DataFrame(rows)
num = T.columns[4:]
T[num] = (T[num].astype(float) * 100).round(1)
T.to_csv('final_summary.csv', index=False)
pd.set_option('display.width', 330); pd.set_option('display.max_rows', 300)
print(T[T.horizon_y == 10].to_string(index=False))
