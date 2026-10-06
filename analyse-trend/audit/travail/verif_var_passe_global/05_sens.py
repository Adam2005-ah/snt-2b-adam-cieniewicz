"""Independent spot-check of L2 sensitivities X2 (costs x2), X3 (margin 23 %), X4 (full-day lag) for v5 and v2."""
import importlib.util, pickle
import pandas as pd
S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
OUT = S + 'audit/verif_var_passe_global/'
spec = importlib.util.spec_from_file_location('r1', OUT + '01_runs.py'); r1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1)
exec(open(OUT + '02_metrics.py').read().split('rows = []')[0])
dec = pd.read_csv(S + 'audit/var_passe_global/decomposition.csv')
rows = []
for v in ('v2', 'v5'):
    runs = {'X2 L2, coûts réels x2': r1.l2(v, cst=2 * r1.REAL, rol=2 * r1.REAL_ROLL),
            'X4 L2, exécution 1 jour': r1.l2(v, lag=2.0)}
    for st, x in runs.items():
        R[(v, 'L2')] = x
        df = build(v, 'L2')
        for p, s in {'1990': '1990-01-01', '2010': '2010-01-01', '2023-07': '2023-07-11'}.items():
            m = met(df, s)
            ref = dec[(dec.variant == VN[v]) & (dec.step == st) & (dec.period == p) & (dec.metric == 'cagr_total')]['value'].iloc[0]
            rows.append(dict(v=v, step=st, period=p, mine=m['cagr_total'], prev=ref))
    R[(v, 'L2')] = pickle.load(open(OUT + 'runs.pkl', 'rb'))['runs'][(v, 'L2')]
    df = build(v, 'L2', margin=0.23)
    for p, s in {'1990': '1990-01-01', '2010': '2010-01-01', '2023-07': '2023-07-11'}.items():
        ref = dec[(dec.variant == VN[v]) & (dec.step == 'X3 L2, marge 23 %') & (dec.period == p) & (dec.metric == 'cagr_total')]['value'].iloc[0]
        rows.append(dict(v=v, step='X3 L2, marge 23 %', period=p, mine=met(df, s)['cagr_total'], prev=ref))
t = pd.DataFrame(rows); t['diff'] = t.mine - t.prev
print(t.round(5).to_string()); t.to_csv(OUT + 'compare_sensitivities.csv', index=False, float_format='%.6f')
