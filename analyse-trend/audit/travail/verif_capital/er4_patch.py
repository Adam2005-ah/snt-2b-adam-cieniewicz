import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/simulation')
import sim
E = sim.E
subs = pickle.load(open(sim.OUT + 'subsets.pkl', 'rb'))
def tabs():
    out = {}
    for k in ['small', 'liquid']:
        tab, N, HS = sim.contract_table(k + '_mult')
        out[k + '_today'] = (tab, pd.DataFrame(np.tile(N.iloc[-1].values, (len(N), 1)), index=N.index, columns=N.columns),
                             pd.DataFrame(np.tile(HS.iloc[-1].values, (len(HS), 1)), index=HS.index, columns=HS.columns))
    return out
orig = tabs()
sim.CC.loc['ER4 Comdty', 'hs'] = 0.0025     # Carver EURIBOR-ICE spread (Bloomberg ER = ICE Euribor) instead of Eurex EURIBOR 0.0072
fixed = tabs()
rows = []
STRICT = list(sim.SP.index[sim.SP['liquid_access'].eq('OK')])
s66 = E.system(STRICT)
for C in [250e3, 500e3, 1e6, 5e6]:
    cols = subs[('sub', C, 1)]; s = E.system(cols)
    for lab, T in [('orig', orig), ('ER4 hs=0.0025', fixed)]:
        sim.TAB.update(T)
        r, _, _ = sim.evaluate('subset_ge1c', cols, C, s, 'liquid_today')
        r66, _, _ = sim.evaluate('strict66', STRICT, C, s66, 'liquid_today')
        for x in r + r66:
            x['case'] = lab; x['ER4_in'] = 'ER4 Comdty' in x['markets'] or x['markets'] == 'all 84' or x['universe']=='strict66'
            rows.append(x)
res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(res[['universe', 'case', 'capital_usd', 'period', 'ER4_in', 'sharpe_net_real', 'fee_plus_halfspread_pct_cap', 'halfspread_pct_cap']].to_string())
res.to_csv('er4_spread_sensitivity.csv', index=False)
