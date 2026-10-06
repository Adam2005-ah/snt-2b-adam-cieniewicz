"""Wide summary of var_passe_global.csv: rows variant x level, columns metric | period (main metrics only)."""
import pandas as pd
d = pd.read_csv('var_passe_global.csv')
keep = ['cagr_total', 'cagr_excess', 'sharpe', 'vol', 'maxdd_total', 'cost_pa', 'cash_shortfall_pa']
w = d[d.metric.isin(keep)].pivot_table(index=['variant', 'level'], columns=['metric', 'period'], values='value', sort=False)
w = w.reindex(columns=keep, level=0)
w.columns = [f'{m} | {p}' for m, p in w.columns]
w.to_csv('resume_large.csv', float_format='%.4f')
print(w.shape)
