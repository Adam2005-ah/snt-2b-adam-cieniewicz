"""(10) Corrected statements of the var_passe_retail summary (the CSV/pickle outputs themselves are correct)."""
import pandas as pd
A = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/'
W = pd.read_csv(A + 'var_passe_retail/var_passe_retail_wide.csv')
h = W[W.execution == 'half_day_lag'].set_index(['key', 'period']); s = W[W.execution == 'same_close'].set_index(['key', 'period'])
d = (h['sharpe'] - s['sharpe']).rename('lag_effect').reset_index()
d['variant'] = d.key.str[:2]
rng_all = d.groupby('variant').lag_effect.agg(['min', 'max'])
rng_long = d[d.period != '2023-07_2026-07'].groupby('variant').lag_effect.agg(['min', 'max'])
rows = []
for v in rng_all.index:
    rows.append(dict(item=f'half-day-lag effect on Sharpe, {v}', previous_claim={'v1': '-0.02..+0.02 (path noise)', 'v2': '-0.02..+0.02', 'v3': '-0.005..-0.018',
                     'v4': '-0.02..+0.02', 'v5': '-0.005..-0.019'}[v],
                     verified=f"all 4 windows {rng_all.loc[v, 'min']:+.3f}..{rng_all.loc[v, 'max']:+.3f}; long windows {rng_long.loc[v, 'min']:+.3f}..{rng_long.loc[v, 'max']:+.3f}"))
rows += [
    dict(item='CAGR net all fees v3_250k 2000-2026', previous_claim='12.6 %', verified='12.65 % -> 12.7 %'),
    dict(item='Published-84 CAGR total (T-bill) since 2000, v4', previous_claim='18.6 %', verified='18.65 % -> 18.7 %'),
    dict(item='margin: computed flat-k on actual contracts vs 17 % x gross-exposure shortcut', previous_claim='agree within about 1 point',
         verified='v5 84 flat-k = 16.8 % (ok) but subsets 12.4/14.2/14.7 % (2000-26), i.e. 2-5 pts below 17 %; v1 shortcut 9.8 % vs computed 16.6-18.4 % (7-9 pts); '
                  'v4 shortcut 14.8 % vs 11.8-14.1 %; cash-leg impact <= ~0.14 %/yr, computed margins are the better estimate'),
    dict(item='retail vs published comparison', previous_claim='published-84 same-close used as reference',
         verified='the gap published-84 -> retail is mostly CUA1/SCO1 removal + half-day lag: tradable-82 half-day-lag Sharpe since 2010 = '
                  'v1 -0.02, v2 0.27, v3 0.32, v4 0.36, v5 0.375 vs retail 100k/250k/1M v5 0.40/0.42/0.38 (reference_84_vs_82lag.csv)'),
]
pd.DataFrame(rows).to_csv('corrections_claims.csv', index=False)
print(pd.DataFrame(rows).to_string())
