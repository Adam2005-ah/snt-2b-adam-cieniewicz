import numpy as np, pandas as pd, core
R = core.rets
px = pd.read_csv(core.MKT + '/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
for c in ['SCO1 Comdty', 'CUA1 Comdty']:
    s = R[c].dropna()
    p = px[c].dropna()
    print(c, 'first', s.index[0].date(), 'price first/last', p.loc[s.index[0]:].iloc[[0, -1]].round(2).to_dict())
    # consistency returns vs price file
    pr = p.pct_change().reindex(s.index)
    print('  corr(ret, price pct_change)=%.4f  max abs diff=%.4g' % (s.corr(pr), (s - pr).abs().max()))
    # day-of-month profile: business-day rank within month, from end
    df = pd.DataFrame({'r': s})
    df['ym'] = df.index.to_period('M')
    df['bd_from_end'] = df.groupby('ym').cumcount(ascending=False)
    df['bd_from_start'] = df.groupby('ym').cumcount()
    g = df.groupby('bd_from_start')['r'].agg(['mean', 'std', 'count'])
    g2 = df.groupby('bd_from_end')['r'].agg(['mean', 'std', 'count'])
    print('  by bd from start (first 5):'); print((g.head(5) * [1e4, 1e4, 1]).round(1).to_string())
    print('  by bd from end (last 5):'); print((g2.head(5) * [1e4, 1e4, 1]).round(1).to_string())
    print('  std by bd from start, avg first half vs second half (bp): %.1f vs %.1f' % (g['std'].iloc[:10].mean() * 1e4, g['std'].iloc[10:20].mean() * 1e4))
    big = s[s.abs() > 4 * s.std()]
    print('  n |r|>4sd:', len(big)); print(big.round(4).to_string()[:1500])
