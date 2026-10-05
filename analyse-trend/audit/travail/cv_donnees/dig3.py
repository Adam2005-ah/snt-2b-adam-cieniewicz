import numpy as np, pandas as pd, core
R = core.rets
px = pd.read_csv(core.MKT + '/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
pd.set_option('display.width', 250)
mk = ['CUA1 Comdty','SCO1 Comdty','CL1 Comdty','NG1 Comdty','HG1 Comdty','GC1 Comdty','LH1 Comdty','C 1 Comdty']
out = {}
for c in mk:
    s = R[c].dropna(); p = px[c].dropna()
    out[(c.split()[0], 'vol')] = s.groupby(s.index.year).std() * np.sqrt(261)
    out[(c.split()[0], 'px')] = p.groupby(p.index.year).last()
print(pd.concat(out, axis=1).loc[2005:].round(2).to_string())
# min back-adjusted price by market (negative/near zero -> additive adjustment trouble)
mn = px.min(); first = px.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan); last = px.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
q = pd.DataFrame({'min': mn, 'first': first, 'last': last, 'last_over_min': last / mn})
print(q.sort_values('last_over_min', ascending=False).head(15).round(3).to_string())
