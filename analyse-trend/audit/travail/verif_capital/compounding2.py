import sys; sys.argv=['x']
exec(open('compounding.py').read().split('rows = []')[0])
rows=[]
for C in [50e3, 100e3]:
    cols = subs[('sub', C, 1)]; s = E.system(cols)
    for st in ['2010-01-01', '2016-01-01', '2023-07-10']:
        comp = run_equity(s, cols, C, start=st)
        fixed, _, _ = sim.evaluate('fixed', cols, C, s, 'liquid_today')
        rows.append(dict(capital=C, start=st, **comp))
print(pd.DataFrame(rows).to_string())
