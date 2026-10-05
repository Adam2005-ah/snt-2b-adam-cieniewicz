import pickle, pandas as pd, numpy as np
exec(open('specs_raw.py').read())
D = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
px = pd.read_csv(D+'mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
last = px.ffill().iloc[-1]; lastdate = px.apply(lambda s: s.last_valid_index())
d = pickle.load(open('v5.pkl', 'rb'))
pos5, pos2, vol, groups, names = d['v5_pos'], d['v2_pos'], d['vol'], d['groups'], d['names']
carry = pd.read_csv(D+'improve/carry/carry_inputs_info.csv', index_col=0)['code']
cfg = pd.read_csv('instrumentconfig.csv', index_col=0)
# FX (USD per 1 unit of currency) from the dataset FX futures, last row 2026-07-10
FX = {'USD': 1.0, 'EUR': last['EC1 Curncy'], 'GBP': last['BP1 Curncy']/100, 'CHF': last['SF1 Curncy']/100,
      'CAD': last['CD1 Curncy']/100, 'JPY': last['JY1 Curncy']/1e4, 'AUD': last['AD1 Curncy']/100,
      'NZD': last['NV1 Curncy']/100, 'SEK': last['SE1 Curncy']/100, 'NOK': last['NO1 Curncy']/100,
      'MXN': last['PE1 Curncy']/100, 'HKD': 1/7.80}
print({k: round(v, 6) for k, v in FX.items()})
cols = ['ticker','exchange','std_contract','std_mult','ccy','price_unit','small_contract','small_mult','small_liquidity','access','note']
sp = pd.DataFrame(SPECS, columns=cols).set_index('ticker')
sp.insert(0, 'name', names.reindex(sp.index) if isinstance(names, pd.Series) else None)
sp.insert(1, 'group', groups.reindex(sp.index))
sp['carver_code'] = carry.reindex(sp.index)
sp['carver_pointsize'] = [cfg['Pointsize'].get(c, np.nan) if isinstance(c, str) else np.nan for c in sp['carver_code']]
sp['last_price'] = last.reindex(sp.index)
sp['last_date'] = lastdate.reindex(sp.index).dt.date
sp['fx_usd'] = sp['ccy'].map(FX)
sp['std_notional_usd'] = sp['std_mult'] * sp['last_price'] * sp['fx_usd']
sp['small_notional_usd'] = sp['small_mult'] * sp['last_price'] * sp['fx_usd']
w = pos5.loc['2023-07-10':'2026-07-10'].abs()
sp['v5_med_abs_pos'] = w.median().reindex(sp.index)
sp['v5_p90_abs_pos'] = w.quantile(0.9).reindex(sp.index)
sp['v5_last_pos'] = pos5.iloc[-1].reindex(sp.index)
sp['v2_med_abs_pos'] = pos2.loc['2023-07-10':'2026-07-10'].abs().median().reindex(sp.index)
sp['mincap_1c_small'] = sp['small_notional_usd'] / sp['v5_med_abs_pos']
sp['mincap_4c_small'] = 4 * sp['mincap_1c_small']
sp['mincap_1c_std'] = sp['std_notional_usd'] / sp['v5_med_abs_pos']
sp['mincap_4c_small_v2'] = 4 * sp['small_notional_usd'] / sp['v2_med_abs_pos']
# Carver AFTS formula: 4 x mult x price x FX x sigma / (IDM x weight x tau)
risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
ncls = risk.value_counts()
sp['weight'] = [1/6/ncls[risk[t]] for t in sp.index]
sp['vol_2026_07_10'] = vol.iloc[-1].reindex(sp.index)
sp['vol_med_3y'] = vol.loc['2023-07-10':].median().reindex(sp.index)
sp['carver_mincap_4c_small'] = 4 * sp['small_notional_usd'] * sp['vol_2026_07_10'] / (2.5 * sp['weight'] * 0.20)
sp['carver_mincap_4c_small_1mkt'] = 4 * sp['small_notional_usd'] * sp['vol_2026_07_10'] / 0.20  # single market, IDM 1, weight 1
liq_ok = sp['small_liquidity'].isin(['H', 'M', 'M-H', 'L-M']) & (sp['access'] == 'OK')
sp['liquid_contract'] = np.where(liq_ok, sp['small_contract'], None); sp['liquid_mult'] = np.where(liq_ok, sp['small_mult'], np.nan)
sp['liquid_liquidity'] = np.where(liq_ok, sp['small_liquidity'], None); sp['liquid_access'] = np.where(liq_ok, 'OK', None)
for t, (c, m, l, a) in LIQUID_OVERRIDE.items():
    sp.loc[t, ['liquid_contract', 'liquid_mult', 'liquid_liquidity', 'liquid_access']] = [c, m, l, a]
sp['liquid_notional_usd'] = sp['liquid_mult'].astype(float) * sp['last_price'] * sp['fx_usd']
sp['mincap_1c_liquid'] = sp['liquid_notional_usd'] / sp['v5_med_abs_pos']
sp['mincap_4c_liquid'] = 4 * sp['mincap_1c_liquid']
def verdict(r):
    if r['access'] in ('NO', 'NO?', 'ILLIQ'): return 'drop (not accessible / illiquid)'
    base = 'KID/IBKR access to verify; ' if r['access'] in ('KID?', 'OK?') else ''
    cap = r['mincap_1c_liquid'] if pd.notna(r['mincap_1c_liquid']) else r['mincap_1c_small']
    if cap > 2e6: return base + 'too big: needs >USD 2M for 1 contract at median position'
    if cap > 5e5: return base + 'needs USD 0.5-2M for 1 contract at median'
    return base + 'feasible below USD 0.5M (1 contract at median)'
sp['retail_verdict'] = sp.apply(verdict, axis=1)
sp.to_csv('specs.csv', float_format='%.6g')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 30)
print(sp[['group','last_price','ccy','std_notional_usd','small_notional_usd','v5_med_abs_pos','mincap_1c_small','mincap_4c_small','carver_mincap_4c_small','access']].round(3).to_string())
