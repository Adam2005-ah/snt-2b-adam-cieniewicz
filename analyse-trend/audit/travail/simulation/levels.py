"""Step 1: actual (unadjusted) price level of each of the 84 markets through time, in the dataset's quote units,
and notional per contract in USD.

Why: the dataset's 'back-adjusted' prices are RATIO-adjusted (P_adj_t = P_today x prod(1+r) backwards; e.g. NG = 924
in 2000 when Henry Hub was ~2.3, wheat 9,922 c/bu vs ~250), so P_t/P_today is the excess-return index, not the price
level. For markets with big roll yield it is off by factors of 3-300. So:
 - 71 markets: daily PRICE (the contract actually held) from Carver's pysystemtrade multiple_prices_csv (local copy
   verify_trend_C/data/mult_<code>.csv, + GASOIL fetched from raw GitHub because the earlier map used GASOILINE=RBOB),
   converted to dataset units by a power-of-ten factor; pysystemtrade ends 2024-03-28, so from then to 2026-07-10 the
   level is P_adj_t x g_t with g interpolated linearly from (pst/P_adj at 2024-03-28) to 1 at 2026-07-10 (P_adj is
   anchored on the actual front price on 2026-07-10). Before the pst start: chained on the ratio path.
 - JGS1 (Nifty), HC1 (HSCEI), TWT1 (TAIEX as proxy): local price indices (mkt/international/indices_local/spliced).
 - MWE1 (spring wheat) <- CBOT wheat level path; CA1 (Euronext milling wheat, EUR) <- CBOT wheat in EUR;
   IJ1 (rapeseed, EUR) <- ICE canola in EUR; each anchored on today's own level.
 - MES1 (MSCI EM), QC1 Index (OMXS30), XP1 (SPI200), PT1 (TSX60), XM1, IR4: ratio path (no level source; flagged).
"""
import pandas as pd, numpy as np, os
S = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
V = S + 'verify_trend_C/data/'
OUT = S + 'audit/simulation/'
MAP = {"ES1 Index":"SP500","NQ1 Index":"NASDAQ","RTY1 Index":"RUSSELL","DM1 Index":"DOW","VG1 Index":"EUROSTX",
 "GX1 Index":"DAX","CF1 Index":"CAC","EO1 Index":"AEX","Z 1 Index":"FTSE100","SM1 Index":"SMI",
 "NO1 Index":"NIKKEI","TP1 Index":"TOPIX","HI1 Index":"HANG","XU1 Index":"FTSECHINAA",
 "TU1 Comdty":"US2","FV1 Comdty":"US5","TY1 Comdty":"US10","UXY1 Comdty":"US10U","US1 Comdty":"US20","WN1 Comdty":"US30",
 "DU1 Comdty":"SHATZ","OE1 Comdty":"BOBL","RX1 Comdty":"BUND","UB1 Comdty":"BUXL","G 1 Comdty":"GILT","JB1 Comdty":"JGB",
 "OAT1 Comdty":"OAT","CN1 Comdty":"CAD10","IK1 Comdty":"BTP",
 "SFR5 Comdty":"SOFR","ER4 Comdty":"EURIBOR","SFI5 Comdty":"SONIA3",
 "EC1 Curncy":"EUR","BP1 Curncy":"GBP","SF1 Curncy":"CHF","CD1 Curncy":"CAD","JY1 Curncy":"JPY","AD1 Curncy":"AUD",
 "NV1 Curncy":"NZD","PE1 Curncy":"MXP","SE1 Curncy":"SEK","NO1 Curncy":"NOK",
 "CL1 Comdty":"CRUDE_W","XB1 Comdty":"GASOILINE","CO1 Comdty":"BRENT_W","HO1 Comdty":"HEATOIL","QS1 Comdty":"GASOIL","CUA1 Comdty":"ETHANOL","NG1 Comdty":"GAS_US",
 "GC1 Comdty":"GOLD","HG1 Comdty":"COPPER","SI1 Comdty":"SILVER","PL1 Comdty":"PLAT","PA1 Comdty":"PALLAD","SCO1 Comdty":"IRON",
 "C 1 Comdty":"CORN","W 1 Comdty":"WHEAT","KW1 Comdty":"REDWHEAT","S 1 Comdty":"SOYBEAN","SM1 Comdty":"SOYMEAL","BO1 Comdty":"SOYOIL",
 "RS1 Comdty":"CANOLA","LC1 Comdty":"LIVECOW","LH1 Comdty":"LEANHOG","FC1 Comdty":"FEEDCOW","SB1 Comdty":"SUGAR11","QW1 Comdty":"SUGAR_WHITE",
 "KC1 Comdty":"COFFEE","DF1 Comdty":"ROBUSTA","CC1 Comdty":"COCOA","QC1 Comdty":"COCOA_LDN","CT1 Comdty":"COTTON2"}
TODAY = pd.Timestamp('2026-07-10')

def daily_price(code):
    f = OUT + 'mp/mult_GASOIL.csv' if code == 'GASOIL' else V + f'mult_{code}.csv'
    df = pd.read_csv(f, usecols=['DATETIME', 'PRICE'], parse_dates=['DATETIME']).dropna()
    df = df[df.PRICE > 0]
    s = df.groupby(df.DATETIME.dt.normalize())['PRICE'].last()
    return s[s.index.dayofweek < 5]

def build():
    px = pd.read_csv(S + 'mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv', index_col=0, parse_dates=True)
    px = px.loc['1985':].ffill()
    idx = px.index
    L = pd.DataFrame(index=idx, columns=px.columns, dtype=float); info = []
    for t in px.columns:
        pa = px[t]
        if t in MAP:
            p = daily_price(MAP[t]).reindex(idx).ffill(limit=5)
            end = p.last_valid_index()
            # unit factor: power of ten nearest to the back-adjusted / actual ratio over the last 60 common days
            common = p.loc[:end].dropna().index[-60:]
            ratio = (pa.reindex(common) / p.reindex(common)).median()
            k = 10 ** np.round(np.log10(ratio))
            lv = p * k
            resid = ratio / k  # carry drift 2024-03 -> 2026-07 (+ any unit oddity)
            # bridge end -> TODAY: P_adj x g, g linear from lv/pa at end to 1 at TODAY
            g_end = lv.loc[end] / pa.loc[end]
            after = idx[(idx > end)]
            w = ((after - end).days / (TODAY - end).days).values.clip(0, 1)
            lv.loc[after] = pa.loc[after].values * (g_end * (1 - w) + w)
            first = p.first_valid_index()
            before = idx[idx < first]
            lv.loc[before] = pa.loc[before] * (lv.loc[first] / pa.loc[first])
            L[t] = lv
            info.append(dict(ticker=t, source=f'pysystemtrade {MAP[t]} PRICE x{k:g}', pst_first=first.date(), pst_last=end.date(),
                             resid_ratio_at_pst_end=round(resid, 3)))
        else:
            info.append(dict(ticker=t, source='ratio path (back-adjusted P_adj)'))
            L[t] = pa
    # proxies
    loc = S + 'mkt/international/indices_local/spliced/'
    def local(name):
        d = pd.read_csv(loc + name + '.csv'); d.columns = [c.lower() for c in d.columns]
        dc = [c for c in d.columns if 'date' in c][0]; vc = [c for c in d.columns if c != dc][0]
        s = d.set_index(pd.to_datetime(d[dc]))[vc].astype(float).sort_index()
        s = s[~s.index.duplicated()]
        return s.reindex(idx, method='ffill')
    for t, name in [('JGS1 Index', 'IN_NIFTY50'), ('HC1 Index', 'CN_HSCEI'), ('TWT1 Index', 'TW_TAIEX')]:
        s = local(name)
        lv = s / s.loc[TODAY] * px[t].loc[TODAY]
        lv = lv.where(lv.notna(), px[t] * (lv.dropna().iloc[0] / px[t].loc[lv.first_valid_index()]) if lv.notna().any() else px[t])
        L[t] = lv
        [r.update(source=f'local index {name}, anchored today') for r in info if r['ticker'] == t]
    eur = L['EC1 Curncy']; cad = L['CD1 Curncy'] / 100
    for t, base, conv in [('MWE1 Comdty', 'W 1 Comdty', None), ('CA1 Comdty', 'W 1 Comdty', eur), ('IJ1 Comdty', 'RS1 Comdty', 'cad')]:
        b = L[base].copy()
        if isinstance(conv, pd.Series): b = b / conv
        elif isinstance(conv, str): b = b * cad / eur
        L[t] = b / b.loc[TODAY] * px[t].loc[TODAY]
        [r.update(source=f'proxy: {base} actual level path{" in EUR" if conv is not None else ""}, anchored today') for r in info if r['ticker'] == t]
    info = pd.DataFrame(info).set_index('ticker')
    # --- fixes for the part BEFORE the pysystemtrade start where the ratio path is clearly biased ---
    # coffee: unadjusted Yahoo front month (KC=F) before 2007-02-20
    y = pd.read_csv(S + 'mkt/commodities/commodity_futures_frontmonth_close_daily_wide.csv', index_col=0, parse_dates=True)['coffee_KC']
    y = y.reindex(idx).ffill(limit=5)
    first = pd.Timestamp(info.loc['KC1 Comdty', 'pst_first'])
    m = (idx < first) & y.notna().values
    L.loc[m, 'KC1 Comdty'] = y[m] * (L.loc[first, 'KC1 Comdty'] / y.loc[first])
    info.loc['KC1 Comdty', 'source'] += '; before 2007-02 Yahoo KC=F front month'
    # bonds whose pst history starts after 1999: before that, follow the US 10y note actual level path (TY1)
    for t in ['UXY1 Comdty', 'DU1 Comdty', 'OE1 Comdty', 'RX1 Comdty', 'UB1 Comdty', 'OAT1 Comdty', 'IK1 Comdty', 'JB1 Comdty', 'TU1 Comdty']:
        first = pd.Timestamp(info.loc[t, 'pst_first'])
        m = idx < first
        L.loc[m, t] = (L.loc[m, 'TY1 Comdty'] * (L.loc[first, t] / L.loc[first, 'TY1 Comdty'])).where(px.loc[m, t].notna())
        info.loc[t, 'source'] += f'; before {first.date()} TY1 level path'
    # MSCI EM (ICE): ratio path is erratic before 2009 -> geometric mean of HSCEI and TAIEX level paths
    a = pd.Timestamp('2009-01-02')
    prox = np.sqrt((L['HC1 Index'] / L.loc[a, 'HC1 Index']) * (L['TWT1 Index'] / L.loc[a, 'TWT1 Index']))
    m = idx < a
    L.loc[m, 'MES1 Index'] = (prox[m] * L.loc[a, 'MES1 Index']).where(px.loc[m, 'MES1 Index'].notna())
    info.loc['MES1 Index', 'source'] = 'ratio path from 2009; before: geo-mean of HSCEI & TAIEX level paths'
    return L, px, info

FXCOL = {'EUR': ('EC1 Curncy', 1), 'GBP': ('BP1 Curncy', 100), 'CHF': ('SF1 Curncy', 100), 'CAD': ('CD1 Curncy', 100),
         'JPY': ('JY1 Curncy', 1e4), 'AUD': ('AD1 Curncy', 100), 'SEK': ('SE1 Curncy', 100), 'NOK': ('NO1 Curncy', 100),
         'MXN': ('PE1 Curncy', 100)}

def fx(L, ccy):
    if ccy == 'USD': return pd.Series(1.0, index=L.index)
    if ccy == 'HKD': return pd.Series(1 / 7.8, index=L.index)
    c, k = FXCOL[ccy]
    return (L[c] / k).bfill()

if __name__ == '__main__':
    L, px, info = build()
    sp = pd.read_csv(S + 'audit/capital/specs.csv', index_col=0)
    FX = pd.DataFrame({t: fx(L, sp.loc[t, 'ccy']) for t in L.columns})
    L.to_pickle(OUT + 'levels.pkl'); FX.to_pickle(OUT + 'fx.pkl')
    chk = []
    for d in ['1999-01-04', '2000-01-03', '2008-07-01', '2010-01-04', '2020-04-20', '2024-03-28', '2026-07-10']:
        dd = pd.Timestamp(d)
        chk.append(L.loc[L.index.asof(dd)].rename(d))
    chk = pd.concat(chk, axis=1)
    chk['P_adj_2000'] = px.loc[px.index.asof(pd.Timestamp('2000-01-03'))]
    out = info.join(chk.round(4))
    out.to_csv(OUT + 'levels_check.csv')
    pd.set_option('display.width', 260); pd.set_option('display.max_rows', 100)
    print(out.drop(columns=['pst_first']).to_string())
