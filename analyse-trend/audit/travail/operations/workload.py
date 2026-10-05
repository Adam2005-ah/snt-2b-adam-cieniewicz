import sys; sys.path.insert(0, '/home/user/snt-2b-adam-cieniewicz/analyse-trend')
import pandas as pd, numpy as np, run_backtests as rb, trend
OUT='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/operations/'
rets=pd.read_pickle(OUT+'rets.pkl'); groups=pd.read_pickle(OUT+'groups.pkl')
ct=pd.read_csv(OUT+'contracts.csv').set_index('ticker')
SUB=['ES1 Index','VG1 Index','NO1 Index','HI1 Index','TU1 Comdty','TY1 Comdty','RX1 Comdty','DU1 Comdty','SFR5 Comdty',
     'EC1 Curncy','JY1 Curncy','BP1 Curncy','AD1 Curncy','CL1 Comdty','NG1 Comdty','GC1 Comdty','HG1 Comdty','C 1 Comdty','S 1 Comdty','LC1 Comdty']

def targets(rets, groups, scale=None, forecast_fn=trend.forecast_regime, risk_target=0.2):
    vol = rets.apply(trend.annual_vol)
    fc = rets.apply(lambda s: forecast_fn(s, vol[s.name]))
    live = (rets.notna().cumsum() > trend.WARMUP_DAYS) & vol.notna() & fc.notna()
    class_count = live.T.groupby(groups).transform("sum").T
    n_classes = live.T.groupby(groups).any().T.sum(axis=1)
    weights = (live / class_count.where(class_count > 0)).div(n_classes, axis=0)
    idm = live.sum(axis=1).map(trend.idm_for)
    unit = (risk_target * weights.mul(idm, axis=0) / vol).where(live)
    if scale is not None: unit = unit.mul(scale.reindex(unit.index).fillna(1.0), axis=0)
    return fc / 10 * unit, unit

def run(rets, groups):
    risk = groups.replace({'Bonds': 'Taux', 'STIR': 'Taux'})
    costs = {c: rb.COSTS[g] for c, g in groups.items()}
    roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
    v5 = trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
    base = trend.run_portfolio(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)
    realised = base["net"].ewm(span=32, min_periods=32).std() * 16
    scale = (0.2 / realised).clip(0.5, 2).shift(1)
    tgt, unit = targets(rets, risk, scale)
    return v5, tgt, unit

def contract_positions(tgt, unit, notional):
    """Carver buffering in contracts: buffer edges rounded; trade to nearest edge if outside."""
    T = (tgt / notional).values; W = (0.1 * unit.fillna(0) / notional).values
    out = np.zeros_like(T); cur = np.zeros(T.shape[1])
    for i in range(T.shape[0]):
        t = T[i]; w = W[i]
        nan = np.isnan(t)
        lo = np.round(np.where(nan, 0, t - w)); hi = np.round(np.where(nan, 0, t + w))
        cur = np.where(nan, 0, np.clip(cur, lo, hi))
        out[i] = cur
    return pd.DataFrame(out, index=tgt.index, columns=tgt.columns)

def analyse(label, rets, groups, start='2016-07-01', end='2026-07-10', tiers=(25e3,50e3,1e5,2.5e5,5e5,1e6,2e6,5e6)):
    v5, tgt, unit = run(rets, groups)
    pos = v5['positions']
    rows = []
    # fractional (infinite capital) changes
    p = pos.loc[start:end]
    ch = (p.diff().abs() > 1e-12).sum(axis=1)
    rows.append(dict(universe=label, capital='infinite (fractional)', markets_ever_held=int((p.abs()>0).any().sum()),
                     avg_markets_held=(p.abs()>0).sum(axis=1).mean(), orders_per_day=ch.mean(), orders_per_week=ch.mean()*5,
                     p95_orders_day=ch.quantile(.95), max_orders_day=ch.max(), days_with_order=(ch>0).mean(),
                     rolls_per_year=float(((p.abs()>0).mean()*groups.map(rb.ROLLS_PER_YEAR)).sum())))
    # index for notional path
    idx = (1 + rets.fillna(0)).cumprod()
    for K in tiers:
        notional = ct.loc[rets.columns, 'notional_usd_2026_07_10'] * idx.div(idx.loc['2026-07-10'])
        # equity of strategy: positions are fractions of current capital; use constant K (rebased) for simplicity
        n = contract_positions(tgt.loc[start:end] * K, unit.loc[start:end] * K, notional.loc[start:end])
        dn = n.diff().abs().fillna(0)
        orders = (dn > 0).sum(axis=1)
        held = (n != 0)
        avg_abs_target = (tgt.loc[start:end].abs() * K / notional.loc[start:end]).mean()
        rows.append(dict(universe=label, capital=K, markets_ever_held=int(held.any().sum()), avg_markets_held=held.sum(axis=1).mean(),
                         markets_avg_target_ge1=int((avg_abs_target >= 1).sum()), markets_avg_target_ge4=int((avg_abs_target >= 4).sum()),
                         orders_per_day=orders.mean(), orders_per_week=orders.mean()*5, p95_orders_day=orders.quantile(.95), max_orders_day=orders.max(),
                         days_with_order=(orders>0).mean(), contracts_traded_per_year=dn.sum().sum()/((n.index[-1]-n.index[0]).days/365.25),
                         rolls_per_year=float((held.mean()*groups.map(rb.ROLLS_PER_YEAR)).sum()),
                         roll_contracts_per_year=float((n.abs().mean()*groups.map(rb.ROLLS_PER_YEAR)).sum())))
        # tracking: pnl with rounded contracts vs fractional (gross, same period)
        frac = (pos.loc[start:end].shift(1) * rets.loc[start:end].fillna(0)).sum(axis=1)
        rnd = ((n * notional.loc[start:end] / K).shift(1) * rets.loc[start:end].fillna(0)).sum(axis=1)
        rows[-1]['tracking_corr_vs_fractional'] = frac.corr(rnd)
        rows[-1]['sharpe_rounded_gross'] = rnd.mean()/rnd.std()*16
        rows[-1]['sharpe_fractional_gross'] = frac.mean()/frac.std()*16
        rows[-1]['vol_rounded'] = rnd.std()*16; rows[-1]['vol_fractional'] = frac.std()*16
        rows[-1]['tracking_error_ann'] = (rnd-frac).std()*16
    return pd.DataFrame(rows), v5

if __name__ == '__main__':
    a, v5all = analyse('84 markets', rets, groups)
    b, v5sub = analyse('20-market subset', rets[SUB], groups[SUB])
    res = pd.concat([a, b]); res.to_csv(OUT+'workload_by_capital.csv', index=False)
    pd.set_option('display.width', 250); print(res.round(3).to_string())
    # subset performance
    for lab, v in [('84', v5all), ('20', v5sub)]:
        for s in ['1990', '2000', '2010', '2023']:
            d = v['net'].loc[s:]; print(lab, s, 'sharpe', round(d.mean()/d.std()*16, 3), 'vol', round(d.std()*16, 3), 'gross mean', round(v['positions'].loc[s:].abs().sum(axis=1).mean(), 2))
    v5sub['net'].to_pickle(OUT+'v5sub_net.pkl'); v5sub['positions'].to_pickle(OUT+'v5sub_positions.pkl')
