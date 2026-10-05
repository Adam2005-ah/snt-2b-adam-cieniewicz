import pandas as pd, numpy as np, time
t0=time.time()
D = pd.read_pickle('daily.pkl').loc['1990-01-01':'2026-07-10']
DPY = 261                       # trading days per calendar year in this data set (9530 days / 36.5 y)
NP, HMAX, BLOCK = 6000, 10*DPY, 60
CASH = 0.030                    # USD cash assumption; EUR investor ~2.5% (ECB depo 2.50% since 16/09/2026)
rng = np.random.default_rng(20261004)
x5 = D['v5_net'].values; x2 = D['v2_net'].values; xe = D['ES1 Index'].values
n = len(x5)
def stationary_bootstrap_idx(n, npaths, h, block, rng):
    p = 1.0/block
    idx = np.empty((npaths, h), dtype=np.int32)
    idx[:,0] = rng.integers(0, n, npaths)
    new = rng.random((npaths, h)) < p
    starts = rng.integers(0, n, (npaths, h))
    for t in range(1, h):
        idx[:,t] = np.where(new[:,t], starts[:,t], (idx[:,t-1]+1) % n)
    return idx
IDX = stationary_bootstrap_idx(n, NP, HMAX, BLOCK, rng)
print('bootstrap idx built', time.time()-t0)
def standardise(x, vol):
    z = (x - x.mean())/x.std()
    return z * vol/np.sqrt(DPY)      # zero-mean daily series with target annual vol
Z5 = standardise(x5, 0.214); Z2 = standardise(x2, 0.176); ZE = standardise(xe, 0.165)
def paths(z, mu_ann):
    return z[IDX] + mu_ann/DPY          # arithmetic excess mean = mu_ann
def metrics(r_ex, h_years, cash=CASH, label=None, tax=None):
    h = int(h_years*DPY); r = r_ex[:, :h]
    tot = r + cash/DPY
    W_ex = np.prod(1+r, axis=1); W = np.cumprod(1+tot, axis=1)
    cagr_ex = W_ex**(1/h_years)-1; cagr = W[:,-1]**(1/h_years)-1
    peak = np.maximum.accumulate(W, axis=1); dd = (W/peak-1).min(axis=1)
    at_high = W >= peak
    ar = np.arange(h)
    last = np.maximum.accumulate(np.where(at_high, ar, -1), axis=1)
    uw = (ar - last).max(axis=1)/DPY
    out = {'excess CAGR p5':np.percentile(cagr_ex,5),'excess CAGR p25':np.percentile(cagr_ex,25),'excess CAGR median':np.median(cagr_ex),
           'excess CAGR p75':np.percentile(cagr_ex,75),'excess CAGR p95':np.percentile(cagr_ex,95),
           'total CAGR p5':np.percentile(cagr,5),'total CAGR median':np.median(cagr),'total CAGR mean of wealth^(1/H)':np.mean(W[:,-1])**(1/h_years)-1,
           'total CAGR p95':np.percentile(cagr,95),
           'P(lose money, nominal)':np.mean(W[:,-1]<1),'P(underperform cash)':np.mean(W_ex<1),
           'P(total CAGR<cash-5%/yr)': np.mean(cagr < cash-0.05),
           'maxDD median':np.median(dd),'maxDD p5 (worst 5%)':np.percentile(dd,5),'P(maxDD<-30%)':np.mean(dd<-0.30),'P(maxDD<-50%)':np.mean(dd<-0.50),
           'longest under water median (y)':np.median(uw),'longest under water p95 (y)':np.percentile(uw,95)}
    if tax is not None:
        # French PFU on yearly realised gains (futures P&L + interest), losses carried forward 10y (simplified: unlimited within horizon)
        yr = tot.reshape(tot.shape[0], int(h_years), DPY)
        W0 = np.ones(tot.shape[0]); carry = np.zeros(tot.shape[0])
        for y in range(int(h_years)):
            g = W0*(np.prod(1+yr[:,y,:],axis=1)-1)
            taxable = g - carry
            taxv = np.where(taxable>0, taxable*tax, 0.0)
            carry = np.where(taxable>0, 0.0, -taxable)  # remaining loss carried
            carry = np.where(g<0, carry, carry)          # (already handled)
            W0 = W0 + g - taxv
        out['after-tax total CAGR median'] = np.median(W0**(1/h_years)-1)
        out['after-tax total CAGR p5'] = np.percentile(W0**(1/h_years)-1,5)
    return out
rows=[]
SRS=[0.0,0.10,0.15,0.25,0.30,0.35,0.45,0.50,0.55]
for var,Z,vol in [('v5 (Les deux)',Z5,0.214),('v2 (base)',Z2,0.176)]:
    for sr in SRS:
        R = paths(Z, sr*vol)
        for H in [5,10]:
            m = metrics(R, H, tax=0.314); m.update(strategy=var, SR=sr, vol=vol, horizon_y=H, arith_excess=sr*vol, geo_excess_approx=sr*vol-vol**2/2)
            rows.append(m)
print('trend done', time.time()-t0)
# equity alone
for mu in [0.015,0.035,0.055]:
    R = paths(ZE, mu)
    for H in [5,10]:
        m=metrics(R,H,tax=0.314); m.update(strategy='S&P 500 futures (excess re-centred)', SR=mu/0.165, vol=0.165, horizon_y=H, arith_excess=mu, geo_excess_approx=mu-0.165**2/2); rows.append(m)
# mixes
ESC={'pess':0.015,'central':0.035,'opt':0.055}
for sr in [0.15,0.35,0.55]:
    for ek,mu in ESC.items():
        RT = paths(Z5, sr*0.214); RE = paths(ZE, mu)
        for lab, R in [('50/50 equity + v5 (capital split)', 0.5*RT+0.5*RE), ('100% equity + v5 overlay at half size', RE+0.5*RT)]:
            for H in [5,10]:
                m=metrics(R,H,tax=0.314); vv=R[:, :H*DPY].std()*np.sqrt(DPY)
                m.update(strategy=lab+f' | equity excess {mu:.1%}', SR=np.nan, vol=vv, horizon_y=H, arith_excess=R.mean()*DPY,
                         geo_excess_approx=np.nan, trend_SR=sr, equity_excess=mu); rows.append(m)
print('mix done', time.time()-t0)
M = pd.DataFrame(rows)
front=['strategy','SR','trend_SR','equity_excess','vol','horizon_y','arith_excess','geo_excess_approx']
M = M[front+[c for c in M.columns if c not in front]]
M.to_csv('monte_carlo.csv', index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
show=['strategy','SR','trend_SR','vol','horizon_y','excess CAGR p5','excess CAGR median','excess CAGR p95','total CAGR median','after-tax total CAGR median','P(lose money, nominal)','P(underperform cash)','maxDD median','maxDD p5 (worst 5%)','longest under water median (y)']
print(M[show].round(3).to_string())
corr = np.corrcoef(Z5[IDX[:200]].ravel(), ZE[IDX[:200]].ravel())[0,1]; print('bootstrap daily corr v5/eq', corr)
