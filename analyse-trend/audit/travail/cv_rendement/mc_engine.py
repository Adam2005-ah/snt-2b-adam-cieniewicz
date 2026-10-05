"""Cross-check Monte Carlo engine (independent of critic/mc_tiers.py and rendement/05_mc.py).

Conventions
- AN = 261 obs/yr (the futures calendar of the data set). Sharpe = arithmetic mean excess / daily-annualised vol.
- Shocks: historical daily excess returns, de-meaned and scaled to unit variance, resampled with a circular stationary
  bootstrap (geometric block length, mean BLOCK days). Each path then gets  r_t = SR*vol/AN + z_t*vol/sqrt(AN)
  (excess over EUR cash), so the arithmetic forward Sharpe is exactly the scenario SR in expectation.
- Wealth is in "USD-equivalent" units so IBKR's USD 10k / USD 100k thresholds apply directly; all returns are EUR returns
  (hedged trend sleeves earn EUR cash + excess; the futures account is margined in EUR cash).
- French tax (2026): PFU 31.4 % (12.8 IR + 18.6 PS).
  * futures P&L (art. 150 ter) and money-market ETF gains (150-0 A) are one 'same-nature' pool, realised every year
    (rolls realise P&L), net losses carried forward 10 y (horizon <= 10 y, so no expiry inside the simulation);
  * interest paid by IBKR (RCM) is taxed every year with NO offset against trading losses;
  * fixed costs (data, VPS) are NOT deductible under the PFU;
  * accumulating UCITS ETFs: tax only on the gain at sale (PMP cost basis), loss carry-forward 10 y.
  * unused loss carry-forward at the horizon is worth 0 in the central case (no other gains assumed);
    flag carry_value=1 values it at 31.4 % (investor can offset it against e.g. QQQ gains - same nature per BOFiP).
"""
import numpy as np

AN = 261
PFU = 0.314


def sb_idx(L, npath, n, block, rng):
    """Circular stationary bootstrap index matrix (npath x n)."""
    p = 1.0 / block
    idx = np.empty((npath, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, L, npath)
    new = rng.random((npath, n)) < p
    starts = rng.integers(0, L, (npath, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % L)
    return idx


def standardise(x):
    x = np.asarray(x, dtype=float)
    return ((x - x.mean()) / x.std()).astype(np.float32)


def maxdd(path):
    return (path / np.maximum.accumulate(path, axis=1) - 1).min(axis=1)


def summarise(W_end, W0, years, path, cash_end, infl=0.02, extra=None):
    """W_end: after-tax terminal wealth; path: account value path (npath x n+1) used for drawdowns."""
    cagr = (np.maximum(W_end, 1e-9) / W0) ** (1 / years) - 1
    real = (1 + cagr) / (1 + infl) - 1
    n = path.shape[1] - 1
    defl = (1 + infl) ** (-np.arange(n + 1) / AN)
    dd_nom = maxdd(path)
    dd_real = maxdd(path * defl[None, :])
    cash_cagr = (cash_end / W0) ** (1 / years) - 1
    out = dict(
        at_med=np.median(cagr), at_p10=np.percentile(cagr, 10), at_p90=np.percentile(cagr, 90), at_mean_wealth=(np.mean(W_end / W0)) ** (1 / years) - 1,
        real_med=np.median(real), real_p10=np.percentile(real, 10), real_p90=np.percentile(real, 90),
        P_nom_loss=np.mean(W_end < W0), P_real_loss=np.mean(W_end * (1 + infl) ** (-years) < W0),
        P_below_cash=np.mean(W_end < cash_end), cash_at=cash_cagr,
        dd_med=np.median(dd_nom), dd_p10=np.percentile(dd_nom, 10),
        ddr_med=np.median(dd_real), ddr_p10=np.percentile(dd_real, 10),
    )
    if extra:
        out.update(extra)
    return out


def cash_benchmark(estr, years, ter=0.001, tax=PFU):
    """After-tax terminal wealth of 1 in an accumulating EUR money-market ETF (XEON-type), taxed at sale."""
    g = (1 + estr - ter) ** years
    return 1 + (g - 1) * (1 - tax)


def sim_diy(z, sr, vol, K0, estr, years=(5, 10), mode='ibkr', fixed_usd=500.0, c=0.40, m=0.17,
            const_drag=0.0, tax_mode='split', carry_value=0.0, ib_spread=0.005, mmf_cost=0.0012,
            abandon_level=0.60, infl=0.02, pfu=PFU):
    """DIY futures account, daily loop.
    mode='ibkr': interest = IBKR rule on (c-m)*W above USD 10k, x min(W/100k,1), rate estr-0.5 %; (1-c)*W in a
                 money-market ETF at estr - mmf_cost; fixed_usd per year (not deductible).
    mode='const': critic's convention: cash earns estr on all W, minus const_drag (% of W/yr, fixed+shortfall),
                 everything (incl. interest, net of drag) in one tax pool.
    Returns dict horizon -> summary (after-tax)."""
    npath, n = z.shape
    W = np.full(npath, float(K0))
    path = np.empty((npath, n + 1), dtype=np.float64)
    path[:, 0] = W
    mu, sd = sr * vol / AN, vol / np.sqrt(AN)
    fut_y = np.zeros(npath); rcm_y = np.zeros(npath); pool_extra_y = np.zeros(npath)
    carry = np.zeros(npath)
    snaps = {}
    touched = np.zeros(npath, dtype=bool)
    for t in range(n):
        Wp = np.maximum(W, 0.0)
        pnl = Wp * (mu + z[:, t] * sd)
        if mode == 'ibkr':
            ib = np.maximum((c - m) * Wp - 10e3, 0.0) * max(estr - ib_spread, 0.0) / AN * np.minimum(Wp / 100e3, 1.0)
            mmf = (1 - c) * Wp * (estr - mmf_cost) / AN
            W = W + pnl + ib + mmf - fixed_usd / AN
            fut_y += pnl; rcm_y += ib; pool_extra_y += mmf
        else:
            cash = Wp * (estr - const_drag) / AN
            W = W + pnl + cash
            fut_y += pnl; pool_extra_y += cash
        touched |= W < abandon_level * K0
        if (t + 1) % AN == 0:
            if tax_mode == 'split':
                g = fut_y + pool_extra_y
                tax = pfu * np.maximum(rcm_y, 0.0)
            else:  # critic: one pool including interest
                g = fut_y + pool_extra_y + rcm_y
                tax = np.zeros(npath)
            pos = np.maximum(g, 0.0)
            use = np.minimum(carry, pos)
            carry = carry - use + np.maximum(-g, 0.0)
            tax = tax + pfu * (pos - use)
            W = W - tax
            fut_y[:] = 0; rcm_y[:] = 0; pool_extra_y[:] = 0
            y = (t + 1) // AN
            if y in years:
                snaps[y] = (W.copy() + carry_value * pfu * carry, touched.copy())
        path[:, t + 1] = np.maximum(W, 1e-6)
    out = {}
    for y in years:
        Wend, touch = snaps[y]
        p = path[:, :y * AN + 1]
        cash_end = K0 * cash_benchmark(estr, y)
        s = summarise(Wend, K0, y, p, cash_end, infl,
                      extra=dict(P_touch_60pct=np.mean(touch)))
        # pre-tax CAGR comparable to critic: rebuild pre-tax by adding back taxes is path-dependent; report pre-tax
        out[y] = s
    return out


def sim_etf(z, sr, vol, estr, years=(5, 10), tax=PFU, carry_value=0.0, infl=0.02, total_mu=None):
    """Accumulating UCITS fund, EUR (hedged): daily r = estr/AN + sr*vol/AN + z*vol/sqrt(AN); tax only at sale.
    total_mu: if given, arithmetic total annual return (used for equities) instead of estr + sr*vol."""
    npath, n = z.shape
    mu = (estr + sr * vol) if total_mu is None else total_mu
    r = mu / AN + z.astype(np.float64) * (vol / np.sqrt(AN))
    path = np.concatenate([np.ones((npath, 1)), np.cumprod(1 + r, axis=1)], axis=1)
    out = {}
    for y in years:
        Wpre = path[:, y * AN]
        gain = Wpre - 1
        Wend = np.where(gain > 0, 1 + gain * (1 - tax), 1 + gain - carry_value * tax * gain)
        cash_end = cash_benchmark(estr, y)
        s = summarise(Wend, 1.0, y, path[:, :y * AN + 1], cash_end, infl,
                      extra=dict(pre_med=np.median(Wpre ** (1 / y) - 1), pre_p10=np.percentile(Wpre ** (1 / y) - 1, 10),
                                 pre_p90=np.percentile(Wpre ** (1 / y) - 1, 90)))
        out[y] = s
    return out


def sim_mix(ze, zt, mu_e, vol_e, sr_t, vol_t, estr, w_e=0.7, years=(5, 10), rebalance=True,
            tax_e=PFU, tax_t=PFU, carry_value=0.0, infl=0.02):
    """Two accumulating ETFs (equity unhedged EUR, trend hedged EUR). Annual rebalancing to w_e (sales realise gains
    on the PMP basis, taxed at 31.4 % with 10-y carry-forward, paid at once). Exit: all sold, tax on remaining gains.
    tax_e < PFU with rebalance=False models equities held in a PEA (18.6 % social charges only, after 5 years)."""
    npath, n = ze.shape
    re = mu_e / AN + ze.astype(np.float64) * (vol_e / np.sqrt(AN))
    rt = (estr + sr_t * vol_t) / AN + zt.astype(np.float64) * (vol_t / np.sqrt(AN))
    Ve = np.full(npath, w_e); Vt = np.full(npath, 1 - w_e)
    Be = Ve.copy(); Bt = Vt.copy()
    carry = np.zeros(npath)
    path = np.empty((npath, n + 1)); path[:, 0] = 1.0
    snaps = {}
    for yr in range(n // AN):
        sl = slice(yr * AN, (yr + 1) * AN)
        ge = np.cumprod(1 + re[:, sl], axis=1); gt = np.cumprod(1 + rt[:, sl], axis=1)
        path[:, yr * AN + 1:(yr + 1) * AN + 1] = Ve[:, None] * ge + Vt[:, None] * gt
        Ve = Ve * ge[:, -1]; Vt = Vt * gt[:, -1]
        y = yr + 1
        if y in years:   # liquidation value at this horizon (before this year's rebalance)
            ge_un = Ve - Be; gt_un = Vt - Bt
            if tax_e == tax_t:
                g = ge_un + gt_un
                pos = np.maximum(g, 0); use = np.minimum(carry, pos)
                taxv = tax_e * (pos - use)
                left = carry - use + np.maximum(-g, 0)
            else:   # separate wrappers (PEA equity / CTO trend): PEA gains taxed at tax_e, CTO at tax_t with carry
                taxv = tax_e * np.maximum(ge_un, 0)
                pos = np.maximum(gt_un, 0); use = np.minimum(carry, pos)
                taxv = taxv + tax_t * (pos - use)
                left = carry - use + np.maximum(-gt_un, 0)
            snaps[y] = Ve + Vt - taxv + carry_value * tax_t * left
        if rebalance and y < max(years):
            tot = Ve + Vt
            te, tt = w_e * tot, (1 - w_e) * tot
            sell_e = np.maximum(Ve - te, 0); sell_t = np.maximum(Vt - tt, 0)
            fe = np.where(Ve > 0, sell_e / Ve, 0); ft = np.where(Vt > 0, sell_t / Vt, 0)
            g = fe * (Ve - Be) + ft * (Vt - Bt)
            Be = Be * (1 - fe) + sell_t; Bt = Bt * (1 - ft) + sell_e
            Ve = Ve - sell_e + sell_t; Vt = Vt - sell_t + sell_e
            pos = np.maximum(g, 0); use = np.minimum(carry, pos)
            carry = carry - use + np.maximum(-g, 0)
            taxv = PFU * (pos - use)
            # pay tax pro rata from both sleeves (basis unchanged per unit)
            tot = Ve + Vt
            k = 1 - taxv / tot          # units sold to pay the tax: basis scaled with the units (2nd-order gain ignored)
            Ve = Ve * k; Vt = Vt * k; Be = Be * k; Bt = Bt * k
            path[:, (yr + 1) * AN] = Ve + Vt
    out = {}
    for y in years:
        cash_end = cash_benchmark(estr, y)
        out[y] = summarise(snaps[y], 1.0, y, path[:, :y * AN + 1], cash_end, infl)
    return out
