"""Independent pre-tax re-implementation (verifier). Shares ONLY series.pkl and the published parameter values with
cv_avant_impot; the bootstrap construction, simulators and summary statistics are written from scratch here.
No tax of any kind exists in this file (there is no tax parameter at all).

Mode 'same'  : reproduces the RNG draw order of the audit (seed 20261005, blocks 63/126/252 on D90 then J00) but builds the
               index matrix with a vectorised cumulative construction (not the audit's loop) -> should give identical shocks.
Mode 'fresh' : different seeds and N paths, to measure how lucky the audit's seed is.
"""
import sys, time, numpy as np, pandas as pd

AN = 261
SER = '/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/cv_avant_impot/series.pkl'
S = pd.read_pickle(SER)
D90 = S.loc['1990-01-01':'2026-07-10']
J00 = S.loc['2000-03-29':'2026-07-10', ['dbi_rep_ex', 'sgtrend_ex', 'eq_sp_eur', 'eq_nq_eur']].dropna()


def boot_idx(L, npath, n, block, rng):
    """Stationary circular bootstrap, vectorised: position = start of current block + steps since block start (mod L)."""
    first = rng.integers(0, L, npath)
    newb = rng.random((npath, n)) < 1.0 / block
    starts = rng.integers(0, L, (npath, n))
    newb[:, 0] = True
    st = np.where(newb, starts, 0)
    st[:, 0] = first
    t = np.arange(n)[None, :]
    last_new = np.maximum.accumulate(np.where(newb, t, 0), axis=1)          # time index of the current block start
    s0 = np.take_along_axis(st, last_new, axis=1)
    return ((s0 + (t - last_new)) % L).astype(np.int64)


def zstd(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


class Shocks:
    def __init__(self, mode='same', seed=20261005, npath=5000, n=10 * AN):
        rng = np.random.default_rng(seed)
        if mode == 'same':     # consume the RNG exactly in the audit's order, keep only block 126
            for b in (63, 126, 252):
                ix = boot_idx(len(D90), npath, n, b, rng)
                if b == 126: self.i90 = ix
            for b in (63, 126, 252):
                ix = boot_idx(len(J00), npath, n, b, rng)
                if b == 126: self.i00 = ix
        else:
            self.i90 = boot_idx(len(D90), npath, n, 126, rng)
            self.i00 = boot_idx(len(J00), npath, n, 126, rng)

    def z(self, name):
        if name in J00.columns:
            return zstd(J00[name].values)[self.i00]
        return zstd(D90[name].values)[self.i90]


def mdd(p):
    return (p / np.maximum.accumulate(p, axis=1) - 1).min(axis=1)


def stats(Wend, W0, y, path, estr, extra=None):
    cagr = (np.maximum(Wend, 1e-9) / W0) ** (1 / y) - 1
    cash = (1 + estr - 0.001) ** y                  # untaxed MMF at ESTR-0.10 %
    dd = mdd(path)
    o = dict(med=np.median(cagr), p10=np.percentile(cagr, 10), p90=np.percentile(cagr, 90),
             real_med=(1 + np.median(cagr)) / 1.02 - 1, P_loss=np.mean(Wend < W0), P_below_cash=np.mean(Wend / W0 < cash),
             dd_med=np.median(dd), dd_p10=np.percentile(dd, 10), mean_log=np.mean(np.log(np.maximum(Wend, 1e-9) / W0)) / y)
    if extra: o.update(extra)
    return o


def diy(z, sr, vol, K0, estr, fixed, c, m, years=(5, 10)):
    """Futures account. Daily: P&L on NAV; IBKR interest on (c-m)*NAV above 10k at ESTR-0.5 %, x min(NAV/100k,1);
    the rest (1-c) in an MMF at ESTR-0.12 %; fixed cost spread daily. NO tax."""
    npath, n = z.shape
    W = np.full(npath, float(K0)); P = np.empty((npath, n + 1)); P[:, 0] = K0
    touch = np.zeros(npath, bool); out = {}
    a, b = sr * vol / AN, vol / np.sqrt(AN)
    rib, rmm = max(estr - 0.005, 0) / AN, (estr - 0.0012) / AN
    for t in range(n):
        Wp = np.clip(W, 0, None)
        W = W + Wp * (a + b * z[:, t]) + np.clip((c - m) * Wp - 1e4, 0, None) * rib * np.clip(Wp / 1e5, None, 1) \
            + (1 - c) * Wp * rmm - fixed / AN
        touch |= W < 0.6 * K0
        P[:, t + 1] = np.clip(W, 1e-6, None)
        if (t + 1) % AN == 0 and (t + 1) // AN in years:
            y = (t + 1) // AN
            out[y] = stats(W.copy(), K0, y, P[:, :t + 2], estr, dict(P_touch60=np.mean(touch)))
    return out


def fund(z, mu, vol, estr, years=(5, 10)):
    r = mu / AN + z * (vol / np.sqrt(AN))
    P = np.hstack([np.ones((z.shape[0], 1)), np.cumprod(1 + r, axis=1)])
    return {y: stats(P[:, y * AN], 1.0, y, P[:, :y * AN + 1], estr) for y in years}, P


def mix(ze, zt, mu_e, vol_e, mu_t, vol_t, estr, w=0.7, years=(5, 10)):
    """Daily paths of two funds, rebalanced to w/1-w at each anniversary (no tax, no cost)."""
    npath, n = ze.shape
    re = mu_e / AN + ze * (vol_e / np.sqrt(AN)); rt = mu_t / AN + zt * (vol_t / np.sqrt(AN))
    P = np.empty((npath, n + 1)); P[:, 0] = 1
    tot = np.ones(npath)
    for k in range(n // AN):
        s = slice(k * AN, (k + 1) * AN)
        P[:, k * AN + 1:(k + 1) * AN + 1] = tot[:, None] * (w * np.cumprod(1 + re[:, s], 1) + (1 - w) * np.cumprod(1 + rt[:, s], 1))
        tot = P[:, (k + 1) * AN].copy()
    return {y: stats(P[:, y * AN], 1.0, y, P[:, :y * AN + 1], estr) for y in years}
