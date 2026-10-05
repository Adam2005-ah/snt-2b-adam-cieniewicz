"""Step 1: replicate critic/mc_tiers.py with my engine (const-drag mode, v5_84 shocks, ESTR 2.3 %, block 126)."""
import time, numpy as np, pandas as pd
import mc_engine as M
t0 = time.time()
D = pd.read_pickle('series.pkl').loc['1990-01-01':'2026-07-10']
NP, NY, BLOCK = 4000, 10, 126
rng = np.random.default_rng(11)
L = len(D)
idx = M.sb_idx(L, NP, NY * M.AN, BLOCK, rng)
z = M.standardise(D['v5_84'].values)[idx]
print('idx', time.time() - t0)
rows = []
tiers = {'100k': (0.005, 0.0093, (0.0, 0.20, 0.40)), '250k': (0.0025, 0.0079, (0.03, 0.24, 0.43)), '1M': (0.001, 0.0072, (0.05, 0.27, 0.47))}
for tier, (fx, sh, srs) in tiers.items():
    for lab, sr in zip(('pess', 'central', 'opt'), srs):
        for vol in (0.214, 0.12):
            at = M.sim_diy(z, sr, vol, 1.0, 0.023, years=(10,), mode='const', const_drag=fx + sh, tax_mode='pool', abandon_level=0.6)[10]
            pre = M.sim_diy(z, sr, vol, 1.0, 0.023, years=(10,), mode='const', const_drag=fx + sh, tax_mode='pool', pfu=0.0)[10]
            rows.append(dict(tier=tier, scen=lab, sr=sr, vol=vol, pre_med=pre['at_med'], at_med=at['at_med'], at_p10=at['at_p10'], at_p90=at['at_p90'],
                             P_loss=at['P_nom_loss'], dd_med=at['dd_med'], dd_p10=at['dd_p10']))
T = pd.DataFrame(rows)
C = pd.read_csv('../critic/mc_tiers.csv')
T['critic_pre_med'] = C['pre_tax_median'].values / 100
T['critic_at_med'] = C['after_tax_median'].values / 100
T['critic_P_loss'] = C['P_nominal_loss_10y'].values / 100
T['critic_dd_med'] = C['maxDD_median'].values / 100
pd.set_option('display.width', 250)
print((T.set_index(['tier', 'scen', 'sr', 'vol']) * 100).round(2).to_string())
T.to_csv('replicate_critic.csv', index=False)
print('time', time.time() - t0)
