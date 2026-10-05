import time, numpy as np, pandas as pd
t0=time.time()
import core
print('load+fc', round(time.time()-t0,1))
t0=time.time(); mine = core.v5(); print('fast v5', round(time.time()-t0,1))
ref_net = pd.read_pickle(core.S+'audit/operations/v5_net.pkl')
ref_pos = pd.read_pickle(core.S+'audit/operations/v5_positions.pkl')
print('max |net diff|', (mine['net']-ref_net['net']).abs().max(), 'max |pos diff|', (mine['positions']-ref_pos).abs().max().max())
print({k: round(v,3) for k,v in core.sharpes(mine['net']).items()})
print({k: round(v,3) for k,v in core.sharpes(mine['net'], an=256).items()})
