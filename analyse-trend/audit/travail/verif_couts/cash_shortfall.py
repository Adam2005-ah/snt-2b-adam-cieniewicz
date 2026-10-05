"""Recompute cash-interest shortfall vs T-bill when commodity-segment cash (futures margin) earns 0 at IBKR.
Source: IBKR 'Margin Interest Calculations' page: 'No interest will be paid on excess funds in the commodities segment';
'interest will be paid to the securities and IBUKL segments ... while no interest will accrue on the commodity balance'."""
import pandas as pd
TIERS=[25e3,50e3,100e3,250e3,500e3,1e6,2e6,5e6]
TBILL=4.00; IB_USD=3.86-0.5; ESTR=2.42; IB_EUR=ESTR-0.5
def ib(cash,nav,r): return max(cash-10e3,0)*r/100*min(nav/100e3,1)
rows=[]
for mte in [0.23,0.30,0.40]:
  for K in TIERS:
    m=mte*K
    # A: all cash at IBKR, margin in commodity segment earns 0
    iA=ib(K-m,K,IB_USD)
    # B: 50% in T-bills (yield TBILL-0.02), 50% cash of which margin m earns 0
    iB=ib(max(0.5*K-m,0),K,IB_USD)+0.5*K*(TBILL-0.02)/100
    # prior agent (margin earning interest)
    pA=ib(K,K,IB_USD); pB=ib(0.5*K,K,IB_USD)+0.5*K*(TBILL-0.02)/100
    rows.append(dict(margin_to_equity=mte,capital=K,prior_allcash=TBILL-pA/K*100,corr_allcash=TBILL-iA/K*100,
                     prior_50tb=TBILL-pB/K*100,corr_50tb=TBILL-iB/K*100))
D=pd.DataFrame(rows).round(2); print(D.to_string())
D.to_csv('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/verif_couts/cash_shortfall_corrected.csv',index=False)
