import pandas as pd, numpy as np, pickle
S='/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/'
px=pd.read_csv(S+'mkt/alternatives/futures_sepp/futures84_backadjusted_prices_daily.csv',index_col=0,parse_dates=True).ffill().iloc[-1]
sp=pd.read_csv(S+'audit/capital/specs.csv',index_col=0)
d=pickle.load(open('mine.pkl','rb'))
med=d['v5'].loc['2023-07-10':'2026-07-10'].abs().median()
# FX: USD per unit, derived independently from dataset quotes (cents -> /100; JPY quote is USD/JPY*1e4)
fx={'USD':1,'EUR':px['EC1 Curncy'],'GBP':px['BP1 Curncy']/100,'JPY':px['JY1 Curncy']/1e4,'AUD':px['AD1 Curncy']/100,'CAD':px['CD1 Curncy']/100,'CHF':px['SF1 Curncy']/100}
# (ticker, contract, units per contract in exchange terms, quote unit in dataset, factor converting dataset price to quote-ccy per unit, ccy, source)
R=[
('ES1 Index','MES',5,'index pts',1,'USD','Lean cme,MES mult 5; IB config SP500_micro 5'),
('NO1 Index','OSE Nikkei 225 Micro',10,'index pts',1,'JPY','OSE micro JPY10 (jpx.co.jp 225micro); exists per N225MC feeds on GitHub'),
('VG1 Index','FSXE',1,'index pts',1,'EUR','Eurex micro EUR1 (Eurex news); IB config n/a'),
('GX1 Index','FDXS',1,'index pts',1,'EUR','Lean eurex,FDXS mult 1; Carver DAX IB mult 1'),
('JB1 Comdty','JGB std (JPY100M face)',1e6,'% of par',1,'JPY','Carver JGB pointsize 1,000,000; IB JGB mult 1e6'),
('JB1 Comdty','JGB mini (JPY10M face)',1e5,'% of par',1,'JPY','IB config JGB-mini MJ OSE mult 100,000'),
('TU1 Comdty','ZT',2000,'% of par',1,'USD','Lean ZT 2000; Carver US2 2000'),
('RX1 Comdty','FGBL',1000,'% of par',1,'EUR','Carver BUND 1000'),
('G 1 Comdty','Long Gilt',1000,'% of par',1,'GBP','Carver GILT 1000'),
('XM1 Comdty','ASX XT (value quote)',1,'AUD value',1,'AUD','dataset quotes contract value; 6% coupon 20 half-years at y=4.88% -> AUD 108.8k'),
('SFR5 Comdty','SR3',2500,'100-rate',1,'USD','Carver SOFR 2500; IB SOFR3 2500'),
('ER4 Comdty','ICE Euribor',2500,'100-rate',1,'EUR','Carver EURIBOR 2500; IB EURIBOR-ICE 2500'),
('SFI5 Comdty','ICE 3M SONIA',2500,'100-rate',1,'GBP','IB SONIA3 ICEEU 2500'),
('IR4 Comdty','ASX 90d bank bill',1e6*90/365/100/(1+0.0445*90/365)**2,'100-rate',1,'AUD','AUD1M x 90/365 / (1+y*90/365)^2 per 1.00 -> ~2,412'),
('JY1 Curncy','MJY (JPY1.25M)',1.25e6,'USD/JPY x1e4',1e-4,'USD','Lean cme,MJY 1,250,000'),
('BP1 Curncy','M6B (GBP6,250)',6250,'US cents/GBP',0.01,'USD','Lean M6B 6250'),
('SE1 Curncy','SEK (SEK2M)',2e6,'US cents/SEK',0.01,'USD','Carver SEK 2,000,000'),
('PE1 Curncy','6M (MXN500k)',5e5,'US cents/MXN',0.01,'USD','Carver MXP 500,000'),
('CL1 Comdty','MCL 100 bbl',100,'$/bbl',1,'USD','Lean MCL 100'),
('NG1 Comdty','MNG 1,000 MMBtu',1000,'$/MMBtu',1,'USD','lumibot MHNG (IBKR conid); 1,000 MMBtu per CME'),
('GC1 Comdty','1OZ',1,'$/oz',1,'USD','Lean comex,1OZ mult 1; lumibot IBKR conid'),
('HG1 Comdty','MHG 2,500 lb',2500,'US cents/lb',0.01,'USD','IB config COPPER-micro 2500'),
('PL1 Comdty','PLM Micro Platinum 10 oz',10,'$/oz',1,'USD','lumibot IBKR PLM; CME metals list "PLM - Micro Platinum Futures"; 10 troy oz'),
('PA1 Comdty','PAM Micro Palladium 10 oz',10,'$/oz',1,'USD','Lean nymex,PAM mult 10; CME metals list "PAM - Micro Palladium"'),
('C 1 Comdty','MZC 500 bu',500,'US cents/bu',0.01,'USD','lumibot IBKR MZC conid 763552571'),
('W 1 Comdty','MZW 500 bu',500,'US cents/bu',0.01,'USD','lumibot IBKR MZW'),
('KC1 Comdty','KC 37,500 lb',37500,'US cents/lb',0.01,'USD','Lean KC 37500; IB 37500 /100'),
('CC1 Comdty','CC 10 t',10,'$/t',1,'USD','Lean CC 10'),
('LC1 Comdty','LE 40,000 lb',40000,'US cents/lb',0.01,'USD','Lean LE 40000 (/100)'),
('SB1 Comdty','SB 112,000 lb',112000,'US cents/lb',0.01,'USD','Lean SB 112000'),
('RS1 Comdty','RS 20 t',20,'CAD/t',1,'CAD','IB CANOLA 20 CAD'),
('CA1 Comdty','EBM 50 t',50,'EUR/t',1,'EUR','IB MILLWHEAT MATIF 50'),
('QS1 Comdty','ICE Gasoil 100 t',100,'$/t',1,'USD','Lean ice,G 100; IB GASOIL 100'),
('SI1 Comdty','SIC 100 oz',100,'$/oz',1,'USD','CME rule filing 26-008: SIC 100 troy oz, financial, first trade 9 Feb 2026'),
]
rows=[]
for t,c,units,q,f,ccy,src in R:
    p=px[t]; notional=p*f*units*fx[ccy]
    rows.append(dict(ticker=t,contract=c,price=p,quote=q,mult_in_dataset_units=units*f,ccy=ccy,notional_usd=round(notional),
        prev_small=sp.loc[t,'small_notional_usd'],prev_std=sp.loc[t,'std_notional_usd'],med_pos=med[t],mincap_1c=round(notional/med[t]),mincap_4c=round(4*notional/med[t]),source=src))
out=pd.DataFrame(rows)
out['ratio_vs_prev']=out.apply(lambda r: r.notional_usd/(r.prev_small if abs(r.notional_usd/r.prev_small-1)<abs(r.notional_usd/r.prev_std-1) else r.prev_std),axis=1).round(3)
out.to_csv('indep_notional.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',30)
print(out.drop(columns=['source']).to_string())
print({k:round(v,6) for k,v in fx.items()})
