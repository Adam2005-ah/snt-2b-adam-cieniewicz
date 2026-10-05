"""Contract specs for the 84 markets: multiplier (per unit of the back-adjusted price in the CSV),
currency, all-in fee per contract per side (IBKR commission + exchange/clearing/regulatory, local ccy),
half bid-ask spread in price units. Primary source: pysystemtrade instrumentconfig.csv (PerBlock) and
spreadcosts.csv (half-spread), github.com/robcarver17/pysystemtrade, fetched 2026-10-04.
src: 'pst' = taken from pysystemtrade; 'asm' = my assumption (flagged)."""
import pandas as pd
# ticker: (mult, ccy, fee, half_spread, src, small_mult, small_fee, small_half_spread, small_name, ibkr_access)
S = {
 'ES1 Index':  (50, 'USD', 2.25, 0.13, 'pst', 5, 0.62, 0.13, 'MES', 'yes'),
 'NQ1 Index':  (20, 'USD', 2.20, 0.32, 'pst', 2, 0.62, 0.24, 'MNQ', 'yes'),
 'RTY1 Index': (50, 'USD', 2.20, 0.094, 'pst', 5, 0.62, 0.11, 'M2K', 'yes'),
 'DM1 Index':  (5, 'USD', 2.25, 0.75, 'pst', 0.5, 0.62, 1.2, 'MYM', 'yes'),
 'MES1 Index': (50, 'USD', 2.07, 0.15, 'asm', None, None, None, None, 'yes'),   # ICE US MSCI EM
 'VG1 Index':  (10, 'EUR', 2.00, 0.48, 'pst', 1, 0.40, 1.0, 'FSXE micro (asm)', 'yes'),
 'GX1 Index':  (25, 'EUR', 2.15, 1.0, 'asm', 1, 0.40, 0.87, 'FDXS micro', 'yes'),
 'CF1 Index':  (10, 'EUR', 2.00, 0.44, 'pst', None, None, None, None, 'yes'),
 'EO1 Index':  (200, 'EUR', 2.80, 0.12, 'pst', None, None, None, None, 'yes'),
 'Z 1 Index':  (10, 'GBP', 1.70, 0.47, 'pst', None, None, None, None, 'yes'),
 'SM1 Index':  (10, 'CHF', 4.00, 1.0, 'pst', None, None, None, None, 'yes'),
 'QC1 Index':  (100, 'SEK', 25.0, 0.25, 'asm', None, None, None, None, 'yes'),  # OMXS30 Nasdaq Stockholm
 'NO1 Index':  (1000, 'JPY', 330, 5.0, 'asm', 100, 100, 3.4, 'Nikkei mini', 'yes'),
 'TP1 Index':  (10000, 'JPY', 330, 0.25, 'asm', 1000, 100, 0.24, 'mini TOPIX', 'yes'),
 'XP1 Index':  (25, 'AUD', 5.00, 0.57, 'pst', None, None, None, None, 'yes'),   # SPI200 (pst code SPI200)
 'PT1 Index':  (200, 'CAD', 3.00, 0.10, 'asm', 40, 1.5, 0.10, 'SXM mini (asm)', 'yes'),
 'TWT1 Index': (40, 'USD', 1.25, 0.14, 'pst', None, None, None, None, 'yes'),   # SGX FTSE Taiwan (pst FTSETAIWAN)
 'HI1 Index':  (50, 'HKD', 30.0, 1.5, 'pst', 10, 17.0, 2.0, 'mini HSI', 'yes'),
 'HC1 Index':  (50, 'HKD', 20.0, 1.0, 'pst', 10, 20.0, 1.5, 'mini HSCEI', 'yes'),
 'XU1 Index':  (1, 'USD', 2.85, 1.3, 'pst', None, None, None, None, 'yes'),
 'JGS1 Index': (2, 'USD', 2.85, 2.0, 'asm', None, None, None, None, 'doubtful (NSE IX GIFT City)'),
 'TU1 Comdty': (2000, 'USD', 1.51, 0.002, 'pst', None, None, None, None, 'yes'),
 'FV1 Comdty': (1000, 'USD', 1.51, 0.0039, 'pst', None, None, None, None, 'yes'),
 'TY1 Comdty': (1000, 'USD', 1.67, 0.008, 'pst', None, None, None, None, 'yes'),
 'UXY1 Comdty':(1000, 'USD', 1.67, 0.0081, 'pst', None, None, None, None, 'yes'),
 'US1 Comdty': (1000, 'USD', 1.74, 0.016, 'pst', None, None, None, None, 'yes'),
 'WN1 Comdty': (1000, 'USD', 1.82, 0.016, 'pst', None, None, None, None, 'yes'),
 'DU1 Comdty': (1000, 'EUR', 2.00, 0.0025, 'pst', None, None, None, None, 'yes'),
 'OE1 Comdty': (1000, 'EUR', 2.00, 0.0051, 'pst', None, None, None, None, 'yes'),
 'RX1 Comdty': (1000, 'EUR', 2.00, 0.0052, 'pst', None, None, None, None, 'yes'),
 'UB1 Comdty': (1000, 'EUR', 2.00, 0.010, 'pst', None, None, None, None, 'yes'),
 'G 1 Comdty': (1000, 'GBP', 1.70, 0.005, 'asm', None, None, None, None, 'yes'),
 'JB1 Comdty': (1000000, 'JPY', 500, 0.005, 'pst', 100000, 85, 0.023, 'mini JGB', 'yes'),
 'OAT1 Comdty':(1000, 'EUR', 2.00, 0.0072, 'pst', None, None, None, None, 'yes'),
 'CN1 Comdty': (1000, 'CAD', 2.40, 0.0056, 'pst', None, None, None, None, 'yes'),
 'XM1 Comdty': (1, 'AUD', 4.50, 22.0, 'asm', None, None, None, None, 'yes'),   # price col = AUD contract value
 'IK1 Comdty': (1000, 'EUR', 2.00, 0.0063, 'pst', None, None, None, None, 'yes'),
 'SFR5 Comdty':(2500, 'USD', 2.12, 0.003, 'pst', None, None, None, None, 'yes'),
 'SFI5 Comdty':(2500, 'GBP', 1.70, 0.0025, 'asm', None, None, None, None, 'yes'),
 'ER4 Comdty': (2500, 'EUR', 2.00, 0.0072, 'pst', None, None, None, None, 'yes'),
 'IR4 Comdty': (2466, 'AUD', 4.50, 0.005, 'asm', None, None, None, None, 'yes'),
 'EC1 Curncy': (125000, 'USD', 2.47, 2.9e-5, 'pst', 12500, 0.41, 5.9e-5, 'M6E', 'yes'),
 'BP1 Curncy': (625, 'USD', 2.47, 6.1e-3, 'pst', 62.5, 0.41, 9.4e-3, 'M6B', 'yes'),          # price in US cents
 'SF1 Curncy': (1250, 'USD', 1.50, 5.8e-3, 'pst', 125, 0.41, 1.1e-2, 'MSF', 'yes'),
 'CD1 Curncy': (1000, 'USD', 2.47, 3.0e-3, 'pst', 100, 0.41, 5.1e-3, 'MCD', 'yes'),
 'JY1 Curncy': (1250, 'USD', 2.47, 2.7e-3, 'pst', 625, 1.37, 1.0e-2, 'J7 E-mini', 'yes'),     # price = USD per 10,000 JPY
 'AD1 Curncy': (1000, 'USD', 2.47, 3.3e-3, 'pst', 100, 0.41, 6.0e-3, 'M6A', 'yes'),
 'NV1 Curncy': (1000, 'USD', 2.47, 2.9e-3, 'pst', None, None, None, None, 'yes'),
 'SE1 Curncy': (20000, 'USD', 2.47, 3.0e-3, 'pst', None, None, None, None, 'yes'),
 'NO1 Curncy': (20000, 'USD', 2.47, 3.6e-3, 'pst', None, None, None, None, 'yes'),
 'PE1 Curncy': (5000, 'USD', 2.47, 5.5e-4, 'pst', None, None, None, None, 'yes'),
 'CL1 Comdty': (1000, 'USD', 2.37, 0.012, 'pst', 100, 0.77, 0.017, 'MCL', 'yes'),
 'CO1 Comdty': (1000, 'USD', 2.37, 0.010, 'asm', None, None, None, None, 'yes'),
 'HO1 Comdty': (420, 'USD', 2.37, 3.4e-2, 'pst', None, None, None, None, 'yes'),             # price in cents/gal
 'XB1 Comdty': (420, 'USD', 2.37, 2.7e-2, 'pst', None, None, None, None, 'yes'),
 'QS1 Comdty': (100, 'USD', 1.70, 0.13, 'pst', None, None, None, None, 'yes'),               # pst GASOIL (not GASOILINE)
 'CUA1 Comdty':(29000, 'USD', 2.97, 5.1e-4, 'pst', None, None, None, None, 'yes (illiquid)'),
 'NG1 Comdty': (10000, 'USD', 2.47, 0.0011, 'pst', 2500, 1.37, 0.0028, 'QG E-mini', 'yes'),
 'GC1 Comdty': (100, 'USD', 2.47, 0.096, 'pst', 10, 0.77, 0.088, 'MGC', 'yes'),
 'HG1 Comdty': (250, 'USD', 2.47, 4.7e-2, 'pst', 25, 1.47, 7.0e-2, 'MHG', 'yes'),           # price in cents/lb
 'SI1 Comdty': (5000, 'USD', 2.47, 0.004, 'asm', 1000, 1.27, 0.004, 'SIL 1000oz', 'yes'),
 'PL1 Comdty': (50, 'USD', 2.47, 0.23, 'pst', None, None, None, None, 'yes'),
 'PA1 Comdty': (100, 'USD', 2.31, 0.56, 'pst', None, None, None, None, 'yes'),
 'SCO1 Comdty':(100, 'USD', 3.25, 0.066, 'pst', None, None, None, None, 'yes'),
 'C 1 Comdty': (50, 'USD', 2.97, 0.16, 'pst', 10, 1.90, 0.14, 'XC mini', 'yes'),
 'W 1 Comdty': (50, 'USD', 2.97, 0.16, 'pst', 10, 1.90, 0.28, 'XW mini', 'yes'),
 'KW1 Comdty': (50, 'USD', 2.97, 0.16, 'pst', None, None, None, None, 'yes'),
 'MWE1 Comdty':(5000, 'USD', 2.97, 0.0025, 'asm', None, None, None, None, 'yes'),          # price in USD/bu
 'CA1 Comdty': (50, 'EUR', 2.00, 0.12, 'pst', None, None, None, None, 'yes'),
 'S 1 Comdty': (50, 'USD', 2.97, 0.17, 'pst', 10, 1.90, 0.35, 'XK mini', 'yes'),
 'SM1 Comdty': (100, 'USD', 2.97, 0.057, 'pst', None, None, None, None, 'yes'),
 'BO1 Comdty': (600, 'USD', 2.97, 0.015, 'pst', None, None, None, None, 'yes'),
 'RS1 Comdty': (20, 'CAD', 2.40, 0.12, 'pst', None, None, None, None, 'yes'),
 'IJ1 Comdty': (50, 'EUR', 2.00, 0.27, 'pst', None, None, None, None, 'yes'),
 'LC1 Comdty': (400, 'USD', 2.97, 0.024, 'pst', None, None, None, None, 'yes'),
 'LH1 Comdty': (400, 'USD', 2.97, 0.017, 'pst', None, None, None, None, 'yes'),
 'FC1 Comdty': (500, 'USD', 2.97, 0.040, 'pst', None, None, None, None, 'yes'),
 'SB1 Comdty': (1120, 'USD', 2.97, 0.005, 'pst', None, None, None, None, 'yes'),
 'QW1 Comdty': (50, 'USD', 2.37, 0.20, 'asm', None, None, None, None, 'yes'),
 'KC1 Comdty': (375, 'USD', 2.97, 0.095, 'pst', None, None, None, None, 'yes'),
 'DF1 Comdty': (10, 'USD', 3.00, 1.9, 'pst', None, None, None, None, 'yes'),
 'CC1 Comdty': (10, 'USD', 2.97, 3.9, 'pst', None, None, None, None, 'yes'),
 'QC1 Comdty': (10, 'GBP', 1.70, 3.2, 'asm', None, None, None, None, 'yes'),
 'CT1 Comdty': (500, 'USD', 2.97, 0.044, 'pst', None, None, None, None, 'yes'),
}
COLS = ['mult', 'ccy', 'fee', 'hs', 'src', 'mult_s', 'fee_s', 'hs_s', 'small_name', 'ibkr_access']
def table():
    return pd.DataFrame.from_dict(S, orient='index', columns=COLS)
# FX: USD per 1 unit of local ccy, from the FX futures price columns
FXCOL = {'EUR': ('EC1 Curncy', 1.0), 'GBP': ('BP1 Curncy', 0.01), 'CHF': ('SF1 Curncy', 0.01),
         'CAD': ('CD1 Curncy', 0.01), 'JPY': ('JY1 Curncy', 1e-4), 'AUD': ('AD1 Curncy', 0.01),
         'SEK': ('SE1 Curncy', 0.01)}
def fx_frame(prices):
    out = pd.DataFrame(index=prices.index)
    out['USD'] = 1.0
    for c, (col, sc) in FXCOL.items():
        out[c] = prices[col].ffill() * sc
    out['HKD'] = 1 / 7.8
    return out
