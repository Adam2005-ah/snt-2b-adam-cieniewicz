import csv
F = []
def f(area, claim, verdict, value, evidence, sources):
    F.append(dict(area=area, claim_tested=claim, verdict=verdict, verified_or_corrected_value=value, evidence=evidence, sources=' ; '.join(sources)))

# ---------- 1. Vehicles ----------
f('vehicle','iMGP DBi R USD UCITS ETF = LU2951555403','corrected',
  'R USD ETF = LU2951555585 (Euronext Paris DBMF, share class 07-Mar-2025; Xetra DBMF:GY since 02-Feb-2026; also LSE). LU2951555403 = R EUR unhedged ETF (Euronext Paris DBMFE, class registered 24/31-Mar-2025). Both accumulating, Luxembourg SICAV sub-fund.',
  'cbonds/finanzfluss/extraETF profiles; etfexpress Xetra article; Fineco hosts an Italian PRIIPs KID for LU2951555403 dated 12-Feb-2026',
  ['https://www.finanzfluss.de/informer/etf/lu2951555585/','https://www.finanzfluss.de/informer/etf/lu2951555403/','https://cbonds.com/etf/235885/','https://etfexpress.com/2026/02/03/imgp-dbi-now-available-on-deutsche-borse-xetra/','https://images.fineco.it/pvt/pdf/kid/IT/LU2951555403_20260212_IT_KID-UCITS.pdf'])
f('vehicle','iMGP DBi R EUR HP (EUR-hedged) LU3359622902 / MFEH, listed Euronext Paris 21-Sep-2026','confirmed',
  'Class launched 13-May-2026; Xetra listing 18-May-2026 (MFEH, accumulating, product costs 0.75%); Euronext Paris MFEH:FP announced 21-Sep-2026.',
  'Deutsche Boerse new listings 18-May-2026 page; rankiapro and france-epargne articles on the 21-Sep-2026 Paris listing',
  ['https://www.cashmarket.deutsche-boerse.com/cash-en/Stay-Informed/newsroom/press-releases/New-ETF-and-ETP-Listings-on-May-18-2026-on-Deutsche-B-rse-5137378','https://rankiapro.com/en/news/imgp-dbi-managed-futures-eur-hedged-etf-euronext-paris/','https://www.france-epargne.fr/news/managed-futures-les-fonds-cta-gagnent-pres-de-16-en-2026-un-etf-couvert-en-euros-arrive-a-paris','https://extraetf.com/fr/etf-profile/LU3359622902'])
f('vehicle','iMGP DBi UCITS ETF TER 0.75%','confirmed (one conflicting source)',
  'TER/product costs 0.75% for all three ETF classes (Deutsche Boerse, finanzfluss, extraETF). cbonds shows 0.55% (outlier, likely management fee only). Non-ETF R GBP class LU2604833231 TER 1.09%.',
  'Deutsche Boerse listing notice says product costs 0.75%','https://www.cashmarket.deutsche-boerse.com/cash-en/Stay-Informed/newsroom/press-releases/New-ETF-and-ETP-Listings-on-May-18-2026-on-Deutsche-B-rse-5137378 ; https://www.finanzfluss.de/informer/etf/lu3359622902/ ; https://script.finanzen.ch/fonds/daten-gebuehr/imgp-dbi-managed-futures-r-lu2604833231'.split(' ; '))
f('vehicle','iMGP DBi UCITS fund AUM','confirmed',
  'USD 698M (whole UCITS sub-fund) at 18-Sep-2026; DBi strategy USD 6.59bn. finanzfluss shows EUR 488.6M (older date).',
  'rankiapro 21-Sep-2026 article',['https://rankiapro.com/en/news/imgp-dbi-managed-futures-eur-hedged-etf-euronext-paris/','https://www.finanzfluss.de/informer/etf/lu2951555403/'])
f('vehicle','iMGP DBi replicates trend following','corrected (nuance)',
  'It replicates the pre-fee return of the SG CTA Index (20 largest CTAs, all styles) with a factor model (Dynamic Beta Engine, ~10-15 futures). It is a CTA-replication product, not a pure trend system and not variant 5.',
  'finanzfluss: tracked index SG CTA PR USD; Deutsche Boerse description',['https://www.finanzfluss.de/informer/etf/lu2951555585/','https://www.cashmarket.deutsche-boerse.com/cash-en/Stay-Informed/newsroom/press-releases/New-ETF-and-ETP-Listings-on-May-18-2026-on-Deutsche-B-rse-5137378'])
f('vehicle','iMGP DBi is Europe\'s only managed-futures UCITS ETF','refuted',
  'BNP Paribas Easy Managed Futures UCITS ETF launched 29-May-2026; iMGP marketing still uses the claim in Sep-2026.',
  'see BNP row',['https://www.finanzfluss.de/informer/etf/lu3307218399/','https://stock3.com/news/managed-futures-im-etf-so-setzt-bnp-auf-steigende-und-fallende-maerkte-17263050'])
f('vehicle','iMGP DBi recent performance','reported (date of snapshot unclear)',
  'DBMFE (EUR unhedged): 1y +28.2%, YTD +13.2% (investing.com snapshot ~Sep-2026). US twin DBMF ~9.8%/yr since May-2019 (to Feb-2026).',
  'investing.com DBMFE page via search; DBi commentary',['https://www.investing.com/etfs/dbmfe-paris'])
f('vehicle','BNP Paribas Easy Managed Futures UCITS ETF: ISIN, TER, launch, strategy','confirmed + detail',
  'LU3307218399 (WKN A42CKK), launched 29-May-2026, TER 0.60%, EUR, accumulating, Luxembourg. Actively managed, no benchmark: in-house systematic trend model, euro-hedged, synthetic (swap-based) long/short exposure to equity, rate, FX and commodity futures/forwards, built with BNP Paribas Global Markets (counterparty risk on BNP). Xetra listing 24-Jun-2026 ticker EEAU; AUM ~EUR 9.9M (Aug-2026). Non-ETF Classic class LU3307218555. Euronext Paris ticker not confirmed by search.',
  'finanzfluss, stock3 (31-Aug-2026), BNP AM press release, Deutsche Boerse 24-Jun-2026 listings',
  ['https://www.finanzfluss.de/informer/etf/lu3307218399/','https://stock3.com/news/managed-futures-im-etf-so-setzt-bnp-auf-steigende-und-fallende-maerkte-17263050','https://www.bnpparibas-am.com/en/press/mediaroom-en-bnp-paribas-asset-management-enters-the-next-generation-of-actively-managed-etfs/','https://www.cashmarket.deutsche-boerse.com/cash-en/Stay-Informed/newsroom/press-releases/New-ETF-and-ETP-Listings-on-June-24-2026-on-Deutsche-B-rse-5347738','https://script.finanzen.ch/fonds/daten-gebuehr/bnp-paribas-easy-managed-futures-classic-lu3307218555'])
f('vehicle','AQR Managed Futures UCITS: retail class, fees, minimum','partly confirmed',
  'A USD LU1103257975: management 0.60% + 10% performance fee, ongoing ~0.81%, minimum ~USD 80k (finect/easybank). RAE EUR LU1662502183 (retail): management 0.60%, TER 1.39%, ongoing incl. transaction costs 2.25%; RAEF LU1662501532 also exists. Minimum for RAE not found. Not an ETF (subscribe via fund platform/bank). A USD: YTD-2026 +10.4%, 5y +12.3%/yr (finanzen.ch, snapshot date unclear).',
  'finanzen.net/finanzen.ch/easybank/finect fund pages',['https://consentmanager.finanzen.net/fonds/daten-gebuehr/aqr-funds-aqr-managed-futures-fund-rae-lu1662502183','https://markets.easybank.at/markets/fonds/FU_100143739/AQR-Managed-Futures-UCITS-Fund-RAE-EUR/profile-kennzahlen','https://www.finect.com/fondos-inversion/LU1103257975-Aqr_mgd_futures_ucits_a_usd','https://www.finanzen.ch/fonds/aqr-funds-aqr-managed-futures-fund-a-lu1103257975'])
f('vehicle','Man AHL Trend Alternative OCF ~1.72%','corrected',
  'DNY H EUR Acc LU0424370004 (retail, since 21-Jul-2009, Man Umbrella SICAV, EUR 329M at 30-Apr-2026): ongoing charges ~2.75% (TER 2.76%), management fee 2.0% plus performance fee. 1.72% applies to cheaper classes (e.g. IN H EUR LU0428380124). Reported YTD-2026 +15.1% (4-Sep-2026), 1y +40-46% (unverified snapshots).',
  'comdirect/finanzen/boursorama via search',['https://www.comdirect.de/inf/fonds/LU0424370004','https://www.comdirect.de/inf/fonds/LU0428380124','https://www.boursorama.com/bourse/opcvm/cours/0P000187W1'])
f('vehicle','Winton Trend UCITS IE00BG382P13, TER ~1.0%','corrected',
  'I EUR Acc = IE00BG382R37: ongoing 1.05%, management 0.80%. 1y return -13.8% to -15.9% in two snapshots (dates unclear, probably stale). IE00BG382P13 not found.',
  'finanzen.ch / onvista',['https://styles.finanzen.ch/fonds/winton-trend-fund-i-ie00bg382r37','https://www.onvista.de/fonds/WINTON-UC-FDS-I-WINT-TREND-FD-REG-SHS-I-EUR-ACC-ON-Fonds-IE00BG382R37'])
f('vehicle','Aspect Diversified Trends IE00B3Q12S92, 1.5% + 20%','confirmed (institutional class)',
  'IE00B3Q12S92 = Class C EUR Institutional: management 1.5%, performance 20%, ongoing 1.30%, subscription fee up to 5%. Retail L EUR class IE00B5BF6770. UCITS since 2010.',
  'finect / fonds-super-markt',['https://www.finect.com/fondos-inversion/IE00B3Q12S92-Aspect_diversified_trends_c_eur_instl','https://www.fonds-super-markt.de/fondsfinder/fondsdetails/ie00b5bf6770-aspect-diversified-trends-fund-l-eur'])
f('vehicle','Man Active Trend UCITS ETF','confirmed as registered only',
  'Registered with the Central Bank of Ireland 2-Jul-2026 (ETF Stream). No listing, ISIN or TER found by 5-Oct-2026: not yet buyable. (US Man Active Trend Enhanced ETF MATE, TER 0.97%, is US-domiciled and not available to EEA retail.)',
  'finantresnoticias citing ETF Stream; trackinsight MATE',['https://finantresnoticias.com/etfs/man-group-etf-ucits-activos-trend-infraestructura/','https://www.trackinsight.com/en/fund/MATE/characteristics'])
f('vehicle','Candriam Diversified Futures FR0010794792, 1.70%','confirmed',
  'French FCP (UCITS) since 16-Nov-2009, C EUR class, management 1.70%, ~EUR 147M, target vol <12%, objective >capitalised EuroSTR; mixes trend, contrarian and pattern models (not pure trend).',
  'finanzfluss factsheet / Candriam SAR',['https://www.finanzfluss.de/informer/fonds/fr0010794792/','https://www.candriam.com/siteassets/funds/regulator-pages/nl---candriam-diversified-futures/2025/candriam-diversified-futures_sar_30.06.2025_en.pdf'])
f('vehicle','PEA eligibility of managed-futures funds/ETFs','confirmed: not eligible',
  'PEA requires a fund to hold permanently >=75% EU/EEA equities (or replicate them synthetically). None of the trend vehicles qualifies.',
  'PEA eligibility rule (L221-31 CMF) as summarised by brokers',['https://finary.com/investir-en-bourse/etfs/etf-pea','https://www.boursedirect.fr/fr/bourse/pea'])
f('vehicle','Availability in French assurance-vie','unverifiable',
  'No insurer (Linxea, Lucya Cardif/CNP, Placement-direct, Generali, Suravenir...) was found listing iMGP DBi, BNP Easy Managed Futures, AQR, Man AHL or Candriam Diversified Futures as a unit-linked fund. Must be checked contract by contract; BNP Easy in Cardif contracts is plausible but unconfirmed.',
  'searches returned only generic AV ETF lists',['https://hellosafe.fr/blog/lucya','https://community.finary.com/t/choix-assurance-vie-linxea-spirit-2-ou-lucya-cnp/39570'])
f('vehicle','Availability on a French CTO','confirmed for ETFs',
  'iMGP DBi (DBMF/DBMFE/MFEH on Euronext Paris) and BNP Easy (Xetra, likely Paris) are exchange-listed UCITS with EU PRIIPs KIDs, so any French CTO with Euronext/Xetra access can buy them. AQR/Man/Winton/Aspect/Candriam are non-listed UCITS funds: need a fund platform/bank that distributes them; minimums may apply.',
  'listings above; Fineco KID for LU2951555403',['https://images.fineco.it/pvt/pdf/kid/IT/LU2951555403_20260212_IT_KID-UCITS.pdf'])
f('market context','Trend performance after the backtest end date (2026-07-10)','new fact',
  'SG CTA +15.91% YTD around 18-21 Sep 2026 (best year since 2022). Local pofo data: SG CTA +8.6% YTD to 10-Jul, +11.1% to 31-Aug; SG Trend +8.4% / +11.1%. The variant-5 backtest stops on 2026-07-10 and misses the Jul-Sep 2026 trend run.',
  'france-epargne article; mkt/alternatives/pofo_indices/SG_*_daily.csv',['https://www.france-epargne.fr/news/managed-futures-les-fonds-cta-gagnent-pres-de-16-en-2026-un-etf-couvert-en-euros-arrive-a-paris'])

# ---------- 2. Tax ----------
f('tax','PFU 31.4% (12.8% IR + 18.6% PS) since 2026','confirmed',
  'LFSS 2026 (loi n. 2025-1403 du 30-Dec-2025, art. 12): CSG on capital income 9.2% -> 10.6%. Applies to revenus du patrimoine (capital gains incl. art 150 ter futures profits) from the taxation of 2025 income, and to produits de placement (interest, dividends) from 1-Jan-2026. Exceptions kept at 17.2%: assurance-vie/capitalisation, PEL/CEL, revenus fonciers, plus-values immobilieres. PEA withdrawals: 18.6% PS.',
  'actu-juridique, Mayer Brown (27-Jan-2026), lafinancepourtous',['https://www.actu-juridique.fr/fiscalite/fiscal-finances/la-csg-en-hausse-sur-les-revenus-du-capital/','https://www.mayerbrown.com/fr/insights/publications/2026/01/budget-social-2026-augmentation-du-taux-de-csg-et-perennisation-du-regime-social-management-package','https://www.lafinancepourtous.com/2025/12/17/csg-un-taux-a-double-vitesse-pour-2026/','https://blog.nalo.fr/flat-tax'])
f('tax','Future changes','new risk',
  'PLF 2027 filed 30-Sep-2026 (Assemblee from 13-Oct-2026): press reports it could extend the 10.6% CSG to assurance-vie/real-estate income; nothing enacted.',
  'meilleurtaux Sep-2026',['https://placement.meilleurtaux.com/placement-financier/actualites/2026-septembre/projet-de-loi-de-finances-2027-calendrier-premieres-pistes-et-enjeux-fiscaux.html'])
f('tax','Futures profits of an individual: art 150 ter CGI, PFU, taxable each roll','confirmed',
  'Taxable event = sale or settlement (denouement) of each contract, so every roll realises P&L. Profits fall under the 12.8% PFU (art 200 A) + 18.6% PS, or the progressive scale on global option. Losses (150-0 D 11) offset same-nature gains (incl. securities capital gains such as QQQ) in the same year and the 10 following years, oldest first. Open positions at 31-Dec are not marked to market.',
  'advizexperts/thetradehub art 150 ter; BOFiP RPPM-PVBMI-20-10-40 via search; actu-juridique',['https://advizexperts.fr/code-general-impots/article-150-ter-cgi-fiscalite-instruments-financiers-terme/','https://www.thetradehub.eu/fr/reglementation/france/codes-nationaux/cgi-art-150-ter','https://bofip.impots.gouv.fr/node/4883','https://www.actu-juridique.fr/fiscalite/fiscal-finances/gains-financiers-quelle-taxation-choisir/'])
f('tax','BNC requalification risk (art 92-2-5 CGI)','confirmed + consequences',
  'Habitual operations on financial futures are BNC (BOI-BNC-SECT-50). Criteria (CE 14-Feb-2001 Boniface, 2003 rulings): number and frequency, spread over time, technicality, diversity of contracts, portfolio size, size of profits. Consequences: progressive scale + social levies (no PFU); deficits only against same-nature BNC profits of the same year and the 6 following years (art 156-I-2), not against global income or QQQ gains. A daily 14-40-market automated system scores high on number, frequency, technicality and diversity.',
  'BOFiP BOI-BNC-SECT-50 (node 13612) and lemondedudroit via search; BOFiP IR-BASE on art 156',['https://bofip.impots.gouv.fr/node/13612','https://www.lemondedudroit.fr/fiscal/277-fiscalite-des-personnes/34016-operations-sur-des-instruments-financiers-a-terme-obligations-declaratives.html','https://bofip.impots.gouv.fr/node/14458'])
f('tax','Form 3916 penalties incl. 5% of balance if >=EUR 50k','corrected',
  'IBKR Ireland account goes on form 3916 every year. Fine EUR 1,500 per undeclared account per year (EUR 10,000 if non-cooperative state); limitation period extended to 10 years; 80% surcharge on tax evaded via undeclared accounts (art 1729-0 A). The 5%-of-balance fine (art 1736 IV-2, balances >= EUR 50k) was struck down by the Conseil constitutionnel (QPC 2016-554, 22-Jul-2016) and replaced by the 80% surcharge: do NOT state it. Crypto accounts abroad (bitcoin) go on 3916-bis: EUR 750 per account, EUR 1,500 if value > EUR 50k.',
  'finalib 2026; Conseil constitutionnel 2016-554 QPC commentary; clubpatrimoine/legifiscal for 3916-bis',['https://finalib.fr/blog/fiscalite/obligation-declaration-compte-etranger-2026','https://webview-ccfr.sites.prod.conseilconstitutionnel.aquaray.com/sites/default/files/as/root/bank_mm/decisions/2016554qpc/2016554qpc_ccc.pdf','https://www.etudes-fiscales-internationales.com/apps/print/25290','https://www.clubpatrimoine.com/faq/faq-contenus/declarer-actifs-numeriques-etranger'])
f('tax','Accumulating UCITS ETF on CTO taxed only at sale','confirmed',
  'No yearly taxation of an accumulating fund held by an individual; PFU 31.4% on the gain at sale, losses 10-year carry-forward. Contrast: futures are taxed every year through rolls.',
  'justETF academy; socic 2026',['https://www.justetf.com/fr/academy/etf-et-fiscalite.html','https://www.socic.fr/ressources-comptabilite/articles/plus-values-mobilieres-2026-flat-tax-314-calcul-declaration'])
f('tax','Interest on IBKR cash: 2778-SD','partly confirmed',
  'Foreign-paid interest: 12.8% non-final prepayment + PS (18.6% from 2026) declared/paid on form 2778-SD (2026 version exists); exemption if RFR(N-2) < EUR 25k single / 50k couple. Monthly deadline (15th of following month) not re-verified.',
  'impots.gouv 2778-SD 2026 form; impots.gouv dispense FAQ',['https://www.impots.gouv.fr/sites/default/files/formulaires/2778-sd/2026/2778-sd_5387.pdf','https://www.impots.gouv.fr/particulier/questions/puis-je-beneficier-dune-dispense-du-prelevement-forfaitaire-non-liberatoire'])

# ---------- 3. Broker / rates ----------
f('broker','IBKR blocks EEA retail from PRIIPs (futures) without a KID','confirmed',
  'IBKR must block a PRIIP for EEA/UK retail when the manufacturer provides no KID; elective professional status lifts it.',
  'IBKR KB 2993',['https://ibkb.interactivebrokers.com/node/2993'])
f('broker','Exchanges with KIDs','confirmed / corrected',
  'KIDs exist: CME Group (product list; micro ag KIDs updated 9-Feb-2026), ICE Futures US/Europe, Eurex, Euronext, OSE/TOCOM (JPX page), HKFE (HKEX publishes an EEA futures KID PDF, 2025-v2) - this contradicts the Darwinex/IBKR page that lists HKEX among no-KID venues. No KID found for SGX or ASX (Darwinex/IBKR page says none) -> SGX iron ore / FTSE Taiwan / Nifty and ASX contracts likely blocked for French retail.',
  'CME PRIIPs list; ICE KID page; JPX EU-PRIIPs page; HKEX PDF; Darwinex IBKR access page; Euronext KID page',['https://www.cmegroup.com/market-regulation/european-regulation/priips-futures-products.html','https://www.ice.com/futures-us/documents/kid','https://www.jpx.co.jp/english/derivatives/eu-priips-regulation/','https://www.hkex.com.hk/-/media/HKEX-Market/Services/Trading/HKEX-Derivatives-Information-for-US-Investors/2025-v2/HKFE_Futures-Contracts_EEA.pdf','https://www.darwinex.com/eu/ibkr/stocks-futures-etfs','https://live.euronext.com/en/resources/key-information-document'])
f('broker','IBKR pays no interest on commodity-segment cash','confirmed',
  'No interest on excess funds in the commodities segment; interest only on securities-segment cash. Excess can be (auto-)swept to securities, so the zero-interest part is the futures margin itself (~17-23% of capital on average for v5).',
  'IBKR Margin Interest Calculations page (several IBKR domains)',['https://www.interactivebrokers.com/en/trading/margin-calculation-details.php','https://www.interactivebrokers.ca/en/trading/margin-calculation-details.php'])
f('broker','IBKR cash interest: BM-0.5%, first 10k unpaid, pro-rata below 100k NAV','confirmed',
  'IBKR Pro pays BM - 0.5% on balances above USD 10,000 (or equivalent; 0% on the first 10k), and a proportionally lower rate when NAV < USD 100,000. Quoted pre-hike: USD 3.12%, EUR 1.700%; EUR benchmark 2.243% on 29-Jul-2026. After the Sep-2026 hikes expect ~EUR 1.9-2.0%, USD ~3.4%.',
  'IBKR interest-rates page via search',['https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php?lp=T','https://www.interactivebrokers.ie/fr/accounts/fees/pricing-interest-rates.php','https://www.interactivebrokers.co.uk/cn/pricing/reference-benchmark-rates-int.php'])
f('rates','ECB deposit rate October 2026','confirmed (resolves earlier dispute)',
  'DFR 2.50% (MRO 2.65%, MLF 2.90%) since 16-Sep-2026, decided 10-Sep-2026; second hike of 2026 after 11-Jun (2.00 -> 2.25). The Banco de Espana \'unchanged in September\' page is about Sept 2025. Next meeting 29-Oct-2026; futures priced ~60% for 2.75% (24-Sep).',
  'il Foglio 10-Sep-2026; pwc; beninwebtv; admiralmarkets',['https://www.ilfoglio.it/en/economy/2026/09/10/news/the-ecb-raises-interest-rates-this-is-the-second-rise-of-2026--407136','https://pwcplus.de/en/article/256769/monetary-policy-decisions-september-2026/','https://beninwebtv.bj/en/eurozone-ecb-raises-rates-deposit-rate-increases-to-2-50/','https://admiralmarkets.com/analytics/traders-blog/ecb-meeting-october-2026'])
f('rates','EuroSTR October 2026 (critic used 2.3%)','corrected (inferred)',
  'EuroSTR 2.19% on 9-Sep-2026 (DFR 2.25 minus ~6 bp). After the 16-Sep hike, ~2.43% (not directly sourced; same spread). Use 2.43%, 2.68% if the ECB hikes on 29-Oct.',
  'global-rates EuroSTR 2026 history',['https://www.global-rates.com/en/interest-rates/ester/historical/2026/'])
f('rates','US policy rate','confirmed',
  'FOMC raised fed funds to 3.75-4.00% on 16-Sep-2026. DTB3 = 4.00% on 2026-10-01 (local FRED file).',
  'securities.io; mkt/us-etf/alm0421_macro/DTB3.csv',['https://www.securities.io/fomc-raises-federal-funds-target-range-to-3-3-4-to-4-percent/'])

# ---------- 4. Liquidity ----------
f('liquidity','1OZ (COMEX 1-Ounce Gold)','confirmed liquid',
  'Record ADV 115,000 contracts (Jan-2026), 80,000 (Oct-2025). Not thin.',
  'CME monthly volume press releases',['https://www.cmegroup.com/media-room/press-releases/2026/2/03/cme_group_januaryvolumesetsnewrecordof296millioncontractsup15yea.html','https://www.cmegroup.com/media-room/press-releases/2025/11/04/cme_group_octobervolumehitsnewrecordof263millioncontractsup8year.html'])
f('liquidity','OSE Nikkei 225 Micro','confirmed very liquid',
  'ADV 0.64M contracts in FY2025 (0.52M FY2024); 22.8M contracts in June 2026; KID published by OSE.',
  'JPX Investor Day 2026; FOW',['https://www.jpx.co.jp/english/corporate/investor-relations/ir-library/events/Investor Day 2026_OSE_English.pdf','https://www.fow.com/insights/nikkei-225-micro-futures-hit-near-record-june-volume-at-jpx'])
f('liquidity','Eurex Micro-DAX FDXS','confirmed liquid',
  'ADV 24,129 (2025), 28,380 (2026 YTD).',
  'Eurex five-year micro futures article',['https://www.eurex.com/ec-en/find/news/A-five-year-success-story-Micro-futures-drive-global-adoption-of-European-benchmarks-5093562'])
f('liquidity','Eurex Micro-Euro Stoxx 50 FSXE','thin - flag',
  'ADV only 1,842 (2025) and 2,429 (2026 YTD) contracts, about EUR 13M notional/day. Fine for 1-10 lots with limit orders, but expect wider spreads than FESX; FESX (EUR 10/pt) is the liquid alternative from ~EUR 55k per contract.',
  'Eurex article',['https://www.eurex.com/ec-en/find/news/A-five-year-success-story-Micro-futures-drive-global-adoption-of-European-benchmarks-5093562'])
f('liquidity','M6E Micro EUR/USD','adequate',
  'About 19k contracts/day for the front contract in a 2025 snapshot; tick USD 1.25 (EUR 12,500).',
  'TradingView M6EM2025 / ironbeam',['https://tradingview.com/symbols/CME_MINI-M6EM2025','https://www.ironbeam.com/knowledge-base/micro-euro-futures-m6e-contract-specifications/'])
f('liquidity','M6A Micro AUD/USD','moderate',
  'About 3.9k contracts/day, open interest 4.8k (TradingView snapshot, undated).',
  'TradingView',['https://in.tradingview.com/symbols/CME_MINI-M6A1!'])
f('liquidity','M6B Micro GBP/USD','thin - flag',
  'About 580 contracts/day, open interest 1.49k (TradingView snapshot, undated). Tick USD 0.63.',
  'TradingView',['https://www.tradingview.com/symbols/CME_MINI-M6B1%21'])
f('liquidity','MJY Micro JPY/USD','unverified - flag',
  'Original MJY delisted 15-Mar-2022 (SER 8890); relaunched 26-Feb-2024 (SER 9326; JPY 1.25M, tick USD 1.25). No ADV/OI found; treat as thin until checked in TWS.',
  'CME SER 8890 / SER 9326',['https://www.cmegroup.com/notices/ser/2021/12/SER-8890.pdf','https://www.cmegroup.com/notices/ser/2024/01/SER-9326.pdf'])
f('liquidity','MNG Micro Henry Hub','moderate/unverified',
  'Launched 6-Nov-2023; 50,000 contracts cumulative in first days; recent per-month daily volumes reported ~1k-14k (undated). Cash-settled, 1,000 MMBtu, tick USD 1.',
  'CME press release 8-Nov-2023; CME/TradingView pages',['https://www.cmegroup.com/media-room/press-releases/2023/11/08/micro_henry_hub_futuressurpass50000contractstraded.html','https://cmegroup.com/markets/energy/natural-gas/micro-henry-hub-natural-gas_contract_specifications.html'])
f('liquidity','MZC / MZW / MZS micro grains','unverified - flag (likely thin)',
  'Launched 24-Feb-2025 (cash-settled, 500 bu). Only launch-day data (54,000 contracts across the five micro ags) and one day of MZC (199 contracts, 29-Mar-2025) found; no 2026 ADV/OI. Tick 0.5 c/bu (USD 2.50) = twice the relative tick of ZC (0.25 c on 5,000 bu), so a one-tick spread costs ~0.12% of notional (corn ~USD 4.2). KIDs exist (CME list updated 9-Feb-2026).',
  'CME launch notice/FAQ; feedstuffs; TradingView; CME PRIIPs page',['https://cmegroup.com/markets/agriculture/grains/micro-corn.contractSpecs.html','https://www.feedstuffs.com/market-news/cme-group-launches-micro-contracts-for-corn-soybeans','https://www.cmegroup.com/market-regulation/european-regulation/priips-futures-products.html'])

with open('/tmp/claude-0/-home-user-snt-2b-adam-cieniewicz/4246b01c-1857-5934-aed2-b3eb85d88a07/scratchpad/audit/cv_fonds_fisc/facts_verified.csv','w',newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(F[0].keys())); w.writeheader(); w.writerows(F)
print(len(F), 'rows')
