"""IB Gateway route for variant 5: facts table (with sources) and running-cost budget.

Writes, in this folder:
  ib_gateway_facts.csv          one row per external fact, with URL and confidence
  ib_gateway_budget.csv         fixed-cost line items x tier x budget level (USD/yr)
  ib_gateway_budget_summary.csv per tier: variable, fixed, cash shortfall, totals (% of capital)
  ib_gateway_time.csv           setup and recurring time estimates
No tax content: every figure is before tax.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "simulation", "results.csv")

EURUSD = 1.1446      # audit FX (10 Jul 2026), specs.csv
USDJPY_INV = 0.006216  # USD per JPY, same source
VAT_FR = 1.20

# ----------------------------------------------------------------------------- facts
F = []


def fact(topic, fact_, value, unit, url, conf, note=""):
    F.append(dict(topic=topic, fact=fact_, value=value, unit=unit, source_url=url, confidence=conf, note=note))


# 1. IB Gateway / TWS / account
fact("gateway", "IB Gateway is a stripped-down, headless-friendly version of TWS: same login, same API, no charts; uses roughly 40% fewer resources; recommended for algo boxes",
     "", "", "https://newyorkcityservers.com/blog/interactive-brokers-vps-tws-ib-gateway ; https://www.interactivebrokers.com/docs/tws-api/doc/architecture/the-trader-workstation/the-ib-gateway", "high")
fact("gateway", "pysystemtrade recommends the Gateway over TWS ('much more stable and lightweight')", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/IB.md", "high")
fact("gateway", "TWS API has no access fee; only market data subscriptions are paid", "0", "USD/month",
     "https://mylinedchart.com/resources/articles/ibkr-market-data-license-costs-for-developers ; https://www.interactivebrokers.com/en/trading/ib-api.php", "high")
fact("gateway", "Commission schedule is per product/exchange, not per channel; no source found that charges API orders differently from TWS orders", "", "",
     "https://www.interactivebrokers.com/en/pricing/commissions-home.php", "medium", "absence of a channel surcharge, not an explicit IBKR statement")
fact("account", "API access requires IBKR Pro; IBKR Lite accounts get 'API support is not available for accounts that support commission free trading'", "", "",
     "https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/interactive-brokers ; https://www.elitetrader.com/et/threads/tws-not-allowing-api-to-connect-ninjatrader-or-any-for-zero-commission-trades-under-ibkr-lite.339450/", "high")
fact("account", "IBKR Lite is offered to US residents; EEA residents are served by Interactive Brokers Ireland on IBKR Pro", "", "",
     "https://brokerchooser.com/broker-reviews/interactive-brokers-review/interactive-brokers-ireland", "medium", "irrelevant anyway: Lite has no API")
fact("account", "No monthly activity/inactivity fee since 1 July 2021 (Lite and Pro)", "0", "USD/month",
     "https://www.financemagnates.com/forex/brokers/interactive-brokers-eliminates-monthly-account-inactivity-fees/", "high")
fact("account", "Additional usernames are free; only one brokerage session per username, so a dedicated API username avoids the Gateway being kicked out when you log in on mobile/TWS",
     "0", "USD", "https://www.interactivebrokers.com/docs/web-api/authentication/multiple-sessions ; https://www.interactivebrokers.com/docs/tws-api/doc/connectivity/logging-into-multiple-applications", "high")
fact("account", "Market data is billed per username and cannot be shared between usernames: subscribe on the API username only", "", "",
     "https://www.interactivebrokers.com/en/pricing/market-data-pricing.php ; https://quietalphalab.com/what-ibkr-market-data-subscriptions-for-automation/", "high")
fact("account", "Real-time data subscriptions need USD 500 equity plus the subscription cost (individual account)", "500", "USD",
     "https://www.interactivebrokers.com/docs/general/market-data-subscriptions/market-data-subscription-minimum-equity-balance-requirements", "high")
fact("paper", "Paper-trading account is free and works with the same API (Gateway port 4002 paper vs 4001 live; TWS 7497/7496)", "0", "USD",
     "https://www.interactivebrokers.com/campus/trading-lessons/installing-configuring-tws-for-the-api/ ; https://github.com/gnzsnz/ib-gateway-docker", "high")
fact("paper", "Live real-time subscriptions can be shared with the paper account (Account Settings > Paper Trading); otherwise paper gets delayed data (US futures 10 min)", "", "",
     "https://www.ibkrguides.com/kb/article-1719.htm ; https://www.interactivebrokers.com/en/trading/papertrader-delayed-data.php", "high")
fact("system", "IB Gateway: about 4 GB RAM (8 GB recommended) on Windows 10+/Ubuntu 18.04+; bundles its own Java; one instance under load can use about 1 GB",
     "4-8", "GB RAM", "https://blog.traderspost.io/article/what-is-ib-gateway ; https://www.interactivebrokers.com/en/trading/tws-requirements.php", "medium")
fact("restart", "TWS/Gateway (v974+) restart automatically each day without re-authentication (Auto Restart, Mon-Sat); choose Auto Restart, not Auto Log Off", "1", "restart/day",
     "https://www.interactivebrokers.com/docs/tws-api/doc/tws-settings/daily-weekly-reauthentication ; https://newyorkcityservers.com/blog/interactive-brokers-vps-tws-ib-gateway", "high")
fact("restart", "Weekly: after the Saturday-night reset, a full login with second factor is required the first time the Gateway runs after 01:00 ET on Sunday", "1", "2FA login/week",
     "https://raw.githubusercontent.com/IbcAlpha/IBC/master/userguide.md", "high")
fact("restart", "2FA is mandatory for live accounts and cannot be disabled for trading platforms; IBKR Mobile (IB Key) push is the practical method", "", "",
     "https://quietalphalab.com/ibkr-2fa-24-7-automated-trading/ ; https://www.interactivebrokers.com/campus/trading-lessons/launching-and-authenticating-the-gateway/", "high")
fact("restart", "IBC cannot approve the 2FA itself; it re-sends the alert if missed (ReloginAfterSecondFactorAuthenticationTimeout=yes), alert times out after about 3 minutes", "3", "minutes",
     "https://raw.githubusercontent.com/IbcAlpha/IBC/master/userguide.md", "high")
fact("restart", "IBC RESTART command cannot bypass the Sunday full second-factor authentication", "", "",
     "https://newreleases.io/project/github/IbcAlpha/IBC/release/3.16.0", "high", "verified in the earlier cost audit")
fact("restart", "Daily IBKR server reset windows: North America 00:15-01:45 ET, Europe 06:25-07:45 CET; expect disconnects then", "", "",
     "https://www.interactivebrokers.com/en/software/systemStatus.php", "medium")
fact("ibc", "IBC (IbcAlpha) starts the Gateway, fills the login, accepts API connections, handles daily auto-restart and 2FA retries; free, open source", "0", "USD",
     "https://github.com/IbcAlpha/IBC", "high")
fact("docker", "Most-used image: gnzsnz/ib-gateway-docker (IB Gateway + IBC + Xvfb + optional VNC + socat). On 5 Oct 2026: stable 10.50.1e, latest 10.51.1b, IBC 3.24.2; TRADING_MODE live/paper/both; ports 4001/4002 bound to 127.0.0.1", "", "",
     "https://raw.githubusercontent.com/gnzsnz/ib-gateway-docker/master/README.md", "high")
fact("docker", "Alternative automaters: QuantConnect IBAutomater; other forks of ib-gateway-docker", "", "",
     "https://github.com/QuantConnect/IBAutomater", "high")
fact("pacing", "API request limit = market-data lines / 2 per second; default 100 lines -> 50 messages/s (TWS can pace instead of disconnecting)", "50", "messages/s",
     "https://www.interactivebrokers.com/docs/tws-api/doc/pacing-limitations/introduction", "high")
fact("pacing", "Default market-data lines: 100 simultaneous streaming quotes (more via commissions or Quote Booster) - far above what a 14-40 market daily system streams", "100", "lines",
     "https://www.interactivebrokers.com/docs/tws-api/doc/pacing-limitations/introduction", "high")
fact("pacing", "Historical pacing: no identical request within 15 s, max 60 requests per 10 min, max 6 per contract within 2 s; strict limits now documented for bars of 30 s or less (eased for larger bars)", "60", "requests/10 min",
     "https://interactivebrokers.github.io/tws-api/historical_limitations.html ; https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/pacing-violations-for-small-bars-30-secs-or-less", "medium", "easing date (Dec 2024) from a secondary source")
fact("orders", "IB Adaptive algo works for US futures (and some foreign products) via the API: algoStrategy='Adaptive', adaptivePriority Urgent/Normal/Patient; DAY only (no GTC)", "", "",
     "https://interactivebrokers.github.io/tws-api/ibalgos.html ; https://www.interactivebrokers.com/campus/trading-lessons/adaptive/", "high")
fact("orders", "pysystemtrade execution algos: default algoOriginalBest (passive limit then aggressive, needs live streaming), algoSnapMkt (market), algoLimit; per-instrument overrides in config", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/sysdata/config/defaults.yaml", "high")

# 2. Data via the API
fact("data", "Free delayed streaming data via reqMarketDataType(3) (4 = delayed-frozen) without subscription; if live data is available it is returned instead", "", "",
     "https://interactivebrokers.github.io/tws-api/delayed_data.html ; https://interactivebrokers.github.io/tws-api/market_data_type.html", "high")
fact("data", "Delay of free data: CME, CBOT, NYMEX, COMEX 10 min; Eurex 15 min; Euronext 15 min; ICE US (NYBOT) 10 min; ICE Futures Europe commodities and financials 10 min; OSE.JPN 20 min", "10-20", "minutes",
     "https://www.interactivebrokers.com/en/pricing/market-data-pricing.php ; https://www.interactivebrokers.com/en/trading/papertrader-delayed-data.php", "medium", "OSE delay from search snippet of the IBKR pricing page")
fact("data", "IBKR API docs: the API always requires Level 1 real-time streaming permission to return historical bars (unlike TWS delayed charts)", "", "",
     "https://interactivebrokers.github.io/tws-api/historical_data.html", "high")
fact("data", "pysystemtrade maintainers: 'market data subscriptions used to be required for IB historical data. This changed in early 2023. There was no announcement... so things may change'", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/production.md", "medium", "contradicts the official docs: must be tested on your own username")
fact("data", "Historical bars of expired futures are available up to 2 years after expiry (includeExpired=True); no continuous contract with end date; expired options not available", "2", "years",
     "https://www.interactivebrokers.com/campus/ibkr-quant-news/historical-options-futures-data-using-tws-api/ ; https://interactivebrokers.github.io/tws-api/classIBApi_1_1Contract.html", "high")
fact("data", "Carver's free pysystemtrade CSVs (multiple and adjusted prices) stop on 28 Mar 2024", "2024-03-28", "date",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/data.md ; https://raw.githubusercontent.com/pst-group/pysystemtrade/master/data/futures/adjusted_prices_csv/SP500.csv", "high", "checked last row of SP500 and CORN files")
fact("data", "Community set bug-or-feature/pst-csv-data: 40 instruments (incl. AEX, CAC, BOBL, BUND, CORN, GOLD_micro, NIKKEI, SOFR, US2-US30, WHEAT) to 23 Sep 2025", "2025-09-23", "date",
     "https://raw.githubusercontent.com/bug-or-feature/pst-csv-data/main/README.md ; https://raw.githubusercontent.com/bug-or-feature/pst-csv-data/main/data/adjusted_prices_csv/CORN.csv", "high", "no warranty")
fact("data", "Norgate futures: about 100 markets on 11 exchanges (incl. Eurex, Euronext, ICE Futures Europe), EOD, back-adjusted continuous; USD 270/12 months", "270", "USD/yr",
     "https://norgatedata.com/futurespackage.php ; https://norgatedata.com/data-content-tables.php", "high")
fact("data", "Norgate Data Updater is Windows-only (Linux/Mac via a Windows VM or WSL2); a Norgate-to-pysystemtrade importer exists", "", "",
     "https://pypi.org/project/norgatedata/ ; https://github.com/davidszp/norgate-pst-utils", "high")
fact("data", "Databento: usage-based historical data from USD 0.50/GB, USD 125 free credit; CME Globex (CME/CBOT/NYMEX/COMEX) with daily OHLCV; Eurex only from 2025-03, ICE Europe financials from 2018", "0.50", "USD/GB",
     "https://databento.com/futures ; https://databento.com/datasets/XEUR.EOBI ; https://databento.com/docs/release-notes", "medium", "a one-off daily back-fill of ~40 markets is likely within the free credit (not tested)")
fact("subs", "US Securities Snapshot and Futures Value Bundle (non-pro): L1 for CME, CBOT, NYMEX, COMEX; USD 10/month, waived in any month with >= USD 30 commissions", "10", "USD/month",
     "https://ibkb.interactivebrokers.com/fr/node/2840 ; https://www.interactivebrokers.com/en/pricing/market-data-pricing.php", "high")
fact("subs", "CME non-professional top-of-book device fee (exchange level, Jan 2026)", "1.60", "USD/month",
     "https://www.cmegroup.com/market-data/files/january-2026-market-data-fee-list.pdf", "medium")
fact("subs", "Carver's own IBKR subscriptions (blog list reproduced in pysystemtrade docs): Eurex Core EUR 8.75, Eurex Retail Europe EUR 2.00, Euronext Data Bundle EUR 3.00, Osaka Exchange JPY 200 per month (no ICE)", "", "per month",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/production.md", "medium", "2021 list; prices may have moved")
fact("subs", "Eurex at IBKR (non-pro): Eurex Retail Europe EUR 2/month; Eurex Core (L2) EUR 12/month per search snippet", "2-12", "EUR/month",
     "https://www.interactivebrokers.com/en/pricing/market-data-pricing.php", "low", "conflicts with Carver's EUR 8.75 for Eurex Core; Deutsche Boerse Retail Europe product EUR 1")
fact("subs", "Euronext Bundle L1 (non-pro) at IBKR: EUR 3/month (equity derivatives top of book)", "3", "EUR/month",
     "https://www.interactivebrokers.com/en/pricing/market-data-pricing.php", "medium")
fact("subs", "Osaka Exchange: IBKR Japan page shows N/A for non-pro and JPY 1,500/month pro; Carver (UK client) lists JPY 200/month", "200-1500", "JPY/month",
     "https://www.interactivebrokers.co.jp/en/pricing/market-data-pricing.php ; https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/production.md", "low", "check in Account Management; delayed 20 min is enough for a daily system")
fact("subs", "ICE non-professional real-time 2026: ICE Futures US USD 148, ICE Futures Europe commodities USD 161, financials USD 139 per month", "148/161/139", "USD/month",
     "https://www.ampfutures.com/trading-info/exchange-data-fees ; https://www.ice.com/publicdocs/data/Market_Data_Subscriber_Fees_2026.pdf", "medium", "exchange fees passed through by brokers; exact IBKR price not found")
fact("subs", "Data subscriptions are billed for the full month (not pro-rated)", "", "",
     "https://www.interactivebrokers.com/docs/general/market-data-subscriptions/market-data-costs-and-fees", "high")

# 3. pysystemtrade
fact("pst", "pysystemtrade moved to github.com/pst-group in Jan 2026 (Carver + Andy Geach); needs Python >= 3.10, MongoDB + Parquet; dependency ib_async>=2,<3", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/README.md ; https://raw.githubusercontent.com/pst-group/pysystemtrade/master/pyproject.toml", "high")
fact("pst", "pysystemtrade switched from ib_insync to ib_async in April 2026", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/IB.md", "high")
fact("pst", "ib_async is the maintained fork of ib_insync (author Ewald de Wit died early 2024), under ib-api-reloaded", "", "",
     "https://github.com/ib-api-reloaded/ib_async ; https://github.com/mattsta/ib_insync", "high")
fact("pst", "Daily processes: run_daily_price_updates (FX, sampled contracts, IB historical prices, multiple/adjusted prices), run_capital_update, run_systems (backtest -> optimal positions), run_strategy_order_generator, run_stack_handler (execution), run_reports, run_backups, run_cleaners", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/production.md", "high")
fact("pst", "Price spikes block the write and email the user for a manual check; rolls via interactive_update_roll_status (manual, semi-auto or auto, passive roll recommended)", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/production.md", "high")
fact("pst", "Download by region possible (e.g. ASIA 07:00, EMEA 18:00, US 20:00 local)", "", "",
     "https://raw.githubusercontent.com/pst-group/pysystemtrade/master/docs/production.md", "high")

# Costs / infra
fact("infra", "Hetzner CX23 (2 vCPU, 4 GB) EUR 5.49 and CX33 (4 vCPU, 8 GB) EUR 8.49 per month from 15 Jun 2026, plus EUR 0.50 IPv4, excl. VAT", "5.49 / 8.49", "EUR/month",
     "https://northflank.com/blog/hetzner-cloud-server-price-increases ; https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/", "high")
fact("infra", "EDF Tarif Bleu base, 1 Aug 2026: EUR 0.2001/kWh incl. taxes (6 kVA)", "0.2001", "EUR/kWh",
     "https://www.fournisseurs-electricite.com/fournisseurs/edf/tarifs/bleu-reglemente", "high")
fact("fx", "IBKR Pro FX conversion: 0.20 bp of value, min USD 2 per order (IdealPro); auto-conversion about 0.03% markup", "2", "USD min/order",
     "https://www.interactivebrokers.com/download/newMark/PDFs/commissionsForex.pdf ; https://quantroutine.com/brokers/interactive-brokers-currency-conversion-guide/", "medium")
fact("cash", "No interest on cash in the commodities (futures) segment; set Excess Funds Sweep to 'sweep to securities' so only required margin stays there", "0", "%",
     "https://www.interactivebrokers.com/en/trading/margin-calculation-details.php ; https://www.ibkrguides.com/orgportal/excessfundssweep.htm", "high")
fact("alt", "QuantConnect Researcher USD 84/month (hosted alternative, still IBKR + weekly 2FA)", "84", "USD/month",
     "https://www.quantconnect.com/pricing.md", "high", "verified in the earlier cost audit")

facts = pd.DataFrame(F)
facts.insert(0, "id", range(1, len(facts) + 1))
facts["retrieved"] = "2026-10-05"
facts.to_csv(os.path.join(HERE, "ib_gateway_facts.csv"), index=False)

# ----------------------------------------------------------------------------- trading volume per tier
sim = pd.read_csv(SIM)
sel = sim[(sim.universe == "b_strict_subset_ge1c") & (sim.notional == "today") & (sim.period == "2010-2026") & (sim.freq == "D")]
TIERS = {100_000: "100k", 250_000: "250k", 1_000_000: "1M"}
# audit's verified fee + half-spread (% of capital/yr), ER4 spread corrected (verif_capital)
VAR = {100_000: (3.30, 1.72), 250_000: (3.65, 2.17), 1_000_000: (2.87, 1.72)}  # (total, half-spread part)
CASH = {100_000: 0.79, 250_000: 0.67, 1_000_000: 0.61}  # shortfall vs ESTR, %/yr (audit)
N_MKTS = {100_000: 14, 250_000: 24, 1_000_000: 40}
ICE = {100_000: [], 250_000: ["ICE US (canola, sugar 11)", "ICE Europe financials (Euribor, SONIA)"],
       1_000_000: ["ICE US (canola, sugar 11, cotton)", "ICE Europe financials (Euribor, SONIA)", "ICE Europe commodities (white sugar)"]}

vol = {}
for cap in TIERS:
    r = sel[sel.capital_usd == cap].iloc[0]
    sides = r.trade_contract_sides_per_yr + r.roll_contract_sides_per_yr
    orders = r.trade_orders_per_yr + r.roll_orders_per_yr
    fees = (VAR[cap][0] - VAR[cap][1]) / 100 * cap        # commissions + exchange fees, USD/yr
    comm_lo, comm_hi = 0.35 * fees, 0.50 * fees          # IBKR commission share of all-in fee (0.25/0.62 micro, 0.85/2.25 std, Eurex higher)
    vol[cap] = dict(sides=int(sides), orders=int(orders), roll_orders=int(r.roll_orders_per_yr), fees=fees,
                    comm_month_lo=comm_lo / 12, comm_month_hi=comm_hi / 12)

# ----------------------------------------------------------------------------- budget lines (USD/yr)
eur = lambda x: x * EURUSD
cx23 = eur((5.49 + 0.50) * VAT_FR * 12)   # ~99
cx33 = eur((8.49 + 0.50) * VAT_FR * 12)   # ~148
home_pc = eur(12 * 8.76 * 0.2001)         # 12 W average x 8760 h at EDF base tariff, ~24
eurex = eur((8.75 + 2.00) * 12)           # Carver's two Eurex lines, ~148
euronext = eur(3.00 * 12)                 # ~41
ose = 200 * USDJPY_INV * 12               # ~15
ice_px = {"ICE US": 148 * 12, "ICE Europe financials": 139 * 12, "ICE Europe commodities": 161 * 12}

rows = []


def line(level, cap, item, usd, basis, conf):
    rows.append(dict(budget=level, capital_usd=cap, tier=TIERS[cap], item=item, usd_per_year=round(usd), basis=basis, confidence=conf))


for cap in TIERS:
    us_bundle_miss = 10 if cap == 100_000 else 0   # expected months below the USD 30 commission threshold
    ice_cost = 0.0
    ice_lab = []
    if cap >= 250_000:
        ice_cost += ice_px["ICE US"] + ice_px["ICE Europe financials"]
        ice_lab += ["ICE US", "ICE Europe financials"]
    if cap >= 1_000_000:
        ice_cost += ice_px["ICE Europe commodities"]
        ice_lab += ["ICE Europe commodities"]

    # minimal
    line("minimal", cap, "Server", cx23, "Hetzner CX23 4 GB, EUR 5.99/month + 20% VAT (alt.: always-on home PC ~USD 25/yr electricity)", "high")
    line("minimal", cap, "US futures data (CME/CBOT/NYMEX/COMEX L1)", us_bundle_miss, "Value Bundle USD 10/month, waived when commissions >= USD 30 that month", "medium")
    line("minimal", cap, "Eurex / Euronext / OSE / ICE data", 0, "free delayed data (10-20 min) + historical bars if IB keeps serving them without subscription (undocumented since 2023)", "low")
    line("minimal", cap, "Back history", 0, "Carver CSV to 2024-03 + community CSV to 2025-09 + IB 2 years of expired contracts + own backtest data", "medium")
    line("minimal", cap, "FX conversions", 25, "about 12 conversions x USD 2 minimum", "medium")
    line("minimal", cap, "Software", 0, "IB Gateway, IBC, Docker image, pysystemtrade/ib_async, MongoDB: free", "high")

    # standard (recommended)
    line("standard", cap, "Server", cx33, "Hetzner CX33 8 GB (Gateway + MongoDB + pysystemtrade), EUR 8.99/month + 20% VAT", "high")
    line("standard", cap, "US futures data (CME/CBOT/NYMEX/COMEX L1)", us_bundle_miss, "Value Bundle, waived in months with >= USD 30 commissions", "medium")
    line("standard", cap, "Eurex data", eurex, "Eurex Core EUR 8.75 + Retail Europe EUR 2 per month (Carver's lines; IBKR may quote EUR 2-12)", "low")
    line("standard", cap, "Euronext data", euronext, "Euronext Bundle L1 EUR 3/month", "medium")
    line("standard", cap, "OSE data", ose, "JPY 200/month if offered to non-pros, else delayed 20 min (USD 0)", "low")
    line("standard", cap, "ICE data", 0, "delayed 10 min + market or Adaptive orders (the cost model already pays the half-spread)" if cap >= 250_000 else "no ICE market held at this tier", "medium")
    line("standard", cap, "Back history", 0, "free CSVs + IB 2 years of expired contracts", "medium")
    line("standard", cap, "FX conversions", 50, "about 2 conversions per month x USD 2", "medium")
    line("standard", cap, "Software", 0, "free", "high")

    # comfortable
    line("comfortable", cap, "Server", cx33 + cx23, "CX33 live + CX23 second instance (paper/backup)", "high")
    line("comfortable", cap, "US futures data (CME/CBOT/NYMEX/COMEX L1)", 2 * us_bundle_miss, "Value Bundle, waiver missed a few months", "medium")
    line("comfortable", cap, "Eurex data", eurex, "Eurex Core + Retail Europe", "low")
    line("comfortable", cap, "Euronext data", euronext, "Euronext Bundle", "medium")
    line("comfortable", cap, "OSE data", ose, "JPY 200/month", "low")
    line("comfortable", cap, "ICE data", ice_cost, ("real-time " + " + ".join(ice_lab)) if ice_lab else "no ICE market held at this tier", "medium")
    line("comfortable", cap, "Back history / cross-check", 270, "Norgate futures (Windows-only updater)", "high")
    line("comfortable", cap, "FX conversions", 100, "monthly sweeps in USD, GBP, JPY, CAD", "medium")
    line("comfortable", cap, "Software", 0, "free", "high")

budget = pd.DataFrame(rows)
tot = budget.groupby(["budget", "capital_usd", "tier"], as_index=False).usd_per_year.sum()
tot["item"] = "TOTAL fixed"
tot["basis"] = "sum of the lines above"
tot["confidence"] = ""
budget = pd.concat([budget, tot[budget.columns]], ignore_index=True)
budget["pct_of_capital"] = (budget.usd_per_year / budget.capital_usd * 100).round(3)
order = {"minimal": 0, "standard": 1, "comfortable": 2}
budget["o"] = budget.budget.map(order)
budget = budget.sort_values(["capital_usd", "o"], kind="stable").drop(columns="o")
budget.to_csv(os.path.join(HERE, "ib_gateway_budget.csv"), index=False)

# ----------------------------------------------------------------------------- summary per tier
S = []
t = tot.set_index(["budget", "capital_usd"]).usd_per_year
for cap, lab in TIERS.items():
    v = vol[cap]
    d = dict(tier=lab, capital_usd=cap, markets=N_MKTS[cap], orders_per_yr=v["orders"], roll_orders_per_yr=v["roll_orders"],
             contract_sides_per_yr=v["sides"],
             ibkr_commissions_usd_per_month_est=f"{v['comm_month_lo']:.0f}-{v['comm_month_hi']:.0f}",
             us_bundle_waiver_met_on_average="yes" if v["comm_month_lo"] >= 30 else "borderline",
             variable_fee_plus_halfspread_pct=VAR[cap][0], of_which_halfspread_pct=VAR[cap][1],
             fixed_minimal_usd=int(t[("minimal", cap)]), fixed_standard_usd=int(t[("standard", cap)]),
             fixed_comfortable_usd=int(t[("comfortable", cap)]),
             fixed_minimal_pct=round(t[("minimal", cap)] / cap * 100, 3),
             fixed_standard_pct=round(t[("standard", cap)] / cap * 100, 3),
             fixed_comfortable_pct=round(t[("comfortable", cap)] / cap * 100, 3),
             cash_shortfall_vs_estr_pct=CASH[cap])
    d["total_standard_pct"] = round(VAR[cap][0] + d["fixed_standard_pct"] + CASH[cap], 2)
    d["not_in_simulated_sharpe_standard_pct"] = round(d["fixed_standard_pct"] + CASH[cap], 2)
    S.append(d)
summary = pd.DataFrame(S)
summary.to_csv(os.path.join(HERE, "ib_gateway_budget_summary.csv"), index=False)

# ----------------------------------------------------------------------------- time
T = [
    ("setup", "IBKR Pro account (IBIE), futures permissions, check each contract is tradeable (KID), 2nd username for the API, paper account", "3-8 h", "+ several days of account approval"),
    ("setup", "VPS: Ubuntu, SSH keys, firewall, Docker, gnzsnz/ib-gateway-docker (IBC), VNC only over SSH tunnel, paper mode first", "4-12 h", ""),
    ("setup", "pysystemtrade + MongoDB/Parquet install; instrument, roll and IB contract config for 14/24/40 markets", "15-40 h", "pysystemtrade ships configs for most of these markets"),
    ("setup", "Port variant 5 (EWMAC, strategy 13 vol regime, portfolio vol targeting) as custom rules; or write a lean ib_async script", "30-100 h", "trend.py is ~200 lines; the hard part is production plumbing, not the formulas"),
    ("setup", "Seed back history (free CSVs + IB expired contracts), splice with backtest data, reconcile live positions vs backtest positions", "15-40 h", ""),
    ("setup", "Paper trading", "8-12 weeks, ~1-2 h/week", "include at least one roll cycle per market and one Gateway upgrade"),
    ("setup", "TOTAL setup", "80-200 h", "for someone fluent in Python; 200-500 h for all 84 markets (audit)"),
    ("recurring", "Approve the weekly 2FA push on IBKR Mobile (Sunday after 01:00 ET or before Monday's first job)", "~2 min/week", "must be reachable; a missed login = no trading until approved"),
    ("recurring", "Read daily email report (spikes, unfilled orders, position breaks)", "5-10 min/trading day", ""),
    ("recurring", "Roll checks and confirmations", "5-15 h/yr", "roll orders/yr 62 (100k), 106 (250k), 169 (1M) from the simulation"),
    ("recurring", "Monthly: statements, FX sweeps, contract chain/expiry config, data fixes", "1-2 h/month", ""),
    ("recurring", "Yearly: margins, costs, contract specs review; IB Gateway/IBC/pysystemtrade upgrades", "10-20 h/yr", ""),
    ("recurring", "TOTAL recurring", "about 1-2 h/week on average, plus incidents", ""),
]
pd.DataFrame(T, columns=["phase", "task", "time", "note"]).to_csv(os.path.join(HERE, "ib_gateway_time.csv"), index=False)

pd.set_option("display.width", 250)
print(summary.T.to_string())
print(budget[budget.item == "TOTAL fixed"].to_string(index=False))
print(f"\n{len(facts)} facts written")
