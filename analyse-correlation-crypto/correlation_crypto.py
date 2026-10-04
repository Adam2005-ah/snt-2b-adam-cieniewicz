"""Matrice de corrélation du top 50 crypto (hors stablecoins) et comparaison avec le BTC.

Sources (accessibles via raw.githubusercontent.com) :
  - Coin Metrics community data (github.com/coinmetrics/data) : historique journalier long,
    dernière date 2026-05-23. Colonne PriceUSD quand elle existe, sinon CapMrktEstUSD
    (capitalisation flottante) utilisée comme proxy des rendements de prix.
  - daily-crypto-tracker-dataset (github.com/UrvishAhir1/daily-crypto-tracker-dataset) :
    instantané quotidien du top 100 CoinGecko, du 2026-04-21 au 2026-10-03. Sert au
    classement actuel et prolonge les séries après 2026-05-23.

Méthode (Carver, Advanced Futures Trading Strategies) : corrélations calculées sur les
rendements (log), pas sur les prix, en hebdomadaire (semaines clôturées le samedi).

Usage : python correlation_crypto.py [--cache DOSSIER] [--out DOSSIER]
"""

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

CM_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{}.csv"
TRACKER_URL = ("https://raw.githubusercontent.com/UrvishAhir1/daily-crypto-tracker-dataset/main/"
               "top_100_cryptocurrency_market_{}.csv")

# Stablecoins, fonds monétaires tokenisés, or tokenisé : corrélation sans intérêt.
EXCLUDED_IDS = {
    "tether", "usd-coin", "figure-heloc", "usds", "ethena-usde", "dai", "usd1-wlfi", "tether-gold",
    "global-dollar", "paypal-usd", "ripple-usd", "hashnote-usyc", "ondo-us-dollar-yield",
    "blackrock-usd-institutional-digital-liquidity-fund", "pax-gold", "falcon-finance",
    "united-stables", "spiko-amundi-overnight-swap-fund-eur", "usdd", "bfusd", "usdgo",
    "superstate-short-duration-us-government-securities-fund-ustb", "gho", "open-usd",
    "blockchain-capital",
}

# id CoinGecko -> code Coin Metrics (absent = pas de série Coin Metrics).
CM_CODES = {
    "bitcoin": "btc", "ethereum": "eth", "binancecoin": "bnb", "ripple": "xrp", "solana": "sol",
    "tron": "trx", "zcash": "zec", "hyperliquid": "hype", "dogecoin": "doge", "monero": "xmr",
    "chainlink": "link", "cardano": "ada", "leo-token": "leo", "stellar": "xlm",
    "bitcoin-cash": "bch", "near": "near", "uniswap": "uni", "litecoin": "ltc",
    "avalanche-2": "avax", "sui": "sui", "hedera-hashgraph": "hbar", "the-open-network": "ton",
    "quant-network": "qnt", "shiba-inu": "shib", "crypto-com-chain": "cro", "pump-fun": "pump",
    "aave": "aave", "okb": "okb", "ethena": "ena", "ondo-finance": "ondo",
}
TICKER_OVERRIDE = {"the-open-network": "TON"}

TOP_N = 50
CM_END = pd.Timestamp("2026-05-23")      # dernière date avec PriceUSD dans Coin Metrics
LONG_START_MAX = pd.Timestamp("2024-12-14")  # une crypto entre dans P1 si son historique commence avant
JUMP_IDIO = 0.30   # proxy cap. flottante : |r - r_btc| au-delà = saut d'offre (unlock/burn), pas un prix
JUMP_ABS = 0.25
PROXY_MIN_CORR = 0.5  # proxy rejeté s'il colle mal aux prix CoinGecko sur la période commune
WEEK = "W-SAT"
SATURDAY = pd.offsets.Week(weekday=5)


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_tracker(cache):
    hist = pd.read_csv(fetch(TRACKER_URL.format("historical"), f"{cache}/tracker_historical.csv"))
    latest = pd.read_csv(fetch(TRACKER_URL.format("latest"), f"{cache}/tracker_latest.csv"))
    hist["fetch_date"] = pd.to_datetime(hist["fetch_date"])
    prices = hist.pivot_table(index="fetch_date", columns="id", values="current_price", aggfunc="last")
    return latest, prices


def load_cm(cache, code):
    df = pd.read_csv(fetch(CM_URL.format(code), f"{cache}/cm_{code}.csv"), low_memory=False)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    if "PriceUSD" in df and df["PriceUSD"].notna().any():
        return df["PriceUSD"].dropna(), "Coin Metrics PriceUSD"
    if "CapMrktEstUSD" in df and df["CapMrktEstUSD"].notna().any():
        return df["CapMrktEstUSD"].dropna(), "Coin Metrics cap. flottante (proxy)"
    return None, None


def log_returns(series):
    s = series[series > 0]
    return np.log(s).diff().dropna()


def build_returns(universe, tracker_prices, cache):
    """Rendements log journaliers : Coin Metrics jusqu'à CM_END, CoinGecko (tracker) ensuite."""
    tracker_ret = {cid: log_returns(tracker_prices[cid].dropna()) for cid in universe.index}
    btc_cm, _ = load_cm(cache, "btc")
    btc_ret = log_returns(btc_cm)
    returns, info, jumps = {}, {}, []
    for cid, row in universe.iterrows():
        tr = tracker_ret[cid]
        source = "CoinGecko (tracker) uniquement"
        cm_ret = None
        code = CM_CODES.get(cid)
        if code:
            level, cm_source = load_cm(cache, code)
            if level is not None:
                cm_ret = log_returns(level[:CM_END])
                if "proxy" in cm_source:
                    overlap = pd.concat([cm_ret, tr], axis=1, join="inner")
                    if len(overlap) > 20 and overlap.corr().iloc[0, 1] < PROXY_MIN_CORR:
                        cm_ret = None
                    else:
                        idio = cm_ret - btc_ret.reindex(cm_ret.index)
                        flag = (idio.abs() > JUMP_IDIO) & (cm_ret.abs() > JUMP_ABS)
                        jumps += [(row.ticker, d.date(), round(v, 2)) for d, v in cm_ret[flag].items()]
                        cm_ret = cm_ret.mask(flag)
                if cm_ret is not None:
                    source = cm_source + " + CoinGecko après " + str(CM_END.date())
        if cm_ret is not None:
            r = pd.concat([cm_ret, tr[tr.index > CM_END]])
        else:
            r = tr
        returns[row.ticker] = r
        info[row.ticker] = {"rang": int(row["rank"]), "nom": row["name"], "source": source}
    return pd.DataFrame(returns).sort_index(), info, jumps


def weekly(daily):
    """Somme des rendements log par semaine (samedi), NaN avant le début de la série."""
    started = daily.notna().cummax()
    wk = daily.fillna(0).where(started).resample(WEEK).sum(min_count=1)
    for col in wk:  # la première semaine est incomplète
        wk.loc[wk.index <= SATURDAY.rollforward(daily[col].first_valid_index()), col] = np.nan
    return wk


def corr_table(wk_all, tickers, start, end):
    wk = wk_all.loc[(wk_all.index > start) & (wk_all.index <= end), tickers]
    return wk.corr(), wk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "resultats"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    latest, tracker_prices = load_tracker(args.cache)
    asof = latest["fetch_date"].iloc[0]
    universe = latest[latest["rank"] <= TOP_N]
    universe = universe[~universe["id"].isin(EXCLUDED_IDS)].set_index("id")
    universe["ticker"] = [TICKER_OVERRIDE.get(i, s.upper()) for i, s in universe["symbol"].items()]

    daily, info, jumps = build_returns(universe, tracker_prices, args.cache)
    end = daily.index.max()
    wk = weekly(daily)

    # Historique disponible par crypto (premier rendement = 2e jour de prix).
    cover = pd.DataFrame({
        t: {**info[t], "debut": daily[t].first_valid_index().date(),
            "fin": daily[t].last_valid_index().date(), "jours": int(daily[t].notna().sum()),
            "semaines": int(wk[t].notna().sum())}
        for t in daily.columns}).T
    cover.index.name = "ticker"
    cover.to_csv(f"{args.out}/historique_donnees.csv")

    # P1 : période commune longue, cryptos dont l'historique commence avant LONG_START_MAX.
    long_tk = [t for t in daily.columns if daily[t].first_valid_index() <= LONG_START_MAX]
    p1_start = SATURDAY.rollforward(max(daily[t].first_valid_index() for t in long_tk))
    p1_corr, p1_wk = corr_table(wk, long_tk, p1_start, end)
    p1_corr.round(3).to_csv(f"{args.out}/matrice_P1_hebdo.csv")

    # P2 : fenêtre récente commune à (presque) toutes les cryptos, 100 % CoinGecko (même horodatage).
    tr_daily = pd.DataFrame({info_t: log_returns(tracker_prices[cid].dropna())
                             for cid, info_t in universe["ticker"].items()})
    p2_start = tracker_prices.index.min()
    p2_tk = [t for t in tr_daily.columns if tr_daily[t].first_valid_index() <= p2_start + pd.Timedelta(days=1)]
    p2_daily = tr_daily[p2_tk]
    p2_corr_d = p2_daily.corr()
    tr_wk = np.log(tracker_prices[[c for c in universe.index if universe.loc[c, "ticker"] in p2_tk]]
                   .rename(columns=universe["ticker"])).resample(WEEK).last().diff().iloc[1:]
    p2_corr_w = tr_wk.corr()
    p2_corr_d.round(3).to_csv(f"{args.out}/matrice_P2_journalier.csv")
    p2_corr_w.round(3).to_csv(f"{args.out}/matrice_P2_hebdo.csv")

    # Comparaison avec le BTC.
    last52_start = end - pd.Timedelta(weeks=52)
    rows = {}
    for t in daily.columns:
        full = wk[["BTC", t]].dropna()
        r = {"rang": info[t]["rang"], "debut_historique": cover.loc[t, "debut"],
             "corr_btc_tout_historique": full.corr().iloc[0, 1] if t != "BTC" else 1.0,
             "semaines_tout_historique": len(full)}
        if t in long_tk:
            x = p1_wk[t]
            b = p1_wk["BTC"]
            r["corr_btc_P1"] = p1_corr.loc[t, "BTC"]
            r["beta_btc_P1"] = x.cov(b) / b.var()
            r["vol_annuelle_P1"] = x.std() * np.sqrt(52)
            r["corr_btc_52_semaines"] = p1_wk[p1_wk.index > last52_start][[t, "BTC"]].corr().iloc[0, 1]
        if t in p2_tk:
            r["corr_btc_P2_journalier"] = p2_corr_d.loc[t, "BTC"]
            r["corr_btc_P2_hebdo"] = p2_corr_w.loc[t, "BTC"]
        rows[t] = r
    vs_btc = pd.DataFrame(rows).T
    num = vs_btc.columns.drop("debut_historique")
    vs_btc[num] = vs_btc[num].apply(pd.to_numeric)
    vs_btc.index.name = "ticker"
    vs_btc.to_csv(f"{args.out}/correlation_vs_btc.csv", float_format="%.3f")

    with open(f"{args.out}/resume.txt", "w") as f:
        f.write(f"Classement CoinGecko au {asof} ; données jusqu'au {end.date()}\n")
        f.write(f"P1 : semaines du {p1_start.date()} (excl.) au {end.date()} -> {len(p1_wk)} rendements hebdo, "
                f"{len(long_tk)} cryptos\n")
        f.write(f"P2 : {p2_start.date()} -> {end.date()} -> {p2_daily.notna().all(axis=1).sum()} rendements "
                f"journaliers, {len(tr_wk)} hebdo, {len(p2_tk)} cryptos\n")
        f.write(f"52 dernières semaines : après le {last52_start.date()}\n")
        f.write("Sauts d'offre neutralisés (proxy cap. flottante) : " + repr(jumps) + "\n")
        f.write("Hors P2 (historique trop court) : " + repr(sorted(set(daily.columns) - set(p2_tk))) + "\n")
    print(open(f"{args.out}/resume.txt").read())


if __name__ == "__main__":
    main()
