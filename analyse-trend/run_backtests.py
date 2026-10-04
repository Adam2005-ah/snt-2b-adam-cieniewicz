"""Lance les backtests de trend following et écrit les résultats dans resultats/.

Usage : python run_backtests.py --data DOSSIER_MKT --crypto DOSSIER_CACHE_CRYPTO
Données : rendements quotidiens de 84 contrats à terme (dépôt GitHub ArturSepp/TrendFollowingSystems,
1959 → 2026-07-10), ETF américains (miroir yfinance alm0421), indices SG Trend et Bloomberg Commodity,
taux des T-bills à 3 mois (FRED DTB3).
"""

import argparse
import os

import numpy as np
import pandas as pd

import trend

# Coût d'une transaction, en fraction du notionnel traité (demi-écart + commission, contrats micro).
COSTS = {"Equities": 1e-4, "Bonds": 1e-4, "STIR": 0.5e-4, "FX": 1e-4,
         "Energy": 2e-4, "Metals": 2e-4, "Agriculture": 4e-4}
ROLLS_PER_YEAR = {"Equities": 4, "Bonds": 4, "STIR": 4, "FX": 4, "Energy": 12, "Metals": 6, "Agriculture": 5}
START = "1990-01-01"   # période principale : la plupart des 84 marchés existent
CRISES = {
    "Krach 1987 (oct.)": ("1987-10-01", "1987-10-31"),
    "Bulle internet (2000-2002)": ("2000-03-24", "2002-10-09"),
    "Crise 2008 (oct. 07 → mars 09)": ("2007-10-31", "2009-03-09"),
    "Covid (19/02 → 23/03/2020)": ("2020-02-19", "2020-03-23"),
    "Inflation 2022": ("2022-01-01", "2022-12-31"),
    "Droits de douane (19/02 → 08/04/2025)": ("2025-02-19", "2025-04-08"),
    "Baisse QQQ (28/01 → 30/03/2026)": ("2026-01-28", "2026-03-30"),
}


def load_futures(data):
    base = f"{data}/alternatives/futures_sepp"
    rets = pd.read_csv(f"{base}/futures84_usd_returns_daily.csv", index_col=0, parse_dates=True)
    desc = pd.read_csv(f"{base}/futures84_descriptive.csv").set_index("ticker")
    groups = desc["group_data"].reindex(rets.columns)
    return rets, groups, desc["names"].reindex(rets.columns)


def load_close(path, col="adj_close"):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[col].dropna()


def load_btc(crypto):
    cm = pd.read_csv(f"{crypto}/cm_btc.csv", usecols=["time", "PriceUSD"], parse_dates=["time"])
    cm = cm.set_index("time")["PriceUSD"].dropna()
    tr = pd.read_csv(f"{crypto}/tracker_historical.csv", parse_dates=["fetch_date"])
    tr = tr[tr["id"] == "bitcoin"].set_index("fetch_date")["current_price"]
    btc = pd.concat([cm, tr[tr.index > cm.index.max()]]).sort_index()
    return btc[btc.index.dayofweek < 5]  # jours ouvrés, comme les autres actifs


def yearly(daily):
    return (1 + daily.dropna()).groupby(daily.dropna().index.year).prod() - 1


def window_return(daily, start, end):
    d = daily.loc[start:end].dropna()
    return (1 + d).prod() - 1 if len(d) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crypto", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rets, groups, names = load_futures(args.data)
    costs = {c: COSTS[g] for c, g in groups.items()}
    # Pour répartir le risque, taux courts et obligations forment une seule classe « Taux ».
    risk_groups = groups.replace({"Bonds": "Taux", "STIR": "Taux"})
    roll_costs = {c: 2 * COSTS[g] * ROLLS_PER_YEAR[g] for c, g in groups.items()}

    # 1. Portefeuille multi-marchés : quotidien, hebdo, mensuel, et variante momentum 12 mois.
    runs = {}
    for freq, label in [("D", "quotidien"), ("W", "hebdomadaire"), ("M", "mensuel")]:
        runs[f"EWMAC {label}"] = trend.run_portfolio(rets, risk_groups, costs, roll_costs, freq=freq)
    runs["Momentum 12 mois, mensuel"] = trend.run_portfolio(rets, risk_groups, costs, roll_costs, freq="M",
                                                            forecast_fn=trend.momentum_12m)
    triple = {c: 3 * v for c, v in costs.items()}
    runs["EWMAC quotidien, coûts x3"] = trend.run_portfolio(
        rets, risk_groups, triple, {c: 3 * v for c, v in roll_costs.items()}, freq="D")

    sg = pd.read_csv(f"{args.data}/alternatives/pofo_indices/SG_Trend_Index_daily.csv", parse_dates=["date"])
    sg = sg.set_index("date")["close"].pct_change()
    tbill = load_close(f"{args.data}/us-etf/alm0421_macro/DTB3.csv", "value") / 100 / 252
    sixty40 = pd.read_csv(f"{args.data}/alternatives/futures_sepp/benchmark_SGTrend_6040_daily.csv",
                          index_col=0, parse_dates=True)["60/40 Equity/Bond"].pct_change()

    summary = {}
    for name, res in runs.items():
        net = res["net"].loc[START:]
        summary[name] = trend.stats(net)
        summary[name]["coûts/an"] = (res["trading_costs"] + res["roll_costs"]).loc[START:].mean() * 252
        summary[name]["levier moyen"] = res["positions"].abs().sum(axis=1).loc[START:].mean()
    summary["Indice SG Trend (frais déduits, excès du monétaire)"] = trend.stats(
        (sg - tbill.reindex(sg.index).ffill()).loc["2000-01-04":])
    for since in ["2000-01-04", "2015-01-01", "2023-01-01"]:
        summary[f"EWMAC quotidien depuis {since[:4]}"] = trend.stats(runs["EWMAC quotidien"]["net"].loc[since:])
        sgx = (sg - tbill.reindex(sg.index).ffill()).loc[since:]
        summary[f"SG Trend depuis {since[:4]}"] = trend.stats(sgx)
    pd.DataFrame(summary).T.to_csv(f"{args.out}/portefeuille_resume.csv", float_format="%.4f")

    main_net = runs["EWMAC quotidien"]["net"]
    def to_week(daily):
        return daily.dropna().add(1).resample("W-FRI").prod(min_count=1).sub(1)

    us = {t: load_close(f"{args.data}/us-etf/alm0421/{t}.csv").pct_change() for t in ["QQQ", "GLD"]}
    btc = load_btc(args.crypto).pct_change()
    weekly = pd.DataFrame({"Trend (ce système)": to_week(main_net.loc[START:]), "SG Trend": to_week(sg),
                           "60/40 actions/obligations": to_week(sixty40.loc[START:]), "QQQ": to_week(us["QQQ"]),
                           "GLD": to_week(us["GLD"]), "BTC": to_week(btc)})
    corr = {}
    for since in ["1990", "2000", "2015", "2022"]:
        corr[f"depuis {since}"] = weekly.loc[since:].corr()["Trend (ce système)"]
    pd.DataFrame(corr).to_csv(f"{args.out}/portefeuille_correlations_hebdo.csv", float_format="%.3f")

    years = pd.DataFrame({"Trend (ce système)": yearly(main_net.loc[START:]), "SG Trend": yearly(sg.loc["2000":]),
                          "60/40": yearly(sixty40.loc[START:]), "QQQ": yearly(us["QQQ"])})
    years.to_csv(f"{args.out}/portefeuille_annees.csv", float_format="%.4f")

    crises = {k: {"Trend (ce système)": window_return(main_net, *v), "SG Trend": window_return(sg, *v),
                  "60/40": window_return(sixty40, *v), "QQQ": window_return(us["QQQ"], *v),
                  "BTC": window_return(btc, *v)} for k, v in CRISES.items()}
    pd.DataFrame(crises).T.to_csv(f"{args.out}/portefeuille_crises.csv", float_format="%.4f")

    by_class = {}
    for g in groups.unique():
        cols = groups.index[groups == g]
        res = runs["EWMAC quotidien"]
        pnl = (res["positions"][cols].shift(1) * rets[cols].fillna(0)).sum(axis=1).loc[START:]
        by_class[g] = {"contribution annuelle": pnl.mean() * 252, "sharpe de la poche": pnl.mean() / pnl.std() * 16,
                       "marchés": len(cols)}
    pd.DataFrame(by_class).T.to_csv(f"{args.out}/portefeuille_par_classe.csv", float_format="%.4f")

    curves = pd.DataFrame({k: (1 + v["net"].loc[START:].fillna(0)).cumprod() for k, v in runs.items()})
    curves["SG Trend (excès du monétaire)"] = (1 + (sg - tbill.reindex(sg.index).ffill()).loc["2000-01-04":]
                                               .fillna(0)).cumprod()
    curves["60/40 actions/obligations"] = (1 + sixty40.loc[START:].fillna(0)).cumprod()
    curves.to_csv(f"{args.out}/courbes_portefeuille.csv", float_format="%.5f")

    # 2. Un seul actif : PDBC, l'indice Bloomberg Commodity depuis 1991, et un filtre de tendance sur QQQ/GLD/BTC.
    pdbc = load_close(f"{args.data}/us-etf/alm0421/PDBC.csv") if os.path.exists(
        f"{args.data}/us-etf/alm0421/PDBC.csv") else None
    if pdbc is None:
        panel = pd.read_csv(f"{args.data}/us-etf/panel_alm0421_adjclose_daily_1990on.csv", index_col=0, parse_dates=True)
        pdbc = panel["PDBC"].dropna()
    bcom = pd.read_csv(f"{args.data}/commodities/indices/pofo_BCOM-ER-USD_daily.csv", comment="#",
                       parse_dates=["date"]).set_index("date")["close"]
    singles = {"PDBC": pdbc.pct_change(), "QQQ": us["QQQ"], "GLD": us["GLD"], "BTC": btc}
    cash_daily = tbill.reindex(pd.date_range("1954-01-01", "2026-12-31", freq="B")).ffill()
    singles["BCOM (indice, depuis 1991)"] = bcom.pct_change() + cash_daily.reindex(bcom.index)
    single_rows, single_curves = {}, {}
    for name, r in singles.items():
        r = r.dropna()
        cash = cash_daily.reindex(r.index).ffill().fillna(0)
        start = r.index[trend.WARMUP_DAYS + 1]
        single_rows[(name, "achat-conservation", "")] = trend.stats(r.loc[start:])
        single_curves[f"{name} achat-conservation"] = (1 + r.loc[start:]).cumprod()
        for mode, mode_label in [("long_cash", "tendance : investi ou monétaire"),
                                 ("long_short", "tendance : achat/vente à découvert")]:
            for freq, flabel in [("D", "quotidien"), ("W", "hebdo"), ("M", "mensuel")]:
                res = trend.run_single(r, cash, freq=freq, mode=mode)
                single_rows[(name, mode_label, flabel)] = trend.stats(res["net"].loc[start:])
                single_rows[(name, mode_label, flabel)]["exposition moyenne"] = res["positions"].loc[start:].mean()
                if freq == "D":
                    single_curves[f"{name} {mode_label}"] = (1 + res["net"].loc[start:].fillna(0)).cumprod()
    single = pd.DataFrame(single_rows).T
    single.index.names = ["actif", "stratégie", "fréquence"]
    single.to_csv(f"{args.out}/actif_seul.csv", float_format="%.4f")
    pd.DataFrame(single_curves).to_csv(f"{args.out}/courbes_actif_seul.csv", float_format="%.5f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(summary).T.to_string())
    print(pd.DataFrame(corr).round(2).to_string())
    print(pd.DataFrame(crises).T.round(3).to_string())
    print(pd.DataFrame(by_class).T.round(3).to_string())
    print(single.round(3).to_string())


if __name__ == "__main__":
    main()
