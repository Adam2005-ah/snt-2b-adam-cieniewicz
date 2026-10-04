"""Le même système de trend sur des sous-ensembles de classes d'actifs.

Usage : python sous_ensembles.py --data DOSSIER_MKT --crypto DOSSIER_CACHE_CRYPTO
Écrit resultats/sous_ensembles_resume.csv, _periodes.csv, _crises.csv, _correlations.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

import listings
import run_backtests as rb
import trend

SUBSETS = {
    "Tout (7 classes)": ["Equities", "Bonds", "STIR", "FX", "Energy", "Metals", "Agriculture"],
    "Énergie + agriculture + métaux + actions": ["Energy", "Agriculture", "Metals", "Equities"],
    "Idem sans minerai de fer ni éthanol": ["Energy", "Agriculture", "Metals", "Equities"],
    "Le reste : obligations + taux courts + devises": ["Bonds", "STIR", "FX"],
    "Matières premières seules (énergie + agriculture + métaux)": ["Energy", "Agriculture", "Metals"],
    "Actions seules": ["Equities"],
}
DUBIOUS = ["SCO1 Comdty", "CUA1 Comdty"]  # données signalées douteuses par la vérification
PERIODS = [("1990", "2026"), ("2000", "2026"), ("2010", "2026"), ("2015", "2026"), ("2023", "2026"),
           ("1990", "1999"), ("2000", "2009"), ("2010", "2019"), ("2020", "2026")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crypto", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats"))
    args = ap.parse_args()

    full, groups, names = rb.load_futures(args.data)
    rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, "1900-01-01"))))
    risk_groups = groups.replace({"Bonds": "Taux", "STIR": "Taux"})
    costs = {c: rb.COSTS[g] for c, g in groups.items()}
    roll_costs = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}

    nets = {}
    for label, classes in SUBSETS.items():
        cols = [c for c in rets.columns if groups[c] in classes]
        if "sans minerai" in label:
            cols = [c for c in cols if c not in DUBIOUS]
        res = trend.run_portfolio(rets[cols], risk_groups[cols], costs, roll_costs, freq="D")
        nets[label] = res["net"].loc[rb.START:]

    summary, periods = {}, {}
    for label, net in nets.items():
        summary[label] = trend.stats(net)
        for a, b in PERIODS:
            s = trend.stats(net.loc[a:b])
            periods[(label, f"{a}-{b}")] = {"rendement annuel": s["rendement annuel"], "volatilité": s["volatilité"],
                                             "sharpe": s["sharpe"], "pire baisse": s["pire baisse"]}
    pd.DataFrame(summary).T.to_csv(f"{args.out}/sous_ensembles_resume.csv", float_format="%.4f")
    per = pd.DataFrame(periods).T
    per.index.names = ["sous-ensemble", "période"]
    per.to_csv(f"{args.out}/sous_ensembles_periodes.csv", float_format="%.4f")

    # Crises, à volatilité égale (10 % par an sur 1990-2026) pour comparer les sous-ensembles entre eux.
    us = {t: rb.load_close(f"{args.data}/us-etf/alm0421/{t}.csv").pct_change() for t in ["QQQ", "GLD"]}
    btc = rb.load_btc(args.crypto).pct_change()
    crises = {}
    for label, net in nets.items():
        scaled = net * 0.10 / (net.std() * 16)
        crises[label] = {k: rb.window_return(scaled, *v) for k, v in rb.CRISES.items()}
    crises["QQQ"] = {k: rb.window_return(us["QQQ"], *v) for k, v in rb.CRISES.items()}
    pd.DataFrame(crises).to_csv(f"{args.out}/sous_ensembles_crises.csv", float_format="%.4f")

    def to_week(daily):
        return daily.dropna().add(1).resample("W-FRI").prod(min_count=1).sub(1)

    weekly = pd.DataFrame({k: to_week(v) for k, v in nets.items()})
    for t in ["QQQ", "GLD"]:
        weekly[t] = to_week(us[t])
    weekly["BTC"] = to_week(btc)
    corr = {}
    for since in ["2000", "2015", "2022"]:
        c = weekly.loc[since:].corr()
        for label in nets:
            corr[(label, f"depuis {since}")] = c.loc[label, ["QQQ", "GLD", "BTC", "Tout (7 classes)"]]
    out = pd.DataFrame(corr).T
    out.index.names = ["sous-ensemble", "période"]
    out.to_csv(f"{args.out}/sous_ensembles_correlations.csv", float_format="%.3f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(summary).T.to_string())
    print(per.round(3).to_string())
    print(pd.DataFrame(crises).round(3).to_string())
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
