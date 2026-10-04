"""Pistes pour améliorer le Sharpe du système de trend, comparées au système de base.

- Taille selon la volatilité : stratégie treize de Carver (régime de volatilité par marché) et ciblage
  de volatilité du portefeuille entier.
- Taille selon la performance passée : réduire après une année perdante, suivre la courbe de capital
  (moyenne 200 jours), ou l'inverse (augmenter après les pertes).
Chaque multiplicateur n'utilise que des données connues la veille.

Usage : python ameliorations.py --data DOSSIER_MKT
Écrit resultats/ameliorations.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

import listings
import run_backtests as rb
import trend

PERIODS = [("1990", "2026"), ("2000", "2026"), ("1990", "1999"), ("2000", "2009"), ("2010", "2019"),
           ("2020", "2026"), ("2023", "2026")]


def summarise(net, base):
    row = {f"sharpe {a}-{b}": net.loc[a:b].mean() / net.loc[a:b].std() * 16 for a, b in PERIODS}
    scaled = net * base.std() / net.std()  # même volatilité que le système de base
    curve = (1 + scaled).cumprod()
    row["rendement annuel à vol égale"] = curve.iloc[-1] ** (252 / len(scaled)) - 1
    row["pire baisse à vol égale"] = (curve / curve.cummax() - 1).min()
    row["skew mensuel"] = ((1 + net).resample("ME").prod() - 1).skew()
    diff = (scaled - base).loc["2000":]
    row["t-stat de l'écart depuis 2000"] = (diff.mean() / diff.std() * np.sqrt(len(diff))
                                            if diff.std() > 1e-12 else np.nan)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats"))
    args = ap.parse_args()

    full, groups, _ = rb.load_futures(args.data)
    rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, "1900-01-01"))))
    risk_groups = groups.replace({"Bonds": "Taux", "STIR": "Taux"})
    costs = {c: rb.COSTS[g] for c, g in groups.items()}
    roll_costs = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}

    def run(**kw):
        return trend.run_portfolio(rets, risk_groups, costs, roll_costs, **kw)["net"]

    base = run()
    equity = (1 + base).cumprod()
    last_year = base.rolling(252).sum()
    variants = {
        "Système de base": base,
        "Stratégie 13 (régime de volatilité)": run(forecast_fn=trend.forecast_regime),
        "Ciblage de volatilité du portefeuille": trend.run_vol_targeted(rets, risk_groups, costs, roll_costs)["net"],
        "Les deux": trend.run_vol_targeted(rets, risk_groups, costs, roll_costs,
                                           forecast_fn=trend.forecast_regime)["net"],
        "Moitié de la taille après une année perdante": run(scale=pd.Series(np.where(last_year > 0, 1.0, 0.5),
                                                                          index=base.index).shift(1)),
        "Moitié sous la moyenne 200 jours du capital": run(scale=pd.Series(
            np.where(equity > equity.rolling(200).mean(), 1.0, 0.5), index=base.index).shift(1)),
        "Inverse : x1,5 après une année perdante": run(scale=pd.Series(np.where(last_year < 0, 1.5, 1.0),
                                                                      index=base.index).shift(1)),
    }
    base = base.loc[rb.START:]
    table = pd.DataFrame({k: summarise(v.loc[rb.START:], base) for k, v in variants.items()}).T
    table.to_csv(f"{args.out}/ameliorations.csv", float_format="%.3f")
    pd.set_option("display.width", 250)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
