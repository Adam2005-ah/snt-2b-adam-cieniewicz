"""Candidats de diversification pour un portefeuille QQQ + GLD + BTC.

Pour chaque candidat : corrélation hebdomadaire (rendements log, semaines closes le vendredi)
avec QQQ, GLD et BTC sur trois fenêtres, comportement pendant les pires semaines de chacun,
volatilité, rendement annualisé et pire baisse. Puis une illustration : le portefeuille
QQQ/GLD/BTC à risque égal, avec ou sans les meilleurs diversifiants.

Données : miroirs GitHub de prix quotidiens (yfinance, indices SG/BCOM, VNI de fonds cat bonds),
voir sources.csv. Usage : python diversification.py --data DOSSIER_MKT --crypto DOSSIER_CACHE_CRYPTO
"""

import argparse
import os

import numpy as np
import pandas as pd

WEEK = "W-FRI"
END = pd.Timestamp("2026-09-25")  # dernière semaine complète des ETF US
WINDOWS = {"2015-2026": pd.Timestamp("2015-01-02"), "depuis 2022": pd.Timestamp("2021-12-31"),
           "52 sem.": END - pd.Timedelta(weeks=52)}
MIN_WEEKS = 52
TAIL = 0.05  # pires 5 % des semaines

# (nom affiché, famille, fichier relatif au dossier de données, colonne de prix)
CANDIDATES = [
    ("Trend SG (indice CTA)", "Trend following", "alternatives/pofo_indices/SG_Trend_Index_daily.csv", "close"),
    ("DBMF", "Trend following", "us", "DBMF"),
    ("KMLM", "Trend following", "us", "KMLM"),
    ("QSPIX (style premia)", "Primes alternatives", "us", "QSPIX"),
    ("Cat bonds (GAM Star, EUR)", "Cat bonds", "alternatives/cat_bond_ils/GAM_Star_Cat_Bond_EURhedged_Acc_NAV_weekly_real.csv", "nav"),
    ("BIL (T-bills)", "Taux", "us", "BIL"),
    ("STIP (TIPS courts)", "Taux", "us", "STIP"),
    ("TLT (Trésor 20 ans+)", "Taux", "us", "TLT"),
    ("MNA (merger arb)", "Primes alternatives", "us", "MNA"),
    ("BCOM ER (matières premières)", "Matières premières", "commodities/indices/pofo_BCOM-ER-USD_daily.csv", "close"),
    ("PDBC (mat. premières optimisé)", "Matières premières", "us", "PDBC"),
    ("BNO (Brent)", "Matières premières", "us", "BNO"),
    ("UNG (gaz naturel)", "Matières premières", "us", "UNG"),
    ("KRBN (carbone)", "Matières premières", "commo", "KRBN"),
    ("URNM (mines d'uranium)", "Matières premières", "commo", "URNM"),
    ("CPER (cuivre)", "Matières premières", "us", "CPER"),
    ("SLV (argent)", "Matières premières", "us", "SLV"),
    ("XLU (services publics US)", "Actions défensives", "us", "XLU"),
    ("XLP (conso. de base US)", "Actions défensives", "us", "XLP"),
    ("XLV (santé US)", "Actions défensives", "us", "XLV"),
    ("INDA (Inde)", "Émergents", "us", "INDA"),
    ("EWZ (Brésil)", "Émergents", "us", "EWZ"),
    ("TUR (Turquie)", "Émergents", "intl", "TUR"),
    ("ARGT (Argentine)", "Émergents", "intl", "ARGT"),
    ("VNM (Vietnam)", "Émergents", "intl", "VNM"),
    ("EWJ (Japon)", "Actions", "us", "EWJ"),
    ("FXY (yen)", "Devises refuges", "us", "FXY"),
    ("FXF (franc suisse)", "Devises refuges", "us", "FXF"),
    ("BTAL (anti-bêta)", "Couvertures", "us", "BTAL"),
    ("TAIL (couverture de queue)", "Couvertures", "us", "TAIL"),
    ("VIXY (long VIX)", "Couvertures", "us", "VIXY"),
    ("SVXY (vente de volatilité)", "Pièges", "us", "SVXY"),
]


def load_csv_series(path, col):
    df = pd.read_csv(path, comment="#")
    date_col = df.columns[0]
    s = pd.Series(df[col].values, index=pd.to_datetime(df[date_col]), dtype=float)
    return s.dropna().sort_index()


def load_btc(crypto_cache):
    cm = pd.read_csv(f"{crypto_cache}/cm_btc.csv", usecols=["time", "PriceUSD"], parse_dates=["time"])
    cm = cm.set_index("time")["PriceUSD"].dropna()
    tr = pd.read_csv(f"{crypto_cache}/tracker_historical.csv", parse_dates=["fetch_date"])
    tr = tr[tr["id"] == "bitcoin"].set_index("fetch_date")["current_price"]
    return pd.concat([cm, tr[tr.index > cm.index.max()]]).sort_index()


def weekly_returns(prices):
    wk = np.log(prices.resample(WEEK).last())
    return wk.diff().where(wk.notna() & wk.shift(1).notna())


def max_drawdown(r):
    curve = r.fillna(0).cumsum()
    return float(np.expm1((curve - curve.cummax()).min()))


def stats(r, ref, start):
    """Indicateurs d'un candidat r (rendements hebdo log) face aux séries de référence ref."""
    out = {}
    for label, s in WINDOWS.items():
        for name in ["QQQ", "GLD", "BTC", "Portefeuille"]:
            pair = pd.concat([r, ref[name]], axis=1).loc[s:END].dropna()
            out[f"corr {name} {label}"] = pair.corr().iloc[0, 1] if len(pair) >= MIN_WEEKS else np.nan
    x = r.loc[start:END].dropna()
    for name in ["QQQ", "GLD", "BTC"]:
        pair = pd.concat([r, ref[name]], axis=1, keys=["c", "r"]).loc[start:END].dropna()
        worst = pair["r"] <= pair["r"].quantile(TAIL)
        out[f"pires sem. {name}"] = np.expm1(pair.loc[worst, "c"]).mean() if len(pair) >= MIN_WEEKS else np.nan
    years = len(x) / 52
    out["début"] = x.index.min().date() if len(x) else None
    out["rendement annuel"] = np.expm1(x.sum() / years) if len(x) else np.nan
    out["volatilité"] = x.std() * np.sqrt(52)
    out["pire baisse"] = max_drawdown(x)
    return out


def inverse_vol_portfolio(rets, lookback=26):
    """Rebalancement hebdo à risque égal, volatilités estimées sur le passé uniquement."""
    vol = rets.rolling(lookback, min_periods=lookback).std().shift(1)
    w = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
    simple = np.expm1(rets)
    return np.log1p((w * simple).sum(axis=1, min_count=1)).where(w.notna().all(axis=1))


def perf(r, rf, target_vol):
    """Sharpe sur l'excès de rendement, puis résultat ramené à la même volatilité (levier implicite)."""
    simple = np.expm1(r).dropna()
    cash = np.expm1(rf).reindex(simple.index).fillna(0)
    excess = simple - cash
    vol = excess.std() * np.sqrt(52)
    lev = target_vol / vol
    scaled = np.log1p(cash + lev * excess)
    return {"sharpe": excess.mean() * 52 / vol, "volatilité brute": vol, "levier pour viser la vol": lev,
            "rendement annuel à vol égale": np.expm1(scaled.mean() * 52), "pire baisse à vol égale": max_drawdown(scaled),
            "début": simple.index.min().date()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crypto", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "resultats"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    panels = {
        "us": pd.read_csv(f"{args.data}/us-etf/panel_alm0421_adjclose_daily_1990on.csv", index_col=0, parse_dates=True),
        "intl": pd.read_csv(f"{args.data}/international/panel_intl_etf_adjclose_daily.csv", index_col=0, parse_dates=True),
        "commo": pd.read_csv(f"{args.data}/commodities/commodity_etfs_adjclose_daily_wide.csv", index_col=0, parse_dates=True),
    }

    ref = pd.DataFrame({
        "QQQ": weekly_returns(panels["us"]["QQQ"].dropna()),
        "GLD": weekly_returns(panels["us"]["GLD"].dropna()),
        "BTC": weekly_returns(load_btc(args.crypto)),
    })
    ref["Portefeuille"] = inverse_vol_portfolio(ref[["QQQ", "GLD", "BTC"]])
    start = WINDOWS["2015-2026"]

    rows, rets = {}, {}
    for name in ["QQQ", "GLD", "BTC"]:
        rows[name] = {"famille": "Déjà détenu", **stats(ref[name], ref, start)}
    for name, family, src, col in CANDIDATES:
        prices = panels[src][col].dropna() if src in panels else load_csv_series(f"{args.data}/{src}", col)
        rets[name] = weekly_returns(prices)
        rows[name] = {"famille": family, **stats(rets[name], ref, start)}
    table = pd.DataFrame(rows).T
    table.index.name = "candidat"
    table.to_csv(f"{args.out}/candidats_hebdo.csv", float_format="%.3f")

    # Illustration : ajout à risque égal des diversifiants les plus solides.
    rets = pd.DataFrame(rets)
    base = ref[["QQQ", "GLD", "BTC"]]
    # Les cat bonds sont exclus : leur VNI lissée sous-estime le risque (saut de -10 à -15 % après un
    # gros ouragan) et la parité de risque leur donnerait un levier absurde (x8,6).
    trend, commo, brent = "Trend SG (indice CTA)", "BCOM ER (matières premières)", "BNO (Brent)"
    combos = {
        "QQQ + GLD + BTC": base,
        "+ trend": base.join(rets[[trend]]),
        "+ mat. premières": base.join(rets[[commo]]),
        "+ trend + mat. premières": base.join(rets[[trend, commo]]),
        "+ trend + mat. premières + Brent": base.join(rets[[trend, commo, brent]]),
    }
    p_start = pd.Timestamp("2015-07-03")  # 26 semaines d'historique de volatilité après 2015
    p_end = pd.Timestamp("2026-08-28")    # dernière semaine de l'indice SG Trend
    rf = rets["BIL (T-bills)"]
    port = {k: inverse_vol_portfolio(v.loc[:p_end]).loc[p_start:p_end] for k, v in combos.items()}
    target = np.expm1(port["QQQ + GLD + BTC"]).std() * np.sqrt(52)
    illus = {k: perf(v, rf, target) for k, v in port.items()}
    pd.DataFrame(illus).T.to_csv(f"{args.out}/illustration_portefeuille.csv", float_format="%.3f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(table.round(2).to_string())
    print(pd.DataFrame(illus).T.round(3).to_string())


if __name__ == "__main__":
    main()
