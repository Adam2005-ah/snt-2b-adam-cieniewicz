"""Graphiques des backtests (lit resultats/, écrit graphiques/). Usage : python graphiques.py"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "resultats"), os.path.join(HERE, "graphiques")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def fr(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def growth_and_drawdown(curves, colors, short, title, subtitle, path, log=True):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    for name, s in curves.items():
        s = s.dropna()
        top.plot(s.index, s.values, color=colors[name], linewidth=2, label=name)
        top.annotate(f"{short[name]} ×{fr(s.iloc[-1])}", (s.index[-1], s.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK)
        dd = s / s.cummax() - 1
        bottom.plot(dd.index, dd.values * 100, color=colors[name], linewidth=1.5)
    if log:
        top.set_yscale("log")
        top.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
        top.yaxis.set_minor_locator(NullLocator())
        top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 0 if v >= 1 else 1)))
    top.set_ylabel("Valeur de 1 € investi")
    fig.text(0.07, 0.965, title, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.07, 0.93, subtitle, fontsize=9, color=INK_2, va="top")
    top.legend(loc="upper left", frameon=False)
    bottom.set_ylabel("Baisse depuis\nle plus haut (%)")
    bottom.axhline(0, color=INK_2, linewidth=0.8)
    fig.subplots_adjust(left=0.1, right=0.82, top=0.88)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)

    port = pd.read_csv(f"{RES}/courbes_portefeuille.csv", index_col=0, parse_dates=True)
    start = "2000-01-04"
    curves = {
        "Ce système (84 marchés)": port["EWMAC quotidien, volatilité de SG Trend"],
        "Indice SG Trend (vrais fonds)": port["SG Trend (excès du monétaire)"],
        "60/40 actions/obligations": port["60/40 actions/obligations"],
    }
    curves = {k: (v.loc[start:] / v.loc[start:].dropna().iloc[0]) for k, v in curves.items()}
    growth_and_drawdown(
        curves, dict(zip(curves, [BLUE, ORANGE, AQUA])), dict(zip(curves, ["Système", "SG Trend", "60/40"])),
        "Trend following multi-marchés, 2000 → juillet 2026",
        "Rendement au-dessus du monétaire, coûts inclus. Système ramené à la volatilité de l'indice SG Trend (13 %), qui est net des frais des fonds.",
        f"{OUT}/portefeuille_2000_2026.png")

    crises = pd.read_csv(f"{RES}/portefeuille_crises.csv", index_col=0)
    crises = crises.drop(index=[c for c in crises.index if "1987" in c])[
        ["Trend (ce système, volatilité de SG Trend)", "SG Trend", "QQQ", "60/40"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    n, height = len(crises.columns), 0.2
    colors = [BLUE, ORANGE, YELLOW, AQUA]
    labels = ["Ce système (même volatilité que SG Trend)", "Indice SG Trend", "QQQ", "60/40"]
    for i, (col, color, label) in enumerate(zip(crises.columns, colors, labels)):
        y = [k + (i - (n - 1) / 2) * (height + 0.02) for k in range(len(crises))]
        vals = crises[col] * 100
        ax.barh(y, vals, height=height, color=color, label=label)
        for yy, v in zip(y, vals):
            if pd.notna(v):
                ax.text(v + (1.5 if v >= 0 else -1.5), yy, f"{v:+.0f} %".replace("-", "−"), va="center",
                        ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(crises)))
    ax.set_yticklabels(crises.index)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlim(min(-100, crises.min().min() * 100 - 15), crises.max().max() * 100 + 20)
    ax.set_xlabel("Performance sur la période (%)")
    ax.set_title("Pendant les crises")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/crises.png", dpi=150)
    plt.close(fig)

    single = pd.read_csv(f"{RES}/courbes_actif_seul.csv", index_col=0, parse_dates=True)
    names = {"PDBC achat-conservation": "PDBC acheté et gardé",
             "PDBC tendance : investi ou monétaire": "Tendance : investi ou monétaire",
             "PDBC tendance : achat/vente à découvert": "Tendance : achat ou vente à découvert"}
    curves = {v: single[k].dropna() for k, v in names.items()}
    growth_and_drawdown(
        curves, dict(zip(curves, [AQUA, BLUE, ORANGE])), dict(zip(curves, ["Acheté-gardé", "Investi/monétaire", "Long/short"])),
        "Trend following sur PDBC seul, nov. 2015 → oct. 2026",
        "Rendement total, coûts inclus (5 points de base par transaction). Signaux recalculés chaque jour.",
        f"{OUT}/pdbc.png", log=False)


if __name__ == "__main__":
    main()
