"""Audit : la variante 5 « Les deux » est-elle applicable en vrai ? Rapport PDF.

Les calculs ont été faits par l'audit (scripts et résultats dans travail/, un dossier par volet ; les rapports
complets des agents et de leurs vérificateurs sont dans travail/rapports_agents/). Ce script ne refait pas les
calculs : il lit les résultats vérifiés et met en page le rapport.

Usage : python rapport_audit.py
Écrit audit/rapport_audit_variante5.pdf et audit/graphiques/*.png.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import re  # noqa: E402

from rapport_pdf import GRID, INK, INK_2, NEGATIVE, PALETTE, escape, fr, num, pct, styles  # noqa: E402
from rapport_pdf import P as _P  # noqa: E402


def P(text, style, **kw):
    """Paragraphe avec espaces insécables dans les nombres et avant les unités."""
    text = re.sub(r"(\d) (\d{3})(?!\d)", "\\1\u00a0\\2", text)
    for unit in (" k$", " M$", " $", " €"):
        text = text.replace(unit, "\u00a0" + unit[1:])
    return _P(text, style, **kw)

W = os.path.join(HERE, "travail")
OUT_FIG = os.path.join(HERE, "graphiques")
GREY, GREY_LIGHT = "#52514e", "#a3a19b"


def read(path, **kw):
    return pd.read_csv(os.path.join(W, path), **kw)


# ---------------------------------------------------------------- données vérifiées

def sharpe_steps():
    """Sharpe selon la période de départ : backtest, univers négociable, portefeuilles retail, vrais fonds."""
    g = read("cv_donnees/restatement_grid.csv")

    def pick(universe, execution, start):
        row = g[(g.universe == universe) & (g.execution == execution) & (g.start.astype(str) == start)]
        return float(row["sharpe16"].iloc[0])

    sim = read("simulation/summary_today_notional.csv")
    er4 = read("verif_capital/er4_spread_sensitivity.csv")
    sub = {c: sim[(sim.universe == "b_strict_subset_ge1c") & (sim.capital_usd == c) & (sim.freq == "D")].iloc[0]
           for c in (100000, 250000)}
    def er4_fixed(period):  # écart de prix de l'Euribor ICE corrigé (vérification du capital)
        return float(er4[(er4.universe == "subset_ge1c") & (er4.capital_usd == 250000) & (er4.case != "orig")
                         & (er4.period.astype(str).str.startswith(period))]["sharpe_net_real"].iloc[0])
    funds = read("rendement/sharpe_par_periode.csv", index_col=0)
    sg = funds["SG Trend SR (daily, net fees)"]
    return pd.DataFrame({
        "Backtest, 84 marchés": [pick("A_backtest_84", "same_close(1.0)", s) for s in ("1990", "2000", "2010")],
        "Marchés négociables, exécution réaliste": [pick("C_ex_CUA1_SCO1", "manual_0.5d(1.5)", s)
                                                    for s in ("1990", "2000", "2010")],
        "Portefeuille à 250 k$ (24 marchés)": [0.84, er4_fixed("2000"), er4_fixed("2010")],
        "Portefeuille à 100 k$ (14 marchés)": [0.73, sub[100000]["sharpe_2000"], sub[100000]["sharpe_2010"]],
        "Vrais fonds (SG Trend, frais déduits)": [np.nan, sg["since 2000"], sg["since 2010"]],
    }, index=["Depuis 1990", "Depuis 2000", "Depuis 2010"]).T


def forward_table():
    """Rendement après impôt sur 10 ans, scénario central (contre-vérification du rendement)."""
    t = read("cv_rendement/final_table.csv")
    t = t[t.horizon == 10]
    central = t[t.scen.astype(str).isin(["central", "eq central / trend central"])]

    def row(case, tier=None, vol=None):
        r = central[central.case == case]
        if tier is not None:
            r = r[r.tier == tier]
        if vol is not None:
            r = r[np.isclose(r.vol.astype(float), vol)]
        return r.iloc[0]

    def scen(case, tier, vol, s):
        r = t[(t.case == case) & (t.tier == tier) & np.isclose(t.vol.astype(float), vol) & (t.scen == s)]
        return float(r["at_med"].iloc[0])

    rows = [
        ("Monétaire en euros (fonds type XEON)", row("d EUR cash (XEON-type MMF, acc., PFU at exit)"), None),
        ("Variante 5 soi-même, 100 k$", row("a/b DIY v5 subset", "100k", 0.214), ("100k", 0.214)),
        ("Variante 5 soi-même, 250 k$", row("a/b DIY v5 subset", "250k", 0.214), ("250k", 0.214)),
        ("Variante 5 soi-même, 1 M$", row("a/b DIY v5 subset", "1M", 0.214), ("1M", 0.214)),
        ("Variante 5 à 12 % de risque, 250 k$", row("a/b DIY v5 subset", "250k", 0.12), ("250k", 0.12)),
        ("ETF trend UCITS (iMGP DBi, couvert €)", row("c UCITS trend ETF"), ("any", 0.12)),
        ("100 % S&P 500 (CTO)", row("e 100% equities CTO (sp)"), "e 100% equities CTO (sp)"),
        ("70 % S&P 500 + 30 % ETF trend (CTO)", row("e 70/30 CTO, annual rebalance (sp)"),
         "e 70/30 CTO, annual rebalance (sp)"),
        ("100 % Nasdaq-100, type QQQ (CTO)", row("e 100% equities CTO (nq)"), "e 100% equities CTO (nq)"),
        ("70 % Nasdaq-100 + 30 % ETF trend (CTO)", row("e 70/30 CTO, annual rebalance (nq)"),
         "e 70/30 CTO, annual rebalance (nq)"),
    ]
    out = []
    for label, r, key in rows:
        pess = opt = np.nan
        if isinstance(key, str):  # actions et mélanges : scénarios pessimiste et optimiste des deux jambes
            lo = "pess" if "100%" in key else "eq pess / trend pess"
            hi = "opt" if "100%" in key else "eq opt / trend opt"
            pess = float(t[(t.case == key) & (t.scen == lo)]["at_med"].iloc[0])
            opt = float(t[(t.case == key) & (t.scen == hi)]["at_med"].iloc[0])
        elif key is not None:
            case = "c UCITS trend ETF" if key[0] == "any" else "a/b DIY v5 subset"
            pess, opt = scen(case, key[0], key[1], "pess"), scen(case, key[0], key[1], "opt")
        out.append({"cas": label, "median": r.at_med, "p10": r.at_p10, "p90": r.at_p90, "reel": r.real_med,
                    "perte": r.P_nom_loss, "sous_cash": r.P_below_cash, "dd": r.dd_med, "dd10": r.dd_p10,
                    "pess": pess, "opt": opt})
    return pd.DataFrame(out).set_index("cas")


# ---------------------------------------------------------------- graphiques

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def chart_sharpe(table, path):
    colors_ = [PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[1], GREY]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.3), sharey=True)
    names = list(table.index)
    y = np.arange(len(names))
    for ax, period in zip(axes, table.columns):
        vals = table[period].values
        ax.set_axisbelow(True)
        ax.barh(y, np.nan_to_num(vals), color=colors_, height=0.7)
        for yy, v in zip(y, vals):
            ax.text((0 if np.isnan(v) else v) + 0.03, yy, "–" if np.isnan(v) else fr(v, 2), va="center",
                    fontsize=7.4, color=INK)
        ax.set_title(period, fontsize=8.6)
        ax.set_xlim(0, 1.45)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 1)))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7.6)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.985, "Du backtest à la réalité : Sharpe selon la période de départ", fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.01, 0.915, "Sharpe = rendement au-dessus du monétaire divisé par la volatilité. Fin des périodes : "
                          "10 juillet 2026.\nPortefeuilles : depuis 2000 et 2010 en contrats entiers avec coûts "
                          "réels ; depuis 1990 sans arrondi, coûts du backtest.", fontsize=7.2, color=INK_2, va="top",
             linespacing=1.4)
    fig.subplots_adjust(left=0.31, right=0.99, top=0.73, bottom=0.09, wspace=0.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_forward(table, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    names = list(table.index)
    y = np.arange(len(names))
    kind = {n: (GREY_LIGHT if "Monétaire" in n else PALETTE[1] if "soi-même" in n or "12 %" in n else
                PALETTE[0] if n.startswith("ETF") else GREY if n.startswith("100") else PALETTE[2]) for n in names}
    ax.set_axisbelow(True)
    for yy, n in zip(y, names):
        r = table.loc[n]
        if not np.isnan(r.p10) and r.p10 != r.p90:
            ax.plot([r.p10 * 100, r.p90 * 100], [yy, yy], color=kind[n], linewidth=3.2, alpha=0.45,
                    solid_capstyle="round")
        ax.plot(r["median"] * 100, yy, "o", color=kind[n], markersize=7)
        ax.text(r["median"] * 100, yy - 0.32, fr(r["median"] * 100), ha="center", va="bottom", fontsize=7.6,
                color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_ylim(len(names) - 0.4, -0.75)
    ax.set_xlabel("Rendement annuel après impôt sur 10 ans (%)", fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 1).replace("-", "−")))
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.985, "Ce qu'on peut raisonnablement attendre sur 10 ans, après impôt", fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.01, 0.935, "Point : médiane des simulations. Trait : 8 cas sur 10 tombent dans cette fourchette "
                          "(10e au 90e centile). Euros, scénario central.", fontsize=7.8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.37, right=0.98, top=0.86, bottom=0.11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- mise en page

def table(rows, widths, st, header_rows=1, negative=True, zebra=True, font=7.2, bold_first=False):
    data = []
    for r in rows:
        data.append([c if not isinstance(c, str) else P(escape(c) if "<" not in c else c,
                                                         st["cell"] if r is rows[0] else st["label"])
                     for c in r])
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#efeee9")),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor(INK_2)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for i in range(header_rows + 1, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f5f1")))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(colors.HexColor(INK_2))
    canvas.drawString(1.5 * cm, 1.0 * cm, "Audit de la variante « Les deux » — estimations, pas une garantie ; "
                                          "ni un conseil fiscal personnalisé")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


def p(x, d=1):
    return pct(x, d)


def build(path, steps, fwd):
    st = styles()
    story = []
    add = story.append
    body, small, h1, h2 = st["body"], st["small"], st["h1"], st["h2"]

    def bullets(items):
        for b in items:
            add(P(b.replace("\n", "<br/>"), st["bullet"], bulletText="•"))

    f = fwd.loc
    add(P("Audit : la variante « Les deux » est-elle applicable dans la réalité ?", st["title"]))
    add(P("Capital nécessaire, contrats entiers, coûts réels, fiscalité française, faisabilité et rendement à "
          "attendre, pour un particulier résident fiscal français. Octobre 2026. Montants en dollars quand il "
          "s'agit de contrats, rendements en euros.", st["subtitle"]))

    add(P("Verdict en bref", h1))
    bullets([
        "<b>La variante 5 telle que backtestée est un système de taille institutionnelle.</b> Pour la reproduire "
        "fidèlement sur ses 84 marchés, il faut environ <b>5 millions de dollars</b>, et au moins 1 à 2 M$ pour "
        "une copie correcte avec des contrats « micro ». En dessous, on ne peut tenir qu'une partie des marchés : "
        "14 à 100 k$, 24 à 250 k$, 40 à 1 M$.",
        "<b>Le Sharpe de 1,17 n'est pas atteignable</b> (Sharpe : rendement au-dessus du monétaire divisé par "
        "la volatilité ; 0,5 est déjà bon). Il doit beaucoup aux années 1990 et à deux marchés douteux : "
        "l'éthanol, un artefact de données, et le minerai de fer de Singapour, probablement inaccessible aux "
        "particuliers européens. Sur les marchés réellement négociables, avec une exécution réaliste, il tombe à "
        f"{fr(steps.iloc[1, 2], 2)} depuis 2010. Les portefeuilles qu'un particulier peut tenir font 0,4 environ "
        "depuis 2010, coûts réels compris. Les vrais fonds de trend ont fait "
        f"{fr(steps.iloc[4, 2], 2)}, frais déduits.",
        "<b>Coûts réels :</b> 2,9 à 3,7 % du capital par an en frais de transaction et de roulement pour un petit "
        "portefeuille (déjà comptés dans les Sharpe ci-dessus), 150 à 1 000 $ par an de frais fixes, environ "
        "0,6 à 0,8 % par an d'intérêts perdus sur la trésorerie, et 80 à 500 heures de mise en place. "
        "<b>Fiscalité :</b> 31,4 % chaque année sur les gains, car chaque roulement de contrat réalise le gain.",
        f"<b>Rendement à attendre sur 10 ans, après impôt</b> (médiane des simulations) : {p(f['Variante 5 soi-même, 100 k$', 'median'])} "
        f"par an à 100 k$, {p(f['Variante 5 soi-même, 250 k$', 'median'])} à 250 k$, "
        f"{p(f['Variante 5 soi-même, 1 M$', 'median'])} à 1 M$. Le risque est lourd : une chance sur trois "
        "environ de perdre de l'argent sur 10 ans, et une pire baisse médiane proche de 50 %. Le monétaire "
        f"rapporte {p(f['Monétaire en euros (fonds type XEON)', 'median'])} après impôt sans risque.",
        f"<b>Un ETF de trend UCITS fait aussi bien ou mieux sans travail</b> : {p(f['ETF trend UCITS (iMGP DBi, couvert €)', 'median'])} "
        "par an attendus, avec une pire baisse médiane deux fois plus faible. L'intérêt réel du trend pour "
        "vous est de diversifier QQQ. Un portefeuille à 70 % Nasdaq-100 et 30 % d'ETF trend garde le même "
        f"rendement médian que 100 % Nasdaq ({p(f['70 % Nasdaq-100 + 30 % ETF trend (CTO)', 'median'])} contre "
        f"{p(f['100 % Nasdaq-100, type QQQ (CTO)', 'median'])}). En revanche, la probabilité de perdre sur 10 "
        f"ans passe de {p(f['100 % Nasdaq-100, type QQQ (CTO)', 'perte'], 0)} à "
        f"{p(f['70 % Nasdaq-100 + 30 % ETF trend (CTO)', 'perte'], 0)}, et la pire baisse médiane de "
        f"{p(f['100 % Nasdaq-100, type QQQ (CTO)', 'dd'], 0)} à {p(f['70 % Nasdaq-100 + 30 % ETF trend (CTO)', 'dd'], 0)}.",
        "<b>Recommandation :</b> ne pas faire tourner la variante 5 soi-même sous 1 M$. Passer par un ETF de "
        "trend UCITS pour la diversification, et garder le système comme outil d'apprentissage (voir la dernière "
        "section pour le faire quand même, proprement).",
    ])

    add(P("Selon votre capital", h2))
    ref = fwd
    rows = [["Capital", "Marchés tenables (≥ 1 contrat)", "Sharpe depuis 2010, coûts réels",
             "Frais de transaction et de roulement / an", "Après impôt, médiane 10 ans [8 cas sur 10]",
             "Probabilité de perte sur 10 ans", "Verdict"]]
    for lab, n, sh, cost, key, verdict in [
        ("25–50 k$", "6 à 11", "0,28–0,34", "≈ 3,3 %", None, "Non : 6 à 11 marchés, baisses de plus de 55 %"),
        ("100 k$", "14", "0,41", "3,3 %", "Variante 5 soi-même, 100 k$", "Non : à peine mieux que le monétaire"),
        ("250 k$", "24", "0,42", "3,7 %", "Variante 5 soi-même, 250 k$", "Moins bien que l'ETF"),
        ("1 M$", "40", "0,39", "2,9 %", "Variante 5 soi-même, 1 M$", "Comme l'ETF, avec beaucoup plus de travail"),
        ("≥ 5 M$", "63 à 84", "≈ 0,38–0,39", "2,3–2,5 %", None, "Réplique fidèle possible"),
        ("ETF trend UCITS", "tout montant", "≈ 0,3 (fonds réels)", "TER 0,75 %", "ETF trend UCITS (iMGP DBi, couvert €)",
         "Le plus simple"),
    ]:
        if key:
            r = ref.loc[key]
            res = f"{p(r['median'])} [{p(r.p10)} ; {p(r.p90)}]"
            loss = p(r.perte, 0)
        else:
            res, loss = "non simulé", "–"
        rows.append([lab, n, sh, cost, res, loss, verdict])
    add(table(rows, [1.8 * cm, 2.0 * cm, 2.0 * cm, 2.3 * cm, 3.5 * cm, 2.0 * cm, 4.4 * cm], st))
    add(P("Sharpe des portefeuilles réellement tenables, simulés en contrats entiers avec les coûts par contrat "
          "d'IBKR et l'écart achat-vente. Rendements : simulation sur 10 ans en euros, après impôt (PFU 31,4 %), "
          "frais fixes et intérêts perdus déduits, au risque de la variante 5 (21 % de volatilité) ; l'ETF est "
          "simulé à 12 % de volatilité.",
          small))

    # ------------------------------------------------ 1. Le point de départ
    add(PageBreak())
    add(P("1. Le 1,17 du backtest n'est pas atteignable", h1))
    add(Image(os.path.join(OUT_FIG, "audit_sharpe.png"), width=18 * cm, height=18 * cm * 3.3 / 7.6))
    bullets([
        "<b>Les années 1990 font l'essentiel du résultat.</b> Sharpe 2,03 de 1990 à 1999, 1,34 dans les années "
        "2000, 0,47 de 2010 à 2019 et −0,46 depuis 2023. Le backtest était encore 43 % sous son sommet "
        "(hors monétaire) au 10 juillet 2026, après un point bas à −54 % en mai 2025.",
        "<b>L'éthanol (CUA1) est un artefact de données.</b> Un jour sur huit sans aucune variation, une "
        "autocorrélation anormale, et un contrat qui aurait monté de 18,7 % par an à l'achat. Il apportait 11 % "
        "des gains depuis 2010 avec 2,4 % du risque. <b>Le minerai de fer de Singapour (SCO1)</b> a des données "
        "propres, mais il pèse très lourd (9 % des gains), ses données ne commencent qu'en 2016, et il est "
        "probablement inaccessible à un particulier européen (aucun document d'information, ou DIC, trouvé).",
        "<b>L'exécution réelle coûte un peu.</b> Le backtest passe ses ordres au cours de clôture qui sert à calculer le "
        "signal, ce qui est impossible. En exécutant dans la séance suivante, le Sharpe perd 0,02 à 0,04 ; il "
        "perdrait 0,07 avec un jour entier de retard.",
        "<b>Choisi après coup.</b> « Les deux » est la meilleure de 53 variantes de trend testées sur ces "
        "données : sa sélection gonfle à elle seule son Sharpe d'environ 0,25. Son avance sur la variante 2 "
        "n'est que tout juste significative sur 1990–2026 (statistique t de 3,3 pour un seuil de 3,1), puis elle "
        "fond : 2,1 depuis 2000, 1,2 depuis 2010 et −0,2 depuis 2020.",
        "<b>Les vrais fonds, à frais réintégrés,</b> retrouvent un Sharpe de 0,41 à 0,49 depuis 2010. Depuis "
        "2010, le backtest ressemble donc à ce qu'obtiennent les professionnels avant leurs frais. L'écart "
        "venait surtout d'avant 2010.",
    ])
    add(P("Sharpe par période : backtest et vrais fonds", h2))
    sp = read("rendement/sharpe_par_periode.csv", index_col=0)
    cols = [("v5 SR (daily)", "Variante 5 (backtest)"), ("v2 SR (daily)", "Variante 2 (backtest)"),
            ("SG Trend SR (daily, net fees)", "Indice SG Trend (net de frais)"),
            ("BTOP50 SR (monthly, net fees)", "BTOP50 (net de frais)"),
            ("AQR TSMOM SR (monthly, gross, no costs)", "Facteur AQR TSMOM (brut, sans coûts)")]
    periods = [("1990-1999", "1990–99"), ("2000-2009", "2000–09"), ("2010-2019", "2010–19"),
               ("2020-2026", "2020–26"), ("since 2023", "Depuis 2023")]
    rows = [["Série"] + [lab for _, lab in periods]]
    for col, lab in cols:
        rows.append([lab] + [("–" if pd.isna(sp.loc[k, col]) else num(sp.loc[k, col])) for k, _ in periods])
    add(table(rows, [5.5 * cm] + [2.5 * cm] * 5, st))
    add(P("Rendement au-dessus du monétaire divisé par la volatilité. L'indice SG Trend n'existe que depuis "
          "2000. Le facteur AQR est une construction académique, sans frais ni coûts.", small))

    # ------------------------------------------------ 2. Capital et contrats
    add(PageBreak())
    add(P("2. Le capital : la taille des contrats décide de tout", h1))
    add(P("Les positions de la variante 5 sont petites par marché : en médiane, 3 % du capital pour une action, "
          "6 % pour une obligation, 12 % pour une devise, 2,5 % pour un produit agricole. Or un contrat a une "
          "taille fixe : on ne peut pas en acheter 0,3.", body))
    rows = [["Contrat", "Valeur d'un contrat (juillet 2026)", "Version micro", "Valeur du micro",
             "Coût par aller (frais + demi-écart)"],
            ["S&P 500 (ES)", "381 000 $", "MES", "38 100 $", "8,75 $ / 1,27 $"],
            ["Pétrole WTI (CL)", "71 400 $", "MCL", "7 100 $", "14,4 $ / 2,5 $"],
            ["Or (GC)", "411 000 $", "MGC (10 onces) ; 1OZ (1 once)", "41 100 $ ; 4 100 $", "12,1 $ / 1,65 $ (MGC)"],
            ["Euro (6E)", "143 000 $", "M6E", "14 300 $", "6,1 $ / 1,15 $"],
            ["Bund allemand", "144 000 $", "mini Euronext, liquidité incertaine", "36 000 $", "8,2 $"],
            ["Taux courts SOFR", "240 000 $", "aucune", "–", "9,6 $"],
            ["Obligation japonaise (JGB)", "791 000 $", "mini, illiquide", "79 000 $", "34 $ / 15 $"],
            ["Maïs (ZC)", "22 000 $", "mini XC ; micro MZC", "4 400 $ ; 2 200 $", "11 $ / 3,3 $ (XC)"]]
    add(table(rows, [3.6 * cm, 3.8 * cm, 2.9 * cm, 2.8 * cm, 4.9 * cm], st))
    add(Spacer(1, 4))
    add(P("Capital pour tenir au moins 1 contrat, puis 4 (règle de Carver), à la position médiane de la variante "
          "5, avec le plus petit contrat disponible : <b>0,63 M$ et 2,5 M$ pour un marché médian</b>, jusqu'à "
          "7,7 M$ et 31 M$ pour le café. Un seul petit contrat dans chacun des 84 marchés représente déjà 4,9 M$ "
          "de valeur.", body))
    rows = [["Capital", "Marchés à ≥ 1 contrat (84)", "Marchés à ≥ 4 contrats", "Accessibles et liquides, ≥ 1",
             "Portefeuille réaliste (re-simulé)"],
            ["25 k$", "0", "0", "0", "6 marchés"],
            ["100 k$", "7", "0", "5", "14 : maïs, blé, soja, or, gaz naturel, yen, livre, dollar australien, euro, "
                                       "Nikkei, Euro Stoxx, CAC, Schatz, SOFR"],
            ["250 k$", "20", "6", "12", "24 : + pétrole, cuivre, colza canadien, sucre, peso mexicain, Dow Jones, "
                                        "Russell, TOPIX, Euribor, SONIA"],
            ["500 k$", "33", "12", "19", "32 : + S&P 500, DAX, AEX, taux US 2 et 5 ans, dollar canadien, coton, porc"],
            ["1 M$", "56", "20", "32", "40 : + Nasdaq, Bobl, taux US 10 ans et 10 ans « ultra », dollar néo-zélandais, "
                                      "blé du Kansas, bovins, sucre blanc"],
            ["2 M$", "71", "33", "43", "50 : + argent, Brent, platine, franc suisse, BTP, Bund, OAT, FTSE, soja "
                                       "(tourteau et huile)"],
            ["5 M$", "81", "58", "61", "63 : réplique quasi fidèle"]]
    add(table(rows, [1.6 * cm, 2.9 * cm, 2.7 * cm, 3.0 * cm, 7.8 * cm], st))
    add(Spacer(1, 4))
    bullets([
        "<b>En contrats entiers, sur les 84 marchés :</b> à 100 k$, 80 % des positions voulues s'arrondissent "
        "à zéro. Le compte ne suit alors le système qu'avec un écart de 12 points par an (corrélation 0,86). À "
        "1 M$, l'écart tombe à 4 points (0,98), à 5 M$ à 1 point.",
        "<b>La bonne méthode pour un petit compte</b> est de choisir moins de marchés, chacun tenable, et de "
        "recalculer les poids. L'arrondi ne coûte alors presque rien ; ce qu'on perd, c'est la diversification "
        "(Sharpe depuis 1990, sans arrondi : 0,73 à 100 k$ et 0,84 à 250 k$, contre 1,17).",
        "<b>Accès :</b> IBKR bloque pour un particulier européen tout contrat sans DIC (règlement PRIIPs). Sont "
        "accessibles le CME, l'ICE, Eurex, Euronext, Osaka et Hong Kong ; Singapour et l'Australie sont "
        "probablement bloqués (aucun DIC trouvé). Certains micros sont minces : le micro Euro Stoxx (2 400 contrats par jour) et le micro "
        "livre sterling (580 par jour). Pour le micro yen et les micros grains, aucun chiffre de volume récent "
        "n'a été trouvé, à vérifier dans TWS. Le statut de « client professionnel » lèverait le blocage, mais "
        "exige deux critères sur trois : plus de 500 000 € de portefeuille, une activité de trading soutenue "
        "pendant un an, ou une expérience professionnelle dans la finance.",
        "<b>Petit compte et baisse :</b> un compte de 100 k$ démarré en 2016 serait tombé à 59 k$, sous son "
        "propre seuil minimal. Il n'aurait alors plus pu tenir ses marchés.",
    ])

    # ------------------------------------------------ 3. Coûts
    add(PageBreak())
    add(P("3. Les coûts réels", h1))
    add(P("Transactions (courtier Interactive Brokers, Irlande)", h2))
    bullets([
        "Commissions : 0,85 $ par contrat américain standard et 0,25 $ par micro, plus les frais de bourse "
        "(environ 1,4 $ sur un E-mini, 0,35 $ sur un micro). Sur Eurex, 0,90 € plus les frais de bourse.",
        "Le vrai coût est surtout l'écart achat-vente, environ 2/3 du total. Les micros coûtent 1,4 à 2 fois "
        "plus cher par dollar de valeur que les contrats standard.",
        "Au total, la variante 5 complète coûterait 2,3 à 2,6 % du capital par an avec des positions "
        "fractionnaires (le backtest supposait 2,4 %). En contrats entiers, c'est 2,7 à 3,1 % avec des "
        "contrats standard et 3,2 à 3,6 % avec des micros. Un petit portefeuille, qui traite et roule plus par "
        "dollar, coûte 2,9 à 3,7 % par an.",
        "Les Sharpe des portefeuilles tenables de ce rapport intègrent ces coûts réels. Ceux des 84 marchés "
        "(1,17 et 0,38) gardent le modèle de coûts du backtest, donc un peu optimiste.",
        "Volume : environ 750 ordres par an à 100 k$, 1 500 à 250 k$, 2 900 à 1 M$ (roulements compris).",
    ])
    add(P("Frais fixes", h2))
    rows = [["Poste", "Coût", "Remarque"],
            ["Logiciel pysystemtrade (Carver)", "gratuit", "Python ; IB Gateway et une machine allumée en permanence"],
            ["Serveur (VPS Hetzner)", "≈ 7 €/mois TTC", "ou un PC à la maison"],
            ["Données historiques (Norgate)", "270 $/an", "ou les fichiers gratuits de Carver, arrêtés en 2024"],
            ["Données de marché IBKR", "0 à 6 200 $/an", "données différées gratuites, suffisantes pour un système "
                                                         "quotidien ; temps réel ICE ≈ 140 à 160 $/mois"],
            ["Alternative clé en main : QuantConnect", "84 $/mois", "plus les nœuds de trading en direct"],
            ["Temps", "80 à 500 h, puis 1 à 2 h/semaine", "plus une connexion manuelle chaque semaine "
                                                         "(double authentification IBKR obligatoire)"]]
    add(table(rows, [4.6 * cm, 3.6 * cm, 9.8 * cm], st))
    add(P("Budget annuel : environ 150 $ au minimum, 1 000 $ en version « sobre », 7 400 $ en temps réel complet. "
          "Hors temps réel, cela représente 0,15 à 1 % du capital à 100 k$ et 0,01 à 0,1 % à 1 M$.", small))
    add(P("Trésorerie : l'intérêt perdu", h2))
    bullets([
        "Le backtest suppose que tout le capital rapporte le taux des T-bills. Chez IBKR, l'argent bloqué en "
        "marge des futures ne rapporte rien : en moyenne 17 à 23 % du capital, bien davantage lors des pics.",
        "Le reste rapporte le taux de référence moins 0,5 point, avec deux restrictions. Les premiers 10 000 $ "
        "(ou €) ne rapportent rien, et le taux est réduit au prorata sous 100 000 $ de compte. Les T-bills ou un fonds "
        "monétaire ne peuvent pas servir de marge.",
        "Pour un investisseur en euros, la référence est l'€STR (environ 2,4 %), pas le T-bill américain "
        "(4,0 %). Les 29,6 % par an affichés par le backtest, monétaire américain compris, sont donc à relire "
        "avec le monétaire euro.",
        "Manque à gagner par rapport à l'€STR, avec la part libre placée dans un fonds monétaire : environ "
        "0,8 % par an à 100 k$, 0,7 % à 250 k$ et 0,6 % à 1 M$.",
    ])

    # ------------------------------------------------ 4. Fiscalité
    add(P("4. La fiscalité française (2026)", h1))
    bullets([
        "<b>Taux :</b> prélèvement forfaitaire unique de <b>31,4 %</b> (12,8 % d'impôt et 18,6 % de prélèvements "
        "sociaux, depuis la hausse de CSG de la loi de financement de la Sécurité sociale 2026). L'option pour "
        "le barème progressif reste possible.",
        "<b>Régime des futures (article 150 ter du CGI) :</b> chaque clôture de contrat est imposable. Comme "
        "chaque roulement clôture l'ancien contrat, presque tout le gain est taxé chaque année, sans report "
        "possible. Les pertes s'imputent sur les gains de même nature, plus-values de titres comprises (donc vos "
        "gains sur QQQ), l'année même et les 10 suivantes. Pas sur le bitcoin, qui a son propre régime.",
        "<b>Déclaration :</b> gain net case 3VG, perte case 3VH de la déclaration 2042, et formulaire 2074-CMV "
        "pour reporter des pertes. IBKR Irlande ne fournit pas d'IFU français : il faut calculer soi-même le "
        "résultat en euros et garder les justificatifs. Les intérêts versés par IBKR sont taxés à part et "
        "ne peuvent pas absorber les pertes sur futures. Ils donnent lieu à un acompte de 12,8 % à verser "
        "soi-même (formulaire 2778-SD).",
        "<b>Compte à l'étranger :</b> le compte IBKR doit être déclaré chaque année (formulaire 3916). En cas "
        "d'oubli : 1 500 € d'amende par compte et par an (10 000 € pour un État non coopératif). S'y ajoute une "
        "majoration de 80 % de l'impôt éludé, et l'administration peut remonter 10 ans en arrière. <b>À vérifier pour votre bitcoin :</b> un compte "
        "sur une plateforme étrangère se déclare sur le 3916-bis (750 €, ou 1 500 € au-delà de 50 000 €).",
        "<b>Risque de requalification :</b> des milliers d'opérations automatisées par an peuvent être jugées "
        "« habituelles » (article 92 du CGI). Les gains relèveraient alors des bénéfices non commerciaux (BNC). "
        "Ils seraient imposés au barème progressif plus les prélèvements sociaux. Les pertes ne seraient "
        "reportables que 6 ans, et seulement sur des bénéfices de même nature, jamais sur QQQ. Le rendement "
        "médian après impôt baisserait de 0,7 à 1,6 point selon votre tranche (30 % ou 41 %).",
        "<b>Enveloppes :</b> les futures ne vont ni dans un PEA ni dans une assurance-vie. Un ETF capitalisant "
        "sur compte-titres n'est taxé qu'à la revente, ce qui vaut environ 0,3 point par an d'avantage sur 10 ans.",
    ])
    add(P("Ceci n'est pas un conseil fiscal personnalisé : les points sensibles (requalification, conversion en "
          "euros, déclaration des comptes crypto) méritent l'avis d'un professionnel.", small))

    # ------------------------------------------------ 5. Faire tourner
    add(Spacer(1, 6))
    add(P("5. Faire tourner le système au quotidien", h1))
    bullets([
        "<b>Marge :</b> en moyenne 17 à 23 % du capital, 26 à 40 % les mauvais jours (1 jour sur 20, selon le "
        "modèle de marge), 33 à 62 % au pire historique. En stress "
        "extrême (bourses qui doublent leurs marges en pleine baisse), 70 à 95 % du capital. IBKR n'envoie pas "
        "d'appel de marge : il liquide automatiquement, au prix du marché, les positions qu'il choisit. Il "
        "faut garder tout le capital chez le courtier, ce qui accroît le manque à gagner sur la trésorerie, et "
        "s'imposer une marge maximale de 40 à 50 %.",
        "<b>Exposition :</b> la variante 5 détient en moyenne 8,8 fois son capital en contrats, jusqu'à 28 fois, "
        "surtout en taux courts et obligations. Un choc brutal (gilts britanniques en 2022, franc "
        "suisse en 2015) n'est représenté que dans la mesure où l'historique en contient.",
        "<b>Charge de travail :</b> 2 à 3 ordres par jour à 100 k$, 4 à 6 à 250 k$, une douzaine à 1 M$, sur les "
        "séances asiatiques, européennes et américaines. Il faut aussi 70 à 300 roulements de contrats par an. "
        "Au-delà de 100 à 250 k$, l'automatisation est indispensable : téléchargement et nettoyage des données, "
        "calcul des signaux, ordres, roulements, rapprochement des positions, rapport de risque, conversion "
        "des devises.",
        "<b>Pannes typiques :</b> prix aberrant dans les données, roulement oublié ou date de premier avis "
        "dépassée, panne du courtier, jours fériés différents selon les bourses, soldes négatifs en devise "
        "facturés au taux de référence plus 1,5 %.",
        "<b>Hebdomadaire plutôt que quotidien :</b> deux fois moins d'ordres, mais les roulements restent. Le "
        "Sharpe baisse d'environ 0,05 à 0,07.",
    ])
    add(P("Tenir dans les mauvaises périodes", h2))
    bullets([
        "Depuis 1990, la variante 5 a connu 7 baisses de plus de 20 %. Deux ont duré environ 4 ans : 2016–2020 "
        "(−47 %, 50 mois) et celle commencée en septembre 2022 (−54 % au plus bas, pas effacée au 10 juillet "
        "2026).",
        "9 années négatives sur 37 ; un an sur cinq, le rendement sur 12 mois glissants est négatif. Sur 10 ans, "
        "jamais. En 2023–2024, la variante 5 perdait 15 % et 19 % par an hors monétaire, pendant que QQQ "
        "gagnait.",
        "Selon Morningstar, les investisseurs perdent en moyenne 1,2 point par an en entrant et sortant au "
        "mauvais moment, davantage sur les fonds volatils. Avec ces baisses, le risque d'abandonner au plus bas "
        "est le principal danger pratique.",
        "À noter : le backtest s'arrête au 10 juillet 2026. Depuis, les fonds de trend ont nettement remonté : "
        "l'indice SG CTA a gagné environ 6,7 % entre le 10 juillet et la mi-septembre (+15,9 % depuis janvier). "
        "Ce rebond n'est pas dans ces chiffres.",
    ])

    # ------------------------------------------------ 6. Rendement
    add(PageBreak())
    add(P("6. Quel rendement attendre ?", h1))
    add(Image(os.path.join(OUT_FIG, "audit_rendement.png"), width=18 * cm, height=18 * cm * 4.5 / 7.6))
    rows = [["Sur 10 ans, après impôt", "Médiane", "8 cas sur 10", "Pessimiste / optimiste",
             "Probabilité de perte", "Probabilité de faire moins que le monétaire", "Pire baisse médiane"]]
    for name in fwd.index:
        r = fwd.loc[name]
        spread = "–" if r.p10 == r.p90 else f"{p(r.p10)} à {p(r.p90)}"
        po = "–" if np.isnan(r.pess) else f"{p(r.pess)} / {p(r.opt)}"
        rows.append([name, p(r["median"]), spread, po, "–" if r.perte == 0 else p(r.perte, 0),
                     "–" if np.isnan(r.sous_cash) else p(r.sous_cash, 0), "–" if r.dd == 0 else p(r.dd, 0)])
    add(table(rows, [5.0 * cm, 1.5 * cm, 2.5 * cm, 2.6 * cm, 1.8 * cm, 2.2 * cm, 2.4 * cm], st))
    add(Spacer(1, 4))
    add(P("Comment ces chiffres sont construits", h2))
    bullets([
        "<b>Sharpe futur supposé</b>, après coûts et exécution réalistes, avant frais fixes, trésorerie et "
        "impôt : 0,20 à 100 k$, 0,24 à 250 k$, 0,27 à 1 M$ (scénario central). Pessimiste : 0 à 0,05 ; "
        "optimiste : 0,40 à 0,47. Ancrages : 0,4 environ depuis 2010 pour les portefeuilles tenables, 0,21 à "
        "0,27 depuis 2010–2015 pour les vrais fonds, et une décote de 25 à 35 % pour le choix après coup et "
        "l'usure après publication. L'ETF est supposé à 0,25 (de 0,10 à 0,40).",
        "<b>Simulation :</b> 5 000 trajectoires de 10 ans tirées par blocs de six mois dans l'historique de "
        "chaque portefeuille. Monétaire à 2,4 % (€STR), PFU de 31,4 % chaque année avec report des "
        "pertes sur 10 ans pour les futures, et à la revente pour les ETF. Actions américaines en euros : "
        "6 % par an avant impôt en central, d'après les hypothèses long terme de J.P. Morgan et Vanguard.",
        "<b>Ce qui pèse le plus :</b> le Sharpe futur. ±0,1 de Sharpe change le rendement de ±1,7 point par an "
        "pour le système soi-même, ±1,0 pour l'ETF, ±0,3 pour le mélange 70/30. ±0,5 point d'€STR change ±0,3. "
        "Valoriser votre temps (80 heures par an à 25 €/h) retire 2,5 points à 100 k$ et 0,9 à 250 k$.",
        "<b>Viser 12 % de risque au lieu de 21 %</b> réduit fortement les baisses (pire baisse médiane −28 % "
        "au lieu de −48 %), mais aussi le rendement médian : 2,2 % au lieu de 2,8 % à 250 k$, 1,0 % au lieu de "
        "1,8 % à 100 k$. Les positions rétrécissent : 250 k$ se comporte alors comme 100 k$.",
        "Rendements réels (après 2 % d'inflation) : retirer environ 2 points. Le système soi-même à 100 k$ est "
        "alors à −0,2 % par an.",
    ])

    # ------------------------------------------------ 7. Alternatives
    add(PageBreak())
    add(P("7. Les alternatives toutes faites", h1))
    rows = [["Fonds", "ISIN", "Type", "Frais", "Remarques"],
            ["iMGP DBi Managed Futures R EUR HP (MFEH, Euronext Paris)", "LU3359622902",
             "ETF UCITS capitalisant, couvert en euros", "0,75 %",
             "Coté à Paris depuis le 21/09/2026. Réplique l'indice SG CTA (toutes stratégies de CTA, pas du "
             "trend pur). Fonds de 698 M$, mais classe couverte récente (mai 2026) et petite."],
            ["iMGP DBi Managed Futures R USD (Euronext Paris : DBMF)", "LU2951555585", "ETF UCITS, en dollars", "0,75 %",
             "Même fonds, risque de change dollar. Le jumeau américain DBMF : 9,2 % par an de 2019 à 2026, "
             "volatilité 12 %, pire baisse −20 %."],
            ["BNP Paribas Easy Managed Futures", "LU3307218399", "ETF UCITS synthétique, couvert en euros",
             "0,60 %", "Modèle de trend maison via un swap. Lancé en mai 2026, environ 10 M€. Coté sur Xetra "
             "(EEAU) ; cotation à Paris non confirmée."],
            ["AQR Managed Futures UCITS (RAE)", "LU1662502183", "Fonds (pas un ETF)", "TER 1,39 %",
             "Le plus proche d'un trend « pur » à la Carver. Classe A en dollars : 0,60 % plus 10 % des gains."],
            ["Winton Trend UCITS (I EUR)", "IE00BG382R37", "Fonds", "1,05 %", "Classe institutionnelle."],
            ["Man AHL Trend Alternative (DNY H EUR)", "LU0424370004", "Fonds", "≈ 2,75 %",
             "2 % de frais plus une commission de performance : cher."]]
    add(table(rows, [4.4 * cm, 2.5 * cm, 2.9 * cm, 1.5 * cm, 6.7 * cm], st))
    add(P("Aucun n'est éligible au PEA ; leur présence dans un contrat d'assurance-vie n'a pu être confirmée "
          "(à vérifier contrat par contrat). Les ETF américains (DBMF coté à New York, KMLM) restent inaccessibles aux "
          "particuliers européens faute de DIC. Vérifiez les cotations et frais sur justETF ou la fiche du "
          "fonds avant d'acheter.", small))

    # ------------------------------------------------ 8. Si vous voulez quand même
    add(P("8. Si vous voulez quand même le faire vous-même", h1))
    bullets([
        "<b>Capital :</b> au moins 250 k$ pour un portefeuille diversifié de 20 à 25 marchés ; l'idéal "
        "commence à 1 M$. En dessous, considérez-le comme un projet d'apprentissage, avec une petite somme "
        "que vous acceptez de voir baisser de moitié.",
        "<b>Commencer sans argent :</b> 6 à 12 mois en compte de démonstration IBKR avec pysystemtrade, pour "
        "roder les données, les ordres et les roulements.",
        "<b>Simplifier :</b> marchés liquides des bourses accessibles (CME, ICE, Eurex, Euronext, Osaka), "
        "micros quand ils sont liquides. Viser 12 à 15 % de risque plutôt que 21 % : baisses bien plus faibles, "
        "mais positions plus petites, donc plus de capital par marché. La stratégie 13 et le pilotage du risque "
        "ont amélioré le backtest, mais leur avance a disparu depuis 2010 : ne pas compter dessus. Pour un "
        "petit compte, la stratégie 25 de Carver (optimisation dynamique) peut faire mieux que l'arrondi "
        "simple ; elle n'a pas été testée ici.",
        "<b>Garde-fous :</b> marge maximale de 40 à 50 % du capital, contrôle automatique des prix aberrants, "
        "rapprochement quotidien des positions, conversion régulière des devises, journal des opérations en "
        "euros pour l'impôt.",
        "<b>Et pour la diversification de QQQ,</b> le plus simple et le plus robuste reste un ETF de trend : "
        "10 à 30 % du portefeuille, en compte-titres.",
    ])

    # ------------------------------------------------ Méthode
    add(PageBreak())
    add(P("Méthode, vérifications et limites", h1))
    bullets([
        "<b>Organisation :</b> quatre volets indépendants (capital et contrats, coûts et fiscalité, rendement "
        "futur, faisabilité) et une simulation en contrats entiers, chacun revu par un vérificateur chargé de "
        "le contredire. Une critique a cherché ce qui manquait, et trois contre-vérifications ont contrôlé ses "
        "résultats nouveaux. Scripts, résultats et rapports complets : analyse-trend/audit/travail/.",
        "<b>Corrections faites en route :</b>\n"
        "– les portefeuilles pour particulier du premier volet ne respectaient pas leur propre règle (refaits) ;\n"
        "– l'écart de prix de l'Euribor était surestimé ;\n"
        "– la marge chez IBKR ne rapporte rien (manque à gagner doublé) ;\n"
        "– l'amende de 5 % pour compte non déclaré n'existe plus ;\n"
        "– les ISIN des ETF iMGP DBi avaient été confondus ;\n"
        "– un autre ETF de trend existe (BNP Paribas Easy), un ETF Man est en préparation ;\n"
        "– il existe des micros platine et palladium ;\n"
        "– le taux de la BCE est de 2,50 % depuis le 16 septembre 2026.",
        "<b>Erreurs de données trouvées dans le projet :</b> le gasoil (QS1) était associé au mauvais code chez "
        "Carver (essence) dans les travaux sur le carry ; l'éthanol (CUA1) est inutilisable. Les autres "
        "résultats du projet ne sont pas recalculés ici.",
        "<b>Principales limites :</b>\n"
        "– le Sharpe futur est un jugement, ce qui donne des fourchettes larges ;\n"
        "– les micros sont supposés avoir toujours existé ;\n"
        "– environ 25 coûts de micros sont estimés ;\n"
        "– les marges sont modélisées, pas relevées chez IBKR ;\n"
        "– le change euro-dollar sur les gains est ignoré ;\n"
        "– le scénario de requalification fiscale n'est chiffré qu'à part ;\n"
        "– les données s'arrêtent au 10 juillet 2026.",
        "<b>Sources principales :</b>\n"
        "– configuration et coûts IBKR publiés par Carver (pysystemtrade) ;\n"
        "– barèmes et pages IBKR, CME et ICE, vus par recherche web ;\n"
        "– BOFiP et presse fiscale (PFU 31,4 %, article 150 ter, formulaire 3916) ;\n"
        "– fiches justETF, finanzfluss et extraETF pour les fonds ;\n"
        "– indices SG Trend, SG CTA, BTOP50 et AQR TSMOM pour les vrais fonds ;\n"
        "– hypothèses long terme 2026 de J.P. Morgan et Vanguard pour les actions.\n"
        "Détail et liens : travail/cv_fonds_fisc/facts_verified.csv et travail/rapports_agents/.",
    ])

    add(P("Lexique", h2))
    for term, expl in [
        ("Backtest", "simulation d'une stratégie sur l'historique des prix."),
        ("Sharpe", "rendement au-dessus du monétaire divisé par la volatilité ; 0,5 est bon, 1 excellent."),
        ("Contrat à terme (future), micro", "engagement d'acheter ou vendre plus tard à un prix fixé aujourd'hui ; "
                                            "un « micro » vaut 1/10e (parfois moins) du contrat standard."),
        ("Roulement", "remplacer un contrat qui arrive à échéance par le suivant."),
        ("Écart achat-vente, demi-écart", "différence entre le meilleur prix d'achat et de vente ; on en paie "
                                          "la moitié à chaque ordre."),
        ("Marge", "dépôt de garantie exigé pour détenir des contrats."),
        ("€STR, T-bill, SOFR, Euribor, SONIA", "taux du monétaire en euros, bons du Trésor américain à 3 mois, "
                                                "et taux courts dollar, euro et sterling."),
        ("Schatz, Bobl, Bund, BTP, OAT, gilts", "emprunts d'État allemands (2, 5 et 10 ans), italiens, français "
                                                 "et britanniques."),
        ("UCITS (OPCVM), TER", "fonds européen réglementé ; frais annuels totaux."),
        ("CTO, PEA, PFU", "compte-titres ordinaire ; plan d'épargne en actions ; prélèvement forfaitaire unique."),
        ("DIC (PRIIPs)", "document d'information clé exigé pour vendre un produit à un particulier européen."),
        ("IFU", "relevé fiscal annuel envoyé par un courtier français."),
        ("BNC", "bénéfices non commerciaux, catégorie d'impôt des activités professionnelles."),
        ("IB Gateway, TWS, VPS", "logiciels de connexion à Interactive Brokers ; serveur loué en ligne."),
        ("Date de premier avis", "date à partir de laquelle un acheteur peut être tenu de prendre livraison."),
        ("Centile, médiane", "le 10e centile est dépassé dans 90 % des cas ; la médiane dans 50 %."),
        ("Statistique t", "mesure si un écart est dû au hasard ; au-delà de 2 à 3, il l'est rarement."),
        ("SG CTA, SG Trend, BTOP50", "indices de vrais fonds de futures gérés ; AQR TSMOM est un facteur "
                                     "académique de trend."),
    ]:
        add(P(f"<b>{term}</b> : {expl}", small))

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.6 * cm, title="Audit de la variante Les deux", author="Analyse trend")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    steps = sharpe_steps()
    fwd = forward_table()
    steps.to_csv(os.path.join(HERE, "sharpe_du_backtest_au_reel.csv"), float_format="%.3f")
    fwd.to_csv(os.path.join(HERE, "rendement_attendu_10ans.csv"), float_format="%.4f")
    print(steps.round(3).to_string())
    print(fwd.round(3).to_string())
    chart_sharpe(steps, os.path.join(OUT_FIG, "audit_sharpe.png"))
    chart_forward(fwd, os.path.join(OUT_FIG, "audit_rendement.png"))
    build(os.path.join(HERE, "rapport_audit_variante5.pdf"), steps, fwd)
    print("PDF écrit")


if __name__ == "__main__":
    main()
