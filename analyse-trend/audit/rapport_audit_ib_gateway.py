"""Audit de la variante 5 « Les deux » en pratique, exploitée via Interactive Brokers et IB Gateway.

Version du rapport d'audit centrée sur une exploitation automatisée chez IBKR par IB Gateway, avec des
rendements simulés bruts (avant impôt). Lit les résultats vérifiés :
- travail/ib_gateway/ et travail/verif_ib_gateway/ : faits et budget d'exploitation via IB Gateway ;
- travail/verif_avant_impot/fresh_full_grid_summary.csv : Monte Carlo brut, 4 tirages indépendants (40 000
  trajectoires), qui corrige le léger biais favorable du tirage unique de travail/cv_avant_impot/ ;
- les autres dossiers de travail/ pour le capital, les contrats et le Sharpe (comme rapport_audit.py).

Usage : python rapport_audit_ib_gateway.py
Écrit audit/rapport_audit_variante5_ib_gateway.pdf et audit/graphiques/audit_ibg_*.png.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, SimpleDocTemplate, Spacer

from rapport_audit import (GREY, GREY_LIGHT, HERE, INK, INK_2, OUT_FIG, PALETTE, P, chart_sharpe, fr, num, pct, read,
                           sharpe_steps, styles, table)


def p(x, d=1):
    return pct(0.0 if abs(x) < 0.5 * 10 ** -(d + 2) else x, d)  # pas de « −0,0 % »


# ---------------------------------------------------------------- rendements bruts (vérifiés, 4 tirages)

CASES = [  # (libellé du rapport, cas dans fresh_full_grid_summary.csv)
    ("Monétaire en euros (fonds monétaire)", None),
    ("Variante 5 soi-même, 100 k$", "DIY 21% 100k"),
    ("Variante 5 soi-même, 250 k$", "DIY 21% 250k"),
    ("Variante 5 soi-même, 1 M$", "DIY 21% 1M"),
    ("Variante 5 à 12 % de risque, 250 k$", "DIY 12% 250k"),
    ("Variante 5 à 12 % de risque, 1 M$", "DIY 12% 1M"),
    ("ETF trend UCITS (iMGP DBi, couvert €)", "ETF"),
    ("100 % S&P 500", "100% sp"),
    ("70 % S&P 500 + 30 % ETF trend", "70/30 sp"),
    ("100 % Nasdaq-100 (type QQQ)", "100% nq"),
    ("70 % Nasdaq-100 + 30 % ETF trend", "70/30 nq"),
]


def forward_table():
    g = read("verif_avant_impot/fresh_full_grid_summary.csv")
    g = g[g.horizon == 10]
    out = []
    for label, case in CASES:
        if case is None:  # fonds monétaire à €STR − 0,10 % : 2,4 % central, ±0,5 point
            out.append({"cas": label, "median": 0.023, "p10": 0.023, "p90": 0.023, "perte": 0.0, "sous_cash": np.nan,
                        "dd": 0.0, "dd10": np.nan, "pess": 0.018, "opt": 0.028})
            continue
        r = {s: g[(g.case == case) & (g.scen == s)].iloc[0] for s in ("pess", "central", "opt")}
        c = r["central"]
        out.append({"cas": label, "median": c.med / 100, "p10": c.p10 / 100, "p90": c.p90 / 100,
                    "perte": c.P_loss / 100, "sous_cash": c.P_below_cash / 100, "dd": c.dd_med / 100,
                    "dd10": c.dd_p10 / 100, "pess": r["pess"].med / 100, "opt": r["opt"].med / 100})
    return pd.DataFrame(out).set_index("cas")


def budget():
    b = read("verif_ib_gateway/recomputed_budget.csv").set_index("capital")
    s = read("ib_gateway/ib_gateway_budget_summary.csv").set_index("capital_usd")
    return b, s


# ---------------------------------------------------------------- graphique

def chart_forward(table_, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    names = list(table_.index)
    y = np.arange(len(names))
    kind = {n: (GREY_LIGHT if "Monétaire" in n else PALETTE[1] if "Variante" in n else
                PALETTE[0] if n.startswith("ETF") else GREY if n.startswith("100") else PALETTE[2]) for n in names}
    ax.set_axisbelow(True)
    for yy, n in zip(y, names):
        r = table_.loc[n]
        if r.p10 != r.p90:
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
    ax.set_xlabel("Rendement annuel sur 10 ans (%)", fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 1).replace("-", "−")))
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.985, "Ce qu'on peut raisonnablement attendre sur 10 ans", fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.01, 0.94, "Point : médiane des simulations. Trait : 8 cas sur 10 tombent dans cette fourchette "
                         "(10e au 90e centile). Euros, scénario central.", fontsize=7.8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.37, right=0.98, top=0.87, bottom=0.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- rapport

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(colors.HexColor(INK_2))
    canvas.drawString(1.5 * cm, 1.0 * cm, "Audit de la variante « Les deux » via IBKR et IB Gateway — estimations, "
                                          "pas une garantie de résultat")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


def build(path, steps, fwd, bud, summ):
    st = styles()
    story = []
    add = story.append
    body, small, h1, h2 = st["body"], st["small"], st["h1"], st["h2"]
    f = fwd.loc

    def bullets(items):
        for b in items:
            add(P(b.replace("\n", "<br/>"), st["bullet"], bulletText="•"))

    std = {c: bud.loc[c, "standard_central"] for c in (100000, 250000, 1000000)}

    add(P("Audit : la variante « Les deux » en pratique, via Interactive Brokers et IB Gateway", st["title"]))
    add(P("Capital nécessaire, contrats entiers, coûts d'une exploitation automatisée chez IBKR avec IB Gateway, "
          "faisabilité et rendement à attendre. Octobre 2026. Montants en dollars pour les contrats, rendements "
          "en euros.", st["subtitle"]))

    add(P("Verdict en bref", h1))
    bullets([
        "<b>La variante 5 telle que backtestée est un système de taille institutionnelle.</b> Pour la reproduire "
        "fidèlement sur ses 84 marchés, il faut environ <b>5 millions de dollars</b>, et au moins 1 à 2 M$ pour "
        "une copie correcte avec des contrats « micro ». En dessous, on ne tient qu'une partie des marchés : 14 à "
        "100 k$, 24 à 250 k$, 40 à 1 M$.",
        "<b>Le Sharpe de 1,17 n'est pas atteignable</b> (Sharpe : rendement au-dessus du monétaire divisé par la "
        "volatilité ; 0,5 est déjà bon). Il doit beaucoup aux années 1990 et à deux marchés douteux : l'éthanol, "
        "un artefact de données, et le minerai de fer de Singapour, probablement inaccessible. Sur les marchés "
        f"réellement négociables, il tombe à {fr(steps.iloc[1, 2], 2)} depuis 2010. Les portefeuilles qu'un "
        "particulier peut tenir font environ 0,4 depuis 2010, coûts réels compris.",
        "<b>Passer par IB Gateway est possible et c'est la bonne voie</b> : l'API d'Interactive Brokers est "
        "gratuite avec un compte IBKR Pro, IB Gateway tourne sur un petit serveur et le logiciel de Carver "
        "(pysystemtrade) s'y branche directement. Les commissions sont celles du compte. Frais fixes : environ "
        f"{std[100000]:.0f} $ par an (serveur et données). Les données différées gratuites suffisent pour un système "
        "quotidien. Seule contrainte : une validation sur le téléphone une fois par semaine.",
        f"<b>Rendement à attendre sur 10 ans</b> (médiane des simulations) : {p(f['Variante 5 soi-même, 100 k$', 'median'])} "
        f"par an à 100 k$, {p(f['Variante 5 soi-même, 250 k$', 'median'])} à 250 k$, "
        f"{p(f['Variante 5 soi-même, 1 M$', 'median'])} à 1 M$, contre "
        f"{p(f['Monétaire en euros (fonds monétaire)', 'median'])} pour le monétaire. Le risque est lourd : "
        f"{p(f['Variante 5 soi-même, 1 M$', 'perte'], 0)} à {p(f['Variante 5 soi-même, 100 k$', 'perte'], 0)} de "
        "chances de perdre de l'argent sur 10 ans, et une pire baisse médiane d'environ 46 %.",
        f"<b>Un ETF de trend UCITS fait presque aussi bien sans travail</b> : {p(f['ETF trend UCITS (iMGP DBi, couvert €)', 'median'])} "
        "par an attendus, avec une pire baisse médiane deux fois plus faible. La variante 5 ne le dépasse qu'à "
        "partir de 1 M$ (d'environ 0,7 point par an, avant de compter votre temps) ; à risque égal (12 %), elle ne "
        "le dépasse jamais.",
        "<b>Pour diversifier QQQ</b>, 70 % Nasdaq-100 et 30 % d'ETF trend rapportent "
        f"{p(f['70 % Nasdaq-100 + 30 % ETF trend', 'median'])} contre {p(f['100 % Nasdaq-100 (type QQQ)', 'median'])} "
        f"pour 100 % Nasdaq. La probabilité de perdre sur 10 ans passe de {p(f['100 % Nasdaq-100 (type QQQ)', 'perte'], 0)} "
        f"à {p(f['70 % Nasdaq-100 + 30 % ETF trend', 'perte'], 0)}, et la pire baisse médiane de "
        f"{p(f['100 % Nasdaq-100 (type QQQ)', 'dd'], 0)} à {p(f['70 % Nasdaq-100 + 30 % ETF trend', 'dd'], 0)}.",
        "<b>Recommandation :</b> sous 1 M$, préférer un ETF de trend. À partir de 1 M$, la variante 5 via IB "
        "Gateway devient défendable si le projet vous intéresse, en commençant par plusieurs mois en compte de "
        "démonstration (plan page 5).",
    ])

    add(P("Selon votre capital", h2))
    rows = [["Capital", "Marchés tenables (≥ 1 contrat)", "Sharpe depuis 2010, coûts réels",
             "Frais de transaction / an", "Frais fixes IB Gateway / an", "Médiane 10 ans [8 cas sur 10]",
             "Probabilité de perte sur 10 ans", "Verdict"]]
    for lab, n, sh, cost, fixed, key, verdict in [
        ("25–50 k$", "6 à 11", "0,28–0,34", "≈ 3,3 %", "≈ 400 $", None, "Non : trop peu de marchés"),
        ("100 k$", "14", "0,41", "3,3 %", f"{std[100000]:.0f} $", "Variante 5 soi-même, 100 k$",
         "Non : moins bien que l'ETF"),
        ("250 k$", "24", "0,42", "3,7 %", f"{std[250000]:.0f} $", "Variante 5 soi-même, 250 k$",
         "Égal à l'ETF, risque double"),
        ("1 M$", "40", "0,39", "2,9 %", f"{std[1000000]:.0f} $", "Variante 5 soi-même, 1 M$",
         "Défendable : +0,7 point/an"),
        ("≥ 5 M$", "63 à 84", "≈ 0,38–0,39", "2,3–2,5 %", "≈ 400 $", None, "Réplique fidèle possible"),
        ("ETF trend", "tout montant", "≈ 0,3 (fonds réels)", "TER 0,75 %", "–", "ETF trend UCITS (iMGP DBi, couvert €)",
         "Le plus simple"),
    ]:
        if key:
            r = fwd.loc[key]
            res, loss = f"{p(r['median'])} [{p(r.p10)} ; {p(r.p90)}]", p(r.perte, 0)
        else:
            res, loss = "non simulé", "–"
        rows.append([lab, n, sh, cost, fixed, res, loss, verdict])
    add(table(rows, [1.6 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm, 3.3 * cm, 1.9 * cm, 3.8 * cm], st))
    add(P("Sharpe des portefeuilles réellement tenables, simulés en contrats entiers avec les commissions IBKR "
          "et l'écart achat-vente. Rendements : simulation sur 10 ans en euros, frais fixes et intérêts perdus "
          "sur la trésorerie déduits, au risque de la variante 5 (21 % de volatilité) ; l'ETF est simulé à 12 %.",
          small))

    # ------------------------------------------------ 1. Le Sharpe
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
        "<b>L'exécution réelle coûte un peu.</b> Le backtest passe ses ordres au cours de clôture qui sert à "
        "calculer le signal, ce qui est impossible. En exécutant dans la séance suivante, le Sharpe perd 0,02 à "
        "0,04 ; il perdrait 0,07 avec un jour entier de retard.",
        "<b>Choisi après coup.</b> « Les deux » est la meilleure de 53 variantes de trend testées sur ces "
        "données : sa sélection gonfle à elle seule son Sharpe d'environ 0,25. Son avance sur la variante 2 "
        "n'est que tout juste significative sur 1990–2026, puis elle fond après 2010.",
        "<b>Les vrais fonds, frais réintégrés,</b> retrouvent un Sharpe de 0,41 à 0,49 depuis 2010 : depuis "
        "2010, le backtest ressemble à ce qu'obtiennent les professionnels avant leurs frais.",
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

    # ------------------------------------------------ 2. Capital
    add(PageBreak())
    add(P("2. Le capital : la taille des contrats décide de tout", h1))
    add(P("Les positions de la variante 5 sont petites par marché : en médiane, 3 % du capital pour une action, "
          "6 % pour une obligation, 12 % pour une devise, 2,5 % pour un produit agricole. Or un contrat a une "
          "taille fixe : on ne peut pas en acheter 0,3.", body))
    rows = [["Contrat", "Valeur d'un contrat (juillet 2026)", "Version réduite", "Valeur réduite",
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
    add(P("Capital pour tenir au moins 1 contrat, puis 4 (règle de Carver), à la position médiane, avec le plus "
          "petit contrat disponible : <b>0,63 M$ et 2,5 M$ pour un marché médian</b>, jusqu'à 7,7 M$ et 31 M$ pour "
          "le café. Un seul petit contrat dans chacun des 84 marchés représente déjà 4,9 M$ de valeur.", body))
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
        "à zéro et le compte s'écarte du système de 12 points par an. À 1 M$, l'écart tombe à 4 points, à 5 M$ "
        "à 1 point.",
        "<b>La bonne méthode pour un petit compte</b> est de choisir moins de marchés, chacun tenable, et de "
        "recalculer les poids. L'arrondi ne coûte alors presque rien ; ce qu'on perd, c'est la diversification "
        "(Sharpe depuis 1990, sans arrondi : 0,73 à 100 k$ et 0,84 à 250 k$, contre 1,17).",
        "<b>Accès :</b> IBKR bloque pour un particulier européen tout contrat sans DIC (règlement PRIIPs). Sont "
        "accessibles le CME, l'ICE, Eurex, Euronext, Osaka et Hong Kong ; Singapour et l'Australie sont "
        "probablement bloqués. Certains micros sont minces : le micro Euro Stoxx (2 400 contrats par jour) et le "
        "micro livre sterling (580 par jour). Pour le micro yen et les micros grains, aucun volume récent n'a été "
        "trouvé : à vérifier dans l'outil de recherche de contrats d'IBKR.",
        "<b>Petit compte et baisse :</b> un compte de 100 k$ démarré en 2016 serait tombé à 59 k$, sous son "
        "propre seuil minimal.",
    ])

    # ------------------------------------------------ 3. IB Gateway
    add(PageBreak())
    add(P("3. Faire tourner le système via IBKR et IB Gateway", h1))
    add(P("Comment ça marche", h2))
    bullets([
        "<b>IB Gateway</b> est la version sans graphiques de la plateforme TWS d'Interactive Brokers : même "
        "connexion, même API, environ 40 % de ressources en moins. C'est la version recommandée par "
        "pysystemtrade. <b>L'API est gratuite</b> ; on ne paie que les commissions habituelles et les données de "
        "marché.",
        "<b>Compte :</b> il faut un compte <b>IBKR Pro</b>. IBKR Lite n'a pas d'accès API, et il n'est de toute "
        "façon pas proposé aux clients européens d'IBKR Irlande. Pas de frais d'inactivité. Créer un "
        "<b>identifiant dédié à l'API</b> (gratuit) : une seule session est possible par identifiant, et se "
        "connecter sur le téléphone couperait sinon IB Gateway. Un <b>compte de démonstration</b> gratuit, avec "
        "la même API, permet de tout tester (port 4002 en démonstration, 4001 en réel).",
        "<b>Machine :</b> un serveur loué (VPS) de 4 à 8 Go de mémoire, ou un mini-PC à la maison. L'image "
        "Docker gnzsnz/ib-gateway-docker regroupe IB Gateway et <b>IBC</b>, un outil gratuit qui automatise le "
        "démarrage, la connexion et le redémarrage quotidien.",
        "<b>La contrainte :</b> une connexion complète avec double authentification est exigée <b>une fois par "
        "semaine</b> (le dimanche à partir de 7 h, heure de Paris) : il faut valider une notification sur "
        "l'application IBKR Mobile dans les 3 minutes. IBC renvoie la demande si on la rate, mais ne peut pas "
        "valider à votre place. Les redémarrages quotidiens, eux, ne demandent rien.",
        "<b>Maintenance d'IBKR :</b> éviter de faire tourner les tâches pendant les coupures quotidiennes "
        "(environ 6 h 15 à 7 h 45, heure de Paris) et le samedi de 6 h à 8 h.",
        "<b>Logiciel :</b> pysystemtrade (Python 3.10 ou plus, MongoDB, bibliothèque ib_async) se connecte à IB "
        "Gateway et fait tourner chaque jour : mise à jour des prix, calcul des positions, génération et envoi "
        "des ordres, enregistrement des exécutions et rapprochement des positions. Les règles de la variante 5 "
        "(environ 200 lignes dans trend.py) doivent être reprogrammées dans pysystemtrade, ou dans un script "
        "Python plus simple branché directement sur ib_async.",
        "<b>Ordres :</b> au marché, à cours limité ou « Adaptive » d'IBKR (exécution intelligente, valable la "
        "journée). Avec des données différées, utiliser des ordres au marché ou Adaptive. Les limites de l'API "
        "(50 messages par seconde, 100 lignes de données) ne gênent pas un système quotidien de 14 à 40 marchés.",
    ])
    add(P("Les données", h2))
    bullets([
        "<b>Les données différées sont gratuites via l'API</b> (10 minutes pour le CME et l'ICE, 15 pour Eurex et "
        "Euronext, 20 pour Osaka) et suffisent pour un système qui décide une fois par jour.",
        "<b>Historique :</b> la documentation récente d'IBKR permet de demander des barres historiques en "
        "différé. pysystemtrade n'en télécharge qu'environ 1 an, et IBKR ne garde les contrats expirés que 2 ans. "
        "Pour démarrer, il faut donc partir de l'historique déjà construit dans ce projet. Les fichiers gratuits "
        "de Carver s'arrêtent en mars 2024. À tester sur votre identifiant avant de s'y fier.",
    ])
    rows = [["Abonnement (particulier)", "Prix", "Utile ?"],
            ["Bundle futures américains (CME, CBOT, NYMEX, COMEX)", "10 $/mois",
             "Offert dès 30 $ de commissions dans le mois (atteint en moyenne, de justesse à 100 k$)"],
            ["Eurex (niveau 1)", "8 à 10,75 €/mois", "Oui dès 100 k$ (Schatz, Euro Stoxx)"],
            ["Euronext", "3 €/mois", "Oui (CAC, colza, sucre blanc)"],
            ["Osaka (OSE)", "0 à 200 ¥/mois", "Oui (Nikkei)"],
            ["ICE (États-Unis ou Europe), temps réel", "123 à 161 $/mois par flux",
             "Non : le différé et des ordres au marché suffisent"]]
    add(table(rows, [6.4 * cm, 3.6 * cm, 8.0 * cm], st))
    add(P("Le temps réel n'est obligatoire nulle part pour un système quotidien ; il exige 500 $ sur le compte et "
          "il est facturé par identifiant.", small))

    add(P("Budget d'exploitation", h2))
    rows = [["Frais fixes par an", "100 k$", "250 k$", "1 M$"]]
    for col, lab in [("minimal", "Minimal : mini-PC ou petit serveur, tout en différé"),
                     ("standard_central", "Standard (retenu) : serveur 8 Go, Eurex, Euronext, Osaka"),
                     ("standard_robust_norgate", "Standard + historique Norgate en secours"),
                     ("comfortable_hi", "Confort : 2e serveur, Norgate, ICE en temps réel")]:
        rows.append([lab] + [f"{bud.loc[c, col]:,.0f} $".replace(",", " ") for c in (100000, 250000, 1000000)])
    add(table(rows, [9.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm], st))
    add(Spacer(1, 4))
    rows = [["Coût total, % du capital par an", "100 k$", "250 k$", "1 M$"]]
    for col, lab in [("variable_fee_plus_halfspread_pct", "Commissions IBKR, frais de bourse et écart achat-vente"),
                     ("fixed_standard_pct", "Frais fixes (budget standard)"),
                     ("cash_shortfall_vs_estr_pct", "Intérêts perdus sur la trésorerie"),
                     ("total_standard_pct", "Total")]:
        rows.append([lab] + [fr(summ.loc[c, col], 2) + " %" for c in (100000, 250000, 1000000)])
    add(table(rows, [9.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm], st))
    add(P("Les commissions et l'écart achat-vente sont déjà déduits des Sharpe et des rendements simulés ; les "
          "frais fixes et les intérêts perdus aussi. Commissions IBKR estimées : 33 à 66 $ par mois à 100 k$, 85 à "
          "154 $ à 250 k$, 275 à 479 $ à 1 M$.", small))
    bullets([
        "<b>Commissions :</b> 0,85 $ par contrat américain standard et 0,25 $ par micro, plus les frais de bourse "
        "(environ 1,4 $ sur un E-mini, 0,35 $ sur un micro) ; 0,90 € sur Eurex. L'écart achat-vente fait environ "
        "la moitié du coût ; les micros coûtent 1,4 à 2 fois plus cher par dollar de valeur.",
        "<b>Trésorerie :</b> l'argent bloqué en marge des futures (17 à 23 % du capital en moyenne) ne rapporte "
        "rien chez IBKR. Le reste rapporte le taux de référence moins 0,5 point, avec deux restrictions : rien "
        "sur les premiers 10 000 $ (ou €), et un taux réduit au prorata sous 100 000 $ de compte. Activer le "
        "transfert automatique des excédents (« Excess Funds Sweep ») vers la partie titres. Le monétaire euro "
        "(€STR, environ 2,4 %) sert de référence.",
        "<b>Serveur :</b> compter 99 à 148 $ par an pour un serveur de 8 Go (OVH, netcup, Hetzner selon "
        "disponibilité). Un mini-PC à la maison coûte environ 25 $ d'électricité par an, un PC de bureau 60 à "
        "120 $.",
    ])

    add(P("Plan de mise en route", h2))
    bullets([
        "<b>1. Compte :</b> ouvrir un compte IBKR Pro (Irlande) et demander les autorisations futures. Vérifier "
        "que chaque contrat visé est négociable (DIC). Créer l'identifiant API et le compte de démonstration, et "
        "partager les données avec ce dernier. Compter 3 à 8 h.",
        "<b>2. Serveur :</b> Ubuntu et Docker, image ib-gateway-docker avec IBC en mode démonstration. Ports "
        "limités à la machine, accès à distance par tunnel SSH, redémarrage automatique. Compter 4 à 12 h.",
        "<b>3. Logiciel :</b> installer pysystemtrade et MongoDB, configurer les 14 à 40 marchés, charger "
        "l'historique et reprogrammer la variante 5. Vérifier ensuite que les positions calculées collent au "
        "backtest. Compter 60 à 180 h.",
        "<b>4. Démonstration :</b> 2 à 3 mois en compte de démonstration, en passant au moins un roulement de "
        "contrats, une validation du dimanche et une mise à jour d'IB Gateway.",
        "<b>5. Réel :</b> démarrer avec une partie du capital (par exemple les marchés du CME, ou un tiers), "
        "puis monter progressivement.",
        "<b>Ensuite :</b> 1 à 2 heures par semaine en moyenne. Il faut valider la connexion du dimanche, lire le "
        "rapport quotidien (5 à 10 min) et suivre les roulements (62 à 169 ordres par an). S'y ajoutent les "
        "alertes de prix aberrants à vérifier, un contrôle mensuel et une revue annuelle.",
    ])

    # ------------------------------------------------ 4. Risques pratiques
    add(P("4. Les risques pratiques", h1))
    bullets([
        "<b>Marge :</b> en moyenne 17 à 23 % du capital, 26 à 40 % les mauvais jours (1 jour sur 20, selon le "
        "modèle de marge), 33 à 62 % au pire historique. En stress extrême (bourses qui doublent leurs marges en "
        "pleine baisse), 70 à 95 % du capital. IBKR n'envoie pas d'appel de marge : il liquide automatiquement, "
        "au prix du marché, les positions qu'il choisit. Il faut donc garder tout le capital chez le courtier et "
        "s'imposer une marge maximale de 40 à 50 %.",
        "<b>Exposition :</b> la variante 5 détient en moyenne 8,8 fois son capital en contrats, jusqu'à 28 fois, "
        "surtout en taux courts et obligations. Un choc brutal (gilts britanniques en 2022, franc suisse en "
        "2015) n'est représenté que dans la mesure où l'historique en contient.",
        "<b>Pannes typiques :</b> prix aberrant dans les données, roulement oublié ou date de premier avis "
        "dépassée, IB Gateway déconnecté (validation du dimanche manquée), jours fériés différents selon les "
        "bourses, soldes négatifs en devise facturés au taux de référence plus 1,5 %.",
        "<b>Tenir dans les mauvaises périodes :</b> depuis 1990, la variante 5 a connu 7 baisses de plus de 20 %. "
        "Deux ont duré environ 4 ans : 2016–2020 (−47 %) et celle commencée en septembre 2022 (−54 % au plus "
        "bas). On compte 9 années négatives sur 37. Le risque d'abandonner au plus bas est le principal danger "
        "pratique.",
        "Le backtest s'arrête au 10 juillet 2026. Depuis, les fonds de trend ont nettement remonté : l'indice SG "
        "CTA a gagné environ 6,7 % entre le 10 juillet et la mi-septembre.",
    ])

    # ------------------------------------------------ 5. Rendement
    add(PageBreak())
    add(P("5. Quel rendement attendre ?", h1))
    add(Image(os.path.join(OUT_FIG, "audit_ibg_rendement.png"), width=18 * cm, height=18 * cm * 4.8 / 7.6))
    rows = [["Sur 10 ans", "Médiane", "8 cas sur 10", "Pessimiste / optimiste", "Probabilité de perte",
             "Probabilité de faire moins que le monétaire", "Pire baisse médiane"]]
    for name in fwd.index:
        r = fwd.loc[name]
        spread = "–" if r.p10 == r.p90 else f"{p(r.p10)} à {p(r.p90)}"
        rows.append([name, p(r["median"]), spread, f"{p(r.pess)} / {p(r.opt)}",
                     "–" if r.perte == 0 else p(r.perte, 0),
                     "–" if np.isnan(r.sous_cash) else p(r.sous_cash, 0), "–" if r.dd == 0 else p(r.dd, 0)])
    add(table(rows, [5.0 * cm, 1.5 * cm, 2.5 * cm, 2.6 * cm, 1.8 * cm, 2.2 * cm, 2.4 * cm], st))
    add(Spacer(1, 4))
    bullets([
        "<b>Sharpe futur supposé</b> (après coûts et exécution réalistes) : 0,20 à 100 k$, 0,24 à 250 k$, 0,27 à "
        "1 M$ en scénario central ; pessimiste 0 à 0,05, optimiste 0,40 à 0,47. Ancrages : environ 0,4 depuis "
        "2010 pour les portefeuilles tenables, 0,21 à 0,27 pour les vrais fonds, et une décote de 25 à 35 % pour "
        "le choix après coup. L'ETF est supposé à 0,25 (de 0,10 à 0,40).",
        "<b>Simulation :</b> trajectoires de 10 ans tirées par blocs de six mois dans l'historique de chaque "
        "portefeuille, quatre tirages indépendants (40 000 trajectoires). Monétaire à 2,4 % (€STR), règles de "
        "rémunération de la trésorerie d'IBKR, frais fixes de 400 $ par an. Actions américaines en euros : 6 % "
        "par an en central, d'après les hypothèses long terme de J.P. Morgan et Vanguard.",
        "<b>Ce qui pèse le plus :</b> le Sharpe futur. ±0,1 de Sharpe change le rendement de la variante 5 "
        "d'environ ±2,2 points par an, de l'ETF de ±1,3, du mélange 70/30 de ±0,4. Le budget fixe ne compte qu'à "
        "100 k$ : chaque 1 000 $ par an de frais fixes en plus retire 1,05 point à 100 k$, 0,36 à 250 k$, 0,09 à "
        "1 M$. Valoriser votre temps (80 h par an à 25 €/h) retire 2,4 points à 100 k$ et 0,2 à 1 M$.",
        "<b>Viser 12 % de risque au lieu de 21 %</b> réduit fortement les baisses (pire baisse médiane −26 % au "
        "lieu de −46 %) mais aussi le rendement : 3,5 % au lieu de 4,6 % à 250 k$, 4,6 % au lieu de 5,4 % à 1 M$. "
        "À ce niveau de risque, la variante 5 ne fait jamais mieux que l'ETF.",
        "En pouvoir d'achat (2 % d'inflation), retirer environ 2 points à tous ces chiffres.",
    ])

    # ------------------------------------------------ 6. Alternatives
    add(PageBreak())
    add(P("6. Les alternatives toutes faites", h1))
    rows = [["Fonds", "ISIN", "Type", "Frais", "Remarques"],
            ["iMGP DBi Managed Futures R EUR HP (MFEH, Euronext Paris)", "LU3359622902",
             "ETF UCITS capitalisant, couvert en euros", "0,75 %",
             "Coté à Paris depuis le 21/09/2026. Réplique l'indice SG CTA (toutes stratégies de CTA, pas du "
             "trend pur). Fonds de 698 M$, mais classe couverte récente (mai 2026) et petite."],
            ["iMGP DBi Managed Futures R USD (Euronext Paris : DBMF)", "LU2951555585", "ETF UCITS, en dollars",
             "0,75 %", "Même fonds, risque de change dollar. Le jumeau américain DBMF : 9,2 % par an de 2019 à "
                       "2026, volatilité 12 %, pire baisse −20 %."],
            ["BNP Paribas Easy Managed Futures", "LU3307218399", "ETF UCITS synthétique, couvert en euros",
             "0,60 %", "Modèle de trend maison via un swap. Lancé en mai 2026, environ 10 M€. Coté sur Xetra "
                       "(EEAU) ; cotation à Paris non confirmée."],
            ["AQR Managed Futures UCITS (RAE)", "LU1662502183", "Fonds (pas un ETF)", "TER 1,39 %",
             "Le plus proche d'un trend « pur » à la Carver. Classe A en dollars : 0,60 % plus 10 % des gains."],
            ["Winton Trend UCITS (I EUR)", "IE00BG382R37", "Fonds", "1,05 %", "Classe institutionnelle."],
            ["Man AHL Trend Alternative (DNY H EUR)", "LU0424370004", "Fonds", "≈ 2,75 %",
             "2 % de frais plus une commission de performance : cher."]]
    add(table(rows, [4.4 * cm, 2.5 * cm, 2.9 * cm, 1.5 * cm, 6.7 * cm], st))
    add(P("Ces ETF s'achètent chez IBKR comme n'importe quelle action, sur Euronext Paris ou Xetra. Les ETF "
          "américains (DBMF coté à New York, KMLM) restent inaccessibles aux particuliers européens faute de DIC. "
          "Vérifiez cotations et frais sur la fiche du fonds avant d'acheter.", small))

    # ------------------------------------------------ Méthode et lexique
    add(PageBreak())
    add(P("Méthode, vérifications et limites", h1))
    bullets([
        "<b>Organisation :</b> volets indépendants (capital et contrats, coûts, IB Gateway, rendement futur, "
        "faisabilité) et simulation en contrats entiers, chacun revu par un vérificateur chargé de le "
        "contredire. Scripts, résultats et rapports complets : analyse-trend/audit/travail/.",
        "<b>Corrections faites en route :</b>\n"
        "– portefeuilles pour particulier refaits ;\n"
        "– écart de prix de l'Euribor corrigé ;\n"
        "– la marge chez IBKR ne rapporte rien ;\n"
        "– ISIN des ETF iMGP DBi corrigés ;\n"
        "– le tirage unique de la simulation flattait la variante 5 d'environ 0,25 point, d'où la moyenne de "
        "quatre tirages ;\n"
        "– les serveurs les moins chers d'Hetzner sont souvent indisponibles depuis septembre 2026 ;\n"
        "– l'abonnement Eurex « retail » ne couvre pas les obligations.",
        "<b>Erreurs de données trouvées dans le projet :</b> le gasoil (QS1) était associé au mauvais code chez "
        "Carver dans les travaux sur le carry ; l'éthanol (CUA1) est inutilisable.",
        "<b>Principales limites :</b>\n"
        "– le Sharpe futur est un jugement, ce qui donne des fourchettes larges ;\n"
        "– les micros sont supposés avoir toujours existé ;\n"
        "– environ 25 coûts de micros sont estimés ;\n"
        "– les prix exacts des données Eurex, Osaka et ICE chez IBKR sont à confirmer dans votre compte ;\n"
        "– les temps de mise en place sont des estimations ;\n"
        "– le change euro-dollar sur les gains est ignoré ;\n"
        "– les données s'arrêtent au 10 juillet 2026.",
        "<b>Sources principales :</b>\n"
        "– documentation IBKR (API, données différées, tarifs) ;\n"
        "– pysystemtrade et ib-gateway-docker sur GitHub ;\n"
        "– configuration et coûts IBKR publiés par Carver ;\n"
        "– barèmes CME, ICE et Eurex ;\n"
        "– fiches justETF, finanzfluss et extraETF ;\n"
        "– indices SG Trend, SG CTA, BTOP50 et AQR TSMOM ;\n"
        "– hypothèses long terme de J.P. Morgan et Vanguard.\n"
        "Liens : travail/ib_gateway/ib_gateway_facts.csv et travail/rapports_agents/.",
    ])
    add(P("Lexique", h2))
    for term, expl in [
        ("Backtest", "simulation d'une stratégie sur l'historique des prix."),
        ("Sharpe", "rendement au-dessus du monétaire divisé par la volatilité ; 0,5 est bon, 1 excellent."),
        ("API", "interface qui permet à un programme de passer des ordres et de lire les prix chez le courtier."),
        ("IB Gateway, TWS", "logiciels de connexion à Interactive Brokers ; IB Gateway est la version légère, "
                            "sans graphiques."),
        ("IBC, Docker, VPS", "outil qui pilote IB Gateway automatiquement ; format d'installation clé en main ; "
                             "serveur loué en ligne."),
        ("Double authentification", "validation de la connexion sur l'application IBKR Mobile."),
        ("pysystemtrade, ib_async", "logiciel de trading systématique de Carver ; bibliothèque Python de "
                                    "connexion à IBKR."),
        ("Contrat à terme (future), micro", "engagement d'acheter ou vendre plus tard à un prix fixé aujourd'hui ; "
                                            "un « micro » vaut 1/10e (parfois moins) du contrat standard."),
        ("Roulement", "remplacer un contrat qui arrive à échéance par le suivant."),
        ("Écart achat-vente, demi-écart", "différence entre le meilleur prix d'achat et de vente ; on en paie "
                                          "la moitié à chaque ordre."),
        ("Marge", "dépôt de garantie exigé pour détenir des contrats."),
        ("€STR, SOFR, Euribor, SONIA", "taux du monétaire en euros, et taux courts dollar, euro et sterling."),
        ("Schatz, Bobl, Bund, BTP, OAT, gilts", "emprunts d'État allemands (2, 5 et 10 ans), italiens, français "
                                                 "et britanniques."),
        ("UCITS, TER, DIC", "fonds européen réglementé ; frais annuels totaux ; document d'information clé exigé "
                            "pour vendre un produit à un particulier européen."),
        ("Centile, médiane", "le 10e centile est dépassé dans 90 % des cas ; la médiane dans 50 %."),
        ("SG CTA, SG Trend, BTOP50", "indices de vrais fonds de futures gérés ; AQR TSMOM est un facteur "
                                     "académique de trend."),
    ]:
        add(P(f"<b>{term}</b> : {expl}", small))

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.6 * cm, title="Audit de la variante Les deux via IB Gateway",
                            author="Analyse trend")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    steps = sharpe_steps()
    fwd = forward_table()
    bud, summ = budget()
    fwd.to_csv(os.path.join(HERE, "rendement_attendu_10ans_ib_gateway.csv"), float_format="%.4f")
    print(fwd.round(3).to_string())
    chart_sharpe(steps, os.path.join(OUT_FIG, "audit_sharpe.png"))
    chart_forward(fwd, os.path.join(OUT_FIG, "audit_ibg_rendement.png"))
    build(os.path.join(HERE, "rapport_audit_variante5_ib_gateway.pdf"), steps, fwd, bud, summ)
    print("PDF écrit")


if __name__ == "__main__":
    main()
