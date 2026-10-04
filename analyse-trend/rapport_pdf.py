"""Rapport PDF : indicateurs, performances année par année et comparaison au S&P 500 et au Nasdaq-100
des variantes du système de trend.

Variantes (84 marchés, 1990 → juillet 2026) :
1. Montant fixe par marché : taille fixée sur la volatilité des 3 premières années, jamais ajustée.
2. Système actuel : stratégie neuf de Carver, taille selon la volatilité courante de chaque marché.
3. + stratégie 13 : prévisions réduites quand la volatilité d'un marché est haute par rapport à son passé.
4. + pilotage du risque du portefeuille : toutes les positions × 20 % / volatilité récente du système.
5. Les deux : 3 et 4 ensemble.
Section à part : trend + carry sur les données de Carver (1990 → mars 2024).

Les futures donnent un rendement au-dessus du monétaire ; on ajoute le taux des T-bills à 3 mois pour obtenir
le rendement total (le capital non utilisé en marge rapporte le monétaire), comparable à un indice actions
dividendes réinvestis.

Usage : python rapport_pdf.py --data DOSSIER_MKT
Écrit rapport/rapport_variantes_trend.pdf, rapport/performances_annuelles.xlsx, resultats/rapport_*.csv,
graphiques/rapport_*.png.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import listings
import run_backtests as rb
import trend

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
NEGATIVE = "#b42318"

VARIANTS = ["1. Montant fixe", "2. Système actuel", "3. + stratégie 13", "4. + pilotage du risque", "5. Les deux"]
SP500, NDX, CASH = "S&P 500", "Nasdaq-100", "Monétaire"
CARRY = ["Trend seul", "Trend + carry", "Trend + carry + strat. 13"]
SUBPERIODS = [("1990", "2026"), ("2000", "2026"), ("2010", "2026"), ("2020", "2026")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ---------------------------------------------------------------- données

def load_series(data):
    full, groups, _ = rb.load_futures(data)
    rets = full.apply(lambda s: s.where(s.index >= pd.Timestamp(listings.LISTING.get(s.name, "1900-01-01"))))
    risk = groups.replace({"Bonds": "Taux", "STIR": "Taux"})
    costs = {c: rb.COSTS[g] for c, g in groups.items()}
    roll = {c: 2 * rb.COSTS[g] * rb.ROLLS_PER_YEAR[g] for c, g in groups.items()}
    excess = pd.DataFrame({
        VARIANTS[0]: trend.run_portfolio(rets, risk, costs, roll, sizing_vol=trend.fixed_vol(rets))["net"],
        VARIANTS[1]: trend.run_portfolio(rets, risk, costs, roll)["net"],
        VARIANTS[2]: trend.run_portfolio(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)["net"],
        VARIANTS[3]: trend.run_vol_targeted(rets, risk, costs, roll)["net"],
        VARIANTS[4]: trend.run_vol_targeted(rets, risk, costs, roll, forecast_fn=trend.forecast_regime)["net"],
    }).loc[rb.START:]
    return excess


def on_calendar(close, index):
    """Rendements d'un indice sur le calendrier des futures (jours fériés américains = rendement nul)."""
    close = close[~close.index.duplicated()]
    return close.reindex(close.index.union(index)).ffill().reindex(index).pct_change()


def load_benchmarks(data, index):
    etf = f"{data}/us-etf/alm0421"
    cash = rb.load_close(f"{data}/us-etf/alm0421_macro/DTB3.csv", "value") / 100 / 252
    cash = cash.reindex(cash.index.union(index)).ffill().reindex(index)
    sp = on_calendar(rb.load_close(f"{etf}/IDX_SP500TR.csv"), index)
    # Nasdaq-100 : indice de prix avant le lancement de QQQ (10 mars 1999), puis QQQ dividendes réinvestis.
    qqq = on_calendar(rb.load_close(f"{etf}/QQQ.csv"), index)
    ndx_price = on_calendar(rb.load_close(f"{etf}/IDX_NDX.csv", "close"), index)
    ndx = ndx_price.where(index <= pd.Timestamp("1999-03-10"), qqq)
    return pd.DataFrame({SP500: sp, NDX: ndx, CASH: cash}, index=index)


# ---------------------------------------------------------------- indicateurs

def years_between(index):
    return (index[-1] - index[0]).days / 365.25 + 1 / 261  # le premier rendement couvre déjà un jour


def cagr(total):
    total = total.dropna()
    return (1 + total).prod() ** (1 / years_between(total.index)) - 1


def drawdown_info(total):
    curve = (1 + total.dropna()).cumprod()
    dd = curve / curve.cummax() - 1
    trough = dd.idxmin()
    peak = curve.loc[:trough].idxmax()
    after = curve.loc[trough:]
    recovered = after[after >= curve.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    underwater = dd < 0
    runs = underwater.ne(underwater.shift()).cumsum()[underwater]
    longest = runs.groupby(runs).apply(lambda g: (g.index[-1] - g.index[0]).days).max() if len(runs) else 0
    return {"max_dd": dd.min(), "peak": peak, "trough": trough, "recovery": recovery,
            "longest_days": longest, "dd": dd}


def metrics(total, excess, cash, sp_excess, ndx_excess, is_cash=False):
    """Indicateurs d'une série. total : rendement total quotidien ; excess : au-dessus du monétaire."""
    total, excess = total.dropna(), excess.dropna()
    monthly = (1 + total).resample("ME").prod() - 1
    yearly = (1 + total).groupby(total.index.year).prod() - 1
    full_years = yearly.loc[: total.index[-1].year - 1] if total.index[-1].month < 12 else yearly
    downside = np.sqrt((excess.clip(upper=0) ** 2).mean())
    info = drawdown_info(total)
    growth = cagr(total)
    # Même risque que le S&P 500 : rendement au-dessus du monétaire × (vol du S&P / vol propre), plus le monétaire.
    same_risk = None if is_cash else excess * (sp_excess.std() / excess.std()) + cash.reindex(excess.index)

    def monthly_corr(other):
        a = (1 + excess).resample("ME").prod() - 1
        b = (1 + other.dropna()).resample("ME").prod() - 1
        return a.corr(b)

    out = {
        "CAGR (rendement total)": growth,
        "CAGR au même risque que le S&P 500": np.nan if is_cash else cagr(same_risk),
        "Volatilité annualisée (écart-type quotidien)": total.std() * 16,
        "Écart-type des performances annuelles": full_years.std(),
        "Sharpe": excess.mean() / excess.std() * 16,
        "Sortino": excess.mean() / downside * 16,
        "Calmar (CAGR / pire baisse)": growth / -info["max_dd"],
        "Pire baisse (max drawdown)": info["max_dd"],
        "Meilleure année": full_years.max(),
        "Pire année": full_years.min(),
        "Années positives": (full_years > 0).mean(),
        "Pire mois": monthly.min(),
        "Mois positifs": (monthly > 0).mean(),
        "Skew mensuel": monthly.skew(),
        "Corrélation S&P 500 (mensuelle)": monthly_corr(sp_excess),
        "Corrélation Nasdaq-100 (mensuelle)": monthly_corr(ndx_excess),
    }
    if is_cash:  # ratios sans objet pour le monétaire
        for k in out:
            if k not in ("CAGR (rendement total)", "Volatilité annualisée (écart-type quotidien)",
                         "Écart-type des performances annuelles", "Meilleure année", "Pire année",
                         "Années positives", "Pire mois", "Mois positifs"):
                out[k] = np.nan
    out.update({"_best_year": int(full_years.idxmax()), "_worst_year": int(full_years.idxmin()),
                "_dd": None if is_cash else info})
    return out


# ---------------------------------------------------------------- mise en forme

def pct(x, digits=1, sign=False):
    if pd.isna(x):
        return "–"
    s = f"{x * 100:+.{digits}f}" if sign else f"{x * 100:.{digits}f}"
    return s.replace("-", "−").replace(".", ",") + " %"


def num(x, digits=2):
    return "–" if pd.isna(x) else f"{x:.{digits}f}".replace("-", "−").replace(".", ",")


MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def month_year(ts):
    return "–" if ts is None else f"{MONTHS[ts.month - 1]} {ts.year}"


def fmt_metric(name, value):
    if name in ("Sharpe", "Sortino", "Calmar (CAGR / pire baisse)", "Skew mensuel") or name.startswith("Corrélation"):
        return num(value)
    if name in ("Années positives", "Mois positifs"):
        return pct(value, 0)
    return pct(value)


# ---------------------------------------------------------------- graphiques

def fr(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def spread_labels(values, min_gap):
    """Écarte des étiquettes (en coordonnées log) pour qu'elles ne se chevauchent pas."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def thousands(v):
    return f"{v:,.0f}".replace(",", " ")


def growth_chart(total, colors_, title, subtitle, path, ticks, dashed=(SP500, NDX), height=4.7):
    fig, ax = plt.subplots(figsize=(7.6, height))
    ends = {}
    for name in total:
        curve = (1 + total[name].dropna()).cumprod()
        ax.plot(curve.index, curve.values, color=colors_[name], linewidth=1.5 if name in dashed else 1.9,
                linestyle=(0, (4, 2)) if name in dashed else "-")
        ends[name] = curve
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: thousands(v) if v >= 1 else fr(v, 1)))
    ax.set_ylim(ticks[0] * 0.7, None)
    first = min(c.index[0] for c in ends.values())
    ax.set_ylabel("Valeur de 1 € (échelle logarithmique)")
    names = list(ends)
    decades = np.log10(ticks[-1] / ticks[0]) + 0.5
    logs = spread_labels([np.log10(ends[n].iloc[-1]) for n in names], 0.034 * decades * 4.7 / height)
    last = max(c.index[-1] for c in ends.values())
    ax.set_xlim(first, last)
    for n, y in zip(names, logs):
        ax.annotate(f"{n} : ×{thousands(ends[n].iloc[-1])}", xy=(ends[n].index[-1], ends[n].iloc[-1]),
                    xytext=(1.03, 10 ** y), textcoords=("axes fraction", "data"), va="center", fontsize=7.6,
                    color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colors_[n], linewidth=0.9, shrinkA=0, shrinkB=0))
    fig.text(0.01, 1 - 0.07 / height, title, fontsize=11, fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.28 / height, subtitle, fontsize=7.8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.09, right=0.69, top=1 - 0.66 / height, bottom=0.3 / height)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cagr_chart(table, colors_, path):
    """Petits multiples : un panneau par période, une barre par série (couleur = identité de la série)."""
    periods = list(table.columns)
    fig, axes = plt.subplots(1, len(periods), figsize=(7.6, 3.6), sharey=True)
    names = list(table.index)
    y = np.arange(len(names))
    hi = table.max().max() * 100
    for ax, p in zip(axes, periods):
        vals = table[p].values * 100
        ax.barh(y, vals, color=[colors_[n] for n in names], height=0.7)
        for yy, v in zip(y, vals):
            ax.text(max(v, 0) + hi * 0.03, yy, fr(v), va="center", fontsize=7.4, color=INK)
        a, b = p.split("-")
        ax.set_title(f"{a} → {'juil. ' if b == '2026' else ''}{b}", fontsize=8.6)
        ax.set_xlim(min(0, vals.min() * 1.15), hi * 1.32)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.985, "CAGR selon la période de départ (% par an, rendement total)", fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.01, 0.925, "Variantes de trend en couleur, indices actions (dividendes réinvestis) en gris. "
                          "Toutes les périodes se terminent le 10 juillet 2026.", fontsize=7.8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.215, right=0.99, top=0.80, bottom=0.08, wspace=0.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def drawdown_chart(total, colors_, path):
    names = list(total.columns)
    rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(7.6, 1.45 * rows + 0.45), sharex=True, sharey=True)
    axes = axes.ravel()
    lo = 0
    for ax, name in zip(axes, names):
        info = drawdown_info(total[name])
        dd = info["dd"] * 100
        lo = min(lo, dd.min())
        ax.fill_between(dd.index, dd.values, 0, color=colors_[name], linewidth=0)
        ax.set_title(f"{name} : pire baisse {pct(info['max_dd'], 0)}", fontsize=8.2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].set_ylim(lo * 1.08, 4)
    axes[0].set_xlim(total.index[0], total.index[-1])
    for ax in axes[::2]:
        ax.set_ylabel("%", fontsize=7)
    fig.text(0.01, 0.995, "Baisse depuis le plus haut précédent (drawdown), rendement total", fontsize=11,
             fontweight="bold", va="top")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.04, hspace=0.45, wspace=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- PDF

def styles():
    pdfmetrics.registerFont(TTFont("DejaVu", f"{FONT_DIR}/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu",
                                  boldItalic="DejaVu-Bold")
    base = dict(fontName="DejaVu", textColor=colors.HexColor(INK), alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", fontSize=17, leading=21, fontName="DejaVu-Bold", spaceAfter=4,
                                **{k: v for k, v in base.items() if k != "fontName"}),
        "subtitle": ParagraphStyle("subtitle", fontSize=9.5, leading=13, textColor=colors.HexColor(INK_2),
                                   spaceAfter=10, fontName="DejaVu"),
        "h1": ParagraphStyle("h1", fontSize=13, leading=16, fontName="DejaVu-Bold", spaceBefore=8, spaceAfter=6,
                             textColor=colors.HexColor(INK)),
        "h2": ParagraphStyle("h2", fontSize=10.5, leading=13, fontName="DejaVu-Bold", spaceBefore=6, spaceAfter=3,
                             textColor=colors.HexColor(INK)),
        "body": ParagraphStyle("body", fontSize=8.8, leading=12, spaceAfter=4, **base),
        "bullet": ParagraphStyle("bullet", fontSize=8.8, leading=12, spaceAfter=2, leftIndent=12, bulletIndent=2,
                                 **base),
        "small": ParagraphStyle("small", fontSize=7.4, leading=9.5, textColor=colors.HexColor(INK_2),
                                fontName="DejaVu", spaceAfter=3),
        "cell": ParagraphStyle("cell", fontSize=6.6, leading=8, fontName="DejaVu-Bold",
                               textColor=colors.HexColor(INK)),
        "label": ParagraphStyle("label", fontSize=7.2, leading=8.4, fontName="DejaVu",
                                textColor=colors.HexColor(INK)),
    }


def data_table(header, rows, col_widths, st, first_col_bold=False, color_negative=False, zebra=True,
               font_size=7.4, row_height=None, bold_rows=()):
    head = [Paragraph(h, st["cell"]) for h in header]
    body = []
    for r in rows:
        body.append([r[0]] + list(r[1:]))
    t = Table([head] + body, colWidths=col_widths, repeatRows=1, rowHeights=row_height)
    style = [
        ("FONT", (0, 0), (-1, -1), "DejaVu", font_size),
        ("FONT", (0, 1), (0, -1), "DejaVu-Bold" if first_col_bold else "DejaVu", font_size),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK_2)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for i in range(2, len(body) + 1, 2):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f5f1")))
    for i in bold_rows:
        style.append(("FONT", (0, i + 1), (-1, i + 1), "DejaVu-Bold", font_size))
        style.append(("LINEABOVE", (0, i + 1), (-1, i + 1), 0.6, colors.HexColor(INK_2)))
    if color_negative:
        for i, r in enumerate(body, start=1):
            for j, v in enumerate(r[1:], start=1):
                if isinstance(v, str) and v.startswith("−"):
                    style.append(("TEXTCOLOR", (j, i), (j, i), colors.HexColor(NEGATIVE)))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(colors.HexColor(INK_2))
    canvas.drawString(1.6 * cm, 1.0 * cm, "Trend following : variantes face au S&P 500 et au Nasdaq-100 — backtest, "
                                          "pas une garantie de performance future")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


def yearly_table(total, columns, end_label):
    yearly = pd.DataFrame({c: (1 + total[c].dropna()).groupby(total[c].dropna().index.year).prod() - 1
                           for c in columns})
    last = yearly.index.max()
    yearly = yearly.rename(index={last: f"{last}*"})
    return yearly




COLORS = {**dict(zip(VARIANTS, PALETTE[:5])), SP500: "#52514e", NDX: "#a3a19b", CASH: "#c9c7c0",
          CARRY[0]: "#008300", CARRY[1]: "#4a3aa7", CARRY[2]: "#e34948"}
REF = "2. Système actuel (84 marchés)"
COLORS[REF] = PALETTE[1]

SECTIONS = {
    "CAGR (rendement total)": "Rendement",
    "Volatilité annualisée (écart-type quotidien)": "Risque",
    "Sharpe": "Ratios rendement / risque",
    "Meilleure année": "Régularité",
    "Corrélation S&P 500 (mensuelle)": "Lien avec les actions",
}


def metrics_rows(m, names):
    rows, section_rows = [], []
    for key in [k for k in m[names[0]] if not k.startswith("_")]:
        if key in SECTIONS:
            section_rows.append(len(rows))
            rows.append([SECTIONS[key]] + [""] * len(names))
        if key == "Meilleure année" or key == "Pire année":
            yk = "_best_year" if key == "Meilleure année" else "_worst_year"
            rows.append([key] + [f"{pct(m[n][key])}\n({m[n][yk]})" for n in names])
        else:
            rows.append([key] + [fmt_metric(key, m[n][key]) for n in names])
        if key == "Pire baisse (max drawdown)":
            dd = {n: m[n]["_dd"] for n in names}
            rows.append(["   début (sommet)"] + ["–" if dd[n] is None else month_year(dd[n]["peak"])
                                                    for n in names])
            rows.append(["   point bas"] + ["–" if dd[n] is None else month_year(dd[n]["trough"]) for n in names])
            rows.append(["   retour au sommet"] + [
                "–" if dd[n] is None else (month_year(dd[n]["recovery"]) if dd[n]["recovery"] is not None
                                           else "pas encore") for n in names])
    return rows, section_rows


def header(name):
    return escape(name).replace("-", "\u2011")  # trait d'union insécable : « Nasdaq-100 » sur une ligne


def metrics_table(m, names, st, label_width=4.4 * cm):
    rows, section_rows = metrics_rows(m, names)
    rows = [[r[0] if i in section_rows else Paragraph(escape(r[0]).replace("   ", "&nbsp;&nbsp;&nbsp;"),
                                                       st["label"])] + r[1:] for i, r in enumerate(rows)]
    width = (18 * cm - label_width) / len(names)
    head = [Paragraph("", st["cell"])] + [Paragraph(header(n), st["cell"]) for n in names]
    t = Table([head] + rows, colWidths=[label_width] + [width] * len(names), repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, -1), "DejaVu", 7.2),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK_2)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in section_rows:
        r = i + 1
        style += [("SPAN", (0, r), (-1, r)), ("FONT", (0, r), (-1, r), "DejaVu-Bold", 7.6),
                  ("TOPPADDING", (0, r), (-1, r), 5), ("LINEBELOW", (0, r), (-1, r), 0.5, colors.HexColor(GRID))]
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r[1:], start=1):
            if isinstance(v, str) and v.startswith("−"):
                style.append(("TEXTCOLOR", (j, i), (j, i), colors.HexColor(NEGATIVE)))
    style += [("LEFTPADDING", (1, 0), (-1, -1), 2), ("RIGHTPADDING", (1, 0), (-1, -1), 2)]
    t.setStyle(TableStyle(style))
    return t


def yearly_block(yearly, total, m, names, st, first_width=1.6 * cm):
    """Tableau année par année + lignes de synthèse."""
    full = yearly.loc[[i for i in yearly.index if not str(i).endswith("*")]]
    rows = [[str(y)] + [pct(yearly.loc[y, n]) for n in names] for y in yearly.index]
    summary_start = len(rows)
    rows.append(["CAGR"] + [pct(m[n]["CAGR (rendement total)"]) for n in names])
    rows.append(["Moyenne des années"] + [pct(full[n].mean()) for n in names])
    rows.append(["Écart-type des années"] + [pct(full[n].std()) for n in names])
    rows.append(["Années positives"] + [f"{int((full[n] > 0).sum())}/{int(full[n].notna().sum())}"
                                        for n in names])
    width = (18 * cm - first_width - 1.2 * cm) / len(names)
    head = [Paragraph("Année", st["cell"])] + [Paragraph(header(n), st["cell"]) for n in names]
    t = Table([head] + rows, colWidths=[first_width + 1.2 * cm] + [width] * len(names), repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, -1), "DejaVu", 7.2),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK_2)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.15), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.15),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, summary_start + 1), (-1, summary_start + 1), 0.8, colors.HexColor(INK_2)),
        ("FONT", (0, summary_start + 1), (-1, -1), "DejaVu-Bold", 7.2),
    ]
    for i in range(2, summary_start + 1, 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f5f1")))
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r[1:], start=1):
            if isinstance(v, str) and v.startswith("−"):
                style.append(("TEXTCOLOR", (j, i), (j, i), colors.HexColor(NEGATIVE)))
    t.setStyle(TableStyle(style))
    return t


def drawdown_table(total, names, st):
    rows = []
    for n in names:
        d = drawdown_info(total[n])
        rec = d["recovery"]
        length = ((rec if rec is not None else total[n].dropna().index[-1]) - d["peak"]).days / 365.25
        rows.append([n, pct(d["max_dd"]), month_year(d["peak"]), month_year(d["trough"]),
                     month_year(rec) if rec is not None else "pas encore",
                     f"{fr(length)} ans" + ("" if rec is not None else " +"),
                     f"{fr(d['longest_days'] / 365.25)} ans", pct(d["dd"].iloc[-1])])
    header = ["Série", "Pire baisse", "Sommet", "Point bas", "Retour au sommet", "Durée sommet → retour",
              "Plus longue période sous un sommet", "Baisse au 10 juil. 2026"]
    widths = [3.6 * cm, 1.75 * cm, 1.95 * cm, 1.95 * cm, 2.1 * cm, 2.15 * cm, 2.5 * cm, 2.0 * cm]
    t = Table([[Paragraph(escape(h), st["cell"]) for h in header]] + rows, colWidths=widths)
    style = [
        ("FONT", (0, 0), (-1, -1), "DejaVu", 7.2),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK_2)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(2, len(rows) + 1, 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f5f1")))
    for i in range(1, len(rows) + 1):
        style.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(NEGATIVE)))
    t.setStyle(TableStyle(style))
    return t


def write_xlsx(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, (df, percent) in sheets.items():
            df.to_excel(xl, sheet_name=name)
            ws = xl.sheets[name]
            ws.column_dimensions["A"].width = 44 if not percent else 12
            ratios = ("Sharpe", "Sortino", "Calmar", "Skew", "Corrélation")
            for col in ws.iter_cols(min_col=2, max_col=ws.max_column):
                ws.column_dimensions[col[0].column_letter].width = 16
                for cell in col[1:]:
                    if isinstance(cell.value, float):
                        label = str(ws.cell(row=cell.row, column=1).value)
                        cell.number_format = "0.00" if not percent and label.startswith(ratios) else "0.0%"
            ws.freeze_panes = "B2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--carry", default=os.path.join(HERE, "experiences", "carry_rendements_quotidiens.csv"))
    args = ap.parse_args()
    out_pdf, out_res, out_fig = (os.path.join(HERE, d) for d in ("rapport", "resultats", "graphiques"))
    os.makedirs(out_pdf, exist_ok=True)

    # ------------------------------------------------ les cinq variantes, 1990 → juillet 2026
    excess = load_series(args.data)
    bench = load_benchmarks(args.data, excess.index)
    cash = bench[CASH]
    total = excess.add(cash, axis=0)
    total[SP500], total[NDX], total[CASH] = bench[SP500], bench[NDX], cash
    excess[SP500], excess[NDX], excess[CASH] = bench[SP500] - cash, bench[NDX] - cash, 0.0
    series = VARIANTS + [SP500, NDX]

    m = {n: metrics(total[n], excess[n], cash, excess[SP500], excess[NDX], is_cash=(n == CASH))
         for n in series + [CASH]}
    keys = [k for k in m[VARIANTS[0]] if not k.startswith("_")]
    table = pd.DataFrame({n: {k: m[n][k] for k in keys} for n in series + [CASH]})
    table.to_csv(f"{out_res}/rapport_indicateurs.csv", float_format="%.4f")
    sub = pd.DataFrame({f"{a}-{b}": {n: cagr(total[n].loc[a:b]) for n in series} for a, b in SUBPERIODS})
    sub.to_csv(f"{out_res}/rapport_cagr_periodes.csv", float_format="%.4f")
    yearly = yearly_table(total, series + [CASH], None)
    yearly.to_csv(f"{out_res}/rapport_annees.csv", float_format="%.4f")

    # ------------------------------------------------ carry : données de Carver, 1990 → mars 2024
    carry_ex = pd.read_csv(args.carry, index_col=0, parse_dates=True).loc[rb.START:]
    carry_ex.columns = CARRY
    cidx = carry_ex.index
    cbench = load_benchmarks(args.data, cidx)
    ccash = cbench[CASH]
    ctotal = carry_ex.add(ccash, axis=0)
    cex = carry_ex.copy()
    ctotal[REF], cex[REF] = total[VARIANTS[1]].reindex(cidx), excess[VARIANTS[1]].reindex(cidx)
    ctotal[SP500], ctotal[NDX], ctotal[CASH] = cbench[SP500], cbench[NDX], ccash
    cex[SP500], cex[NDX], cex[CASH] = cbench[SP500] - ccash, cbench[NDX] - ccash, 0.0
    cseries = CARRY + [REF, SP500, NDX]
    cm_ = {n: metrics(ctotal[n], cex[n], ccash, cex[SP500], cex[NDX]) for n in cseries}
    ctable = pd.DataFrame({n: {k: cm_[n][k] for k in keys} for n in cseries})
    ctable.to_csv(f"{out_res}/rapport_carry_indicateurs.csv", float_format="%.4f")
    cyearly = yearly_table(ctotal, cseries + [CASH], None)
    cyearly.to_csv(f"{out_res}/rapport_carry_annees.csv", float_format="%.4f")

    write_xlsx(f"{out_pdf}/performances_annuelles.xlsx", {
        "Années 1990-2026": (yearly, True),
        "Indicateurs 1990-2026": (table, False),
        "CAGR par période": (sub, True),
        "Carry années 1990-2024": (cyearly, True),
        "Carry indicateurs": (ctable, False),
    })

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(table.round(3).to_string())
    print(sub.round(3).to_string())
    print(ctable.round(3).to_string())
    if os.environ.get("RAPPORT_CALCUL_SEUL"):
        return

    # ------------------------------------------------ graphiques
    growth_chart(total[series], COLORS, "Croissance de 1 €, janvier 1990 → juillet 2026",
                 "Rendement total (monétaire inclus), coûts de transaction et de roulement inclus, sans frais de "
                 "gestion. Pointillés : indices actions.", f"{out_fig}/rapport_croissance.png",
                 [1, 10, 100, 1000, 10000])
    cagr_chart(sub, COLORS, f"{out_fig}/rapport_cagr.png")
    drawdown_chart(total[series], COLORS, f"{out_fig}/rapport_baisses.png")
    growth_chart(ctotal[cseries], COLORS, "Ajouter le carry : croissance de 1 €, janvier 1990 → mars 2024",
                 "Trend seul, trend + carry et trend + carry + stratégie 13 sur les données de Carver (71 marchés) ; "
                 "système actuel sur les 84 marchés.", f"{out_fig}/rapport_carry.png", [1, 10, 100, 1000],
                 height=3.2)

    build_pdf(f"{out_pdf}/rapport_variantes_trend.pdf", m, sub, yearly, total, series, cm_, cyearly, ctotal,
              cseries, out_fig)
    print("PDF écrit :", f"{out_pdf}/rapport_variantes_trend.pdf")


def build_pdf(path, m, sub, yearly, total, series, cm_, cyearly, ctotal, cseries, out_fig):
    st = styles()
    C = "CAGR (rendement total)"
    V = "Volatilité annualisée (écart-type quotidien)"
    DD = "Pire baisse (max drawdown)"
    v1, v2, v3, v4, v5 = VARIANTS
    p = lambda x: pct(x)  # noqa: E731
    pts = lambda x: f"{x * 100:+.1f}".replace(".", ",").replace("-", "−") + " pt"  # noqa: E731
    since10 = sub["2010-2026"]
    corr = [m[n]["Corrélation S&P 500 (mensuelle)"] for n in VARIANTS]
    dd4, dd5, dd1 = m[v4]["_dd"], m[v5]["_dd"], m[v1]["_dd"]
    sr2, vol2 = m[v2]["Sharpe"], m[v2][V]
    real_lo, real_hi = (s * vol2 - vol2 ** 2 / 2 for s in (0.3, 0.5))
    still_under = [n.split(". ")[0] for n in VARIANTS if m[n]["_dd"]["recovery"] is None]

    story = []
    story.append(Paragraph("Trend following : cinq variantes face au S&amp;P 500 et au Nasdaq-100", st["title"]))
    story.append(Paragraph(
        "Backtest du 1<super>er</super> janvier 1990 au 10 juillet 2026 · 84 marchés à terme · coûts de "
        "transaction et de roulement inclus, sans frais de gestion ni impôts · rendements totaux en dollars "
        "(intérêts du monétaire inclus), indices actions dividendes réinvestis.", st["subtitle"]))

    story.append(Paragraph("À retenir", st["h1"]))
    bullets = [
        f"<b>Sur toute la période</b>, le système actuel (variante 2) a rapporté <b>{p(m[v2][C])} par an</b> "
        f"contre {p(m[SP500][C])} pour le S&amp;P 500 et {p(m[NDX][C])} pour le Nasdaq-100, avec une volatilité "
        f"proche de celle du S&amp;P 500 ({p(vol2)} contre {p(m[SP500][V])}) et une pire baisse plus faible "
        f"({p(m[v2][DD])} contre {p(m[SP500][DD])} et {p(m[NDX][DD])}).",
        f"<b>L'avance vient surtout des années 1990 et de 2008.</b> Depuis 2010, toutes les variantes font moins "
        f"bien que les deux indices : de {p(since10[VARIANTS].min())} à {p(since10[VARIANTS].max())} par an, "
        f"contre {p(since10[SP500])} pour le S&amp;P 500 et {p(since10[NDX])} pour le Nasdaq-100. "
        f"2023 et 2024 sont négatives pour les cinq variantes.",
        f"<b>Chaque amélioration ajoute du rendement</b> par rapport à la variante 2 : stratégie 13 "
        f"{pts(m[v3][C] - m[v2][C])} par an, pilotage du risque {pts(m[v4][C] - m[v2][C])}, les deux "
        f"{pts(m[v5][C] - m[v2][C])}. À risque égal (même volatilité que le S&amp;P 500), l'écart se réduit : "
        f"{p(m[v2]['CAGR au même risque que le S&P 500'])} pour la variante 2, "
        f"{p(m[v5]['CAGR au même risque que le S&P 500'])} pour la variante 5. Le pilotage du risque augmente aussi "
        f"la volatilité ({p(m[v4][V])}) et la pire baisse ({p(m[v4][DD])}) : les variantes 4 et 5 sont toujours "
        f"sous leur sommet de {month_year(dd4['peak'])}." if dd4["recovery"] is None and dd5["recovery"] is None
        else "",
        f"<b>Sans ajuster la taille à la volatilité</b> (variante 1), c'est nettement moins bon : {p(m[v1][C])} par "
        f"an, une pire baisse de {p(m[v1][DD])}, et toujours sous son sommet de {month_year(dd1['peak'])} "
        f"en juillet 2026." if dd1["recovery"] is None else "",
        f"<b>Le trend est décorrélé des actions</b> (corrélation mensuelle avec le S&amp;P 500 de "
        f"{num(min(corr))} à {num(max(corr))}) et a gagné les années où les actions chutaient : 2000, 2001, 2002, "
        f"2008, 2022. C'est son principal intérêt dans un portefeuille QQQ / or / bitcoin.",
        f"<b>Ce sont des résultats de backtest</b>, optimistes (voir la dernière page). Pour l'avenir, un Sharpe de "
        f"0,3 à 0,5 est une hypothèse plus prudente que le {num(sr2)} mesuré ici, soit, au niveau de risque de la "
        f"variante 2, environ {fr(real_lo * 100, 0)} à {fr(real_hi * 100, 0)} % par an au-dessus du monétaire.",
    ]
    for b in bullets:
        if b:
            story.append(Paragraph(b, st["bullet"], bulletText="•"))

    story.append(Paragraph("Les cinq variantes", st["h1"]))
    desc = [
        (v1, "Témoin. La taille de chaque position est calculée une seule fois, sur la volatilité des trois "
             "premières années du marché, puis n'est plus jamais ajustée."),
        (v2, "Stratégie 9 de Carver (Advanced Futures Trading Strategies) : quatre croisements de moyennes mobiles "
             "exponentielles (8/32 à 64/256 jours), taille inversement proportionnelle à la volatilité récente de "
             "chaque marché, risque visé 20 % par an, poids égal par classe d'actifs."),
        (v3, "Variante 2, mais la prévision d'un marché est réduite quand sa volatilité est haute par rapport à son "
             "propre passé (×0,5 au plus agité, ×2 au plus calme)."),
        (v4, "Variante 2, mais toutes les positions sont multipliées chaque jour par 20 % / volatilité récente du "
             "portefeuille entier (multiplicateur borné entre 0,5 et 2)."),
        (v5, "Variantes 3 et 4 combinées."),
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", st["body"]), Paragraph(b, st["body"])] for a, b in desc],
              colWidths=[3.9 * cm, 14.1 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(GRID))]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Les paramètres sont ceux publiés par Carver, pas optimisés sur ces données. Le pilotage du risque utilise "
        "seulement des informations connues la veille.", st["small"]))

    # ------------------------------------------------ indicateurs
    story.append(PageBreak())
    story.append(Paragraph("Indicateurs, janvier 1990 → juillet 2026", st["h1"]))
    story.append(metrics_table(m, series + [CASH], st))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Comment lire ces indicateurs", st["h2"]))
    gloss = [
        ("CAGR", "taux de croissance annuel composé : le rendement constant qui mène du capital de départ au capital "
                 "final."),
        ("CAGR au même risque que le S&amp;P 500", "la série est ramenée, avec plus ou moins de levier, à la "
         f"volatilité du S&amp;P 500 sur la période ({p(m[SP500][V])}) : compare les rendements à risque égal."),
        ("Volatilité", "écart-type des rendements quotidiens, annualisé (× 16, soit √256 jours)."),
        ("Écart-type des performances annuelles", "dispersion des rendements des années civiles complètes "
         "(1990–2025). Il dépasse la volatilité pour le trend parce que les tendances durent et que les très "
         "bonnes années se composent."),
        ("Sharpe", "rendement moyen au-dessus du monétaire divisé par sa volatilité, annualisé."),
        ("Sortino", "comme le Sharpe, mais divisé par l'écart-type des seules baisses : les fortes hausses ne sont "
                    "pas pénalisées."),
        ("Calmar", "CAGR divisé par la pire baisse."),
        ("Pire baisse", "plus forte chute du capital entre un sommet et le point bas qui suit."),
        ("Skew", "asymétrie des rendements mensuels ; positif = les grosses surprises sont plutôt à la hausse."),
        ("Corrélation", "entre −1 (sens opposé) et +1 (même sens), sur les rendements mensuels au-dessus du "
                        "monétaire."),
    ]
    for a, b in gloss:
        story.append(Paragraph(f"<b>{a}</b> : {b}", st["small"]))

    # ------------------------------------------------ graphiques CAGR et croissance
    story.append(PageBreak())
    story.append(Paragraph("CAGR des variantes face au S&amp;P 500 et au Nasdaq-100", st["h1"]))
    story.append(Image(f"{out_fig}/rapport_cagr.png", width=18 * cm, height=18 * cm * 3.6 / 7.6))
    story.append(Spacer(1, 4))
    story.append(Image(f"{out_fig}/rapport_croissance.png", width=18 * cm, height=18 * cm * 4.7 / 7.6))
    story.append(Paragraph(
        "Échelle logarithmique : une même pente correspond à un même rendement en pourcentage. Les écarts finaux "
        "très importants viennent de la composition sur 36 ans de rendements de backtest élevés, surtout avant 2010.",
        st["small"]))

    # ------------------------------------------------ baisses
    story.append(PageBreak())
    story.append(Paragraph("Les baisses", st["h1"]))
    story.append(Image(f"{out_fig}/rapport_baisses.png", width=18 * cm, height=18 * cm * (1.45 * 4 + 0.45) / 7.6))
    story.append(Spacer(1, 6))
    story.append(drawdown_table(total, series, st))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "« + » : baisse pas encore effacée au 10 juillet 2026. La plus longue période sous un sommet compte toutes "
        "les baisses, pas seulement la pire.", st["small"]))

    # ------------------------------------------------ années
    story.append(PageBreak())
    story.append(Paragraph("Performances année par année (rendement total)", st["h1"]))
    story.append(yearly_block(yearly, total, m, series + [CASH], st))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "* 2026 : du 1<super>er</super> janvier au 10 juillet, non annualisé. Moyenne, écart-type et années "
        "positives portent sur les années complètes 1990–2025. Nasdaq-100 : indice de prix (sans dividendes) "
        "avant mars 1999, puis l'ETF QQQ dividendes réinvestis. Fichier Excel joint avec les mêmes chiffres.",
        st["small"]))

    # ------------------------------------------------ carry
    story.append(PageBreak())
    T, TC, TC13 = CARRY
    story.append(Paragraph("Ajouter le carry (données de Carver, janvier 1990 → mars 2024)", st["h1"]))
    story.append(Paragraph(
        "Le carry demande le prix du contrat suivant, absent des 84 marchés : il a été testé sur les données "
        "publiées par Carver (71 marchés), qui s'arrêtent le 28 mars 2024. « Trend seul » est le même système que "
        "la variante 2 appliqué à ces données ; la comparaison avec « trend + carry » isole donc l'apport du carry. "
        "Trend + carry = stratégie 11 de Carver (60 % trend, 40 % carry) ; « + strat. 13 » ajoute le réglage de la "
        "variante 3 sur la partie trend. Pas de pilotage du risque du portefeuille ici. La variante 2 sur 84 marchés "
        "est montrée sur la même période pour situer ces chiffres.", st["body"]))
    story.append(Paragraph(
        f"Sur ces données, le carry fait passer le CAGR de {p(cm_[T][C])} à {p(cm_[TC][C])} "
        f"({p(cm_[TC13][C])} avec la stratégie 13), le Sharpe de {num(cm_[T]['Sharpe'])} à "
        f"{num(cm_[TC]['Sharpe'])} ({num(cm_[TC13]['Sharpe'])}) et la pire baisse de {p(cm_[T][DD])} à "
        f"{p(cm_[TC][DD])} ({p(cm_[TC13][DD])}). Prudence : Carver a choisi le carry en connaissant une bonne partie "
        "de cette histoire, le gain est donc en partie mesuré « dans l'échantillon ».", st["body"]))
    story.append(metrics_table(cm_, cseries, st))
    story.append(Spacer(1, 6))
    story.append(Image(f"{out_fig}/rapport_carry.png", width=18 * cm, height=18 * cm * 3.2 / 7.6))
    story.append(PageBreak())
    story.append(Paragraph("Carry : performances année par année (rendement total)", st["h1"]))
    story.append(yearly_block(cyearly, ctotal, cm_, cseries, st))
    story.append(Paragraph("* 2024 : du 1<super>er</super> janvier au 28 mars, non annualisé.", st["small"]))

    # ------------------------------------------------ méthode
    story.append(PageBreak())
    story.append(Paragraph("Méthode et limites", st["h1"]))
    method = [
        "<b>Données.</b> 84 contrats à terme (actions, obligations, taux courts, devises, énergie, métaux, "
        "agriculture), rendements quotidiens en dollars des contrats roulés. Avant la date de cotation réelle d'un "
        "contrat, l'historique reconstitué est ignoré. Taux courts et obligations forment une seule classe "
        "« taux » pour les poids.",
        "<b>Rendement total.</b> Un contrat à terme ne rapporte que l'écart au monétaire ; le capital qui sert de "
        "marge reste placé. On ajoute donc chaque jour le taux des T-bills américains à 3 mois (FRED DTB3), "
        f"{p(m[CASH][C])} par an en moyenne sur la période. Les indices actions sont dividendes réinvestis "
        "(S&amp;P 500 total return ; Nasdaq-100 : indice de prix avant le 10 mars 1999, ce qui sous-estime "
        "légèrement ces années-là, puis l'ETF QQQ). Tout est en dollars : pour un investisseur en euros, il "
        "faudrait ajouter l'effet du change ou le coût de couverture.",
        "<b>Coûts.</b> 1 point de base (0,5 pour les taux courts) par unité de notionnel échangée, plus le coût de "
        "roulement des contrats (4 à 12 fois par an selon la classe). Pas de frais de gestion : un fonds de trend "
        "prend souvent 1 à 2 % par an, plus parfois 20 % des gains.",
        "<b>Calculs.</b> Calendrier des jours ouvrés (environ 261 par an), jours fériés américains comptés à "
        "rendement nul pour les indices. CAGR en temps calendaire. Volatilité, Sharpe et Sortino annualisés × 16. "
        "Sharpe et Sortino sur le rendement au-dessus du monétaire (le Sortino divise par la racine de la moyenne "
        "des rendements négatifs au carré). Pire baisse, CAGR et années sur le rendement total ; les pires baisses "
        "citées plus tôt dans la conversation étaient mesurées hors monétaire, d'où quelques points d'écart. Les "
        "rendements annuels moyens cités plus tôt comptaient 252 jours par an au lieu d'environ 261, ce qui les "
        "sous-estimait d'environ 0,5 point ; c'est corrigé ici et dans les fichiers de résultats.",
        "<b>Levier.</b> Pour viser 20 % de risque avec des marchés peu volatils (obligations, taux courts), "
        "l'exposition totale atteint en moyenne environ 5 à 6 fois le capital. Ce n'est réalisable qu'avec des "
        "contrats à terme, et il faut un capital important pour détenir 84 marchés avec des contrats entiers.",
        "<b>Pourquoi ces chiffres sont optimistes.</b> Les règles de Carver ont été publiées en 2022 par quelqu'un "
        "qui connaissait cette histoire ; le choix même du trend following repose sur son succès passé. Les "
        "variantes 3 à 5 et le carry ont été ajoutés après avoir vu les résultats du système de base. Les années "
        "1990 et 2008 pèsent très lourd dans le total. Les vrais fonds de trend (indice SG Trend) ont fait "
        "nettement moins bien depuis 2000, frais déduits. D'où l'hypothèse d'un Sharpe futur de 0,3 à 0,5.",
        "<b>Reproduire.</b> <font face='DejaVu'>python rapport_pdf.py --data DOSSIER_MKT</font> dans "
        "analyse-trend/ ; chiffres détaillés dans resultats/rapport_*.csv.",
    ]
    for b in method:
        story.append(Paragraph(b, st["body"]))

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.6 * cm, title="Trend following : variantes face au S&P 500 et au Nasdaq-100",
                            author="Analyse trend")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
