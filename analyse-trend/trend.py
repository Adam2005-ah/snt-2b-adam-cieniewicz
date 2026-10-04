"""Système de trend following à la Carver (Advanced Futures Trading Strategies, stratégie neuf).

Étapes, pour chaque marché :
1. Volatilité : écart-type pondéré exponentiellement des rendements quotidiens (span 32 jours),
   mélangé à 30 % avec sa moyenne sur 10 ans (stratégie trois).
2. Signaux : croisements de moyennes mobiles exponentielles EWMAC(n, 4n) pour n = 8, 16, 32, 64,
   divisés par la volatilité en prix, multipliés par les « forecast scalars » de Carver (table 29)
   et plafonnés à ±20.
3. Prévision combinée : moyenne des quatre signaux × FDM 1.13 (table 36), plafonnée à ±20.
4. Position (exposition en fraction du capital) = prévision/10 × risque cible × IDM × poids / volatilité.
5. Tampon : on ne traite que si la position sort d'une zone de ±10 % de la position moyenne.
Le signal du jour t ne s'applique qu'au rendement du jour t+1 (pas de regard vers le futur).

Les paramètres sont ceux publiés par Carver : rien n'est optimisé sur ces données.
"""

import numpy as np
import pandas as pd

SPEEDS = [8, 16, 32, 64]
SCALARS = {2: 12.1, 4: 8.53, 8: 5.95, 16: 4.10, 32: 2.79, 64: 1.91}  # AFTS table 29
FDM = 1.13                    # AFTS table 36, quatre filtres EWMAC8 à 64
CAP = 20.0
VOL_SPAN = 32
VOL_LONG_DAYS = 2520          # 10 ans de jours ouvrés
VOL_FLOOR_DAYS = 500          # plancher de volatilité (pysystemtrade) : 5e centile des 500 derniers jours
VOL_FLOOR_QUANTILE = 0.05
WARMUP_DAYS = 256             # historique minimum avant de trader un marché
BUFFER = 0.10                 # AFTS stratégie huit
IDM_TABLE = [(1, 1.00), (2, 1.20), (3, 1.48), (4, 1.56), (5, 1.70), (6, 1.90), (7, 2.10),
             (8, 2.20), (15, 2.30), (25, 2.40), (30, 2.50)]  # AFTS table 16


def idm_for(n):
    value = 1.0
    for threshold, idm in IDM_TABLE:
        if n >= threshold:
            value = idm
    return value


def annual_vol(returns):
    """Volatilité annualisée de Carver : 70 % EWMA 32 jours + 30 % de sa moyenne sur 10 ans.

    Plancher au 5e centile des 500 derniers jours, sinon un marché très calme (taux courts à zéro)
    reçoit un levier démesuré."""
    daily = returns.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
    long_run = daily.rolling(VOL_LONG_DAYS, min_periods=VOL_SPAN).mean()
    blended = 0.7 * daily + 0.3 * long_run
    floor = blended.rolling(VOL_FLOOR_DAYS, min_periods=100).quantile(VOL_FLOOR_QUANTILE)
    return blended.clip(lower=floor) * 16


def forecast(returns, vol_ann):
    """Prévision combinée de trend, entre -20 et +20 (10 = conviction moyenne)."""
    price = (1 + returns.fillna(0)).cumprod().where(returns.notna().cummax())
    price_vol = price * vol_ann / 16
    signals = []
    for n in SPEEDS:
        raw = (price.ewm(span=n, min_periods=n).mean() - price.ewm(span=4 * n, min_periods=4 * n).mean()) / price_vol
        signals.append((raw * SCALARS[n]).clip(-CAP, CAP))
    return (sum(signals) / len(signals) * FDM).clip(-CAP, CAP)


def rebalance_mask(index, freq):
    """Jours où l'on recalcule les positions : chaque jour, en fin de semaine ou en fin de mois."""
    if freq == "D":
        return pd.Series(True, index=index)
    period = index.to_period("W-FRI" if freq == "W" else "M")
    last = pd.Series(index, index=index).groupby(period).transform("max")
    return pd.Series(index == last.values, index=index)


def buffered_positions(target, unit, mask):
    """Tampon de Carver : on ne bouge que si la cible sort de ±10 % de la position moyenne (unit)."""
    tgt, width, do = target.values, (BUFFER * unit).values, mask.values
    out = np.zeros(len(tgt))
    current = 0.0
    for i in range(len(tgt)):
        if np.isnan(tgt[i]):
            current = 0.0          # marché pas encore (ou plus) tradable
        elif do[i]:
            current = min(max(current, tgt[i] - width[i]), tgt[i] + width[i])
        out[i] = current
    return pd.Series(out, index=target.index)


def momentum_12m(returns, vol_ann):
    """Variante mensuelle classique (Moskowitz, Ooi, Pedersen) : signe du rendement sur 12 mois, ±10."""
    price = (1 + returns.fillna(0)).cumprod().where(returns.notna().cummax())
    return np.sign(price / price.shift(252) - 1) * 10


def run_portfolio(returns, groups, costs, roll_costs, risk_target=0.20, freq="D", forecast_fn=forecast):
    """Backtest multi-marchés. returns : rendements quotidiens (excès de rendement des futures).

    groups : classe d'actifs de chaque marché ; costs : coût par unité de notionnel traitée ;
    roll_costs : coût annuel de roulement par unité de notionnel détenue.
    """
    vol = returns.apply(annual_vol)
    fc = returns.apply(lambda s: forecast_fn(s, vol[s.name]))
    age = returns.notna().cumsum()
    live = (age > WARMUP_DAYS) & vol.notna() & fc.notna()

    # Poids : égal entre classes d'actifs présentes, puis égal entre marchés d'une même classe.
    class_count = live.T.groupby(groups).transform("sum").T
    n_classes = live.T.groupby(groups).any().T.sum(axis=1)
    weights = live / class_count.where(class_count > 0)
    weights = weights.div(n_classes, axis=0)
    idm = live.sum(axis=1).map(idm_for)

    unit = (risk_target * weights.mul(idm, axis=0) / vol).where(live)  # position pour une prévision de 10
    target = fc / 10 * unit
    mask = rebalance_mask(returns.index, freq)
    pos = pd.DataFrame({c: buffered_positions(target[c], unit[c].fillna(0), mask) for c in returns})

    held = pos.shift(1).fillna(0)
    gross = (held * returns.fillna(0)).sum(axis=1)
    trading = (pos.diff().abs().fillna(pos.abs()) * pd.Series(costs)).sum(axis=1)
    rolling = (held.abs() * pd.Series(roll_costs) / 252).sum(axis=1)
    return {"net": gross - trading - rolling, "gross": gross, "positions": pos, "forecasts": fc,
            "trading_costs": trading, "roll_costs": rolling, "live": live}


def run_single(returns, cash, freq="D", mode="long_cash", risk_target=0.20, cost=0.0005):
    """Trend sur un seul actif (ETF). returns : rendements totaux ; cash : rendement monétaire.

    mode "long_cash" : investi de 0 à 100 % selon la prévision, le reste en monétaire, sans levier.
    mode "long_short" : exposition = prévision/10 × risque cible / volatilité, à découvert possible.
    """
    excess = returns - cash
    vol = annual_vol(excess)
    fc = forecast(excess, vol)
    live = (excess.notna().cumsum() > WARMUP_DAYS) & vol.notna() & fc.notna()
    if mode == "long_cash":
        target, unit = (fc / 10).clip(0, 1), pd.Series(1.0, index=fc.index)
    else:
        unit = risk_target / vol
        target = fc / 10 * unit
    target, unit = target.where(live), unit.where(live).fillna(0)
    pos = buffered_positions(target, unit, rebalance_mask(returns.index, freq))
    held = pos.shift(1).fillna(0)
    net = held * excess.fillna(0) + cash.fillna(0) - pos.diff().abs().fillna(0) * cost
    return {"net": net.where(live.shift(1, fill_value=False)), "positions": pos, "forecasts": fc}


def stats(daily, label=None):
    """Indicateurs d'une série de rendements quotidiens."""
    d = daily.dropna()
    monthly = (1 + d).resample("ME").prod() - 1
    curve = (1 + d).cumprod()
    years = len(d) / 252
    return pd.Series({
        "début": d.index.min().date(), "fin": d.index.max().date(),
        "rendement annuel": curve.iloc[-1] ** (1 / years) - 1,
        "volatilité": d.std() * 16,
        "sharpe": d.mean() / d.std() * 16,
        "pire baisse": (curve / curve.cummax() - 1).min(),
        "skew mensuel": monthly.skew(),
        "% mois positifs": (monthly > 0).mean(),
    }, name=label)
