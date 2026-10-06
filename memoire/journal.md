# Journal des conversations (mémoire du projet)

Résumé de ce qui a été fait et décidé avec l'utilisateur, du plus ancien au plus récent. À compléter à la fin
de chaque séance de travail importante (voir la règle dans CLAUDE.md).

## Séance d'octobre 2026 (session cloud, branche claude/quirky-dijkstra-hphu32)

### 1. Cryptos (dossier analyse-correlation-crypto/)
- Matrice de corrélation des cryptos du top 10–50 face au BTC, période commune, puis corrélations glissantes
  hebdomadaires (26 et 52 semaines) face au BTC.

### 2. Diversification (dossier analyse-diversification/)
- Recherche d'actifs décorrélés de QQQ, GLD et BTC, structurellement haussiers ou très volatils.
- Véhicules UCITS vérifiés pour un investisseur français : resultats/sources.csv.
- Explications données sur l'indice SG Trend, DBMF, KRBN, PDBC et URNM.

### 3. Trend following à la Carver (dossier analyse-trend/)
- Moteur trend.py : stratégie 9 de Carver (EWMAC 8/32 à 64/256, FDM 1,13, IDM, risque visé 20 %, tampon 10 %),
  stratégie 13 (régime de volatilité), pilotage du risque du portefeuille, variante à montant fixe.
- Données : 84 contrats à terme (Artur Sepp), voir analyse-trend/donnees/.
- Les 5 variantes, backtest 1990 → 10/07/2026, rendement total avec T-bill américain :

  | Variante | CAGR | Sharpe | Vol | Pire baisse | Exposition |
  |---|---|---|---|---|---|
  | 1. Montant fixe | 13,0 % | 0,50 | 24,6 % | −66,9 % | 5,3× |
  | 2. Système actuel | 19,3 % | 0,92 | 17,6 % | −40,2 % | 5,6× |
  | 3. + stratégie 13 | 22,3 % | 1,03 | 18,3 % | −41,4 % | 6,8× |
  | 4. + pilotage du risque | 28,1 % | 1,13 | 21,2 % | −48,4 % | 7,8× |
  | 5. Les deux (13 + pilotage) | 29,6 % | 1,17 | 21,4 % | −47,8 % | 8,8× |

- « Les deux » = trend + stratégie 13 + pilotage du risque. Pas de carry.
- Carry (stratégie 11, données de Carver, 71 marchés, 1990 → 03/2024) : Sharpe trend seul 0,72 → trend + carry
  1,00 → + stratégie 13 1,08 ; pire baisse −35 % → −27 % → −26 %. En partie dans l'échantillon.
- Ce qui n'a pas marché : réduire ou augmenter la taille selon les performances passées.
- Rapport PDF des variantes : analyse-trend/rapport/rapport_variantes_trend.pdf (+ Excel des années).
- Brent seul (règles de la variante 2) : Sharpe 0,43 depuis 1990, 0,34 depuis 2010, −0,64 depuis 2023
  (marché en allers-retours, 22 changements de signe en 3,5 ans). Acheter-garder le Brent : Sharpe 0,37.

### 4. Audit de faisabilité réelle (dossier analyse-trend/audit/)
- Rapport sans fiscalité, centré sur IBKR + IB Gateway (version que l'utilisateur veut) :
  audit/rapport_audit_variante5_ib_gateway.pdf (générateur rapport_audit_ib_gateway.py).
  L'ancienne version avec fiscalité : audit/rapport_audit_variante5.pdf (ne plus l'utiliser).
- Conclusions principales :
  - réplique fidèle des 84 marchés : environ 5 M$ ; en dessous, sous-portefeuilles de 14 (100 k$),
    24 (250 k$), 40 (1 M$) marchés ;
  - l'éthanol (CUA1) est un artefact de données ; le minerai de fer SGX (SCO1) n'est pas accessible ;
    marchés négociables + exécution réaliste : Sharpe de la variante 5 = 0,38 depuis 2010 (au lieu de 0,54) ;
  - coûts via IB Gateway : API gratuite (compte IBKR Pro), ~400 $/an de frais fixes, coût total 4,5 % du capital
    par an à 100–250 k$, 3,5 % à 1 M$ ; validation 2FA hebdomadaire sur IBKR Mobile ;
  - rendement médian attendu sur 10 ans (avant impôt), variante 5 : 3,3 % (100 k$), 4,6 % (250 k$),
    5,4 % (1 M$) ; ETF trend UCITS (iMGP DBi couvert €, MFEH, LU3359622902) : 4,7 % ; monétaire 2,3 % ;
  - 70 % Nasdaq-100 + 30 % ETF trend : 6,4 % contre 5,9 % pour 100 % Nasdaq, probabilité de perte sur 10 ans
    12 % contre 24 % ;
  - recommandation : sous 1 M$, un ETF de trend ; à partir de 1 M$, la variante 5 via IB Gateway est défendable.
- Scripts, résultats et rapports complets des agents et vérificateurs : audit/travail/.

### 5. Rendements nets de frais par variante (en cours au moment de la sauvegarde)
- Calculés et vérifiés (audit/travail/var_passe_global, var_passe_retail, verif_*) :
  - passé net réaliste, 84 marchés négociables (sans CUA1/SCO1, exécution à une demi-journée, coûts IBKR
    réels, trésorerie IBKR), CAGR total depuis 2010 : v1 −1,8 %, v2 4,5 %, v3 5,7 %, v4 7,1 %, v5 7,5 % ;
  - passé à taille de particulier (contrats entiers, frais IB Gateway), CAGR 2010–2026 à 100 k / 250 k / 1 M :
    v1 −2,8 / −4,2 / −2,5 %, v2 3,6 / 4,6 / 4,3 %, v3 4,4 / 6,4 / 5,6 %, v4 6,9 / 7,2 / 6,9 %,
    v5 7,2 / 7,9 / 7,1 %.
- Attendu sur 10 ans (audit/travail/var_futur, PAS ENCORE VÉRIFIÉ : le vérificateur a échoué sur une limite
  d'usage), médiane à 100 k / 250 k / 1 M : v1 −5,2 / −6,1 / −4,0 %, v2 2,5 / 3,6 / 3,9 %,
  v3 1,9 / 3,9 / 4,4 %, v4 3,3 / 4,2 / 5,3 %, v5 3,3 / 4,6 / 5,4 %.
- À FAIRE : vérifier var_futur, puis ajouter une section « Rendements nets par variante » (passé + attendu)
  au rapport IB Gateway et le régénérer.

### 6. Pistes d'amélioration proposées (non encore testées)
- Protocole : liste fixée à l'avance, décision sur 1990–2009, jugement hors échantillon sur 2010–2026,
  sur le portefeuille réellement tenable avec coûts IBKR ; refuser les gains non significatifs (t < 2).
- Pistes : carry sur les 84 marchés (puis trend + carry + 13), signaux de cassure (breakout), autres familles
  de signaux de Carver (stratégies 15–20), optimisation dynamique (stratégie 25) pour petits comptes,
  pondération par corrélations. L'utilisateur n'a pas encore dit lequel lancer.
