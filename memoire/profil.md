# Profil et conventions (chargé seulement avec /reprendre)

Ce dépôt est le dossier de travail d'un investisseur particulier. Ce fichier et memoire/journal.md servent de
mémoire entre les conversations : ils résument ce qui a déjà été fait, décidé et demandé.


## Qui est l'utilisateur et comment lui répondre
- Francophone : répondre en français, de façon pédagogique, avec des chiffres vérifiés.
- Détient QQQ, de l'or (GLD) et du bitcoin. Cherche à diversifier (trend following, actifs décorrélés).
- Lit Robert Carver (Systematic Trading, Leveraged Trading, Advanced Futures Trading Strategies, Smart Portfolios).
- Ne veut PAS qu'on lui parle d'impôts ou de fiscalité : rendements avant impôt, aucune section fiscale.
- S'il exécute un système lui-même, ce sera chez Interactive Brokers via IB Gateway (API) : bâtir les coûts
  et la faisabilité sur cette hypothèse.
- Aime les livrables en PDF (reportlab, police DejaVu), avec tableaux et graphiques, et les explications
  « par variante ». Veut des résultats vérifiés de façon indépendante avant d'être présentés.

## Organisation du dépôt
- analyse-correlation-crypto/ : corrélations des cryptos face au BTC.
- analyse-diversification/ : actifs décorrélés de QQQ/GLD/BTC, véhicules UCITS vérifiés.
- analyse-trend/ : système de trend following à la Carver.
  - trend.py (moteur), run_backtests.py, listings.py, rapport_pdf.py (rapport des variantes),
    ameliorations.py, sous_ensembles.py, resultats/, graphiques/, rapport/.
  - audit/ : audit de faisabilité réelle ; rapport à jour = audit/rapport_audit_variante5_ib_gateway.pdf
    (générateur rapport_audit_ib_gateway.py) ; travail/ contient tous les scripts, résultats et rapports
    d'agents.
  - donnees/ : copies compressées des données de marché. Pour les utiliser :
    `python analyse-trend/donnees/installer.py /tmp/mkt` puis `python run_backtests.py --data /tmp/mkt ...`
    (attention : certains scripts d'audit dans audit/travail/ ont des chemins absolus vers l'ancien
    dossier temporaire ; remplacer ce chemin par /tmp/mkt).

## Conventions de calcul
- Sharpe annualisé ×16 (convention de Carver), CAGR en temps calendaire (jours / 365,25).
- Les futures donnent un rendement au-dessus du monétaire ; rendement total = excès + T-bill (FRED DTB3).
- Pour un investisseur en euros, référence monétaire = €STR (≈ 2,4 % en octobre 2026).
- Exclure l'éthanol (CUA1, artefact de données) et le minerai de fer SGX (SCO1, inaccessible) dès qu'on parle
  de résultats réalistes.

## Mise à jour de la mémoire
Seulement quand l'utilisateur lance /memoriser : ajouter un paragraphe daté
dans memoire/journal.md (ce qui a été fait, les chiffres clés, où sont les fichiers, ce qui reste à faire),
puis committer et pousser.
