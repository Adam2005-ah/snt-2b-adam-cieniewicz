"""Décompresse les données de marché du projet dans un dossier utilisable par les scripts.

Usage : python analyse-trend/donnees/installer.py [DOSSIER]   (défaut : /tmp/mkt)
Puis : python run_backtests.py --data DOSSIER ...

Sources d'origine (copies figées au 2026-10-04) :
- futures 84 marchés : github.com/ArturSepp/TrendFollowingSystems (tf_system_data_prices.csv et rendements USD)
- prix d'ETF et indices (QQQ, GLD, PDBC, ^SP500TR, ^NDX) et taux T-bill DTB3 : github.com/alm0421/alm0421
- indices de fonds (SG Trend, SG CTA, réplique DBi, BTOP50, AQR TSMOM) : github.com/bpineau/pofo
"""

import gzip
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mkt"
    src = os.path.join(HERE, "mkt")
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".gz"):
                continue
            out = os.path.join(dest, os.path.relpath(root, src), f[:-3])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with gzip.open(os.path.join(root, f), "rb") as i, open(out, "wb") as o:
                shutil.copyfileobj(i, o)
            n += 1
    print(f"{n} fichiers installés dans {dest}")


if __name__ == "__main__":
    main()
