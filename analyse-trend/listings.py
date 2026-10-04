"""Dates approximatives de première cotation des contrats à terme (ou de leur équivalent direct).

Les séries de la base Sepp remontent parfois bien avant l'existence du contrat (indices reconstitués,
séries dérivées des taux). On n'utilise les rendements qu'à partir de ces dates. Sources : historiques
des Bourses, dates approximatives relevées pendant la vérification. Les marchés absents de la liste sont
supposés cotés dès le début de leur série (devises 1972-78, matières premières anciennes, etc.).
"""

LISTING = {
 "ES1 Index":"1982-04-21","NQ1 Index":"1996-04-10","RTY1 Index":"1993-02-04","DM1 Index":"1997-10-06","MES1 Index":"2009-01-01",
 "VG1 Index":"1998-06-22","GX1 Index":"1990-11-23","CF1 Index":"1988-12-08","EO1 Index":"1988-10-03","Z 1 Index":"1984-05-03",
 "SM1 Index":"1990-11-09","QC1 Index":"1987-01-01","NO1 Index":"1986-09-03","TP1 Index":"1988-09-03","XP1 Index":"1983-02-16",
 "PT1 Index":"1999-09-07","TWT1 Index":"1997-01-09","HI1 Index":"1986-05-06","HC1 Index":"2003-12-08","XU1 Index":"2006-09-05",
 "JGS1 Index":"2000-09-25",
 "TU1 Comdty":"1990-06-25","FV1 Comdty":"1988-05-20","TY1 Comdty":"1982-05-03","UXY1 Comdty":"2016-01-11","US1 Comdty":"1977-08-22",
 "WN1 Comdty":"2010-01-11","DU1 Comdty":"1997-03-07","OE1 Comdty":"1990-10-04","RX1 Comdty":"1988-09-29","UB1 Comdty":"2005-09-12",
 "G 1 Comdty":"1982-11-18","JB1 Comdty":"1985-10-19","OAT1 Comdty":"2012-04-16","CN1 Comdty":"1989-09-15","XM1 Comdty":"1984-12-01",
 "IK1 Comdty":"2009-09-14",
 "SFR5 Comdty":"2018-05-07","SFI5 Comdty":"2018-06-01","ER4 Comdty":"1998-12-01","IR4 Comdty":"1988-01-01",
 "EC1 Curncy":"1999-01-04","PE1 Curncy":"1995-04-25",
 "CUA1 Comdty":"2005-03-23","XB1 Comdty":"2005-10-03","NG1 Comdty":"1990-04-03","SCO1 Comdty":"2013-04-01",
 "LH1 Comdty":"1996-01-01",
}
