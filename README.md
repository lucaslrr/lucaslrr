# Estimateur de coût d'exécution — SBF 120

Petit outil Python qui estime le coût d'exécution d'un ordre actions sur 10 valeurs du SBF 120.

1. **Données** : un an de cours de clôture (ajustés) et de volumes quotidiens via `yfinance`.
2. **Statistiques par titre** :
   - ADV 20 jours (moyenne des volumes des 20 dernières séances), en titres et en M€ ;
   - volatilité quotidienne σ = écart-type des rendements logarithmiques sur un an (et σ annualisée = σ·√252).
3. **Pour un ordre (titre, quantité Q)** :
   - **% de l'ADV** = Q / ADV ;
   - **durée en POV 15 %** = (Q / ADV) / 15 %, arrondie à la séance supérieure ;
   - **impact estimé, loi en racine carrée** : `I = Y · σ · √(Q / ADV)`, avec Y ≈ 1 (réglable), en bps et en €.

Titres retenus : 6 grandes capitalisations (LVMH, TotalEnergies, Airbus, Sanofi, BNP Paribas, Schneider)
et 4 valeurs moyennes (Soitec, Nexans, Rubis, Vallourec), pour faire ressortir l'écart de liquidité.

## Utilisation

```bash
pip install -r requirements.txt

python estimateur_cout.py                                  # tableau + saisie interactive (ex. « MC 50000 »)
python estimateur_cout.py --ordre MC.PA 50000 --ordre SOI 200000
python estimateur_cout.py --pov 0.10 --y 0.7 --ordre NEX 30000
python estimateur_cout.py --demo --ordre MC 50000          # données simulées, sans connexion
python -m pytest                                           # tests
```

Exemple de sortie (mode `--demo`, donc chiffres simulés) :

```
Ordre : NEX.PA (Nexans) — 30 000 titres
  Cours de référence        161,18 €
  Notionnel                 4 835 351 €
  ADV 20 j                  103 868 titres (16,9 M€)
  % de l'ADV                28,88 %
  Volatilité quotidienne    2,14 %
  Durée en POV 15 %         1,93 séance(s) → 2 jour(s) conseillé(s)
  Impact estimé (Y = 1,00)  114,9 bps ≈ 55 542 €
  → Ordre important : étaler sur plusieurs séances, surveiller le risque de marché.
```

## Points à discuter en entretien

- **Pourquoi la racine carrée ?** L'impact est concave en taille : doubler l'ordre multiplie l'impact par
  √2 ≈ 1,41, pas par 2. C'est une loi empirique robuste (Torre/Barra, Almgren et al. 2005, Tóth et al. 2011),
  valable en gros pour Q/ADV entre 0,1 % et ~20-30 %.
- **La durée n'apparaît pas au premier ordre** : l'impact dépend surtout de Q/ADV. Étaler réduit l'impact
  *temporaire* (moins agressif sur le carnet), mais augmente le **risque de marché** (σ·√T) : c'est
  l'arbitrage coût/risque d'Almgren-Chriss, et le choix du taux de POV en découle.
- **Limites** : ADV moyenne sensible aux jours exceptionnels (fixings, rebalancements d'indices : la médiane
  est plus robuste) ; pas de spread (ajouter ½ spread pour un coût total) ; pas de profil intraday ; Y à
  calibrer sur ses propres exécutions (TCA) ; volumes Yahoo = marché principal Euronext, sans les MTF
  (Cboe, Turquoise, Aquis), qui représentent une part importante de la liquidité européenne.
