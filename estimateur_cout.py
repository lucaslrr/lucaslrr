"""Estimateur de coût d'exécution pour des actions du SBF 120.

Pour chaque titre :
  * ADV (Average Daily Volume) sur 20 séances, en titres et en euros ;
  * volatilité quotidienne (écart-type des rendements logarithmiques sur un an).

Pour un ordre (titre, quantité) :
  * taille de l'ordre en % de l'ADV ;
  * nombre de jours conseillé en POV (Percentage Of Volume) à 15 % ;
  * impact de marché estimé avec la loi en racine carrée :
        impact = Y * sigma_quotidienne * sqrt(Q / ADV)

Usage :
    python estimateur_cout.py                          # tableau des 10 titres + saisie interactive
    python estimateur_cout.py --ordre MC.PA 50000      # un ou plusieurs ordres
    python estimateur_cout.py --demo --ordre SOI.PA 200000   # données simulées, hors ligne
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 10 valeurs du SBF 120 : 6 grandes capitalisations du CAC 40 et 4 valeurs moyennes,
# pour montrer l'écart de liquidité entre les deux segments.
TITRES = {
    "MC.PA": "LVMH",
    "TTE.PA": "TotalEnergies",
    "AIR.PA": "Airbus",
    "SAN.PA": "Sanofi",
    "BNP.PA": "BNP Paribas",
    "SU.PA": "Schneider Electric",
    "SOI.PA": "Soitec",
    "NEX.PA": "Nexans",
    "RUI.PA": "Rubis",
    "VK.PA": "Vallourec",
}

FENETRE_ADV = 20
POV_DEFAUT = 0.15
Y_DEFAUT = 1.0  # constante de la loi en racine carrée, typiquement entre 0,5 et 1
SEANCES_PAR_AN = 252


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def telecharger(tickers: list[str], periode: str = "1y") -> dict[str, pd.DataFrame]:
    """Télécharge cours de clôture ajustés et volumes quotidiens via yfinance."""
    import yfinance as yf

    brut = yf.download(
        tickers, period=periode, auto_adjust=True, progress=False, group_by="column"
    )
    donnees = {}
    for t in tickers:
        try:
            df = pd.DataFrame({"Close": brut["Close"][t], "Volume": brut["Volume"][t]})
        except KeyError:
            continue
        df = df.dropna()
        if not df.empty:
            donnees[t] = df
    manquants = set(tickers) - set(donnees)
    if manquants:
        print(f"Attention : aucune donnée pour {', '.join(sorted(manquants))}", file=sys.stderr)
    if not donnees:
        raise RuntimeError("Aucune donnée téléchargée (connexion à Yahoo Finance ?).")
    return donnees


# Paramètres de simulation (cours initial, volume moyen, volatilité quotidienne)
# proches des ordres de grandeur réels, pour le mode --demo.
_PARAMS_DEMO = {
    "MC.PA": (600.0, 450_000, 0.018),
    "TTE.PA": (57.0, 4_500_000, 0.013),
    "AIR.PA": (160.0, 1_200_000, 0.016),
    "SAN.PA": (90.0, 2_000_000, 0.013),
    "BNP.PA": (65.0, 3_500_000, 0.017),
    "SU.PA": (230.0, 900_000, 0.017),
    "SOI.PA": (60.0, 350_000, 0.035),
    "NEX.PA": (110.0, 110_000, 0.022),
    "RUI.PA": (27.0, 250_000, 0.017),
    "VK.PA": (16.0, 1_300_000, 0.024),
}


def donnees_demo(tickers: list[str], jours: int = SEANCES_PAR_AN, graine: int = 42) -> dict[str, pd.DataFrame]:
    """Génère des séries simulées (marche aléatoire géométrique + volumes log-normaux)."""
    rng = np.random.default_rng(graine)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=jours)
    donnees = {}
    for t in tickers:
        prix0, volume_moyen, sigma = _PARAMS_DEMO.get(t, (50.0, 500_000, 0.02))
        rendements = rng.normal(0.0, sigma, jours)
        cours = prix0 * np.exp(np.cumsum(rendements))
        volumes = volume_moyen * rng.lognormal(-0.125, 0.5, jours)  # moyenne ~ volume_moyen
        donnees[t] = pd.DataFrame({"Close": cours, "Volume": volumes.round()}, index=dates)
    return donnees


# ---------------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------------

@dataclass
class Statistiques:
    ticker: str
    dernier_cours: float
    adv_titres: float
    adv_eur: float
    vol_quotidienne: float
    nb_seances: int

    @property
    def vol_annualisee(self) -> float:
        return self.vol_quotidienne * math.sqrt(SEANCES_PAR_AN)


def statistiques(ticker: str, df: pd.DataFrame, fenetre_adv: int = FENETRE_ADV) -> Statistiques:
    """ADV sur `fenetre_adv` séances et volatilité quotidienne sur tout l'historique."""
    recent = df.tail(fenetre_adv)
    adv_titres = recent["Volume"].mean()
    adv_eur = (recent["Volume"] * recent["Close"]).mean()
    rendements = np.log(df["Close"]).diff().dropna()
    return Statistiques(
        ticker=ticker,
        dernier_cours=float(df["Close"].iloc[-1]),
        adv_titres=float(adv_titres),
        adv_eur=float(adv_eur),
        vol_quotidienne=float(rendements.std()),
        nb_seances=len(df),
    )


@dataclass
class Estimation:
    stats: Statistiques
    quantite: float
    pov: float
    y: float

    @property
    def notionnel(self) -> float:
        return self.quantite * self.stats.dernier_cours

    @property
    def pct_adv(self) -> float:
        return self.quantite / self.stats.adv_titres

    @property
    def jours_pov(self) -> float:
        """Durée nécessaire pour exécuter l'ordre en participant à `pov` du volume."""
        return self.pct_adv / self.pov

    @property
    def jours_conseilles(self) -> int:
        return max(1, math.ceil(self.jours_pov))

    @property
    def impact(self) -> float:
        """Loi en racine carrée : Y * sigma * sqrt(Q / ADV), en fraction du prix."""
        return self.y * self.stats.vol_quotidienne * math.sqrt(self.pct_adv)

    @property
    def impact_eur(self) -> float:
        return self.impact * self.notionnel


def estimer(stats: Statistiques, quantite: float, pov: float = POV_DEFAUT, y: float = Y_DEFAUT) -> Estimation:
    if quantite <= 0:
        raise ValueError("La quantité doit être strictement positive.")
    if not 0 < pov <= 1:
        raise ValueError("Le taux de participation doit être dans ]0, 1].")
    return Estimation(stats, quantite, pov, y)


def commentaire(pct_adv: float, pov: float = POV_DEFAUT) -> str:
    if pct_adv < 0.01:
        return "Ordre petit devant la liquidité : exécution intraday sans difficulté."
    if pct_adv <= pov:
        return "Ordre significatif : un algo POV/VWAP sur la séance est adapté."
    if pct_adv < 0.30:
        return "Ordre important : étaler sur plusieurs séances, surveiller le risque de marché."
    return ("Ordre très large : au-delà du domaine de calibration de la loi en racine carrée ; "
            "envisager un block / capital risk ou un étalement long.")


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def fmt(x: float, decimales: int = 0) -> str:
    """Format français : espace pour les milliers, virgule décimale."""
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")


def fmt_pct(x: float, decimales: int = 2) -> str:
    return fmt(100 * x, decimales) + " %"


def tableau(stats: list[Statistiques]) -> str:
    lignes = [
        {
            "Titre": s.ticker,
            "Société": TITRES.get(s.ticker, ""),
            "Cours (€)": fmt(s.dernier_cours, 2),
            f"ADV {FENETRE_ADV} j (titres)": fmt(s.adv_titres),
            f"ADV {FENETRE_ADV} j (M€)": fmt(s.adv_eur / 1e6, 1),
            "Vol. quot.": fmt_pct(s.vol_quotidienne),
            "Vol. ann.": fmt_pct(s.vol_annualisee, 1),
        }
        for s in sorted(stats, key=lambda s: s.adv_eur, reverse=True)
    ]
    return pd.DataFrame(lignes).to_string(index=False)


def rapport(e: Estimation) -> str:
    s = e.stats
    jours = f"{fmt(e.jours_pov, 2)} séance(s) → {e.jours_conseilles} jour(s) conseillé(s)"
    lignes = [
        ("Cours de référence", f"{fmt(s.dernier_cours, 2)} €"),
        ("Notionnel", f"{fmt(e.notionnel)} €"),
        (f"ADV {FENETRE_ADV} j", f"{fmt(s.adv_titres)} titres ({fmt(s.adv_eur / 1e6, 1)} M€)"),
        ("% de l'ADV", fmt_pct(e.pct_adv)),
        ("Volatilité quotidienne", fmt_pct(s.vol_quotidienne)),
        (f"Durée en POV {fmt(100 * e.pov, 0)} %", jours),
        (f"Impact estimé (Y = {fmt(e.y, 2)})", f"{fmt(e.impact * 1e4, 1)} bps ≈ {fmt(e.impact_eur)} €"),
    ]
    titre = f"Ordre : {s.ticker} ({TITRES.get(s.ticker, '?')}) — {fmt(e.quantite)} titres"
    largeur = max(len(k) for k, _ in lignes)
    corps = "\n".join(f"  {k:<{largeur}}  {v}" for k, v in lignes)
    return f"{titre}\n{corps}\n  → {commentaire(e.pct_adv, e.pov)}"


# ---------------------------------------------------------------------------
# Interface en ligne de commande
# ---------------------------------------------------------------------------

def _normaliser_ticker(saisie: str, disponibles: dict) -> str:
    t = saisie.strip().upper()
    if t not in disponibles and f"{t}.PA" in disponibles:
        t = f"{t}.PA"
    if t not in disponibles:
        raise ValueError(f"Titre inconnu : {saisie}. Disponibles : {', '.join(disponibles)}")
    return t


def _quantite(saisie: str) -> float:
    try:
        return float(saisie.replace(" ", "").replace("_", "").replace(",", "."))
    except ValueError:
        raise ValueError(f"Quantité invalide : {saisie}") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Estimateur de coût d'exécution (SBF 120).")
    p.add_argument("--ordre", nargs=2, action="append", metavar=("TITRE", "QUANTITE"),
                   help="ordre à estimer, ex. --ordre MC.PA 50000 (répétable)")
    p.add_argument("--pov", type=float, default=POV_DEFAUT, help="taux de participation (défaut 0.15)")
    p.add_argument("--y", type=float, default=Y_DEFAUT, help="constante Y de la loi en racine carrée (défaut 1.0)")
    p.add_argument("--tickers", nargs="+", default=list(TITRES), help="liste de tickers Yahoo")
    p.add_argument("--demo", action="store_true", help="données simulées (hors ligne)")
    args = p.parse_args(argv)

    if args.demo:
        print("*** MODE DÉMO : données simulées, pas de vrais cours ***\n")
        donnees = donnees_demo(args.tickers)
    else:
        print("Téléchargement d'un an de cours et volumes via yfinance…\n")
        donnees = telecharger(args.tickers)

    stats = {t: statistiques(t, df) for t, df in donnees.items()}
    print(tableau(list(stats.values())))
    print()

    if args.ordre:
        for ticker, quantite in args.ordre:
            e = estimer(stats[_normaliser_ticker(ticker, stats)], _quantite(quantite), args.pov, args.y)
            print(rapport(e), end="\n\n")
        return 0

    if not sys.stdin.isatty():
        return 0
    print("Saisir un ordre « TITRE QUANTITÉ » (ex. MC 50000), ou Entrée pour quitter.")
    while True:
        try:
            saisie = input("> ").strip()
        except EOFError:
            break
        if not saisie:
            break
        try:
            ticker, quantite = saisie.rsplit(maxsplit=1)
            e = estimer(stats[_normaliser_ticker(ticker, stats)], _quantite(quantite), args.pov, args.y)
            print(rapport(e), end="\n\n")
        except ValueError as err:
            print(f"Erreur : {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
