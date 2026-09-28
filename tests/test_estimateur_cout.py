import math

import numpy as np
import pandas as pd
import pytest

import estimateur_cout as ec


def _serie(cours, volumes):
    dates = pd.bdate_range("2025-01-01", periods=len(cours))
    return pd.DataFrame({"Close": cours, "Volume": volumes}, index=dates)


def test_adv_sur_20_dernieres_seances():
    volumes = [1_000_000] * 30 + [200_000] * 20
    df = _serie([10.0] * 50, volumes)
    s = ec.statistiques("X.PA", df)
    assert s.adv_titres == pytest.approx(200_000)
    assert s.adv_eur == pytest.approx(2_000_000)
    assert s.dernier_cours == 10.0


def test_volatilite_quotidienne_rendements_log():
    rendements = np.array([0.01, -0.02, 0.015, -0.005, 0.0])
    cours = 100 * np.exp(np.concatenate([[0.0], np.cumsum(rendements)]))
    s = ec.statistiques("X.PA", _serie(cours, [1] * len(cours)))
    assert s.vol_quotidienne == pytest.approx(rendements.std(ddof=1))
    assert s.vol_annualisee == pytest.approx(s.vol_quotidienne * math.sqrt(252))


def test_estimation_ordre():
    s = ec.Statistiques("X.PA", dernier_cours=50.0, adv_titres=1_000_000,
                        adv_eur=50e6, vol_quotidienne=0.02, nb_seances=252)
    e = ec.estimer(s, 250_000, pov=0.15, y=1.0)
    assert e.pct_adv == pytest.approx(0.25)
    assert e.jours_pov == pytest.approx(0.25 / 0.15)
    assert e.jours_conseilles == 2
    assert e.impact == pytest.approx(0.02 * 0.5)  # 100 bps
    assert e.impact_eur == pytest.approx(0.01 * 250_000 * 50)


def test_petit_ordre_au_moins_un_jour():
    s = ec.Statistiques("X.PA", 50.0, 1_000_000, 50e6, 0.02, 252)
    assert ec.estimer(s, 100).jours_conseilles == 1


@pytest.mark.parametrize("quantite,pov", [(0, 0.15), (-5, 0.15), (100, 0), (100, 1.5)])
def test_parametres_invalides(quantite, pov):
    s = ec.Statistiques("X.PA", 50.0, 1_000_000, 50e6, 0.02, 252)
    with pytest.raises(ValueError):
        ec.estimer(s, quantite, pov=pov)


def test_demo_de_bout_en_bout(capsys):
    assert ec.main(["--demo", "--ordre", "MC", "50000", "--ordre", "SOI.PA", "1 000"]) == 0
    sortie = capsys.readouterr().out
    assert "LVMH" in sortie and "Soitec" in sortie
    assert "% de l'ADV" in sortie
