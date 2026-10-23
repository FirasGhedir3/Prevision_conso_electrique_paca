"""Chargement et nettoyage des données éCO2mix régionales (RTE / ODRÉ).

Le fichier brut contient un pas de 30 minutes. Les dates sont fournies avec
un décalage horaire (+01:00 / +02:00) : on les convertit en UTC pour obtenir
un axe temporel strictement monotone, puis on reconstruit une grille régulière
de 30 minutes (les passages heure d'été / heure d'hiver créent des doublons
et des trous dans le fichier brut).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FREQ = "30min"
STEPS_PER_DAY = 48
TARGET_RAW = "Consommation (MW)"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lit le CSV brut (séparateur ';', BOM UTF-8 éventuel)."""
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def clean_series(
    df: pd.DataFrame,
    region: str | None = None,
    max_gap_steps: int = 4,
) -> tuple[pd.Series, dict]:
    """Retourne la consommation sur une grille régulière UTC + un rapport de qualité.

    Étapes :
      1. filtre sur la région (si fournie) ;
      2. conversion de 'Date - Heure' en UTC ;
      3. suppression des doublons horaires (changement d'heure de printemps) ;
      4. reindexation sur une grille régulière de 30 minutes ;
      5. interpolation temporelle des trous courts (<= max_gap_steps pas) ;
      6. suppression des valeurs aberrantes physiquement impossibles (<= 0).
    """
    report: dict = {"lignes_brutes": int(len(df))}

    if region is not None:
        df = df[df["Région"] == region]
    report["region"] = region if region else "toutes (agrégées)"
    report["lignes_region"] = int(len(df))

    ts = pd.to_datetime(df["Date - Heure"], utc=True)
    s = pd.Series(df[TARGET_RAW].to_numpy(dtype="float64"), index=ts, name="conso")
    s = s.sort_index(kind="stable")

    n_dup = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="first")]
    report["doublons_horodatage_supprimes"] = n_dup

    n_neg = int((s <= 0).sum())
    s = s.mask(s <= 0)
    report["valeurs_non_positives_supprimees"] = n_neg

    full_index = pd.date_range(s.index.min(), s.index.max(), freq=FREQ, tz="UTC")
    s = s.reindex(full_index)
    n_missing_before = int(s.isna().sum())
    s = s.interpolate(method="time", limit=max_gap_steps, limit_area="inside")
    n_missing_after = int(s.isna().sum())

    report["pas_manquants_apres_reindexation"] = n_missing_before
    report["pas_interpoles"] = n_missing_before - n_missing_after
    report["pas_manquants_restants"] = n_missing_after
    report["lignes_grille_reguliere"] = int(len(s))
    report["debut"] = str(s.index.min())
    report["fin"] = str(s.index.max())
    s.index.name = "datetime_utc"
    return s, report
