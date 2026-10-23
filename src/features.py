"""Création des variables : calendrier cyclique, jours fériés, lags et fenêtres glissantes.

Cadre de prévision : « J-1 glissant ». Pour prédire la consommation à l'instant t,
on n'utilise que des informations disponibles à t - 24 h ou avant. Tous les lags
et statistiques glissantes sont donc décalés d'au moins 48 pas (24 h).
Aucune donnée de production (solaire, thermique...) n'est utilisée : elle n'est
pas connue à l'avance et fuiterait de l'information sur la cible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

from data import STEPS_PER_DAY

TZ = "Europe/Paris"
HORIZON = STEPS_PER_DAY  # 24 h

CALENDAR_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "holiday_eve", "holiday_next",
]
CYCLIC_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
]
LAG_FEATURES = [
    "lag_24h", "lag_48h", "lag_7d",
    "roll_mean_24h", "roll_std_24h", "roll_mean_7d",
]
ALL_FEATURES = CALENDAR_FEATURES + CYCLIC_FEATURES + LAG_FEATURES

# Jeu de variables proche du script initial (pour mesurer l'apport du travail)
BASELINE_FEATURES = ["hour", "dayofweek", "month", "is_holiday", "lag_24h"]

# Groupes pour l'importance par permutation (variables corrélées permutées ensemble)
FEATURE_GROUPS = {
    "Heure (hour, sin/cos)": ["hour", "hour_sin", "hour_cos"],
    "Jour de semaine (dow, week-end)": ["dayofweek", "dow_sin", "dow_cos", "is_weekend"],
    "Mois / saison": ["month", "month_sin", "month_cos", "doy_sin", "doy_cos"],
    "Jours fériés (+ veille/lendemain)": ["is_holiday", "holiday_eve", "holiday_next"],
    "Lag 24 h": ["lag_24h"],
    "Lag 48 h": ["lag_48h"],
    "Lag 7 jours": ["lag_7d"],
    "Moyennes/écart-type glissants": ["roll_mean_24h", "roll_std_24h", "roll_mean_7d"],
}


def _cyc(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_features(s: pd.Series) -> pd.DataFrame:
    """Construit le tableau de variables à partir de la série régulière (index UTC)."""
    if not s.index.is_monotonic_increasing:
        raise ValueError("L'index doit être trié.")

    local = s.index.tz_convert(TZ)  # heures locales : les habitudes de conso suivent l'heure légale
    df = pd.DataFrame({"conso": s.to_numpy()}, index=s.index)

    hour_frac = local.hour + local.minute / 60
    df["hour"] = hour_frac
    df["dayofweek"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)

    years = range(local.year.min() - 1, local.year.max() + 2)
    fr = holidays.France(years=years)
    dates = pd.Series(local.date, index=s.index)
    df["is_holiday"] = dates.map(lambda d: d in fr).astype(int)
    df["holiday_eve"] = dates.map(lambda d: (d + pd.Timedelta(days=1)) in fr).astype(int)
    df["holiday_next"] = dates.map(lambda d: (d - pd.Timedelta(days=1)) in fr).astype(int)

    # Variables cycliques : 23h30 et 00h00 deviennent voisines (ce que 'hour' brut ne permet pas)
    df["hour_sin"], df["hour_cos"] = _cyc(hour_frac.to_numpy(), 24)
    df["dow_sin"], df["dow_cos"] = _cyc(local.dayofweek.to_numpy(), 7)
    df["month_sin"], df["month_cos"] = _cyc(local.month.to_numpy() - 1, 12)
    df["doy_sin"], df["doy_cos"] = _cyc(local.dayofyear.to_numpy() - 1, 365.25)

    # Lags : la grille étant régulière, shift(k) == décalage de k * 30 min
    c = df["conso"]
    df["lag_24h"] = c.shift(HORIZON)
    df["lag_48h"] = c.shift(2 * HORIZON)
    df["lag_7d"] = c.shift(7 * HORIZON)

    past = c.shift(HORIZON)  # tout ce qui suit n'utilise que le passé >= 24 h
    df["roll_mean_24h"] = past.rolling(STEPS_PER_DAY, min_periods=STEPS_PER_DAY).mean()
    df["roll_std_24h"] = past.rolling(STEPS_PER_DAY, min_periods=STEPS_PER_DAY).std()
    df["roll_mean_7d"] = past.rolling(7 * STEPS_PER_DAY, min_periods=7 * STEPS_PER_DAY).mean()

    return df
