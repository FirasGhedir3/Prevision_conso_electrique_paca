"""Tests de non-régression : pas de fuite de données, grille régulière, cyclicité."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as D
import features as F


def _toy_series(n_days=20):
    idx = pd.date_range("2022-01-01", periods=n_days * 48, freq="30min", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float) + 1000, index=idx, name="conso")


def test_lags_use_only_past_values():
    s = _toy_series()
    df = F.build_features(s)
    # lag_24h(t) doit valoir conso(t - 24 h) exactement
    t = s.index[500]
    assert df.loc[t, "lag_24h"] == s.loc[t - pd.Timedelta(hours=24)]
    assert df.loc[t, "lag_7d"] == s.loc[t - pd.Timedelta(days=7)]


def test_rolling_features_do_not_see_current_day():
    s = _toy_series()
    df = F.build_features(s)
    t = s.index[600]
    window = s.loc[t - pd.Timedelta(hours=24 + 23.5): t - pd.Timedelta(hours=24)]
    assert np.isclose(df.loc[t, "roll_mean_24h"], window.mean())


def test_cyclic_hour_is_continuous_across_midnight():
    s = _toy_series()
    df = F.build_features(s)
    # 23h30 et 00h00 (heure locale) sont voisines sur le cercle
    local = df.index.tz_convert(F.TZ)
    a = df[(local.hour == 23) & (local.minute == 30)].iloc[0]
    b = df[(local.hour == 0) & (local.minute == 0)].iloc[1]
    d = np.hypot(a.hour_sin - b.hour_sin, a.hour_cos - b.hour_cos)
    assert d < 0.15


def test_clean_series_regular_grid_and_duplicates():
    raw = pd.DataFrame({
        "Région": ["X"] * 5,
        "Date - Heure": ["2022-03-27T03:00:00+02:00", "2022-03-27T03:00:00+02:00",
                         "2022-03-27T03:30:00+02:00", "2022-03-27T04:30:00+02:00",
                         "2022-03-27T05:00:00+02:00"],
        "Consommation (MW)": [100.0, 100.0, 110.0, 130.0, 140.0],
    })
    s, rep = D.clean_series(raw, region="X")
    assert rep["doublons_horodatage_supprimes"] == 1
    assert s.index.is_monotonic_increasing and s.index.is_unique
    assert (s.index.to_series().diff().dropna() == pd.Timedelta("30min")).all()
    assert s.isna().sum() == 0  # le trou de 30 min est interpolé
