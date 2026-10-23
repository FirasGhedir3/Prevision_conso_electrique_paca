"""Entraînement LightGBM, baselines, métriques et importance des variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import ALL_FEATURES, BASELINE_FEATURES, FEATURE_GROUPS, TZ

SEED = 42

# Découpage temporel strict (heure locale) : aucun mélange aléatoire sur une série temporelle
TRAIN_END = pd.Timestamp("2022-10-01", tz=TZ)
VALID_END = pd.Timestamp("2023-01-01", tz=TZ)

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def split_temporal(df: pd.DataFrame, features: list[str]):
    """Train (début -> 30/09/2022), validation (T4 2022), test (année 2023 complète)."""
    d = df.dropna(subset=features + ["conso"])
    train = d[d.index < TRAIN_END]
    valid = d[(d.index >= TRAIN_END) & (d.index < VALID_END)]
    test = d[d.index >= VALID_END]
    return train, valid, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / y_true) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


class Forecaster:
    """LightGBM + (optionnel) cible résiduelle.

    residual=True : le modèle apprend l'écart conso(t) - conso(t-24h) ; la prédiction finale
    est lag_24h + écart prédit. Cela évite que les arbres (constants par morceaux) aient à
    reproduire le niveau absolu, ce qui les rend plus robustes aux dérives de niveau
    (crise énergétique 2022-2023, sobriété...).
    """

    def __init__(self, features: list[str], residual: bool, n_estimators: int):
        self.features = features
        self.residual = residual
        self.est = lgb.LGBMRegressor(n_estimators=n_estimators, **LGB_PARAMS)

    @property
    def booster_(self):
        return self.est.booster_

    def _y(self, d: pd.DataFrame) -> pd.Series:
        return d["conso"] - d["lag_24h"] if self.residual else d["conso"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.est.predict(X[self.features])
        return p + X["lag_24h"].to_numpy() if self.residual else p


def fit_lgbm(train, valid, features: list[str], residual: bool = True):
    """Entraîne avec arrêt anticipé sur la validation, puis ré-entraîne sur train+valid
    avec le nombre d'itérations optimal (ajusté de 10 % pour la taille de données accrue)."""
    first = Forecaster(features, residual, n_estimators=5000)
    first.est.fit(
        train[features], first._y(train),
        eval_set=[(valid[features], first._y(valid))],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iter = int(first.est.best_iteration_)
    full = pd.concat([train, valid])
    final = Forecaster(features, residual, n_estimators=max(50, int(best_iter * 1.1)))
    final.est.fit(full[features], final._y(full))
    return final, best_iter


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                                   n_repeats: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Hausse de MAE quand on permute un groupe de variables corrélées (sur le test).
    Plus fiable que le gain pour des variables redondantes (hour / hour_sin / hour_cos)."""
    rng = np.random.default_rng(seed)
    # Si le modèle est résiduel, permuter le groupe « Lag 24 h » détruit aussi l'ancre
    # lag_24h ajoutée à la prédiction : on mesure bien la perte d'information associée.
    base = np.mean(np.abs(model.predict(X) - y.to_numpy()))
    rows = []
    for name, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            deltas.append(np.mean(np.abs(model.predict(Xp) - y.to_numpy())) - base)
        rows.append({"groupe": name, "delta_MAE_MW": float(np.mean(deltas)),
                     "ecart_type": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("delta_MAE_MW", ascending=False).reset_index(drop=True)


def gain_importance(model, features: list[str]) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    out = pd.DataFrame({"variable": features, "gain": gain})
    out["gain_%"] = out["gain"] / out["gain"].sum() * 100
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def segment_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """Erreurs par segment : heure, mois, jours fériés, week-end, Noël."""
    t = test.copy()
    t["pred"] = pred
    t["ape"] = np.abs(t["pred"] - t["conso"]) / t["conso"] * 100
    t["ae"] = np.abs(t["pred"] - t["conso"])
    local = t.index.tz_convert(TZ)
    t["h"] = local.hour
    t["m"] = local.month

    out = {
        "par_heure": t.groupby("h")[["ae", "ape"]].mean().rename(
            columns={"ae": "MAE_MW", "ape": "MAPE_%"}),
        "par_mois": t.groupby("m")[["ae", "ape"]].mean().rename(
            columns={"ae": "MAE_MW", "ape": "MAPE_%"}),
    }
    seg = {}
    for label, mask in {
        "jour_ouvre_normal": (t["is_holiday"] == 0) & (t["is_weekend"] == 0),
        "week_end": t["is_weekend"] == 1,
        "jour_ferie": t["is_holiday"] == 1,
    }.items():
        if mask.any():
            seg[label] = {"MAE_MW": float(t.loc[mask, "ae"].mean()),
                          "MAPE_%": float(t.loc[mask, "ape"].mean()),
                          "n": int(mask.sum())}
    dates = pd.Series(local.date, index=t.index)
    noel = dates.isin([pd.Timestamp("2023-12-25").date(), pd.Timestamp("2023-12-26").date()])
    if noel.any():
        seg["noel_25_26_dec_2023"] = {"MAE_MW": float(t.loc[noel.to_numpy(), "ae"].mean()),
                                      "MAPE_%": float(t.loc[noel.to_numpy(), "ape"].mean()),
                                      "n": int(noel.sum())}
    out["segments"] = seg
    return out
