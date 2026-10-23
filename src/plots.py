"""Figures (matplotlib) : palette catégorielle validée (bleu = réel, orange = prédit)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import TZ

BLUE = "#2a78d6"     # série 1
ORANGE = "#eb6834"   # série 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _fmt_dates(ax, fmt="%d/%m\n%Hh"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=TZ))


def plot_first_week(df: pd.DataFrame, out: Path, region: str):
    """Première semaine complète de 2020 (corrige le titre trompeur du script initial)."""
    w = df["conso"].dropna().iloc[:336]
    w = w.tz_convert(TZ)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(w.index, w, color=BLUE, lw=2)
    ax.set_title(f"Consommation électrique, {region} — 7 premiers jours de 2020")
    ax.set_ylabel("Consommation (MW)")
    ax.set_xlabel("Date (heure locale)")
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    _fmt_dates(ax, "%a %d/%m")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_overview(s: pd.Series, out: Path, region: str):
    """Moyenne journalière sur toute la période : saisonnalité et tendance."""
    daily = s.tz_convert(TZ).resample("D").mean()
    roll = daily.rolling(30, center=True, min_periods=15).mean()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(daily.index, daily, color=BLUE, lw=0.9, alpha=0.55, label="Moyenne journalière")
    ax.plot(roll.index, roll, color=ORANGE, lw=2, label="Moyenne glissante 30 jours")
    ax.set_title(f"Consommation moyenne journalière, {region} (2020–2023)")
    ax.set_ylabel("Consommation (MW)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_pred(test: pd.DataFrame, pred: np.ndarray, out: Path, days: int = 7,
                        end: str | None = None, label: str | None = None):
    t = test.copy()
    t["pred"] = pred
    if end is not None:
        t = t[t.index < pd.Timestamp(end, tz=TZ)]
    w = t.iloc[-days * 48:]
    idx = w.index.tz_convert(TZ)
    mape = float(np.mean(np.abs(w["pred"] - w["conso"]) / w["conso"]) * 100)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(idx, w["conso"], color=BLUE, lw=2, label="Réel")
    ax.plot(idx, w["pred"], color=ORANGE, lw=2, ls="--", label="Prédit (LightGBM)")
    ax.set_title(f"Réel vs prédit — {label or str(days) + ' derniers jours du jeu de test'} (MAPE {mape:.2f} %)")
    ax.set_ylabel("Consommation (MW)")
    ax.set_xlabel("Date (heure locale)")
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    _fmt_dates(ax, "%a %d/%m")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(test: pd.DataFrame, pred: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(test["conso"], pred, s=4, color=BLUE, alpha=0.25, linewidths=0)
    lo, hi = float(min(test["conso"].min(), pred.min())), float(max(test["conso"].max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color=ORANGE, lw=1.8, label="Prédiction parfaite")
    ax.set_title("Prédit vs réel (année 2023)")
    ax.set_xlabel("Réel (MW)")
    ax.set_ylabel("Prédit (MW)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_gain_importance(gain: pd.DataFrame, out: Path, top: int = 15):
    g = gain.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(g["variable"], g["gain_%"], color=BLUE, height=0.62)
    for y, v in enumerate(g["gain_%"]):
        ax.text(v + 0.4, y, f"{v:.1f} %", va="center", color=INK2, fontsize=9)
    ax.set_title("Importance (gain LightGBM) — arbres entraînés sur l'écart à J-1")
    ax.set_xlabel("Part du gain total (%)")
    ax.set_xlim(0, g["gain_%"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_perm_importance(perm: pd.DataFrame, out: Path):
    p = perm.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.barh(p["groupe"], p["delta_MAE_MW"], xerr=p["ecart_type"], color=BLUE, height=0.62,
            error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.set_title("Importance par permutation de groupes — hausse de MAE sur le test 2023")
    ax.set_xlabel("Hausse de MAE quand le groupe est mélangé (MW)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_hour(by_hour: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(by_hour.index, by_hour["MAE_MW"], color=BLUE, width=0.7)
    ax.set_title("Erreur absolue moyenne par heure locale (test 2023)")
    ax.set_xlabel("Heure locale")
    ax.set_ylabel("MAE (MW)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame, out: Path):
    """Compare MAE des baselines et des modèles sur le test."""
    t = table.sort_values("MAE_MW", ascending=False)
    colors = [ORANGE if "LightGBM (complet)" in n else BLUE for n in t.index]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.barh(t.index, t["MAE_MW"], color=colors, height=0.6)
    for y, v in enumerate(t["MAE_MW"]):
        ax.text(v + 1, y, f"{v:.0f} MW", va="center", color=INK2, fontsize=9)
    ax.set_title("Comparaison des modèles — MAE sur le test 2023 (plus bas = mieux)")
    ax.set_xlabel("MAE (MW)")
    ax.set_xlim(0, t["MAE_MW"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
