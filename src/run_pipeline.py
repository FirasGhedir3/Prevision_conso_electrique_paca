"""Pipeline complet : nettoyage -> features -> LightGBM -> évaluation -> figures.

Usage :
    python src/run_pipeline.py                      # PACA, fichier par défaut
    python src/run_pipeline.py --region "Bretagne"  # autre région (si présente dans le CSV)
    python src/run_pipeline.py --region ALL         # toutes les régions du CSV (une par une)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D
import features as F
import model as M
import plots as P

ROOT = Path(__file__).resolve().parents[1]


def run(csv: Path, region: str, out_res: Path, out_fig: Path) -> dict:
    print(f"[1/6] Chargement : {csv.name}")
    raw = D.load_raw(csv)

    print(f"[2/6] Nettoyage ({region})")
    s, quality = D.clean_series(raw, region=region)
    print("      ", quality)

    print("[3/6] Création des variables (cycliques, fériés, lags)")
    df = F.build_features(s)

    print("[4/6] Entraînement et évaluation")
    train, valid, test = M.split_temporal(df, F.ALL_FEATURES)
    print(f"       train={len(train)}  valid={len(valid)}  test={len(test)}")

    model, best_iter = M.fit_lgbm(train, valid, F.ALL_FEATURES, residual=True)
    pred = model.predict(test)

    # Ablation : variables du script initial (5 variables), cible directe puis résiduelle
    tr_b, va_b, te_b = M.split_temporal(df, F.BASELINE_FEATURES)
    model_b, _ = M.fit_lgbm(tr_b, va_b, F.BASELINE_FEATURES, residual=False)
    pred_b = model_b.predict(te_b)
    model_br, _ = M.fit_lgbm(tr_b, va_b, F.BASELINE_FEATURES, residual=True)
    pred_br = model_br.predict(te_b)

    rows = {
        "Naïf : même heure J-1": M.metrics(test["conso"], test["lag_24h"]),
        "Naïf : même heure J-7": M.metrics(test["conso"], test["lag_7d"]),
        "LightGBM initial (5 var., cible directe)": M.metrics(te_b["conso"], pred_b),
        "LightGBM 5 var., cible résiduelle": M.metrics(te_b["conso"], pred_br),
        "LightGBM (complet)": M.metrics(test["conso"], pred),
    }
    table = pd.DataFrame(rows).T
    print(table.round(3))

    print("[5/6] Importance des variables et erreurs par segment")
    gain = M.gain_importance(model, F.ALL_FEATURES)
    perm = M.grouped_permutation_importance(model, test[F.ALL_FEATURES], test["conso"])
    seg = M.segment_metrics(test, pred)

    print("[6/6] Sauvegarde des résultats et figures")
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    table.round(4).to_csv(out_res / "metrics_comparison.csv", encoding="utf-8")
    gain.round(4).to_csv(out_res / "feature_importance_gain.csv", index=False)
    perm.round(4).to_csv(out_res / "feature_importance_permutation.csv", index=False)
    seg["par_heure"].round(4).to_csv(out_res / "errors_by_hour.csv")
    seg["par_mois"].round(4).to_csv(out_res / "errors_by_month.csv")
    pd.DataFrame({
        "datetime_utc": test.index,
        "reel_MW": test["conso"].to_numpy(),
        "predit_MW": pred.round(1),
    }).to_csv(out_res / "predictions_test.csv", index=False)

    summary = {
        "region": region,
        "qualite_donnees": quality,
        "decoupage": {
            "train": f"{train.index.min().tz_convert(F.TZ).date()} -> {train.index.max().tz_convert(F.TZ).date()} ({len(train)} pas)",
            "validation": f"{valid.index.min().tz_convert(F.TZ).date()} -> {valid.index.max().tz_convert(F.TZ).date()} ({len(valid)} pas)",
            "test": f"{test.index.min().tz_convert(F.TZ).date()} -> {test.index.max().tz_convert(F.TZ).date()} ({len(test)} pas)",
        },
        "meilleure_iteration_early_stopping": best_iter,
        "metriques_test": rows,
        "segments_test": seg["segments"],
        "top_variables_gain": gain.head(5)[["variable", "gain_%"]].round(2).to_dict("records"),
        "importance_permutation": perm.round(3).to_dict("records"),
    }
    (out_res / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    P.plot_first_week(df, out_fig / "01_premiere_semaine_2020.png", region)
    P.plot_overview(s, out_fig / "02_vue_ensemble_2020_2023.png", region)
    P.plot_actual_vs_pred(test, pred, out_fig / "03_reel_vs_predit_7_derniers_jours.png")
    P.plot_actual_vs_pred(test, pred, out_fig / "04_reel_vs_predit_semaine_noel_2023.png",
                          end="2023-12-28", label="7 jours autour de Noël 2023 (21–27 déc.)")
    P.plot_scatter(test, pred, out_fig / "05_scatter_reel_vs_predit.png")
    P.plot_gain_importance(gain, out_fig / "06_importance_gain.png")
    P.plot_perm_importance(perm, out_fig / "07_importance_permutation.png")
    P.plot_error_by_hour(seg["par_heure"], out_fig / "08_erreur_par_heure.png")
    P.plot_model_comparison(table, out_fig / "09_comparaison_modeles.png")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "raw" / "eco2mix-regional-cons-def.csv"))
    ap.add_argument("--region", default="Provence-Alpes-Côte d'Azur")
    args = ap.parse_args()

    csv = Path(args.csv)
    regions = [args.region]
    if args.region == "ALL":
        regions = sorted(D.load_raw(csv)["Région"].dropna().unique())

    for r in regions:
        slug = "" if len(regions) == 1 else "/" + r.replace(" ", "_").replace("'", "")
        run(csv, r, ROOT / "results" / slug.strip("/"), ROOT / "figures" / slug.strip("/"))


if __name__ == "__main__":
    main()
