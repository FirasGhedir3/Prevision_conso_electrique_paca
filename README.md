# Prévision de la consommation électrique régionale (PACA) avec LightGBM

Pipeline complet de prévision de la consommation électrique à partir des données éCO2mix de RTE : nettoyage de la série temporelle, création de variables cycliques et de lags, modèle LightGBM, analyse de l'importance des variables et comparaison réel vs prédit.

**Résultat principal** : sur l'année 2023 complète (jamais vue à l'entraînement), le modèle atteint une **MAE de 128 MW** et un **MAPE de 2,85 %** (R² = 0,954), soit **26 % d'erreur en moins** que la baseline « même heure la veille » (MAE 174 MW, MAPE 3,91 %).

![Réel vs prédit](figures/03_reel_vs_predit_7_derniers_jours.png)

## Contexte et périmètre

- Données : éCO2mix régional consolidé, **Provence-Alpes-Côte d'Azur**, 2020–2023, pas de 30 minutes, soit **70 128 observations** (voir [data/README.md](data/README.md)).
- Tâche : prédire la consommation à l'instant *t* en n'utilisant que des informations disponibles à *t − 24 h* ou avant (prévision « J-1 glissant »).
- Aucune donnée de production (solaire, thermique…) ni de météo n'est utilisée : elles ne sont pas connues à l'avance ou ne sont pas dans le fichier.

## Résultats (jeu de test : 2023, 17 520 pas de 30 min)

| Modèle | MAE (MW) | RMSE (MW) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| Naïf : même heure J-7 | 321,0 | 438,2 | 6,91 | 0,713 |
| Naïf : même heure J-1 | 173,9 | 237,7 | 3,91 | 0,916 |
| LightGBM initial (5 variables, cible directe) | 130,3 | 178,3 | 2,91 | 0,953 |
| LightGBM 5 variables, cible résiduelle | 128,0 | 175,7 | 2,85 | 0,954 |
| **LightGBM complet (cible résiduelle)** | **128,1** | **175,5** | **2,85** | **0,954** |

Lecture honnête de ces chiffres :

- Le gain majeur vient du modèle lui-même par rapport aux baselines (−26 % de MAE vs J-1).
- Les variables supplémentaires (cycliques, lags 48 h / 7 j, fenêtres glissantes) **n'améliorent pas mesurablement le test 2023** par rapport aux 5 variables du script initial (128,1 vs 128,0 MW : écart dans le bruit). Elles sont conservées pour l'interprétation, mais ce n'est pas un gain de performance démontré.
- Erreurs par segment (test) : jours ouvrés 2,78 %, week-ends 2,98 %, jours fériés 3,84 %, Noël 2023 (25–26 déc.) 5,44 %.
- Saisonnalité de l'erreur : meilleure en octobre (MAE 60 MW), pire en décembre (202 MW) et février (187 MW), mois où la consommation est probablement plus sensible à la température, qui n'est pas dans les variables (hypothèse non testée).

## Importance des variables

![Permutation](figures/07_importance_permutation.png)

Le lag 24 h domine très largement (+771 MW de MAE s'il est détruit), suivi du jour de semaine (+70 MW) et de l'heure (+28 MW). Le graphique de gain LightGBM ([figures/06](figures/06_importance_gain.png)) décrit l'importance *dans un modèle entraîné sur l'écart à J-1* : le lag 24 h y paraît faible car il sert d'ancre à la prédiction, d'où l'usage de la permutation par groupes comme mesure de référence.

## Ce qui a été corrigé par rapport au script initial

1. **Fuseau horaire** : `hour` et les jours fériés étaient calculés sur l'heure UTC (décalage de 1–2 h). Tout est maintenant calculé en heure locale (Europe/Paris).
2. **Lags sur grille irrégulière** : 8 horodatages en double et des trous (changements d'heure) rendaient `shift(48)` incorrect à ces moments. La série est reconstruite sur une grille régulière de 30 min (doublons supprimés, trous interpolés).
3. **Commentaire contradictoire** : « 96 pas de 15 min » alors que le code décalait 48 pas de 30 min. Un seul pas de temps, documenté.
4. **Jeu de test** : 30 derniers jours seulement (décembre 2023), et le MAPE « 25–26 déc. 2020 » ne pouvait jamais être calculé (absent du test). Remplacé par un test sur 2023 entier et une évaluation sur Noël 2023.
5. **Pas de validation ni de baseline** : ajout d'un jeu de validation (T4 2022) avec arrêt anticipé, de deux baselines naïves et d'une ablation.
6. **Variables calendaires brutes** : `hour`/`month` entiers remplacés (en complément) par des encodages sinus/cosinus.
7. **Graphique de la première semaine** : le titre annonçait « 1ʳᵉ semaine de janvier 2020 » alors que les 48 premiers pas avaient été supprimés par le décalage et l'axe était en UTC. Corrigé.
8. **Taille des données** : le fichier fourni contient 70 128 lignes (une seule région), et non plus de 500 000. Le code traite plusieurs régions via `--region ALL` si vous disposez de l'export complet.

## Structure du dépôt

```
├── README.md
├── requirements.txt
├── data/README.md            # source des données et emplacement du CSV
├── src/
│   ├── data.py               # chargement, nettoyage, grille régulière
│   ├── features.py           # variables calendaires/cycliques, fériés, lags
│   ├── model.py              # LightGBM, baselines, métriques, importance
│   ├── plots.py              # figures
│   └── run_pipeline.py       # exécution de bout en bout
├── tests/test_pipeline.py    # tests anti-fuite de données
├── results/                  # métriques, importances, prédictions, summary.json, log
├── figures/                  # 9 graphiques
└── docs/rapport.md           # rapport détaillé
```

## Reproduire

```bash
pip install -r requirements.txt
# placer eco2mix-regional-cons-def.csv dans data/raw/
python src/run_pipeline.py          # ~10 s, régénère results/ et figures/
python -m pytest -q tests           # tests unitaires
```

## Limites et pistes

- Pas de météo (température) : principale piste d'amélioration, surtout pour l'hiver.
- Une seule région et 4 ans de données, dont 2020 (Covid) et 2022–2023 (sobriété énergétique) atypiques.
- Pas de réglage poussé des hyperparamètres ; pas d'intervalles de prédiction (régression quantile possible avec LightGBM).
- Validation sur un seul découpage temporel ; une validation croisée en fenêtre glissante renforcerait les conclusions.

Rapport détaillé : [docs/rapport.md](docs/rapport.md).
