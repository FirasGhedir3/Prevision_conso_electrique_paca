# Données

Le pipeline attend le fichier `data/raw/eco2mix-regional-cons-def.csv`.

- **Source** : RTE, jeu de données « éCO2mix régional consolidé et définitif », publié sur le portail Open Data Réseaux Énergies (ODRÉ). Données sous Licence Ouverte (Etalab).
- **Contenu du fichier utilisé ici** : 70 128 lignes, une seule région (Provence-Alpes-Côte d'Azur, code INSEE 93), du 01/01/2020 au 31/12/2023, pas de 30 minutes.
- **Colonne cible** : `Consommation (MW)`.
- Le CSV n'est pas versionné dans Git (`.gitignore`). Placez-le dans `data/raw/` avant de lancer `python src/run_pipeline.py`.

Si vous téléchargez l'export multi-régions, lancez `python src/run_pipeline.py --region ALL` pour traiter chaque région.
