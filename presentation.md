---
marp: true
theme: default
paginate: false
size: 16:9
---

# Projet de classification supervisée
## Prédiction des sinistres en assurance auto

**Objectif :** estimer la probabilité qu'un client déclare un sinistre afin d'aider la décision métier.

**Dataset :** 10 000 clients, 18 variables (profil socio-éco, historique de conduite, véhicule).

---

# Contexte et cible

**Problématique :** identifier les profils à risque de manière fiable et interprétable.

**Variable cible :** `outcome` (1 = sinistre, 0 = aucun).

**Contraintes :** données hétérogènes, valeurs manquantes, variables ordinales et catégorielles.

**Approche :** pipeline complet (nettoyage → encodage → normalisation → modèles → évaluation).

---

# Pipeline de préparation

- **Nettoyage :** suppression des valeurs aberrantes et colonnes non informatives.
- **Imputation :** médiane pour les numériques, mode pour les qualitatives si besoin.
- **Encodage :** mapping manuel pour les ordinales, encodage binaire pour les autres.
- **Normalisation :** StandardScaler pour rendre les variables comparables.
- **Sortie :** données numériques prêtes pour l'apprentissage.

---

# Nettoyage des données

- **Outliers :** suppression des lignes où `speeding_violations > 100` (erreurs de saisie extrêmes).
- **Colonnes retirées :** `id` (identifiant aléatoire) et `postal_code` (faible granularité).
- **Pourquoi :** éviter les biais sur les distances et les coefficients des modèles.

**Résultat :** distribution stabilisée, bruit réduit, variables plus cohérentes.

---

# Valeurs manquantes et encodage

- **Valeurs manquantes :** `credit_score` et `annual_mileage` imputés par la médiane.
- **Raison :** la médiane est robuste aux extrêmes et limite l'introduction de bruit.
- **Encodage ordinal :** `education`, `income`, `driving_experience` mappés pour préserver la hiérarchie.
- **Encodage binaire :** `vehicle_year`, `vehicle_type` via LabelEncoder.

**Impact :** le modèle conserve le sens de progression des variables ordinales.

---

# Normalisation (StandardScaler)

- Les variables ont des échelles très différentes (ex: `credit_score` vs `annual_mileage`).
- Sans normalisation, les variables à grande amplitude dominent l'apprentissage.
- Le centrage-réduction garantit une influence comparable sur la régression logistique et le KNN.

**Résultat :** distances et coefficients plus équilibrés.

---

# Graphiques : distributions

Les histogrammes montrent des formes variées (asymétrie, dispersion), ce qui justifie la normalisation.

![width:900px](histograms.png)

---

# Graphiques : relations entre variables

La scatter matrix met en évidence des séparations partielles, sans frontière nette.
Cela favorise un modèle probabiliste plutôt qu'une règle déterministe.

![width:900px](scatter_matrix.png)

---

# Corrélations clés avec `outcome`

- **driving_experience (-0.50)** : facteur protecteur majeur.
- **age (-0.45)** : moins de sinistres chez les conducteurs plus âgés.
- **income (-0.42)** : revenus plus élevés → moins de sinistres.
- **vehicle_ownership (-0.38), past_accidents (-0.31), credit_score (-0.31)**.
- **vehicle_year (~0.29)** : effet positif sur le risque.

**Lecture :** corrélation ≠ causalité, mais utile pour prioriser les variables.

---

# Modèles et validation

**Modèles comparés (GridSearchCV) :**
- Régression logistique (baseline interprétable)
- KNN (distance entre profils)
- Perceptron (frontière linéaire simple)

**Validation :**
- Split train/test 80/20
- Cross-validation 5 folds
- Score moyen : **84.12% ± 0.78%** (stabilité du modèle)

---

# Comparaison visuelle des modèles

![width:900px](comparaison_modeles.png)

**Lecture :** la régression logistique reste la meilleure, KNN proche, perceptron en retrait.

---

# Comprendre les métriques

- **Accuracy :** part de prédictions correctes, mais sensible au déséquilibre des classes.
- **Précision :** proportion de sinistres réels parmi les clients prédits à risque.
- **Rappel :** proportion de sinistres détectés parmi les sinistres réels.
- **F1-score :** compromis global entre précision et rappel.

**Pourquoi :** une seule métrique ne suffit pas pour mesurer le risque métier.

---

# Résultats du modèle final

**Régression logistique optimisée** (`C=0.1`, `solver=liblinear`).

| Métrique | Valeur |
| --- | --- |
| Accuracy | **83.94%** |
| Précision | **74.88%** |
| Rappel | **73.57%** |
| F1-score | **74.22%** |

**Lecture :** bon compromis entre détection et limitation des fausses alertes.

---

# Matrice de confusion (lecture métier)

- **Vrais négatifs :** 1216
- **Vrais positifs :** 462
- **Faux positifs :** 155
- **Faux négatifs :** 166

**Interprétation :** les faux négatifs sont les sinistres manqués ; le seuil peut être ajusté si l'objectif devient prioritairement le rappel.

---

# Conclusion

La **régression logistique** est retenue pour sa performance et son interprétabilité.

Elle fournit un modèle robuste, explicable aux équipes métier, avec un bon équilibre entre précision et rappel.
