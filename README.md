# Projet Science des Données : Classification Supervisée

Ce projet vise à prédire si un client fera une demande d'indemnisation auprès de son assurance automobile (`outcome`) en utilisant un jeu de données de 10 000 entrées.

## Choix Techniques et Préparation des Données (Parties 1 à 5)

### 1. Analyse et Nettoyage des Outliers
*   **Observation :** La colonne `speeding_violations` présentait des valeurs aberrantes extrêmes (ex: 41056), probablement dues à des erreurs de saisie.
*   **Décision :** Suppression des lignes où `speeding_violations > 100`.
*   **Justification :** Ces valeurs faussent la distribution et les calculs de distance pour les modèles de classification.

### 2. Gestion des Variables Inutiles
*   **Décision :** Suppression des colonnes `id` et `postal_code`.
*   **Justification :** L'`id` est un identifiant aléatoire sans pouvoir prédictif. Le `postal_code` bien que numérique, représente des zones géographiques ; avec seulement 4 codes postaux distincts, il a été jugé moins pertinent que d'autres variables socio-économiques pour cette phase initiale.

### 3. Traitement des Valeurs Manquantes
*   **Observation :** `credit_score` et `annual_mileage` ont environ 10% de valeurs manquantes.
*   **Décision :** Imputation par la **médiane**.
*   **Justification :** Contrairement à l'approche précédente (aléatoire entre Q3 et Q4), la médiane est une mesure de tendance centrale robuste qui minimise l'influence des valeurs extrêmes. Bien qu'elle crée un pic dans la distribution, elle reste le standard pour éviter d'introduire du bruit aléatoire injustifié dans un modèle de classification.

### 4. Encodage des Variables Qualitatives
*   **Variables Ordinales (`driving_experience`, `education`, `income`) :**
    *   **Méthode :** Mapping manuel (ex: `none: 0`, `high school: 1`, `university: 2`).
    *   **Justification :** Il existe une hiérarchie logique. Un `LabelEncoder` classique aurait pu assigner des valeurs arbitraires (ex: high school=0, none=1), perdant ainsi l'information de progression.
*   **Variables Binaires (`vehicle_year`, `vehicle_type`) :**
    *   **Méthode :** `LabelEncoder`.
    *   **Justification :** Transformation simple en 0/1 car il n'y a pas d'ordre intrinsèque complexe.

### 5. Normalisation
*   **Méthode :** `StandardScaler`.
*   **Justification :** Les variables ont des échelles très différentes (ex: `credit_score` entre 0 et 1 vs `annual_mileage` en milliers). La normalisation est cruciale pour que les algorithmes (comme la Régression Logistique ou les K-NN) ne donnent pas une importance disproportionnée aux variables avec de grandes valeurs numériques.

## Structure du Projet
- `main.py` : Script principal contenant le pipeline de préparation.
- `car_insurance.csv` : Jeu de données source.
- `instructions.pdf` : Cahier des charges du projet.
- `car_insurance_desc.pdf` : Description des variables.
