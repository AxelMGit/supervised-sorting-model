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

## 6. Recherche de corrélations

L'analyse de corrélation (Partie 6) a permis d'identifier les variables ayant le plus d'influence sur le risque de sinistre (`outcome`) :

*   **Variables les plus corrélées (en valeur absolue) :**
    1.  `driving_experience` (-0.50)
    2.  `age` (-0.45)
    3.  `income` (-0.42)
    4.  `vehicle_ownership` (-0.38)
    5.  `past_accidents` (-0.31)
    6.  `credit_score` (-0.31)
*   **Signification :**
    *   Les corrélations négatives fortes (ex: `driving_experience`) indiquent que plus l'expérience de conduite ou l'âge augmente, plus la probabilité de faire une demande d'indemnisation diminue.
    *   `vehicle_year` présente une corrélation positive (~0.29), suggérant une influence de l'âge du véhicule sur le risque.
*   **Variables pertinentes :** Les variables avec une corrélation absolue > 0.2 ont été sélectionnées comme "prometteuses" pour l'entraînement des modèles.

## 8. Entraînement du modèle (Régression Logistique)

Un modèle de **Régression Logistique** a été entraîné sur le jeu d'apprentissage. Voici les réponses aux questions théoriques :

*   **Hypothèse sur le logit :** On suppose que le logarithme du rapport des vraisemblances (logit) est une fonction linéaire des variables d'entrée :  
    $log(\frac{p}{1-p}) = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$
*   **Minimisation de la fonction de coût :** L'algorithme cherche à minimiser la fonction de perte (Log-Loss / Cross-Entropy) en utilisant des techniques d'optimisation numérique comme la **Descente de Gradient** ou des solveurs plus complexes (ex: 'lbfgs', 'liblinear').
*   **Paramètres calculés :** Pendant la phase d'apprentissage, l'algorithme calcule les **coefficients** ($\beta_i$) associés à chaque variable ainsi que l'**ordonnée à l'origine** (intercept, $\beta_0$).

## 9. Évaluation du modèle

L'évaluation du modèle est cruciale pour comprendre ses performances réelles sur des données non vues. Nous utilisons plusieurs métriques issues de la **matrice de confusion** :

### Métriques expliquées :
1.  **Accuracy (Précision globale) :** C'est le taux de bonnes prédictions totales (Vrais Positifs + Vrais Négatifs) par rapport au nombre total d'échantillons. Elle donne une idée générale de la performance mais peut être trompeuse si les classes sont déséquilibrées.
2.  **Precision (Précision) :** Répond à la question : "Sur tous ceux que le modèle a prédit comme 'Positif' (indemnisation), combien l'étaient réellement ?". C'est crucial pour éviter les "Faux Positifs" (prédire un sinistre là où il n'y en a pas).
3.  **Recall (Rappel / Sensibilité) :** Répond à la question : "Sur tous ceux qui étaient réellement 'Positif', combien le modèle a-t-il réussi à en détecter ?". C'est crucial pour éviter les "Faux Négatifs" (manquer un sinistre réel).
4.  **F1-Score :** C'est la moyenne harmonique de la Précision et du Rappel. Il permet d'avoir une mesure unique qui équilibre les deux, ce qui est particulièrement utile quand on veut un bon compromis entre détection exhaustive et fiabilité des prédictions.

### Résultats du modèle actuel (Régression Logistique) :
Suite à l'entraînement sur 80% des données, le modèle obtient les performances suivantes sur le jeu de test :

| Métrique | Valeur |
| :--- | :--- |
| **Accuracy** | **83.94%** |
| **Precision** | **74.88%** |
| **Recall** | **73.57%** |
| **F1-Score** | **74.22%** |

### Matrice de Confusion :
La matrice de confusion permet de visualiser les erreurs du modèle :
*   **Vrais Négatifs (1216) :** Clients n'ayant pas fait de demande, bien prédits.
*   **Vrais Positifs (462) :** Clients ayant fait une demande, bien prédits.
*   **Faux Positifs (155) :** Clients n'ayant pas fait de demande, mais prédits comme en ayant fait une (Erreur de Type I).
*   **Faux Negatifs (166) :** Clients ayant fait une demande, mais prédits comme n'en ayant pas fait une (Erreur de Type II).

## Structure du Projet
- `main.py` : Script principal contenant le pipeline de préparation.
- `car_insurance.csv` : Jeu de données source.
- `instructions.pdf` : Cahier des charges du projet.
- `car_insurance_desc.pdf` : Description des variables.
