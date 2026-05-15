import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas.plotting import scatter_matrix


def main():
    # --- 3. Importation des données ---
    try:
        df = pd.read_csv('car_insurance.csv')

        print("--- 3. Importation des données ---")
        print("Aperçu des 5 premières lignes :")
        print(df.head())
        print("\nInformations sur les colonnes :")
        df.info()

        # --- 4. Examen des données ---
        print("\n--- 4. Examen des données ---")
        print("Taille du jeu de données :", df.shape)
        
        print("\nNombre de valeurs manquantes par colonne :")
        print(df.isnull().sum())

        # Visualisation des histogrammes pour les variables numériques existantes
        # Avant transformations pour voir l'état initial
        print("\nGénération des histogrammes (état initial)...")
        df.hist(figsize=(15, 10))
        plt.tight_layout()
        # plt.show() # Commenté pour éviter de bloquer en environnement non-GUI

        # --- 5. Préparation des données ---
        print("\n--- 5. Préparation des données ---")

        # a. Suppression des colonnes inutiles
        # 'id' est un identifiant unique, 'postal_code' a peu de valeurs et est souvent exclu dans un premier temps
        df.drop(columns=['id', 'postal_code'], inplace=True)
        print("Colonnes 'id' et 'postal_code' supprimées.")

        # b. Gestion des données aberrantes
        # On a observé des valeurs extrêmes (>100) pour speeding_violations (ex: 41056)
        outliers_count = df[df['speeding_violations'] > 100].shape[0]
        if outliers_count > 0:
            print(f"Suppression de {outliers_count} lignes avec des valeurs aberrantes dans 'speeding_violations'.")
            df = df[df['speeding_violations'] <= 100]

        # c. Gestion des valeurs manquantes
        # Pour les variables numériques : remplacement par la médiane
        for col in ['credit_score', 'annual_mileage']:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Valeurs manquantes dans '{col}' remplacées par la médiane : {median_val}")

        # d. Transformation des variables qualitatives
        # Variables ordinales : mapping manuel pour respecter l'ordre
        print("Transformation des variables qualitatives...")
        df['driving_experience'] = df['driving_experience'].map({'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3})
        df['education'] = df['education'].map({'none': 0, 'high school': 1, 'university': 2})
        df['income'] = df['income'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})

        # Variables binaires : utilisation de LabelEncoder comme suggéré
        le = LabelEncoder()
        df['vehicle_year'] = le.fit_transform(df['vehicle_year']) # before 2015 -> 0, after 2015 -> 1
        df['vehicle_type'] = le.fit_transform(df['vehicle_type']) # sedan -> 0, sports car -> 1

        # e. Séparation des caractéristiques et de la cible
        X = df.drop(columns=['outcome'])
        y = df['outcome'].astype(int)

        # f. Normalisation des données
        print("Normalisation des données avec StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Conversion en DataFrame pour garder les noms de colonnes
        X_final = pd.DataFrame(X_scaled, columns=X.columns)
        # On rajoute la cible pour l'analyse de corrélation
        df_final = X_final.copy()
        df_final['outcome'] = y.values

        print("\nPréparation terminée.")
        print("Nouvelle taille de X :", X_final.shape)

        # --- 6. Recherche de corrélations ---
        print("\n--- 6. Recherche de corrélations ---")
        
        # Calcul de la matrice de corrélation
        corr_matrix = df_final.corr()

        # Affichage de la corrélation de chaque variable avec la sortie 'outcome'
        print("Corrélation des variables avec 'outcome' (triée) :")
        target_corr = corr_matrix['outcome'].sort_values(ascending=False)
        print(target_corr)

        # Identification des variables les plus prometteuses (valeur absolue > 0.2 par exemple)
        threshold = 0.2
        promising_features = target_corr[abs(target_corr) > threshold].index.tolist()
        promising_features.remove('outcome')
        print(f"\nVariables les plus prometteuses (abs(corr) > {threshold}) : {promising_features}")

        # Visualisation avec scatter_matrix (sur un sous-ensemble pour la lisibilité)
        print("\nGénération de la scatter_matrix pour les variables prometteuses...")
        # scatter_matrix(df_final[promising_features], figsize=(12, 8), diagonal='kde')
        # plt.tight_layout()
        # plt.show()

        # Visualisation finale après préparation
        print("\nGénération des histogrammes après préparation...")
        # X_final.hist(figsize=(15, 10))
        # plt.tight_layout()
        # plt.show()

    except FileNotFoundError:
        print("Erreur : Fichier 'car_insurance.csv' non trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == '__main__':
    main()
