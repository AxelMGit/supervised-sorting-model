import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas.plotting import scatter_matrix

def load_data(file_path):
    """3. Importation des données"""
    print(f"--- 3. Importation de {file_path} ---")
    df = pd.read_csv(file_path)
    print(df.head())
    return df

def examine_data(df):
    """4. Examen des données"""
    print("\n--- 4. Examen des données ---")
    df.info()
    print("\nValeurs manquantes :\n", df.isnull().sum())
    # df.hist(figsize=(15, 10))
    # plt.show()

def preprocess_data(df):
    """5. Préparation des données"""
    print("\n--- 5. Préparation des données ---")
    
    # a. Nettoyage
    df = df.drop(columns=['id', 'postal_code'])
    df = df[df['speeding_violations'] <= 100]
    
    # b. Imputation
    for col in ['credit_score', 'annual_mileage']:
        df[col] = df[col].fillna(df[col].median())
    
    # c. Encodage Qualitatif
    df['driving_experience'] = df['driving_experience'].map({'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3})
    df['education'] = df['education'].map({'none': 0, 'high school': 1, 'university': 2})
    df['income'] = df['income'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})
    
    le = LabelEncoder()
    df['vehicle_year'] = le.fit_transform(df['vehicle_year'])
    df['vehicle_type'] = le.fit_transform(df['vehicle_type'])
    
    # d. Séparation X, y
    X = df.drop(columns=['outcome'])
    y = df['outcome'].astype(int)
    
    # e. Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"Préparation terminée. Taille : {X_scaled.shape}")
    return X_scaled, y

def analyze_correlations(X, y):
    """6. Recherche de corrélations"""
    print("\n--- 6. Recherche de corrélations ---")
    df_temp = X.copy()
    df_temp['outcome'] = y.values
    
    corrs = df_temp.corr()['outcome'].sort_values(ascending=False)
    print("Corrélations avec outcome :\n", corrs)
    
    promising = corrs[abs(corrs) > 0.2].index.tolist()
    promising.remove('outcome')
    print(f"\nVariables prometteuses (>0.2) : {promising}")
    return promising

def main():
    try:
        # Pipeline principal
        df = load_data('car_insurance.csv')
        examine_data(df)
        X, y = preprocess_data(df)
        promising_features = analyze_correlations(X, y)
        
        # Prêt pour la partie 7...
        print("\nPrêt pour l'étape 7 (Extraction des jeux).")
        
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == '__main__':
    main()
