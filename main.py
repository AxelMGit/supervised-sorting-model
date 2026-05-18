import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pandas.plotting import scatter_matrix

def load_data(file_path):
    """3. Importation des données"""
    print(f"--- 3. Importation de {file_path} ---")
    df = pd.read_csv(file_path)
    print(df.head())
    print("\nObjectif : Prédire si un client fera une demande d'indemnisation (outcome=1) ou non (outcome=0).")
    return df

def examine_data(df):
    """4. Examen des données"""
    print("\n--- 4. Examen des données ---")
    print(f"Taille du jeu de données : {df.shape}")
    print("\nTypes des données :\n", df.dtypes)
    
    # Visualisation des histogrammes (demandé section 4)
    print("\nGénération des histogrammes des variables numériques...")
    df.hist(figsize=(15, 10))
    plt.tight_layout()
    plt.savefig('histograms.png')
    print("Graphique sauvegardé : histograms.png")

def preprocess_data(df):
    """5. Préparation des données"""
    print("\n--- 5. Préparation des données ---")
    
    # Suppression des colonnes inutiles (ID et Postal Code)
    df = df.drop(columns=['id', 'postal_code'])
    
    # Traitement des valeurs aberrantes (Outliers) identifiées dans speeding_violations
    df = df[df['speeding_violations'] <= 100]
    
    # Gestion des valeurs manquantes (Section 5 du PDF)
    # Pour les numériques : Médiane
    for col in ['credit_score', 'annual_mileage']:
        df[col] = df[col].fillna(df[col].median())
    
    # Pour les qualitatives (si existantes) : Valeur la plus fréquente (Mode)
    qualitative_cols = df.select_dtypes(include=['object']).columns
    for col in qualitative_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Transformation des variables qualitatives (LabelEncoder ou Mapping manuel pour l'ordinal)
    # Note : Le mapping manuel est utilisé pour préserver la hiérarchie (poverty < working class < ...)
    df['driving_experience'] = df['driving_experience'].map({'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3})
    df['education'] = df['education'].map({'none': 0, 'high school': 1, 'university': 2})
    df['income'] = df['income'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})
    
    le = LabelEncoder()
    df['vehicle_year'] = le.fit_transform(df['vehicle_year'])
    df['vehicle_type'] = le.fit_transform(df['vehicle_type'])
    
    X = df.drop(columns=['outcome'])
    y = df['outcome'].astype(int)
    
    # Normalisation (StandardScaler)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
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
    
    # Scatter Matrix (demandé section 6)
    print(f"\nGénération de la Scatter Matrix pour : {promising}")
    scatter_matrix(df_temp[promising + ['outcome']], figsize=(15, 15), alpha=0.3)
    plt.savefig('scatter_matrix.png')
    print("Graphique sauvegardé : scatter_matrix.png")
    
    return promising

def main():
    try:
        # 3 & 4. Import et Examen
        df = load_data('car_insurance.csv')
        examine_data(df)
        
        # 5. Préparation
        X, y = preprocess_data(df)
        
        # 6. Corrélations
        promising_features = analyze_correlations(X, y)
        
        # 7. Split (test_size=0.2 comme standard)
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )
        
        # 8, 10 & 11. Apprentissage et Optimisation
        print("\n--- 8, 10, 11. Apprentissage et Optimisation ---")
        
        # Validation croisée explicite avec KFold (Section 10)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        configs = [
            {"name": "Logit", "model": LogisticRegression(max_iter=1000), "params": {'C': [0.1, 1, 10]}},
            {"name": "KNN", "model": KNeighborsClassifier(), "params": {'n_neighbors': [5, 11, 21]}},
            {"name": "Perceptron", "model": Perceptron(), "params": {'alpha': [0.0001, 0.01]}}
        ]
        
        best_overall_model = None
        best_overall_acc = 0
        
        print(f"{'Modèle':<20} | {'CV Accuracy':<15}")
        print("-" * 40)
        
        for config in configs:
            grid = GridSearchCV(config['model'], config['params'], cv=kf, scoring='accuracy')
            grid.fit(X_train, y_train)
            acc_cv = grid.best_score_
            print(f"{config['name']:<20} | {acc_cv:.2%}")
            
            if acc_cv > best_overall_acc:
                best_overall_acc = acc_cv
                best_overall_model = grid.best_estimator_
        
        # 9. Évaluation (Section 9 du PDF)
        print("\n--- 9. Évaluation du meilleur modèle sur le jeu de test ---")
        y_pred = best_overall_model.predict(X_test)
        
        # Boucle d'affichage demandée par le PDF (sur les 10 premiers)
        print("\nComparaison (10 premiers échantillons) :")
        for i in range(10):
            print(f"Echantillon {i} : Prédit={y_pred[i]}, Réel={y_test[i]}")
            
        print(f"\nAccuracy Finale  : {accuracy_score(y_test, y_pred):.2%}")
        print(f"F1-Score Final   : {f1_score(y_test, y_pred):.2%}")
        print("\nMatrice de Confusion :\n", confusion_matrix(y_test, y_pred))

        # 12. Sauvegarde (Section 12)
        print("\n--- 12. Sauvegarde du modèle ---")
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_overall_model, f)
        print("Modèle sauvegardé dans best_model.pkl")

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == '__main__':
    main()
