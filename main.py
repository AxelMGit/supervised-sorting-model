import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def load_data(file_path):
    """3. Importation des données"""
    print(f"--- 3. Importation de {file_path} ---")
    df = pd.read_csv(file_path)
    print(f"OK : {len(df)} lignes chargées.")
    return df

def examine_data(df):
    """4. Examen des données"""
    print("\n--- 4. Examen des données ---")
    print(f"Valeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

def preprocess_data(df):
    """5. Préparation des données"""
    print("\n--- 5. Préparation des données ---")
    df = df.drop(columns=['id', 'postal_code'])
    df = df[df['speeding_violations'] <= 100]
    for col in ['credit_score', 'annual_mileage']:
        df[col] = df[col].fillna(df[col].median())
    
    df['driving_experience'] = df['driving_experience'].map({'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3})
    df['education'] = df['education'].map({'none': 0, 'high school': 1, 'university': 2})
    df['income'] = df['income'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})
    
    le = LabelEncoder()
    df['vehicle_year'] = le.fit_transform(df['vehicle_year'])
    df['vehicle_type'] = le.fit_transform(df['vehicle_type'])
    
    X = df.drop(columns=['outcome'])
    y = df['outcome'].astype(int)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"OK : Données nettoyées et normalisées. Taille : {X_scaled.shape}")
    return X_scaled, y

def analyze_correlations(X, y):
    """6. Recherche de corrélations"""
    print("\n--- 6. Recherche de corrélations ---")
    df_temp = X.copy()
    df_temp['outcome'] = y.values
    corrs = df_temp.corr()['outcome'].sort_values(ascending=False)
    promising = corrs[abs(corrs) > 0.2].index.tolist()
    promising.remove('outcome')
    print(f"Variables prometteuses (>0.2) : {promising}")
    return promising

def optimiser_tous_les_modeles(X_train, y_train):
    """11. Optimisation et recherche des meilleurs paramètres"""
    print("\n--- 11. Optimisation des modèles (GridSearchCV) ---")
    
    configs = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        },
        {
            "name": "KNN",
            "model": KNeighborsClassifier(),
            "params": {
                'n_neighbors': [5, 11, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        {
            "name": "Perceptron",
            "model": Perceptron(random_state=42),
            "params": {
                'alpha': [0.0001, 0.001, 0.01],
                'penalty': ['l2', 'l1', 'elasticnet']
            }
        }
    ]
    
    results_summary = []
    best_models = {}
    
    for config in configs:
        print(f"Analyse de {config['name']}...")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_models[config['name']] = {
            "model": grid.best_estimator_,
            "params": grid.best_params_,
            "cv_score": grid.best_score_
        }
        
    return best_models

def main():
    try:
        # Pipeline
        df = load_data('car_insurance.csv')
        examine_data(df)
        X, y = preprocess_data(df)
        _ = analyze_correlations(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )
        
        # Optimisation
        best_configs = optimiser_tous_les_modeles(X_train, y_train)
        
        # Comparaison et affichage final
        print("\n" + "="*60)
        print("         COMPARAISON FINALE DES MODÈLES OPTIMISÉS")
        print("="*60)
        
        final_data = []
        for name, info in best_configs.items():
            model = info["model"]
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            final_data.append({
                "Modèle": name,
                "Best Params": str(info["params"]),
                "CV Accuracy": f"{info['cv_score']:.2%}",
                "Test Accuracy": f"{acc:.2%}",
                "Test F1-Score": f"{f1:.2%}"
            })
            
        df_results = pd.DataFrame(final_data)
        print(df_results.to_string(index=False))
        
        # Conclusion
        winner_name = df_results.iloc[df_results['Test Accuracy'].str.rstrip('%').astype(float).idxmax()]['Modèle']
        print("\n" + "="*60)
        print(f"🏆 GAGNANT : {winner_name}")
        print("="*60)
        
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == '__main__':
    main()
