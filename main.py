import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

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
            "name": "Logit",
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
    
    best_models_info = {}
    for config in configs:
        print(f"Optimisation de {config['name']}...")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models_info[config['name']] = {
            "model": grid.best_estimator_,
            "params": grid.best_params_,
            "cv_score": grid.best_score_
        }
    return best_models_info

def generer_graphiques_comparaison(df_results):
    """Génère et sauvegarde un graphique de comparaison des modèles"""
    print("\n--- Génération du graphique de comparaison ---")
    models = df_results['Modèle']
    accuracy = df_results['Accuracy'].str.rstrip('%').astype(float)
    f1_score = df_results['F1-Score'].str.rstrip('%').astype(float)
    
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#3498db')
    ax.bar(x + width/2, f1_score, width, label='F1-Score', color='#e74c3c')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Comparaison des Performances Finales')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig('comparaison_modeles.png')
    print("Graphique sauvegardé : comparaison_modeles.png")

def sauvegarder_modele(model, filename='best_model.pkl'):
    """12. Sauvegarde du modèle entraîné"""
    print(f"\n--- 12. Sauvegarde du modèle dans {filename} ---")
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print("Modèle sauvegardé avec succès.")

def charger_et_tester_modele(X_test, filename='best_model.pkl'):
    """12. Chargement et vérification du modèle"""
    print(f"--- Vérification du modèle chargé ---")
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    predictions = loaded_model.predict(X_test[:5])
    print(f"Prédictions (5 premiers) : {predictions}")

def main():
    try:
        # Pipeline initial
        df = load_data('car_insurance.csv')
        examine_data(df)
        X, y = preprocess_data(df)
        _ = analyze_correlations(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )
        
        # 11. Optimisation et Comparaison
        best_configs = optimiser_tous_les_modeles(X_train, y_train)
        
        final_data = []
        best_acc = 0
        winner_model = None
        winner_name = ""

        for name, info in best_configs.items():
            model = info["model"]
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            final_data.append({
                "Modèle": name,
                "Accuracy": f"{acc:.2%}",
                "F1-Score": f"{f1:.2%}"
            })
            
            if acc > best_acc:
                best_acc = acc
                winner_model = model
                winner_name = name

        df_results = pd.DataFrame(final_data)
        print("\n" + df_results.to_string(index=False))
        
        generer_graphiques_comparaison(df_results)
        
        # 12. Sauvegarde du gagnant
        print(f"\n🏆 Gagnant sélectionné : {winner_name}")
        sauvegarder_modele(winner_model)
        charger_et_tester_modele(X_test)
        
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == '__main__':
    main()
