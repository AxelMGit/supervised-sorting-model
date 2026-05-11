import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#  1   age                  10000 non-null  int64  
#  2   gender               10000 non-null  int64  
#  3   driving_experience   10000 non-null  object 
#  4   education            10000 non-null  object 
#  5   income               10000 non-null  object 
#  6   credit_score         9018 non-null   float64
#  7   vehicle_ownership    10000 non-null  float64
#  8   vehicle_year         10000 non-null  object 
#  9   married              10000 non-null  float64
#  10  children             10000 non-null  float64
#  11  postal_code          10000 non-null  int64  
#  12  annual_mileage       9043 non-null   float64
#  13  vehicle_type         10000 non-null  object 
#  14  speeding_violations  10000 non-null  int64  
#  15  duis                 10000 non-null  int64  
#  16  past_accidents       10000 non-null  int64  
#  17  outcome              10000 non-null  float64

def main():
    # Lire le fichier CSV dans un DataFrame pandas.
    try:
        df = pd.read_csv('car_insurance.csv')

        # Afficher les premières lignes pour vérifier que le chargement est correct
        print("Aperçu des 5 premières lignes du DataFrame :\n")
        print(df.head())

        # Afficher les informations sur les variables et leurs types
        print("\n\nInformations sur les colonnes :")
        df.info()

        # Detection des valeurs manquantes
        print("\n\nNombre de valeurs manquantes par colonne :")
        print(df.isnull().sum())


        #for every column with data type object, print all the unique values
        for col in df.select_dtypes(include=['object']).columns:
            print(f"\n\nValeurs uniques pour la colonne '{col}':")
            print(df[col].unique())

        # remplacer les valeur des colonnes non numérique par des valeurs numériques
        #Valeurs uniques pour la colonne 'driving_experience':
        # ['0-9y' '10-19y' '20-29y' '30y+']
        # Valeurs uniques pour la colonne 'education':
        # ['high school' 'none' 'university']
        # Valeurs uniques pour la colonne 'income':
        # ['upper class' 'poverty' 'working class' 'middle class']
        # Valeurs uniques pour la colonne 'vehicle_year':
        # ['after 2015' 'before 2015']
        # Valeurs uniques pour la colonne 'vehicle_type':
        # ['sedan' 'sports car']

        df['driving_experience'] = df['driving_experience'].map({'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3})
        df['education'] = df['education'].map({'none': 0, 'high school': 1, 'university': 2})
        df['income'] = df['income'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})
        df['vehicle_year'] = df['vehicle_year'].map({'before 2015': 0, 'after 2015': 1})
        df['vehicle_type'] = df['vehicle_type'].map({'sedan': 0, 'sports car': 1})
        


        



        print("\n\nGénération des histogrammes...")

        # retirer speeding violation si la valeur depasse 500
        numeric_cols = ['speeding_violations']    
        for col in numeric_cols:
            df = df[df[col] <= 100]

        # pour annuel et credit score, remplacer les valeurs manquantes par la moyenne
        # df['annual_mileage'].fillna(df['annual_mileage'].mean(), inplace=True)
        # df['credit_score'].fillna(df['credit_score'].mean(), inplace=True)

        #  annuel et credit score, remplacer les valeurs manquantes par une valeur aléatoire entre le premier quartile et le troisième quartile
        for col in ['annual_mileage', 'credit_score']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            df[col].fillna(df[col].apply(lambda x: np.random.uniform(q1, q3) if pd.isnull(x) else x), inplace=True)


        # Génération des multiples boîtes à moustaches
        df.hist(
            figsize=(18, 12))
        
        plt.tight_layout()
        plt.show()

        df.plot(
            kind='box',         
            subplots=True,      
            layout=(-1, 4),        
            figsize=(18, 12),     
            sharex=False,         
            sharey=False         
        )

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Erreur : Fichier non trouvé")

if __name__ == '__main__':
    main()
