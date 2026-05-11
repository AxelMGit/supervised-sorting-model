import pandas as pd

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
    except FileNotFoundError:
        print("Erreur : Fichier non trouvé")

if __name__ == '__main__':
    main()
