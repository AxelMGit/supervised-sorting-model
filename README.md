# supervised-sorting-model

1. Visualisation par histogramme : peu compréhensible
2. Visualition par plot : plus lisible pour notre usecase

3. Essai de tri par quartiles : pas assez de précision
4. Essai de tri en définissant une valeur maximale manuellement (ex speeding_violation) : OK

5. Compléter les valeurs manquantes :
    - Essai d'utiliser médiane et moyenne pour remplacer les valeurs manquantes: Les données ne sont pas réparties et il y à une sur-représentation artificielle des valeurs à la moyenne.
    - Solution adoptée : utiliser une valeur aléatoire comprise entre le 3e et le 4e quartile pour éviter une sur-représentation artificielle.


6.Transformation des variables qualitatives en variables numériques :
    - Valeurs identifiées :
        Valeurs uniques pour la colonne 'driving_experience':
            ['0-9y' '10-19y' '20-29y' '30y+']

        Valeurs uniques pour la colonne 'education':
            ['high school' 'none' 'university']

        Valeurs uniques pour la colonne 'income':
            ['upper class' 'poverty' 'working class' 'middle class']

        Valeurs uniques pour la colonne 'vehicle_year':
            ['after 2015' 'before 2015']

        Valeurs uniques pour la colonne 'vehicle_type':
            ['sedan' 'sports car']


    - Solution adoptée : df.map