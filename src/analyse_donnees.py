### 1. Importation des bibliothèques nécessaires
from pathlib import Path

import matplotlib as plt
import numpy as np
import pandas as pd

### 2. Définition des chemins du projet

#on repère la racine du projet en remontant de deux dossiers
RACINE_PROJET = Path(__file__).resolve().parent.parent

#on définit le chemin du fichier de données et des dossiers de sortie
CHEMIN_CSV = RACINE_PROJET / "donnees" / "donnees_ia.csv"
DOSSIER_TABLEAUX = RACINE_PROJET / "resultats" / "tableaux"
DOSSIER_GRAPHIQUES = RACINE_PROJET / "resultats" / "graphiques"

#on crée les dossiers de sortie s'ils n'existent pas déjà
DOSSIER_TABLEAUX.mkdir(
    parents=True,
    exist_ok=True
)

DOSSIER_GRAPHIQUES.mkdir(
    parents=True,
    exist_ok=True
)

### 3. Chargement de la base de données

#On vérifie que le fichier CSV existe avant de le lire
assert CHEMIN_CSV.exists(), (
    f"Le fichier CSV est introuvable: {CHEMIN_CSV}"
)

#on charge le fichier:
donnees = pd.read_csv(CHEMIN_CSV)

### 4. Inspection initiale de la base

print("\nPremières observations :")
print(donnees.head())

print("\nDimensions de la base :", donnees.shape)

print("\nInformations générales :")
donnees.info()

assert not donnees.empty
assert donnees["id"].is_unique
assert not donnees.isna().any().any()

### 5. Définition des variables et des catégories

#définition des variables
variables_qualitatives = [
    "session",
    "utilisation_recente",
    "usage_principal",
    "outil_principal",
    "verification",
    "utilite_percue"
]

variables_quantitatives = [
    "jours_utilisation",
    "duree_minutes"
]

#définition des ordres
ordre_session = ["1re", "2e", "3e", "4e", "5e ou plus"]
ordre_utilisation = ["Non", "Oui"]
ordre_verification = [
    "Je n'utilise pas l'IA",
    "Jamais",
    "Rarement",
    "Parfois",
    "Souvent",
    "Toujours"
]
ordre_utilite = [
    "Je n'utilise pas l'IA",
    "Pas du tout utile",
    "Peu utile",
    "Moyennement utile",
    "Assez utile",
    "Très utile"
]

#conversion en variables catégorielles
donnees["session"] = pd.Categorical(
    donnees["session"],
    categories=ordre_session,
    ordered=True
)

donnees["utilisation_recente"] = pd.Categorical(
    donnees["utilisation_recente"],
    categories=ordre_utilisation,
    ordered=False
)

donnees["verification"] = pd.Categorical(
    donnees["verification"],
    categories=ordre_verification,
    ordered=True
)

donnees["utilite_percue"] = pd.Categorical(
    donnees["utilite_percue"],
    categories=ordre_utilite,
    ordered=True
)

#vérifications après ces conversions:
assert set(variables_qualitatives).issubset(donnees.columns)
assert set(variables_quantitatives).issubset(donnees.columns)

print("\nTypes après conversion :")
print(donnees.dtypes)

print("\nCatégories de verification :")
print(donnees["verification"].cat.categories)

print("\nCatégories de utilite_percue :")
print(donnees["utilite_percue"].cat.categories)

assert donnees["utilisation_recente"].notna().all()
assert donnees["session"].notna().all()
assert donnees["verification"].notna().all()
assert donnees["utilite_percue"].notna().all()

### 6. Tableaux de fréquences des variables qualitatives

# on procède par la création d'une fonction, par respect de la norme DRY (Don't repeat yourself)
def creer_tableau_frequences(variable, cumulatif=False):
    """
    Construit un tableau de fréquences pour une variable qualitative

    Paramètres
    ----------
    variable: str
        Nom de la colonne de la base à analyser.
    cumulatif: bool
        Ajoute les effectifs et pourcentages cumulés
        lorsque la variable est ordinale.
    
    Retour
    ------
    pandas.DataFrame
        Tableau contenant les effectifs et les fréquences
    """

    serie = donnees[variable]

    # On veut respecter l'ordre défini pour les variables catégorielles.
    if isinstance(serie.dtype, pd.CategoricalDtype):
        effectifs = serie.value_counts(
            sort=False,
            dropna=False
        )
    else:
        # Pour les variables nominales, classement par effectif décroissant.
        effectifs = serie.value_counts(
            sort=True,
            dropna=False
        )
    
    total = effectifs.sum()

    tableau = pd.DataFrame({
        "Effectif": effectifs,
        "Fréquence relative": effectifs / total
    })

    # Ajout des fréquences cumulées pour les variables ordinales:
    if cumulatif:
        tableau["Effectif cumulé"] = tableau["Effectif"].cumsum()
        tableau["Fréquence cumulée"] = tableau["Fréquence relative"].cumsum()
    
    # Nom de l'index.
    tableau.index.name = "Modalité"

    # Arrondissement uniquement pour la présentation.
    tableau["Fréquence relative"] = tableau["Fréquence relative"].round(4)

    if cumulatif:
        tableau["Fréquence cumulée"] = tableau["Fréquence cumulée"].round(4)

    return tableau

