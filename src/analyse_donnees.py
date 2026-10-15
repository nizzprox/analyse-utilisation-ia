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

# Ensemble de l'échantillon.
donnees_completes = donnees.copy()


# IMPORTANT: donnees_utilisateurs représente le sous-échantillon d'analyse des habitudes d'utilisation.
# Les valeurs correspondant à un non-utilisateur ne sont donc pas prises en compte.

# Étudiants ayant utilisé l'IA récemment.
donnees_utilisateurs = donnees.loc[
    donnees["utilisation_recente"] == "Oui"
].copy()

print("Nombre total d'étudiants:", len(donnees_completes))
print("Nombre d'utilisateurs récents:", len(donnees_utilisateurs))

### 6. Tableaux de fréquences des variables qualitatives

# on procède par la création d'une fonction, par respect de la norme DRY (Don't repeat yourself)
def creer_tableau_frequences(
    base,
    variable,
    cumulatif=False,
    ordonner_valeurs=False
):

    """
    Construit un tableau de fréquences pour une variable.

    Paramètres
    ----------
    base: object
        Indique quelle base de données utiliser.
    variable: str
        Nom de la colonne de la base à analyser.
    cumulatif: bool
        Ajoute les effectifs et pourcentages cumulés
        lorsque la variable est ordinale.
    ordonner_valeurs: bool
        Ordonne les modalités ou valeurs selon leur
        index avant de construire le tableau.

    Retour
    ------
    pandas.DataFrame
        Tableau contenant les effectifs et les fréquences
    """

    serie = base[variable]

    # On veut respecter l'ordre défini pour les variables catégorielles.
    if isinstance(serie.dtype, pd.CategoricalDtype):
        effectifs = serie.value_counts(
            sort=False,
            dropna=False
        )
    
    elif ordonner_valeurs:
        effectifs = serie.value_counts(
            sort=False,
            dropna=False
        ).sort_index()
    
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

    # Création de la ligne Total.
    ligne_total = {
        "Effectif": tableau["Effectif"].sum(),
        "Fréquence relative" : tableau["Fréquence relative"].sum()
    }

    if cumulatif:
        ligne_total["Effectif cumulé"] = tableau["Effectif"].sum()
        ligne_total["Fréquence cumulée"] = tableau["Fréquence relative"].sum()
    
    # Ajout de la ligne Total au tableau.
    tableau = pd.concat([
        tableau,
        pd.DataFrame(
            [ligne_total],
            index=["Total"]
        )
    ])
# Arrondissement uniquement pour la présentation.
    tableau["Fréquence relative"] = tableau["Fréquence relative"].round(4)

    if cumulatif:
        tableau["Fréquence cumulée"] = tableau["Fréquence cumulée"].round(4)

    return tableau


# Tableaux portant sur l'ensemble de l'échantillon:

tableau_session = creer_tableau_frequences(
    donnees_completes,
    "session",
    cumulatif=True  # Il s'agit d'une variable ordinale: les fréquences cumulés sont pertinentes.
)

# Nominale, fréquences cumulatives non pertinentes.
tableau_utilisation = creer_tableau_frequences(donnees_completes, "utilisation_recente")  


#Tableaux portant uniquement sur les utilisateurs récents:

# Variables ordinales: les fréquences cumulées sont pertinentes.
tableau_verification = creer_tableau_frequences(
    donnees_utilisateurs,
    "verification",
    cumulatif=True
)

tableau_utilite = creer_tableau_frequences(
    donnees_utilisateurs,
    "utilite_percue",
    cumulatif=True
)

# Variables nominales: les fréquences cumulées ne sont pas pertinentes.

tableau_usage = creer_tableau_frequences(donnees_utilisateurs, "usage_principal")
tableau_outil = creer_tableau_frequences(donnees_utilisateurs, "outil_principal")

# Affichage des résultats:
tableaux_qualitatifs = {
    "session": (tableau_session, donnees_completes),
    "utilisation_recente": (tableau_utilisation, donnees_completes), 
    "usage_principal": (tableau_usage, donnees_utilisateurs),
    "outil_principal": (tableau_outil, donnees_utilisateurs),
    "verification": (tableau_verification, donnees_utilisateurs),
    "utilite_percue": (tableau_utilite, donnees_utilisateurs)
}

titres_tableaux_qualitatifs = {
    "session": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "selon leur session d'études"
    ),
    "utilisation_recente": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "selon leur utilisation récente de l'intelligence artificielle"
    ),
    "usage_principal": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "leur usage principal de l'intelligence artificielle"
    ),
    "outil_principal": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "l'outil d'intelligence artificielle principalement utilisé"
    ),
    "verification": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "leur fréquence de vérification des informations fournies "
        "par l'intelligence artificielle"
    ),
    "utilite_percue": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "l'utilité perçue de l'intelligence artificielle"
        
    )
}

for numero, (variable, (tableau, base)) in enumerate(
    tableaux_qualitatifs.items(),
    start=1
):
    print("\n" + "=" * 80)
    print(f"Tableau {numero}: {titres_tableaux_qualitatifs[variable]}")
    print("=" * 80)
    print(tableau)

# Vérification rapide des tableaux créés:
for variable, (tableau, base) in tableaux_qualitatifs.items():

    # On exclut la ligne total avant de vérifier la somme des modalités:
    modalites = tableau.drop(index="Total")

    assert modalites["Effectif"].sum() == len(base), (
        f"Erreur dans le tableau : {variable}"
    )

print("\nVérification des tableaux des variables qualitatives ok.")

### 7. Tableaux des variables quantitative

### 7.1. Tableau des jours d'utilisation chez les utilisateurs récents de l'IA

# Création du tableau:
tableau_jours_utilisation = creer_tableau_frequences(
    donnees_utilisateurs,
    "jours_utilisation",
    cumulatif=True,
    ordonner_valeurs=True
)

# Ajout à un dictionnaire sous la forme (tableau, base).
tableaux_quantitatifs = {
    "jours_utilisation": (
        tableau_jours_utilisation,
        donnees_utilisateurs
    )
}

# Ajout du titre du tableau à un dictionnaire pour les titres des tableaux quantitatifs.
titres_tableaux_quantitatifs = {
    "jours_utilisation": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon le "
        "nombre de jours d'utilisation au cours des 7 derniers jours"
    )
}

### 7.2. Tableau de la durée d'utilisation chez les utilisateur récents de l'IA

# Nombre d'observation dans le sous-échantillon d'utilisateurs.
nb_utilisateurs = len(donnees_utilisateurs)
print(f"Nombre d'utilisateurs: {nb_utilisateurs}")

# Calcul du nombre de classes selon la règle de Sturges, arrondi vers le haut.
nb_classes = int(
    np.ceil(1 + 3.322 * np.log10(nb_utilisateurs))
)
print(f"Nombre de classes, selon Sturges: {nb_classes}")

# Calcul de l'étendue
duree_min = donnees_utilisateurs["duree_minutes"].min() # 16 minutes
duree_max = donnees_utilisateurs["duree_minutes"].max() # 650 minutes

etendue_duree = duree_max - duree_min   # De 16 à 650 minutes => étendue de 634 minutes.
print(f"Étendue: {etendue_duree} minutes")

# Calcul de l'amplitude
amplitude_duree = int(
    np.ceil(etendue_duree / nb_classes)
)
print(f"Amplitude des classes: {amplitude_duree} minutes")

# Construction des classes.
bornes_duree = [16, 96, 176, 256, 336, 416, 496, 576, 656]

etiquettes_classes_duree = [
    "[16 ; 96[",
    "[96 ; 176[",
    "[176 ; 256[",
    "[256 ; 336[",
    "[336 ; 416[",
    "[416 ; 496[",
    "[496 ; 576[",
    "[576 ; 656["
]

# Regroupement des durées en huit classes de 80 minutes.
donnees_utilisateurs["classe_duree"] = pd.cut(
    donnees_utilisateurs["duree_minutes"],
    bins=bornes_duree,
    labels=etiquettes_classes_duree,
    right=False # Classes fermées à gauche et ouvertes à droite.
)

# Vérification essentielle à faire maintenant:
# On s'assure de ne pas avoir de classes vides.
assert donnees_utilisateurs["classe_duree"].notna().all()

# Création du tableau:
tableau_duree_utilisation = creer_tableau_frequences(
    donnees_utilisateurs,
    "classe_duree",
    cumulatif=True
)

tableaux_quantitatifs["duree_minutes"] = (
    tableau_duree_utilisation,
    donnees_utilisateurs
)

titres_tableaux_quantitatifs["duree_minutes"] = (
    "Distribution des étudiants inscrits en Science de la nature "
    "ayant utilisé récemment l'intelligence artificielle selon la "
    "durée d'utilisation, en minutes"
)

# Affichage des tableaux quantitatifs.
for numero, (variable, (tableau, base)) in enumerate(
    tableaux_quantitatifs.items(),
    start=1
):
    print("\n" + "=" * 80)
    print(f"Tableau {numero} : {titres_tableaux_quantitatifs[variable]}")
    print("=" * 80)
    print(tableau)

# Vérification rapide des tableaux créés:
for variable, (tableau, base) in tableaux_quantitatifs.items():

    # On exclut la ligne total avant de vérifier la somme des modalités:
    modalites = tableau.drop(index="Total")

    assert modalites["Effectif"].sum() == len(base), (
        f"Erreur dans le tableau : {variable}"
    )