### 1. Importation des bibliothèques nécessaires
from pathlib import Path

import matplotlib.pyplot as plt
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


# Tableaux portant uniquement sur les utilisateurs récents:

# Petites modifications à faire avant:
donnees_utilisateurs["verification"] = donnees_utilisateurs["verification"].cat.remove_unused_categories()
donnees_utilisateurs["utilite_percue"] = donnees_utilisateurs["utilite_percue"].cat.remove_unused_categories()

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
        "selon la session d'études"
    ),
    "utilisation_recente": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "selon l'utilisation récente de l'intelligence artificielle"
    ),
    "usage_principal": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "l'usage principal de l'intelligence artificielle"
    ),
    "outil_principal": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "l'outil d'intelligence artificielle principalement utilisé"
    ),
    "verification": (
        "Répartition des étudiants inscrits en Sciences de la nature "
        "ayant utilisé récemment l'intelligence artificielle selon "
        "la fréquence de vérification des informations fournies "
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

### 8. Indicateurs statistiques des variables quantitatives

### 8.1. Indicateurs du nombre de jours d'utilisation

# On extrait la série quantitative non groupée.
donnees_jours = donnees_utilisateurs["jours_utilisation"]

# Nombre d'observations:
nb_jours = donnees_jours.count()

# Indicateurs de tendance centrale:
moyenne_jours = donnees_jours.mean()
mediane_jours = donnees_jours.median()
modes_jours = donnees_jours.mode().to_list()    # .to_list() au cas où il y aurait plusieurs modes.

# Indicateurs de dispersion:
minimum_jours = donnees_jours.min() 
maximum_jours = donnees_jours.max()
etendue_jours = maximum_jours - minimum_jours

# Variance:
variance_jours = donnees_jours.var(ddof=0)

# Écart-type:
ecart_type_jours = donnees_jours.std(ddof=0)

# Coefficient de variation (pourcentage):
coeff_variation_jours = (ecart_type_jours / moyenne_jours) * 100

# On regroupe les indicateurs dans une seule série pandas:
indicateurs_jours = pd.Series({
    "Effectif": nb_jours,
    "Moyenne": moyenne_jours,
    "Médiane": mediane_jours,
    "Mode": ", ".join(map(str, modes_jours)),
    "Minimum": minimum_jours,
    "Maximum": maximum_jours,
    "Étendue": etendue_jours,
    "Variance": variance_jours,
    "Écart-type": ecart_type_jours,
    "Coefficient de variation (%)": coeff_variation_jours
})

print("\nIndicateurs statistiques du nombre de jours d'utilisation:")
print(indicateurs_jours)

### 8.2. Indicateurs de la durée d'utilisation

tableau_classes_duree = tableau_duree_utilisation.drop(index="Total")

effectifs_duree = tableau_classes_duree["Effectif"].to_numpy()

bornes_inferieures = np.array([16, 96, 176, 256, 336, 416, 496, 576])

bornes_superieures = np.array([96, 176, 256, 336, 416, 496, 576, 656])

centres_classes = (bornes_inferieures + bornes_superieures) / 2

# Moyenne estimée:
nb_total_durees = effectifs_duree.sum()

moyenne_groupee_duree = (
    effectifs_duree * centres_classes
).sum() / nb_total_durees # (n_i * c_i) / N

# Médiane estimée:
# On trouve la classe médiane:
position_mediane = np.searchsorted(
    np.cumsum(effectifs_duree), 
    nb_total_durees / 2
)

# On détermine la borne inférieure et le nombre d'effectifs de la classe médiane:
borne_inf_mediane = bornes_duree[position_mediane]
n_classe_mediane = effectifs_duree[position_mediane]

if position_mediane == 0:   # Si la classe médiane est à l'index zéro,
    n_cumule_precedant = 0  # l'effectif cumulé de la classe précédente
                            # est de zéro lui aussi.
else:
    n_cumule_precedant = np.cumsum( # On calcule l'effectif cumulé de la
        effectifs_duree             # classe précédant la classe médiane.
    )[position_mediane - 1]

h_mediane_duree = ( # Calcul de l'amplitude de la classe médiane
    bornes_duree[position_mediane + 1]
    - bornes_duree[position_mediane]
)

mediane_duree = (   # On peut enfin estimer la médiane
    borne_inf_mediane +
    h_mediane_duree
    * (
        (nb_total_durees / 2 - n_cumule_precedant)
        / n_classe_mediane
    )
)

# Mode estimé:
position_mode = np.argmax(effectifs_duree) # On trouve l'index de la classe modale

borne_inf_mode = bornes_duree[position_mode]        # Amplitude de la classe modale
n_classe_modale = effectifs_duree[position_mode]    # Effectif de la classe modale

n_precedent = ( # Effectif de la classe précédant la classe modale
    effectifs_duree[position_mode - 1]
    if position_mode > 0
    else 0
)

n_suivant = (   # Effectif de la classe suivant la classe modale
    effectifs_duree[position_mode + 1]
    if position_mode < len(effectifs_duree) - 1
    else 0
)

h_mode = (
    bornes_duree[position_mode + 1]
    - bornes_duree[position_mode]
)

mode_duree = (
    borne_inf_mode
    + h_mode
    * (
        (n_classe_modale - n_precedent)
        / (
            (n_classe_modale - n_precedent)
            + (n_classe_modale - n_suivant)
        )
    )
)

# On a déjà calculé l'étendue à la ligne 385

# Variance estimée avec les centres et effectifs de classes:
variance_duree = (
    effectifs_duree
    * (centres_classes - moyenne_groupee_duree) ** 2
).sum() / nb_total_durees

# Écart-type estimé:
ecart_type_duree = np.sqrt(variance_duree)

# Coefficient de variation estimé:
coeff_variation_duree = (
    ecart_type_duree / moyenne_groupee_duree * 100
)

# On regroupe les indicateurs en série pandas:
indicateurs_duree = pd.Series({
    "Effectif": nb_total_durees,
    "Moyenne": moyenne_groupee_duree,
    "Médiane": mediane_duree,
    "Mode": mode_duree,
    "Minimum": duree_min,
    "Maximum": duree_max,
    "Étendue": etendue_duree,
    "Variance": variance_duree,
    "Écart-type": ecart_type_duree,
    "Coefficient de variation (%)": coeff_variation_duree
})

# Vérifications 
assert nb_total_durees == len(donnees_utilisateurs)

assert 16 <= moyenne_groupee_duree < 656
assert 16 <= mediane_duree < 656
assert bornes_duree[0] <= mode_duree < bornes_duree[-1]

assert variance_duree >= 0
assert ecart_type_duree >= 0
assert coeff_variation_duree >= 0

print("\nIndicateurs statistiques de la durée d'utilisation:")
print(indicateurs_duree)

# Regrouper en un seul tableau
indicateurs_statistiques = pd.concat(
    [
        indicateurs_jours.rename(
            "Nombre de jours d'utilisation"
        ),
        indicateurs_duree.rename(
            "Durée d'utilisation (minutes)"
        )
    ],
    axis=1
)

print("\nIndicateurs statistiques des variables quantitatives:")
print(indicateurs_statistiques)

### 9. Exportation des tableaux en CSV

# Tableaux qualitatifs:
for variable, (tableau, base) in tableaux_qualitatifs.items():

    tableau.to_csv(
        DOSSIER_TABLEAUX / f"{variable}.csv",
        sep=",",
        encoding="utf-8-sig",
        float_format="%.4f",
        index_label="Modalité"
    )

# Tableaux de distribution quantitative:
tableau_jours_utilisation.to_csv(
    DOSSIER_TABLEAUX / "jours_utilisation.csv",
    sep=",",
    encoding="utf-8-sig",
    float_format="%.2f",
    index_label="Valeur"
)

tableau_duree_utilisation.to_csv(
    DOSSIER_TABLEAUX / "duree_utilisation.csv",
    sep=",",
    encoding="utf-8-sig",
    float_format="%.2f",
    index_label="Valeur"
)

# Tableau des indicateurs statistiques:

# Copie pour l'export:
indicateurs_export = indicateurs_statistiques.copy()

# Définition des colonnes numériques à convertir:
colonnes_num = [
    "Nombre de jours d'utilisation",
    "Durée d'utilisation (minutes)"
]

# Conversion:
for colonne in colonnes_num:
    indicateurs_export[colonne] = pd.to_numeric(
        indicateurs_export[colonne],
        errors="coerce"
    ).round(2)

indicateurs_export.to_csv(
    DOSSIER_TABLEAUX / "indicateurs_statistiques.csv",
    sep=",",
    encoding="utf-8-sig",
    float_format="%.2f",
    index_label="Indicateur"
)

print("\nExportation des tableaux en CSV terminée.")

### 10. Représentations graphiques

# Couleurs qui seront utilisées dans les graphiques:
COULEUR_BANDES = "#4E79A7"
COULEUR_HISTOGRAMME = "#F28E2B"
COULEURS_CIRCULAIRE = ["#4E79A7", "#F28E2B"]

def enregistrer_graphique(nom_fichier):
    """
    Ajuste la mise en page, enregistre le graphique au format PNG,
    affique le graphique puis libère la mémoire.
    """

    plt.tight_layout()

    plt.savefig(
        DOSSIER_GRAPHIQUES / nom_fichier,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

def creer_diagramme_bandes(
    tableau,
    titre,
    nom_fichier,
    etiquette_axe_x,
    numero_graphique,
    horizontal=False
):
    """
    Crée un diagramme à bande à partir d'un tableau de fréquences.
    """

    titre_complet = f"Graphique {numero_graphique}: {titre}"

    # On ne veut pas afficher la ligne Total dans le graphique
    donnees_graphique = tableau.drop(index="Total")

    # Identifie les modalités et effectifs
    modalites = donnees_graphique.index.astype(str)
    effectifs = donnees_graphique["Effectif"]

    # Initialise le graphique
    plt.figure(figsize=(10,7))

    if horizontal:  # Si on décide d'afficher le graphique à l'horizontal pour faciliter la lecture des longues modalités
        plt.barh(
            modalites,
            effectifs,
            color=COULEUR_BANDES,
            edgecolor="black"
        )

        plt.xlabel("Effectif")
        plt.ylabel(etiquette_axe_x)
        plt.grid(axis="x", alpha=0.25)

        for position, valeur in enumerate(effectifs):
            plt.text(
                valeur + 0.3,
                position,
                str(valeur),
                va="center"
            )
    else:
        plt.bar(
            modalites,
            effectifs,
            color=COULEUR_BANDES,
            edgecolor="black"
        )

        plt.xlabel(etiquette_axe_x)
        plt.ylabel("Effectif")
        plt.grid(axis="y", alpha=0.25)
        
        # On fait pivoter les étiquettes de l'axe x de 25 degrés vers la droite,
        # ce qui aide pour les longues modalités:
        plt.xticks( 
            rotation=25,
            ha="right"
        )

        for position, valeur in enumerate(effectifs):
            plt.text(
                position,
                valeur + 0.3,
                str(valeur),
                ha="center"
            )
    
    plt.subplots_adjust(bottom=0.34)

    # Ajout du titre:
    plt.figtext(
        0.5,
        0.02,
        titre_complet,
        ha="center",
        va="top",
        wrap=True,
        fontsize=10
    )

    # Enregistrement du graphique:
    enregistrer_graphique(nom_fichier)

def creer_diagramme_circulaire(
    tableau,
    titre,
    nom_fichier,
    numero_graphique
):
    """
    Crée un diagramme circulaire à partir d'un tableau de fréquences
    Cette fonction est utilisée pour la variable dichotomique
    utilisation_recente.
    """

    titre_complet = f"Graphique {numero_graphique}: {titre}"

    # On ne veut pas ajouter la ligne Total du graphique
    donnees_graphique = tableau.drop(index="Total")

    # Initialisation du graphique
    plt.figure(figsize=(8, 7))

    # Création du diagramme circulaire
    plt.pie(
        donnees_graphique["Effectif"],
        labels=donnees_graphique.index.astype(str),
        autopct="%1.1f %%", # Affiche le pourcentage des modalités avec une décimale
        startangle=90,
        colors=COULEURS_CIRCULAIRE,
        wedgeprops={"edgecolor": "white"} # Ajoute une fine démarcation entre les modalités
    )

    # Ajoute le titre
    plt.figtext(
        0.5,
        0.03,
        titre_complet,
        ha="center",
        va="bottom",
        wrap=True,
        fontsize=10
    )

    # Enregistre le graphique
    enregistrer_graphique(nom_fichier)

def creer_histogramme_duree(
    effectifs,
    bornes,
    titre,
    nom_fichier,
    numero_graphique
):
    """
    Crée un histogramme de la durée d'utilisation.
    """

    titre_complet = f"Graphique {numero_graphique}: {titre}"

    largeurs_classes = np.diff(bornes) # Détermine la largeur des classes selon les bornes fournies

    # Initialise le graphique
    plt.figure(figsize=(11, 6))

    plt.bar(
        bornes[:-1], # Inclut toutes les bornes sauf la dernière.
        effectifs, # Donne la hauteur de chaque rectangle.
        width=largeurs_classes, # Donne la largeur réelle de chaque rectangle (80 min)
        align="edge",   # Indique que la valeur fournie correspond au bord gauche, et non au centre du rectangle.
        color=COULEUR_HISTOGRAMME,
        edgecolor="black"
    )

    plt.xlabel("Durée totale d'utilisation lors des 7 derniers jours (minutes)")
    plt.ylabel("Effectif")  # Indique ce que représente la hauteur.
    plt.xticks(bornes)  # Affiche les bornes sur l'axe horizontal.
    plt.grid(axis="y", alpha=0.25)

    plt.subplots_adjust(bottom=0.34)

    # Ajout du titre:
    plt.figtext(
        0.5,
        0.02,
        titre_complet,
        ha="center",
        va="top",
        wrap=True,
        fontsize=10
    )

    enregistrer_graphique(nom_fichier)

# Les titres sont communs aux tableaux et aux graphiques:
titres_graphiques = {
    **titres_tableaux_qualitatifs,
    **titres_tableaux_quantitatifs
}

# Diagrammes en bandes: variables qualitatives et jours d'utilisation:
creer_diagramme_bandes(
    tableau_session,
    titres_graphiques["session"],
    "session.png",
    "Session d'études",
    1
)

creer_diagramme_bandes(
    tableau_usage,
    titres_graphiques["usage_principal"],
    "usage_principal.png",
    "Usage principal",
    2,
    horizontal=True
)

creer_diagramme_bandes(
    tableau_outil,
    titres_graphiques["outil_principal"],
    "outil_principal.png",
    "Outil principal",
    3,
    horizontal=True
)

creer_diagramme_bandes(
    tableau_verification,
    titres_graphiques["verification"],
    "verification.png",
    "Fréquence de vérification",
    4
)

creer_diagramme_bandes(
    tableau_utilite,
    titres_graphiques["utilite_percue"],
    "utilite_percue.png",
    "Utilité perçue",
    5
)

creer_diagramme_bandes(
    tableau_jours_utilisation,
    titres_graphiques["jours_utilisation"],
    "jours_utilisation.png",
    "Nombre de jours d'utilisation",
    6
)

# Diagramme circulaire: variable qualitative nominale (Oui/Non):
creer_diagramme_circulaire(
    tableau_utilisation,
    titres_graphiques["utilisation_recente"],
    "utilisation_recente.png",
    7
)

# Histogramme: durée d'utilisation (en classes):
creer_histogramme_duree(
    effectifs_duree,
    bornes_duree,
    titres_graphiques["duree_minutes"],
    "duree_utilisation.png",
    8
)
