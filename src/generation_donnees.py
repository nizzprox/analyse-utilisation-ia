"""
Génération d'une base de données fictive portant sur les habitudes
d'utilisation de l'intelligence artificielle générative chez les
étudiants en Sciences de la nature au Cégep Édouard-Montpetit.

Les données produites sont fictives et sont utilisées uniquement
à des fins pédagogiques.
"""
from pathlib import Path

import numpy as np
import pandas as pd

### 1. paramètres généraux

N_ÉTUDIANTS = 75
RAND_SEED = 2026

DOSSIER_PROJET = Path(__file__).resolve().parent.parent #Path choisira automatiquement la synthaxe appropriée au système d'exploitation lors du téléchargement depuis github par un tiers parti
DOSSIER_DONNEES = DOSSIER_PROJET / "donnees" #définition du chemin vers le dossier "donnees"
CHEMIN_CSV = DOSSIER_DONNEES / "donnees_ia.csv" #définition du chemin pour la création du ficher .csv de la base de données fictive

DOSSIER_DONNEES.mkdir(parents=True, exist_ok=True) #redondance utile pour éviter des problèmes

### 2. Initialisation du générateur aléatoire

rng = np.random.default_rng(RAND_SEED) #création du générateur de nombres aléatoire NumPy, pour que le programme produise les mêmes données chaque fois qu'il est exécuté

### 3. Création des identifiants

ids = np.arange(1, N_ÉTUDIANTS + 1)

### 4. Création initiale de la base de données fictive

donnees = pd.DataFrame({
    "id" : ids
})

### 5. Visualisation de la base

print("Cinq premières lignes: ")
print(donnees.head())

print("\nCinq dernières lignes: ")
print(donnees.tail())

print("\nDimensions de la base: ")
print(donnees.shape)

### 6. Vérifs initiales

#on vérifie qu'on a bien créé 75 identifiants
assert len(donnees) == N_ÉTUDIANTS
#on vérifie qu'il n'y a pas de doublons
assert donnees["id"].is_unique
#on vérifie ensuite que la premier et le dernier identifiants valent respectivement 1 et 75
assert donnees["id"].min() == 1
assert donnees["id"].max() == N_ÉTUDIANTS 

#si tout va bien, la ligne suivante s'exécute:
print("\nToutes les vérifications initiales ont réussi.")

### 7. Ajout de la variable "session" à la base

choix_session = ["1re", "2e", "3e", "4e", "5e ou plus"]
prob_session = [0.225, 0.225, 0.225, 0.225, 0.10]

#petites vérifications des paramètres
assert np.isclose(sum(prob_session), 1.00) 
assert len(choix_session) == len(prob_session)
assert all(0 <= probabilite <= 1 for probabilite in prob_session)

#création des sessions pour les étudiants
sessions = rng.choice(
    a=choix_session,
    size=N_ÉTUDIANTS,
    replace=True,
    p=prob_session
)

#ajout à la base de données fictive
donnees["session"] = sessions

#petites vérifications des sessions
print(donnees.head())
print(donnees["session"].value_counts().sort_index())
assert donnees["session"].isin(choix_session).all()
assert len(donnees["session"]) == N_ÉTUDIANTS
print("\nVérification des sessions ok.")

### 8. Ajout des variables jours_utilisation et utilisation_recente

nb_jours_possibles = np.arange(0, 8)
prob_nb_jours = [0.12, 0.12, 0.16, 0.20, 0.16, 0.12, 0.07, 0.05]

#petites vérifications des paramètres
assert np.isclose(sum(prob_nb_jours), 1.00) 
assert len(nb_jours_possibles) == len(prob_nb_jours)
assert all(0 <= probabilite <= 1 for probabilite in prob_nb_jours)

#création du nombre de jours où l'IA a été utilisée pour les étudiants
nb_jours_utilisation = rng.choice(
    a=nb_jours_possibles,
    size=N_ÉTUDIANTS,
    replace=True,
    p=prob_nb_jours
)

#ajout à la base
donnees["jours_utilisation"] = nb_jours_utilisation

#petites vérifications des nombres de jours d'utilisation
print(donnees.head())
print(donnees["jours_utilisation"].value_counts().sort_index())
assert donnees["jours_utilisation"].isin(nb_jours_possibles).all()
assert len(donnees["jours_utilisation"]) == N_ÉTUDIANTS
assert donnees["jours_utilisation"].between(0, 7).all()
print("\nVérification des jours d'utilisation ok.")

#déduction de l'utilisation récente de l'IA selon le nombre de jours d'utilisation
utilisation_recente = np.where(nb_jours_utilisation > 0, "Oui", "Non")

#ajout à la base
donnees["utilisation_recente"] = utilisation_recente

#Vérifications croisées de utilisation_recente avec jours_utilisation
condition_utilisateur = donnees["jours_utilisation"] > 0
condition_non_utilisateur = donnees["jours_utilisation"] == 0

#On vérifie d'abord si, parmi les lignes où le nombre de jours est supérieur à zéro,
#toutes les valeurs d'utilisation récente sont égales à "Oui"
assert donnees.loc[
    condition_utilisateur,
    "utilisation_recente"
].eq("Oui").all()

#On vérifie le cas inverse
assert donnees.loc[
    condition_non_utilisateur,
    "utilisation_recente"
].eq("Non").all() 

#Finalement, on vérifie qu'il ne manque pas de données
assert donnees["jours_utilisation"].notna().all()
assert donnees["utilisation_recente"].notna().all()
assert donnees["utilisation_recente"].isin(["Oui", "Non"]).all()

print(donnees[
    ["jours_utilisation", "utilisation_recente"]
].head(10))

print(donnees["utilisation_recente"].value_counts())

print("Vérifications croisées ok.")

### 9. Ajout de la variable duree_minutes

#il faut d'abord s'assurer que les non-utilisateurs ne sont pas considérés par l'algorithme
nb_utilisateurs = condition_utilisateur.sum()

#création d'un tableau initial pour les durées
tableau_durees_minutes = np.zeros(N_ÉTUDIANTS, dtype=int)

#on va utiliser une distribution gamma pour obtenir des durées quotidiennes qui
#sont concentrées autour de des petites ou moyennes valeurs, mais présentant
#quelques valeurs plus élevées. La distribution gamma est aussi utile puisqu'elle 
#ne génère pas de valeurs négatives. Nous n'avons pas de données réelles qui démontrent
#que la durée d'utilisation de l'IA suit une loi Gamma; nous posons seulement une
#hypothèse plausible.
durees_quoti_utilisateurs = rng.gamma(
    shape=3,
    scale=15,
    size=nb_utilisateurs
)

#préparation de la série contenant les jours d'utilisation des utilisateurs de l'IA
jours_des_utilisateurs = donnees.loc[
    condition_utilisateur,
    "jours_utilisation"
]
#on vérifie qu'on a bel et bien sélectionné seulement les valeurs plus grandes que zéro
assert len(jours_des_utilisateurs) == nb_utilisateurs
assert (jours_des_utilisateurs > 0).all()

#on calcule enfin les durées hebdomadaires des utilisateurs
durees_hebdomadaires = np.rint(
    jours_des_utilisateurs.to_numpy() * durees_quoti_utilisateurs
).astype(int)

durees_hebdomadaires = np.clip(
    durees_hebdomadaires,
    a_min=1,
    a_max=None
)

#on insère les valeurs générées
tableau_durees_minutes[condition_utilisateur] = durees_hebdomadaires

#on ajoute à la base
donnees["duree_minutes"] = tableau_durees_minutes

#vérifications nécessaires
assert len(donnees["duree_minutes"]) == N_ÉTUDIANTS
assert donnees["duree_minutes"].notna().all()
assert (donnees["duree_minutes"] >= 0).all()
assert pd.api.types.is_integer_dtype(
    donnees["duree_minutes"]
)


#On vérifie que, parmi les lignes où le nombre de jours est supérieur à zéro,
#la durée hebdomadaire est strictement supérieure à zéro:
assert (
    donnees.loc[
        condition_utilisateur,
        "duree_minutes"
    ] > 0
).all()

#on vérifie ensuite le cas inverse:
assert (
    donnees.loc[
        condition_non_utilisateur,
        "duree_minutes"
    ] == 0
).all()

#inspection visuelle
print(
        donnees[
        [
            "jours_utilisation",
            "utilisation_recente",
            "duree_minutes"
        ]
    ].head(10)
)

### 10. Ajout de la variable usage_principal

#initialisation du tableau:
tableau_usages_principaux = np.full(
    N_ÉTUDIANTS,
    "Aucun usage",
    dtype=object    # permet de stocker des chaînes des longueurs différentes
)

#définition des paramètres
usages_possibles = [
    "Comprendre une notion",
    "Résoudre un problème ou un exercice",
    "Résumer",
    "Rédiger ou reformuler",
    "Préparer un examen",
    "Écrire ou comprendre du code",
    "Autre"
]

probabilites_usages_possibles = [0.24, 0.22, 0.12, 0.16, 0.11, 0.10, 0.05]

#vérifications des paramètres
assert np.isclose(sum(probabilites_usages_possibles), 1.0)
assert len(usages_possibles) == len(probabilites_usages_possibles)
assert all(0 <= p <= 1 for p in probabilites_usages_possibles)

#on génère les usages des utilisateurs
usages_generes = rng.choice(
    a=usages_possibles,
    size=nb_utilisateurs,
    replace=True,
    p=probabilites_usages_possibles
)

#on insère les usages dans le tableau des usages
tableau_usages_principaux[condition_utilisateur] = usages_generes

#ajout à la base
donnees["usage_principal"] = tableau_usages_principaux

#vérifications des usages principaux
assert len(usages_generes) == nb_utilisateurs
assert len(donnees["usage_principal"]) == N_ÉTUDIANTS
assert donnees["usage_principal"].notna().all()

assert donnees.loc[
    condition_utilisateur,
    "usage_principal"
].isin(usages_possibles).all()

assert donnees.loc[
    condition_non_utilisateur,
    "usage_principal"
].eq("Aucun usage").all()

#inspection visuelle
print(
        donnees[
        [
            "utilisation_recente",
            "usage_principal"
        ]
    ].head(10)
)
print(donnees["usage_principal"].value_counts())

### 11. Ajout de la variable outil_principal

#étant donné que l'implémentation de cette variable est pratiquement 
#identique à la variable précédente, les commentaires seront omis.

tableau_outil_principal = np.full(
    N_ÉTUDIANTS,
    "Aucun outil",
    dtype=object
)

assert len(tableau_outil_principal) == N_ÉTUDIANTS

outils_possibles = [
    "ChatGPT",
    "Gemini",
    "Copilot",
    "Perplexity",
    "Claude",
    "DeepSeek",
    "Autre"
]

probabilites_outils = [0.44, 0.06, 0.10, 0.10, 0.20, 0.05, 0.05]

assert len(outils_possibles) == len(probabilites_outils)
assert all(0 <= p <= 1 for p in probabilites_outils)
assert np.isclose(sum(probabilites_outils), 1.0)

outils_generes = rng.choice(
    a=outils_possibles,
    size=nb_utilisateurs,
    replace=True,
    p=probabilites_outils
)

tableau_outil_principal[condition_utilisateur] = outils_generes

donnees["outil_principal"] = tableau_outil_principal

assert len(outils_generes) == nb_utilisateurs
assert len(donnees["outil_principal"]) == N_ÉTUDIANTS
assert donnees["outil_principal"].notna().all()

assert donnees.loc[
    condition_utilisateur,
    "outil_principal"
].isin(outils_possibles).all()

assert donnees.loc[
    condition_non_utilisateur,
    "outil_principal"
].eq("Aucun outil").all()

print(
        donnees[
        [
            "utilisation_recente",
            "outil_principal"
        ]
    ].head(10)
)
print(donnees["outil_principal"].value_counts())

### 12. Ajout de la variable verification

#encore une fois, le code suivra la même structure que pour les
#deux variables précédentes.

tableau_verification = np.full(
    N_ÉTUDIANTS,
    "Je n'utilise pas l'IA",
    dtype=object
)

assert len(tableau_verification) == N_ÉTUDIANTS

niveaux_verif_possibles = [
    "Jamais",
    "Rarement",
    "Parfois",
    "Souvent",
    "Toujours"
]

probabilites_niveau_verif = [0.20, 0.15, 0.30, 0.25, 0.10]

assert len(niveaux_verif_possibles) == len(probabilites_niveau_verif)
assert np.isclose(sum(probabilites_niveau_verif), 1.0)
assert all(0 <= p <= 1 for p in probabilites_niveau_verif)

niveaux_verif_generes = rng.choice(
    a=niveaux_verif_possibles,
    size=nb_utilisateurs,
    replace=True,
    p=probabilites_niveau_verif
)

tableau_verification[condition_utilisateur] = niveaux_verif_generes

donnees["verification"] = tableau_verification

assert len(donnees["verification"]) == N_ÉTUDIANTS
assert len(niveaux_verif_generes) == nb_utilisateurs
assert donnees["verification"].notna().all()

assert donnees.loc[
    condition_utilisateur,
    "verification"
].isin(niveaux_verif_possibles).all()

assert donnees.loc[
    condition_non_utilisateur,
    "verification"
].eq("Je n'utilise pas l'IA").all()

print(
        donnees[
        [
            "utilisation_recente",
            "verification"
        ]
    ].head(10)
)
print(donnees["verification"].value_counts())