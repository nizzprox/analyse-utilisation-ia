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

