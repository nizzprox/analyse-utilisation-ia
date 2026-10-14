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

#1. paramètres généraux

N_ÉTUDIANTS = 75
RAND_SEED = 2026

DOSSIER_PROJET = Path(__file__).resolve().parent.parent #Path choisira automatiquement la synthaxe appropriée au système d'exploitation lors du téléchargement depuis github par un tiers parti
DOSSIER_DONNEES = DOSSIER_PROJET / "donnees" #définition du chemin vers le dossier "donnees"
CHEMIN_CSV = DOSSIER_DONNEES / "donnees_ia.csv" #définition du chemin pour la création du ficher .csv de la base de données fictive

DOSSIER_DONNEES.mkdir(parents=True, exist_ok=True) #redondance utile pour éviter des problèmes

#2. Initialisation du générateur aléatoire

rng = np.random.default_rng(RAND_SEED) #création du générateur de nombres aléatoire NumPy, pour que le programme produise les mêmes données chaque fois qu'il est exécuté

#3. Création des identifiants

ids = np.arange(1, N_ÉTUDIANTS + 1)

#4. Création initiale de la base de données fictive

donnees = pd.DataFrame({
    "id" : ids
})

#5. Visualisation de la base

print("Cinq premières lignes: ")
print(donnees.head())

print("\nCinq dernières lignes: ")
print(donnees.tail())

print("\nDimensions de la base: ")
print(donnees.shape)

