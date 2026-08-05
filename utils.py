import pandas as pd
import numpy as np
import os

# Fixer la graine pour avoir des résultats reproductibles hors ligne
np.random.seed(42)
n_lignes = 10000

print("Génération des données en cours...")

# 1. Variables indépendantes (Caractéristiques des entreprises)
age_entreprise = np.random.randint(1, 15, size=n_lignes)  # de 1 à 15 ans
nombre_employes = np.random.randint(1, 25, size=n_lignes)  # de 1 à 25 employés
capital_initial_kvd = (
    np.random.exponential(scale=500, size=n_lignes) + 100
)  # Capital en vD (valeur dollar)

# Heures de connexion nocturne utilisées (simulation de votre propre routine !)
heures_connexion_nuit = np.random.uniform(1.0, 8.0, size=n_lignes)

# 2. Variable pour la RÉGRESSION LINÉAIRE : Chiffre d'affaires annuel (avec un bruit statistique)
# Le chiffre d'affaires dépend positivement de l'âge, des employés et fortement des heures de connexion
bruit_lineaire = np.random.normal(0, 150, size=n_lignes)
chiffre_affaires_an = (
    (age_entreprise * 120)
    + (nombre_employes * 250)
    + (heures_connexion_nuit * 450)
    + (capital_initial_kvd * 1.2)
    + bruit_lineaire
)
# S'assurer qu'il n'y a pas de valeurs négatives absurdes
chiffre_affaires_an = np.clip(chiffre_affaires_an, 300, None)

# 3. Variable pour la RÉGRESSION LOGISTIQUE : Adoption d'un outil IA / Numérique (0 ou 1)
# La probabilité dépend du nombre d'employés et des heures de connexion nocturne
score_logistique = -3 + (nombre_employes * 0.15) + (heures_connexion_nuit * 0.5)
probabilite = 1 / (1 + np.exp(-score_logistique))  # Fonction sigmoïde
adoption_ia = np.random.binomial(1, probabilite)

# 4. Assemblage du DataFram
df = pd.DataFrame(
    {
        "ID_Entreprise": range(1, n_lignes + 1),
        "Age_Entreprise_Ans": age_entreprise,
        "Nombre_Employes": nombre_employes,
        "Capital_Initial_USD": np.round(capital_initial_kvd, 2),
        "Heures_Connexion_Nuit": np.round(heures_connexion_nuit, 1),
        "Chiffre_Affaires_Annuel_USD": np.round(chiffre_affaires_an, 2),
        "Adoption_Outil_IA": adoption_ia,
    }
)

# Sauvegarde locale du fichier CSV
nom_fichier = "donnees_goma_2026.csv"
folder = "data"
root = os.path.join(folder, nom_fichier)
df.to_csv(root, index=False)
print(f"✅ Fichier '{nom_fichier}' généré avec succès ({n_lignes} lignes) !")
