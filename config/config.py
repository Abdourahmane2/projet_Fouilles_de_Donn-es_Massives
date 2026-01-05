"""
Configuration globale du projet de détection de fraudes
========================================================

Ce fichier centralise tous les paramètres du projet pour assurer
la reproductibilité et faciliter les modifications.
"""

import os
from pathlib import Path

# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

# Chemins des données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Chemins des outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Créer les dossiers s'ils n'existent pas
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMÈTRES DES DONNÉES
# =============================================================================

# Nom du fichier de données (À MODIFIER selon ton fichier)
DATA_FILENAME = "données.txt" 

# Paramètres de lecture du fichier
FILE_PARAMS = {
    "sep": ";",              # Séparateur
    "decimal": ",",          # Séparateur décimal français
    "encoding": "utf-8",     # Essayer "latin-1" si erreur
}

# Variable cible
TARGET_COLUMN = "FlagImpaye"

# Colonnes à exclure de la modélisation
COLUMNS_TO_EXCLUDE = [
    "ZIBZIN",                    # Identifiant client
    "IDAvisAutorisationCheque",  # Identifiant transaction
    "DateTransaction",           # Date (utilisée pour split uniquement)
    "CodeDecision",              # Information post-transaction !
]

# Colonne de date pour le split temporel
DATE_COLUMN = "DateTransaction"

# =============================================================================
# PARAMÈTRES DE SÉPARATION TRAIN/TEST
# =============================================================================

# Dates de séparation (selon le sujet)
TRAIN_START_DATE = "2017-02-01"
TRAIN_END_DATE = "2017-08-31"
TEST_START_DATE = "2017-09-01"
TEST_END_DATE = "2017-11-30"

# =============================================================================
# PARAMÈTRES DE MODÉLISATION
# =============================================================================

# Seed pour reproductibilité
RANDOM_STATE = 42

# Validation croisée
CV_FOLDS = 5

# Métrique principale à optimiser
PRIMARY_METRIC = "f1"

# Métriques secondaires à suivre
SECONDARY_METRICS = ["precision", "recall", "roc_auc", "average_precision"]

# =============================================================================
# PARAMÈTRES DE LA MATRICE DE COÛTS (PARTIE 2)
# =============================================================================

# Taux de marge pour les transactions acceptées
MARGIN_RATE = 0.05  # 5%

# Taux de récupération pour les FP (transactions refusées à tort)
FP_RECOVERY_RATE = 0.70  # 70%

# Fonction de perte pour les FN (fraudes non détectées)
def calculate_fn_loss(amount):
    """
    Calcule la perte pour une fraude non détectée (FN)
    selon le montant de la transaction.
    
    Args:
        amount: Montant de la transaction
        
    Returns:
        Perte associée
    """
    if amount <= 20:
        return 0
    elif amount <= 50:
        return 0.2 * amount
    elif amount <= 100:
        return 0.3 * amount
    elif amount <= 200:
        return 0.5 * amount
    else:
        return 0.8 * amount

# =============================================================================
# PARAMÈTRES DES ALGORITHMES
# =============================================================================

# Hyperparamètres par défaut pour Random Forest
RF_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Hyperparamètres par défaut pour XGBoost
XGB_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Hyperparamètres par défaut pour LightGBM
LGBM_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": -1,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# =============================================================================
# PARAMÈTRES D'ÉCHANTILLONNAGE
# =============================================================================

# SMOTE
SMOTE_PARAMS = {
    "sampling_strategy": "auto",
    "k_neighbors": 5,
    "random_state": RANDOM_STATE,
}

# ADASYN
ADASYN_PARAMS = {
    "sampling_strategy": "auto",
    "n_neighbors": 5,
    "random_state": RANDOM_STATE,
}

# =============================================================================
# PARAMÈTRES DE VISUALISATION
# =============================================================================

# Style des graphiques
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# Taille par défaut des figures
FIGURE_SIZE = (12, 8)

# Palette de couleurs
COLOR_PALETTE = "viridis"

# DPI pour sauvegarde
FIGURE_DPI = 300

# =============================================================================
# AFFICHAGE DE LA CONFIGURATION
# =============================================================================

def print_config():
    """Affiche la configuration actuelle."""
    print("=" * 60)
    print("CONFIGURATION DU PROJET")
    print("=" * 60)
    print(f"\n📁 Chemins:")
    print(f"   - Projet: {PROJECT_ROOT}")
    print(f"   - Données: {RAW_DATA_DIR}")
    print(f"   - Figures: {FIGURES_DIR}")
    print(f"\n📊 Données:")
    print(f"   - Fichier: {DATA_FILENAME}")
    print(f"   - Cible: {TARGET_COLUMN}")
    print(f"   - Colonnes exclues: {COLUMNS_TO_EXCLUDE}")
    print(f"\n📅 Split temporel:")
    print(f"   - Train: {TRAIN_START_DATE} → {TRAIN_END_DATE}")
    print(f"   - Test: {TEST_START_DATE} → {TEST_END_DATE}")
    print(f"\n⚙️ Modélisation:")
    print(f"   - Random state: {RANDOM_STATE}")
    print(f"   - CV folds: {CV_FOLDS}")
    print(f"   - Métrique: {PRIMARY_METRIC}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
