"""
Module de chargement et validation des données
===============================================

Ce module fournit les fonctions pour charger les données brutes
et effectuer les premières validations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    RAW_DATA_DIR,
    DATA_FILENAME,
    FILE_PARAMS,
    TARGET_COLUMN,
    DATE_COLUMN,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)


def load_raw_data(filename: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Charge les données brutes depuis le fichier.
    
    Args:
        filename: Nom du fichier (utilise DATA_FILENAME par défaut)
        verbose: Afficher les informations de chargement
        
    Returns:
        DataFrame contenant les données brutes
    """
    if filename is None:
        filename = DATA_FILENAME
    
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {filepath}\n"
            f"Veuillez placer le fichier de données dans: {RAW_DATA_DIR}"
        )
    
    if verbose:
        print(f" Chargement des données depuis: {filepath}")
    
    # Chargement avec les paramètres français
    df = pd.read_csv(filepath, **FILE_PARAMS)
    
    if verbose:
        print(f" Données chargées: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    
    return df


def validate_columns(df: pd.DataFrame) -> dict:
    """
    Valide que les colonnes attendues sont présentes.
    
    Args:
        df: DataFrame à valider
        
    Returns:
        Dictionnaire avec le statut de validation
    """
    expected_columns = [
        "ZIBZIN", "IDAvisAutorisationCheque", "FlagImpaye", "Montant",
        "DateTransaction", "CodeDecision", "VerifianceCPT1", "VerifianceCPT2",
        "VerifianceCPT3", "D2CB", "ScoringFP1", "ScoringFP2", "ScoringFP3",
        "TauxImpNb_RB", "TauxImpNB_CPM", "EcartNumCheq", "NbrMagasin3J",
        "DiffDateTr1", "DiffDateTr2", "DiffDateTr3", "CA3TRetMtt", "CA3TR", "Heure"
    ]
    
    actual_columns = set(df.columns)
    expected_set = set(expected_columns)
    
    missing = expected_set - actual_columns
    extra = actual_columns - expected_set
    
    validation = {
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "expected_count": len(expected_columns),
        "actual_count": len(actual_columns),
    }
    
    return validation


def convert_data_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Conversion de la date
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce"
        )

        invalid_dates = df[DATE_COLUMN].isna().sum()
        if verbose:
            print(f" Colonne '{DATE_COLUMN}' convertie en datetime")
            if invalid_dates > 0:
                print(f" {invalid_dates} dates invalides converties en NaT")

        df = df.dropna(subset=[DATE_COLUMN])

    # Conversion de la cible
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        if verbose:
            print(f" Colonne '{TARGET_COLUMN}' convertie en int")

    # CodeDecision en catégorie
    if "CodeDecision" in df.columns:
        df["CodeDecision"] = df["CodeDecision"].astype("category")
        if verbose:
            print(f" Colonne 'CodeDecision' convertie en category")

    return df  



def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Génère un résumé des données.
    
    Args:
        df: DataFrame à résumer
        
    Returns:
        Dictionnaire avec les statistiques résumées
    """
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_total": df.isnull().sum().sum(),
        "missing_by_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    
    # Statistiques sur la cible
    if TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts()
        summary["target_distribution"] = target_counts.to_dict()
        summary["imbalance_ratio"] = target_counts.min() / target_counts.max()
    
    # Période des données
    if DATE_COLUMN in df.columns:
        df_dates = pd.to_datetime(df[DATE_COLUMN])
        summary["date_min"] = df_dates.min()
        summary["date_max"] = df_dates.max()
        summary["date_range_days"] = (df_dates.max() - df_dates.min()).days
    
    return summary


def split_by_date(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Sépare les données en train/test selon les dates du sujet.
    
    Args:
        df: DataFrame complet
        verbose: Afficher les informations
        
    Returns:
        Tuple (df_train, df_test)
    """
    df = df.copy()
    
    # S'assurer que la date est au bon format
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    
    # Créer une colonne date seule (sans heure)
    date_only = df[DATE_COLUMN].dt.date
    
    # Séparation selon les dates du sujet
    train_mask = (date_only >= pd.to_datetime(TRAIN_START_DATE).date()) & \
                 (date_only <= pd.to_datetime(TRAIN_END_DATE).date())
    test_mask = (date_only >= pd.to_datetime(TEST_START_DATE).date()) & \
                (date_only <= pd.to_datetime(TEST_END_DATE).date())
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    if verbose:
        print(f"\n Séparation temporelle:")
        print(f"   Train: {TRAIN_START_DATE} → {TRAIN_END_DATE}")
        print(f"   - {len(df_train):,} transactions")
        print(f"   - Fraudes: {df_train[TARGET_COLUMN].sum():,} ({df_train[TARGET_COLUMN].mean()*100:.2f}%)")
        print(f"\n   Test: {TEST_START_DATE} → {TEST_END_DATE}")
        print(f"   - {len(df_test):,} transactions")
        print(f"   - Fraudes: {df_test[TARGET_COLUMN].sum():,} ({df_test[TARGET_COLUMN].mean()*100:.2f}%)")
    
    return df_train, df_test


def print_data_summary(summary: dict):
    """
    Affiche un résumé formaté des données.
    
    Args:
        summary: Dictionnaire de résumé (de get_data_summary)
    """
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES DONNÉES")
    print("=" * 60)
    
    print(f"\n Dimensions:")
    print(f"   - Lignes: {summary['n_rows']:,}")
    print(f"   - Colonnes: {summary['n_cols']}")
    print(f"   - Mémoire: {summary['memory_mb']:.2f} MB")
    
    print(f"\n Valeurs manquantes:")
    print(f"   - Total: {summary['missing_total']:,}")
    if summary['missing_total'] > 0:
        missing = {k: v for k, v in summary['missing_by_column'].items() if v > 0}
        for col, count in missing.items():
            print(f"   - {col}: {count:,}")
    
    if "target_distribution" in summary:
        print(f"\n Distribution de la cible ({TARGET_COLUMN}):")
        for label, count in summary["target_distribution"].items():
            pct = count / summary["n_rows"] * 100
            label_str = "Normal" if label == 0 else "Fraude"
            print(f"   - {label} ({label_str}): {count:,} ({pct:.2f}%)")
        print(f"   - Ratio de déséquilibre: 1:{1/summary['imbalance_ratio']:.1f}")
    
    if "date_min" in summary:
        print(f"\n Période:")
        print(f"   - Début: {summary['date_min']}")
        print(f"   - Fin: {summary['date_max']}")
        print(f"   - Durée: {summary['date_range_days']} jours")
    
    print("=" * 60)


# =============================================================================
# FONCTION PRINCIPALE DE CHARGEMENT
# =============================================================================

def load_and_prepare_data(filename: str = None, verbose: bool = True) -> tuple:
    """
    Fonction principale qui charge et prépare les données.
    
    Args:
        filename: Nom du fichier de données
        verbose: Afficher les informations
        
    Returns:
        Tuple (df_train, df_test, summary)
    """
    # 1. Charger les données brutes
    df = load_raw_data(filename, verbose)
    
    # 2. Valider les colonnes
    validation = validate_columns(df)
    if not validation["valid"]:
        print(f" Colonnes manquantes: {validation['missing_columns']}")
    if validation["extra_columns"]:
        print(f" Colonnes supplémentaires: {validation['extra_columns']}")
    
    # 3. Convertir les types
    df = convert_data_types(df, verbose)
    
    # 4. Résumé des données
    summary = get_data_summary(df)
    if verbose:
        print_data_summary(summary)
    
    # 5. Séparation train/test
    df_train, df_test = split_by_date(df, verbose)
    
    return df_train, df_test, summary


# =============================================================================
# TEST DU MODULE
# =============================================================================

if __name__ == "__main__":
    print("Test du module data_loader")
    print("-" * 40)
    
    try:
        df_train, df_test, summary = load_and_prepare_data()
        print("\n Chargement réussi!")
    except FileNotFoundError as e:
        print(f"\n Erreur: {e}")
        print("\nPour tester ce module:")
        print(f"1. Placez votre fichier de données dans: {RAW_DATA_DIR}")
        print(f"2. Modifiez DATA_FILENAME dans config/config.py si nécessaire")
