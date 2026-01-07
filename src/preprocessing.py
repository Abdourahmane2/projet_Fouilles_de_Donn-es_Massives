"""
Module de preprocessing des donnees
====================================

Ce module fournit les fonctions pour preparer les donnees
avant la modelisation : selection des features, normalisation,
gestion du desequilibre, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    TARGET_COLUMN,
    COLUMNS_TO_EXCLUDE,
    RANDOM_STATE,
    PROCESSED_DATA_DIR,
)


# =============================================================================
# SELECTION DES FEATURES
# =============================================================================

def get_feature_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Retourne la liste des colonnes a utiliser comme features.
    
    Args:
        df: DataFrame source
        exclude_cols: Colonnes a exclure (utilise COLUMNS_TO_EXCLUDE par defaut)
        
    Returns:
        Liste des noms de colonnes features
    """
    if exclude_cols is None:
        exclude_cols = COLUMNS_TO_EXCLUDE
    
    # Ajouter la cible aux colonnes a exclure
    exclude_set = set(exclude_cols) | {TARGET_COLUMN}
    
    # Colonnes temporelles creees lors de l'exploration
    temp_cols = {'Date', 'Month', 'DayOfWeek', 'Hour'}
    exclude_set = exclude_set | temp_cols
    
    # Selectionner les colonnes numeriques non exclues
    feature_cols = [col for col in df.columns 
                    if col not in exclude_set 
                    and df[col].dtype in ['int64', 'float64', 'int32', 'float32' , 'int']]
    
    print("COLUMNS_TO_EXCLUDE:", COLUMNS_TO_EXCLUDE)
    print("TARGET_COLUMN:", TARGET_COLUMN)
    
    
    return feature_cols


def prepare_features_target(df: pd.DataFrame, 
                           feature_cols: list = None,
                           verbose: bool = True) -> tuple:
    """
    Separe les features et la variable cible.
    
    Args:
        df: DataFrame source
        feature_cols: Liste des colonnes features (auto-detecte si None)
        verbose: Afficher les informations
        
    Returns:
        Tuple (X, y) avec X = features, y = cible
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    
    if verbose:
        print(f"Features selectionnees ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"   {i:2d}. {col}")
        print(f"\nDimensions: X = {X.shape}, y = {y.shape}")
    
    return X, y


# =============================================================================
# ANALYSE DES FEATURES
# =============================================================================

def analyze_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse detaillee des features.
    
    Args:
        X: DataFrame des features
        
    Returns:
        DataFrame avec les statistiques de chaque feature
    """
    stats = []
    
    for col in X.columns:
        col_stats = {
            'feature': col,
            'dtype': str(X[col].dtype),
            'missing': X[col].isnull().sum(),
            'missing_pct': X[col].isnull().sum() / len(X) * 100,
            'unique': X[col].nunique(),
            'min': X[col].min(),
            'max': X[col].max(),
            'mean': X[col].mean(),
            'std': X[col].std(),
            'median': X[col].median(),
            'skew': X[col].skew(),
            'kurtosis': X[col].kurtosis(),
            'zeros': (X[col] == 0).sum(),
            'zeros_pct': (X[col] == 0).sum() / len(X) * 100,
        }
        stats.append(col_stats)
    
    return pd.DataFrame(stats)


def detect_outliers_iqr(X: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detecte les outliers avec la methode IQR.
    
    Args:
        X: DataFrame des features
        factor: Facteur IQR (1.5 = standard, 3 = extreme)
        
    Returns:
        DataFrame avec le nombre d'outliers par feature
    """
    outlier_stats = []
    
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        
        outlier_stats.append({
            'feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': outliers,
            'pct_outliers': outliers / len(X) * 100
        })
    
    return pd.DataFrame(outlier_stats)


# =============================================================================
# NORMALISATION / STANDARDISATION
# =============================================================================

def scale_features(X_train: pd.DataFrame, 
                   X_test: pd.DataFrame,
                   method: str = 'standard',
                   verbose: bool = True) -> tuple:
    """
    Normalise/Standardise les features.
    
    Args:
        X_train: Features d'entrainement
        X_test: Features de test
        method: 'standard', 'minmax', ou 'robust'
        verbose: Afficher les informations
        
    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Methode inconnue: {method}")
    
    # Fit sur train, transform sur train et test
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if verbose:
        print(f"Scaling applique: {method}")
        print(f"   X_train: {X_train_scaled.shape}")
        print(f"   X_test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# CREATION D'UN SET DE VALIDATION
# =============================================================================

def create_validation_split(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           val_size: float = 0.2,
                           stratify: bool = True,
                           verbose: bool = True) -> tuple:
    """
    Cree un ensemble de validation a partir du train.
    
    Args:
        X_train: Features d'entrainement
        y_train: Cible d'entrainement
        val_size: Proportion pour la validation
        stratify: Stratifier selon la cible
        verbose: Afficher les informations
        
    Returns:
        Tuple (X_train_new, X_val, y_train_new, y_val)
    """
    stratify_param = y_train if stratify else None
    
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=stratify_param,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"Split train/validation ({1-val_size:.0%}/{val_size:.0%}):")
        print(f"   Train: {len(X_train_new):,} ({y_train_new.mean()*100:.2f}% fraudes)")
        print(f"   Val:   {len(X_val):,} ({y_val.mean()*100:.2f}% fraudes)")
    
    return X_train_new, X_val, y_train_new, y_val


# =============================================================================
# STATISTIQUES PAR CLASSE
# =============================================================================

def compute_class_statistics(X: pd.DataFrame, 
                            y: pd.Series,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Calcule les statistiques des features par classe.
    
    Args:
        X: Features
        y: Cible
        verbose: Afficher les informations
        
    Returns:
        DataFrame avec les stats par classe
    """
    # Combiner X et y
    df_combined = X.copy()
    df_combined[TARGET_COLUMN] = y
    
    # Stats par classe
    stats_by_class = df_combined.groupby(TARGET_COLUMN).agg(['mean', 'std', 'median'])
    
    # Calculer la difference entre classes
    diff_stats = []
    for col in X.columns:
        mean_0 = stats_by_class.loc[0, (col, 'mean')]
        mean_1 = stats_by_class.loc[1, (col, 'mean')]
        std_0 = stats_by_class.loc[0, (col, 'std')]
        std_1 = stats_by_class.loc[1, (col, 'std')]
        
        # Difference relative des moyennes
        if mean_0 != 0:
            diff_pct = (mean_1 - mean_0) / abs(mean_0) * 100
        else:
            diff_pct = np.inf if mean_1 != 0 else 0
        
        diff_stats.append({
            'feature': col,
            'mean_normal': mean_0,
            'mean_fraud': mean_1,
            'diff_pct': diff_pct,
            'std_normal': std_0,
            'std_fraud': std_1,
        })
    
    diff_df = pd.DataFrame(diff_stats)
    diff_df = diff_df.sort_values('diff_pct', key=abs, ascending=False)
    
    if verbose:
        print("\nDifferences de moyennes entre classes (triees par importance):")
        print("-" * 70)
        for _, row in diff_df.head(10).iterrows():
            print(f"   {row['feature']:25s} | Normal: {row['mean_normal']:10.2f} | "
                  f"Fraud: {row['mean_fraud']:10.2f} | Diff: {row['diff_pct']:+8.1f}%")
    
    return diff_df


# =============================================================================
# SAUVEGARDE DES DONNEES PREPROCESSEES
# =============================================================================

def save_preprocessed_data(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: pd.Series,
                          y_test: pd.Series,
                          suffix: str = '',
                          verbose: bool = True) -> dict:
    """
    Sauvegarde les donnees preprocessees.
    
    Args:
        X_train, X_test, y_train, y_test: Donnees a sauvegarder
        suffix: Suffixe pour les noms de fichiers
        verbose: Afficher les informations
        
    Returns:
        Dictionnaire avec les chemins des fichiers sauvegardes
    """
    suffix_str = f"_{suffix}" if suffix else ""
    
    paths = {
        'X_train': PROCESSED_DATA_DIR / f'X_train{suffix_str}.pkl',
        'X_test': PROCESSED_DATA_DIR / f'X_test{suffix_str}.pkl',
        'y_train': PROCESSED_DATA_DIR / f'y_train{suffix_str}.pkl',
        'y_test': PROCESSED_DATA_DIR / f'y_test{suffix_str}.pkl',
    }
    
    X_train.to_pickle(paths['X_train'])
    X_test.to_pickle(paths['X_test'])
    y_train.to_pickle(paths['y_train'])
    y_test.to_pickle(paths['y_test'])
    
    if verbose:
        print(f"\nDonnees sauvegardees:")
        for name, path in paths.items():
            print(f"   {name}: {path}")
    
    return paths


def load_preprocessed_data(suffix: str = '', verbose: bool = True) -> tuple:
    """
    Charge les donnees preprocessees.
    
    Args:
        suffix: Suffixe des fichiers
        verbose: Afficher les informations
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    suffix_str = f"_{suffix}" if suffix else ""
    
    X_train = pd.read_pickle(PROCESSED_DATA_DIR / f'X_train{suffix_str}.pkl')
    X_test = pd.read_pickle(PROCESSED_DATA_DIR / f'X_test{suffix_str}.pkl')
    y_train = pd.read_pickle(PROCESSED_DATA_DIR / f'y_train{suffix_str}.pkl')
    y_test = pd.read_pickle(PROCESSED_DATA_DIR / f'y_test{suffix_str}.pkl')
    
    if verbose:
        print(f"Donnees chargees:")
        print(f"   X_train: {X_train.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_train: {y_train.shape} ({y_train.mean()*100:.2f}% fraudes)")
        print(f"   y_test: {y_test.shape} ({y_test.mean()*100:.2f}% fraudes)")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# FONCTION PRINCIPALE DE PREPROCESSING
# =============================================================================

def preprocess_pipeline(df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       scale_method: str = None,
                       create_validation: bool = False,
                       val_size: float = 0.2,
                       verbose: bool = True) -> dict:
    """
    Pipeline complet de preprocessing.
    
    Args:
        df_train: DataFrame d'entrainement
        df_test: DataFrame de test
        scale_method: Methode de scaling ('standard', 'minmax', 'robust', None)
        create_validation: Creer un set de validation
        val_size: Taille du set de validation
        verbose: Afficher les informations
        
    Returns:
        Dictionnaire avec toutes les donnees preprocessees
    """
    if verbose:
        print("=" * 60)
        print("PIPELINE DE PREPROCESSING")
        print("=" * 60)
    
    # 1. Selection des features
    if verbose:
        print("\n[1/4] Selection des features...")
    feature_cols = get_feature_columns(df_train)
    
    X_train, y_train = prepare_features_target(df_train, feature_cols, verbose)
    X_test, y_test = prepare_features_target(df_test, feature_cols, verbose=False)
    
    # 2. Analyse des features
    if verbose:
        print("\n[2/4] Analyse des features...")
    feature_stats = analyze_features(X_train)
    class_stats = compute_class_statistics(X_train, y_train, verbose)
    
    # 3. Scaling (optionnel)
    scaler = None
    if scale_method:
        if verbose:
            print(f"\n[3/4] Scaling ({scale_method})...")
        X_train, X_test, scaler = scale_features(X_train, X_test, scale_method, verbose)
    else:
        if verbose:
            print("\n[3/4] Scaling: Non applique")
    
    # 4. Validation split (optionnel)
    X_val, y_val = None, None
    if create_validation:
        if verbose:
            print(f"\n[4/4] Creation du set de validation...")
        X_train, X_val, y_train, y_val = create_validation_split(
            X_train, y_train, val_size, verbose=verbose
        )
    else:
        if verbose:
            print("\n[4/4] Set de validation: Non cree")
    
    if verbose:
        print("\n" + "=" * 60)
        print("PREPROCESSING TERMINE")
        print("=" * 60)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_val': X_val,
        'y_val': y_val,
        'feature_cols': feature_cols,
        'feature_stats': feature_stats,
        'class_stats': class_stats,
        'scaler': scaler,
    }


# =============================================================================
# TEST DU MODULE
# =============================================================================

if __name__ == "__main__":
    print("Test du module preprocessing")
    print("-" * 40)
    
    # Charger les donnees
    try:
        df_train = pd.read_pickle(PROCESSED_DATA_DIR / 'df_train.pkl')
        df_test = pd.read_pickle(PROCESSED_DATA_DIR / 'df_test.pkl')
        
        # Executer le pipeline
        result = preprocess_pipeline(df_train, df_test, scale_method='standard')
        
        print("\nTest reussi!")
        
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        print("Executez d'abord le notebook 01_exploration.ipynb")
