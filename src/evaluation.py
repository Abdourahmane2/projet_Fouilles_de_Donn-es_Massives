"""
Module d'evaluation des modeles
================================

Ce module fournit les fonctions pour evaluer les performances
des modeles de classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import FIGURES_DIR


def compute_all_metrics(y_true, y_pred, y_proba=None):
    """
    Calcule toutes les metriques de classification.
    
    Args:
        y_true: Vraies etiquettes
        y_pred: Predictions (0 ou 1)
        y_proba: Probabilites de la classe positive (optionnel)
        
    Returns:
        Dictionnaire avec toutes les metriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['TP'] = tp
    metrics['TN'] = tn
    metrics['FP'] = fp
    metrics['FN'] = fn
    
    # Specificite
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Metriques basees sur les probabilites
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics


def print_metrics(metrics, model_name="Modele"):
    """
    Affiche les metriques de maniere formatee.
    """
    print(f"\n{'='*60}")
    print(f"RESULTATS: {model_name}")
    print('='*60)
    
    print(f"\n[Matrice de confusion]")
    print(f"   TP (Fraudes detectees):     {metrics['TP']:>10,}")
    print(f"   TN (Normales correctes):    {metrics['TN']:>10,}")
    print(f"   FP (Fausses alertes):       {metrics['FP']:>10,}")
    print(f"   FN (Fraudes manquees):      {metrics['FN']:>10,}")
    
    print(f"\n[Metriques principales]")
    print(f"   F1-Score:    {metrics['f1_score']:.4f}  <-- METRIQUE A OPTIMISER")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   Specificite: {metrics['specificity']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"\n[Metriques probabilistes]")
        print(f"   ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"   Avg Prec:    {metrics['avg_precision']:.4f}")
    
    print('='*60)


def plot_confusion_matrix(y_true, y_pred, model_name="Modele", save=False):
    """
    Affiche la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Normal (0)', 'Fraude (1)'],
                yticklabels=['Normal (0)', 'Fraude (1)'])
    ax.set_xlabel('Prediction', fontsize=12)
    ax.set_ylabel('Realite', fontsize=12)
    ax.set_title(f'Matrice de confusion - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name="Modele", ax=None, save=False):
    """
    Affiche la courbe ROC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('Taux de Faux Positifs (1 - Specificite)', fontsize=11)
    ax.set_ylabel('Taux de Vrais Positifs (Recall)', fontsize=11)
    ax.set_title('Courbe ROC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save and ax is None:
        plt.savefig(FIGURES_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
    
    return ax


def plot_precision_recall_curve(y_true, y_proba, model_name="Modele", ax=None, save=False):
    """
    Affiche la courbe Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})', linewidth=2)
    
    # Ligne de base (proportion de positifs)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.4f})', linewidth=1)
    
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Courbe Precision-Recall', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save and ax is None:
        plt.savefig(FIGURES_DIR / 'pr_curve.png', dpi=300, bbox_inches='tight')
    
    return ax


def compare_models(results_dict, metric='f1_score'):
    """
    Compare plusieurs modeles sur une metrique.
    
    Args:
        results_dict: Dictionnaire {nom_modele: metrics}
        metric: Metrique a comparer
        
    Returns:
        DataFrame de comparaison
    """
    comparison = []
    for name, metrics in results_dict.items():
        comparison.append({
            'Modele': name,
            'F1-Score': metrics.get('f1_score', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'Accuracy': metrics.get('accuracy', 0),
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('F1-Score', ascending=False)
    
    return df


def plot_model_comparison(comparison_df, save=False):
    """
    Visualise la comparaison des modeles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Barplot F1-Score
    models = comparison_df['Modele']
    f1_scores = comparison_df['F1-Score']
    
    colors = sns.color_palette('viridis', len(models))  
    bars = axes[0].barh(models, f1_scores, color=colors, edgecolor='black')
    axes[0].set_xlabel('F1-Score', fontsize=12)
    axes[0].set_title('Comparaison des modeles (F1-Score)', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, max(f1_scores) * 1.15)
    
    for bar, score in zip(bars, f1_scores):
        axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{score:.4f}', va='center', fontsize=10)
    
    # Barplot multi-metriques pour le meilleur modele
    best_model = comparison_df.iloc[0]['Modele']
    best_metrics = comparison_df.iloc[0][['F1-Score', 'Precision', 'Recall', 'ROC-AUC']].values
    metric_names = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']
    
    bars2 = axes[1].bar(metric_names, best_metrics, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'], 
                        edgecolor='black')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title(f'Metriques du meilleur modele: {best_model}', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    
    for bar, score in zip(bars2, best_metrics):
        axes[1].text(bar.get_x() + bar.get_width()/2, score + 0.02, 
                     f'{score:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def find_best_threshold(y_true, y_proba, metric='f1'):
    """
    Trouve le seuil optimal pour une metrique donnee.
    
    Args:
        y_true: Vraies etiquettes
        y_proba: Probabilites
        metric: 'f1', 'precision', 'recall'
        
    Returns:
        Tuple (meilleur_seuil, meilleur_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = 0 
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def plot_threshold_analysis(y_true, y_proba, model_name="Modele", save=False):
    """
    Analyse l'impact du seuil sur les metriques.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2, color='green')
    ax.plot(thresholds, precisions, label='Precision', linewidth=2, color='blue')
    ax.plot(thresholds, recalls, label='Recall', linewidth=2, color='red')
    
    # Marquer le meilleur seuil pour F1
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    ax.axvline(x=best_thresh, color='green', linestyle='--', alpha=0.7)
    ax.scatter([best_thresh], [best_f1], color='green', s=100, zorder=5)
    ax.annotate(f'Best F1: {best_f1:.4f}\nThreshold: {best_thresh:.2f}',
                xy=(best_thresh, best_f1), xytext=(best_thresh + 0.1, best_f1),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('Seuil de decision', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Impact du seuil sur les metriques - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save:
        filename = f"threshold_analysis_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return best_thresh, best_f1
