# Détection de Fraudes - Transactions par Chèque

## 📋 Description du projet

Projet de Fouille de Données Massives (M2 SISE - Université Lyon 2) portant sur la détection de fraudes dans un contexte de données déséquilibrées.

**Objectifs :**
1. Construire un modèle de classification optimisant la **F-mesure**
2. Adapter le modèle pour maximiser la **marge financière** de l'enseigne

## 📁 Structure du projet

```
fraud-detection-project/
│
├── data/
│   └── raw/                    # Données brutes (non versionnées)
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_exploration.ipynb    # Analyse exploratoire
│   ├── 02_preprocessing.ipynb  # Préparation des données
│   ├── 03_modeling.ipynb       # Modélisation
│   └── 04_cost_optimization.ipynb  # Optimisation marge
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Chargement des données
│   ├── preprocessing.py        # Fonctions de prétraitement
│   ├── models.py               # Définition des modèles
│   ├── evaluation.py           # Métriques et évaluation
│   └── cost_analysis.py        # Analyse coûts/marge
│
├── reports/
│   ├── figures/                # Graphiques générés
│   └── rapport_final.pdf       # Rapport final
│
├── config/
│   └── config.py               # Configuration globale
│
├── requirements.txt            # Dépendances Python
├── .gitignore                  # Fichiers à ignorer
├── README.md                   # Ce fichier
└── main.py                     # Script principal
```

## 🔧 Installation

### 1. Cloner le repository
```bash
git clone https://github.com/Abdourahmane2/projet_Fouilles_de_Donn-es_Massives
cd projet_Fouilles_de_Donn-es_Massives
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter les données
Placer le fichier de données dans `data/raw/`

## 📊 Données

- **Source** : Enseigne de grande distribution + FNCI + Banque de France
- **Période** : 02/2017 - 11/2017
- **Variables** : 23 features
- **Target** : `FlagImpaye` (0 = normal, 1 = fraude)

### Séparation temporelle
- **Train** : 01/02/2017 - 31/08/2017
- **Test** : 01/09/2017 - 30/11/2017

## 🚀 Utilisation

### Exécuter l'analyse complète
```bash
python main.py
```

### Exécuter les notebooks
```bash
jupyter notebook notebooks/
```

## 📈 Méthodologie

### Partie 1 : Optimisation F-mesure
1. Analyse exploratoire des données
2. Prétraitement et feature engineering
3. Gestion du déséquilibre (SMOTE, ADASYN, Under-sampling)
4. Comparaison d'algorithmes (RF, XGBoost, SVM, NN...)
5. Optimisation des hyperparamètres

### Partie 2 : Optimisation de la marge
- Matrice de coûts asymétrique basée sur le montant
- Optimisation du seuil de décision

## 📝 Résultats

| Modèle | F-mesure | Precision | Recall | AUC-ROC |
|--------|----------|-----------|--------|---------|
|  | LightGBM + ADASYN  | 0.107 | - | - |

## 👥 Auteurs

- Abdourahmane Timera

## 📄 Licence

Projet académique - M2 SISE - Université Lyon 2


=======
# projet_Fouilles_de_Donn-es_Massives

