# MNIST RGB : classification par CNN et transfert d'apprentissage

Projet de S5 (2025-2026) : classification de chiffres manuscrits MNIST en version couleur, avec un réseau de neurones convolutif (CNN) puis du transfert d'apprentissage.

## Description

Chaque image MNIST (28×28, niveaux de gris) est transformée en image RGB (28×28×3) :

- le **fond** est uni, avec des composantes R, G, B tirées uniformément dans [0,2 ; 0,8] (arrondies à 2 décimales) ;
- le **chiffre** prend l'une des 8 couleurs suivantes : noir, blanc, rouge, vert, bleu, jaune, magenta, cyan.

La génération est reproductible (graine 42, fonction `make_rgb`).

Le projet traite quatre tâches :

1. classification des chiffres en **10 classes** avec un CNN ;
2. classification **pair/impair** (2 classes) avec le même type de CNN ;
3. **transfert d'apprentissage** : les features de la couche `Dense(100)` du CNN 10 classes alimentent un petit modèle pair/impair ;
4. même principe avec les features de la couche `Dense(40)`.

## Structure du dépôt

```
.
├── MNIST_RGB.ipynb                  # notebook : dataset, CNN, pair/impair, transfert, analyses
├── train_all.py                     # ré-entraînement reproductible de tous les modèles
├── requirements.txt                 # dépendances Python
├── README.md
└── 2025-2026-S5-Projet-MNIST-RGB/   # sorties : poids et paramètres de coloration
    ├── weights_cnn_10classes.weights.h5
    ├── weights_cnn_binaire.weights.h5
    ├── weights_petit_modele.weights.h5
    ├── weights_petit_modele_v2.weights.h5
    └── params_coloration.npz
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Ré-entraîner tous les modèles et régénérer les poids (15 époques) :

```bash
python train_all.py
```

Test rapide avec 2 époques :

```bash
EPOCHS=2 python train_all.py
```

Ouvrir le notebook, qui charge les poids sauvegardés (`TrainReseau=False`) :

```bash
jupyter notebook MNIST_RGB.ipynb
```

Pour ré-entraîner depuis le notebook, passer `TrainReseau` à `True` dans la cellule concernée.

## Résultats

Accuracy sur le jeu de test :

| Tâche | Accuracy |
|---|---|
| CNN, 10 classes | 97,91 % |
| CNN, pair/impair | 98,54 % |
| Transfert d'apprentissage, `Dense(100)` | 99,06 % |
| Transfert d'apprentissage, `Dense(40)` | 99,02 % |

Les erreurs du CNN 10 classes viennent surtout d'ambiguïtés entre certains chiffres : 3 et 8, 7 et 2, 4 et 9.

Ce projet a été réalisé au S5 2025-2026 (rendu en janvier 2026) et publié sur GitHub en octobre 2026.

Projet réalisé avec Gaspard Beaudic
