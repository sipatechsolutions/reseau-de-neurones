# Réseau de neurones from scratch

Implémentation, uniquement avec **NumPy**, des deux mécanismes fondamentaux
d'un réseau de neurones : la **forward propagation** (propagation avant) et la
**backward propagation** (rétropropagation). Le réseau apprend à résoudre le
problème du **XOR**, l'exemple classique qu'un simple neurone ne peut pas
séparer.

## Contenu

| Fichier | Rôle |
|---|---|
| `reseau_de_neurones.py` | Le réseau complet : forward, coût, backward, entraînement. Tout est commenté pas à pas. |
| `generer_figures.py` | Produit les illustrations (courbe d'apprentissage, frontière de décision, schéma). |
| `generer_pdf.py` | Assemble le document explicatif au format PDF. |
| `comprendre_la_propagation.pdf` | Document pédagogique complet, pensé pour débuter de zéro. |
| `figures/` | Images générées. |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Entraîner le réseau et afficher les résultats :

```bash
python reseau_de_neurones.py
```

Régénérer les figures puis le PDF :

```bash
python generer_figures.py
python generer_pdf.py
```

## Résultat attendu

Après 10 000 époques, le réseau atteint **100 % d'exactitude** sur le XOR :

| Entrée | Prédiction | Attendu |
|---|---|---|
| (0, 0) | 0 | 0 |
| (0, 1) | 1 | 1 |
| (1, 0) | 1 | 1 |
| (1, 1) | 0 | 0 |

## L'architecture

```
Entrées (2)  ->  Couche cachée (4 neurones, sigmoïde)  ->  Sortie (1, sigmoïde)
```

Le document `comprendre_la_propagation.pdf` explique chaque étape mathématique
et son intuition, avec un glossaire des termes clés.
