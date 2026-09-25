# Fiche de présentation — Réseau de neurones « from scratch »

## Objectif
Implémenter, **sans aucune bibliothèque de deep learning**, les deux mécanismes
au cœur de tout réseau de neurones : la **propagation avant** (forward) et la
**rétropropagation** (backward). Le but est de comprendre *comment* un réseau
apprend, et non de simplement appeler une fonction toute faite.

## Ce que fait le projet
Un réseau à une couche cachée apprend à résoudre le **XOR** (« OU exclusif »),
l'exemple historique qu'un simple neurone ne peut pas séparer.

## Méthode (en 3 idées)
1. **Forward** : les entrées traversent le réseau (multiplication par les poids →
   addition du biais → fonction d'activation) pour produire une prédiction.
2. **Coût** : on mesure l'erreur entre la prédiction et la bonne réponse.
3. **Backward** : on calcule, couche par couche (règle de la chaîne), comment
   corriger chaque poids, puis on les ajuste (descente de gradient). On répète.

## Résultat
Parti de poids aléatoires, le réseau atteint **100 % d'exactitude** sur le XOR
après 10 000 itérations. La courbe de coût et la frontière de décision sont dans
le dossier `figures/`.

## Comment lancer
```bash
pip install -r requirements.txt
python reseau_de_neurones.py     # entraînement + résultats
python generer_figures.py        # (optionnel) régénère les figures
```

## Pour aller plus loin
Le document **`comprendre_la_propagation.pdf`** explique chaque étape et chaque
formule, avec un glossaire — idéal pour suivre pas à pas.

## À dire à l'oral (points clés)
- Un neurone = somme pondérée + fonction d'activation ; c'est l'activation
  (non-linéaire) qui donne au réseau sa puissance.
- Forward = « deviner » ; backward = « apprendre de son erreur ».
- La rétropropagation repose sur la **règle de la chaîne** (dérivation en cascade).
- Le XOR démontre l'utilité d'une couche cachée.
