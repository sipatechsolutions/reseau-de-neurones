"""
Réseau de neurones from scratch (uniquement avec NumPy)
=======================================================

Ce fichier implémente, sans aucune bibliothèque de deep learning,
les deux mécanismes fondamentaux d'un réseau de neurones :

    1. La FORWARD PROPAGATION  (propagation avant)
       -> on part des entrées et on calcule une prédiction.

    2. La BACKWARD PROPAGATION  (rétropropagation)
       -> on mesure l'erreur et on calcule, couche par couche,
          comment corriger chaque poids pour réduire cette erreur.

Le réseau utilisé ici est un perceptron multicouche (MLP) :

        Entrées (2)  ->  Couche cachée (n neurones)  ->  Sortie (1)

Tout est commenté ligne par ligne pour rester accessible à un débutant.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. FONCTIONS D'ACTIVATION
# ---------------------------------------------------------------------------
# Une fonction d'activation introduit de la "non-linéarité".
# Sans elle, empiler des couches reviendrait à une simple multiplication
# de matrices : le réseau ne pourrait apprendre que des relations linéaires.

def sigmoid(z):
    """Transforme n'importe quel nombre réel en une valeur entre 0 et 1.

    sigma(z) = 1 / (1 + e^{-z})
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivee(a):
    """Dérivée de la sigmoïde, exprimée à partir de sa sortie 'a'.

    Si a = sigma(z), alors sigma'(z) = a * (1 - a).
    C'est cette dérivée qui indique "de combien" la sortie change
    quand on bouge un peu l'entrée : indispensable pour la rétropropagation.
    """
    return a * (1.0 - a)


# ---------------------------------------------------------------------------
# 2. LA CLASSE DU RESEAU
# ---------------------------------------------------------------------------
class ReseauDeNeurones:
    """Un réseau à une couche cachée, entièrement écrit à la main."""

    def __init__(self, n_entrees, n_caches, n_sorties, taux_apprentissage=0.5,
                 graine=42):
        """Initialise les poids et les biais du réseau.

        Paramètres
        ----------
        n_entrees : nombre de caractéristiques en entrée (ici 2)
        n_caches  : nombre de neurones dans la couche cachée
        n_sorties : nombre de neurones en sortie (ici 1)
        taux_apprentissage : "vitesse" à laquelle on corrige les poids
        graine    : fixe le hasard pour obtenir des résultats reproductibles
        """
        rng = np.random.default_rng(graine)

        # Les poids sont initialisés avec de petites valeurs aléatoires.
        # W1 relie les entrees a la couche cachee : forme (n_entrees, n_caches)
        # W2 relie la couche cachee a la sortie    : forme (n_caches, n_sorties)
        self.W1 = rng.normal(0, 1, size=(n_entrees, n_caches))
        self.b1 = np.zeros((1, n_caches))

        self.W2 = rng.normal(0, 1, size=(n_caches, n_sorties))
        self.b2 = np.zeros((1, n_sorties))

        self.taux_apprentissage = taux_apprentissage

        # On garde l'historique de l'erreur pour tracer la courbe d'apprentissage
        self.historique_cout = []

    # -------------------------------------------------------------------
    # 2.a  FORWARD PROPAGATION
    # -------------------------------------------------------------------
    def forward(self, X):
        """Calcule la prédiction du réseau pour un lot d'exemples X.

        X a la forme (m, n_entrees) ou m = nombre d'exemples.

        Étapes :
            Z1 = X . W1 + b1        (combinaison linéaire, couche cachée)
            A1 = sigmoid(Z1)        (activation de la couche cachée)
            Z2 = A1 . W2 + b2       (combinaison linéaire, couche de sortie)
            A2 = sigmoid(Z2)        (prédiction finale, entre 0 et 1)
        """
        self.X = X                                   # on garde X pour la backprop

        self.Z1 = X @ self.W1 + self.b1              # @ = produit matriciel
        self.A1 = sigmoid(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2

    # -------------------------------------------------------------------
    # 2.b  FONCTION DE COUT
    # -------------------------------------------------------------------
    @staticmethod
    def cout(Y_vrai, Y_pred):
        """Entropie croisée binaire : mesure l'écart entre la vérité et la prédiction.

        L = -(1/m) * somme[ y*log(p) + (1-y)*log(1-p) ]

        Plus la prédiction est proche de la vérité, plus L est petit.
        """
        m = Y_vrai.shape[0]
        eps = 1e-9                                    # évite log(0)
        cout = -np.mean(
            Y_vrai * np.log(Y_pred + eps)
            + (1 - Y_vrai) * np.log(1 - Y_pred + eps)
        )
        return cout

    # -------------------------------------------------------------------
    # 2.c  BACKWARD PROPAGATION
    # -------------------------------------------------------------------
    def backward(self, Y):
        """Calcule les gradients de l'erreur par rapport à chaque poids.

        On applique la règle de la chaîne (dérivation en cascade),
        de la sortie vers l'entrée.

        Astuce : avec une sortie sigmoïde ET une entropie croisée,
        la première dérivée se simplifie magnifiquement en :
            dZ2 = A2 - Y
        """
        m = Y.shape[0]                                # nombre d'exemples

        # ---- Couche de sortie -------------------------------------------------
        dZ2 = self.A2 - Y                            # (m, n_sorties)
        dW2 = (self.A1.T @ dZ2) / m                  # gradient des poids W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m # gradient des biais b2

        # ---- Couche cachée ----------------------------------------------------
        dA1 = dZ2 @ self.W2.T                        # on "renvoie" l'erreur en arrière
        dZ1 = dA1 * sigmoid_derivee(self.A1)         # on traverse l'activation
        dW1 = (self.X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # ---- Descente de gradient : on corrige les poids ----------------------
        # Chaque poids se déplace dans la direction opposée à son gradient,
        # d'un pas proportionnel au taux d'apprentissage.
        self.W2 -= self.taux_apprentissage * dW2
        self.b2 -= self.taux_apprentissage * db2
        self.W1 -= self.taux_apprentissage * dW1
        self.b1 -= self.taux_apprentissage * db1

    # -------------------------------------------------------------------
    # 2.d  ENTRAINEMENT
    # -------------------------------------------------------------------
    def entrainer(self, X, Y, n_epoques=10000, afficher_tous=1000):
        """Répète forward + backward pour faire baisser l'erreur.

        Une "époque" = un passage complet forward puis backward sur les données.
        """
        for epoque in range(n_epoques):
            Y_pred = self.forward(X)                 # 1) prédiction
            cout = self.cout(Y, Y_pred)              # 2) mesure de l'erreur
            self.historique_cout.append(cout)
            self.backward(Y)                         # 3) correction des poids

            if afficher_tous and epoque % afficher_tous == 0:
                print(f"Époque {epoque:5d} | coût = {cout:.6f}")

        cout_final = self.cout(Y, self.forward(X))
        print(f"Époque {n_epoques:5d} | coût = {cout_final:.6f}  (final)")

    def predire(self, X, seuil=0.5):
        """Renvoie 0 ou 1 selon que la probabilité dépasse le seuil."""
        proba = self.forward(X)
        return (proba >= seuil).astype(int)


# ---------------------------------------------------------------------------
# 3. DEMONSTRATION : LE PROBLEME DU XOR
# ---------------------------------------------------------------------------
# XOR ("OU exclusif") renvoie 1 quand les deux entrées sont DIFFERENTES.
# C'est l'exemple historique qu'un simple neurone ne peut PAS résoudre,
# mais qu'un réseau avec une couche cachée résout parfaitement.
#
#     entrée      sortie attendue
#     (0, 0)  ->      0
#     (0, 1)  ->      1
#     (1, 0)  ->      1
#     (1, 1)  ->      0

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)

    Y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=float)

    print("=" * 60)
    print("Entraînement d'un réseau de neurones sur le problème XOR")
    print("=" * 60)

    reseau = ReseauDeNeurones(n_entrees=2, n_caches=4, n_sorties=1,
                              taux_apprentissage=0.5, graine=42)
    reseau.entrainer(X, Y, n_epoques=10000, afficher_tous=1000)

    print("\nRésultats après apprentissage :")
    print("-" * 60)
    proba = reseau.forward(X)
    predictions = reseau.predire(X)
    for i in range(len(X)):
        print(f"  Entrée {X[i].astype(int)}  ->  "
              f"probabilité = {proba[i, 0]:.3f}  ->  "
              f"prédiction = {predictions[i, 0]}  "
              f"(attendu {int(Y[i, 0])})")

    exactitude = np.mean(predictions == Y) * 100
    print("-" * 60)
    print(f"Exactitude : {exactitude:.0f} %")
