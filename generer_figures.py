"""
Génère les illustrations utilisées dans le document explicatif :
  - figures/courbe_cout.png        : l'erreur qui diminue au fil de l'entraînement
  - figures/frontiere_decision.png : la zone que le réseau classe en 0 ou en 1
  - figures/schema_reseau.png      : le schéma de l'architecture du réseau
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # backend sans écran (pour serveur)
import matplotlib.pyplot as plt

from reseau_de_neurones import ReseauDeNeurones

DOSSIER = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(DOSSIER, exist_ok=True)

# Couleurs sobres et cohérentes
BLEU = "#2563eb"
ORANGE = "#ea580c"
GRIS = "#334155"

# ---------------------------------------------------------------------------
# On (ré)entraîne un réseau identique à celui de la démonstration
# ---------------------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

reseau = ReseauDeNeurones(2, 4, 1, taux_apprentissage=0.5, graine=42)
reseau.entrainer(X, Y, n_epoques=10000, afficher_tous=0)


# ---------------------------------------------------------------------------
# Figure 1 : la courbe de coût
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
plt.plot(reseau.historique_cout, color=BLEU, linewidth=2)
plt.title("La courbe d'apprentissage : l'erreur diminue à chaque époque",
          fontsize=12, color=GRIS)
plt.xlabel("Époque (nombre de passages sur les données)")
plt.ylabel("Coût (erreur)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "courbe_cout.png"), dpi=150)
plt.close()
print("OK -> courbe_cout.png")


# ---------------------------------------------------------------------------
# Figure 2 : la frontière de décision
# ---------------------------------------------------------------------------
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300),
                     np.linspace(-0.3, 1.3, 300))
grille = np.c_[xx.ravel(), yy.ravel()]
zz = reseau.forward(grille).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.75)
plt.colorbar(label="Probabilité de sortie du réseau")

# On place les 4 points du XOR
for i in range(len(X)):
    couleur = ORANGE if Y[i, 0] == 1 else BLEU
    marqueur = "^" if Y[i, 0] == 1 else "o"
    plt.scatter(X[i, 0], X[i, 1], c=couleur, s=220, marker=marqueur,
                edgecolors="white", linewidths=2, zorder=5)
    plt.annotate(f"{int(Y[i,0])}", (X[i, 0], X[i, 1]),
                 color="white", ha="center", va="center",
                 fontsize=11, fontweight="bold", zorder=6)

plt.title("Frontière de décision apprise sur le XOR", fontsize=12, color=GRIS)
plt.xlabel("Entrée 1")
plt.ylabel("Entrée 2")
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "frontiere_decision.png"), dpi=150)
plt.close()
print("OK -> frontiere_decision.png")


# ---------------------------------------------------------------------------
# Figure 3 : le schéma de l'architecture (2 -> 4 -> 1)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

couches = [2, 4, 1]
noms = ["Entrées", "Couche cachée", "Sortie"]
positions = {}

for c, n in enumerate(couches):
    x = c * 4.0
    ys = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    for j, y in enumerate(ys):
        positions[(c, j)] = (x, y)

# Traits (connexions) d'abord, pour qu'ils passent derrière les cercles
for c in range(len(couches) - 1):
    for j in range(couches[c]):
        for k in range(couches[c + 1]):
            x1, y1 = positions[(c, j)]
            x2, y2 = positions[(c + 1, k)]
            ax.plot([x1, x2], [y1, y2], color="#cbd5e1", linewidth=1, zorder=1)

# Cercles (neurones)
couleurs_couche = [BLEU, GRIS, ORANGE]
for c, n in enumerate(couches):
    for j in range(n):
        x, y = positions[(c, j)]
        cercle = plt.Circle((x, y), 0.32, color=couleurs_couche[c],
                            zorder=2, ec="white", lw=2)
        ax.add_patch(cercle)
    x = c * 4.0
    ax.text(x, (max(couches) / 2.0) + 0.9, noms[c], ha="center",
            fontsize=12, color=GRIS, fontweight="bold")

ax.set_xlim(-1, 9)
ax.set_ylim(-2.5, 3.2)
ax.set_aspect("equal")
ax.set_title("Architecture du réseau : 2 entrées -> 4 neurones cachés -> 1 sortie",
             fontsize=12, color=GRIS)
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER, "schema_reseau.png"), dpi=150)
plt.close()
print("OK -> schema_reseau.png")


# ---------------------------------------------------------------------------
# Figure 4 : comparaison des trois fonctions d'activation
# ---------------------------------------------------------------------------
z = np.linspace(-5, 5, 400)


def relu(v):
    return np.maximum(0, v)


fonctions = [
    ("Sigmoide", 1.0 / (1.0 + np.exp(-z)), BLEU, "entre 0 et 1"),
    ("tanh", np.tanh(z), ORANGE, "entre -1 et 1"),
    ("ReLU", relu(z), "#15803d", "0 si negatif, sinon la valeur"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (nom, y, couleur, sous_titre) in zip(axes, fonctions):
    ax.plot(z, y, color=couleur, linewidth=2.5)
    ax.axhline(0, color="#94a3b8", linewidth=0.8)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_title(nom, fontsize=13, color=GRIS, fontweight="bold")
    ax.set_xlabel(sous_titre, fontsize=9, color=GRIS)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-5, 5)

axes[0].set_ylim(-0.15, 1.15)
axes[1].set_ylim(-1.2, 1.2)
axes[2].set_ylim(-0.5, 5)
fig.suptitle("Les trois fonctions d'activation a connaitre",
             fontsize=13, color=GRIS)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(DOSSIER, "activations.png"), dpi=150)
plt.close()
print("OK -> activations.png")

print("\nToutes les figures ont été générées dans le dossier 'figures/'.")
