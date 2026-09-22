# -*- coding: utf-8 -*-
"""
Construit le document PDF explicatif :  comprendre_la_propagation.pdf

Le texte est écrit pour un grand débutant. Les équations sont rendues en
belle qualité mathématique (via matplotlib) puis assemblées avec reportlab.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---------------------------------------------------------------------------
DOSSIER = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(DOSSIER, "figures")
EQ = os.path.join(FIG, "equations")
os.makedirs(EQ, exist_ok=True)

# Palette
BLEU = HexColor("#1d4ed8")
BLEU_CLAIR = HexColor("#eff6ff")
GRIS = HexColor("#334155")
GRIS_CLAIR = HexColor("#f1f5f9")
ORANGE = HexColor("#ea580c")
NOIR = HexColor("#0f172a")
VERT_FONCE = HexColor("#166534")
VERT_CLAIR = HexColor("#f0fdf4")


# ---------------------------------------------------------------------------
# Rendu des équations en images PNG
# ---------------------------------------------------------------------------
def equation(nom, latex, taille=20, couleur="#0f172a"):
    """Rend une formule LaTeX en image PNG et renvoie son chemin."""
    chemin = os.path.join(EQ, nom + ".png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=taille, color=couleur)
    fig.savefig(chemin, dpi=200, bbox_inches="tight",
                pad_inches=0.12, transparent=True)
    plt.close(fig)
    return chemin


# On prépare toutes les équations dont on aura besoin
EQS = {
    "neurone": r"z = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b",
    "sigmoid": r"\sigma(z) = \dfrac{1}{1 + e^{-z}}",
    "activation": r"a = \sigma(z)",
    "tanh": r"\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}",
    "relu": r"\mathrm{ReLU}(z) = \max(0,\; z)",
    "fwd1": r"Z^{[1]} = X \cdot W^{[1]} + b^{[1]}",
    "fwd2": r"A^{[1]} = \sigma\left(Z^{[1]}\right)",
    "fwd3": r"Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}",
    "fwd4": r"A^{[2]} = \sigma\left(Z^{[2]}\right) = \hat{y}",
    "cout": r"L = -\frac{1}{m}\sum_{i=1}^{m}\left[\,y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\,\right]",
    "gradient": r"w \leftarrow w - \eta\,\frac{\partial L}{\partial w}",
    "chaine": r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}",
    "sig_deriv": r"\sigma'(z) = \sigma(z)\,\left(1 - \sigma(z)\right) = a\,(1-a)",
    "bwd1": r"dZ^{[2]} = A^{[2]} - Y",
    "bwd2": r"dW^{[2]} = \frac{1}{m}\,{A^{[1]}}^{\top} \cdot dZ^{[2]}",
    "bwd3": r"db^{[2]} = \frac{1}{m}\sum dZ^{[2]}",
    "bwd4": r"dZ^{[1]} = \left(dZ^{[2]} \cdot {W^{[2]}}^{\top}\right)\, \odot\, \sigma'\!\left(Z^{[1]}\right)",
    "bwd5": r"dW^{[1]} = \frac{1}{m}\,X^{\top} \cdot dZ^{[1]}",
    "bwd6": r"db^{[1]} = \frac{1}{m}\sum dZ^{[1]}",
}
CHEMINS_EQ = {k: equation(k, v) for k, v in EQS.items()}


# ---------------------------------------------------------------------------
# Styles de texte
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

style_titre = ParagraphStyle(
    "TitrePrincipal", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=30, leading=36, textColor=NOIR, alignment=TA_CENTER, spaceAfter=6)

style_sous_titre = ParagraphStyle(
    "SousTitre", parent=styles["Normal"], fontName="Helvetica",
    fontSize=14, leading=20, textColor=GRIS, alignment=TA_CENTER, spaceAfter=4)

style_h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=18, leading=24, textColor=BLEU, spaceBefore=18, spaceAfter=10)

style_h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=13.5, leading=18, textColor=NOIR, spaceBefore=12, spaceAfter=6)

style_corps = ParagraphStyle(
    "Corps", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=16.5, textColor=NOIR, alignment=TA_JUSTIFY,
    spaceAfter=8)

style_puce = ParagraphStyle(
    "Puce", parent=style_corps, leftIndent=16, bulletIndent=4, spaceAfter=4)

style_legende = ParagraphStyle(
    "Legende", parent=styles["Normal"], fontName="Helvetica-Oblique",
    fontSize=9.5, leading=13, textColor=GRIS, alignment=TA_CENTER,
    spaceBefore=4, spaceAfter=10)

style_code = ParagraphStyle(
    "Code", parent=styles["Code"], fontName="Courier", fontSize=8.6,
    leading=11.5, textColor=NOIR)

style_encadre = ParagraphStyle(
    "Encadre", parent=style_corps, spaceAfter=0)


# ---------------------------------------------------------------------------
# Helpers de mise en page
# ---------------------------------------------------------------------------
def img_equation(nom, largeur_cm=None, hauteur_cm=1.0):
    """Insère une équation centrée, dimensionnée par sa hauteur."""
    chemin = CHEMINS_EQ[nom]
    from PIL import Image as PILImage
    try:
        w, h = PILImage.open(chemin).size
    except Exception:
        w, h = (400, 100)
    ratio = w / h
    hauteur = hauteur_cm * cm
    largeur = hauteur * ratio
    img = Image(chemin, width=largeur, height=hauteur)
    img.hAlign = "CENTER"
    return img


def bloc_equation(nom, hauteur_cm=1.0, fond=BLEU_CLAIR):
    """Équation centrée dans un léger bandeau coloré."""
    img = img_equation(nom, hauteur_cm=hauteur_cm)
    t = Table([[img]], colWidths=[16 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fond),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def encadre(titre, texte, fond=VERT_CLAIR, bordure=VERT_FONCE, titre_couleur=VERT_FONCE):
    """Encadré 'à retenir' / 'intuition'."""
    para_titre = Paragraph(
        f'<font color="#{titre_couleur.hexval()[2:]}"><b>{titre}</b></font>',
        ParagraphStyle("et", parent=style_corps, fontName="Helvetica-Bold",
                       fontSize=11.5, spaceAfter=4))
    para_txt = Paragraph(texte, style_encadre)
    t = Table([[para_titre], [para_txt]], colWidths=[15.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fond),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bordure),
    ]))
    return t


def para(txt):
    return Paragraph(txt, style_corps)


def puce(txt):
    return Paragraph(txt, style_puce, bulletText="•")


def image_fig(nom_fichier, largeur_cm=13.5, legende=None):
    chemin = os.path.join(FIG, nom_fichier)
    from PIL import Image as PILImage
    w, h = PILImage.open(chemin).size
    largeur = largeur_cm * cm
    hauteur = largeur * h / w
    img = Image(chemin, width=largeur, height=hauteur)
    img.hAlign = "CENTER"
    elems = [img]
    if legende:
        elems.append(Paragraph(legende, style_legende))
    return KeepTogether(elems)


# ---------------------------------------------------------------------------
# Construction du contenu
# ---------------------------------------------------------------------------
histoire = []
A = histoire.append


# ---- Page de titre --------------------------------------------------------
A(Spacer(1, 3.2 * cm))
A(Paragraph("Comprendre la propagation<br/>dans un réseau de neurones", style_titre))
A(Spacer(1, 0.3 * cm))
A(Paragraph("Forward propagation & backward propagation, expliquées pas à pas",
            style_sous_titre))
A(Spacer(1, 0.2 * cm))
A(Paragraph("Une implémentation from scratch en Python, avec NumPy uniquement",
            style_sous_titre))
A(Spacer(1, 1.4 * cm))
A(image_fig("schema_reseau.png", largeur_cm=15))
A(Spacer(1, 1.0 * cm))
A(Paragraph(
    "Ce document part de zéro : aucune connaissance préalable en apprentissage "
    "automatique n'est nécessaire. Chaque terme est défini au moment où il apparaît.",
    ParagraphStyle("intro_titre", parent=style_corps, alignment=TA_CENTER,
                   textColor=GRIS, fontSize=11)))
A(PageBreak())


# ---- 0. Sommaire ----------------------------------------------------------
A(Paragraph("Sommaire", style_h1))
sommaire = [
    "1.  L'idée générale : qu'est-ce qu'un réseau de neurones ?",
    "2.  Un neurone, brique de base",
    "3.  La fonction d'activation",
    "4.  Les trois fonctions d'activation à connaître : sigmoïde, tanh, ReLU",
    "5.  La forward propagation (propagation avant)",
    "6.  Mesurer l'erreur : la fonction de coût",
    "7.  Apprendre = descendre la pente (descente de gradient)",
    "8.  La backward propagation (rétropropagation)",
    "9.  L'algorithme complet, en une page",
    "10. Exemple concret : le problème du XOR",
    "11. Le code Python commenté",
    "12. Glossaire des termes importants",
]
for s in sommaire:
    A(Paragraph(s, ParagraphStyle("som", parent=style_corps, spaceAfter=6,
                                  leftIndent=6)))
A(PageBreak())


# ---- 1. L'idée générale ---------------------------------------------------
A(Paragraph("1. L'idée générale", style_h1))
A(para(
    "Un <b>réseau de neurones</b> est une machine à faire des prédictions. "
    "On lui donne des <b>entrées</b> (par exemple les pixels d'une image, "
    "ou deux nombres 0 et 1) et il produit une <b>sortie</b> "
    "(par exemple « c'est un chat » ou « la réponse est 1 »)."))
A(para(
    "Au départ, le réseau se trompe presque tout le temps, car il est rempli de "
    "nombres au hasard appelés <b>poids</b>. Tout l'enjeu de "
    "l'<b>entraînement</b> est de corriger progressivement ces poids pour que "
    "les prédictions deviennent justes. Ce processus repose sur deux mouvements "
    "qui se répètent des milliers de fois :"))
A(puce("<b>La forward propagation</b> : on pousse les entrées à travers le "
       "réseau pour obtenir une prédiction."))
A(puce("<b>La backward propagation</b> : on compare la prédiction à la bonne "
       "réponse, puis on remonte le réseau à l'envers pour savoir comment "
       "corriger chaque poids."))
A(Spacer(1, 0.2 * cm))
A(encadre("En une phrase",
          "Forward = <b>deviner</b>. Backward = <b>apprendre de son erreur</b>. "
          "On répète, et le réseau devient de plus en plus juste."))
A(Spacer(1, 0.3 * cm))
A(para(
    "Imagez un archer : il tire une flèche (forward), regarde de combien il a "
    "raté la cible (l'erreur), puis ajuste son geste pour le prochain tir "
    "(backward). Après beaucoup d'essais, il vise juste."))


# ---- 2. Un neurone --------------------------------------------------------
A(Paragraph("2. Un neurone, brique de base", style_h1))
A(para(
    "Un <b>neurone</b> reçoit plusieurs nombres en entrée. Il multiplie chaque "
    "entrée par un <b>poids</b> (son importance), additionne le tout, puis "
    "ajoute un petit décalage appelé <b>biais</b>. On obtient un nombre "
    "intermédiaire noté <i>z</i> :"))
A(bloc_equation("neurone", hauteur_cm=0.75))
A(para(
    "Ici, les <i>x</i> sont les entrées, les <i>w</i> (weights) sont les poids, "
    "et <i>b</i> (bias) est le biais. C'est une simple <b>somme pondérée</b> : "
    "les poids disent à quel point chaque entrée compte, en bien (poids positif) "
    "ou en mal (poids négatif)."))
A(encadre("À retenir",
          "Un neurone, c'est juste une multiplication suivie d'une addition. "
          "Rien de magique : ce sont les <b>bonnes valeurs</b> des poids "
          "(trouvées pendant l'entraînement) qui rendent le réseau intelligent.",
          fond=BLEU_CLAIR, bordure=BLEU, titre_couleur=BLEU))


# ---- 3. Activation --------------------------------------------------------
A(Paragraph("3. La fonction d'activation", style_h1))
A(para(
    "Si on empilait uniquement des sommes pondérées, tout le réseau resterait "
    "une grosse opération linéaire (une droite), incapable d'apprendre des "
    "formes compliquées. Pour lui donner de la souplesse, on fait passer <i>z</i> "
    "dans une <b>fonction d'activation</b>. La plus classique est la "
    "<b>sigmoïde</b> :"))
A(bloc_equation("sigmoid", hauteur_cm=1.1))
A(para(
    "La sigmoïde écrase n'importe quel nombre dans l'intervalle "
    "<b>]0 ; 1[</b>. Un grand nombre positif donne une valeur proche de 1, "
    "un grand nombre négatif une valeur proche de 0, et 0 donne exactement 0,5. "
    "C'est pratique : la sortie ressemble alors à une <b>probabilité</b>."))
A(para("La valeur produite par le neurone après activation est notée <i>a</i> :"))
A(bloc_equation("activation", hauteur_cm=0.7))
A(encadre("Pourquoi c'est indispensable",
          "Sans activation, mille couches ne valent pas mieux qu'une seule. "
          "L'activation (non-linéaire) est ce qui permet au réseau de tracer des "
          "frontières courbes et de résoudre des problèmes que ligne droite ne "
          "peut pas séparer.", fond=BLEU_CLAIR, bordure=BLEU, titre_couleur=BLEU))


# ---- 4. Les fonctions d'activation à connaître ----------------------------
A(PageBreak())
A(Paragraph("4. Les trois fonctions d'activation à connaître", style_h1))
A(para(
    "On vient de voir la sigmoïde. En pratique, la fonction d'activation se "
    "choisit dans une petite « boîte à outils ». Voici les <b>trois noms</b> "
    "que vous croiserez tout le temps dans un cours ou un article : il faut "
    "savoir les reconnaître et comprendre à quoi chacune sert. Ce sont "
    "simplement trois façons différentes de « plier » le nombre <i>z</i> "
    "produit par un neurone."))

A(Paragraph("La sigmoïde", style_h2))
A(bloc_equation("sigmoid", hauteur_cm=1.0))
A(para(
    "<b>Se prononce</b> « sig-mo-ïde ». Elle écrase n'importe quel nombre "
    "<b>entre 0 et 1</b>. On l'utilise surtout sur le <b>dernier</b> neurone, "
    "quand on veut une sortie qui ressemble à une probabilité (par exemple "
    "« 87 % de chances que ce soit un chat »). Son défaut : dans les réseaux "
    "profonds, elle a tendance à <b>ralentir l'apprentissage</b>."))

A(Paragraph("La tangente hyperbolique (tanh)", style_h2))
A(bloc_equation("tanh", hauteur_cm=1.15))
A(para(
    "<b>Se prononce</b> « tan-ache », abréviation de « tangente hyperbolique ». "
    "C'est la cousine de la sigmoïde, mais elle écrase le nombre <b>entre "
    "-1 et 1</b> (elle est donc centrée sur 0). Dans les couches cachées, on "
    "la préfère souvent à la sigmoïde car le réseau apprend un peu plus vite."))

A(Paragraph("ReLU (unité linéaire rectifiée)", style_h2))
A(bloc_equation("relu", hauteur_cm=0.7))
A(para(
    "<b>Se prononce</b> « ré-lou ». C'est l'abréviation de l'anglais "
    "<i>Rectified Linear Unit</i> (« unité linéaire rectifiée »). C'est la plus "
    "utilisée aujourd'hui dans les couches cachées, et pourtant la plus "
    "simple : elle <b>garde le nombre s'il est positif</b>, et le "
    "<b>remplace par 0 s'il est négatif</b>. Rapide à calculer et sans le "
    "défaut de lenteur de la sigmoïde, c'est le <b>choix par défaut</b> des "
    "réseaux modernes."))

A(image_fig("activations.png", largeur_cm=15.5,
            legende="Figure — Les trois fonctions côte à côte. Notez la forme : "
                    "la sigmoïde et tanh « écrasent », ReLU coupe simplement le "
                    "négatif à zéro."))

A(encadre("Comment choisir ? (le réflexe à retenir)",
          "&#8226; <b>Couches cachées</b> : ReLU par défaut (simple et rapide). "
          "<br/>&#8226; <b>Dernier neurone</b>, si on veut une probabilité : "
          "sigmoïde.<br/>&#8226; <b>tanh</b> : une bonne alternative à la "
          "sigmoïde dans les couches cachées.<br/><br/>"
          "Dans notre exemple (section 10), on utilise la <b>sigmoïde partout</b>, "
          "uniquement pour n'avoir qu'une seule formule à expliquer. Le principe "
          "reste identique quelle que soit la fonction choisie.",
          fond=BLEU_CLAIR, bordure=BLEU, titre_couleur=BLEU))


# ---- 5. Forward -----------------------------------------------------------
A(PageBreak())
A(Paragraph("5. La forward propagation (propagation avant)", style_h1))
A(para(
    "On travaille rarement neurone par neurone : on regroupe les calculs de "
    "toute une couche dans une seule <b>multiplication de matrices</b>, ce qui "
    "est beaucoup plus rapide. Voici les quatre étapes pour notre réseau à une "
    "couche cachée (<i>X</i> contient les entrées, empilées en lignes)."))

A(Paragraph("Étape 1 — combinaison linéaire de la couche cachée", style_h2))
A(bloc_equation("fwd1", hauteur_cm=0.7))
A(Paragraph("Étape 2 — activation de la couche cachée", style_h2))
A(bloc_equation("fwd2", hauteur_cm=0.8))
A(Paragraph("Étape 3 — combinaison linéaire de la couche de sortie", style_h2))
A(bloc_equation("fwd3", hauteur_cm=0.7))
A(Paragraph("Étape 4 — activation finale : la prédiction", style_h2))
A(bloc_equation("fwd4", hauteur_cm=0.8))
A(para(
    "Le symbole <b>·</b> représente le produit matriciel. Les exposants entre "
    "crochets, comme <i>W</i><super>[1]</super>, indiquent le numéro de la "
    "couche (1 = cachée, 2 = sortie). Le chapeau sur "
    "<i>y&#770;</i> signifie « valeur prédite », par opposition à <i>y</i>, "
    "la vraie réponse."))
A(encadre("À retenir",
          "La forward propagation, c'est simplement enchaîner : "
          "<b>multiplier par les poids &#8594; ajouter le biais &#8594; activer</b>, "
          "couche après couche, jusqu'à obtenir la prédiction.",
          fond=BLEU_CLAIR, bordure=BLEU, titre_couleur=BLEU))


# ---- 5. Coût --------------------------------------------------------------
A(Paragraph("6. Mesurer l'erreur : la fonction de coût", style_h1))
A(para(
    "Une fois la prédiction obtenue, il faut chiffrer « à quel point on s'est "
    "trompé ». C'est le rôle de la <b>fonction de coût</b> (ou fonction de "
    "perte). Pour une sortie qui vaut 0 ou 1, on utilise "
    "l'<b>entropie croisée binaire</b> :"))
A(bloc_equation("cout", hauteur_cm=1.15))
A(para(
    "Ici <i>m</i> est le nombre d'exemples. La formule punit fortement les "
    "prédictions confiantes mais fausses (par exemple prédire 0,99 alors que la "
    "vérité est 0). Quand la prédiction colle à la vérité, le coût tend vers 0."))
A(encadre("Intuition",
          "Le coût est une <b>note d'erreur</b> : plus il est bas, mieux le "
          "réseau prédit. Tout l'entraînement consiste à faire baisser ce nombre."))


# ---- 6. Descente de gradient ----------------------------------------------
A(Paragraph("7. Apprendre = descendre la pente", style_h1))
A(para(
    "Comment corriger les poids pour faire baisser le coût ? On utilise la "
    "<b>descente de gradient</b>. Imaginez le coût comme une vallée : chaque "
    "poids est une position, et on veut atteindre le fond (l'erreur minimale)."))
A(para(
    "Le <b>gradient</b> d'un poids, noté &#8706;L/&#8706;w, indique la "
    "<b>pente</b> : dans quel sens et de combien le coût augmente si on augmente "
    "ce poids. Pour descendre, il suffit d'aller dans le sens <b>opposé</b> :"))
A(bloc_equation("gradient", hauteur_cm=0.85))
A(para(
    "La lettre grecque &#951; (êta) est le <b>taux d'apprentissage</b> : la "
    "taille du pas. Trop grand, on saute par-dessus le fond de la vallée ; trop "
    "petit, l'apprentissage est très lent. C'est un réglage à doser."))
A(encadre("Image mentale",
          "Vous êtes dans le brouillard sur une colline et voulez rejoindre le "
          "fond. Vous tâtez le sol pour sentir la pente (le gradient) et faites "
          "un pas vers le bas. Répété, ce geste vous mène au point le plus bas.",
          fond=VERT_CLAIR, bordure=VERT_FONCE))


# ---- 7. Backward ----------------------------------------------------------
A(PageBreak())
A(Paragraph("8. La backward propagation (rétropropagation)", style_h1))
A(para(
    "Le problème : le coût dépend des poids de la sortie, qui dépendent des "
    "poids de la couche cachée, qui dépendent des entrées… Pour trouver "
    "l'influence de <i>chaque</i> poids sur le coût, on utilise la "
    "<b>règle de la chaîne</b> (dérivation en cascade) :"))
A(bloc_equation("chaine", hauteur_cm=1.0))
A(para(
    "Autrement dit, on multiplie les petites influences le long du chemin. On "
    "part de l'erreur à la sortie et on la <b>propage vers l'arrière</b>, couche "
    "par couche — d'où le nom. On aura besoin de la dérivée de la sigmoïde, qui "
    "s'exprime très simplement à partir de sa sortie :"))
A(bloc_equation("sig_deriv", hauteur_cm=0.75))

A(Paragraph("Les formules, de la sortie vers l'entrée", style_h2))
A(para("<b>Couche de sortie.</b> Avec une sortie sigmoïde et l'entropie "
       "croisée, la première dérivée se simplifie de façon spectaculaire :"))
A(bloc_equation("bwd1", hauteur_cm=0.62))
A(bloc_equation("bwd2", hauteur_cm=0.75))
A(bloc_equation("bwd3", hauteur_cm=0.75))
A(para("<b>Couche cachée.</b> On renvoie l'erreur vers l'arrière en la "
       "multipliant par les poids de sortie, puis on traverse l'activation. "
       "Le petit rond barré dans la formule ci-dessous désigne une "
       "multiplication <b>terme à terme</b> (chaque case avec la case "
       "correspondante), à ne pas confondre avec le produit matriciel :"))
A(bloc_equation("bwd4", hauteur_cm=0.75))
A(bloc_equation("bwd5", hauteur_cm=0.75))
A(bloc_equation("bwd6", hauteur_cm=0.75))
A(para(
    "Les quantités notées <i>dW</i> et <i>db</i> sont exactement les "
    "<b>gradients</b> attendus par la descente de gradient de la section 7. "
    "Une fois calculés, on met à jour tous les poids et biais, et on recommence."))
A(encadre("À retenir",
          "Backward = <b>remonter l'erreur à l'envers</b> pour attribuer à "
          "chaque poids sa part de responsabilité, grâce à la règle de la chaîne. "
          "Le résultat, ce sont les gradients qui disent comment corriger.",
          fond=BLEU_CLAIR, bordure=BLEU, titre_couleur=BLEU))


# ---- 8. Algorithme complet ------------------------------------------------
A(PageBreak())
A(Paragraph("9. L'algorithme complet, en une page", style_h1))
A(para("Entraîner le réseau, c'est répéter la boucle suivante des milliers de "
       "fois. Une répétition complète s'appelle une <b>époque</b> :"))

etapes = [
    ("1", "Forward", "Calculer la prédiction à partir des entrées "
     "(section 5)."),
    ("2", "Coût", "Mesurer l'erreur entre prédiction et vérité "
     "(section 6)."),
    ("3", "Backward", "Calculer les gradients de chaque poids par "
     "rétropropagation (section 8)."),
    ("4", "Mise à jour", "Corriger chaque poids d'un petit pas dans le sens "
     "opposé à son gradient (section 7)."),
    ("5", "Répéter", "Recommencer à l'étape 1 jusqu'à ce que le coût soit "
     "suffisamment bas."),
]
lignes = []
for num, titre, desc in etapes:
    lignes.append([
        Paragraph(f'<b>{num}</b>', ParagraphStyle(
            "num", parent=style_corps, textColor=HexColor("#ffffff"),
            alignment=TA_CENTER, fontSize=14, fontName="Helvetica-Bold")),
        Paragraph(f"<b>{titre}</b> — {desc}", style_corps),
    ])
tbl = Table(lignes, colWidths=[1.2 * cm, 14.4 * cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, -1), BLEU),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("LEFTPADDING", (1, 0), (1, -1), 12),
    ("LINEBELOW", (1, 0), (1, -2), 0.5, GRIS_CLAIR),
    ("ROWBACKGROUNDS", (1, 0), (1, -1), [HexColor("#ffffff"), GRIS_CLAIR]),
]))
A(tbl)
A(Spacer(1, 0.4 * cm))
A(encadre("Le cœur de tout l'apprentissage profond",
          "Cette boucle « deviner &#8594; mesurer l'erreur &#8594; corriger » "
          "est <b>exactement</b> ce qui fait fonctionner les plus grands modèles "
          "d'intelligence artificielle. Ils ont plus de couches et de neurones, "
          "mais le principe est identique.", fond=VERT_CLAIR, bordure=VERT_FONCE))


# ---- 9. Exemple XOR -------------------------------------------------------
A(PageBreak())
A(Paragraph("10. Exemple concret : le problème du XOR", style_h1))
A(para(
    "Le <b>XOR</b> (« OU exclusif ») renvoie 1 quand les deux entrées sont "
    "<b>différentes</b>, et 0 quand elles sont identiques. C'est l'exemple "
    "historique : un neurone seul <b>ne peut pas</b> le résoudre, mais un réseau "
    "avec une couche cachée le résout parfaitement."))

table_xor = Table(
    [["Entrée 1", "Entrée 2", "Sortie attendue"],
     ["0", "0", "0"],
     ["0", "1", "1"],
     ["1", "0", "1"],
     ["1", "1", "0"]],
    colWidths=[3.2 * cm, 3.2 * cm, 4.2 * cm])
table_xor.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLEU),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 11),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), GRIS_CLAIR]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
]))
t_wrap = Table([[table_xor]], colWidths=[15.6 * cm])
t_wrap.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
A(t_wrap)
A(Spacer(1, 0.5 * cm))

A(para(
    "Après <b>10 000 époques</b> d'entraînement, le coût s'effondre : le réseau "
    "apprend. La courbe ci-dessous montre l'erreur qui diminue au fil du temps."))
A(image_fig("courbe_cout.png", largeur_cm=13,
            legende="Figure — Le coût (l'erreur) chute rapidement puis se "
                    "stabilise près de zéro."))

A(para(
    "On peut aussi visualiser ce que le réseau a « compris » en coloriant le "
    "plan selon sa prédiction. Les zones rouges correspondent à une sortie "
    "proche de 1, les zones bleues à une sortie proche de 0. On retrouve bien le "
    "motif en damier caractéristique du XOR."))
A(image_fig("frontiere_decision.png", largeur_cm=11,
            legende="Figure — La frontière de décision. Les triangles oranges "
                    "(sortie 1) et les cercles bleus (sortie 0) sont "
                    "correctement séparés."))

A(para("Résultat final obtenu par le programme :"))
resultats = Table(
    [["Entrée", "Probabilité prédite", "Prédiction", "Attendu"],
     ["(0, 0)", "0,001", "0", "0"],
     ["(0, 1)", "0,999", "1", "1"],
     ["(1, 0)", "0,999", "1", "1"],
     ["(1, 1)", "0,002", "0", "0"]],
    colWidths=[3.4 * cm, 4.6 * cm, 3.4 * cm, 3.0 * cm])
resultats.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), GRIS),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10.5),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), VERT_CLAIR]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
]))
A(resultats)
A(Spacer(1, 0.3 * cm))
A(encadre("Exactitude : 100 %",
          "Le réseau classe correctement les quatre cas. Parti de poids "
          "aléatoires, il a « appris » le XOR uniquement en répétant "
          "forward + backward.", fond=VERT_CLAIR, bordure=VERT_FONCE))


# ---- 10. Code -------------------------------------------------------------
A(PageBreak())
A(Paragraph("11. Le code Python commenté", style_h1))
A(para(
    "Voici le cœur de l'implémentation. Le fichier complet, abondamment "
    "commenté, accompagne ce document (<font name='Courier'>reseau_de_neurones.py</font>). "
    "Remarquez comme le code suit exactement les formules des sections "
    "précédentes."))

extrait_forward = [
    "def forward(self, X):",
    "    # Couche cachee : combinaison lineaire puis activation",
    "    self.Z1 = X @ self.W1 + self.b1",
    "    self.A1 = sigmoid(self.Z1)",
    "    # Couche de sortie : idem, la sortie est la prediction",
    "    self.Z2 = self.A1 @ self.W2 + self.b2",
    "    self.A2 = sigmoid(self.Z2)",
    "    return self.A2",
]
extrait_backward = [
    "def backward(self, Y):",
    "    m = Y.shape[0]",
    "    # --- Couche de sortie ---",
    "    dZ2 = self.A2 - Y",
    "    dW2 = (self.A1.T @ dZ2) / m",
    "    db2 = np.sum(dZ2, axis=0, keepdims=True) / m",
    "    # --- Couche cachee (regle de la chaine) ---",
    "    dA1 = dZ2 @ self.W2.T",
    "    dZ1 = dA1 * sigmoid_derivee(self.A1)",
    "    dW1 = (self.X.T @ dZ1) / m",
    "    db1 = np.sum(dZ1, axis=0, keepdims=True) / m",
    "    # --- Mise a jour des poids (descente de gradient) ---",
    "    self.W2 -= self.taux_apprentissage * dW2",
    "    self.b2 -= self.taux_apprentissage * db2",
    "    self.W1 -= self.taux_apprentissage * dW1",
    "    self.b1 -= self.taux_apprentissage * db1",
]


def bloc_code(titre, lignes):
    para_titre = Paragraph(titre, style_h2)
    corps = "<br/>".join(
        l.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
        for l in lignes)
    p = Paragraph(corps, style_code)
    t = Table([[p]], colWidths=[15.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    # Re-teindre le texte en clair
    p2 = Paragraph(corps, ParagraphStyle("codeclair", parent=style_code,
                                         textColor=HexColor("#e2e8f0")))
    t = Table([[p2]], colWidths=[15.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#0f172a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return KeepTogether([para_titre, t, Spacer(1, 0.3 * cm)])


A(bloc_code("La forward propagation en Python", extrait_forward))
A(bloc_code("La backward propagation en Python", extrait_backward))
A(para(
    "Le symbole <font name='Courier'>@</font> est le produit matriciel de NumPy, "
    "et <font name='Courier'>.T</font> transpose une matrice (échange lignes et "
    "colonnes). Ces deux fonctions font tout le travail lourd à notre place."))


# ---- 11. Glossaire --------------------------------------------------------
A(PageBreak())
A(Paragraph("12. Glossaire des termes importants", style_h1))
glossaire = [
    ("Poids (weight)", "Nombre qui règle l'importance d'une entrée. C'est ce "
     "que le réseau apprend."),
    ("Biais (bias)", "Décalage ajouté à la somme pondérée, qui permet de "
     "déplacer la fonction."),
    ("Fonction d'activation", "Fonction non-linéaire qui donne au réseau sa "
     "souplesse. Les trois classiques : sigmoïde, tanh, ReLU."),
    ("Sigmoïde", "Fonction d'activation qui écrase un nombre entre 0 et 1. "
     "Utile pour produire une probabilité en sortie."),
    ("tanh", "« Tangente hyperbolique ». Comme la sigmoïde mais entre -1 et 1 ; "
     "souvent préférée dans les couches cachées."),
    ("ReLU", "« Rectified Linear Unit ». Garde le nombre s'il est positif, "
     "sinon renvoie 0. Le choix par défaut des réseaux modernes."),
    ("Forward propagation", "Passage des entrées vers la sortie pour produire "
     "une prédiction."),
    ("Fonction de coût", "Mesure chiffrée de l'erreur entre prédiction et "
     "vérité."),
    ("Gradient", "Pente du coût par rapport à un poids : dans quel sens et de "
     "combien le corriger."),
    ("Descente de gradient", "Méthode qui corrige les poids en suivant la pente "
     "vers le bas."),
    ("Backward propagation", "Calcul des gradients en remontant le réseau à "
     "l'envers (règle de la chaîne)."),
    ("Taux d'apprentissage", "Taille du pas de correction. À régler avec soin."),
    ("Époque", "Une itération complète : forward + coût + backward + mise à "
     "jour sur toutes les données."),
]
lignes_glo = [[Paragraph(f"<b>{terme}</b>", style_corps),
               Paragraph(defn, style_corps)] for terme, defn in glossaire]
tglo = Table(lignes_glo, colWidths=[4.6 * cm, 11.0 * cm])
tglo.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [HexColor("#ffffff"), GRIS_CLAIR]),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor("#e2e8f0")),
]))
A(tglo)
A(Spacer(1, 0.6 * cm))
A(KeepTogether(encadre("Félicitations",
          "Vous connaissez maintenant les deux mécanismes qui font battre le "
          "cœur de tout réseau de neurones. Le reste de l'apprentissage profond "
          "n'est qu'une extension de ces mêmes idées.",
          fond=BLEU_CLAIR, bordure=BLEU, titre_couleur=BLEU)))


# ---------------------------------------------------------------------------
# Assemblage du document avec numéros de page
# ---------------------------------------------------------------------------
def pied_de_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(GRIS)
    numero = canvas.getPageNumber()
    if numero > 1:
        canvas.drawRightString(
            A4[0] - 2 * cm, 1.1 * cm, f"{numero}")
        canvas.drawString(
            2 * cm, 1.1 * cm, "Comprendre la propagation dans un réseau de neurones")
        canvas.setStrokeColor(HexColor("#e2e8f0"))
        canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


chemin_pdf = os.path.join(DOSSIER, "comprendre_la_propagation.pdf")
doc = BaseDocTemplate(
    chemin_pdf, pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2 * cm, bottomMargin=2 * cm,
    title="Comprendre la propagation dans un reseau de neurones",
    author="")
frame = Frame(doc.leftMargin, doc.bottomMargin,
              doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="tpl", frames=[frame],
                                   onPage=pied_de_page)])
doc.build(histoire)
print(f"PDF genere : {chemin_pdf}")
