# 🎮 Projet Morpion ML - Examen M1 S1,  MACHINE LEARNING

[Site Web ISPM](https://ispm-edu.com) - https://ispm-edu.com

## 👥 Informations Groupe
**NOM DU GROUPE :** `404 NOT FOUND`  
**PROMOTION :** `IMTICIA 4`

| N° | Membres du Groupe |
|:---|:---|
| 07 | **RAKOMAMPIONONA** Fitahiana Herizo | |
| 10 | **RAKOTOBE** Lori Emmanuela | |
| 12 | **RAKOTONINDRINA** Andry Anicet | |
| 15 | **RAZAFIMAHANDRY** Herintsoa Fitahiana | |
| 19 | **HARINIRIANA** Nomena Niaina Kévin | |
| 20 | **RANDRIANARISOA** Notahiniela Olly Desto | |
| 24 | **RAKOTONANDRASANA** Rova Fanantenana | |

---

## 📝 Description du Projet
Ce projet consiste en la création d'un **jeu de Morpion (Tic-Tac-Toe)** complet intégrant une **Intelligence Artificielle** comme adversaire. 

Le pipeline inclut :
*   La génération automatique de données.
*   L'entraînement de modèles de Machine Learning (Régression Logistique, Random Forest, etc.).
*   Une interface graphique interactive permettant de défier l'IA en temps réel.







Ce projet présente un pipeline complet de Machine Learning pour le jeu du Morpion (Tic-Tac-Toe), allant de la génération automatisée de données par Minimax jusqu'à une interface de jeu réactive intégrant des modèles d'IA.


[Lien Video](https://drive.google.com/drive/folders/10IBLziiX-BiaDlof8i4wAqiBojoWreUX?usp=sharing)
## 📋 Table des Matières
1. [Génération du Dataset](#1-génération-du-dataset)
2. [Analyse Exploratoire (EDA)](#2-analyse-exploratoire)
3. [Modélisation & Performance](#3-modélisation--performance)
4. [Interface Jouable](#4-interface-jouable)
5. [Réponses aux Questions (Q1-Q4)](#5-réponses-aux-questions)

---

# Instruction et instruction :
```
    make  #compiler le projet
```
```
    make run #lancer le projet
```
---


## 1. Génération du Dataset
Le dataset a été généré via un script Python utilisant l'algorithme **Minimax avec élagage Alpha-Bêta**.
- **Périmètre** : Tous les états valides où c'est au tour de **X** de jouer.
- **Volume** : 2423 états uniques.
- **Features** : 18 colonnes binaires (occupation par X et O pour chaque case).
- **Labels** : `x_wins` et `is_draw` basés sur un jeu parfait.

## 2. Analyse Exploratoire (EDA)
L'EDA a révélé un fort déséquilibre de classes :
- **X gagne** : 75.5 % des cas.
- **Match Nul** : 18.2 % des cas.
- **O gagne** : 6.3 % des cas.

Les matrices d'occupation montrent que le centre (Case 4) et les coins sont les zones les plus corrélées à la victoire de X.

## 3. Modélisation & Performance

### Baseline (Régression Logistique)
- **x_wins** : F1-Score ~0.87 | Accuracy ~77%
- **is_draw** : F1-Score 0.00 (Le modèle prédit systématiquement "Pas Nul" à cause du déséquilibre).

### Modèles Avancés
L'exploration de modèles plus puissants a donné les résultats suivants pour `x_wins` :
| Modèle | Accuracy | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- |
| **XGBoost (Meilleur)** | **88.2 %** | **0.925** | **0.922** |
| Random Forest | 87.4 % | 0.923 | 0.891 |
| MLP Neural Net | 83.5 % | 0.897 | 0.814 |
| Decision Tree | 78.8 % | 0.860 | 0.766 |

## 4. Interface Jouable
Située dans `interfaces/`, l'interface React propose :
- **vs Humain** : Jeu à deux en local.
- **vs IA (ML)** : Utilise les poids de la Régression Logistique.
- **vs IA (Hybride)** : Minimax depth 3 + évaluation ML des feuilles.
On a utilisé customTkinter.


## 5. Réponses aux Questions

### Q1 : Influence des cases et stratégie humaine
L'analyse des coefficients de la Régression Logistique montre que les cases les plus influentes pour la victoire de X sont les **coins** (0, 2, 6, 8) et le **centre** (4). Ces cases ont des coefficients positifs élevés, ce qui signifie que leur occupation par X augmente fortement la probabilité de victoire prédite. C'est parfaitement cohérent avec la stratégie humaine qui consiste à occuper le centre pour contrôler les diagonales et les coins pour créer des doubles menaces.

### Q2 : Déséquilibre des classes et métriques
Le dataset est fortement déséquilibré (75% X gagne). Utiliser l'**Accuracy** seule est trompeur : un modèle qui prédit toujours "X gagne" aurait 75% d'accuracy mais ne reconnaîtrait jamais un nul. Nous avons donc privilégié le **F1-Score** et l'**AUC-ROC**, qui pénalisent les faux positifs et les faux négatifs de manière équilibrée, particulièrement pour la classe minoritaire `is_draw` où la Régression Logistique a totalement échoué (F1 = 0).

### Q3 : Comparaison des classificateurs et erreurs types
Le classificateur `x_wins` est beaucoup plus performant (F1 ~0.92) que `is_draw` (F1 ~0.15). Cela s'explique par le fait que "gagner" au Morpion suit des motifs géométriques simples (lignes/colonnes/diagonales) faciles à apprendre pour un modèle. En revanche, un "match nul" est souvent la conséquence d'une suite complexe de blocages subtils, ce qui rend cette classe beaucoup plus difficile à modéliser avec des features simples et peu d'exemples.

### Q4 : Comportement IA-ML vs Hybride
L'IA purement ML a tendance à faire des coups "statistiquement bons" mais peut tomber dans des pièges bêtes car elle n'a aucune vision à long terme. Le mode **Hybride** est nettement plus solide car le Minimax lui permet d'anticiper les menaces immédiates sur 3 coups, tout en utilisant le modèle ML pour juger de la "qualité" de la position finale atteinte. L'Hybride est moins prévisible et beaucoup plus difficile à battre qu'une IA qui ne ferait que suivre une probabilité figée.

---
**Équipe R&D EdTech Madagascar**
*Hackathon Mars 2026*
