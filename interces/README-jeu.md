# 🎮 Guide du Jeu — Morpion ML Adaptatif

Bienvenue dans l'interface de démonstration de notre IA de Morpion. Ce projet illustre l'intégration de modèles de Machine Learning dans une application web interactive.

## 🚀 Comment lancer le jeu

1. Assurez-vous d'avoir [Node.js](https://nodejs.org/) installé.
2. Ouvrez un terminal dans le dossier `interfaces/`.
3. Installez les dépendances :
   ```bash
   npm install
   ```
4. Lancez l'application :
   ```bash
   npm start
   ```
5. Le jeu s'ouvrira automatiquement sur [http://localhost:3000](http://localhost:3000).

## 🕹️ Les Modes de Jeu

### 👤 vs Humain
Le mode classique pour tester l'interface ou jouer avec un ami sur le même écran.

### 🧠 vs IA (ML)
L'IA utilise directement les prédictions d'un modèle de **Régression Logistique** entraîné sur plus de 2000 états de jeu parfaits. Elle choisit le coup qui maximise statistiquement ses chances de victoire ou de nul.
> *Note : Ce mode est purement statistique et peut parfois manquer de vision à long terme.*

### ⚡ vs IA (Hybride)
Le mode ultime. L'IA combine l'algorithme **Minimax** (profondeur 3) avec une fonction d'évaluation basée sur nos modèles ML. Elle "réfléchit" à l'avance et utilise l'IA pour évaluer la qualité des positions futures.

## 📊 Analyse en Temps Réel
Pendant que vous jouez, le panneau de droite affiche une estimation en temps réel de l'avantage de X (basée sur le modèle ML). Plus la barre est remplie, plus X est proche d'une victoire théorique en jeu parfait.

---
*Réalisé dans le cadre du Hackathon EdTech 2026.*
