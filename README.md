<div align="center">

# 🚗 Détection de somnolence du conducteur

**Prototype académique de détection de somnolence au volant par vision par ordinateur**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Statut](https://img.shields.io/badge/Statut-Prototype-orange?style=for-the-badge)]()
[![Validation](https://img.shields.io/badge/Validation-Non%20certifi%C3%A9-red?style=for-the-badge)]()

</div>

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Captures d'écran](#-captures-décran)
- [Architecture](#-architecture)
- [Utilisation](#-utilisation)
- [Modèle](#-modèle)
- [Métriques](#-métriques)
- [Limites](#-limites)
- [Avertissement](#-avertissement)
- [Auteur](#-auteur)

---

## 🎯 À propos

Ce projet est un **prototype académique** de détection de somnolence au volant, basé sur l'analyse des points faciaux via **MediaPipe Face Landmarker** et sur le calcul de deux métriques reconnues dans la littérature :

- **EAR** (*Eye Aspect Ratio*) : rapport d'ouverture des yeux, calculé à partir de 6 points par œil.
- **PERCLOS** (*Percentage of Eye Closure*) : pourcentage de temps où les yeux sont fermés sur une fenêtre glissante.

L'application propose deux modes d'analyse : **vidéo enregistrée** et **webcam en direct**, avec une interface interactive **Streamlit** affichant KPI, graphiques, vidéo annotée et journal des événements.

> ⚠️ **Ce système n'est pas certifié pour la conduite réelle.** Il est destiné à la recherche, à l'enseignement et à la démonstration technique.

---

## ✨ Fonctionnalités

### 🎥 Analyse vidéo
- Import de vidéos aux formats **MP4, AVI, MOV, MKV**.
- Traitement image par image avec **OpenCV** + **MediaPipe**.
- Génération d'une **vidéo annotée** (cadre coloré, EAR, seuil, état, PERCLOS).
- Conversion en **H.264** pour compatibilité navigateur.
- Aperçu en direct pendant l'analyse.

### 📹 Webcam en direct
- Flux vidéo en temps réel via **streamlit-webrtc**.
- Affichage des métriques en continu (EAR, seuil, PERCLOS, état).
- Courbe EAR en temps réel.
- Analyse post-session avec export des mesures.

### 📊 Métriques et analytics
- **EAR personnalisé** : calibration baseline sur yeux ouverts, seuil adaptatif.
- **PERCLOS glissant** sur fenêtre de 60 s, avec filtrage des clignotements.
- **Détection de fermetures prolongées** (≥ 0,5 s) avec confirmation temporelle.
- **Couverture PERCLOS** affichée pour indiquer la fiabilité de la mesure.
- **Diagnostic de calibration** : signalement si la baseline est incertaine.

### 📈 Interface
- KPI synthétiques (EAR, seuil, PERCLOS, fermetures, état).
- Graphique d'évolution de l'EAR et du seuil.
- Journal des fermetures prolongées (début, fin, durée, type).
- Tableau image par image.
- Export **CSV** des mesures et des événements.

---

## 📸 Captures d'écran

> *À compléter avec vos propres captures d'écran.*

| Mode vidéo | Mode webcam |
|---|---|
| ![Mode vidéo](docs/screenshot-video.png) | ![Mode webcam](docs/screenshot-webcam.png) |

| Graphique EAR | Journal des événements |
|---|---|
| ![Graphique EAR](docs/screenshot-chart.png) | ![Journal](docs/screenshot-events.png) |

---

## 💻 Utilisation

*Mode vidéo enregistrée
Sélectionner « Vidéo enregistrée ».

Importer une vidéo (MP4, AVI, MOV, MKV).

Cliquer sur « Analyser la vidéo ».

Consulter l'aperçu en direct, puis les résultats :

Vidéo annotée

KPI (EAR, seuil, PERCLOS, fermetures, état)

Graphique d'évolution

Journal des fermetures prolongées

Tableau image par image

Boutons d'export CSV

*Mode webcam en direct
Sélectionner « Webcam en direct ».

Autoriser l'accès à la caméra dans le navigateur.

Garder le visage face à la caméra pendant la calibration (2 s).

Consulter les métriques en temps réel.

Cliquer sur « Analyser la session webcam » pour récupérer les mesures et événements.

---

## 🧠 Modèle
Le fichier face_landmarker.task est un modèle binaire MediaPipe (quelques Mo). Il n'est pas versionné dans ce dépôt (voir .gitignore) pour éviter de gonfler l'historique Git.
Téléchargez le modèle depuis le dépôt officiel MediaPipe.
Placez ensuite le fichier face_landmarker.task à la racine du projet, au même niveau que app.py.

---

## 📐 Métriques
EAR — Eye Aspect Ratio
L'EAR mesure l'ouverture des yeux à partir de 6 points par œil :


EAR = (|p2 - p6| + |p3 - p5|) / (2 × |p1 - p4|)

EAR élevé → yeux ouverts
EAR bas → yeux fermés ou mi-clos

PERCLOS — Percentage of Eye Closure
Le PERCLOS mesure le pourcentage de temps où les yeux sont fermés sur une fenêtre glissante (60 s par défaut) :

PERCLOS = (durée yeux fermés ≥ 0,5 s) / (durée totale valide) × 100
Les clignotements (< 0,5 s) sont exclus du calcul.

Le PERCLOS n'est publié qu'après une fenêtre complète de 60 s, avec au moins 75 % de couverture valide.

Calibration baseline
Pendant les 2 premières secondes, l'application accumule des mesures d'yeux ouverts.

Le seuil est ensuite calculé comme : seuil = médiane(baseline) × 0,75.

Si la baseline est jugée incertaine (médiane < 0,15), le seuil de secours 0,15 est utilisé.

---

## 🛠️ Stack technique

| Composant | Rôle |
|---|---|
| **Python 3.10+** | Langage principal |
| **MediaPipe Tasks** | Détection des points faciaux (Face Landmarker) |
| **OpenCV** | Lecture, écriture et annotation des vidéos |
| **Streamlit** | Interface web interactive |
| **streamlit-webrtc** | Flux webcam en temps réel |
| **streamlit-autorefresh** | Rafraîchissement automatique des métriques |
| **NumPy** | Calculs numériques (EAR, médianes) |
| **pandas** | Manipulation des données et export CSV |
| **imageio-ffmpeg** | Conversion H.264 des vidéos annotées |

---
## 🏗️ Architecture

detection-somnolence-conducteur/
│
├── app.py # Application principale
├── requirements.txt # Dépendances Python
├── .gitignore # Fichiers ignorés par Git
├── README.md # Ce fichier
│
├── face_landmarker.task # Modèle MediaPipe (non versionné)
│
└── docs/ # Captures d'écran et documentation

---
##⚠️ Limites
Limite	Détail
Prototype académique	: Aucune validation clinique ou réglementaire.
Seuil PERCLOS expérimental	: Le seuil de 15 % est indicatif et doit être validé sur des données annotées.
Sensibilité aux conditions	: Éclairage, lunettes, angle du visage, résolution.
Pas de validation quantitative	: Aucun dataset annoté n'a été utilisé pour mesurer précision/rappel.
Head pose non géré	: Un regard détourné peut fausser l'EAR.
Pas de détection multi-visages	: Le système ne traite qu'un visage à la fois.

---
##🚨 Avertissement
Ce système n'est pas un dispositif de sécurité.

Il ne remplace en aucun cas les équipements de sécurité automobile, ni le jugement du conducteur.

Il est destiné exclusivement à :

la recherche académique,

l'enseignement,

la démonstration technique.

Ne l'utilisez pas pour surveiller un conducteur réel sans son consentement éclairé.

---
##👤 Auteur
Nakbi Meriem
GitHub :UserMariam50


##🙏 Remerciements
MediaPipe — Google

OpenCV

Streamlit

streamlit-webrtc


<div align="center">
⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐

</div> ```
