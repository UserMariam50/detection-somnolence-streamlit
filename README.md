 <div align="center">

# 🚗 Détection de somnolence du conducteur

**Prototype académique de détection de somnolence au volant par vision par ordinateur**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![Statut](https://img.shields.io/badge/Statut-Prototype-orange?style=for-the-badge)
![Validation](https://img.shields.io/badge/Validation-Non%20certifi%C3%A9-red?style=for-the-badge)

</div>

---

## 📋 Table des matières

- [🎯 À propos](#a-propos)
- [✨ Fonctionnalités](#fonctionnalites)
- [📸 Captures d'écran](#captures-decran)
- [🏗️ Architecture](#architecture)
- [💻 Utilisation](#utilisation)
- [🧠 Modèle](#modele)
- [📐 Métriques](#metriques)
- [🛠️ Stack technique](#stack-technique)
- [⚠️ Limites](#limites)
- [🚨 Avertissement](#avertissement)
- [👤 Auteur](#auteur)
- [🙏 Remerciements](#remerciements)

---

<a id="a-propos"></a>
## 🎯 À propos

Ce projet est un **prototype académique de détection de somnolence au volant**, basé sur l'analyse des points faciaux avec **MediaPipe Face Landmarker** et sur le calcul de métriques utilisées dans l'étude de la fermeture des yeux.

Les principales métriques sont :

- **EAR (Eye Aspect Ratio)** : mesure géométrique de l'ouverture des yeux, calculée à partir de six points faciaux par œil.
- **PERCLOS (Percentage of Eye Closure)** : estimation du pourcentage de temps pendant lequel les yeux sont fermés sur une fenêtre temporelle donnée.

L'application propose deux modes d'analyse :

- **Vidéo enregistrée** : analyse d'une vidéo importée et génération de résultats annotés.
- **Webcam en direct** : suivi des métriques pendant une session webcam.

L'interface développée avec **Streamlit** présente des indicateurs synthétiques, des graphiques, une vidéo annotée, un journal des événements et des possibilités d'exportation des résultats.

> ⚠️ **Important :** ce système est un prototype académique non certifié. Il est destiné à l'apprentissage, à la recherche et à la démonstration technique. Il n'est pas validé pour une utilisation en conduite réelle.

---

<a id="fonctionnalites"></a>
## ✨ Fonctionnalités

### 🎥 Analyse vidéo

- Importation de vidéos aux formats MP4, AVI, MOV et MKV.
- Traitement image par image avec OpenCV et MediaPipe.
- Génération d'une vidéo annotée affichant les informations de détection.
- Affichage de l'EAR, du seuil, du PERCLOS et de l'état estimé.
- Conversion vidéo en H.264 pour faciliter la lecture dans le navigateur, selon la configuration disponible.
- Présentation des résultats après l'analyse.

### 📹 Webcam en direct

- Capture vidéo en temps réel avec `streamlit-webrtc`.
- Affichage des métriques de détection.
- Visualisation de l'évolution de l'EAR.
- Calibration initiale du seuil.
- Analyse des données enregistrées pendant la session.
- Exportation des mesures et des événements, selon les fonctionnalités activées dans l'application.

### 📊 Métriques et analyse

- Calcul de l'EAR à partir des points faciaux.
- Calibration initiale d'une référence d'ouverture des yeux.
- Utilisation d'un seuil adaptatif basé sur la calibration.
- Estimation du PERCLOS sur une fenêtre glissante de 60 secondes, selon l'implémentation.
- Détection de fermetures prolongées des yeux.
- Filtrage des clignotements courts dans le calcul du PERCLOS, selon les règles définies.
- Affichage de la couverture des données valides.
- Diagnostic de calibration lorsque la référence estimée est incertaine.

### 📈 Interface et visualisation

- Indicateurs synthétiques : EAR, seuil, PERCLOS, nombre de fermetures et état estimé.
- Graphique d'évolution de l'EAR et du seuil.
- Journal des fermetures prolongées.
- Tableau des mesures image par image.
- Exportation des mesures et des événements au format CSV.

---

<a id="captures-decran"></a>
## 📸 Captures d'écran

Les captures d'écran ci-dessous sont à ajouter après avoir exécuté l'application.

| Analyse vidéo | Webcam en direct |
|---|---|
| ![Analyse vidéo](docs/screenshot-video.png) | ![Webcam en direct](docs/screenshot-webcam.png) |

| Évolution de l'EAR | Journal des événements |
|---|---|
| ![Graphique EAR](docs/screenshot-chart.png) | ![Journal des événements](docs/screenshot-events.png) |


---

<a id="architecture"></a>
## 🏗️ Architecture

Voici la structure recommandée du dépôt :

```text
detection-somnolence-conducteur/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
│
├── face_landmarker.task
│
└── docs/
  
```

### Description des fichiers

| Fichier ou dossier | Rôle |
|---|---|
| `app.py` | Application principale et logique de détection |
| `requirements.txt` | Dépendances Python nécessaires |
| `.gitignore` | Fichiers et dossiers exclus du suivi Git |
| `README.md` | Documentation du projet |
| `LICENSE` | Licence du projet, si elle est incluse |
| `face_landmarker.task` | Modèle MediaPipe requis pour la détection faciale |
| `docs/` | Captures d'écran et documentation visuelle |

Le fichier `face_landmarker.task` peut être exclu du dépôt afin d'éviter d'y ajouter un fichier binaire volumineux. Dans ce cas, il faut documenter clairement son téléchargement et son emplacement.

---

<a id="utilisation"></a>
## 💻 Utilisation

### 1. Prérequis

- Python 3.10 ou une version compatible avec les dépendances installées.
- Git.
- Une webcam pour le mode temps réel.
- Un navigateur compatible avec les fonctionnalités utilisées.
- Le fichier de modèle MediaPipe requis.

### 2. Cloner le dépôt

### 3. Créer un environnement virtuel

### 4. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Télécharger le modèle MediaPipe

Le fichier `face_landmarker.task` doit être placé à la racine du projet, au même niveau que `app.py`.

### 6. Lancer l'application

```bash
streamlit run app.py
```

Streamlit affiche généralement une adresse locale dans le terminal. Ouvre cette adresse dans ton navigateur.

### 7. Mode vidéo enregistrée

1. Sélectionner le mode **Vidéo enregistrée**.
2. Importer une vidéo au format pris en charge.
3. Cliquer sur le bouton d'analyse.
4. Attendre la fin du traitement.
5. Consulter la vidéo annotée, les indicateurs, les graphiques et le journal des événements.
6. Exporter les résultats disponibles au format CSV.

### 8. Mode webcam en direct

1. Sélectionner le mode **Webcam en direct**.
2. Autoriser l'accès à la caméra.
3. Placer le visage face à la caméra et effectuer la calibration initiale.
4. Observer les métriques pendant la session.
5. Terminer la session selon les commandes disponibles dans l'application.
6. Consulter et exporter les mesures enregistrées.

> Les intitulés exacts des boutons et les étapes de l'interface doivent correspondre à la version effective de `app.py`.

---

<a id="modele"></a>
## 🧠 Modèle

Le projet utilise **MediaPipe Face Landmarker** pour détecter les points de repère du visage nécessaires à l'analyse géométrique des yeux.

Le fichier `face_landmarker.task` contient le modèle utilisé par l'application. Il s'agit d'un fichier binaire qui n'est pas nécessairement versionné dans Git.

### Installation du modèle

1. Télécharger le fichier compatible depuis les ressources officielles de MediaPipe.
2. Placer `face_landmarker.task` à la racine du projet.
3. Vérifier que le chemin utilisé dans le code correspond à son emplacement.
4. Lancer l'application et vérifier que le modèle est correctement chargé.

Documentation officielle : [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker).

### Pourquoi utiliser MediaPipe ?

MediaPipe permet d'obtenir des points de repère faciaux à partir des images. Ces points peuvent ensuite être utilisés pour calculer des indicateurs géométriques tels que l'EAR.

Le système ne repose donc pas uniquement sur la classification directe d'une image : il utilise les points faciaux détectés pour construire des mesures exploitables par la logique de détection.

---

<a id="metriques"></a>
## 📐 Métriques

### 1. EAR — Eye Aspect Ratio

L'EAR est un indicateur géométrique de l'ouverture de l'œil calculé à partir de six points de repère.

Une formule couramment utilisée est :

\[
EAR = \frac{\|p_2-p_6\|+\|p_3-p_5\|}{2\|p_1-p_4\|}
\]

Où :

- \(p_1\) et \(p_4\) représentent les extrémités horizontales de l'œil.
- \(p_2\), \(p_3\), \(p_5\) et \(p_6\) représentent des points situés sur les contours supérieur et inférieur.

Interprétation générale :

- **EAR relativement élevé** : œil généralement ouvert.
- **EAR relativement faible** : œil potentiellement fermé ou partiellement fermé.

L'EAR dépend de la géométrie des points détectés, de la personne et des conditions d'acquisition. Un seuil doit donc être choisi et évalué avec prudence.

### 2. PERCLOS — Percentage of Eye Closure

Le PERCLOS est une mesure temporelle de la fermeture des yeux.

Dans ce projet, son estimation est décrite comme suit :

\[
PERCLOS = \frac{\text{Durée estimée des fermetures}}{\text{Durée valide observée}}\times100
\]

La fenêtre glissante est configurée sur 60 secondes par défaut, selon l'implémentation.

Le calcul peut exclure les clignotements de courte durée, selon les règles définies dans le code.

**Important :** la formule exacte, les conditions de validité, le traitement des clignotements et les critères de publication du PERCLOS doivent être vérifiés dans `app.py`. Les résultats sont des estimations expérimentales et ne constituent pas une mesure médicale ou une certification de somnolence.

### 3. Calibration du seuil

La calibration initiale vise à estimer une référence d'ouverture des yeux à partir de mesures obtenues pendant une période où les yeux sont supposés ouverts.

Si l'implémentation utilise la médiane des valeurs de référence, le seuil peut être calculé sous la forme :

\[
Seuil = \operatorname{médiane}(EAR_{\text{baseline}})\times0{,}75
\]

Un seuil de secours peut être utilisé lorsque la calibration est jugée incertaine, conformément au code.

Ces valeurs sont des paramètres expérimentaux. Elles doivent être évaluées sur plusieurs personnes et dans différentes conditions avant de tirer des conclusions sur les performances.

### 4. Fermetures prolongées

Le système peut détecter des périodes durant lesquelles l'EAR reste inférieur au seuil pendant une durée minimale.

Le projet utilise une durée de référence de 0,5 seconde pour identifier les fermetures prolongées, selon la logique implémentée.

Cette détection doit être distinguée d'une véritable classification de la somnolence : une fermeture des yeux n'indique pas à elle seule l'état physiologique du conducteur.

---

<a id="stack-technique"></a>
## 🛠️ Stack technique

| Technologie | Rôle |
|---|---|
| **Python 3.10+** | Langage principal |
| **MediaPipe Tasks** | Détection des points de repère faciaux |
| **OpenCV** | Lecture, traitement et annotation des vidéos |
| **Streamlit** | Interface web interactive |
| **streamlit-webrtc** | Transmission vidéo depuis la webcam |
| **streamlit-autorefresh** | Actualisation périodique de l'interface, si utilisée |
| **NumPy** | Calculs numériques et opérations sur les mesures |
| **pandas** | Organisation des données et exportation CSV |
| **imageio-ffmpeg** | Conversion vidéo, si utilisée dans le pipeline |

Les versions exactes et les dépendances nécessaires sont définies dans `requirements.txt`.

---

<a id="limites"></a>
## ⚠️ Limites

| Limite | Description |
|---|---|
| **Prototype académique** | Le système n'est pas certifié pour un usage automobile réel. |
| **Absence de validation réglementaire** | Aucune certification de sécurité n'est revendiquée. |
| **Seuils expérimentaux** | Les seuils doivent être évalués sur des données représentatives. |
| **Absence de validation quantitative** | Sans jeu de données annoté et protocole de référence, la précision, le rappel et le F1-score ne peuvent pas être établis de manière fiable. |
| **Conditions d'éclairage** | Les variations de luminosité peuvent affecter la qualité de la détection. |
| **Lunettes et occultations** | Les lunettes, reflets ou autres occultations peuvent perturber les points faciaux. |
| **Orientation du visage** | Une rotation ou une inclinaison importante peut réduire la fiabilité des mesures. |
| **Clignotements naturels** | Les clignotements peuvent être confondus avec des fermetures prolongées selon les paramètres choisis. |
| **Détection du visage** | La perte temporaire des points faciaux peut interrompre ou fausser les mesures. |
| **Nombre de visages** | La gestion de plusieurs visages dépend de la configuration du modèle et de son implémentation dans l'application. |
| **Généralisation** | Les performances peuvent varier selon les personnes, les caméras et les environnements. |

### Améliorations possibles

- Évaluer le système sur un jeu de données annoté.
- Calculer la précision, le rappel, le F1-score et la matrice de confusion.
- Étudier la robustesse aux changements de luminosité et d'orientation.
- Comparer plusieurs seuils et méthodes de calibration.
- Ajouter une analyse de la pose de la tête si nécessaire.
- Mesurer la latence et la fréquence de traitement en temps réel.
- Documenter les erreurs de détection et les cas d'échec.

---

<a id="avertissement"></a>
## 🚨 Avertissement

**Ce projet est un prototype académique et ne constitue pas un dispositif de sécurité automobile.**

Il ne remplace pas les équipements de sécurité, le repos nécessaire à la conduite ou le jugement du conducteur.

Il est destiné exclusivement à :

- La recherche académique.
- L'enseignement.
- L'apprentissage de la vision par ordinateur.
- La démonstration technique.

Ne l'utilisez pas comme unique moyen de prévention des accidents ou de surveillance d'un conducteur réel. Respectez le consentement des personnes filmées ainsi que les règles applicables à la protection des données personnelles.

---

<a id="auteur"></a>
## 👤 Auteur

**Nakbi Meriem**

- GitHub : [@UserMariam50](https://github.com/UserMariam50)

Projet académique réalisé dans le domaine de la vision par ordinateur et de la Data Science.

---

<a id="remerciements"></a>
## 🙏 Remerciements

Remerciements aux équipes et communautés qui développent et maintiennent les outils utilisés dans ce projet :

- [MediaPipe](https://developers.google.com/mediapipe)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)

---

<div align="center">

⭐ **Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !** ⭐

</div>
