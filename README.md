# Classification d'images sportives

Ce projet utilise un modèle de Deep Learning pour classer des images de sports de raquette (badminton, tennis, tennis de table). 

## Fonctionnement du modèle

Le modèle est basé sur l'architecture ResNet50 pré-entraînée sur ImageNet, avec des couches supplémentaires pour la classification spécifique aux sports de raquette. Il a été entraîné sur un ensemble de données d'images de ces sports et optimisé pour obtenir une précision maximale.

## Utilisation de l'application Streamlit

1. Téléversez une image de sport de raquette.
2. Cliquez sur le bouton "Prédire".
3. L'application affichera la classe prédite (badminton, tennis, ou tennis de table) avec son pourcentage de confiance.
4. Les probabilités pour les autres classes seront également affichées.

## Dépendances

* Python 3.9
* PyTorch 
* torchvision
* torchaudio
* Streamlit
* Pillow
* joblib

## Installation

1. Clonez ce dépôt.
2. Créez un environnement virtuel Python.
3. Installez les dépendances : `pip install -r requirements.txt`
4. Exécutez l'application Streamlit : `streamlit run app.py`

## Dockerisation

Un Dockerfile est inclus pour dockeriser l'application. Vous pouvez construire et exécuter l'image Docker avec les commandes suivantes :

1. docker build -t sport-image-classifier .
2. docker run -p 8501:8501 sport-image-classifier
