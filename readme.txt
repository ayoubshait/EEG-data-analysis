Projet EEG : Classification des périodes d'yeux ouverts et fermés

Ce projet utilise des signaux EEG (Électroencéphalogramme) pour classifier les périodes pendant lesquelles les yeux sont ouverts ou fermés. À partir des signaux EEG, nous extrayons les puissances des bandes Alpha et Theta et entraînons un modèle de RandomForest pour faire cette classification.

Objectifs du projet
- Extraction des bandes de fréquences Alpha (8-12 Hz) et Theta (4-8 Hz) à partir des signaux EEG.
- Classification des périodes d'yeux ouverts et fermés à l'aide d'un modèle d'apprentissage automatique.
- Affichage des matrices de confusion brute et normalisée pour visualiser les performances du modèle.

Prérequis

Avant d'exécuter ce projet, vous devez disposer des éléments suivants :

Les bibliothèques Python suivantes :
    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn

# il faudra executer ces commandes:

pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn

pip install numpy pandas scikit-learn matplotlib seaborn

Exécution du projet :

1. Modifier le chemin du fichier EEG:
Avant de commencer, vous devez modifier le chemin du fichier EEG dans partie1.py pour qu'il corresponde à l'emplacement de vos données brutes :

file_path = 'votre/chemin/vers/le/fichier/OpenBCI-RAW.csv'

2. Exécuter partie1.py:

Pour exécuter partie1.py, utilisez la commande suivante :

python partie1.py

Ce script va nous permettre aussi :
    Sauvegarder les données dans alpha_data.csv et theta_data.csv.
    Enregistrer ce graphique dans le répertoire courant.

3. Exécuter partie2.py

python partie2.py




