import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger les données Alpha et Theta calculées à partir de la partie 1
alpha_data = np.loadtxt('alpha_data.csv', delimiter=',')  # 300 échantillons, 8 canaux
theta_data = np.loadtxt('theta_data.csv', delimiter=',')  # 300 échantillons, 8 canaux

# Vérifier la forme des données
print(f"Shape of alpha_data: {alpha_data.shape}")
print(f"Shape of theta_data: {theta_data.shape}")

# 2. Créer les labels en fonction des intervalles de 30 secondes
labels = np.concatenate([
    np.ones(30),     # 0s ~ 30s : Yeux ouverts (préparation)
    np.zeros(30),    # 30s ~ 1min : Yeux fermés
    np.ones(30),     # 1min ~ 1min30s : Yeux ouverts
    np.zeros(30),    # 1min30s ~ 2mins : Yeux fermés
    np.ones(30),     # 2mins ~ 2mins30s : Yeux ouverts
    np.zeros(30),    # 2mins30s ~ 3mins : Yeux fermés
    np.ones(30),     # 3mins ~ 3mins30s : Yeux ouverts
    np.zeros(30),    # 3mins30s ~ 4mins : Yeux fermés
    np.ones(30),     # 4mins ~ 4mins30s : Yeux ouverts
    np.ones(30)      # 4mins30s ~ 5mins : Yeux ouverts (fin)
])

# Vérifier que les labels correspondent bien aux données (300 échantillons)
print(f"Shape of labels: {labels.shape}")

# 3. Concaténer les puissances Alpha et Theta en un tableau de caractéristiques (features)
features = np.concatenate([alpha_data, theta_data], axis=1)

# Vérifier la forme des données fusionnées
print(f"Shape of features: {features.shape}")

# 4. Diviser les données en ensemble d'entraînement (80%) et ensemble de test (20%)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 5. Créer et entraîner le modèle RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Faire des prédictions sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Évaluer le modèle : afficher la précision pendant l'entraînement et la précision finale
print("----- Training done -----")
training_mean_accuracy = np.mean(cross_val_score(rf_model, X_train, y_train, cv=5))
final_accuracy = rf_model.score(X_test, y_test)
print(f"Training Mean Accuracy = {training_mean_accuracy}")
print(f"Training Final Accuracy = {final_accuracy}")

# Afficher le rapport de classification (précision, rappel, F1-score)
print(classification_report(y_test, y_pred))

# Générer la matrice de confusion brute
conf_matrix = confusion_matrix(y_test, y_pred)

# Normaliser la matrice de confusion de manière sécurisée
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Si certaines lignes ont une somme égale à zéro, remplacez les NaN par zéro :
conf_matrix_normalized = np.nan_to_num(conf_matrix_normalized)

# figure avec deux sous-graphiques pour fusionner les deux matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# la matrice de confusion brute (gauche)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix, without normalization")
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted label')

# la matrice de confusion normalisée (droite)
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap="Blues", ax=axes[1])
axes[1].set_title("Normalized Confusion Matrix")
axes[1].set_ylabel('True label')
axes[1].set_xlabel('Predicted label')

# Ajuster la mise en page
plt.tight_layout()
plt.show()

