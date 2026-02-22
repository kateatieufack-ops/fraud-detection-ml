import pandas as pd

# Charger les données
df = pd.read_csv("creditcard.csv")

# Afficher les 5 premières lignes
print(df.head())

# Dimensions du dataset
print("Shape :", df.shape)

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Distribution des classes
print(df["Class"].value_counts())

from sklearn.model_selection import train_test_split
# Séparer variables explicatives et cible
X = df.drop("Class", axis=1)
y = df["Class"]
# Séparer train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape :", X_train.shape)
print("Test shape :", X_test.shape)

print("Fraudes dans train :", y_train.value_counts())
print("Fraudes dans test :", y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"  # important pour déséquilibre
) 
# Entraîner
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Matrice de confusion
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# Rapport complet
print("\nClassification report :")
print(classification_report(y_test, y_pred))

import joblib

# Sauvegarder le modèle
joblib.dump(model, "fraud_model.pkl")

print("Modèle sauvegardé avec succès.")