import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import joblib
import os

# 1. Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# 2. Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# 3. Définir features et cible
feature_cols = ['age','sexe_encoded','temperature','tension_sys',
                'toux','fatigue','maux_tete','region_encoded']
X = df[feature_cols]
y = df['diagnostic']
print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")

# 4. Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nEntraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# 5. Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"\nModèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Classes : {list(model.classes_)}")

# 6. Évaluer
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")
print("\nMatrice de confusion :")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# 7. Sérialiser
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
size = os.path.getsize("models/model.pkl")
print(f"\nModèle sauvegardé : models/model.pkl ({size/1024:.1f} Ko)")

# 8. Tester le modèle rechargé
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
print(f"\nModèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

nouveau_patient = {
    'age': 28, 'sexe': 'F', 'temperature': 39.5,
    'tension_sys': 110, 'toux': True,
    'fatigue': True, 'maux_tete': True, 'region': 'Dakar'
}
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]
features = [nouveau_patient['age'], sexe_enc,
            nouveau_patient['temperature'],
            nouveau_patient['tension_sys'],
            int(nouveau_patient['toux']),
            int(nouveau_patient['fatigue']),
            int(nouveau_patient['maux_tete']), region_enc]
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()
print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")
print(f"\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")

# ==========================================
# EXERCICE 1 - Importance des features
# ==========================================
print("\n--- Exercice 1 : Importance des features ---")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")

# ==========================================
# EXERCICE 2 - 3 patients fictifs
# ==========================================
print("\n--- Exercice 2 : Test de 3 patients fictifs ---")

patients_test = [
    {'age': 8,  'sexe': 'M', 'temperature': 36.8,
     'tension_sys': 95,  'toux': False, 'fatigue': False,
     'maux_tete': False, 'region': 'Dakar'},
    {'age': 45, 'sexe': 'F', 'temperature': 40.2,
     'tension_sys': 130, 'toux': True,  'fatigue': True,
     'maux_tete': True,  'region': 'Thies'},
    {'age': 68, 'sexe': 'M', 'temperature': 38.0,
     'tension_sys': 150, 'toux': True,  'fatigue': True,
     'maux_tete': False, 'region': 'Kaolack'},
]
descriptions = [
    "Enfant 8 ans, sans symptômes",
    "Adulte 45 ans, forte fièvre + symptômes",
    "Patient 68 ans, toux + fatigue",
]
for i, (p, desc) in enumerate(zip(patients_test, descriptions)):
    s_enc = le_sexe.transform([p['sexe']])[0]
    r_enc = le_region.transform([p['region']])[0]
    feat = [p['age'], s_enc, p['temperature'], p['tension_sys'],
            int(p['toux']), int(p['fatigue']),
            int(p['maux_tete']), r_enc]
    diag = model.predict([feat])[0]
    proba = model.predict_proba([feat])[0].max()
    print(f"\nPatient {i+1} : {desc}")
    print(f"  Diagnostic : {diag} ({proba:.1%})")