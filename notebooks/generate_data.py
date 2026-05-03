import pandas as pd
import numpy as np

np.random.seed(42)
n = 500
regions = ['Dakar','Thies','Saint-Louis','Ziguinchor','Kaolack',
           'Tambacounda','Diourbel','Fatick','Kolda','Louga']

rows = []
diagnostics = ['palu']*135 + ['grippe']*130 + ['sain']*125 + ['typh']*110

for i, diag in enumerate(diagnostics):
    age = np.random.randint(5, 75)
    sexe = np.random.choice(['M','F'])
    if diag == 'palu':
        temp = round(np.random.uniform(38.5, 41.0), 1)
        toux = bool(np.random.choice([True,False]))
        fatigue = True
        maux_tete = True
    elif diag == 'grippe':
        temp = round(np.random.uniform(38.0, 39.5), 1)
        toux = True
        fatigue = True
        maux_tete = bool(np.random.choice([True,False]))
    elif diag == 'typh':
        temp = round(np.random.uniform(38.5, 40.5), 1)
        toux = False
        fatigue = True
        maux_tete = bool(np.random.choice([True,False]))
    else:
        temp = round(np.random.uniform(36.0, 37.5), 1)
        toux = False
        fatigue = False
        maux_tete = False
    tension = np.random.randint(90, 160)
    region = np.random.choice(regions)
    rows.append([age, sexe, temp, tension, toux, fatigue, maux_tete, region, diag])

df = pd.DataFrame(rows, columns=['age','sexe','temperature','tension_sys',
                                  'toux','fatigue','maux_tete','region','diagnostic'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/patients_dakar.csv', index=False)
print(f"Dataset créé : {df.shape}")
print(df['diagnostic'].value_counts())