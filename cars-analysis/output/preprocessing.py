import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== 1. Remplacement des valeurs catégorielles ========
def replace_categorical_by_numerical(data):
    data = data.copy()

    # Nettoyage robuste de la colonne Levy
    print("Cleaning 'Levy' column...")
    data["Levy"] = (
        data["Levy"]
        .astype(str)                 # Convertir en string
        .str.strip()                 # Enlever les espaces
        .replace({"-": "0", "": "0"}) # Remplacer tous les tirets ou vides
    )
    data["Levy"] = pd.to_numeric(data["Levy"], errors="coerce").fillna(0)

    # Nettoyage Engine volume
    data["Engine volume"] = data["Engine volume"].astype(str)
    data["Engine volume"] = data["Engine volume"].str.replace("Turbo", "", case=False)
    data["Engine volume"] = pd.to_numeric(data["Engine volume"], errors='coerce')

    # Nettoyage Mileage
    data["Mileage"] = data["Mileage"].astype(str)
    data["Mileage"] = data["Mileage"].str.replace("km", "")
    data["Mileage"] = pd.to_numeric(data["Mileage"], errors='coerce')

    return data

# ======== 2. Transformation des colonnes ========
def column_transformations(data):
    data["Mileage_log"] = np.log(data["Mileage"]).replace(-np.inf, 1e-6)
    data["Levy_log"] = np.log(data["Levy"].replace(0, 1e-6)).replace(-np.inf, 1e-6)
    data["Engine_volume_log"] = np.log(data["Engine volume"]).replace(-np.inf, 1e-6)
    return data

# ======== 3. Nettoyage des outliers ========
def clean_outliers(df, col):
    try:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    except Exception as e:
        print(f"⚠️ Error cleaning column {col}: {e}")
    return df

# ======== 4. Feature engineering ========
def engineer_features(df):
    current_year = pd.Timestamp.now().year
    df["age"] = current_year - df["Prod. year"]
    return df

# ======== 5. Pipeline complète ========
def preprocessing_pipline(df: pd.DataFrame):
    print("Preprocessing started.....")
    print(f"Initial shape : {df.shape}")

    # Suppression des doublons
    df = df.drop_duplicates()
    print(f"After dropping duplicates: {df.shape}")

    # Remplacement des valeurs catégorielles
    print("Replacing categorical values.....")
    df = replace_categorical_by_numerical(df)

    # Nettoyage des outliers
    print("Cleaning outliers.....")
    for col in ["Price", "Levy", "Engine volume", "Mileage"]:
        df = clean_outliers(df, col)
    print(f"After cleaning outliers: {df.shape}")

    # Feature engineering
    print("Feature engineering.....")
    df = engineer_features(df)

    # Transformations log
    print("Column transformations...")
    df = column_transformations(df)

    # Suppression de certaines colonnes inutiles
    print("Dropping columns...")
    df = df.drop(["Doors", "Prod. year"], axis=1, errors="ignore")

    print("✅ Preprocessing completed successfully!")
    print("Final shape:", df.shape)

    return df
