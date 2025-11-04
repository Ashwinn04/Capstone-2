import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_dataset() -> pd.DataFrame:
    # Prefer engineered_data.csv if present (as in Capstone (4).ipynb), else Dataset.csv
    candidates = [
        os.path.join(PROJECT_ROOT, "engineered_data.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "engineered_data.csv"),
        os.path.join(PROJECT_ROOT, "Dataset.csv"),
    ]
    data_path = next((p for p in candidates if os.path.exists(p)), None)
    if not data_path:
        raise FileNotFoundError("No dataset found. Expected engineered_data.csv or Dataset.csv in project root.")
    df = pd.read_csv(data_path)
    # Normalize label column: Capstone (4).ipynb uses 'SepsisLabel'
    label_col = None
    for cand in ["SepsisLabel", "Sepsis_Label", "sepsis_label", "sepsisLabel"]:
        if cand in df.columns:
            label_col = cand
            break
    if not label_col:
        raise ValueError("Label column not found. Expected one of: SepsisLabel, Sepsis_Label")
    # Rename internally to Sepsis_Label for downstream consistency
    if label_col != "Sepsis_Label":
        df = df.rename(columns={label_col: "Sepsis_Label"})
    return df


def build_feature_list(df: pd.DataFrame, label_col: str = "Sepsis_Label"):
    # Prefer Person B's common feature set when available; otherwise use numeric columns (excluding label/IDs)
    preferred = [
        'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','FiO2','pH','PaCO2','SaO2',
        'BaseExcess','HCO3','WBC','Platelets','Creatinine','Bilirubin_total','Age',
        'MAP_rolling_mean_6hr','MAP_delta_1hr','qSOFA_Resp','qSOFA_MAP','qSOFA_Score_simple'
    ]
    drop_like = {"Patient_ID", "patient_id", "RecordID", "Unnamed: 0"}
    available_pref = [c for c in preferred if c in df.columns]
    if available_pref:
        feature_list = available_pref
    else:
        # Fallback: all non-label, non-ID columns
        feature_list = [c for c in df.columns if c not in drop_like and c != label_col]
    return feature_list


def coerce_features(df: pd.DataFrame, feature_list):
    X = df[feature_list].copy()
    # Minimal categorical handling: map Gender; everything else -> numeric
    for col in X.columns:
        if X[col].dtype == object:
            if X[col].nunique() <= 3 and set(map(str, X[col].dropna().unique())) <= {"Male", "Female", "Other"}:
                X[col] = X[col].map({"Male": 1, "Female": 0, "Other": 0}).astype(float)
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def train_hgb_pipeline(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_dataset()

    feature_list = build_feature_list(df)
    X = coerce_features(df, feature_list)
    y = df["Sepsis_Label"].astype(int).values

    # Patient-wise split is ideal; in absence, use stratified split for artifact creation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train_hgb_pipeline(X_train, y_train)

    # Persist artifacts expected by integration_real.py
    with open(os.path.join(OUT_DIR, "feature_list.json"), "w") as f:
        json.dump(feature_list, f, indent=2)

    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    with open(os.path.join(OUT_DIR, "imputer_final.pkl"), "wb") as f:
        pickle.dump(imputer, f)
    with open(os.path.join(OUT_DIR, "scaler_final.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT_DIR, "model_final_hgb.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print("✓ Saved baseline artifacts to outputs/: feature_list.json, imputer_final.pkl, scaler_final.pkl, model_final_hgb.pkl")


if __name__ == "__main__":
    main()


