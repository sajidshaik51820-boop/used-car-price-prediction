import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# ============================================================
# USED CAR AI - MODEL TRAINING
# ============================================================

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cardekho_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_car_price_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ============================================================
# 3. REMOVE UNNECESSARY COLUMNS
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "selling_price"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\n❌ Missing columns:")
    print(missing_columns)
    raise ValueError("Dataset does not contain all required columns.")

# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"\nDuplicates removed: {before - after}")

# ============================================================
# 6. CONVERT NUMERICAL COLUMNS
# ============================================================

numeric_columns = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "selling_price"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ============================================================
# 7. REMOVE INVALID TARGET VALUES
# ============================================================

df = df.dropna(
    subset=["selling_price"]
)

# Price must be positive
df = df[
    df["selling_price"] > 0
]

# ============================================================
# 8. REMOVE EXTREME / INVALID VALUES
# ============================================================

df = df[
    (df["vehicle_age"] >= 0) &
    (df["vehicle_age"] <= 30)
]

df = df[
    (df["km_driven"] >= 0) &
    (df["km_driven"] <= 1000000)
]

df = df[
    (df["selling_price"] > 10000)
]

print(f"\nClean dataset rows: {len(df):,}")

# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

df["km_per_year"] = (
    df["km_driven"] /
    (df["vehicle_age"] + 1)
)

print("\nFeature engineering completed.")

# ============================================================
# 10. FEATURES
# ============================================================

features = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "km_per_year",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

target = "selling_price"

X = df[features]

y = df[target]

# ============================================================
# 11. COLUMN TYPES
# ============================================================

categorical_features = [
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

numerical_features = [
    "vehicle_age",
    "km_driven",
    "km_per_year",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

# ============================================================
# 12. PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)

# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nData split:")
print(f"Training rows : {len(X_train):,}")
print(f"Testing rows  : {len(X_test):,}")

# ============================================================
# 14. LOG TRANSFORM TARGET
# ============================================================

# This helps the model handle very cheap and very expensive
# cars without allowing expensive cars to dominate training.

y_train_log = np.log1p(y_train)

# ============================================================
# 15. XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# 16. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n========================================")
print("       TRAINING XGBOOST MODEL")
print("========================================")

pipeline.fit(
    X_train,
    y_train_log
)

print("\n✅ Training completed!")

# ============================================================
# 18. PREDICTION
# ============================================================

predicted_log = pipeline.predict(
    X_test
)

# Convert prediction back to original ₹ price

y_pred = np.expm1(
    predicted_log
)

# Make sure predictions cannot be negative

y_pred = np.maximum(
    y_pred,
    0
)

# ============================================================
# 19. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

# Mean percentage error

percentage_error = (
    np.abs(
        (y_test - y_pred) /
        y_test
    ) * 100
)

mean_percentage_error = percentage_error.mean()

accuracy = 100 - mean_percentage_error

# ============================================================
# 20. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("        MODEL PERFORMANCE")
print("========================================")

print(
    f"MAE                 : ₹{mae:,.2f}"
)

print(
    f"RMSE                : ₹{rmse:,.2f}"
)

print(
    f"R² Score            : {r2:.4f}"
)

print(
    f"Mean % Error        : {mean_percentage_error:.2f}%"
)

print(
    f"Approx. Accuracy    : {accuracy:.2f}%"
)

print("========================================")

# ============================================================
# 21. SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Difference"] = (
    results["Predicted Price"]
    - results["Actual Price"]
)

results["Error %"] = (
    np.abs(results["Difference"])
    / results["Actual Price"]
) * 100

print("\nSample predictions:\n")

print(
    results.head(10).to_string(
        index=False
    )
)

# ============================================================
# 22. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n========================================")
print("        MODEL SAVED SUCCESSFULLY")
print("========================================")

print(
    f"Model path: {MODEL_PATH}"
)

# ============================================================
# 23. VERIFY MODEL
# ============================================================

if os.path.exists(MODEL_PATH):

    file_size = os.path.getsize(
        MODEL_PATH
    )

    print(
        f"Model size: {file_size:,} bytes"
    )

    print(
        "\n✅ MODEL FILE VERIFIED!"
    )

else:

    print(
        "\n❌ MODEL FILE NOT FOUND!"
    )