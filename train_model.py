
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the original dataset
df = pd.read_csv("student-mat.csv", sep=";")

# 2. Define target and input features
target = "G3"
features = [col for col in df.columns if col != target]

# 3. Define categorical columns
categorical = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic"
]

# Check that the expected categorical columns exist
missing = set(categorical) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

numeric = [col for col in features if col not in categorical]

# 4. Convert feature columns to appropriate types
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 5. Preprocess numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric),
    ("cat", categorical_pipeline, categorical)
])

# 6. Create the ML pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2
    ))
])

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8. Train
model.fit(X_train, y_train)

# 9. Evaluate on unseen test records
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Model Evaluation")
print("------------------------------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Mean Absolute Error: {mae:.2f} grade points")
print(f"R2 Score: {r2:.3f}")

# 10. Save the trained model and feature configuration
joblib.dump({
    "model": model,
    "features": features,
    "numeric": numeric,
    "categorical": categorical,
    "mae": mae,
    "r2": r2
}, "student_model.pkl")

print("\nModel saved to student_model.pkl")
