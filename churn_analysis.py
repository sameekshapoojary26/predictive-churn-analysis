import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Load dataset
files = glob.glob("*.csv")
if not files:
    raise FileNotFoundError("Upload your churn CSV dataset first.")

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip()

# Detect churn target
targets = ["Churn", "churn", "Exited", "Attrition", "Customer_Status"]
target = next((c for c in targets if c in df.columns), None)

if target is None:
    raise ValueError("Churn target column not found. Check dataset columns.")

df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = LabelEncoder().fit_transform(df[target].astype(str))

# Remove ID columns
X = X.drop(columns=[
    c for c in X.columns
    if "id" in c.lower() or c.lower() in ["customerid"]
], errors="ignore")

# Preprocessing
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    print("\n", name)
    print(classification_report(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": auc
    })

    if name == "Random Forest":
        pd.DataFrame({
            "Actual_Churn": y_test,
            "Predicted_Churn": predictions,
            "Churn_Probability": probabilities
        }).to_csv("churn_predictions.csv", index=False)

pd.DataFrame(results).to_csv("model_results.csv", index=False)

print("\nANALYSIS COMPLETED SUCCESSFULLY!")
print(pd.DataFrame(results))
