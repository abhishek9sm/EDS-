from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "cleaned_form_responses.csv")
df["baseline_wpm"] = df.groupby("person_id")["wpm_avg"].transform("mean")
df["baseline_acc"] = df.groupby("person_id")["acc_avg"].transform("mean")

features = ["baseline_wpm", "baseline_acc", "sleep_hours", "sleep_quality", "caffeine_mg", "stress"]
X = pd.get_dummies(df[features], columns=["stress"], drop_first=True)
y = df[["wpm_avg", "acc_avg"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

print(f"WPM -> R²: {r2_score(y_test['wpm_avg'], preds[:, 0]):.3f} | MAE: {mean_absolute_error(y_test['wpm_avg'], preds[:, 0]):.2f} WPM")
print(f"Acc -> MAE: {mean_absolute_error(y_test['acc_avg'], preds[:, 1]):.2f}%")

model.fit(X, y)
joblib.dump({"model": model, "columns": list(X.columns)}, BASE_DIR / "typing_model.joblib")
print(f"Saved trained model to {BASE_DIR / 'typing_model.joblib'}")
