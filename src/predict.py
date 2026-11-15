from pathlib import Path
import joblib
import pandas as pd

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "typing_model.joblib"

saved = joblib.load( project_root / "typing_model.joblib")
model, cols = saved["model"], saved["columns"]

print("--- Anonymous Performance Predictor ---")
b_wpm = float(input("Baseline WPM (e.g. 50): ") or 50.0)
b_acc = float(input("Baseline Accuracy % (e.g. 94): ") or 94.0)
s_hrs = float(input("Sleep hours last night (e.g. 7.5): ") or 7.5)
s_qual = int(input("Sleep quality 1-5 (e.g. 4): ") or 4)
caf = float(input("Caffeine in mg (0 / 50 / 100): ") or 50.0)
stress = (input("Stress level (low / medium / high): ") or "low").strip().lower()

user_df = pd.DataFrame([{
    "baseline_wpm": b_wpm,
    "baseline_acc": b_acc,
    "sleep_hours": s_hrs,
    "sleep_quality": s_qual,
    "caffeine_mg": caf,
    "stress_low": 1 if stress == "low" else 0,
    "stress_medium": 1 if stress == "medium" else 0
}])[cols]

pred_wpm, pred_acc = model.predict(user_df)[0]
print(f"\nPredicted Typing Speed:    {pred_wpm:.2f} WPM")
print(f"Predicted Typing Accuracy: {min(pred_acc, 100.0):.2f}%")
