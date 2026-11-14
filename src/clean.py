import numpy as np
import pandas as pd
from pathlib import Path

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
path_in = BASE_DIR / "data" / "raw_response_form.csv"

RENAME_COLS = {
    "Timestamp": "timestamp",
    "Select your person ID": "person_id",
    "Enter the current date": "date",
    "Enter your day number": "day_number",
    "Enter the test time": "test_time",
    "Enter your sleep hours for last night": "sleep_hours",
    "Rate your sleep quality from the past night": "sleep_quality",
    "Select your stress level before the test": "stress",
    "Are you a caffeine user?": "caffeine_user_reported",
    "How do you like your coffee/tea monsieur?": "caffeine_mg",
    "Enter the time of your last caffeine intake": "last_caffeine_time",
    "Enter WPM for run 1": "wpm_run1",
    "Enter WPM for run 2": "wpm_run2",
    "Enter accuracy percentage for run 1": "acc_run1",
    "Enter accuracy percentage for run 2": "acc_run2",
    "Enter any additional notes": "user_notes",
}

LEVEL_TO_MG = {"light": 50.0, "medium": 100.0, "heavy": 150.0}
NON_CAFFEINE_USERS = {"P3"}


def parse_accuracy(val):
    if pd.isna(val):
        return np.nan
    num = float(str(val).replace("%", "").strip())
    if num > 100:
        num /= 100.0
    elif num <= 1:
        num *= 100.0
    return round(num, 2)


def clean_form_data(path_in, path_out="cleaned_form_responses.csv"):

    df = pd.read_csv(BASE_DIR / path_in, dtype=str, keep_default_na=False)
    df = df.replace("", np.nan).rename(columns=RENAME_COLS)

    df["id_upper"] = df["person_id"].str.strip().str.upper()
    df = df[df["id_upper"] != "DEMO"].copy()
    feature_cols = [c for c in df.columns if c not in ["timestamp", "id_upper"]]
    df = df.dropna(subset=feature_cols, how="all").copy()

    missing_p4 = df["person_id"].isna() & (df["date"] == "10/28/2026") & (df["test_time"] == "7:20:00 PM")
    df.loc[missing_p4, "person_id"] = "P4"

    df["person_id"] = df["person_id"].str.strip().str.upper()
    df["stress"] = df["stress"].str.strip().str.lower()

    df.loc[(df["person_id"] == "P2") & (df["date"].isin(["8/19/2026", "2026-08-19"])), "date"] = "10/12/2026"
    df.loc[(df["person_id"] == "P2") & (df["date"].isna()) & (df["day_number"] == "3"), "date"] = "10/13/2026"
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d")

    for col in ["test_time", "last_caffeine_time"]:
        df[col] = pd.to_datetime(df[col], format="mixed").dt.strftime("%H:%M")

    p4_mask = df["person_id"] == "P4"
    df.loc[p4_mask & (df["last_caffeine_time"] == "17:30"), "last_caffeine_time"] = "05:30"
    df.loc[p4_mask & (df["last_caffeine_time"] == "17:45"), "last_caffeine_time"] = "05:45"

    p5_mask = df["person_id"] == "P5"
    df.loc[p5_mask & (df["date"] == "2026-10-23") & (df["last_caffeine_time"] == "09:00"), "last_caffeine_time"] = "08:45"
    df.loc[p5_mask & (df["date"] == "2026-10-28") & (df["last_caffeine_time"] == "09:00"), "last_caffeine_time"] = "08:25"

    for col in ["acc_run1", "acc_run2"]:
        df[col] = df[col].map(parse_accuracy)
    for col in ["wpm_run1", "wpm_run2", "sleep_hours", "sleep_quality"]:
        df[col] = pd.to_numeric(df[col])

    df.loc[(df["person_id"] == "P2") & (df["date"] == "2026-10-24") & (df["wpm_run2"] == 88.0), "wpm_run2"] = 48.0

    df["caffeine_user_reported"] = df["caffeine_user_reported"].replace({"0": "No"}).str.strip().str.title()
    raw_caf = df["caffeine_mg"].fillna("0").str.strip().str.lower()
    mg = raw_caf.map(LEVEL_TO_MG).astype(float).fillna(pd.to_numeric(raw_caf, errors="coerce")).fillna(0.0)
    df["caffeine_level"] = np.where(raw_caf.isin(LEVEL_TO_MG), raw_caf.str.title(), np.where(mg > 0, "Numeric", pd.NA))
    df["caffeine_mg"] = mg

    df.loc[df["person_id"].isin(NON_CAFFEINE_USERS), ["caffeine_mg", "caffeine_level", "last_caffeine_time", "caffeine_user_reported"]] = [0.0, np.nan, np.nan, "No"]
    df.loc[p5_mask, "caffeine_user_reported"] = "Yes"

    df["_sort_date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["person_id", "_sort_date", "test_time"]).reset_index(drop=True)
    df["day_number"] = df.groupby("person_id").cumcount() + 1

    df["wpm_avg"] = ((df["wpm_run1"] + df["wpm_run2"]) / 2.0).round(2)
    df["acc_avg"] = ((df["acc_run1"] + df["acc_run2"]) / 2.0).round(2)
    df["user_notes"] = df["user_notes"].replace({"None": np.nan, "none": np.nan})

    final_cols = [
        "person_id", "date", "day_number", "test_time",
        "sleep_hours", "sleep_quality", "stress",
        "caffeine_mg", "caffeine_level", "last_caffeine_time", "caffeine_user_reported",
        "wpm_run1", "wpm_run2", "wpm_avg",
        "acc_run1", "acc_run2", "acc_avg",
        "user_notes"
    ]

    df[final_cols].to_csv(BASE_DIR / path_out, index=False)
    print(f"Successfully cleaned {len(df)} rows -> saved to {path_out}")
    return df[final_cols]


if __name__ == "__main__":
    clean_form_data("data/raw_response_form.csv", "data/cleaned_form_responses.csv")
