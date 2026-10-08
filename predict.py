"""Interactive heart disease risk estimator (educational use only).

Usage: .venv/bin/python predict.py
"""
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

from heart_features import FEATURES

MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"

DISCLAIMER = """\
==================================================================
  DISCLAIMER: This is an EDUCATIONAL tool, NOT medical advice.
  It was trained on a small, decades-old research dataset (920
  patients from 4 hospitals) and cannot diagnose anything. The
  number it produces is a statistical estimate that may be wrong.
  If you have symptoms or concerns, talk to a qualified doctor.
  If you have chest pain right now, call emergency services.
=================================================================="""

# (feature, question, kind, spec)
#   numeric:     spec = (min, max, integer_only)
#   categorical: spec = [(display label, value stored in the dataset), ...]
YES_NO = [("Yes", "True"), ("No", "False")]
QUESTIONS = [
    ("age", "Age in years", "numeric", (18, 100, True)),
    ("sex", "Sex", "categorical", [("Male", "Male"), ("Female", "Female")]),
    ("cp", "Chest pain type", "categorical", [
        ("Typical angina (pressure on exertion, relieved by rest)", "typical angina"),
        ("Atypical angina", "atypical angina"),
        ("Non-anginal chest pain", "non-anginal"),
        ("No chest pain (asymptomatic)", "asymptomatic"),
    ]),
    ("trestbps", "Resting systolic blood pressure (mm Hg, e.g. 120)", "numeric", (80, 220, False)),
    ("chol", "Serum cholesterol (mg/dL, e.g. 200)", "numeric", (100, 600, False)),
    ("fbs", "Is your fasting blood sugar above 120 mg/dL?", "categorical", YES_NO),
    ("restecg", "Resting ECG result", "categorical", [
        ("Normal", "normal"),
        ("ST-T wave abnormality", "st-t abnormality"),
        ("Left ventricular hypertrophy", "lv hypertrophy"),
    ]),
    ("thalch", "Maximum heart rate reached in an exercise test (bpm)", "numeric", (60, 220, False)),
    ("exang", "Did exercise bring on chest pain (angina)?", "categorical", YES_NO),
    ("oldpeak", "ST depression during exercise vs. rest (mm, e.g. 1.0)", "numeric", (-3, 7, False)),
    ("slope", "Slope of the peak exercise ST segment", "categorical", [
        ("Upsloping", "upsloping"),
        ("Flat", "flat"),
        ("Downsloping", "downsloping"),
    ]),
    ("ca", "Number of major vessels seen on fluoroscopy (0-3)", "numeric", (0, 3, True)),
    ("thal", "Thallium stress test result", "categorical", [
        ("Normal", "normal"),
        ("Fixed defect", "fixed defect"),
        ("Reversible defect", "reversable defect"),
    ]),
]
assert sorted(q[0] for q in QUESTIONS) == sorted(FEATURES)


def ask_numeric(question, lo, hi, integer):
    while True:
        raw = input(f"{question} [{lo} to {hi}, Enter to skip]: ").strip()
        if not raw:
            return math.nan
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number, or press Enter to skip.")
            continue
        if integer and not value.is_integer():
            print("  Please enter a whole number.")
        elif not lo <= value <= hi:
            print(f"  Please enter a value between {lo} and {hi}.")
        else:
            return value


def ask_choice(question, options):
    print(question)
    for i, (label, _) in enumerate(options, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input(f"Choose 1-{len(options)} [Enter to skip]: ").strip()
        if not raw:
            return math.nan
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(f"  Please enter a number from 1 to {len(options)}, or press Enter to skip.")


def interview():
    print("\nAnswer each question. Press Enter to skip anything you don't know;"
          "\nmany items come from clinical tests (ECG, stress test, fluoroscopy).\n")
    answers = {}
    for feature, question, kind, spec in QUESTIONS:
        if kind == "numeric":
            answers[feature] = ask_numeric(question, *spec)
        else:
            answers[feature] = ask_choice(question, spec)
        print()
    return answers


def report(model, answers):
    proba = model.predict_proba(pd.DataFrame([answers]))[0, 1]
    skipped = [f for f, v in answers.items() if isinstance(v, float) and math.isnan(v)]
    if proba < 0.3:
        band = "lower"
    elif proba < 0.6:
        band = "moderate"
    else:
        band = "higher"

    print("-" * 66)
    print(f"  Estimated probability of heart disease: {proba:.0%}  ({band} estimated risk)")
    print("-" * 66)
    if skipped:
        print(f"You skipped {len(skipped)} of {len(answers)} questions; those were filled in with"
              "\ntypical values, so this estimate is less reliable.")
    print("This estimate is based on patterns in historical data and is not a"
          "\ndiagnosis. Please discuss any concerns with a doctor.\n")


def main():
    print(DISCLAIMER)
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    try:
        while True:
            report(model, interview())
            if input("Assess another person? [y/N]: ").strip().lower() != "y":
                break
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    print("\nReminder: educational tool only, not medical advice.")


if __name__ == "__main__":
    main()
