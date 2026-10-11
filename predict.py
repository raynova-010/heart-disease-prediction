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

# Plain-language context per question: (short hint shown beside the label, longer explanation).
CONTEXT = {
    "age": ("", "Your age in whole years."),
    "sex": ("", "Sex as recorded in the original study data (male or female)."),
    "cp": ("the kind you usually have, if any",
           "Typical angina: pressure or tightness in the chest brought on by exertion or stress "
           "and relieved by rest. Atypical angina: chest pain with only some of those features. "
           "Non-anginal: chest pain that doesn't seem heart-related, e.g. sharp pain that changes "
           "with breathing or when pressing on the chest. Asymptomatic: no chest pain."),
    "trestbps": ("normal is under 120",
                 "The top (systolic) number of a blood pressure reading taken at rest, e.g. 120 "
                 "in '120/80'."),
    "chol": ("from a blood test; under 200 is desirable",
             "Total cholesterol from a blood test. If your report shows mmol/L, multiply by 38.67 "
             "to get mg/dL (e.g. 5.2 mmol/L ≈ 200 mg/dL)."),
    "fbs": ("from a fasting blood test",
            "Blood sugar measured after not eating for at least 8 hours. 120 mg/dL is about "
            "6.7 mmol/L. Choose \"Don't know\" if you haven't had this test."),
    "restecg": ("from an ECG at rest",
                "The result of an electrocardiogram (ECG) taken while resting. ST-T wave "
                "abnormality: changes in part of the heartbeat tracing. Left ventricular "
                "hypertrophy: signs that the heart's main pumping chamber has a thickened wall."),
    "thalch": ("from an exercise stress test",
               "The highest heart rate reached on a treadmill or bike stress test, not your "
               "everyday resting pulse. A rough expected maximum is 220 minus your age."),
    "exang": ("during an exercise stress test",
              "Whether you had chest pain (angina) during an exercise stress test."),
    "oldpeak": ("from an exercise ECG; 0 means no change",
                "How far the ST segment of the ECG tracing dropped during exercise compared "
                "with rest, in millimetres. Larger drops are more concerning."),
    "slope": ("from an exercise ECG",
              "The shape of the ST segment of the ECG tracing at peak exercise. Upsloping is "
              "usually the least concerning; flat and downsloping are more concerning."),
    "ca": ("from cardiac fluoroscopy",
           "The number of major heart vessels (0 to 3) highlighted on a fluoroscopy (X-ray "
           "imaging) report. Most people haven't had this test; choose \"Don't know\" if so."),
    "thal": ("from a nuclear (thallium) stress test",
             "A scan of blood flow to the heart muscle at rest and during exercise. Fixed "
             "defect: an area with low blood flow both at rest and during exercise, often from "
             "old damage. Reversible defect: low blood flow only during exercise."),
}
assert set(CONTEXT) == set(FEATURES)


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


def risk_band(proba):
    if proba < 0.3:
        return "lower"
    if proba < 0.6:
        return "moderate"
    return "higher"


def report(model, answers):
    proba = model.predict_proba(pd.DataFrame([answers]))[0, 1]
    skipped = [f for f, v in answers.items() if isinstance(v, float) and math.isnan(v)]
    band = risk_band(proba)

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
