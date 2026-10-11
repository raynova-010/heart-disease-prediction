"""Streamlit frontend for the heart disease risk model (educational use only).

Usage: .venv/bin/streamlit run app.py
"""
import math
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from predict import CONTEXT, DISCLAIMER, MODEL_PATH, QUESTIONS, risk_band

REPORTS = Path(__file__).parent / "reports"
COMPARISON_CSV = REPORTS / "comparison.csv"
CONFUSION_PNG = REPORTS / "confusion_matrices.png"
ACCENT = "#e34948"  # keep in sync with .streamlit/config.toml
DONT_KNOW = "Don't know"
MANY_SKIPPED = 4  # warn from this many skipped answers
METRICS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
# The CLI disclaimer without its terminal border, as one paragraph.
DISCLAIMER_TEXT = " ".join(l.strip() for l in DISCLAIMER.splitlines() if not l.startswith("="))

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def disclaimer():
    st.warning(DISCLAIMER_TEXT, icon="⚠️")


def ask(feature, question, kind, spec):
    """Render one question; returns None when the user doesn't know."""
    hint, details = CONTEXT[feature]
    label = f"{question} :gray[· {hint}]" if hint else question
    if kind == "numeric":
        lo, hi, integer = spec
        cast = int if integer else float
        return st.number_input(
            label, min_value=cast(lo), max_value=cast(hi), value=None,
            step=cast(1 if integer else 0.1), format="%d" if integer else "%.1f",
            placeholder=f"{DONT_KNOW} (leave blank)", key=feature, help=details,
        )
    options = [DONT_KNOW] + [text for text, _ in spec]
    return dict(spec).get(st.selectbox(label, options, key=feature, help=details))


def predict_page(threshold):
    st.subheader("Estimate your risk")
    st.caption("Answer what you can. Many items come from clinical tests (ECG, exercise "
               "stress test, fluoroscopy); leave those as \"Don't know\" if you haven't had them.")
    with st.form("patient", enter_to_submit=False):  # avoids the hint overlapping placeholders
        cols = st.columns(3)
        raw = {}
        for i, (feature, question, kind, spec) in enumerate(QUESTIONS):
            with cols[i % 3]:
                raw[feature] = ask(feature, question, kind, spec)
        submitted = st.form_submit_button("Estimate risk", type="primary")
    if not submitted:
        return

    answers = {f: math.nan if v is None else v for f, v in raw.items()}
    proba = float(load_model().predict_proba(pd.DataFrame([answers]))[0, 1])
    skipped = sum(v is None for v in raw.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated probability of heart disease", f"{proba:.0%}")
    c2.metric("Estimated risk level", risk_band(proba).capitalize(),
              help="Lower < 30%, moderate 30–60%, higher ≥ 60% (same bands as predict.py).")
    c3.metric("Answers skipped", f"{skipped} of {len(QUESTIONS)}")
    st.progress(proba, text=f"{proba:.0%}")
    call = "screen positive" if proba >= threshold else "screen negative"
    st.caption(f"At the sidebar decision threshold of {threshold:.0%}, the model would "
               f"**{call}**. The test-set metrics were measured at 50%.")

    if skipped == len(QUESTIONS):
        st.warning("You skipped every question, so this number says nothing about you. It "
                   "mostly reflects that, in the training data, patients with many unrecorded "
                   "values came from hospitals where most patients had heart disease.")
    elif skipped >= MANY_SKIPPED:
        st.warning(f"You skipped {skipped} questions. Skipped answers are filled in with typical "
                   "values and pull the estimate toward the middle; skipping cholesterol or blood "
                   "sugar pushes it up (see About & limitations). This estimate is less reliable.")
    elif skipped:
        st.info(f"You skipped {skipped} question(s); those were filled in with typical values.")


def comparison_page():
    st.subheader("Model comparison (held-out test set, 184 patients)")
    if not COMPARISON_CSV.exists():
        st.error(f"{COMPARISON_CSV} not found. Run train.py first.")
        return
    df = pd.read_csv(COMPARISON_CSV)
    table = df.set_index("model")[list(METRICS) + ["cv_recall", "cv_f1"]]
    table = table.rename(columns=METRICS | {"cv_recall": "CV recall", "cv_f1": "CV F1"})
    st.dataframe(
        table.style.format("{:.3f}").highlight_max(
            axis=0, props="background-color: rgba(227, 73, 72, 0.18); font-weight: 600"),
        width="stretch",
    )
    st.caption("Best value per column highlighted (ties share it). Test metrics use a 0.5 "
               "threshold; CV columns are 5-fold cross-validation on the training set.")
    st.info("The differences are small: on recall the three models are within one patient of "
            "each other on the test set, so the ranking is largely noise. Random Forest's "
            "clearest lead is ROC-AUC.")

    long = df.melt(id_vars="model", value_vars=list(METRICS), var_name="metric")
    long["metric"] = long["metric"].map(METRICS)
    base = alt.Chart().encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                axis=alt.Axis(format=".1f", tickCount=5, grid=True)),
        y=alt.Y("model:N", title=None, sort=list(df["model"])),
    )
    bars = base.mark_bar(color=ACCENT, cornerRadiusEnd=4, height=14).encode(
        tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")])
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("value:Q", format=".2f"))
    chart = (alt.layer(bars, labels, data=long)
             .properties(width=190, height=110)
             .facet(column=alt.Column("metric:N", title=None, sort=list(METRICS.values())))
             .configure_view(stroke=None))
    st.altair_chart(chart)

    st.markdown("**Confusion matrices (test set)**")
    if CONFUSION_PNG.exists():
        st.image(str(CONFUSION_PNG), width="stretch")
    with st.expander("Best hyperparameters (5-fold GridSearchCV, scored on recall)"):
        st.dataframe(df[["model", "best_params"]], hide_index=True, width="stretch")


def about_page():
    st.subheader("About & limitations")
    st.markdown("""
**Data.** The UCI Heart Disease dataset (via Kaggle): 920 patients pooled from 4 hospitals:
Cleveland (304), Hungary (293), VA Long Beach (200) and Switzerland (123). The target is
binarized: any diagnosed disease (`num > 0`) counts as disease; 55% of patients have it.

**Model.** A scikit-learn pipeline (cleaning → imputation → scaling/one-hot encoding → model).
Logistic regression, random forest and XGBoost were tuned with 5-fold stratified grid search on
recall; the best by test recall, then F1, was a random forest. See *Model comparison*.

**Limitations**
- **Many missing values.** Only Cleveland is nearly complete. Overall, `ca` (vessels on
  fluoroscopy) is 66% missing, `thal` 53%, `slope` 34%, and cholesterol about 22% once its
  impossible zeros are treated as missing.
- **The hospital effect.** Hospitals differ in disease rate (Switzerland: 93%) *and* in what
  they left unrecorded, so "this value is missing" partly reveals which hospital a patient came
  from. Skipping **cholesterol or fasting blood sugar** (mostly unrecorded at the high-disease
  Swiss and VA hospitals) **pushes estimates up**; skipping the clinical-test items mostly pulls
  them toward the middle (low risks rise, high risks fall). Skipping everything gives about 62%.
  Test scores are also likely optimistic for patients from a new hospital.
- **Untuned threshold.** Reported metrics use a 0.5 cutoff that was not tuned. Lowering it
  catches more true cases at the cost of more false alarms; the sidebar slider shows how the
  model's call changes, but the reported metrics stay at 0.5.
- **Small, old data.** 920 patients from the 1980s; it may not reflect today's patients.

**This is an educational tool, not medical advice.** It cannot diagnose anything. If you
have symptoms or concerns, talk to a doctor; if you have chest pain now, call emergency services.
""")


disclaimer()
st.title("Heart disease risk estimator")
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Probability at or above which the model's call is "
                               "'screen positive'. Metrics in this app were measured at 0.50.")
    st.caption("The lower/moderate/higher label uses fixed bands; the threshold only changes "
               "the model's yes/no call.")
predict_tab, compare_tab, about_tab = st.tabs(["Predict", "Model comparison", "About & limitations"])
with predict_tab:
    predict_page(threshold)
with compare_tab:
    comparison_page()
with about_tab:
    about_page()
st.divider()
disclaimer()
