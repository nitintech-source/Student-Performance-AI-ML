from pathlib import Path
from datetime import datetime
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "student_performance.csv"
FEATURES = ["study_hours", "attendance", "previous_score", "sleep_hours", "assignments_completed"]
TARGET = "final_score"

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction")
st.caption("AI/ML training project • Python + Scikit-learn + Streamlit")
st.info(
    "Educational demonstration only. Predictions are estimates from a synthetic demo dataset, "
    "not official grades or guaranteed outcomes."
)

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(df):
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds),
    }
    return model, metrics, X_test, y_test, preds

def validate_dataset(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, "Missing required columns: " + ", ".join(missing)
    clean = df[required].copy()
    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna()
    if len(clean) < 30:
        return False, "Please provide at least 30 valid rows to train a demo model."
    if ((clean["attendance"] < 0) | (clean["attendance"] > 100)).any():
        return False, "Attendance must be between 0 and 100."
    if ((clean[TARGET] < 0) | (clean[TARGET] > 100)).any():
        return False, "final_score must be between 0 and 100."
    return True, clean

default_df = load_default_data()
st.sidebar.header("⚙️ Data & student inputs")
uploaded = st.sidebar.file_uploader("Optional: upload your own CSV dataset", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        valid, result = validate_dataset(raw_df)
        if valid:
            df = result
            st.sidebar.success(f"Loaded {len(df)} valid rows from your CSV.")
        else:
            st.sidebar.error(result)
            df = default_df
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
        df = default_df
else:
    df = default_df

st.sidebar.markdown("Enter a student's details:")
study_hours = st.sidebar.slider("Daily study hours", 0.0, 12.0, 3.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
previous_score = st.sidebar.slider("Previous score (%)", 0, 100, 65)
sleep_hours = st.sidebar.slider("Sleep per night (hours)", 0.0, 12.0, 7.0, 0.5)
assignments_completed = st.sidebar.slider("Assignments completed (%)", 0, 100, 75)

model, metrics, X_test, y_test, test_preds = train_model(df)
input_df = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "assignments_completed": assignments_completed,
}])
prediction = float(np.clip(model.predict(input_df[FEATURES])[0], 0, 100))
category = "High" if prediction >= 75 else "Moderate" if prediction >= 50 else "Needs support"

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Dataset insights", "🧪 Model evaluation", "📁 Data preview"])

with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Estimated final score")
        st.metric("Predicted score (out of 100)", f"{prediction:.1f}")
        st.metric("Performance band", category)
        st.caption("The estimate is based on patterns learned by a Random Forest regression model.")
        if category == "High":
            st.success("Keep the current study routine and maintain consistent attendance.")
        elif category == "Moderate":
            st.warning("Consider regular revision, completing assignments, and improving attendance.")
        else:
            st.warning("Consider a structured study plan and support from a teacher or mentor.")
    with right:
        st.subheader("Student input summary")
        st.dataframe(input_df.rename(columns={
            "study_hours": "Study hours/day",
            "attendance": "Attendance (%)",
            "previous_score": "Previous score (%)",
            "sleep_hours": "Sleep hours",
            "assignments_completed": "Assignments completed (%)",
        }), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download this prediction as CSV",
            data=pd.concat([input_df, pd.DataFrame([{"predicted_final_score": round(prediction, 2),
                                                     "performance_band": category}])], axis=1).to_csv(index=False).encode("utf-8"),
            file_name="student_prediction.csv",
            mime="text/csv",
        )

with tab2:
    st.subheader("Explore the dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid student records", f"{len(df):,}")
    c2.metric("Average final score", f"{df[TARGET].mean():.1f}/100")
    c3.metric("Average attendance", f"{df['attendance'].mean():.1f}%")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["study_hours"], df[TARGET], alpha=0.55)
    ax1.set_xlabel("Study hours per day")
    ax1.set_ylabel("Final score")
    ax1.set_title("Study hours vs final score")
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(df[TARGET], bins=12, edgecolor="black")
    ax2.set_xlabel("Final score")
    ax2.set_ylabel("Number of students")
    ax2.set_title("Distribution of final scores")
    st.pyplot(fig2)
    plt.close(fig2)

    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig3, ax3 = plt.subplots()
    image = ax3.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax3.set_yticks(range(len(corr.index)), corr.index)
    ax3.set_title("Feature correlation matrix")
    fig3.colorbar(image, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with tab3:
    st.subheader("Model evaluation on a held-out test set")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R² score", f"{metrics['R2']:.3f}")
    st.write("**How to interpret:** MAE/RMSE are average error measures (lower is better). R² compares the model with a simple mean-based baseline; it can be negative.")
    compare_df = pd.DataFrame({"Actual score": y_test.to_numpy(), "Predicted score": test_preds})
    fig4, ax4 = plt.subplots()
    ax4.scatter(compare_df["Actual score"], compare_df["Predicted score"], alpha=0.65)
    ax4.plot([0, 100], [0, 100], linestyle="--")
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.set_xlabel("Actual score")
    ax4.set_ylabel("Predicted score")
    ax4.set_title("Actual vs predicted scores")
    st.pyplot(fig4)
    plt.close(fig4)
    st.caption("Metrics depend on the dataset and random split. Do not present demo metrics as real-world evidence.")

with tab4:
    st.subheader("Dataset preview")
    st.write("Required CSV columns: `study_hours`, `attendance`, `previous_score`, `sleep_hours`, `assignments_completed`, `final_score`.")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download current dataset", df.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance_dataset.csv", mime="text/csv")

st.divider()
st.caption("Built as a learning project. The included dataset is synthetic and created for demonstration; replace it with permissioned, representative data before any real-world use.")
