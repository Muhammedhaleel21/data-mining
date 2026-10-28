
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Early Warning System",
    page_icon="🎓",
    layout="wide"
)

# Load the saved model
saved = joblib.load("student_model.pkl")
model = saved["model"]
features = saved["features"]

st.title("🎓 Student Performance & Early Warning System")
st.write(
    "Upload student records to predict final grades "
    "and identify students who may need additional support."
)

st.info(
    "The model predicts G3 using earlier grades and other "
    "student information. Predictions are estimates, not guarantees."
)

# Show model evaluation
with st.expander("View model evaluation"):
    st.metric("Test MAE", f'{saved["mae"]:.2f} grade points')
    st.metric("Test R²", f'{saved["r2"]:.3f}')
    st.caption(
        "These metrics describe the held-out test set, "
        "not guaranteed performance on new students."
    )

st.subheader("1. Upload Student Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing student records",
    type=["csv"]
)

# Optional download of a blank input template
template = pd.DataFrame(columns=features)

st.download_button(
    "Download CSV Template",
    data=template.to_csv(index=False),
    file_name="student_input_template.csv",
    mime="text/csv"
)

if uploaded_file is not None:
    try:
        # Detect comma-separated or semicolon-separated CSV
        data = pd.read_csv(uploaded_file, sep=None, engine="python")
        data.columns = data.columns.str.strip()

        # Required model input columns
        missing = [col for col in features if col not in data.columns]

        if missing:
            st.error(
                "The uploaded file is missing required columns: "
                + ", ".join(missing)
            )
            st.stop()

        if data.empty:
            st.error("The uploaded CSV contains no student records.")
            st.stop()

        # Preserve optional student identifiers for the report
        if "Student_ID" in data.columns:
            student_ids = data["Student_ID"].astype(str).tolist()
        elif "student_id" in data.columns:
            student_ids = data["student_id"].astype(str).tolist()
        else:
            student_ids = [
                f"Student {i + 1:03d}"
                for i in range(len(data))
            ]

        if "Student_Name" in data.columns:
            student_names = data["Student_Name"].astype(str).tolist()
        elif "student_name" in data.columns:
            student_names = data["student_name"].astype(str).tolist()
        else:
            student_names = student_ids.copy()

        # Prepare only the features used during training.
        # G3, if supplied, is never used as a prediction input.
        X = data[features].copy()

        numeric = saved["numeric"]
        categorical = saved["categorical"]

        for col in numeric:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        for col in categorical:
            X[col] = X[col].replace(r"^\s*$", pd.NA, regex=True)

        # Check that each row contains usable input data
        usable = X.notna().any(axis=1)

        if not usable.all():
            st.warning(
                f"{int((~usable).sum())} row(s) contain no usable "
                "feature values and will be excluded."
            )

        X_valid = X.loc[usable]

        if X_valid.empty:
            st.error("No usable student records were found.")
            st.stop()

        # Predict all valid students together
        predicted_grades = model.predict(X_valid)
        predicted_grades = [
            max(0.0, min(20.0, float(grade)))
            for grade in predicted_grades
        ]

        results = pd.DataFrame({
            "Student_ID": [
                student_ids[i] for i in X_valid.index
            ],
            "Student_Name": [
                student_names[i] for i in X_valid.index
            ],
            "Predicted_G3": [
                round(grade, 2) for grade in predicted_grades
            ]
        })

        # Apply the project's proposed support thresholds
        def assign_label(grade):
            if grade < 10:
                return "High Priority"
            elif grade < 14:
                return "Needs Attention"
            return "On Track"

        results["Support_Label"] = results["Predicted_G3"].apply(
            assign_label
        )

        def assign_action(label):
            if label == "High Priority":
                return "Review promptly and discuss support"
            elif label == "Needs Attention":
                return "Monitor progress and offer support"
            return "Continue regular monitoring"

        results["Suggested_Action"] = results[
            "Support_Label"
        ].apply(assign_action)

        st.subheader("2. Class Overview")

        total = len(results)
        high = (results["Support_Label"] == "High Priority").sum()
        attention = (
            results["Support_Label"] == "Needs Attention"
        ).sum()
        on_track = (results["Support_Label"] == "On Track").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students Processed", total)
        c2.metric("High Priority", int(high))
        c3.metric("Needs Attention", int(attention))
        c4.metric("On Track", int(on_track))

        st.caption(
            "Support labels are based on predicted grades and "
            "proposed thresholds. They are not definitive judgments."
        )

        st.subheader("3. Student Prediction Report")

        selected = st.selectbox(
            "Filter by support label",
            [
                "All Students",
                "High Priority",
                "Needs Attention",
                "On Track"
            ]
        )

        if selected == "All Students":
            displayed = results
        else:
            displayed = results[
                results["Support_Label"] == selected
            ]

        st.dataframe(
            displayed,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("4. Download Results")

        st.download_button(
            "Download Complete Prediction Report",
            data=results.to_csv(index=False),
            file_name="student_performance_report.csv",
            mime="text/csv"
        )

        st.subheader("5. Grade Prediction Distribution")

        st.bar_chart(
            results["Support_Label"].value_counts()
            .reindex(
                ["High Priority", "Needs Attention", "On Track"],
                fill_value=0
            )
        )

    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as e:
        st.error(f"Could not process the uploaded CSV: {e}")
