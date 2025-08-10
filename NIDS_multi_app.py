import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === Load model and dependencies ===
model = joblib.load("best_attack_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")  # List of 40 feature names

# === Page Configuration ===
st.set_page_config(
    page_title="🛡️ Network Attack Classifier",
    layout="wide",
    page_icon="🛡️"
)

# === Title Section ===
st.markdown("""
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1 style='color:#003366;'>🛡️ Network Attack Type Classifier - BY Rushikesh Gurav</h1>
        <p style='font-size:18px;'>Predict the type of network attack using UNSW-NB15 features</p>
    </div>
""", unsafe_allow_html=True)

# === Sidebar Instructions ===
with st.sidebar:
    st.header("📌 Instructions")
    st.markdown("""
    - Choose input method: Manual or CSV
    - For CSV: Upload file with **1 row**, **40 values**, **no header**
    - For Manual: Enter all 40 features
    - Press **Predict** to get results
    """)

# === Input Method Selection ===
st.subheader("🧠 Select Input Method")
input_method = st.radio("Choose how to enter feature values:", ["Manual Input", "Upload CSV"])

input_array = None

# === Manual Input ===
if input_method == "Manual Input":
    with st.form("manual_form"):
        st.subheader("📥 Enter Feature Values Manually")

        cols = st.columns(2)
        user_input = []

        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                val = st.number_input(
                    label=f"{feature}",
                    value=0.0,
                    format="%.4f",
                    step=1.0,
                    key=feature
                )
                user_input.append(val)

        submitted_manual = st.form_submit_button("🔍 Predict Attack Type")
        if submitted_manual:
            input_array = np.array(user_input).reshape(1, -1)

# === CSV Upload Input ===
elif input_method == "Upload CSV":
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file with 1 row and 40 values (no header)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file, header=None)
            if df_input.shape != (1, len(selected_features)):
                st.error(f"⚠️ File must have exactly 1 row and {len(selected_features)} columns. Found: {df_input.shape}")
            else:
                input_array = df_input.to_numpy()
                st.success("✅ File uploaded and validated successfully.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    if input_array is not None:
        if st.button("🔍 Predict Attack Type"):
            st.info("Running prediction...")

# === Prediction & Output ===
if input_array is not None:
    try:
        prediction = model.predict(input_array)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display Prediction
        st.markdown("---")
        st.markdown(f"""
            <div style="padding: 20px; background-color: #f0f8ff; border-left: 6px solid #007acc;">
                <h3>🚨 Predicted Attack Type:</h3>
                <h1 style="color:#ff5733;">{predicted_label}</h1>
            </div>
        """, unsafe_allow_html=True)

        # Top probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_array)[0]
            top_indices = np.argsort(probs)[::-1][:3]
            st.markdown("### 🔢 Top Class Probabilities:")
            for idx in top_indices:
                label = label_encoder.inverse_transform([idx])[0]
                st.write(f"- **{label}**: `{probs[idx]*100:.2f}%`")

        # Expand Raw
        st.markdown("---")
        with st.expander("📊 Show Raw Inputs & Output"):
            st.write("📥 Input Sample (first 5):", input_array[0][:5])
            st.write("🔬 Raw Prediction:", prediction)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
