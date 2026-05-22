import streamlit as st
import pandas as pd
import joblib
# import ollama
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="CardioInsight AI", layout="wide")


def create_pdf(probability, ai_insight):
    import io
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("CardioInsight AI Clinical Report", styles['Title']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Predicted Cardiac Risk: {probability * 100:.1f}%", styles['BodyText']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(ai_insight, styles['BodyText']))

    doc.build(content)
    buffer.seek(0)
    return buffer


#Load model
model = joblib.load("models/heart_model.pkl")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("CardioInsight AI")
st.markdown("### AI-Powered Cardiac Risk Intelligence Dashboard")
st.caption("Model: Logistic Regression (selected by highest recall)")

# ── Sidebar Inputs ───────────────────────────────────────────────────────────
st.sidebar.header("Patient Clinical Inputs")

age       = st.sidebar.slider("Age", 1, 90, 50)
sex       = st.sidebar.selectbox("Sex", [0, 1])
cp        = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
chol      = st.sidebar.slider("Cholesterol", 100, 400, 200)
trestbps  = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
fbs       = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
restecg   = st.sidebar.slider("Rest ECG", 0, 2, 1)
thalach   = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang     = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak   = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope     = st.sidebar.slider("Slope", 0, 2, 1)
ca        = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
thal      = st.sidebar.slider("Thal", 0, 3, 1)

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict Risk"):
    st.session_state.prediction_done = False

    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    explainer   = shap.LinearExplainer(model, masker=shap.maskers.Independent(input_data))
    shap_values = explainer(input_data)

    st.session_state.probability      = probability
    st.session_state.prediction_done  = True

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Heart Disease Risk Probability", value=f"{probability * 100:.1f}%")
        if prediction == 1:
            st.error("High Heart Disease Risk Detected")
        else:
            st.success("Low Cardiac Risk")

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Cardiac Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40],  'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100],'color': "salmon"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── AI Insight ───────────────────────────────────────────────────────────
    prompt = f"""
Patient Information:

Age: {age}
Cholesterol: {chol}
Blood Pressure: {trestbps}
Maximum Heart Rate: {thalach}

Predicted Cardiac Risk: {probability * 100:.1f}%

Provide:
1. Clinical interpretation
2. Key contributing factors
3. General lifestyle recommendations

Keep response concise.
"""
    # response = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': prompt}])

    st.subheader("AI Clinical Insight")
    # ai_insight = response['message']['content']
    insights = []

    if chol > 240:
        insights.append("High cholesterol contributes to increased cardiac risk.")

    if exang == 1:
        insights.append("Exercise-induced angina may indicate reduced cardiac function.")

    if trestbps > 140:
        insights.append("Elevated blood pressure is associated with cardiovascular strain.")

    if probability > 0.7:
        insights.append("Overall prediction indicates elevated cardiac disease risk.")

    ai_insight = " ".join(insights) if insights else "No major risk indicators detected based on current inputs."
    st.write(ai_insight)
    st.session_state.ai_insight = ai_insight

    st.divider()

    # ── SHAP Explainability ──────────────────────────────────────────────────
    st.subheader("Prediction Explainability")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    st.divider()

# ── PDF Report ───────────────────────────────────────────────────────────────
if st.session_state.get("prediction_done"):
    st.subheader("Clinical Report")
    pdf_buffer = create_pdf(
        st.session_state.probability,
        st.session_state.ai_insight
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="cardio_report.pdf",
        mime="application/pdf"
    )

    st.divider()

# ── Disclaimer ───────────────────────────────────────────────────────────────
st.warning(
    "**Disclaimer:** This dashboard is an AI-powered screening tool for educational "
    "purposes only. It does not replace professional medical advice, diagnosis, or treatment."
)
