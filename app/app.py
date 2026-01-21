import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Intelligence Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        artifacts = joblib.load("models/churn_stacking_tuned.pkl")
        return artifacts
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please run the training script to generate 'models/churn_stacking_tuned.pkl'.")
        return None

artifacts = load_artifacts()

if artifacts:
    pipeline = artifacts["pipeline"]
    feature_names = artifacts["feature_names"]
    optimal_threshold = artifacts.get("threshold", 0.5) 
    
    preprocessor = pipeline.named_steps["preprocessing"]
    stacking_model = pipeline.named_steps["stacking"]
    rf_explainer_model = stacking_model.estimators_[0] 

st.sidebar.header("👤 Customer Profile")
st.sidebar.markdown("---")

def user_input_features():
    with st.sidebar.expander("Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with st.sidebar.expander("Services Config", expanded=False):
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with st.sidebar.expander("Account Details", expanded=True):
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total = st.number_input("Total Charges ($)", 18.0, 9000.0, 1000.0)

    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech_support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    return pd.DataFrame([data])

if artifacts:
    input_df = user_input_features()

    df_processed = input_df.copy()
    
    tenure_safe = df_processed["tenure"].replace(0, 1)
    
    df_processed["AvgMonthlySpend"] = df_processed["TotalCharges"] / tenure_safe
    df_processed["TenureYears"] = df_processed["tenure"] / 12
    
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df_processed["ServiceCount"] = df_processed[service_cols].apply(lambda x: sum(x == "Yes"), axis=1)
    df_processed["MonthlyPerService"] = df_processed["MonthlyCharges"] / (df_processed["ServiceCount"] + 1)
    
    df_processed["IsSenior"] = df_processed["SeniorCitizen"] 
    df_processed = df_processed.drop("SeniorCitizen", axis=1)

    st.title("📡 Churn Intelligence Hub")
    
    X_transformed = preprocessor.transform(df_processed)

    prediction_prob = pipeline.predict_proba(df_processed)[0][1]
    
    
    t_high = optimal_threshold
    t_moderate = optimal_threshold * 0.75 
    
    if prediction_prob >= t_high:
        status = "HIGH RISK"
        color = "#FF4B4B"
        bg_color = "rgba(255, 75, 75, 0.1)"
    elif prediction_prob >= t_moderate:
        status = "MODERATE RISK"
        color = "#FFA500"
        bg_color = "rgba(255, 165, 0, 0.1)"
    else:
        status = "SAFE"
        color = "#28a745"
        bg_color = "rgba(40, 167, 69, 0.1)"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background-color:{bg_color}; padding: 10px; border-radius: 5px; text-align: center;">
            <h3 style="color:{color}; margin:0;">{status}</h3>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.metric("Churn Probability", f"{prediction_prob:.1%}")
    with col3:
        st.metric("Critical Threshold", f"{optimal_threshold:.3f}", help="Probabilities above this trigger High Risk")

    st.markdown("---")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Risk Gauge")
        
        gauge_max = 1.0 
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_prob,
            number = {'valueformat': '.1%'},
            title = {'text': "Churn Probability"},
            gauge = {
                'axis': {'range': [0, gauge_max], 'tickformat': '.0%'},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, t_moderate], 'color': "rgba(40, 167, 69, 0.2)"},
                    {'range': [t_moderate, t_high], 'color': "rgba(255, 165, 0, 0.3)"},
                    {'range': [t_high, gauge_max], 'color': "rgba(255, 75, 75, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': optimal_threshold
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.subheader("🔍 What is driving this prediction?")
        
        explainer = shap.TreeExplainer(rf_explainer_model)
        shap_values = explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values
            
        if hasattr(vals, 'ndim') and vals.ndim == 3:
             vals = vals[0, :, 1]
        elif hasattr(vals, 'ndim') and vals.ndim == 2:
             vals = vals[0]
             
        vals = np.array(vals).flatten()

        feature_names_out = [
             "tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", 
             "ServiceCount", "MonthlyPerService", "TenureYears", "IsSenior"
        ] + list(preprocessor.named_transformers_["cat"].get_feature_names_out())
        
        shap_df = pd.DataFrame({
            'Feature': feature_names_out,
            'SHAP Value': vals
        })
        
        shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(8)
        
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=shap_df['Feature'],
            x=shap_df['SHAP Value'],
            orientation='h',
            marker=dict(color=shap_df['SHAP Value'].apply(lambda x: '#ef553b' if x > 0 else '#00cc96')),
            text=shap_df['SHAP Value'].apply(lambda x: f"{x:.2f}"),
            textposition='auto'
        ))
        
        fig_shap.update_layout(
            title="Top Factors Influencing this Decision",
            xaxis_title="Impact (Right=Churn, Left=Stay)",
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    if status == "HIGH RISK":
        st.error(f"🚨 **Action Required:** Customer probability ({prediction_prob:.1%}) is above the critical cutoff of {optimal_threshold:.1%}.")
    elif status == "MODERATE RISK":
        st.warning(f"⚠️ **Watchlist:** Customer is approaching the cutoff. Current: {prediction_prob:.1%}. (Buffer starts at {t_moderate:.1%})")
    else:
        st.success(f"✅ **Healthy:** Customer is well below risk levels.")
    st.markdown("### 📋 AI-Driven Retention Strategy")
    with st.expander("View Action Plan", expanded=True):
        recommendations = []
        
        if input_df["Contract"].iloc[0] == "Month-to-month":
            if status == "HIGH RISK":
                recommendations.append("🔴 **Urgent Contract Upgrade:** User is High Risk & Monthly. **Offer:** 20% discount for 1-year commitment.")
            elif status == "MODERATE RISK":
                recommendations.append("🟡 **Contract Nudge:** User is Monthly. **Action:** Send 'Benefits of Annual Plan' email.")

        if input_df["InternetService"].iloc[0] == "Fiber optic":
             if prediction_prob > 0.4:
                recommendations.append("⚡ **Fiber Quality Check:** High probability user on Fiber. Check for recent service outages.")

        if input_df["PaymentMethod"].iloc[0] == "Electronic check":
             if status == "HIGH RISK":
                 recommendations.append("💳 **Payment Friction:** High Risk. **Action:** Call to setup Auto-Pay with $10 credit.")
             else:
                 recommendations.append("💳 **Payment Friction:** Suggest Auto-Pay in next billing email.")

        if input_df["TechSupport"].iloc[0] == "No" and input_df["InternetService"].iloc[0] != "No":
             recommendations.append("🛠️ **Value Add:** Offer 3 months free Tech Support to increase product stickiness.")

        if not recommendations:
            st.info("👍 No specific retention actions triggered. Customer looks healthy.")
        else:
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}")

else:
    st.warning("Please train the model first!")