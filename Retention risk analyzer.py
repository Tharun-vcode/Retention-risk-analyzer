import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Retention Risk Analyzer", layout="wide")
st.title("📊 Retention Risk Analyzer")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        encoder = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = encoder.fit_transform(df[col])

        X = df.drop(columns=["Churn", "customerID"], errors="ignore")
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        tab1, tab2, tab3 = st.tabs(["📁 Data & EDA", "📈 Model Performance", "🔮 Prediction"])

        with tab1:
            st.subheader("Data Overview")
            st.dataframe(df.head(), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            st.markdown("**Churn Distribution**")
            churn_counts = df["Churn"].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=["#4CAF50", "#F44336"])
            st.pyplot(fig_pie)

            st.markdown("**Numeric Summary**")
            st.dataframe(df.describe(), use_container_width=True)

        with tab2:
            st.subheader("Model Accuracy")
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", ax=ax_cm)
            st.pyplot(fig_cm)

            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
            ax_roc.plot([0, 1], [0, 1], "r--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

        with tab3:
            st.subheader("Enter New Customer Details")
            input_data = {}
            for col in X.columns:
                if df[col].dtype == "int64" and len(df[col].unique()) <= 10:
                    input_data[col] = st.selectbox(col, sorted(df[col].unique()))
                elif df[col].dtype in ["float64", "int64"]:
                    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    input_data[col] = st.text_input(col)

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df)[0][prediction] * 100

            if prediction == 1:
                st.error(f"🚨 High Risk of Churn — Confidence: {confidence:.1f}%")
                st.markdown("**Business Insight:** This customer shows a high probability of leaving. Consider offering loyalty rewards, personalized discounts, or targeted engagement campaigns to retain them.")
            else:
                st.success(f"✅ Low Risk of Churn — Confidence: {confidence:.1f}%")
                st.markdown("**Business Insight:** This customer is likely satisfied with the service. Maintain engagement and consider upselling additional features or services to increase lifetime value.")

    except Exception as e:
        st.error(f"Error: {e}")
