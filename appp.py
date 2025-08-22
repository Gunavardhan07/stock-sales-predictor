# If needed, install:
# pip install streamlit pandas numpy seaborn matplotlib xgboost scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

st.set_page_config(page_title="Crowdfunding ROI Predictor", layout="wide")

st.title("📊 Crowdfunding ROI Predictor using XGBoost")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Dataset Preview")
    st.write(df.head())

    if "ROI" not in df.columns:
        st.error("⚠️ Your dataset must have a column named 'ROI' (the target).")
    else:
        # -----------------------------
        # Correlation Heatmap
        # -----------------------------
        st.subheader("📌 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # -----------------------------
        # Features & Target
        # -----------------------------
        X = df.drop("ROI", axis=1)
        y = df["ROI"]

        # -----------------------------
        # Train-Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -----------------------------
        # Train XGBoost
        # -----------------------------
        st.subheader("⚙️ Training Model...")
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -----------------------------
        # Metrics
        # -----------------------------
        st.subheader("📈 Model Performance")
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R²:** {r2:.2f}")

        # -----------------------------
        # Graphs
        # -----------------------------
        st.subheader("📊 Graphs")

        # ROI Distribution
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(y, kde=True, color="purple", ax=ax)
        ax.set_title("Distribution of ROI")
        st.pyplot(fig)

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
        ax.set_xlabel("Actual ROI")
        ax.set_ylabel("Predicted ROI")
        ax.set_title("Actual vs Predicted ROI")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔑 Feature Importance")
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x="Importance", y="Feature", data=importance, palette="viridis", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # Download Trained Model (Optional)
        st.subheader("💾 Save Model")
        import joblib
        joblib.dump(model, "xgboost_roi_model.pkl")
        st.success("Model saved as xgboost_roi_model.pkl")
