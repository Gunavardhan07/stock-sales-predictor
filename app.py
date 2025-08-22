import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

st.set_page_config(page_title="STOCK MARKET sales Predictor", layout="wide")
st.title("STOCK MARKET sales Predictor using XGBoost")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Dataset Preview")
    st.write(df.head())

    if "ROI" not in df.columns:
        st.error("Your dataset must have a column named 'ROI' (the target).")
    else:
        st.subheader("📌 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.corr()
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        ticks = np.arange(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

        X = df.drop("ROI", axis=1)
        y = df["ROI"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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

        st.subheader("📈 Model Performance")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")

        st.subheader("📊 Graphs")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(y, bins=30, color="purple", alpha=0.7)
        ax.set_title("Distribution of ROI")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Actual ROI")
        ax.set_ylabel("Predicted ROI")
        ax.set_title("Actual vs Predicted ROI")
        st.pyplot(fig)

        st.subheader("🔑 Feature Importance")
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Importance'], color="green")
        ax.invert_yaxis()
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        st.subheader("💾 Save Model")
        joblib.dump(model, "xgboost_roi_model.pkl")
        st.success("Model saved as xgboost_roi_model.pkl")
