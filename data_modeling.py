import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDRegressor, SGDClassifier, Perceptron, PassiveAggressiveClassifier,
    RidgeClassifier, LogisticRegressionCV, TheilSenRegressor, HuberRegressor, RANSACRegressor
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score, max_error, median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils.validation import check_is_fitted

def modeling_page():
    st.title("Data Modeling")

    if 'df' in st.session_state:
        df = st.session_state.df

        uploaded_model = st.file_uploader("Upload a trained model", type=["pkl"])

        if uploaded_model:
            try:
                model = joblib.load(uploaded_model)
                st.write("Model loaded successfully!")

                model_type = "classification" if hasattr(model, "predict_proba") else "regression"

                target_column = st.selectbox("Select Target Column:", df.columns)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                st.subheader("Model Evaluation")
                y_pred = model.predict(X_test)

                if model_type == "classification":
                    st.subheader("Classification Metrics")
                    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}" )
                    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}" )
                    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}" )
                    st.write(f"Matthews Correlation Coefficient (MCC): {matthews_corrcoef(y_test, y_pred)}")
                    st.write(f"Cohen's Kappa Score: {cohen_kappa_score(y_test, y_pred)}")

                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(X_test)
                            if proba.shape[1] == 2:
                                roc_auc = roc_auc_score(y_test, proba[:, 1])
                            else:
                                roc_auc = roc_auc_score(y_test, proba, multi_class='ovr')
                            st.write(f"ROC-AUC Score: {roc_auc}")
                            st.write(f"Log Loss: {log_loss(y_test, proba)}")
                        except Exception as e:
                            st.warning(f"ROC-AUC/Log Loss calculation failed: {e}")

                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
                    st.pyplot(fig)

                else:
                    st.subheader("Regression Metrics")
                    st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
                    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
                    st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred))}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, y_pred)}")
                    st.write(f"R-Squared (R²): {r2_score(y_test, y_pred)}")
                    st.write(f"Explained Variance Score: {explained_variance_score(y_test, y_pred)}")
                    st.write(f"Max Error: {max_error(y_test, y_pred)}")
                    st.write(f"Median Absolute Error: {median_absolute_error(y_test, y_pred)}")

            except Exception as e:
                st.error(f"Error while loading the model: {str(e)}")

        else:
            task_type = st.radio("Select Modeling Task:", ["Regression", "Classification"])
            target_column = st.selectbox("Select Target Column:", df.columns)

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_options = {
                "Regression": [
                    "Random Forest", "Gradient Boosting", "XGBoost", "Linear Regression",
                    "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                    "SGD Regression", "Theil-Sen Estimator", "Huber Regression", "RANSAC Regression"
                ],
                "Classification": [
                    "Logistic Regression", "Logistic Regression CV", "Random Forest",
                    "Gradient Boosting", "XGBoost", "Bagging", "SGD Classifier",
                    "Perceptron", "Passive Aggressive Classifier", "Ridge Classifier"
                ]
            }
            model_type = st.selectbox("Select Algorithm:", model_options[task_type])

            model_classes = {
                "Random Forest": (RandomForestClassifier, RandomForestRegressor),
                "Gradient Boosting": (GradientBoostingClassifier, GradientBoostingRegressor),
                "XGBoost": (xgb.XGBClassifier, xgb.XGBRegressor),
                "Logistic Regression": (LogisticRegression, None),
                "Bagging": (BaggingClassifier, None),
                "Linear Regression": (None, LinearRegression),
                "Ridge Regression": (None, Ridge),
                "Lasso Regression": (None, Lasso),
                "ElasticNet Regression": (None, ElasticNet),
                "SGD Regression": (None, SGDRegressor),
                "Theil-Sen Estimator": (None, TheilSenRegressor),
                "Huber Regression": (None, HuberRegressor),
                "RANSAC Regression": (None, RANSACRegressor),
                "SGD Classifier": (SGDClassifier, None),
                "Perceptron": (Perceptron, None),
                "Passive Aggressive Classifier": (PassiveAggressiveClassifier, None),
                "Ridge Classifier": (RidgeClassifier, None),
                "Logistic Regression CV": (LogisticRegressionCV, None),
            }

            params_grid = {
                "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
                "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
                "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
                "Logistic Regression": {'C': [0.1, 1], 'solver': ['liblinear', 'lbfgs']},
                "Bagging": {'n_estimators': [10, 50], 'max_samples': [0.7, 1.0]},
                "Linear Regression": {'fit_intercept': [True, False]},
                "Ridge Regression": {'alpha': [0.1, 1.0], 'solver': ['auto', 'svd']},
                "Lasso Regression": {'alpha': [0.1, 1.0], 'selection': ['cyclic', 'random']},
                "ElasticNet Regression": {'alpha': [0.1, 1.0], 'l1_ratio': [0.5, 0.8]},
                "SGD Regression": {'loss': ['squared_error', 'huber'], 'alpha': [0.0001, 0.001]},
                "Theil-Sen Estimator": {'n_subsamples': [50, 100]},
                "Huber Regression": {'epsilon': [1.35, 1.5], 'alpha': [0.001, 0.01]},
                "RANSAC Regression": {'min_samples': [0.5, None]},
                "SGD Classifier": {'loss': ['hinge', 'log_loss'], 'alpha': [0.0001, 0.001]},
                "Perceptron": {'penalty': ['l2', 'l1'], 'alpha': [0.0001, 0.001]},
                "Passive Aggressive Classifier": {'C': [0.1, 1.0], 'loss': ['hinge', 'squared_hinge']},
                "Ridge Classifier": {'alpha': [0.1, 1.0], 'solver': ['auto', 'svd']},
                "Logistic Regression CV": {'Cs': [10, 100], 'solver': ['liblinear', 'lbfgs']}
            }

            train_button = st.button("Train Model")

            if train_button:
                # Determine model class based on task
                model_class_tuple = model_classes[model_type]
                model_class = model_class_tuple[0] if task_type == "Classification" else model_class_tuple[1]
                model = model_class()

                # Get parameter grid
                param_grid = params_grid[model_type]

                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if task_type == "Classification" else 'r2')
                try:
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_

                    try:
                        check_is_fitted(best_model)
                    except Exception as e:
                        st.error(f"Model not fitted: {e}")

                    y_pred = best_model.predict(X_test)

                    if task_type == "Regression":
                        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
                        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
                        st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred))}")
                        st.write(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, y_pred)}")
                        st.write(f"R-Squared (R²): {r2_score(y_test, y_pred)}")
                        st.write(f"Explained Variance Score: {explained_variance_score(y_test, y_pred)}")
                        st.write(f"Max Error: {max_error(y_test, y_pred)}")
                        st.write(f"Median Absolute Error: {median_absolute_error(y_test, y_pred)}")

                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
                        ax.set_xlabel('True Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title('True vs Predicted Values')
                        st.pyplot(fig)

                    elif task_type == "Classification":
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}" )
                        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}" )
                        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}" )
                        st.write(f"Matthews Correlation Coefficient (MCC): {matthews_corrcoef(y_test, y_pred)}")
                        st.write(f"Cohen's Kappa Score: {cohen_kappa_score(y_test, y_pred)}")

                        if hasattr(best_model, 'predict_proba'):
                            try:
                                proba = best_model.predict_proba(X_test)
                                if proba.shape[1] == 2:
                                    roc_auc = roc_auc_score(y_test, proba[:, 1])
                                else:
                                    roc_auc = roc_auc_score(y_test, proba, multi_class='ovr')
                                st.write(f"ROC-AUC Score: {roc_auc}")
                                st.write(f"Log Loss: {log_loss(y_test, proba)}")
                            except Exception as e:
                                st.warning(f"ROC-AUC/Log Loss calculation failed: {e}")

                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
                        st.pyplot(fig)

                    model_filename = "trained_model.pkl"
                    joblib.dump(best_model, model_filename)

                    with open(model_filename, "rb") as file:
                        st.download_button(
                            label="Download Trained Model",
                            data=file,
                            file_name=model_filename,
                            mime="application/octet-stream"
                        )

                except Exception as e:
                    st.error(f"Error while fitting the model: {str(e)}")

    else:
        st.warning("Please upload a dataset in the Data page first.")
