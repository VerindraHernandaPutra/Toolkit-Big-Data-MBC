import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def competition():
    st.title("🏆 Kaggle-style Competition")
    
    # Session state initialization
    if 'submission' not in st.session_state:
        st.session_state.submission = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    
    # Upload section
    with st.expander("📤 Upload Files", expanded=True):
        col1, col2, col3 = st.columns(3)
        model_file = col1.file_uploader("Trained Model (.pkl)", type=["pkl"])
        test_file = col2.file_uploader("Test Data (test.csv)", type=["csv"])
        sample_file = col3.file_uploader("Sample Submission", type=["csv"])
    
    # Load data
    if model_file and test_file and sample_file:
        try:
            model = joblib.load(model_file)
            test_df = pd.read_csv(test_file).drop(columns=['Unnamed: 0'], errors='ignore')
            sample_sub = pd.read_csv(sample_file).drop(columns=['Unnamed: 0'], errors='ignore')
            
            # Store in session state
            st.session_state.model = model
            st.session_state.test_data = test_df
            st.session_state.sample_sub = sample_sub
            
            # Auto-detect target column
            target_options = [col for col in sample_sub.columns if col != 'ID']
            if target_options:
                st.session_state.target_column = target_options[0]
            
            st.success("✅ All files loaded successfully!")
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
    
    # Data preview section
    if 'test_data' in st.session_state and 'sample_sub' in st.session_state:
        with st.expander("🔍 Data Preview", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Test Data")
                st.write(f"Shape: {st.session_state.test_data.shape}")
                st.dataframe(st.session_state.test_data.head(), use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Sample Submission")
                st.write(f"Shape: {st.session_state.sample_sub.shape}")
                st.dataframe(st.session_state.sample_sub.head(), use_container_width=True, hide_index=True)
    
    # Target column selection
    if 'sample_sub' in st.session_state and st.session_state.sample_sub is not None:
        target_options = [col for col in st.session_state.sample_sub.columns if col != 'ID']
        if target_options:
            new_target = st.selectbox(
                "Select Target Column Name:",
                options=target_options,
                index=0,
                key='target_select'
            )
            st.session_state.target_column = new_target
    
    # Prediction section
    if 'model' in st.session_state and 'test_data' in st.session_state and st.session_state.target_column:
        st.markdown("---")
        with st.expander("🔮 Make Predictions", expanded=True):
            if st.button("🚀 Generate Predictions"):
                try:
                    # Preprocess test data
                    test_df = st.session_state.test_data.copy().drop(columns=['Unnamed: 0'], errors='ignore')
                    X_test = test_df.drop('ID', axis=1)
                    
                    # Handle categorical features
                    categorical_cols = X_test.select_dtypes(include=['object']).columns
                    if not categorical_cols.empty:
                        encoder = OneHotEncoder(handle_unknown='ignore')
                        X_test_encoded = encoder.fit_transform(X_test[categorical_cols])
                        X_test = X_test.drop(categorical_cols, axis=1)
                        X_test = pd.concat([X_test, pd.DataFrame(X_test_encoded.toarray())], axis=1)
                    
                    # Align features with model
                    if hasattr(st.session_state.model, 'feature_names_in_'):
                        missing_features = set(st.session_state.model.feature_names_in_) - set(X_test.columns)
                        for feature in missing_features:
                            X_test[feature] = 0
                        X_test = X_test[st.session_state.model.feature_names_in_]
                    
                    # Make predictions
                    predictions = st.session_state.model.predict(X_test)
                    
                    # Create submission
                    submission = st.session_state.sample_sub.copy().drop(columns=['Unnamed: 0'], errors='ignore')
                    submission[st.session_state.target_column] = predictions.astype(int)
                    st.session_state.submission = submission
                    
                    st.success(f"✅ Successfully predicted {len(predictions)} samples!")

                    edited_sub = st.data_editor(st.session_state.submission, use_container_width=True, hide_index=True)
            
                    csv = edited_sub.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Submission CSV",
                        data=csv,
                        file_name="submission.csv",
                        mime="text/csv",
                        help="Download your predictions in Kaggle submission format"
                    )
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
