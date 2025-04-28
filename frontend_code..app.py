import os

# Check if it's not running on Streamlit Cloud
if os.getenv('STREMLIT_ENV') != "cloud":
    import tenseal as ts
    # Your encryption code goes here
else:
    # Skip encryption or handle differently in the cloud
    print("Encryption skipped (Cloud environment)")

# Rest of your code follows



import streamlit as st
import numpy as np
import pickle
import tenseal as ts

# Set page config
st.set_page_config(
    page_title="Medical Diagnosis with Homomorphic Encryption",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        color: #2e86de;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #f8f9fa;
        border-left: 5px solid #2e86de;
    }
    .feature-input {
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #2e86de;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        margin-top: 20px;
    }
    .model-selector {
        margin-bottom: 20px;
    }
    .model-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .metric-box {
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #e8f4fd;
    }
</style>
""", unsafe_allow_html=True)

# Load models and components
@st.cache_resource
def load_models():
    try:
        # Load XGBoost model
        with open('xgb_encrypted_model.pkl', 'rb') as f:
            xgb_components = pickle.load(f)
        
        # Load Random Forest model
        with open('rf_encrypted_model.pkl', 'rb') as f:
            rf_components = pickle.load(f)
        
        # Deserialize TenSEAL contexts
        xgb_components['private_ctx'] = ts.context_from(xgb_components['private_ctx'])
        xgb_components['public_ctx'] = ts.context_from(xgb_components['public_ctx'])
        
        rf_components['private_ctx'] = ts.context_from(rf_components['private_ctx'])
        rf_components['public_ctx'] = ts.context_from(rf_components['public_ctx'])
        
        return {
            'xgb': xgb_components,
            'rf': rf_components
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

model_components = load_models()

def make_prediction(sample, model_type):
    """Generic prediction function that works for both models"""
    components = model_components[model_type]
    
    try:
        # Preprocess
        sample = components['imputer'].transform(sample)
        sample = components['scaler'].transform(sample)
        
        # Encrypt
        encrypted_sample = ts.ckks_vector(components['public_ctx'], sample.tolist()[0])
        enc_feature_importances = ts.ckks_vector(components['public_ctx'], 
                                              components['feature_importances'].tolist())
        enc_threshold = ts.ckks_vector(components['public_ctx'], 
                                    [components['threshold']])
        
        # Homomorphic prediction
        weighted_sum = encrypted_sample.dot(enc_feature_importances)
        encrypted_result = weighted_sum - enc_threshold
        
        # Decrypt
        decrypted_result = encrypted_result.decrypt(components['private_ctx'].secret_key())
        y_pred = 1 if decrypted_result[0] > 0 else 0
        prediction_label = components['label_encoder'].inverse_transform([y_pred])[0]
        
        return {
            'prediction': y_pred,
            'label': prediction_label,
            'confidence': abs(decrypted_result[0]),
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

# Main app
def main():
    st.markdown('<h1 class="title">🔒 Secure Medical Diagnosis</h1>', unsafe_allow_html=True)
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses Homomorphic Encryption to securely predict medical conditions 
        without exposing sensitive patient data.
        
        *How it works:*
        1. Enter patient features
        2. Choose prediction model
        3. Data gets encrypted
        4. Prediction happens in encrypted form
        5. Only final result is decrypted
        """)
        
        st.header("Model Comparison")
        
        if model_components:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.subheader("XGBoost")
            st.write("- Gradient boosting algorithm")
            st.write("- Typically higher accuracy")
            st.write("- More complex to train")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.subheader("Random Forest")
            st.write("- Ensemble of decision trees")
            st.write("- More interpretable")
            st.write("- Robust to outliers")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Models failed to load. Please check the model files.")

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Patient Data Input")
        
        if not model_components:
            st.error("Cannot load input form without models. Please ensure both model files exist.")
            return
        
        # Dynamic feature input based on the model
        num_features = len(model_components['xgb']['feature_importances'])
        feature_values = []
        
        # Create 2 columns for feature inputs
        cols = st.columns(2)
        for i in range(num_features):
            with cols[i % 2]:
                val = st.number_input(
                    f"Feature {i}",
                    key=f"feature_{i}",
                    help=f"Enter value for feature {i}",
                    step=0.01,
                    format="%.2f"
                )
                feature_values.append(val)
    
    with col2:
        st.header("Model Selection & Prediction")
        
        if not model_components:
            return
        
        # Model selection
        model_type = st.radio(
            "Select Prediction Model:",
            options=["XGBoost", "Random Forest"],
            index=0,
            key="model_selector",
            horizontal=True
        )
        model_key = 'xgb' if model_type == "XGBoost" else 'rf'
        
        if st.button(f"🔒 Make Secure Prediction ({model_type})"):
            with st.spinner(f"Making {model_type} prediction securely..."):
                try:
                    # Prepare input data
                    sample = np.array(feature_values).reshape(1, -1)
                    
                    # Make prediction
                    result = make_prediction(sample, model_key)
                    
                    if not result['success']:
                        st.error(f"Prediction failed: {result['error']}")
                        return
                    
                    # Display results
                    with st.container():
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.subheader("Prediction Result")
                        
                        if result['prediction'] == 1:
                            st.error(f"🔴 High Risk: {result['label']}")
                        else:
                            st.success(f"🟢 Low Risk: {result['label']}")
                        
                        st.write(f"*Model Used:* {model_type}")
                        st.write("*Confidence Score:*", f"{result['confidence']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Explanation
                        st.info("""
                        *Privacy Note:* Your data was encrypted before processing. 
                        The actual values were never exposed to the prediction system.
                        """)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")


