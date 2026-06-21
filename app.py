import streamlit as st
import joblib
import tempfile
import os
from streamlit_mic_recorder import mic_recorder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "parkinsons_rf_model.pkl")

# NOTE: You MUST ensure that 'src/audio_prepro.py' is available and 
# that the functions 'extract_features' and 'preprocess_audio' are correctly defined in it.
# We only need 'extract_features' here.
# For this example to run, this import must succeed:
try:
    from src.audio_prepro import extract_features
except ImportError:
    st.error("🚨 Error: Could not import 'extract_features' from 'src.audio_prepro'.")
    st.info("Please ensure you have a directory named 'src' with an 'audio_prepro.py' file inside.")
    # Define a mock function to prevent the app from crashing if the module is missing
    def extract_features(file_path):
        st.error("Using Mock Feature Extraction: Please fix the import!")
        # Return dummy features (assuming your model needs 21 features based on the UCI dataset)
        return st.session_state.get('mock_features', [0.5] * 21)


# --- 1. Model Loading (Cached for Efficiency) ---

# Use Streamlit's cache to load the model only once, even across user interactions.
@st.cache_resource
def load_parkinsons_model(model_path, model_mtime):
    """Loads the trained Random Forest model."""
    try:
        model = joblib.load(model_path)
        st.success("✅ ML Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_parkinsons_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))


# --- 2. Prediction Function (Replacing Flask Logic) ---

def predict_on_audio(audio_bytes):
    """
    Saves recorded audio to a temp file, extracts features, predicts, and cleans up.
    This logic directly replaces the core of your Flask /predict endpoint.
    """
    # 1. Save to a Temporary File
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    
    # Initialize variables
    risk = None
    confidence = None
    
    try:
        # 2. Preprocess and Extract Features
        # The reshape(1, -1) is crucial for a single sample prediction
        features = extract_features(tmp_file_path).reshape(1, -1)
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is None and hasattr(model, "named_steps"):
            expected_features = getattr(model.named_steps.get("classifier"), "n_features_in_", None)
        if expected_features is not None and features.shape[1] != expected_features:
            raise ValueError(
                f"Feature mismatch: extractor produced {features.shape[1]} features, "
                f"but the loaded model expects {expected_features}. "
                "Restart Streamlit or retrain the model if this persists."
            )
        
        # 3. Predict using the loaded model
        risk = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
        
        return {'label': int(risk), 'confidence': float(confidence)}
    
    except Exception as e:
        st.error(f"Prediction Error: Could not process audio or predict. Details: {e}")
        return None
    
    finally:
        # 4. Cleanup the temporary file (ensures system stability)
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# --- 3. Streamlit UI and Execution ---

st.set_page_config(page_title="Voice Analysis for Parkinson's", layout="centered")
st.title("🎙️ Voice Analysis for Parkinson's Risk")
st.markdown("Record a short 'ah' sound. The audio will be analyzed by a Random Forest model to assess risk.")

# Use the mic_recorder component
audio_data = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹️ Stop Recording and Predict",
    just_once=True,
    format="wav",
    key='mic_recorder_component'
)

# Only run prediction when the component has returned new data
if audio_data is not None and audio_data['bytes'] is not None:
    st.info("✅ Audio recording complete. Sending to model...")

    with st.spinner('Running AI Model...'):
        # Call the real prediction function
        result = predict_on_audio(audio_data['bytes'])
    
    # --- Display Result ---
    if result:
        st.subheader("📊 Prediction Results")
        label = result['label']
        confidence = result['confidence']
        
        status_text = 'Parkinson’s Risk' if label == 1 else 'Healthy'
        
        if label == 1:
            st.error(f"**🔴 Prediction: {status_text}**")
            st.metric("Confidence", f"{(confidence * 100):.2f}%")
            st.warning("⚠️ **Disclaimer:** This is a screening tool, not a medical diagnosis. Consult a professional.")
        else:
            st.success(f"**🟢 Prediction: {status_text}**")
            st.metric("Confidence", f"{(confidence * 100):.2f}%")
            st.balloons()
            
        # Optional: Playback the recorded audio
        st.markdown("---")
        st.caption("Review your recording:")
        st.audio(audio_data['bytes'], format='audio/wav')
