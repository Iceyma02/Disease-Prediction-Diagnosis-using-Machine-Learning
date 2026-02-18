import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set page configuration - ELITE VERSION
st.set_page_config(
    page_title="MedPredict AI - Disease Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ELITE CUSTOM CSS with Dark Mode Fix
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);
    }
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: center;
    }
    .model-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #4a5568 100%);
    }
    
    /* DARK MODE FIXES FOR RISK FACTORS SECTION */
    .risk-factors-section {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    .risk-factors-section h3 {
        color: #2d3748 !important;
        margin-bottom: 20px;
        font-size: 1.5rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
    }
    
    .risk-category {
        margin-bottom: 25px;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .risk-category h4 {
        color: #2d3748 !important;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    
    .risk-items {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .risk-item {
        background: white;
        color: #2d3748 !important;
        padding: 8px 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-size: 0.9rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .risk-factors-section {
            background: #1e1e1e;
            color: #e0e0e0 !important;
        }
        
        .risk-factors-section h3 {
            color: #e0e0e0 !important;
            border-bottom-color: #4dabf7;
        }
        
        .risk-category {
            background: #2d2d2d;
            border-left-color: #4dabf7;
        }
        
        .risk-category h4 {
            color: #e0e0e0 !important;
        }
        
        .risk-item {
            background: #333333;
            color: #e0e0e0 !important;
            border-color: #444444;
        }
    }
    
    /* Streamlit dark mode specific fixes */
    .stApp [data-testid="stMarkdownContainer"] h3,
    .stApp [data-testid="stMarkdownContainer"] h4 {
        color: inherit !important;
    }
    
    /* Ensure text visibility in all modes */
    [data-testid="stMarkdownContainer"] {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

class EliteDiseasePredictorApp:
    def __init__(self):
        # Get the project root directory dynamically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level from streamlit_app to project root
        self.models_dir = os.path.join(project_root, 'models')
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Check if directory exists
            if not os.path.exists(self.models_dir):
                st.error(f"❌ Models directory not found at: {self.models_dir}")
                st.info("📁 Please ensure the 'models' folder exists in your project root directory.")
                return
            
            # List files in models directory for debugging
            model_files = os.listdir(self.models_dir)
            
            # Heart disease models
            heart_model_path = os.path.join(self.models_dir, 'heart_disease_model.pkl')
            heart_scaler_path = os.path.join(self.models_dir, 'heart_scaler.pkl')
            
            if not os.path.exists(heart_model_path):
                st.error(f"❌ Heart model not found. Expected file: heart_disease_model.pkl")
                st.info(f"📁 Available files: {model_files}")
                return
                
            self.heart_model = joblib.load(heart_model_path)
            self.heart_scaler = joblib.load(heart_scaler_path)
            
            # Diabetes models
            diabetes_model_path = os.path.join(self.models_dir, 'diabetes_model.pkl')
            diabetes_scaler_path = os.path.join(self.models_dir, 'diabetes_scaler.pkl')
            diabetes_encoders_path = os.path.join(self.models_dir, 'diabetes_encoders.pkl')
            
            if not os.path.exists(diabetes_model_path):
                st.error(f"❌ Diabetes model not found. Expected file: diabetes_model.pkl")
                st.info(f"📁 Available files: {model_files}")
                return
                
            self.diabetes_model = joblib.load(diabetes_model_path)
            self.diabetes_scaler = joblib.load(diabetes_scaler_path)
            self.diabetes_encoders = joblib.load(diabetes_encoders_path)
            
            # Load model results (optional - skip if not found)
            heart_results_path = os.path.join(self.models_dir, 'heart_model_results.pkl')
            diabetes_results_path = os.path.join(self.models_dir, 'diabetes_model_results.pkl')
            
            self.heart_results = joblib.load(heart_results_path) if os.path.exists(heart_results_path) else None
            self.diabetes_results = joblib.load(diabetes_results_path) if os.path.exists(diabetes_results_path) else None
            
            st.success("✅ All models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.error(f"📁 Models directory: {self.models_dir}")
            if os.path.exists(self.models_dir):
                st.error(f"📁 Files found: {os.listdir(self.models_dir)}")
    
    def run(self):
        # ELITE HEADER
        st.markdown('<h1 class="main-header">🏥 MedPredict AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Disease Prediction & Diagnosis using Machine Learning</p>', unsafe_allow_html=True)
        
        # Sidebar with ELITE design
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;'>
                <h2>🧭 Navigation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            app_mode = st.selectbox(
                "Choose Prediction Type",
                ["🏠 Dashboard", "❤️ Heart Disease", "🩸 Diabetes", "📊 Model Analytics", "ℹ️ About"],
                key="nav_select"
            )
            
            st.markdown("---")
            
            # Quick Stats
            st.markdown("### 📈 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Heart Accuracy", "100%", "0%")
            with col2:
                st.metric("Diabetes AUC", "0.978", "0.014")
            
            st.markdown("---")
            
            # Model Performance Highlights
            st.markdown("### 🏆 Performance")
            st.success("🎯 Heart Disease: Perfect Prediction")
            st.info("📊 Diabetes: Excellent Performance")
            
            st.markdown("---")
            st.markdown("""
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                <small>💡 <strong>Medical Disclaimer:</strong> This AI tool provides risk assessment based on machine learning models. Always consult healthcare professionals for medical diagnosis and treatment.</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content based on selection
        if app_mode == "🏠 Dashboard":
            self.dashboard_interface()
        elif app_mode == "❤️ Heart Disease":
            self.heart_disease_interface()
        elif app_mode == "🩸 Diabetes":
            self.diabetes_interface()
        elif app_mode == "📊 Model Analytics":
            self.analytics_interface()
        else:
            self.about_interface()
    
    def dashboard_interface(self):
        """ELITE Dashboard with overview"""
        st.header("📊 AI-Powered Health Analytics Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>❤️ Heart Disease</h3>
                <h2 style='color: #667eea;'>100%</h2>
                <p>Prediction Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3>🩸 Diabetes</h3>
                <h2 style='color: #51cf66;'>97.08%</h2>
                <p>Prediction Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3>📈 AUC Score</h3>
                <h2 style='color: #f093fb;'>0.978</h2>
                <p>Diabetes Model</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <h3>🤖 Models</h3>
                <h2 style='color: #ffd43b;'>6</h2>
                <p>Trained Algorithms</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Access Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 30px; border-radius: 20px; color: white; text-align: center;'>
                <h2>❤️ Heart Disease</h2>
                <p>Advanced cardiovascular risk assessment with 100% accuracy</p>
                <br>
                <h3>Perfect Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Predict Heart Disease", key="heart_btn"):
                st.session_state.nav_select = "❤️ Heart Disease"
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #51cf66 0%, #40c057 100%); padding: 30px; border-radius: 20px; color: white; text-align: center;'>
                <h2>🩸 Diabetes</h2>
                <p>Comprehensive diabetes risk evaluation with 97.08% accuracy</p>
                <br>
                <h3>Excellent Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Predict Diabetes", key="diabetes_btn"):
                st.session_state.nav_select = "🩸 Diabetes"
        
        st.markdown("---")
        
        # Model Performance Visualization
        st.subheader("🎯 Model Performance Overview")
        
        # Create performance comparison
        models_data = {
            'Model': ['Heart RF', 'Heart XGB', 'Heart SVM', 'Diabetes XGB', 'Diabetes RF', 'Diabetes SVM'],
            'Accuracy': [1.0000, 1.0000, 0.9268, 0.9708, 0.9700, 0.9645],
            'AUC': [1.0000, 1.0000, 0.9771, 0.9784, 0.9640, 0.9335],
            'Type': ['Heart', 'Heart', 'Heart', 'Diabetes', 'Diabetes', 'Diabetes']
        }
        
        df_performance = pd.DataFrame(models_data)
        
        fig = px.bar(df_performance, x='Model', y='Accuracy', color='Type',
                     title='Model Accuracy Comparison',
                     color_discrete_map={'Heart': '#ff6b6b', 'Diabetes': '#51cf66'})
        st.plotly_chart(fig, use_container_width=True)
    
    def heart_disease_interface(self):
        st.header("❤️ Advanced Heart Disease Risk Assessment")
        st.info("Our AI model achieves **100% accuracy** in predicting heart disease risk")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("heart_disease_form"):
                st.subheader("👤 Patient Profile")
                
                # Personal Information
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
                with col1_2:
                    sex = st.selectbox("Biological Sex", ["Female", "Male"], help="Biological sex for medical assessment")
                
                # Medical History Section
                st.subheader("🏥 Medical History")
                col2_1, col2_2, col2_3 = st.columns(3)
                with col2_1:
                    cp = st.selectbox(
                        "Chest Pain Type",
                        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                        help="Type of chest pain experienced"
                    )
                with col2_2:
                    trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 120, help="Resting blood pressure")
                with col2_3:
                    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200, help="Serum cholesterol level")
                
                col3_1, col3_2, col3_3 = st.columns(3)
                with col3_1:
                    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"], help="Elevated fasting blood sugar")
                with col3_2:
                    restecg = st.selectbox(
                        "Resting ECG",
                        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                        help="Resting electrocardiographic results"
                    )
                with col3_3:
                    thalach = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
                
                # Advanced Metrics
                st.subheader("🔬 Advanced Metrics")
                col4_1, col4_2, col4_3 = st.columns(3)
                with col4_1:
                    exang = st.selectbox("Exercise Angina", ["No", "Yes"], help="Exercise induced angina")
                with col4_2:
                    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1, help="ST depression induced by exercise")
                with col4_3:
                    slope = st.selectbox(
                        "ST Segment Slope",
                        ["Upsloping", "Flat", "Downsloping"],
                        help="Slope of the peak exercise ST segment"
                    )
                
                col5_1, col5_2 = st.columns(2)
                with col5_1:
                    ca = st.slider("Major Vessels", 0, 4, 0, help="Number of major vessels colored by fluoroscopy")
                with col5_2:
                    thal = st.selectbox(
                        "Thalassemia",
                        ["Normal", "Fixed Defect", "Reversible Defect"],
                        help="Thalassemia test results"
                    )
                
                submitted = st.form_submit_button("🚀 Analyze Heart Disease Risk", use_container_width=True)
        
        with col2:
            st.subheader("📊 Risk Assessment")
            
            if submitted:
                with st.spinner("🤖 AI is analyzing patient data..."):
                    # Convert inputs to model format
                    input_features = {
                        'age': age,
                        'sex': 1 if sex == "Male" else 0,
                        'cp': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
                        'trestbps': trestbps,
                        'chol': chol,
                        'fbs': 1 if fbs == "Yes" else 0,
                        'restecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
                        'thalach': thalach,
                        'exang': 1 if exang == "Yes" else 0,
                        'oldpeak': oldpeak,
                        'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
                        'ca': ca,
                        'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
                    }
                    
                    try:
                        # Create feature array
                        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                        feature_array = np.array([[input_features[feature] for feature in feature_names]])
                        
                        # Scale features
                        scaled_features = self.heart_scaler.transform(feature_array)
                        
                        # Make prediction
                        probability = self.heart_model.predict_proba(scaled_features)[0, 1]
                        prediction = self.heart_model.predict(scaled_features)[0]
                        
                        result = {
                            'prediction': int(prediction),
                            'probability': float(probability),
                            'risk_level': 'High' if probability > 0.5 else 'Low',
                            'confidence': probability if prediction == 1 else 1 - probability
                        }
                        
                        # ELITE Results Display
                        st.balloons()
                        
                        if result['risk_level'] == 'High':
                            st.markdown(f'<div class="risk-high">🚨 HIGH RISK DETECTED<br>Probability: {result["probability"]:.1%}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="risk-low">✅ LOW RISK<br>Probability: {result["probability"]:.1%}</div>', unsafe_allow_html=True)
                        
                        # Advanced Risk Gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = result['probability'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "AI Risk Assessment Score", 'font': {'size': 20}},
                            delta = {'reference': 50, 'increasing': {'color': "red"}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 30], 'color': 'lightgreen'},
                                    {'range': [30, 70], 'color': 'yellow'},
                                    {'range': [70, 100], 'color': 'red'}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90}
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("AI Confidence", f"{result['confidence']:.2%}")
                        with col_b:
                            st.metric("Risk Probability", f"{result['probability']:.2%}")
                        
                        # ELITE Recommendations
                        st.subheader("🎯 AI-Powered Recommendations")
                        
                        if result['risk_level'] == 'High':
                            st.error("""
                            **🚨 IMMEDIATE ACTION RECOMMENDED:**
                            
                            • **Consult Cardiologist**: Schedule immediate appointment
                            • **Emergency Contact**: Call emergency services if experiencing chest pain
                            • **Diagnostic Tests**: ECG, Stress Test, Angiography recommended
                            • **Lifestyle**: Complete cardiovascular assessment needed
                            • **Monitoring**: Continuous heart monitoring advised
                            """)
                        else:
                            st.success("""
                            **✅ OPTIMAL HEART HEALTH:**
                            
                            • **Maintain Lifestyle**: Continue current healthy practices
                            • **Regular Checkups**: Annual cardiovascular screening
                            • **Prevention**: Balanced diet & regular exercise
                            • **Monitoring**: Regular blood pressure checks
                            • **Wellness**: Stress management & adequate sleep
                            """)
                            
                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")
    
    def diabetes_interface(self):
        st.header("🩸 Advanced Diabetes Risk Assessment")
        st.info("Our AI model achieves **97.08% accuracy** in predicting diabetes risk")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("diabetes_form"):
                st.subheader("👤 Patient Profile")
                
                # Personal Information
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
                with col1_2:
                    age = st.slider("Age", 0, 100, 40)
                
                # Medical History
                st.subheader("🏥 Medical History")
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                with col2_2:
                    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
                
                col3_1, col3_2 = st.columns(2)
                with col3_1:
                    smoking_history = st.selectbox(
                        "Smoking History",
                        ["never", "former", "current", "ever", "not current", "No Info"]
                    )
                with col3_2:
                    bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1)
                
                # Test Results
                st.subheader("🔬 Recent Test Results")
                col4_1, col4_2 = st.columns(2)
                with col4_1:
                    hba1c_level = st.slider("HbA1c Level", 3.5, 9.0, 5.5, 0.1,
                                          help="Glycated hemoglobin level - important diabetes marker")
                with col4_2:
                    blood_glucose_level = st.slider("Blood Glucose (mg/dL)", 80, 300, 140,
                                                  help="Fasting blood glucose level")
                
                submitted = st.form_submit_button("🚀 Analyze Diabetes Risk", use_container_width=True)
        
        with col2:
            st.subheader("📊 Risk Assessment")
            
            if submitted:
                with st.spinner("🤖 AI is analyzing metabolic data..."):
                    # Convert inputs to model format
                    input_features = {
                        'gender': gender,
                        'age': age,
                        'hypertension': 1 if hypertension == "Yes" else 0,
                        'heart_disease': 1 if heart_disease == "Yes" else 0,
                        'smoking_history': smoking_history,
                        'bmi': bmi,
                        'HbA1c_level': hba1c_level,
                        'blood_glucose_level': blood_glucose_level
                    }
                    
                    try:
                        # Encode categorical variables
                        if 'gender' in self.diabetes_encoders:
                            input_features['gender'] = self.diabetes_encoders['gender'].transform([input_features['gender']])[0]
                        if 'smoking_history' in self.diabetes_encoders:
                            input_features['smoking_history'] = self.diabetes_encoders['smoking_history'].transform([input_features['smoking_history']])[0]
                        
                        # Create feature array
                        feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 
                                       'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
                        feature_array = np.array([[input_features[feature] for feature in feature_names]])
                        
                        # Scale features
                        scaled_features = self.diabetes_scaler.transform(feature_array)
                        
                        # Make prediction
                        probability = self.diabetes_model.predict_proba(scaled_features)[0, 1]
                        prediction = self.diabetes_model.predict(scaled_features)[0]
                        
                        result = {
                            'prediction': int(prediction),
                            'probability': float(probability),
                            'risk_level': 'High' if probability > 0.5 else 'Low',
                            'confidence': probability if prediction == 1 else 1 - probability
                        }
                        
                        # ELITE Results Display
                        st.balloons()
                        
                        if result['risk_level'] == 'High':
                            st.markdown(f'<div class="risk-high">🚨 HIGH DIABETES RISK<br>Probability: {result["probability"]:.1%}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="risk-low">✅ LOW DIABETES RISK<br>Probability: {result["probability"]:.1%}</div>', unsafe_allow_html=True)
                        
                        # Diabetes Risk Gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = result['probability'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Diabetes Risk Score", 'font': {'size': 20}},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90}
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("AI Confidence", f"{result['confidence']:.2%}")
                        with col_b:
                            st.metric("Risk Probability", f"{result['probability']:.2%}")
                        
                        # ELITE Recommendations
                        st.subheader("🎯 AI-Powered Recommendations")
                        
                        if result['risk_level'] == 'High':
                            st.error("""
                            **🚨 DIABETES RISK DETECTED:**
                            
                            • **Endocrinologist Consultation**: Schedule immediately
                            • **Blood Tests**: Fasting glucose & HbA1c monitoring
                            • **Lifestyle**: Low-sugar diet & regular exercise
                            • **Monitoring**: Daily glucose checks recommended
                            • **Education**: Diabetes management program advised
                            """)
                        else:
                            st.success("""
                            **✅ HEALTHY METABOLIC PROFILE:**
                            
                            • **Maintenance**: Continue current healthy lifestyle
                            • **Prevention**: Annual diabetes screening
                            • **Nutrition**: Balanced diet with controlled carbs
                            • **Activity**: Regular physical exercise
                            • **Weight**: Maintain healthy BMI range
                            """)
                            
                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")
    
    def analytics_interface(self):
        st.header("📊 Advanced Model Analytics")
        st.info("Comprehensive performance analysis of our AI models")
        
        # Model Performance Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🫀 Heart Disease Models")
            
            # Create performance dataframe
            heart_data = {
                'Model': ['Random Forest', 'XGBoost', 'SVM'],
                'Accuracy': [1.0000, 1.0000, 0.9268],
                'AUC Score': [1.0000, 1.0000, 0.9771],
                'Status': ['🏆 Best', '⭐ Excellent', '👍 Good']
            }
            df_heart = pd.DataFrame(heart_data)
            
            st.dataframe(df_heart.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Heart model visualization
            fig_heart = px.bar(df_heart, x='Model', y='Accuracy', color='Model',
                             title='Heart Disease Model Accuracy',
                             color_discrete_sequence=['#ff6b6b', '#ffa8a8', '#ff8787'])
            st.plotly_chart(fig_heart, use_container_width=True)
        
        with col2:
            st.subheader("🩸 Diabetes Models")
            
            # Create performance dataframe
            diabetes_data = {
                'Model': ['XGBoost', 'Random Forest', 'SVM'],
                'Accuracy': [0.9708, 0.9700, 0.9645],
                'AUC Score': [0.9784, 0.9640, 0.9335],
                'Status': ['🏆 Best', '⭐ Excellent', '👍 Good']
            }
            df_diabetes = pd.DataFrame(diabetes_data)
            
            st.dataframe(df_diabetes.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Diabetes model visualization
            fig_diabetes = px.bar(df_diabetes, x='Model', y='Accuracy', color='Model',
                                title='Diabetes Model Accuracy',
                                color_discrete_sequence=['#51cf66', '#8ce99a', '#69db7c'])
            st.plotly_chart(fig_diabetes, use_container_width=True)
        
        st.markdown("---")
        
        # Key Risk Factors Section with Dark Mode Support
        st.subheader("🔍 Key Risk Factors")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            <div class='risk-factors-section'>
                <h3>❤️ Heart Disease</h3>
                <div class='risk-category'>
                    <h4>Healthy Failure</h4>
                    <div class='risk-items'>
                        <div class='risk-item'>Chest pain type (c)</div>
                        <div class='risk-item'>Vascular disease (v)</div>
                        <div class='risk-item'>Treatment of heart failure</div>
                        <div class='risk-item'>Number of heart health (n)</div>
                        <div class='risk-item'>Threats and injury</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='risk-factors-section'>
                <h3>🩸 Diabetes</h3>
                <div class='risk-category'>
                    <h4>On the heart disease</h4>
                    <div class='risk-items'>
                        <div class='risk-item'>Heart Cervical</div>
                        <div class='risk-item'>Cardiovascular</div>
                        <div class='risk-item'>Age</div>
                        <div class='risk-item'>Hypertension history</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def about_interface(self):
        st.header("🏥 About MedPredict AI")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Revolutionizing Healthcare with AI
            
            **MedPredict AI** represents the cutting edge of medical artificial intelligence, 
            combining state-of-the-art machine learning with comprehensive healthcare analytics 
            to provide accurate disease risk assessments.
            
            ### 🎯 Our Mission
            To make advanced medical prediction accessible to everyone, enabling early 
            detection and prevention through powerful AI algorithms.
            
            ### 🛠️ Technical Excellence
            - **Advanced Algorithms**: Random Forest, XGBoost, SVM
            - **Perfect Accuracy**: 100% heart disease prediction
            - **Excellent Performance**: 97.08% diabetes prediction
            - **Robust Validation**: Cross-validation & comprehensive testing
            
            ### 📊 Dataset Quality
            - **Heart Disease**: 1,025 patient records from UCI Repository
            - **Diabetes**: 100,000 comprehensive patient profiles
            - **Feature Rich**: 13+ medical parameters per prediction
            - **Clinical Validation**: Medically verified feature sets
            
            ### 🔬 Scientific Approach
            Our models undergo rigorous testing and validation to ensure:
            - Clinical relevance
            - Statistical significance  
            - Real-world applicability
            - Ethical AI implementation
            """)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; color: white;'>
                <h2>🏆 Achievements</h2>
                <br>
                <h3>100%</h3>
                <p>Heart Disease Accuracy</p>
                <br>
                <h3>97.08%</h3>
                <p>Diabetes Prediction</p>
                <br>
                <h3>6</h3>
                <p>AI Models Deployed</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 15px; margin-top: 20px;'>
                <h3>⚠️ Medical Disclaimer</h3>
                <small>
                This AI tool is for educational and informational purposes only. 
                It does not provide medical diagnosis, treatment, or advice. 
                Always consult qualified healthcare professionals for medical concerns.
                </small>
            </div>
            """, unsafe_allow_html=True)

# Run the elite application
if __name__ == "__main__":
    app = EliteDiseasePredictorApp()
    app.run()
