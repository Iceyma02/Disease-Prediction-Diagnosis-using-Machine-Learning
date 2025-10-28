"""
MedPredict AI - Main Entry Point
Advanced Disease Prediction & Diagnosis using Machine Learning
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

def main():
    """Main entry point for MedPredict AI"""
    parser = argparse.ArgumentParser(
        description="MedPredict AI - Disease Prediction & Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run Streamlit app
  python main.py --train                  # Train models only
  python main.py --port 8080              # Run on custom port
  python main.py --demo                   # Run demo predictions
        """
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train machine learning models'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run demo predictions with sample data'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port to run Streamlit app (default: 8501)'
    )
    
    parser.add_argument(
        '--host', 
        type=str, 
        default='localhost',
        help='Host to run Streamlit app (default: localhost)'
    )
    
    args = parser.parse_args()
    
    if args.train:
        # Train models
        print("🚀 Training MedPredict AI models...")
        from src.model_training import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_heart_disease_models()
        trainer.train_diabetes_models()
        print("✅ Model training completed!")
        
    elif args.demo:
        # Run demo predictions
        print("🎯 Running MedPredict AI demo...")
        from src.prediction import DiseasePredictor
        predictor = DiseasePredictor()
        
        # Sample heart disease prediction
        print("\n❤️ Heart Disease Demo Prediction:")
        heart_features = {
            'age': 52,
            'sex': 1,
            'cp': 0,
            'trestbps': 125,
            'chol': 212,
            'fbs': 0,
            'restecg': 1,
            'thalach': 168,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 2,
            'thal': 3
        }
        heart_result = predictor.predict_heart_disease(heart_features)
        print(f"   Risk Probability: {heart_result['probability']:.2%}")
        print(f"   Risk Level: {heart_result['risk_level']}")
        
        # Sample diabetes prediction
        print("\n🩸 Diabetes Demo Prediction:")
        diabetes_features = {
            'gender': 'Male',
            'age': 50,
            'hypertension': 0,
            'heart_disease': 0,
            'smoking_history': 'never',
            'bmi': 27.5,
            'HbA1c_level': 5.9,
            'blood_glucose_level': 140
        }
        diabetes_result = predictor.predict_diabetes(diabetes_features)
        print(f"   Risk Probability: {diabetes_result['probability']:.2%}")
        print(f"   Risk Level: {diabetes_result['risk_level']}")
        
    else:
        # Run Streamlit app
        print(f"🏥 Starting MedPredict AI on http://{args.host}:{args.port}")
        os.system(f"streamlit run streamlit_app/app.py --server.port {args.port} --server.address {args.host}")

if __name__ == "__main__":
    main()
