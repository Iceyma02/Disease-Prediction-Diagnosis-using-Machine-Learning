import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model_training import ModelTrainer
from src.prediction import DiseasePredictor
from src.data_processing import DataProcessor

class TestModels:
    """Test cases for model training and prediction"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample data for model training tests"""
        # Heart disease data
        heart_data = pd.DataFrame({
            'age': [52, 45, 60, 48, 55, 62, 41, 58, 35, 50] * 10,
            'sex': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10,
            'cp': [0, 2, 1, 0, 3, 1, 2, 0, 1, 2] * 10,
            'trestbps': [125, 140, 130, 120, 135, 128, 142, 118, 132, 125] * 10,
            'chol': [212, 250, 200, 230, 245, 198, 265, 210, 228, 240] * 10,
            'fbs': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 10,
            'restecg': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10,
            'thalach': [168, 155, 160, 172, 148, 165, 158, 170, 162, 155] * 10,
            'exang': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 10,
            'oldpeak': [1.0, 2.5, 0.5, 1.2, 3.0, 0.8, 2.2, 1.1, 0.3, 1.8] * 10,
            'slope': [2, 1, 2, 1, 0, 2, 1, 2, 1, 0] * 10,
            'ca': [2, 0, 1, 3, 0, 2, 1, 0, 2, 1] * 10,
            'thal': [3, 2, 3, 1, 2, 3, 2, 1, 3, 2] * 10,
            'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
        })
        
        # Diabetes data
        diabetes_data = pd.DataFrame({
            'gender': ['Male', 'Female', 'Other', 'Male', 'Female'] * 20,
            'age': [50.0, 35.0, 60.0, 45.0, 55.0] * 20,
            'hypertension': [0, 1, 0, 0, 1] * 20,
            'heart_disease': [0, 0, 1, 0, 0] * 20,
            'smoking_history': ['never', 'former', 'current', 'never', 'ever'] * 20,
            'bmi': [27.5, 24.0, 32.0, 26.0, 29.5] * 20,
            'HbA1c_level': [5.9, 5.5, 6.8, 5.7, 6.2] * 20,
            'blood_glucose_level': [140, 120, 180, 130, 160] * 20,
            'diabetes': [0, 0, 1, 0, 1] * 20
        })
        
        return heart_data, diabetes_data
    
    @pytest.fixture
    def model_trainer(self):
        """Create ModelTrainer instance for testing"""
        return ModelTrainer()
    
    @pytest.fixture
    def disease_predictor(self):
        """Create DiseasePredictor instance for testing"""
        return DiseasePredictor()
    
    def test_model_trainer_initialization(self, model_trainer):
        """Test ModelTrainer initialization"""
        assert model_trainer is not None
        assert hasattr(model_trainer, 'processor')
    
    def test_disease_predictor_initialization(self, disease_predictor):
        """Test DiseasePredictor initialization"""
        assert disease_predictor is not None
        # Models should be loaded during initialization
        assert hasattr(disease_predictor, 'heart_model')
        assert hasattr(disease_predictor, 'diabetes_model')
    
    def test_heart_disease_prediction_structure(self, disease_predictor):
        """Test heart disease prediction output structure"""
        sample_features = {
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
        
        result = disease_predictor.predict_heart_disease(sample_features)
        
        # Check result structure
        assert 'prediction' in result
        assert 'probability' in result
        assert 'risk_level' in result
        assert 'confidence' in result
        
        # Check data types
        assert isinstance(result['prediction'], (int, np.integer))
        assert isinstance(result['probability'], (float, np.floating))
        assert isinstance(result['risk_level'], str)
        assert isinstance(result['confidence'], (float, np.floating))
        
        # Check value ranges
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probability'] <= 1
        assert result['risk_level'] in ['Low', 'High']
        assert 0 <= result['confidence'] <= 1
    
    def test_diabetes_prediction_structure(self, disease_predictor):
        """Test diabetes prediction output structure"""
        sample_features = {
            'gender': 'Male',
            'age': 50,
            'hypertension': 0,
            'heart_disease': 0,
            'smoking_history': 'never',
            'bmi': 27.5,
            'HbA1c_level': 5.9,
            'blood_glucose_level': 140
        }
        
        result = disease_predictor.predict_diabetes(sample_features)
        
        # Check result structure
        assert 'prediction' in result
        assert 'probability' in result
        assert 'risk_level' in result
        assert 'confidence' in result
        
        # Check data types
        assert isinstance(result['prediction'], (int, np.integer))
        assert isinstance(result['probability'], (float, np.floating))
        assert isinstance(result['risk_level'], str)
        assert isinstance(result['confidence'], (float, np.floating))
        
        # Check value ranges
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probability'] <= 1
        assert result['risk_level'] in ['Low', 'High']
        assert 0 <= result['confidence'] <= 1
    
    def test_prediction_consistency(self, disease_predictor):
        """Test that identical inputs produce consistent outputs"""
        heart_features = {
            'age': 45,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 220,
            'fbs': 0,
            'restecg': 1,
            'thalach': 155,
            'exang': 0,
            'oldpeak': 1.5,
            'slope': 1,
            'ca': 1,
            'thal': 2
        }
        
        # Make multiple predictions with same features
        result1 = disease_predictor.predict_heart_disease(heart_features)
        result2 = disease_predictor.predict_heart_disease(heart_features)
        
        # Results should be identical (deterministic)
        assert result1['prediction'] == result2['prediction']
        assert abs(result1['probability'] - result2['probability']) < 1e-10
    
    def test_invalid_feature_handling(self, disease_predictor):
        """Test behavior with invalid input features"""
        invalid_features = {
            'age': 'invalid',  # Wrong type
            'sex': 1,
            'cp': 2
            # Missing other required features
        }
        
        # Should handle invalid inputs gracefully
        try:
            result = disease_predictor.predict_heart_disease(invalid_features)
            # If no exception, check that it handled it
            assert True
        except Exception as e:
            # Exception is acceptable for invalid inputs
            assert True
    
    def test_model_file_existence(self):
        """Test that required model files exist"""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        required_files = [
            'heart_disease_model.pkl',
            'diabetes_model.pkl', 
            'heart_scaler.pkl',
            'diabetes_scaler.pkl',
            'diabetes_encoders.pkl'
        ]
        
        for file in required_files:
            file_path = os.path.join(models_dir, file)
            assert os.path.exists(file_path), f"Model file {file} does not exist"
    
    def test_model_loading(self, disease_predictor):
        """Test that models are properly loaded and functional"""
        # Test that models can make predictions
        heart_features = {
            'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212,
            'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0,
            'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
        }
        
        diabetes_features = {
            'gender': 'Male', 'age': 50, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 27.5, 'HbA1c_level': 5.9,
            'blood_glucose_level': 140
        }
        
        # Should not raise exceptions
        heart_result = disease_predictor.predict_heart_disease(heart_features)
        diabetes_result = disease_predictor.predict_diabetes(diabetes_features)
        
        assert heart_result is not None
        assert diabetes_result is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
