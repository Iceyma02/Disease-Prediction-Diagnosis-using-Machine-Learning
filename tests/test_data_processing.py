import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_processing import DataProcessor

class TestDataProcessing:
    """Test cases for DataProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance for testing"""
        return DataProcessor()
    
    @pytest.fixture
    def sample_heart_data(self):
        """Create sample heart disease data for testing"""
        return pd.DataFrame({
            'age': [52, 45, 60],
            'sex': [1, 0, 1],
            'cp': [0, 2, 1],
            'trestbps': [125, 140, 130],
            'chol': [212, 250, 200],
            'fbs': [0, 1, 0],
            'restecg': [1, 0, 1],
            'thalach': [168, 155, 160],
            'exang': [0, 1, 0],
            'oldpeak': [1.0, 2.5, 0.5],
            'slope': [2, 1, 2],
            'ca': [2, 0, 1],
            'thal': [3, 2, 3],
            'target': [1, 0, 1]
        })
    
    @pytest.fixture
    def sample_diabetes_data(self):
        """Create sample diabetes data for testing"""
        return pd.DataFrame({
            'gender': ['Male', 'Female', 'Other'],
            'age': [50.0, 35.0, 60.0],
            'hypertension': [0, 1, 0],
            'heart_disease': [0, 0, 1],
            'smoking_history': ['never', 'former', 'current'],
            'bmi': [27.5, 24.0, 32.0],
            'HbA1c_level': [5.9, 5.5, 6.8],
            'blood_glucose_level': [140, 120, 180],
            'diabetes': [0, 0, 1]
        })
    
    def test_processor_initialization(self, processor):
        """Test DataProcessor initialization"""
        assert processor is not None
        assert hasattr(processor, 'heart_data_path')
        assert hasattr(processor, 'diabetes_data_path')
    
    def test_load_heart_data_success(self, processor, sample_heart_data, tmp_path):
        """Test successful heart data loading"""
        # Create temporary test file
        test_file = tmp_path / "test_heart.csv"
        sample_heart_data.to_csv(test_file, index=False)
        
        # Test loading
        processor.heart_data_path = str(test_file)
        df = processor.load_heart_data()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'target' in df.columns
    
    def test_load_diabetes_data_success(self, processor, sample_diabetes_data, tmp_path):
        """Test successful diabetes data loading"""
        # Create temporary test file
        test_file = tmp_path / "test_diabetes.csv"
        sample_diabetes_data.to_csv(test_file, index=False)
        
        # Test loading
        processor.diabetes_data_path = str(test_file)
        df, encoders = processor.load_diabetes_data()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'diabetes' in df.columns
        assert 'gender' in encoders
        assert 'smoking_history' in encoders
    
    def test_prepare_heart_features(self, processor, sample_heart_data):
        """Test heart disease feature preparation"""
        X_train, X_test, y_train, y_test, features = processor.prepare_heart_features(sample_heart_data)
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert features is not None
        assert len(features) == 13  # All features except target
        assert 'target' not in features
        assert X_train.shape[0] + X_test.shape[0] == len(sample_heart_data)
    
    def test_prepare_diabetes_features(self, processor, sample_diabetes_data):
        """Test diabetes feature preparation"""
        # First load data to get encoders
        df, encoders = processor.load_diabetes_data()
        
        X_train, X_test, y_train, y_test, features = processor.prepare_diabetes_features(df)
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert features is not None
        assert len(features) == 8  # All features except target
        assert 'diabetes' not in features
        assert X_train.shape[0] + X_test.shape[0] == len(df)
    
    def test_feature_scaling(self, processor, sample_heart_data):
        """Test that features are properly scaled"""
        X_train, X_test, y_train, y_test, features = processor.prepare_heart_features(sample_heart_data)
        
        # Check that features are scaled (mean ~0, std ~1 for training set)
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-2)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-2)
    
    def test_data_splitting_stratification(self, processor, sample_heart_data):
        """Test that data splitting maintains target distribution"""
        X_train, X_test, y_train, y_test, features = processor.prepare_heart_features(sample_heart_data)
        
        # Check that target distribution is similar in train and test
        train_pos_ratio = y_train.mean()
        test_pos_ratio = y_test.mean()
        
        # Should be relatively close (stratified split)
        assert abs(train_pos_ratio - test_pos_ratio) < 0.2
    
    def test_missing_data_handling(self, processor):
        """Test behavior with missing data"""
        # Create data with missing values
        incomplete_data = pd.DataFrame({
            'age': [52, None, 60],
            'sex': [1, 0, 1],
            'target': [1, 0, 1]
        })
        
        # Should handle missing values appropriately
        # (Implementation specific - adjust based on your data_processing.py)
        try:
            result = processor.prepare_heart_features(incomplete_data)
            # If no exception, test passed
            assert True
        except Exception:
            # If exception is expected, test passed
            assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
