```markdown
# 🔌 API Documentation

## Overview
MedPredict AI provides both a web interface and programmatic APIs for disease prediction. This document covers the programmatic usage of the prediction models.

## Quick Start

```python
from src.prediction import DiseasePredictor

# Initialize the predictor
predictor = DiseasePredictor()

# Make predictions
heart_result = predictor.predict_heart_disease(age=45, sex=1, cp=2, trestbps=130, chol=250)
diabetes_result = predictor.predict_diabetes(age=35, gender="Male", bmi=26.5, HbA1c_level=5.8)

print(f"Heart Disease Risk: {heart_result['probability']:.2%}")
print(f"Diabetes Risk: {diabetes_result['probability']:.2%}")
```

## DiseasePredictor Class

### Initialization
```python
predictor = DiseasePredictor()
```
Loads all trained models and preprocessing objects automatically.

### Methods

#### `predict_heart_disease(input_features)`
Predicts heart disease risk based on clinical parameters.

**Parameters:**
- `input_features` (dict): Dictionary containing patient features

**Required Features:**
```python
{
    'age': 45,                    # Age in years (20-100)
    'sex': 1,                     # Sex (0=Female, 1=Male)
    'cp': 2,                      # Chest pain type (0-3)
    'trestbps': 130,              # Resting blood pressure (90-200)
    'chol': 250,                  # Cholesterol (100-600)
    'fbs': 0,                     # Fasting blood sugar (0=False, 1=True)
    'restecg': 1,                 # Resting ECG (0-2)
    'thalach': 150,               # Max heart rate (60-220)
    'exang': 0,                   # Exercise angina (0=No, 1=Yes)
    'oldpeak': 1.2,               # ST depression (0.0-6.0)
    'slope': 1,                   # ST segment slope (0-2)
    'ca': 0,                      # Major vessels (0-4)
    'thal': 2                     # Thalassemia (0-3)
}
```

**Returns:**
```python
{
    'prediction': 1,              # Binary prediction (0=No, 1=Yes)
    'probability': 0.8765,        # Risk probability (0.0-1.0)
    'risk_level': 'High',         # Risk category ('Low', 'High')
    'confidence': 0.8765          # Model confidence
}
```

#### `predict_diabetes(input_features)`
Predicts diabetes risk based on health indicators.

**Parameters:**
- `input_features` (dict): Dictionary containing patient features

**Required Features:**
```python
{
    'gender': 'Male',             # Gender ('Female', 'Male', 'Other')
    'age': 45.0,                  # Age in years (0.08-80)
    'hypertension': 0,            # Hypertension (0=No, 1=Yes)
    'heart_disease': 0,           # Heart disease (0=No, 1=Yes)
    'smoking_history': 'never',   # Smoking history
    'bmi': 27.5,                  # Body Mass Index (10.0-95.0)
    'HbA1c_level': 5.7,           # HbA1c level (3.5-9.0)
    'blood_glucose_level': 135    # Blood glucose (80-300)
}
```

**Smoking History Options:**
- `'never'`, `'former'`, `'current'`, `'ever'`, `'not current'`, `'No Info'`

**Returns:**
```python
{
    'prediction': 0,              # Binary prediction (0=No, 1=Yes)
    'probability': 0.2341,        # Risk probability (0.0-1.0)
    'risk_level': 'Low',          # Risk category ('Low', 'High')
    'confidence': 0.7659          # Model confidence
}
```

## DataProcessor Class

### Usage for Training
```python
from src.data_processing import DataProcessor

processor = DataProcessor()

# Load data
heart_df = processor.load_heart_data()
diabetes_df, encoders = processor.load_diabetes_data()

# Prepare features for training
X_train_heart, X_test_heart, y_train_heart, y_test_heart, heart_features = processor.prepare_heart_features(heart_df)
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes, diabetes_features = processor.prepare_diabetes_features(diabetes_df)
```

## ModelTrainer Class

### Usage for Training Models
```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()

# Train models
heart_model = trainer.train_heart_disease_models()
diabetes_model = trainer.train_diabetes_models()
```

## Error Handling

### Common Exceptions

```python
try:
    result = predictor.predict_heart_disease(features)
except KeyError as e:
    print(f"Missing feature: {e}")
except ValueError as e:
    print(f"Invalid feature value: {e}")
except Exception as e:
    print(f"Prediction error: {e}")
```

### Input Validation
All input features are automatically validated:
- Type checking
- Range validation
- Required feature checking

## Batch Prediction

### Multiple Predictions
```python
# List of patient records
patients = [
    {'age': 45, 'sex': 1, 'cp': 2, ...},
    {'age': 52, 'sex': 0, 'cp': 1, ...},
    {'age': 38, 'sex': 1, 'cp': 0, ...}
]

results = []
for patient in patients:
    result = predictor.predict_heart_disease(patient)
    results.append(result)
```

### DataFrame Integration
```python
import pandas as pd

# Create DataFrame with patient data
df_patients = pd.DataFrame(patients)

# Add predictions
df_patients['heart_risk'] = df_patients.apply(
    lambda row: predictor.predict_heart_disease(row.to_dict())['probability'], 
    axis=1
)
```

## Configuration

### Model Paths
Models are automatically loaded from:
- `models/heart_disease_model.pkl`
- `models/diabetes_model.pkl`
- `models/heart_scaler.pkl`
- `models/diabetes_scaler.pkl`
- `models/diabetes_encoders.pkl`

### Custom Paths
```python
# For custom model locations
predictor = DiseasePredictor(custom_models_path='/path/to/models')
```

## Performance Considerations

### Memory Usage
- Models require ~50-100MB RAM
- Single prediction: <1ms
- Batch processing recommended for large datasets

### Thread Safety
- Predictor is thread-safe for concurrent predictions
- Model loading should be done once per process

## Example Use Cases

### Healthcare Application Integration
```python
class HealthcareApp:
    def __init__(self):
        self.predictor = DiseasePredictor()
    
    def assess_patient_risk(self, patient_data):
        heart_risk = self.predictor.predict_heart_disease(patient_data)
        diabetes_risk = self.predictor.predict_diabetes(patient_data)
        
        return {
            'heart_disease': heart_risk,
            'diabetes': diabetes_risk,
            'overall_risk': max(heart_risk['probability'], diabetes_risk['probability'])
        }
```

### Research and Analysis
```python
# Analyze feature importance
def analyze_risk_factors(predictor, base_features, variations):
    results = []
    for feature, values in variations.items():
        for value in values:
            test_features = base_features.copy()
            test_features[feature] = value
            result = predictor.predict_heart_disease(test_features)
            results.append({
                'feature': feature,
                'value': value,
                'risk': result['probability']
            })
    return pd.DataFrame(results)
```

## Support

For API issues or questions:
1. Check the input feature requirements
2. Verify model files are present
3. Ensure all dependencies are installed
4. Review the error messages for specific guidance

For additional support, create an issue on the GitHub repository.
```

