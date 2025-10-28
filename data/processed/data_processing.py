import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class DataProcessor:
    def __init__(self):
        # Use absolute paths to your files
        self.heart_data_path = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning\src\heart.csv'
        self.diabetes_data_path = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning\src\diabetes_prediction_dataset.csv'
        
    def load_heart_data(self):
        """Load and process heart disease data"""
        try:
            df = pd.read_csv(self.heart_data_path)
            print(f"✅ Heart Disease Data Shape: {df.shape}")
            print(f"✅ Heart Data Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Error loading heart data from {self.heart_data_path}: {e}")
            return None
    
    def load_diabetes_data(self):
        """Load and process diabetes data"""
        try:
            df = pd.read_csv(self.diabetes_data_path)
            print(f"✅ Diabetes Data Shape: {df.shape}")
            print(f"✅ Diabetes Data Columns: {df.columns.tolist()}")
            
            # Encode categorical variables
            label_encoders = {}
            categorical_columns = ['gender', 'smoking_history']
            
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
                    print(f"✅ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
            return df, label_encoders
            
        except Exception as e:
            print(f"❌ Error loading diabetes data from {self.diabetes_data_path}: {e}")
            return None, {}
    
    def prepare_heart_features(self, df):
        """Prepare features for heart disease prediction"""
        if df is None:
            print("❌ No heart data provided")
            return None, None, None, None, None
            
        X = df.drop('target', axis=1)
        y = df['target']
        
        print(f"✅ Heart Features: {X.columns.tolist()}")
        print(f"✅ Heart Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create models directory if it doesn't exist
        models_dir = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning\models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save scaler
        scaler_path = os.path.join(models_dir, 'heart_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ Heart scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def prepare_diabetes_features(self, df):
        """Prepare features for diabetes prediction"""
        if df is None:
            print("❌ No diabetes data provided")
            return None, None, None, None, None
            
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        print(f"✅ Diabetes Features: {X.columns.tolist()}")
        print(f"✅ Diabetes Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        models_dir = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning\models'
        scaler_path = os.path.join(models_dir, 'diabetes_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ Diabetes scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

# Test the data processor
if __name__ == "__main__":
    print("🧪 TESTING DATA PROCESSOR WITH ABSOLUTE PATHS")
    print("=" * 50)
    
    processor = DataProcessor()
    
    # Test heart data
    print("\n🫀 PROCESSING HEART DATA")
    heart_df = processor.load_heart_data()
    if heart_df is not None:
        X_train_heart, X_test_heart, y_train_heart, y_test_heart, heart_features = processor.prepare_heart_features(heart_df)
        print(f"✅ Heart features prepared: {len(heart_features)} features")
    else:
        print("❌ Failed to process heart data")
    
    # Test diabetes data
    print("\n🩸 PROCESSING DIABETES DATA")
    diabetes_df, encoders = processor.load_diabetes_data()
    if diabetes_df is not None:
        X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes, diabetes_features = processor.prepare_diabetes_features(diabetes_df)
        print(f"✅ Diabetes features prepared: {len(diabetes_features)} features")
        print(f"✅ Label encoders created: {list(encoders.keys())}")
    else:
        print("❌ Failed to process diabetes data")
    
    print("\n🎉 DATA PROCESSING COMPLETED!")