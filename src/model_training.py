import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import os
import sys

# Add the current directory to path to import data_processing
sys.path.append(os.path.dirname(__file__))
from data_processing import DataProcessor

class ModelTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        self.models_dir = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning\models'
        
    def train_heart_disease_models(self):
        print("🫀 TRAINING HEART DISEASE MODELS")
        print("=" * 50)
        
        # Load and prepare data
        heart_df = self.processor.load_heart_data()
        if heart_df is None:
            print("❌ Cannot train heart disease models - no data")
            return None
            
        X_train, X_test, y_train, y_test, features = self.processor.prepare_heart_features(heart_df)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        results = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ {name} Results:")
            print(f"   - Accuracy: {accuracy:.4f}")
            print(f"   - AUC: {auc_score:.4f}")
            
            results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'model': model
            }
            
            # Save best model
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_model_name = name
        
        # Save best model
        if best_model is not None:
            model_path = os.path.join(self.models_dir, 'heart_disease_model.pkl')
            joblib.dump(best_model, model_path)
            print(f"\n🎯 Best Heart Disease Model: {best_model_name}")
            print(f"🎯 Best AUC Score: {best_score:.4f}")
            print(f"💾 Model saved to: {model_path}")
            
            # Save results
            results_path = os.path.join(self.models_dir, 'heart_model_results.pkl')
            joblib.dump(results, results_path)
        else:
            print("❌ No model was trained successfully")
        
        return best_model
    
    def train_diabetes_models(self):
        print("\n🩸 TRAINING DIABETES MODELS")
        print("=" * 50)
        
        # Load and prepare data
        diabetes_df, encoders = self.processor.load_diabetes_data()
        if diabetes_df is None:
            print("❌ Cannot train diabetes models - no data")
            return None
            
        X_train, X_test, y_train, y_test, features = self.processor.prepare_diabetes_features(diabetes_df)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        results = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ {name} Results:")
            print(f"   - Accuracy: {accuracy:.4f}")
            print(f"   - AUC: {auc_score:.4f}")
            
            results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'model': model
            }
            
            # Save best model
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_model_name = name
        
        # Save best model and encoders
        if best_model is not None:
            model_path = os.path.join(self.models_dir, 'diabetes_model.pkl')
            encoders_path = os.path.join(self.models_dir, 'diabetes_encoders.pkl')
            
            joblib.dump(best_model, model_path)
            joblib.dump(encoders, encoders_path)
            
            print(f"\n🎯 Best Diabetes Model: {best_model_name}")
            print(f"🎯 Best AUC Score: {best_score:.4f}")
            print(f"💾 Model saved to: {model_path}")
            print(f"💾 Encoders saved to: {encoders_path}")
            
            # Save results
            results_path = os.path.join(self.models_dir, 'diabetes_model_results.pkl')
            joblib.dump(results, results_path)
        else:
            print("❌ No model was trained successfully")
        
        return best_model

if __name__ == "__main__":
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 50)
    
    # Create models directory
    models_dir = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning\models'
    os.makedirs(models_dir, exist_ok=True)
    
    trainer = ModelTrainer()
    
    # Train both models
    heart_model = trainer.train_heart_disease_models()
    diabetes_model = trainer.train_diabetes_models()
    
    print("\n" + "=" * 50)
    if heart_model is not None and diabetes_model is not None:
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("🫀 Heart Disease Models: Random Forest, XGBoost, SVM")
        print("🩸 Diabetes Models: Random Forest, XGBoost, SVM")
        print("💾 All models saved to models/ directory")
    else:
        print("⚠️  Some models failed to train. Check the errors above.")