# Enhanced Visualizations for Heart Disease
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("📈 CREATING COMPREHENSIVE DATA VISUALIZATIONS")

# Load the data first
print("🔍 Loading data files...")

# Set the path to your data files
project_path = r'C:\Users\Icey_m_a\Documents\Icey\Icey\School\Python\Disease-Prediction-Diagnosis-using-Machine-Learning'
heart_path = os.path.join(project_path, 'src', 'heart.csv')
diabetes_path = os.path.join(project_path, 'src', 'diabetes_prediction_dataset.csv')

# Load heart data
heart_df = pd.read_csv(heart_path)
print(f"✅ Heart data loaded: {heart_df.shape}")

# Load diabetes data
diabetes_df = pd.read_csv(diabetes_path)
print(f"✅ Diabetes data loaded: {diabetes_df.shape}")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# 1. Heart Disease Visualizations
print("\n🫀 HEART DISEASE VISUALIZATIONS")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Heart Disease Dataset Analysis', fontsize=16, fontweight='bold')

# Target distribution
heart_df['target'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
axes[0,0].set_title('Heart Disease Distribution\n(0=No, 1=Yes)')
axes[0,0].set_xlabel('Heart Disease')
axes[0,0].set_ylabel('Count')

# Age distribution by target
sns.histplot(data=heart_df, x='age', hue='target', bins=20, ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Heart Disease')

# Cholesterol levels
sns.boxplot(data=heart_df, x='target', y='chol', ax=axes[0,2])
axes[0,2].set_title('Cholesterol Levels by Heart Disease')

# Maximum heart rate
sns.boxplot(data=heart_df, x='target', y='thalach', ax=axes[1,0])
axes[1,0].set_title('Max Heart Rate by Heart Disease')

# Chest pain type
sns.countplot(data=heart_df, x='cp', hue='target', ax=axes[1,1])
axes[1,1].set_title('Chest Pain Type by Heart Disease\n(0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)')

# Sex distribution
sns.countplot(data=heart_df, x='sex', hue='target', ax=axes[1,2])
axes[1,2].set_title('Sex Distribution by Heart Disease\n(0=Female, 1=Male)')

plt.tight_layout()
plt.show()

# Correlation heatmap for heart disease
plt.figure(figsize=(12, 8))
correlation = heart_df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, center=0)
plt.title('Heart Disease Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 2. Diabetes Visualizations
print("\n🩸 DIABETES DATASET VISUALIZATIONS")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Diabetes Dataset Analysis', fontsize=16, fontweight='bold')

# Diabetes distribution
diabetes_df['diabetes'].value_counts().plot(kind='bar', ax=axes[0,0], color=['lightgreen', 'red'])
axes[0,0].set_title('Diabetes Distribution\n(0=No, 1=Yes)')
axes[0,0].set_xlabel('Diabetes')
axes[0,0].set_ylabel('Count')

# Age distribution
sns.histplot(data=diabetes_df, x='age', hue='diabetes', bins=30, ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Diabetes Status')

# HbA1c levels
sns.boxplot(data=diabetes_df, x='diabetes', y='HbA1c_level', ax=axes[0,2])
axes[0,2].set_title('HbA1c Levels by Diabetes Status')

# Blood glucose levels
sns.boxplot(data=diabetes_df, x='diabetes', y='blood_glucose_level', ax=axes[1,0])
axes[1,0].set_title('Blood Glucose Levels by Diabetes Status')

# BMI distribution
sns.boxplot(data=diabetes_df, x='diabetes', y='bmi', ax=axes[1,1])
axes[1,1].set_title('BMI by Diabetes Status')

# Gender distribution
sns.countplot(data=diabetes_df, x='gender', hue='diabetes', ax=axes[1,2])
axes[1,2].set_title('Gender Distribution by Diabetes Status\n(0=Female, 1=Male, 2=Other)')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Feature importance analysis for diabetes
print("\n📊 DIABETES RISK FACTOR ANALYSIS")

# Calculate mean values by diabetes status
diabetes_stats = diabetes_df.groupby('diabetes').agg({
    'age': 'mean',
    'bmi': 'mean', 
    'HbA1c_level': 'mean',
    'blood_glucose_level': 'mean',
    'hypertension': 'mean',
    'heart_disease': 'mean'
}).round(2)

print("Average Values by Diabetes Status:")
print(diabetes_stats)

# Visualize risk factors
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
titles = ['Age', 'BMI', 'HbA1c Level', 'Blood Glucose', 'Hypertension', 'Heart Disease']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    row, col = i // 3, i % 3
    diabetes_stats[metric].plot(kind='bar', ax=axes[row, col], color=['lightgreen', 'red'])
    axes[row, col].set_title(f'Average {title} by Diabetes Status')
    axes[row, col].set_xlabel('Diabetes (0=No, 1=Yes)')
    axes[row, col].set_ylabel(title)

plt.tight_layout()
plt.show()

print("\n🎉 VISUALIZATIONS COMPLETED!")