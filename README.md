# 🩺 Disease Prediction & Diagnosis using Machine Learning

🤖 Disease Prediction & Diagnosis using ML 🩺 Predict heart disease & diabetes risk with Random Forest & XGBoost. Interactive Streamlit app for real-time risk assessment! 📊🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive machine learning system for predicting the likelihood of diseases (Diabetes, Heart Disease) using patient data with an interactive web interface.

## 🌟 Features

- **🤖 Multiple ML Models**: Random Forest, XGBoost, SVM
- **🎯 Disease Prediction**: Diabetes & Heart Disease
- **📊 Interactive Dashboard**: Real-time predictions
- **📈 Data Visualization**: EDA and model performance
- **🔍 Explainable AI**: Feature importance and SHAP values
- **🚀 Production Ready**: Dockerized and deployable

## 📸 Demo

![Demo](streamlit_app/assets/images/demo.gif)

## 🛠️ Installation

### Method 1: Using pip
```bash
git clone https://github.com/yourusername/disease-prediction-ml.git
cd disease-prediction-ml
pip install -r requirements.txt
```

### Method 2: Using Conda
```bash
conda env create -f environment.yml
conda activate disease-prediction
```

### Method 3: Using Docker
```bash
docker build -t disease-prediction .
docker run -p 8501:8501 disease-prediction
```

## 🚀 Quick Start

1. **Prepare Data**: Download datasets from Kaggle and place in `data/raw/`
2. **Train Models**:
   ```bash
   python src/model_training.py
   ```
3. **Launch App**:
   ```bash
   streamlit run streamlit_app/app.py
   ```
4. **Access**: Open `http://localhost:8501` in your browser

## 📊 Datasets

- **Heart Disease**: [Kaggle - Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)
- **Diabetes**: [Kaggle - Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## 🏗️ Project Structure

```
disease-prediction-ml/
├── data/          # Raw and processed data
├── src/           # Source code
├── models/        # Trained models
├── streamlit_app/ # Web application
├── notebooks/     # Jupyter notebooks
└── tests/         # Test cases
```

## 🤖 Models Implemented

- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine
- Logistic Regression
- Gradient Boosting

## 📈 Performance

| Disease | Model | Accuracy | Precision | Recall | F1-Score |
|---------|-------|----------|-----------|--------|----------|
| Heart | XGBoost | 92% | 91% | 90% | 91% |
| Diabetes | Random Forest | 89% | 88% | 87% | 88% |

## 🎯 Usage

### Web Interface
1. Select disease type (Heart/Diabetes)
2. Input patient parameters
3. Get instant risk prediction
4. View feature importance

### Programmatic
```python
from src.prediction import DiseasePredictor

predictor = DiseasePredictor()
result = predictor.predict_heart_disease(age=45, bp=120, cholesterol=200)
print(f"Risk Score: {result['probability']:.2%}")
```

## 🧪 Testing

```bash
pytest tests/ -v
pytest --cov=src tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for the datasets
- Scikit-learn team
- Streamlit team
- Open source community

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/disease-prediction-ml](https://github.com/yourusername/disease-prediction-ml)
```
