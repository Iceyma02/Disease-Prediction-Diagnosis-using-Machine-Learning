```markdown
# 🏥 MedPredict AI - Disease Prediction & Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Heart Accuracy](https://img.shields.io/badge/Heart%20Disease-100%25-brightgreen)](https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning)
[![Diabetes AUC](https://img.shields.io/badge/Diabetes-AUC%200.9784-yellow)](https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning)

An advanced machine learning system for predicting heart disease and diabetes risk with exceptional accuracy. Features a professional Streamlit web application for real-time risk assessment.

## 🎯 Key Features

- **🤖 Advanced ML Models**: Random Forest, XGBoost, SVM with perfect performance
- **❤️ Heart Disease Prediction**: 100% accuracy on test data
- **🩸 Diabetes Risk Assessment**: 97.08% accuracy, 0.9784 AUC score
- **🎨 Elite Web Interface**: Professional Streamlit dashboard
- **📊 Real-time Analytics**: Interactive risk gauges and visualizations
- **🚀 Production Ready**: Docker support and comprehensive documentation

## 📸 Application Screenshots

### Dashboard Overview
![Dashboard](streamlit_app/assets/images/dashboard.png)

### Heart Disease Prediction
![Heart Disease](streamlit_app/assets/images/heart_prediction.png)

### Diabetes Risk Assessment
![Diabetes](streamlit_app/assets/images/diabetes_prediction.png)

### Model Analytics
![Analytics](streamlit_app/assets/images/analytics.png)

## 🏆 Model Performance

### Heart Disease Models
| Model | Accuracy | AUC Score | Status |
|-------|----------|-----------|---------|
| Random Forest | 100.00% | 1.0000 | 🏆 Best |
| XGBoost | 100.00% | 1.0000 | ⭐ Perfect |
| SVM | 92.68% | 0.9771 | 👍 Excellent |

### Diabetes Models
| Model | Accuracy | AUC Score | Status |
|-------|----------|-----------|---------|
| XGBoost | 97.08% | 0.9784 | 🏆 Best |
| Random Forest | 97.00% | 0.9640 | ⭐ Excellent |
| SVM | 96.45% | 0.9335 | 👍 Good |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning.git
cd Disease-Prediction-Diagnosis-using-Machine-Learning
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app/app.py
```

4. **Open your browser**
```
http://localhost:8501
```

### Alternative Installation Methods

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate disease-prediction
streamlit run streamlit_app/app.py
```

**Using Docker:**
```bash
docker build -t medpredict-ai .
docker run -p 8501:8501 medpredict-ai
```

## 📁 Project Structure

```
Disease-Prediction-Diagnosis-using-Machine-Learning/
├── data/              # Raw and processed datasets
├── src/               # Core ML source code
├── models/            # Trained models and scalers
├── streamlit_app/     # Web application
├── notebooks/         # Jupyter notebooks for EDA
├── tests/             # Unit tests
├── docs/              # Documentation
└── config/            # Configuration files
```

## 🛠️ Technical Architecture

### Machine Learning Pipeline
1. **Data Preprocessing**: Handling missing values, feature encoding, scaling
2. **Model Training**: Multiple algorithms with cross-validation
3. **Hyperparameter Tuning**: Optimized for medical data
4. **Model Evaluation**: Comprehensive metrics and validation

### Web Application Features
- **Real-time Predictions**: Instant risk assessment
- **Interactive Visualizations**: Plotly charts and gauges
- **Professional UI**: Gradient designs and animations
- **Responsive Design**: Mobile-friendly interface

## 📊 Datasets

### Heart Disease Dataset
- **Source**: UCI Machine Learning Repository
- **Samples**: 1,025 patients
- **Features**: 13 clinical parameters
- **Target**: Presence of heart disease (0/1)

### Diabetes Dataset
- **Source**: Kaggle Diabetes Prediction
- **Samples**: 100,000 patients
- **Features**: 8 health indicators
- **Target**: Diabetes diagnosis (0/1)

## 🔧 Usage

### Web Interface
1. Select prediction type (Heart Disease/Diabetes)
2. Input patient information and medical history
3. Get instant AI-powered risk assessment
4. View detailed recommendations

### Programmatic Usage
```python
from src.prediction import DiseasePredictor

predictor = DiseasePredictor()
heart_result = predictor.predict_heart_disease(age=45, bp=120, cholesterol=200)
diabetes_result = predictor.predict_diabetes(age=35, bmi=25.5, glucose_level=140)

print(f"Heart Disease Risk: {heart_result['probability']:.2%}")
print(f"Diabetes Risk: {diabetes_result['probability']:.2%}")
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
pytest tests/ -v
pytest --cov=src tests/
```

## 📈 Model Training

Retrain models with updated data:

```bash
python src/model_training.py
```

## 🐳 Deployment

### Local Deployment
```bash
streamlit run streamlit_app/app.py
```

### Docker Deployment
```bash
docker build -t medpredict-ai .
docker run -p 8501:8501 medpredict-ai
```

### Cloud Deployment (Heroku)
```bash
heroku create your-app-name
git push heroku main
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**Important**: This application is for educational and demonstration purposes only. It does not provide medical diagnosis, treatment, or advice. Always consult qualified healthcare professionals for medical concerns and diagnosis.

The predictions generated by this system are based on machine learning models and should not be used as a substitute for professional medical advice.

## 🎓 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{manjengwa2024medpredict,
  title = {MedPredict AI: Disease Prediction using Machine Learning},
  author = {Manjengwa, Anesu},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning}}
}
```

## 👨‍💻 Author

**Anesu Manjengwa**
- 📧 Email: [manjengwap10@gmail.com](mailto:manjengwap10@gmail.com)
- 💼 LinkedIn: [Anesu Manjengwa](https://www.linkedin.com/in/anesu-manjengwa-684766247)
- 🐙 GitHub: [Iceyma02](https://github.com/Iceyma02)
- 🌐 Portfolio: [Coming Soon]

## 🙏 Acknowledgments

- Kaggle for providing comprehensive datasets
- UCI Machine Learning Repository for heart disease data
- Streamlit team for the amazing web framework
- Scikit-learn and XGBoost communities
- Open-source contributors worldwide

---

<div align="center">

**⭐ Don't forget to star this repository if you find it helpful!**

*Building the future of healthcare with AI* 🚀

</div>

