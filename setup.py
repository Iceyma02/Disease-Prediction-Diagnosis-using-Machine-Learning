from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medpredict-ai",
    version="1.0.0",
    author="Anesu Manjengwa",
    author_email="manjengwap10@gmail.com",
    description="Advanced Disease Prediction & Diagnosis using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "models": ["*.pkl"],
        "data": ["raw/*.csv"],
    },
    entry_points={
        "console_scripts": [
            "medpredict-train=src.model_training:main",
            "medpredict-serve=main:main",
        ],
    },
    keywords=[
        "machine-learning",
        "healthcare",
        "disease-prediction", 
        "artificial-intelligence",
        "medical-ai",
        "heart-disease",
        "diabetes",
        "streamlit",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning/issues",
        "Source": "https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning",
        "Documentation": "https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning/docs",
    },
)
