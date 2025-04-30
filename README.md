# End-to-End Predictive Maintenance System Using Machine Learning and Edge AI

![Predictive Maintenance](https://img.shields.io/badge/AI-Predictive%20Maintenance-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete predictive maintenance solution for predicting machine failures using sensor data. This system includes data ingestion, preprocessing, model training, edge deployment, and real-time anomaly detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Model Performance](#model-performance)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements an end-to-end predictive maintenance system that uses machine learning to predict equipment failures before they occur. The system works with sensor data from industrial equipment and provides early warnings when a machine is likely to fail in the near future.

By enabling proactive maintenance scheduling, this system helps reduce unexpected downtime, minimize maintenance costs, and extend equipment lifetime.

## ✨ Features

- **Data Collection & Preprocessing**: Comprehensive pipeline for sensor data cleaning and feature engineering
- **Dual Model Approach**:
  - Traditional ML model (XGBoost) for interpretability and efficiency
  - Deep Learning model (LSTM) for capturing complex temporal patterns
- **Edge AI Deployment**: Models optimized for edge devices using TensorFlow Lite and ONNX
- **Real-Time Inference**: Live processing of streaming sensor data with failure predictions
- **Monitoring & Retraining**: Concept drift detection and model retraining strategy
- **Interactive Dashboard**: Real-time visualization of machine health and predictions
- **Performance Evaluation**: Comprehensive metrics for model evaluation and comparison

## 🏗️ System Architecture

The system consists of six main components:

1. **Data Preprocessor**: Cleans, normalizes, and transforms raw sensor data into features
2. **Model Trainer**: Trains and evaluates XGBoost and LSTM models
3. **Edge Optimizer**: Converts models to efficient formats for edge deployment
4. **Inference Engine**: Processes live data streams and generates predictions
5. **Monitoring System**: Tracks model performance and detects concept drift
6. **Dashboard**: Visualizes predictions, alerts, and system status

## 📥 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Pedasoft-Consult/predictive-maintenance-system.git
   cd predictive-maintenance-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the NASA Turbofan Engine Degradation dataset:
   ```bash
   mkdir -p data/raw
   # Download and place the dataset in the data/raw directory
   # Dataset available at: https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
   ```

## 🚀 Usage

### 1. Data Preprocessing

Process the raw data to create features and training datasets:

```bash
python src/data_preprocessing/preprocessing.py
```

### 2. Model Training

Train both the XGBoost and LSTM models:

```bash
python src/model_training/model_training.py
```

### 3. Edge Deployment

Optimize models for edge deployment:

```bash
python src/edge_deployment/edge_deployment.py
```

### 4. Real-Time Inference

Run the real-time inference system:

```bash
python src/real_time_inference/real_time_inference.py
```

### 5. Launch Dashboard

Start the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

## 🧩 Components

### Data Preprocessing Module

The preprocessor handles:
- Missing value imputation
- Sensor data normalization
- Feature engineering (rolling statistics, trends)
- Sequence creation for deep learning
- Train/validation/test splitting

### Model Training Module

Implements and trains:
- XGBoost model for tabular sensor data
- LSTM network for temporal sequences
- Hyperparameter tuning and evaluation

### Edge Deployment Module

Optimizes models for edge devices:
- TensorFlow Lite conversion with quantization
- ONNX conversion for cross-platform compatibility
- Benchmarking of model size, latency and accuracy

### Real-Time Inference System

Provides:
- Sensor data simulation or ingestion
- Real-time prediction with optimized models
- Configurable alert thresholds
- Prediction smoothing to reduce false alarms

### Monitoring and Retraining Module

Implements:
- Performance tracking over time
- Concept drift detection
- Incremental model retraining
- A/B testing for model improvements

### Dashboard

Features:
- Real-time RUL predictions
- Machine health indicators
- Performance metrics visualization
- Sensor data visualization
- System control panel

## 📊 Model Performance

| Metric | XGBoost | LSTM |
|--------|---------|------|
| RMSE   | 15.32   | 12.76 |
| MAE    | 10.45   | 8.93  |
| R²     | 0.83    | 0.88  |

Edge-optimized model performance:

| Model | Format | Size | Avg Inference Time | Memory Usage |
|-------|--------|------|-------------------|--------------|
| XGBoost | ONNX | 0.8 MB | 1.2 ms | 12 MB |
| LSTM | TFLite | 1.2 MB | 3.6 ms | 28 MB |
| LSTM | ONNX | 1.4 MB | 3.8 ms | 32 MB |

## 🖥️ Dashboard

The Streamlit dashboard provides:

- **Status Panel**: Current machine health status
- **RUL Chart**: Real-time visualization of remaining useful life
- **Sensor Data**: Visualization of key sensor readings
- **Performance Metrics**: Model accuracy and error trends
- **Alert History**: Record of previous warnings and alerts
- **System Controls**: Configuration options for the system

## 📁 Project Structure

```
project_root/
├── data/
│   ├── raw/                # Raw dataset files
│   └── processed/          # Preprocessed data
├── notebooks/              # Exploratory analysis notebooks
├── src/
│   ├── data_preprocessing/ # Data preprocessing scripts
│   │   └── preprocessing.py
│   ├── model_training/     # Model training scripts
│   │   └── model_training.py
│   ├── edge_deployment/    # Model optimization for edge
│   │   └── edge_deployment.py
│   ├── real_time_inference/# Real-time inference system
│   │   └── real_time_inference.py
│   └── monitoring/         # Monitoring and retraining
│       └── monitoring_retraining.py
├── models/                 # Trained and optimized models
├── dashboard/              # Dashboard for visualization
│   └── app.py
├── logs/                   # System logs
├── visualizations/         # Generated visualizations
├── docs/                   # Documentation
├── requirements.txt        # Project dependencies
└── README.md
```

## 🔮 Future Improvements

- Unsupervised anomaly detection for novel failure patterns
- Transfer learning for adaptation to new equipment types
- Multi-model ensemble for improved accuracy
- AutoML for automated hyperparameter tuning
- Integration with industrial IoT platforms
- Containerization for easier deployment
- Extended support for additional sensor types

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- NASA Turbofan Engine Degradation Simulation Dataset: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- TensorFlow Lite: https://www.tensorflow.org/lite
- ONNX: https://onnx.ai/
- XGBoost: https://xgboost.readthedocs.io/
- Streamlit: https://streamlit.io/