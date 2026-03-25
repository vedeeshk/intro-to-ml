# Human Activity Recognition with CNN, LSTM, and Hybrid Models

This project investigates machine learning approaches for **Human Activity Recognition (HAR)** using smartphone inertial sensor data. It compares classical machine learning methods with deep learning architectures, and explores hybrid models that combine learned representations with nonlinear classifiers.

The work is based on the **UCI Human Activity Recognition (HAR) dataset** and follows a **subject-independent evaluation protocol**, ensuring realistic generalisation to unseen users.

---

## 📌 Overview

Human Activity Recognition (HAR) uses sensor data from smartphones to classify physical activities such as walking, sitting, or standing.

This project:
- Uses **raw sensor signals** (no handcrafted features)
- Compares **classical ML vs deep learning**
- Implements a **full reproducible pipeline**
- Explores a **CNN–Random Forest hybrid**, which achieves the best performance

---

## 🧠 Models Implemented

### Classical Baselines
- **Logistic Regression**
  - Linear baseline on flattened input
- **Random Forest**
  - Nonlinear ensemble model

### Deep Learning Models
- **1D Convolutional Neural Network (CNN)**
  - Extracts local temporal patterns
- **LSTM**
  - Models sequential dependencies
- **CNN–LSTM Hybrid**
  - Combines convolutional feature extraction with sequence modelling

### Hybrid Model
- **CNN–Random Forest (CNN–RF)**
  - CNN used as a feature extractor
  - Random Forest used as classifier
  - Achieves the best performance

---

## 📊 Results Summary

| Model                  | Accuracy | Macro-F1 |
|------------------------|----------|----------|
| Logistic Regression    | 0.3156   | 0.30     |
| Random Forest          | 0.8703   | 0.87     |
| 1D-CNN                 | 0.8546 ± 0.0312 | 0.88 |
| LSTM                   | 0.6290 ± 0.0033 | 0.54 |
| CNN–LSTM               | 0.6540 ± 0.0154 | 0.65 |
| **CNN–RF Hybrid**      | **0.9119 ± 0.0092** | **0.92** |

### Key Findings
- Linear models perform poorly on raw time-series data
- Random Forest is a strong classical baseline
- CNN performs best among neural models
- LSTM-based models underperform on short windows
- **CNN–RF hybrid achieves highest accuracy (~92%)**

---

## 📁 Project Structure
intro-to-ml/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── pipeline/
│   │   ├── acquire.py
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── cnn1d.py
│   │   ├── lstm.py
│   │   ├── cnn_lstm.py
│   │   └── cnn_rf.py
│   ├── train/
│   │   └── trainer.py
│   ├── eval/
│   │   └── evaluate.py
│   └── utils/
│       └── seed.py
└── run.py

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/vedeeshk/intro-to-ml.git
cd intro-to-ml
```

### 2. Create environment (recommended)
```bash
conda create -n har python=3.10
conda activate har
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

The entire pipeline is controlled through a **single entry point**:

```bash
python run.py <stage>
```

### Available stages

#### Data processing
```bash
python run.py preprocess
```

#### Training models
```bash
python run.py train --model cnn
python run.py train --model lstm
python run.py train --model cnn_lstm
python run.py train --model cnn_rf
```

#### Evaluation
```bash
python run.py evaluate --model cnn --model_path artifacts/models/cnn1d_seed999.pt
```

Outputs include:
- Accuracy
- Confusion matrices
- Classification reports

---

## 📦 Data

This project uses the **UCI Human Activity Recognition dataset**:

- 30 subjects
- 6 activity classes:
  - Walking  
  - Walking upstairs  
  - Walking downstairs  
  - Sitting  
  - Standing  
  - Laying  
- 50 Hz sampling rate  
- Window size: 128 timesteps  
- 6 channels (accelerometer + gyroscope)  

### Processing steps
- Merge original train/test splits  
- Perform **subject-wise split**  
- Apply **channel-wise normalisation**  
- Convert to tensor format `(C, T)`  

---

## 📈 Evaluation

Models are evaluated using:
- Accuracy  
- Macro-averaged F1 score  
- Confusion matrices  
- Precision and recall per class  

### Key observations
- Dynamic activities are easier to classify  
- Static activities (sitting vs standing) are harder  
- Hybrid models reduce confusion in static classes  

---

## 🔬 Reproducibility

- All experiments use fixed seeds:
  - `42`, `123`, `999`  
- Best model selected using validation set  
- Final evaluation performed on held-out test set  

## 📚 References

- UCI HAR Dataset  
- Ordonez & Roggen (2016) — DeepConvLSTM  
- Hammerla et al. (2016) — Deep Learning for HAR  

---

## 👨‍💻 Authors

- 22000557  
- 22085358  
- 22023456  
- 22155918  

GitHub: https://github.com/vedeeshk/intro-to-ml

---

## 📜 License

This project is released for academic and educational use.