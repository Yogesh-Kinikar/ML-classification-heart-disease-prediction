# 🫀 Heart Disease Prediction - Classification Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.2-FF9900.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%3E%3D1.7-66CC00.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

A clean, end‑to‑end machine learning pipeline to **predict heart disease** from clinical attributes.  
It includes EDA, model benchmarking, evaluation reports, and ready‑to‑run code.

---

## 🔎 Problem Statement

Given a patient’s clinical and demographic features (e.g., age, chest pain type, cholesterol, max heart rate), predict whether **heart disease is present**.

- **Task:** Binary classification (`target`: 1 = disease, 0 = no disease)  
- **Objective:** Build an interpretable and accurate model with robust evaluation

---

## 📂 Dataset

- Typical schema (columns may vary with source):

| Feature     | Description (short)                               |
|-------------|---------------------------------------------------|
| `age`       | Age in years                                      |
| `sex`       | 1 = male, 0 = female                              |
| `cp`        | Chest pain type (0–3)                             |
| `trestbps`  | Resting blood pressure (mm Hg)                    |
| `chol`      | Serum cholesterol (mg/dl)                         |
| `fbs`       | Fasting blood sugar > 120 mg/dl (1 = true)        |
| `restecg`   | Resting ECG results                               |
| `thalach`   | Max heart rate achieved                           |
| `exang`     | Exercise induced angina (1 = yes)                 |
| `oldpeak`   | ST depression induced by exercise                 |
| `slope`     | Slope of the peak exercise ST segment             |
| `ca`        | Number of major vessels (0–3) colored by flouros. |
| `thal`      | Thalassemia (0–3 encoded)                         |
| `target`    | 1 = disease, 0 = no disease                       |

---

## 📊 Exploratory Data Analysis (EDA)

Key visuals included in the repo:

- **Age Distribution**
  
| Age group          | Numbers  |  
|--------------------|---------:|
| 29 - 41            | **28**   | 
| 41 - 53            | **99**   | 
| 53 - 65            | **142**  |
| 65 - 77            | **33**   |

<img width="722" height="595" alt="age dist" src="https://github.com/user-attachments/assets/fa83d202-6312-45fe-8fe6-0baa317a5573" />

- **Gender Distribution**

| Gender             | Numbers  |  
|--------------------|---------:|
| Male               | **207**  | 
| Female             | **96**   | 

  
<img width="696" height="501" alt="gender" src="https://github.com/user-attachments/assets/64b543d0-b113-4432-896f-d336fe89eed3" />

- **Cholesterol Distribution**
<img width="707" height="565" alt="cholestrl dist" src="https://github.com/user-attachments/assets/0fc3260b-6d76-488b-b7c1-10db1a202750" />

- **Correlation Heatmap**
<img width="1007" height="710" alt="corelation" src="https://github.com/user-attachments/assets/eb97941b-a36d-46f7-b9d3-baddb7125b58" />

**Insights (high level):**
- `thalach` (max heart rate) and `cp` (chest pain type) tend to have **positive** association with `target`(Heart disease).
- `oldpeak`(ST depression induced by exercise) and `ca`(Number of major vessels (0–3) colored by flouros) show **negative** association with `target`(Heart disease).

---

## 🤖 Models Trained

- Logistic Regression  
- K‑Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost

---

## ✅ Results

### Per‑model Classification Reports (test set)

- **KNN**  
<img width="838" height="791" alt="KNN Neighbours 1" src="https://github.com/user-attachments/assets/3fb0166d-508b-47c9-9c44-cd57f46cd850" />
<img width="577" height="220" alt="KNN Neighbours" src="https://github.com/user-attachments/assets/6851288c-edc1-4b42-a812-5ca551644680" />

  **Accuracy:** `77.05%`

- **Logistic Regression**  
<img width="880" height="821" alt="logistic regression 1" src="https://github.com/user-attachments/assets/2c71416c-ddbe-4357-a657-cebf428d1815" />
<img width="583" height="210" alt="logistic regression results" src="https://github.com/user-attachments/assets/ec9cedaf-e3a1-473e-8b0a-38e1c9d0d03f" />

  **Accuracy:** `80.33%`

- **SVM**  
<img width="892" height="792" alt="SVM classifier 1" src="https://github.com/user-attachments/assets/f650b1bd-3942-4344-9ac2-95606038d26a" />
<img width="573" height="226" alt="SVM classifier" src="https://github.com/user-attachments/assets/699eedbc-aeb2-4cd8-8935-a6e762fa8a70" />

  **Accuracy:** `81.97%`

- **Random Forest**  
<img width="852" height="792" alt="Random Forest classifier 1" src="https://github.com/user-attachments/assets/6c2d64fe-5587-412b-b6d2-1b2811f7dec4" />
<img width="565" height="217" alt="Random Forest classifier" src="https://github.com/user-attachments/assets/6d37456c-2192-4886-96a1-9e6b593b35b5" />

  **Accuracy:** `80.33%`

- **Decision Tree**  
<img width="852" height="793" alt="Decision Tree classifier 1" src="https://github.com/user-attachments/assets/9ed064ec-cc77-4649-8eb5-44b9f01c4f6c" />
<img width="571" height="220" alt="Decision Tree classifier" src="https://github.com/user-attachments/assets/a05d8c5e-d0d9-410c-bb3b-e6bbaa737937" />

  **Accuracy:** `78.69%`

- **XGBoost**  
<img width="765" height="663" alt="XGBOOST 1" src="https://github.com/user-attachments/assets/a9eed762-cf19-4f4e-a9c9-b3eccf40f229" />
<img width="848" height="387" alt="XGBOOST 2" src="https://github.com/user-attachments/assets/83ba834b-ccf8-4d50-8e90-54cbc38e2f07" />
<img width="582" height="222" alt="XGBOOST" src="https://github.com/user-attachments/assets/370541ae-8fd7-4305-9fd3-2dc96250ec64" />

  **Accuracy:** `78.69%`

### 📌 Model Performance Summary

| Model                     | Train Accuracy | Test Accuracy | Precision (1) | Recall (1) |
|---------------------------|---------------:|--------------:|--------------:|-----------:|
| Logistic Regression      | 85.54%         | 80.33%        | 81.8%         | 81.8%      |
| KNN Classifier           | 88.84%         | 77.05%        | 77.1%         | 81.8%      |
| Decision Tree Classifier |   100%         | 78.69%        | 79.4%         | 81.8%      |
| Random Forest Classifier |   100%         | 80.33%        | 83.9%         | 78.8%      |
| SVM                      | **85.95%**     | **81.97%**    | **80.6%**     | **87.9%**  |
| XGBoost                  | 97.52%         | 78.69%        | 79.4%         | 81.8%      |

> ℹ️SVM Classifier stands out with highest Test accuracy(81.97%) and Recall (87.9%) with one of the top precision (80.6%)

> **We tested the SVM model to predict on a dummy feature database and we received 3 correct answers out of 4**

---

## ⚠️ Limitations

- Dataset contains limited rows (~300), which may restrict generalization.
- Some features show correlation overlap (multicollinearity).
- Hyperparameter tuning not deeply explored in this version.
- Slight class imbalance affects precision/recall for class 0.
- Medical dataset may not generalize across demographics or regions.

---

## 📈 Future Improvements

- Apply hyperparameter tuning using GridSearchCV / Optuna
- Build model explainability using SHAP / LIME
- Add cross-validation model comparison
- Engineer additional features for improved accuracy
- Introduce MLflow for experiment tracking
- Deploy model with Streamlit / FastAPI for user interface

---
## 🛠️ Tech Stack

- **Python 3.9+**
- **NumPy**
- **Pandas**
- **Matplotlib & Seaborn**
- **Scikit‑Learn**
- **XGBoost**
- **Jupyter Notebook**

---

## 🪪 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more details.

---

## 👤 Author

**Yogesh Kinikar**  
Analyst (BASES IM)
yogeshsk19.pumba@gmail.com
If you have suggestions or feedback, feel free to open an Issue or Pull Request.

---
