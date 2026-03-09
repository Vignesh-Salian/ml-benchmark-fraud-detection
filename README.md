# ML Benchmark: Credit Card Fraud Detection

This project demonstrates a comprehensive machine learning pipeline for detecting fraudulent credit card transactions. It features an interactive **Streamlit Dashboard** that enables side-by-side comparisons of various ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes, LightGBM) as well as real-time fraud predictions.

## Features
- **Exploratory Data Analysis (EDA)** and Data Preprocessing
- **Handling Imbalanced Data** via SMOTE (Synthetic Minority Over-sampling Technique)
- **Model Benchmarking** across 7 different algorithms
- **Interactive Streamlit Web App** with multiple pages:
  - **Model Comparison:** Displays performance metrics (Accuracy, F1 Score, ROC-AUC) visually.
  - **Fraud Prediction System:** Real-time form for entering transaction features and predicting the likelihood of fraud using the best performing model.

## Evaluation Metrics & Visualizations

### Application Dashboard
| Model Comparison Page | Fraud Prediction Page |
|---|---|
| ![Model Comparison](images/model_comparison_dashboard.png) | ![Fraud Prediction](images/fraud_prediction_dashboard.png) |

### Fraud vs Normal Transactions
![Fraud vs Normal Transactions](images/fraud_vs_normal.png)

### Model Comparison Dashboard
![Model Comparison Dashboard](images/model_comparison.png)

### Model Confusion Matrix (Random Forest)
![Confusion Matrix](images/confusion_matrix.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vignesh-Salian/ml-benchmark-fraud-detection.git
   cd ml-benchmark-fraud-detection
   ```

2. **Set up virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**
   Ensure you have Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM, Seaborn, Matplotlib, and Streamlit installed.
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm seaborn matplotlib streamlit
   ```

4. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
.
├── app.py                # Main Streamlit application
├── data/                 # Directory containing the original dataset (e.g., creditcard.csv)
├── images/               # Directory for storing generated plots and dashboard screenshots
├── models/               # Pre-trained ML models and Scaler file (.pkl)
├── src/                  # Source code containing the Jupyter notebook for training
├── .gitignore            # Gitignore file
└── README.md             # Project documentation
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 👨‍💻 Author

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Vignesh-Salian">
        <img src="https://img.shields.io/badge/Vignesh%20Salian-Developer-blue?style=for-the-badge&logo=github" alt="Vignesh Salian"/>
      </a>
      <br/>
      <a href="https://github.com/Vignesh-Salian">
        <img src="https://img.shields.io/github/followers/Vignesh-Salian?label=Follow&style=social" alt="GitHub Followers"/>
      </a>
    </td>
  </tr>
</table>

---
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>
