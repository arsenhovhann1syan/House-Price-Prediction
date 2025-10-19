# 🏡 House Prices: Advanced Regression Techniques

This repository contains the solution for the Kaggle competition 'House Prices - Advanced Regression Techniques'. The goal is to predict the final price of residential homes in Ames, Iowa, based on 79 explanatory variables describing almost every aspect of residential homes.

## ⚙️ Project Structure and Methodology

The project follows a standard Data Science workflow, emphasizing feature engineering, data cleaning, and robust modeling techniques.

### 1. Data Cleaning and Preprocessing
- **Missing Value Imputation:** Handled missing values by dropping heavily sparse columns (`PoolQC`, `MiscFeature`, etc.) and imputing the rest. Categorical features were imputed with 'None', and numerical features with the median.
- **Target Transformation:** The target variable (`SalePrice`) was found to be highly skewed. It was **log-transformed** (`np.log1p`) to normalize the distribution, which significantly improves the performance of linear models and gradient boosting methods.

### 2. Feature Engineering
Several new, meaningful features were created to improve predictive power:
- **`TotalSF`:** Total Square Footage of the house (Basement + 1st Floor + 2nd Floor).
- **`HouseAge`:** Age of the house at the time of sale (`YrSold` - `YearBuilt`).
- **`RemodAge`:** Time since the last remodeling (`YrSold` - `YearRemodAdd`).
- **`TotalBath`:** Combined full and half bathrooms.

### 3. Outlier and Redundancy Management
- **Outlier Removal:** Outliers were removed from the training set based on highly important numerical features (e.g., `TotalSF`, `GrLivArea`) using the **Interquartile Range (IQR)** method to ensure the model is not biased by extreme values.
- **Feature Reduction:** - **Variance Threshold:** Removed features with near-zero variance.
    - **Correlation Filtering:** Removed one feature from highly correlated pairs (correlation > 0.9) to prevent multicollinearity.

### 4. Feature Selection
- **Random Forest Importance:** A Random Forest model was trained on the encoded features to rank them by importance. The final modeling stages used only the **Top 11 Most Important Features** to reduce dimensionality and training time.

### 5. Modeling and Evaluation

Multiple regression models were tested and compared, often trained on the log-transformed target:

| Model | Evaluation Method | Key Result |
| :--- | :--- | :--- |
| **Linear Regression** | K-Fold Cross-Validation | Established a stable baseline performance. |
| **Ridge / Lasso** | Cross-Validation (CV) | Used for regularization (L2 and L1) to select optimal `alpha` and improve generalization. |
| **XGBoost Regressor** | Training with Early Stopping | **Selected as the final model.** Gradient Boosting typically yields the best results on this type of structured data. |

### 🚀 Final Result

The final predictions were generated using the **XGBoost model**, trained on the log-transformed `SalePrice`, and transformed back to the original dollar scale (`np.expm1`) for the submission file.

- **Final Submission File:** `submission_xgb_final.csv`
- **Key Metrics (Validation Set):**
    - **RMSE (XGBoost):** [Insert your final RMSE value here, e.g., 28000.00] 
    - **R2 Score (XGBoost):** [Insert your final R2 score here, e.g., 0.9250]

## 🛠️ Requirements

To run this notebook, the following libraries are required:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost