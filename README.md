# Bank Customer Churn Prediction Model

## Project Overview

This project builds a machine learning model to predict bank customer churn - identifying which customers are likely to exit or leave the bank. By identifying at-risk customers early, the bank can implement targeted retention strategies and improve customer lifetime value.

The model combines both **microeconomic factors** (customer-level data) and **macroeconomic indicators** (broader financial environment) to provide a comprehensive understanding of churn risk.

## Project Goal

Develop a reliable predictive model that:
- Identifies customers at risk of leaving the bank
- Enables early intervention through targeted retention strategies
- Improves decision-making and resource allocation
- Supports planning and revenue forecasting

## Dataset

**File:** `data/botswana_bank_customer_churn.csv`

The dataset contains customer banking information and churn indicators, including:
- Customer demographics
- Banking behavior metrics
- Account information
- Churn status (target variable)

## Project Structure

```
Bank_churn_model/
├── README.md                              # Project documentation
├── data/
│   └── botswana_bank_customer_churn.csv  # Raw dataset
├── notebooks/
│   ├── Data_Exploration.ipynb             # EDA and data analysis
│   └── Macro-intergrated_Random_Forest.ipynb  # Model development and training
└── output/
    ├── preprocessed_data.csv              # Cleaned and processed data
    ├── random_forest.sav                  # Trained Random Forest model
    └── macro_random_forest1.sav           # Production-ready model variant
```

## Workflow

### Phase 1: Environment and Data Setup
- Import required libraries (pandas, scikit-learn, matplotlib, etc.)
- Load and inspect the dataset
- Handle missing values and data quality issues

**Notebook:** Data_Exploration.ipynb

### Phase 2: Exploratory Data Analysis (EDA)
- Examine data structure and distributions
- Identify assumptions and potential issues
- Detect and handle missing data
- Visualize data patterns and relationships
- Inform feature engineering strategy

**Notebook:** Data_Exploration.ipynb

### Phase 3: Feature Engineering
- Prepare and transform variables for modeling
- Encode categorical variables
- Scale or normalize features where needed
- Create derived features combining micro and macroeconomic factors

**Notebook:** Macro-intergrated_Random_Forest.ipynb

### Phase 4: Model Training
- Train Random Forest classifier on prepared data
- Tune hyperparameters for optimal performance
- Validate model using cross-validation
- Compare different model configurations

**Notebook:** Macro-intergrated_Random_Forest.ipynb

### Phase 5: Model Evaluation
- Measure performance using key metrics (accuracy, precision, recall, F1-score, AUC-ROC)
- Generate confusion matrices and ROC curves
- Analyze feature importance
- Assess how well the model identifies churn risk

**Notebook:** Macro-intergrated_Random_Forest.ipynb

## Key Outputs

| File | Description |
|------|-------------|
| `preprocessed_data.csv` | Cleaned and transformed dataset ready for modeling |
| `random_forest.sav` | First trained Random Forest model |
| `macro_random_forest1.sav` | Production-ready Random Forest model with macro indicators |


## How to Use

### 1. Explore the Data
Open and run `notebooks/Data_Exploration.ipynb` to:
- Understand dataset characteristics
- View visualizations and distributions
- Identify data quality issues

### 2. Train the Model
Open and run `notebooks/Macro-intergrated_Random_Forest.ipynb` to:
- Execute the complete pipeline
- Train the Random Forest model
- Evaluate performance on test data
- Save the trained model

### 3. Make Predictions
Load the saved model and apply to new customer data:

```python
import joblib

# Load the trained model
model = joblib.load('output/macro_random_forest1.sav')

# Make predictions on new data
predictions = model.predict(new_customer_data)
churn_probabilities = model.predict_proba(new_customer_data)
```


## Future Improvements

Potential enhancements to the model:
- Experiment with additional algorithms (Gradient Boosting, XGBoost, Neural Networks)
- Incorporate time-series macroeconomic data
- Implement SHAP values for better model interpretability
- Add real-time prediction capabilities
- Develop a pipeline for continuous monitoring
- Test with larger or more recent datasets

## Author

Nancia Mwaramba

## License

[Add your license information here]
