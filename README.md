# Loan Default / Credit Risk Prediction using Logistic Regression

A machine learning project that predicts the probability of loan default using applicant financial and credit-related information.

## Project Overview

Loan default is a major risk for banks and financial institutions. Before approving a loan, lenders need to estimate whether an applicant is likely to repay the loan or default.

This project builds a binary classification model using Logistic Regression to estimate the likelihood of loan default based on factors such as:

* Annual income
* Credit score
* Loan amount
* Debt-to-income ratio
* Employment length
* Previous defaults
* Applicant age
* Loan repayment term

The project also includes model evaluation, feature-importance analysis, and an interactive slider-based prediction demo.

## Business Problem

The objective is to answer:

> "Given an applicant's financial and credit information, how likely are they to default on a loan?"

The model produces:

* A predicted class: `Default Risk` or `Likely to Repay`
* An estimated probability of default

In a real lending environment, this probability could be used as one input for loan approval, risk assessment, pricing, or manual review.

## Dataset

The project uses a synthetic dataset containing 2,000 loan applicants.

Real customer credit data is generally private and cannot be freely used for a student project, so the dataset is artificially generated with realistic ranges and relationships between variables.

### Features

| Feature                   | Description                                     |
| ------------------------- | ----------------------------------------------- |
| `age`                     | Applicant age                                   |
| `annual_income_k`         | Annual income in ₹ thousands                    |
| `credit_score`            | Credit score ranging approximately from 300–900 |
| `loan_amount_k`           | Requested loan amount in ₹ thousands            |
| `loan_term_months`        | Loan repayment period                           |
| `employment_length_years` | Years in current employment                     |
| `debt_to_income`          | Debt-to-income ratio                            |
| `previous_defaults`       | Number of previous defaults                     |

### Target

`default`

* `0` = Repaid normally
* `1` = Defaulted

The target variable is generated from a risk score influenced by the applicant's financial characteristics, along with random noise.

## Machine Learning Approach

The project uses Logistic Regression for binary classification.

The workflow is:

```text
Synthetic Applicant Data
        ↓
Exploratory Data Analysis
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Default Probability
        ↓
Classification
        ↓
Model Evaluation
```

### Why Logistic Regression?

Logistic Regression is commonly used as a baseline for credit-risk classification because it:

* Works well for binary classification
* Produces probability estimates
* Is relatively fast to train
* Is easier to interpret than many complex models
* Allows the effect and direction of features to be examined

The model is not intended to represent a production banking credit-scoring system.

## Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of test cases classified correctly.

### ROC-AUC

Measures how well the model distinguishes between default and non-default applicants across different classification thresholds.

### Classification Report

Includes:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

These metrics provide more information than accuracy alone, which is particularly important in risk prediction.

## Feature Impact

Because Logistic Regression produces coefficients, the project examines the direction of each feature's relationship with default risk.

For example, within this synthetic dataset:

* Higher debt-to-income ratio generally increases predicted risk.
* More previous defaults generally increase predicted risk.
* Higher credit scores generally reduce predicted risk.
* Higher income generally reduces predicted risk.
* Larger loan amounts generally increase predicted risk.

The coefficients are calculated after standardizing the features, making their magnitudes more comparable.

## Interactive Prediction Demo

The notebook includes an interactive interface built with `ipywidgets`.

Users can modify:

* Age
* Income
* Credit score
* Loan amount
* Loan term
* Employment length
* Debt-to-income ratio
* Previous defaults

The model then updates the predicted default probability.

Example:

```text
Applicant Information
        ↓
StandardScaler
        ↓
Trained Logistic Regression
        ↓
Default Probability
        ↓
Prediction
```

This makes the project easier to demonstrate without manually editing Python code for every applicant.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* IPyWidgets
* Google Colab / Jupyter Notebook

## Project Structure

```text
loan-default-prediction/
│
├── Loan_Default_Credit_Risk_Prediction.ipynb
└── README.md
```

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Run the installation/import cell.
3. Run the cells from top to bottom.
4. Check the generated dataset and visualizations.
5. Review the model evaluation results.
6. Use the interactive sliders to test hypothetical applicants.

### Option 2: Jupyter Notebook

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets
```

Then open the notebook using Jupyter:

```bash
jupyter notebook
```

Run the notebook cells sequentially.

## Example Prediction

The notebook tests a hypothetical applicant with characteristics such as:

```text
Age: 29
Annual Income: ₹420,000
Credit Score: 610
Loan Amount: ₹350,000
Loan Term: 36 months
Employment Length: 2.5 years
Debt-to-Income Ratio: 0.55
Previous Defaults: 1
```

The trained model returns a predicted class and estimated probability of default.

The exact probability depends on the generated dataset and trained model.

## Responsible AI Considerations

A real credit-risk system would require considerably more work than this demonstration.

Important considerations include:

* Testing for bias across relevant demographic groups
* Monitoring model performance over time
* Handling missing and incorrect financial data
* Preventing data leakage
* Calibrating predicted probabilities
* Choosing an appropriate decision threshold
* Providing appropriate explanations for decisions
* Human review for borderline cases
* Protecting sensitive financial information
* Regularly validating and retraining the model

This project should therefore be treated as an educational prototype, not as a system for making real lending decisions.

## Limitations

The biggest limitation is the synthetic dataset.

Although the variables and relationships are designed to resemble a real credit-risk dataset, the model has not been trained or validated on actual bank/NBFC customer data.

Other limitations include:

* Only 2,000 synthetic records are generated.
* The target variable is artificially created.
* Real-world financial behavior is more complex.
* No external credit bureau data is used.
* No categorical applicant information is included.
* No time-series repayment history is modeled.
* The classification threshold is not optimized for a specific lender's business costs.

Therefore, the performance obtained on this dataset should not be interpreted as real-world lending performance.

## Future Improvements

Possible extensions include:

1. Train the model on a real publicly available credit-risk dataset.
2. Compare Logistic Regression with Random Forest, XGBoost, and other models.
3. Perform hyperparameter tuning.
4. Handle class imbalance using appropriate techniques.
5. Add probability calibration.
6. Optimize the classification threshold based on the cost of false approvals and false rejections.
7. Add explainability using SHAP or similar techniques.
8. Build a web dashboard using Flask, FastAPI, or Streamlit.
9. Store applicant predictions in a database.
10. Add model monitoring and drift detection.

## Key Learning Outcomes

This project demonstrates an end-to-end machine learning workflow:

* Synthetic data generation
* Data preparation
* Exploratory data analysis
* Train/test splitting
* Feature scaling
* Logistic Regression
* Probability prediction
* Classification metrics
* Confusion matrix analysis
* ROC-AUC evaluation
* Model coefficient interpretation
* Interactive ML prediction

## Disclaimer

This project is created for educational and portfolio purposes.

The dataset is synthetic, and the predictions should not be used to approve, reject, price, or otherwise make decisions about real loan applications.

Author
Khushmeet Kaur

Machine Learning | AI | Business Analytics
