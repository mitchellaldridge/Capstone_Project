# Employee Attrition Prediction & Workforce Segmentation

## Project Overview

Employee attrition can create significant operational and financial challenges through recruitment costs, onboarding time, productivity loss, and the loss of experienced employees. This project evaluates whether machine learning can help identify employees at elevated risk of leaving an organization and whether employee segmentation can improve predictive performance.

The analysis uses a synthetic employee burnout and turnover dataset containing approximately **850,000 employee records**. The project combines supervised classification, feature engineering, model tuning, threshold optimization, model interpretability, and unsupervised clustering.

---

## Objectives

The main objectives of this project were to:

* Predict employee attrition using machine learning
* Compare Logistic Regression, Random Forest, and XGBoost
* Address class imbalance using class weighting and SMOTE
* Identify the strongest predictors of employee attrition
* Optimize the classification threshold for improved recall and F1 score
* Use K-Means clustering to identify employee segments
* Test whether segmentation could improve supervised model performance

---

## Dataset

The project uses the **Employee Burnout and Turnover Prediction** dataset available through Hugging Face.

* Approximately **850,000 observations**
* Binary target variable: `left_company`
* **71.5%** of employees stayed
* **28.5%** of employees left

Predictor variables include information related to:

* Stress
* Burnout
* Satisfaction
* Salary
* Workload
* Overtime
* Employee sentiment
* Tenure
* Job level
* Department
* Collaboration
* Performance

**Dataset:**
[https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k](https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k)

---

## Data Cleaning and Feature Engineering

Initial preparation included reviewing:

* Data types
* Missing values
* Target class distribution
* Duplicate variables
* Low-variation variables
* Potential target leakage

Variables removed because of leakage, redundancy, or lack of predictive usefulness included:

* `employee_id`
* `burnout_risk`
* `turnover_reason`
* `risk_factors_summary`
* `turnover_probability_generated`
* `persona_name`
* `slack_activity`
* `meeting_participation`
* `performance_score`
* `technical_skills`
* `soft_skills`

### Engineered Features

#### Feedback Sentiment

VADER sentiment analysis was applied to `recent_feedback` to create a numerical `feedback_sentiment` score.

#### Underpaid Overtime

`underpaid_overtime` identifies employees who:

* Earn below the overall mean salary
* Work overtime hours

#### Burnout Indicator Score

A composite burnout-related predictor was created using:

```text
stress_level + workload_score - satisfaction_score
```

#### New Employee

`new_employee` identifies employees with fewer than 12 months of tenure.

---

## Preprocessing

Preprocessing was performed using scikit-learn pipelines and `ColumnTransformer`.

* Numerical variables: `StandardScaler`
* Department: One-hot encoding
* Job level: Ordinal encoding
* Boolean variables: Passed through unchanged

Job level was encoded in the following order:

```text
Entry → Mid → Senior → Lead → Manager
```

The dataset was split into:

* **80% training data**
* **20% testing data**

A stratified split was used to preserve the attrition distribution in both sets.

---

## Exploratory Data Analysis

EDA focused on:

* Target class distribution
* Numerical distributions
* Outlier detection
* Correlation analysis
* Attrition differences across predictors
* Department and job-level comparisons

Some of the clearest differences between employees who stayed and left appeared in:

* `stress_level`
* `satisfaction_score`

After redundant features were removed, no remaining predictor had a strong linear correlation with `left_company`. The highest correlation was approximately **0.11**, suggesting that attrition was influenced by multiple interacting variables rather than a single dominant predictor.

---

## Supervised Modeling

The following classification approaches were evaluated:

* Logistic Regression
* Balanced Logistic Regression
* Random Forest
* XGBoost
* XGBoost with SMOTE
* XGBoost with threshold optimization

### Baseline Logistic Regression

The baseline Logistic Regression model achieved approximately **71.5% accuracy**, but predicted every employee as staying.

This resulted in:

* Precision: **0.0%**
* Recall: **0.0%**
* F1 Score: **0.0%**

This demonstrated why accuracy alone was not appropriate for the imbalanced target.

### Class Balancing

Balanced class weights were used with Logistic Regression and Random Forest.

XGBoost used `scale_pos_weight`, calculated from the ratio of employees who stayed to employees who left.

This substantially improved the models' ability to identify actual leavers.

---

## Hyperparameter Tuning

Random Forest and XGBoost were tuned using:

* `RandomizedSearchCV`
* 10 randomly sampled parameter combinations
* Stratified 5-fold cross-validation
* F1 score as the model-selection metric

Five-fold cross-validation provided a balance between estimate stability and computational cost while keeping each validation fold large and representative.

---

## Model Performance

| Model                        | Accuracy | Precision |    Recall |  F1 Score | ROC-AUC |
| ---------------------------- | -------: | --------: | --------: | --------: | ------: |
| Logistic Regression          |    71.5% |      0.0% |      0.0% |      0.0% |   58.0% |
| Balanced Logistic Regression |    52.2% |     32.9% |     64.8% |     43.6% |   58.0% |
| Tuned Random Forest          |    53.0% |     33.0% |     63.1% |     43.4% |   58.0% |
| Tuned XGBoost                |    51.8% |     32.8% |     66.0% |     43.9% |   58.1% |
| XGBoost + SMOTE              |    55.4% |     32.9% |     53.9% |     40.8% |   56.6% |
| XGBoost, Threshold 0.40      |    37.0% |     30.1% | **91.0%** | **45.2%** |   58.1% |

---

## Threshold Optimization

The tuned XGBoost model was evaluated across classification thresholds from **0.10 to 0.90**.

### Default Threshold: 0.50

* Recall: **66.0%**
* F1 Score: **43.9%**
* False Negatives: **16,478**
* False Positives: **65,517**

### Selected Threshold: 0.40

* Recall: **91.0%**
* F1 Score: **45.2%**
* False Negatives: **4,378**
* False Positives: **102,681**

Lowering the classification threshold significantly reduced the number of actual leavers missed by the model, but also increased the number of employees incorrectly flagged as likely to leave.

This makes the final model more appropriate for **risk screening** than for automated decision-making.

---

## Feature Importance

Both XGBoost and Random Forest identified `stress_level` as the strongest predictor.

Other important predictors included:

* `burnout_indicator_score`
* `satisfaction_score`
* `email_sentiment`
* `workload_score`
* `overtime_hours`
* `collaboration_score`
* `tenure_months`

Feature importance reflects contribution to model predictions, but does not establish causation or indicate the direction of a relationship.

---

## K-Means Employee Segmentation

K-Means clustering was used to determine whether employees formed meaningful groups based on numerical characteristics.

### Cluster Selection

* Numerical predictors were standardized before clustering
* The elbow method was used to evaluate the number of clusters
* Silhouette analysis was performed on a 40,000-row sample
* Four clusters were selected
* PCA was used only for visualization

Cluster attrition rates ranged from approximately:

**21.6% to 33.1%**

One of the most notable findings was that the segment with the lowest attrition rate also had substantially lower stress levels than the other groups.

---

## Did Segmentation Improve Prediction?

Two approaches were tested:

1. Adding cluster membership as an XGBoost predictor
2. Training separate XGBoost models within individual clusters

Neither approach produced meaningful improvements over the global model.

The segmentation results were therefore more useful for **descriptive employee profiling** than for improving predictive performance.

---

## Key Findings

* Baseline accuracy was misleading because of class imbalance
* Class weighting was more effective than SMOTE
* XGBoost produced the strongest default-threshold performance
* Lowering the threshold to 0.40 increased recall to **91.0%**
* Higher recall came at the cost of substantially more false positives
* Stress and burnout-related predictors were consistently influential
* K-Means identified employee groups with different attrition patterns
* Segmentation did not improve supervised prediction
* ROC-AUC remained near **58%**, indicating limited overall class separation

---

## Business Application

The final model is best viewed as an **employee attrition risk-screening tool**.

A potential workflow could be:

```text
Employee Data
      ↓
Attrition Risk Model
      ↓
Higher-Risk Employees Flagged
      ↓
Additional Information Gathering
      ↓
Confidential Survey / Manager Review
      ↓
Supportive Retention Actions
```

Rather than automatically determining that an employee will leave, model predictions could be used to prioritize additional information gathering related to:

* Stress
* Burnout
* Satisfaction
* Workload
* Career development
* Employee engagement

---

## Limitations

* ROC-AUC remained near **58%**
* Precision remained relatively low
* Lower thresholds produced many false positives
* The dataset is synthetic and may not represent real organizational behavior
* Important real-world attrition drivers may not be included
* Predictions should not be used for automated employment decisions

---

## Future Work

Potential extensions include:

* Validate the model using real organizational data
* Evaluate performance across departments and job levels
* Add employee engagement and manager-feedback variables
* Evaluate probability calibration
* Explore SHAP or permutation importance
* Test cost-sensitive threshold selection
* Monitor model drift over time
* Compare additional classification approaches

---

## Repository Structure

```text
employee-attrition-project/
│
├── notebooks/
│   ├── 01_EDA_Feature_Engineering.ipynb
│   ├── 02_Supervised_Modeling.ipynb
│   └── 03_KMeans_Segmentation.ipynb
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* imbalanced-learn
* NLTK / VADER
* Matplotlib
* K-Means
* PCA

---

## References

BrotherTony. *Employee Burnout & Turnover Prediction 800k*. Hugging Face.
[https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k](https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k)

Govindarajan, R., Komal, K., Sudhakar, R., Pravallika, S., B, D., & G, P. K. (2025). Predicting employee attrition: A comparative analysis of machine learning models using the IBM Human Resource Analytics Dataset. *Procedia Computer Science, 258*, 4084–4093.
[https://doi.org/10.1016/j.procs.2025.04.659](https://doi.org/10.1016/j.procs.2025.04.659)

Iparraguirre-Villanueva, O., Chauca-Huete, L., Prieto-Chavez, R., & Paulino-Moreno, C. (2024). Employee attrition prediction using machine learning models. *Proceedings of the 22nd LACCEI International Multi-Conference for Engineering, Education and Technology*.
[https://doi.org/10.18687/laccei2024.1.1.498](https://doi.org/10.18687/laccei2024.1.1.498)
