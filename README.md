## Exploratory Data Analysis (EDA)

A comprehensive Exploratory Data Analysis (EDA) was performed to understand customer behavior, uncover churn patterns, and guide feature engineering and model selection.  
All visualizations generated during EDA are stored in the **`outputs/` directory**.

---

### 1️⃣ Churn Distribution (Target Balance)

**File:** `outputs/01_churn_distribution.png`  
**Chart Type:** Pie Chart  

This visualization shows the proportion of churned vs retained customers in the dataset. As commonly observed in real-world telecom datasets, the target variable is **imbalanced**, with non-churn customers forming the majority class.

**Key Insight:**
- Class imbalance justifies the use of **SMOTE** and **recall-focused evaluation**
- Accuracy alone would be misleading for churn prediction

---

### 2️⃣ Numerical Feature Distributions

These plots examine the distribution of key numerical features and their relationship with churn.

#### a) Customer Tenure Distribution  
**File:** `outputs/02_dist_tenure.png`

- Strong right skew indicating many short-tenure customers
- Early-lifecycle customers exhibit higher churn probability

#### b) Monthly Charges Distribution  
**File:** `outputs/02_dist_MonthlyCharges.png`

- Churned customers tend to have **higher monthly charges**
- Suggests pricing sensitivity and perceived value mismatch

#### c) Total Charges Distribution  
**File:** `outputs/02_dist_TotalCharges.png`

- Closely correlated with tenure
- Lower total charges often indicate early churn behavior

**Business Insight:**  
Customers with **high monthly charges and low tenure** represent the most vulnerable churn segment.

---

### 3️⃣ Categorical Feature vs Churn Analysis

These bar charts highlight churn behavior across major categorical variables.

#### a) Contract Type  
**File:** `outputs/03_cat_Contract.png`

- **Month-to-month contracts exhibit the highest churn**
- Long-term contracts significantly reduce churn risk

#### b) Internet Service Type  
**File:** `outputs/03_cat_InternetService.png`

- Fiber optic users show higher churn compared to DSL users
- Indicates potential service quality or pricing concerns

#### c) Payment Method  
**File:** `outputs/03_cat_PaymentMethod.png`

- Electronic check users churn more frequently
- Automated or long-term payment methods correlate with retention

#### d) Technical Support  
**File:** `outputs/03_cat_TechSupport.png`

- Customers without tech support churn at much higher rates
- Support services act as a strong retention mechanism

#### e) Online Security  
**File:** `outputs/03_cat_OnlineSecurity.png`

- Customers lacking security add-ons show elevated churn
- Bundled security services improve perceived value

---

### 4️⃣ Correlation Analysis

**File:** `outputs/04_correlation_matrix.png`  
**Chart Type:** Heatmap  

The correlation heatmap visualizes relationships between numerical and engineered features.

**Key Observations:**
- Tenure and Total Charges show strong positive correlation
- Monthly Charges moderately influence churn
- Engineered features improve predictive separability

---

### 📌 EDA Summary

The EDA confirms that customer churn is driven by a combination of:
- Contract flexibility
- Pricing pressure
- Service engagement
- Early customer lifecycle behavior

These insights directly informed **feature engineering, imbalance handling, and model selection**, ensuring the system remains both **predictively strong and business-aligned**.

## 🚀 Live Application (Streamlit Deployment)

The Customer Churn Prediction & Retention Analysis System has been successfully deployed using **Streamlit Cloud**, allowing users to interactively predict churn risk and receive actionable retention insights.

🔗 **Live App URL:**  
https://customer-churn-prediction-and-retention-analysis-system.streamlit.app

---

### 🖥️ Application Features

The deployed web application allows users to:

- Input customer details through an intuitive UI
- Predict **churn probability** in real time
- View **risk categorization** (Low / Medium / High)
- Identify **key factors contributing to churn**
- Receive **data-driven retention recommendations**

---

### 🎯 Business Value of Deployment

- Enables **non-technical stakeholders** to use ML insights
- Supports **proactive customer retention strategies**
- Bridges the gap between **model output and business action**
- Demonstrates end-to-end ML workflow: data → model → deployment

This deployment showcases the project’s readiness for **real-world business use** rather than being limited to offline experimentation.
