# Course-4-End-of-Course Project: Regression Modeling & Machine Learning Workflows

This repository contains hands-on data science labs and end-of-course capstone projects focusing on multiple linear regression, binomial logistic regression, and end-to-end predictive modeling workflows using the PACE (Plan, Analyze, Construct, Execute) framework.

---

## 📂 Repository Contents

### **1. Jupyter Notebooks**

#### [`Activity_Course 5 TikTok project lab.ipynb`](https://github.com/dyyland23-glitch/Course-4-End-of-course-project-Build-a-multiple-linear-regression-model/blob/main/Activity_Course%205%20TikTok%20project%20lab.ipynb)
* **Project Overview**: Demonstrates an end-to-end binary logistic regression workflow to predict user account verification status (`verified_status`) based on video engagement characteristics and metadata. This predictive pipeline helps prioritize user reports and streamline content moderation (differentiating between claims and opinions).
* **Key Tasks & Methods**:
  * **Exploratory Data Analysis (EDA)**: Investigates distribution of engagement metrics (`video_view_count`, `video_like_count`, `video_comment_count`, `video_duration_sec`) and analyzes transcription lengths across account types.
  * **Class Imbalance Handling**: Addresses severe class imbalance (approx. 94.2% unverified vs. 5.8% verified) via minority class upsampling (`resample`).
  * **Feature Engineering & Multicollinearity**: Generates text-length features (`video_transcription_text` length) and evaluates feature relationships using correlation matrices to avoid violating logistic regression assumptions.
  * **Categorical Encoding**: Prepares features (`claim_status`, `author_ban_status`) and targets using one-hot encoding (`OneHotEncoder`).
  * **Model Training & Evaluation**: Fits a `LogisticRegression` classifier, validates on held-out test data, and assesses performance using confusion matrices (`ConfusionMatrixDisplay`), classification reports (precision, recall, F1-score), and log-odds coefficient interpretation.
* **Core Libraries & Key Functions**:
  * `pandas`: `read_csv()`, `isna()`, `drop_duplicates()`, `describe()`, `reset_index()`
  * `numpy`: Array manipulations and transformations (`.ravel()`)
  * `sklearn.utils`: `resample()` (for balancing class distributions)
  * `sklearn.preprocessing`: `OneHotEncoder()` (for categorical feature transformations)
  * `sklearn.model_selection`: `train_test_split()`
  * `sklearn.linear_model`: `LogisticRegression()`
  * `sklearn.metrics`: `confusion_matrix()`, `ConfusionMatrixDisplay()`, `classification_report()`
  * `seaborn` & `matplotlib.pyplot`: `boxplot()`, `histplot()`, `heatmap()`

---

#### [`Activity_ Course 5 Automatidata project lab.ipynb`](https://github.com/dyyland23-glitch/Course-4-End-of-course-project-Build-a-multiple-linear-regression-model/blob/main/Activity_%20Course%205%20Automatidata%20project%20lab.ipynb)
* **Project Overview**: Constructs an Ordinary Least Squares (OLS) multiple linear regression model for the NYC Taxi and Limousine Commission (TLC) to predict continuous taxi fares (`fare_amount`) from trip parameters and location aggregations.
* **Key Tasks & Methods**:
  * **Data Cleaning & Outlier Imputation**: Converts datetime strings, computes trip durations, and imputes upper/lower bound outliers using IQR thresholds ($Q_3 + 6 \times \text{IQR}$).
  * **Route-Level Feature Engineering**: Groups trip combinations (`pickup_dropoff`) to derive aggregated behavioral variables (`mean_distance`, `mean_duration`), rush hour indicators, and temporal attributes.
  * **Model Building & Diagnostics**: Standardizes features using `StandardScaler`, fits multiple linear regression models, and inspects residual distributions to confirm normality and homoscedasticity.
* **Core Libraries & Key Functions**:
  * `pandas`: Datetime conversions (`to_datetime()`), `groupby()`, `map()`, `get_dummies()`
  * `sklearn.preprocessing`: `StandardScaler()`
  * `sklearn.linear_model`: `LinearRegression()`
  * `sklearn.metrics`: `r2_score()`, `mean_absolute_error()`, `mean_squared_error()`
  * `statsmodels`: Regression diagnostics and statistical modeling

---

### **2. Datasets (`.csv`)**
* [`tiktok_dataset.csv`](https://github.com/dyyland23-glitch/Course-4-End-of-course-project-Build-a-multiple-linear-regression-model/blob/main/tiktok_dataset.csv): Contains 19,382 video records with engagement metrics, verification status, author account status, and transcription text.
* [`2017_Yellow_Taxi_Trip_Data.csv`](https://github.com/dyyland23-glitch/Course-4-End-of-course-project-Build-a-multiple-linear-regression-model/blob/main/2017_Yellow_Taxi_Trip_Data.csv): Contains 22,699 trip records with timestamped pickup/dropoff points, trip distances, and fare breakdowns.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dyyland23-glitch/Course-4-End-of-course-project-Build-a-multiple-linear-regression-model.git](https://github.com/dyyland23-glitch/Course-4-End-of-course-project-Build-a-multiple-linear-regression-model.git)
