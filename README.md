# Hotel Cancellation Prediction

### Project Overview

This project aims to predict **hotel booking cancellations** using a rich dataset of booking information. By analyzing factors such as lead time, stay duration, guest demographics, deposit type, and special requests, the goal is to develop a machine learning model that can accurately forecast which bookings are likely to be canceled. This enables hotels to optimize their overbooking strategies, manage revenue, and improve overall operational efficiency.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Hotel Bookings Analysis Dataset](https://www.kaggle.com/datasets/qucwang/hotel-bookings-analysis-dataset)
  * **Size**: 119390 entries, 32 columns. After data cleaning (dropping duplicates and columns with high missing values), the dataset size is reduced.
  * **Key Features**:
      * lead\_time, arrival\_date\_year, arrival\_date\_month, stays\_in\_weekend\_nights, stays\_in\_week\_nights, adults, children, babies, meal, country, market\_segment, distribution\_channel, is\_repeated\_guest, previous\_cancellations, previous\_bookings\_not\_canceled, reserved\_room\_type, assigned\_room\_type, booking\_changes, deposit\_type, days\_in\_waiting\_list, customer\_type, adr, required\_car\_parking\_spaces, total\_of\_special\_requests, reservation\_status, reservation\_status\_date.
  * **Approach**:
      * Data Cleaning: Dropped duplicates (31994 found). Dropped columns with more than 50% missing values (`company` and `agent`). The remaining columns with null values (`children`, `country`) were implicitly handled by Label Encoding.
      * Exploratory Data Analysis: Histograms, Boxplots, and a heatmap were used for visualization.
      * Label Encoding: Applied to all columns, including categorical features and the target `is_canceled`.
      * Binary Classification: The target variable `is_canceled` indicates whether a booking was canceled (`1`) or not (`0`). The original dataset is imbalanced but this was not explicitly handled in the provided notebook.
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * 100% with XGBoost, Random Forest, AdaBoost, Gradient Boosting, Bagging, and Decision Tree.
      * 98.8% with Ridge Classifier.
      * The extremely high accuracies for most models suggest potential data leakage. Features like `reservation_status` or `reservation_status_date` are often directly indicative of the target variable and should be handled with care or excluded to build a truly predictive model.

-----

### Purpose and Applications

  * Enable hotels to **predict booking cancellations** and manage their inventory more effectively.
  * Implement dynamic pricing or targeted offers to reduce the likelihood of cancellations.
  * Optimize revenue management by anticipating no-shows.
  * Support data-driven decision-making in marketing and operations.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Hotel-Cancelation-Prediction-Using-ML.git
cd Hotel-Cancelation-Prediction-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * **Investigating and addressing the issue of data leakage**, particularly by carefully selecting features that are available at the time of booking, not after the cancellation status is known.
  * Performing comprehensive hyperparameter tuning and cross-validation for all models to ensure robustness.
  * Exploring advanced feature engineering techniques, such as combining time-related features or creating dummy variables for categorical data.
  * Adding explainability (e.g., SHAP or LIME) to understand which factors genuinely contribute to a booking being canceled.
