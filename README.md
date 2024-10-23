# Fraud-Detection-Model-Python
Fraud Detection Model for Credit card Transaction of Bank Customers in Python - Google Colab notebook

# **Overview**

The goal of this project is to accurately detect fraudulent credit card transactions in real-time, minimizing financial losses for banks and protecting customer trust. This project utilizes machine learning techniques, specifically Logistic Regression and Random Forest Classifier, to identify fraudulent activities based on a dataset collected from Kaggle.

# **Dataset Description**

**Collection:** Credit card transactions dataset from Kaggle for fraud detection analysis.

**Features:**

- ```V1 to V28:``` Principal components derived from PCA transformation (confidential)

- ```Time:``` Seconds elapsed between each transaction and the first transaction

- ```Amount:``` Transaction amount, relevant for cost-sensitive learning

- ```Class:``` Response variable (1 for fraud, 0 for non-fraud) - Dependent Variable

Note: The dataset contains anonymized numerical features due to confidentiality, with the majority of features being the result of a PCA transformation.

# **Project Steps**

1. Import Necessary Libraries
   
Import libraries such as pandas, numpy, scikit-learn, matplotlib, seaborn, and imbalanced-learn for data manipulation, visualization, and modeling.
```python
# Data Manipulation Libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations

# Data Visualization Libraries
import matplotlib.pyplot as plt  # For creating static visualizations
import seaborn as sns           # For enhanced data visualization

# Machine Learning Libraries
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.preprocessing import StandardScaler       # For feature scaling to standardize the dataset
from sklearn.ensemble import RandomForestClassifier     # For implementing the Random Forest Classifier
from sklearn.model_selection import cross_val_score, KFold  # For cross-validation and model evaluation
from sklearn.linear_model import LogisticRegression     # For implementing the Logistic Regression model
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_score, recall_score,
                             accuracy_score)  # For model evaluation metrics

# Handling Imbalanced Datasets
from imblearn.over_sampling import SMOTE  # For oversampling minority class to handle class imbalance

# Data Cleaning and Imputation
from sklearn.impute import SimpleImputer  # For handling missing values in the dataset
```     
2. Exploratory Data Analysis (EDA)
   
Data Summary: Check the dataset's summary and class distribution:
- Total Transactions: 51,590
- Fraudulent Transactions (Class = 1): 150
- Feature Insights
- The features V1 to V28 vary widely, with some extreme values.
- The average transaction amount is $94, with significant variability.
- The Time feature shows low correlations with other features, indicating minimal impact on fraud detection.
- Correlation Matrix: Visualize relationships between features to identify anomalies.
```python
# Step 2: Load the dataset
dataset = pd.read_csv('creditcard.csv')

# Step 3: Initial Data Exploration
print(dataset.info())  # Check data types and null values
print(dataset.describe())  # Summary statistics
print(dataset['Class'].value_counts())  # Check imbalance in the target variable
```  
3. Data Preprocessing
 ```python  
- Handle Missing Values: Impute or remove missing values.
- Address Class Imbalance: Use techniques like oversampling, undersampling, or SMOTE.
- Standardization: Scale the features for better model performance.
- Split Dataset: Divide the dataset into training and testing sets.
  # Define the list of features to include in the correlation matrix
features_to_include = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Time']

# Select only the columns of interest
subset = dataset_imputed[features_to_include]

# Calculate the correlation matrix
corr_matrix = subset.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Selected Features")
plt.show()
# Convert hyphens to NaN
dataset.replace('-', np.nan, inplace=True)

# Now impute the missing values with median
imputer = SimpleImputer(strategy='median')
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns) 

#Seperating the independnet features V1 to V28, 'Amount', 'Time' and the dependent variable 'class'
X = dataset_imputed.drop(columns=['Class'])
y = dataset_imputed['Class']

# Standardizing the 'Amount' and 'Time' columns
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# Handling Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Splitting the datset into training and test set for model training & validations
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
```      
     
4. Logistics Regression Model Training and K-Fold Cross-Validation
   
Logistic Regression (LR): 
-Simple and interpretable for binary classification
-Effective with smaller datasets.

- K-Fold Cross-Validation: Conduct cross-validation for reliable performance assessment for LR model to reduce overfitting issue.

Random Forest Classifier:
- Robust and effective for complex datasets.
- Reduces overfitting through ensemble learning.
 ```python   
# Define the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Set up K-Fold Cross-Validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold Cross-Validation and get accuracy scores for each fold
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=kf, scoring='accuracy')
     
# Calculate the average accuracy across the folds
average_cv_accuracy = cv_scores.mean()

# Print the cross-validation accuracy for each fold and the average accuracy
print(f"K-Fold Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Average Cross-Validation Accuracy: {average_cv_accuracy:.4f}")

# Train the Logistic Regression model on the full training set
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)
y_pred_prob_lr = lr_model.predict_proba(X_test)[:, 1]

lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# Misclassification rate
misclassification_rate_lr = 1 - lr_accuracy

# Print the results
print(f"LR model has a precision score of: {lr_precision}")
print(f"LR Model has  recall score of: {lr_recall}")
print(f"\nLR model has an Accuracy on Test Set: {lr_accuracy:.4f}")
print("LR model has a misclassifcation rate of:", misclassification_rate_lr*100, "%")
```      

5. Visualization of Confusion Matrix and Classification Report for Logistic Regression Model

- ROC Curve: Visualize the performance of models.
- Confusion Matrix: Display true positives, true negatives, false positives, and false negatives.
```python 
print("\nLogistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix (Logistic Regression)")
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraudulent', 'Fraudulent'], yticklabels=['Non-Fraudulent', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix LR Model')
plt.show()

# Logistic Regression ROC-AUC
lr_roc_auc = roc_auc_score(y_test, y_pred_prob_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()
``` 

6. Random Forest Classifier Model Training, Evaluation & Visualizations
```python 
# Step 8: Model Training - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
#Prediction on Test set
y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate Precision, Recall, and Accuracy for Random Forest
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Calculate misclassification rate
misclassification_rate_rf = 1 - rf_accuracy

# Print the results
print(f"Random Forest Classifier - Precision: {rf_precision}")
print(f"Random Forest Classifier - Recall: {rf_recall}")
print(f"Random Forest Classifier - Accuracy: {rf_accuracy}")
print("RF model has a misclassifcation rate of:", misclassification_rate_rf*100, "%")

print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest)")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraudulent', 'Fraudulent'], yticklabels=['Non-Fraudulent', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Random Forest Classifier')
plt.show()

# Random Forest ROC-AUC
rf_roc_auc = roc_auc_score(y_test, y_pred_prob_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()
```

7. ROC-AUC Curve comparison for both models Visually
```python 
rf_roc_auc = roc_auc_score(y_test, y_pred_prob_rf)
lr_roc_auc = roc_auc_score(y_test, y_pred_prob_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)

plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_roc_auc:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```
8. Feature Importance
   
Extract and compare feature importance from both models (LR and RF) to identify key drivers of fraud detection.

> LR Model Feature Importance & Visualization
```python 
# Access the coefficients from Logistic Regression
lr_coefficients = lr_model.coef_[0]

# Get the feature names from your dataset
feature_names = X_train.columns

# Print feature names and their coefficients from Logistic Regression
print("\nLogistic Regression Feature Coefficients:")
for feature_name, coefficient in zip(feature_names, lr_coefficients):
    print(f"Feature {feature_name}: Coefficient = {coefficient:.4f}")

# Accessing Feature Importances for Logistic Regression (using absolute coefficients)
lr_coefficients = np.abs(lr_model.coef_[0])  # Coefficients for Logistic Regression

# Create a DataFrame for Logistic Regression feature importances
lr_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lr_coefficients
})

# Sort by importance and get the top 5 features for Logistic Regression
top_lr_features = lr_importances_df.sort_values(by='Importance', ascending=False).head(5)
print("\nTop 5 Features from Logistic Regression:")
print(top_lr_features)

#Plotting Top 5 Feature Importance/Significant co-efficients for LR model
plt.figure(figsize=(10, 6))
plt.barh(top_lr_features['Feature'], top_lr_features['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 5 Features Importance from Logistic Regression')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()
```
> RandomForest Classifier Features Importance & Visualization
```python
# Access the feature importances from Random Forest
rf_feature_importances = rf_model.feature_importances_

# Get the feature names from your dataset
feature_names = X_train.columns

# Print feature names and their importances from Random Forest
print("Random Forest Feature Importances:")
for feature_name, importance in zip(feature_names, rf_feature_importances):
    print(f"Feature {feature_name}: Importance = {importance:.4f}")

# Create a DataFrame for Random Forest feature importances
rf_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_feature_importances
})

# Sort by importance and get the top 5 features for Random Forest
top_rf_features = rf_importances_df.sort_values(by='Importance', ascending=False).head(5)
print("Top 5 Features from Random Forest:")
print(top_rf_features)
```

# **Results Summary & Findings**

- **Logistic Regression Results:**
  
- Precision: 0.984

- Recall: 0.974

- Accuracy: 0.979

- Misclassification Rate: 2.06%

- The model performs very well in terms of precision and recall, indicating it is effective at detecting fraud while minimizing false alarms and missing fraud cases.

- The high accuracy and low misclassification rate reflect overall strong performance. However, given the class imbalance (where fraudulent transactions are rare), precision and recall are more critical metrics for evaluating the model's effectiveness in fraud detection.

- The confusion matrix for the Logistic Regression model in fraud detection provides that True Negatives (Non-Fraudulent correctly identified): 9923 transactions were correctly classified as non-fraudulent, 

- False Positives (Non-Fraudulent incorrectly identified as fraudulent): 155 non-fraudulent transactions were mistakenly classified as fraudulent, 

- False Negatives (Fraudulent incorrectly identified as non-fraudulent): 261 fraudulent transactions were incorrectly classified as non-fraudulent and 

- True Positives (Fraudulent correctly identified): 9856 fraudulent transactions were correctly classified as fraudulent.

- This proves that the model has a strong ability to detect both fraudulent and non-fraudulent transactions, but there are still some misclassifications, particularly with fraudulent transactions being missed (false negatives).

Overall, the LR model is robust for fraud detection, effectively identifying fraudulent transactions while maintaining a low rate of false positives and negatives.

- The ROC curve for the LR model to detect credit card fraud gives almost a perfect performance:

- AUC = 0.97: This indicates a perfect classifier with extremely few errors in distinguishing between fraudulent and non-fraudulent transactions.

- The curve follows the top-left corner with a small curve, meaning both True Positive Rate (TPR) is almost maximized and False Positive Rate (FPR) is almost minimized. Only a few errors there.

This suggests the model performs exceptionally well at detecting fraud.

- **Random Forest Results:**
  
- Precision: 0.9998

- Recall: 1.0

- Accuracy: 0.9999

- Misclassification Rate: 0.0099%

- From the above resuult, it clears that the Random Forest Classifier model for detecting credit card fraud performs exceptionally well.

- With a precision of 0.9998, it almost perfectly identifies fraudulent transactions, minimizing false positives.

- A recall of 1.0 means it catches all actual fraud cases, showing no false negatives. The accuracy of 0.9999 indicates overall excellent performance.

- The very low misclassification rate of 0.0099% further supports its high reliability in detecting fraud.

- The Random Forest model performs exceptionally well, with almost perfect classification (no false negatives and only 2 false positives). This suggests it is highly effective for fraud detection in this dataset.

- The ROC curve for the Random Forest model to detect credit card fraud shows a perfect performance:

- AUC = 1.00: This indicates a perfect classifier with no errors in distinguishing between fraudulent and non-fraudulent transactions.

- The curve follows the top-left corner, meaning both True Positive Rate (TPR) is maximized and False Positive Rate (FPR) is minimized.

The AUC curve plotting suggests the model performs exceptionally well at detecting fraud.

# **Feature Importance**

LR Model: Top 5 important Features are -

- V4 (Coefficient: 2.833): The most influential feature with the highest positive impact on detecting fraud. A higher value increases fraud probability significantly.

- V24 (Coefficient: 1.912): Strong positive effect on fraud detection. Contributes notably to the model's prediction of fraud.

- V6 (Coefficient: 1.620): Also positively affects fraud detection, though less than V4 and V21.

- V25 (Coefficient: 1.432): Significant positive contributor, enhancing the likelihood of fraud detection.

- V21 (Coefficient: 1.403): High positive coefficient indicating that larger transaction amounts increase the fraud detection probability.

- These features have the highest positive coefficients, meaning they play a crucial role in identifying fraudulent transactions in the model.

RF Classifier Model: Top 5 significant feature variables here are -

- V14 (Importance: 0.179): Most important feature for fraud detection in the Random Forest model, contributing significantly to the prediction.

- V3 (Importance: 0.125): Second most important feature, with a notable impact on detecting fraud.

- V4 (Importance: 0.125): Also significant, though slightly less impactful than the top three features.

- V10 (Importance: 0.101): Important feature with a substantial role in the Random Forest’s fraud detection.

- V12 (Importance: 0.081): Contributes to fraud detection, but with a smaller impact compared to the others.

**Conclusion**

Both models exhibit excellent performance, with Random Forest slightly outperforming Logistic Regression in minimizing misclassifications. However, when analyzing the feature importance of both models, the Random Forest classifier provides more nuanced insights, highlighting different variables as critical for detecting fraud compared to the Logistic Regression model. Despite this, V4 stands out as the most influential feature in both models, indicating that this variable plays a crucial role in determining fraudulent activity across different algorithms.

**Recommendation**
- Implement the Random Forest model for fraud detection due to its superior performance.
- Monitor the most influential feature, V4, for anomalies.
- Continuously update models with new data to adapt to evolving fraud patterns.

(Note: Please Check the Google Colab notebook for detailed explanation and graphs of the project.)

# Author

Debolina Dutta

LinkedIn: https://www.linkedin.com/in/duttadebolina/
