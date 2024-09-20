# Fraud-Detection-Model-Python
Fraud Detection Model for Credit card Transaction of Bank Customers in Python - Google Colab notebook

# **Overview**

The goal of this project is to accurately detect fraudulent credit card transactions in real-time, minimizing financial losses for banks and protecting customer trust. This project utilizes machine learning techniques, specifically Logistic Regression and Random Forest Classifier, to identify fraudulent activities based on a dataset collected from Kaggle.

# **Dataset Description**

**Collection:** Credit card transactions dataset from Kaggle for fraud detection analysis.

**Features:**

- V1 to V28: Principal components derived from PCA transformation (confidential)

- Time: Seconds elapsed between each transaction and the first transaction

- Amount: Transaction amount, relevant for cost-sensitive learning

- Class: Response variable (1 for fraud, 0 for non-fraud) - Dependent Variable

Note: The dataset contains anonymized numerical features due to confidentiality, with the majority of features being the result of a PCA transformation.

# **Project Steps**

1. Import Necessary Libraries
Import libraries such as pandas, numpy, scikit-learn, matplotlib, seaborn, and imbalanced-learn for data manipulation, visualization, and modeling.

2. Exploratory Data Analysis (EDA)
Data Summary: Check the dataset's summary and class distribution:
- Total Transactions: 51,590
- Fraudulent Transactions (Class = 1): 150
- Feature Insights
- The features V1 to V28 vary widely, with some extreme values.
- The average transaction amount is $94, with significant variability.
- The Time feature shows low correlations with other features, indicating minimal impact on fraud detection.
- Correlation Matrix: Visualize relationships between features to identify anomalies.

3. Data Preprocessing
   
- Handle Missing Values: Impute or remove missing values.
- Address Class Imbalance: Use techniques like oversampling, undersampling, or SMOTE.
- Standardization: Scale the features for better model performance.
- Split Dataset: Divide the dataset into training and testing sets.
  
5. Model Selection and Evaluation
   
Logistic Regression (LR): 
-Simple and interpretable for binary classification
-Effective with smaller datasets.

- K-Fold Cross-Validation: Conduct cross-validation for reliable performance assessment for LR model to reduce overfitting issue.

Random Forest Classifier:
- Robust and effective for complex datasets.
- Reduces overfitting through ensemble learning.
  
5. Model Evaluation

Metrics: Evaluate models using Precision, Recall, Accuracy, and AUC curve.

6. Feature Importance Analysis
Extract and compare feature importance from both models (LR and RF) to identify key drivers of fraud detection.

7. Visualization
ROC Curve: Visualize the performance of models.
Confusion Matrix: Display true positives, true negatives, false positives, and false negatives.

# **Results Summary & Findings**

- **Logistic Regression Results:**
  
Precision: 0.984

Recall: 0.974

Accuracy: 0.979

Misclassification Rate: 2.06%

The model performs very well in terms of precision and recall, indicating it is effective at detecting fraud while minimizing false alarms and missing fraud cases.

The high accuracy and low misclassification rate reflect overall strong performance. However, given the class imbalance (where fraudulent transactions are rare), precision and recall are more critical metrics for evaluating the model's effectiveness in fraud detection.

The confusion matrix for the Logistic Regression model in fraud detection provides that True Negatives (Non-Fraudulent correctly identified): 9923 transactions were correctly classified as non-fraudulent, False Positives (Non-Fraudulent incorrectly identified as fraudulent): 155 non-fraudulent transactions were mistakenly classified as fraudulent, False Negatives (Fraudulent incorrectly identified as non-fraudulent): 261 fraudulent transactions were incorrectly classified as non-fraudulent and True Positives (Fraudulent correctly identified): 9856 fraudulent transactions were correctly classified as fraudulent.

This proves that the model has a strong ability to detect both fraudulent and non-fraudulent transactions, but there are still some misclassifications, particularly with fraudulent transactions being missed (false negatives).

Overall, the LR model is robust for fraud detection, effectively identifying fraudulent transactions while maintaining a low rate of false positives and negatives.

The ROC curve for the LR model to detect credit card fraud gives almost a perfect performance:

AUC = 0.97: This indicates a perfect classifier with extremely few errors in distinguishing between fraudulent and non-fraudulent transactions.

The curve follows the top-left corner with a small curve, meaning both True Positive Rate (TPR) is almost maximized and False Positive Rate (FPR) is almost minimized. Only a few errors there.

This suggests the model performs exceptionally well at detecting fraud.

- **Random Forest Results:**
  
Precision: 0.9998

Recall: 1.0

Accuracy: 0.9999

Misclassification Rate: 0.0099%

From the above resuult, it clears that the Random Forest Classifier model for detecting credit card fraud performs exceptionally well.

With a precision of 0.9998, it almost perfectly identifies fraudulent transactions, minimizing false positives.

A recall of 1.0 means it catches all actual fraud cases, showing no false negatives. The accuracy of 0.9999 indicates overall excellent performance.

The very low misclassification rate of 0.0099% further supports its high reliability in detecting fraud.

the Random Forest model performs exceptionally well, with almost perfect classification (no false negatives and only 2 false positives). This suggests it is highly effective for fraud detection in this dataset.

The ROC curve for the Random Forest model to detect credit card fraud shows a perfect performance:

AUC = 1.00: This indicates a perfect classifier with no errors in distinguishing between fraudulent and non-fraudulent transactions.

The curve follows the top-left corner, meaning both True Positive Rate (TPR) is maximized and False Positive Rate (FPR) is minimized.

This suggests the model performs exceptionally well at detecting fraud.

# **Feature Importance**

LR Model: Top 5 important Features are -

V4 (Coefficient: 2.833): The most influential feature with the highest positive impact on detecting fraud. A higher value increases fraud probability significantly.

V24 (Coefficient: 1.912): Strong positive effect on fraud detection. Contributes notably to the model's prediction of fraud.

V6 (Coefficient: 1.620): Also positively affects fraud detection, though less than V4 and V21.

V25 (Coefficient: 1.432): Significant positive contributor, enhancing the likelihood of fraud detection.

V21 (Coefficient: 1.403): High positive coefficient indicating that larger transaction amounts increase the fraud detection probability.

These features have the highest positive coefficients, meaning they play a crucial role in identifying fraudulent transactions in the model.

RF Classifier Model: Top 5 significant feature variables here are -

V14 (Importance: 0.179): Most important feature for fraud detection in the Random Forest model, contributing significantly to the prediction.

V3 (Importance: 0.125): Second most important feature, with a notable impact on detecting fraud.

V4 (Importance: 0.125): Also significant, though slightly less impactful than the top three features.

V10 (Importance: 0.101): Important feature with a substantial role in the Random Forest’s fraud detection.

V12 (Importance: 0.081): Contributes to fraud detection, but with a smaller impact compared to the others.

**Conclusion**

Both models exhibit excellent performance, with Random Forest slightly outperforming Logistic Regression in minimizing misclassifications. However, when analyzing the feature importance of both models, the Random Forest classifier provides more nuanced insights, highlighting different variables as critical for detecting fraud compared to the Logistic Regression model. Despite this, V4 stands out as the most influential feature in both models, indicating that this variable plays a crucial role in determining fraudulent activity across different algorithms.

**Recommendation**
- Implement the Random Forest model for fraud detection due to its superior performance.
- Monitor the most influential feature, V4, for anomalies.
- Continuously update models with new data to adapt to evolving fraud patterns.
