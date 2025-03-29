# Comparing-Classifiers

## Business Objective

The objective of this project is to optimize the predictive performance of various classifiers for predicting the success of a bank marketing campaign. The campaign aims to predict whether a customer will subscribe to a product based on their characteristics (e.g., demographic data, past behavior). The classifiers compared in this project are:

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Trees
4. Support Vector Machines (SVM)

By comparing the performance of these models, we aim to determine the best classification model to accurately predict customer responses and improve the bank's marketing campaign's effectiveness.

## Dataset
The dataset used for this project is from the UCI Machine Learning repository related to marketing bank products over the telephone. 
  
  Data: <ins>bank-additional-full.xls</ins>

The features include various customer attributes such as:
Demographic information (e.g., age, job, marital status)
Contact-related information (e.g., duration of the call, contact communication method)
Previous marketing campaign results (e.g., whether the customer subscribed to a product in a past campaign)
The target variable is whether or not the customer will subscribe to the product during the current marketing campaign.

## Project Workflow
### Step 1: Understanding the Data
In this step, we familiarize ourselves with the dataset by examining its structure, features, and overall composition. This includes checking for any inconsistencies, such as missing values or erroneous data entries, to ensure that we can proceed with a clean dataset for analysis.

### Step 2: Load the Data
Import the dataset into our environment, typically using pandas library. This will allow us to manipulate and explore the data, preparing it for analysis and model training.

### Step 3: Understanding the Features
Identify the various features of the dataset, which might include customer demographics, previous interactions, and marketing campaign results. We also check for any missing values and identify whether any features need to be converted to different data types (e.g., categorical variables needing encoding or continuous variables requiring scaling).

### Step 4: Understanding the Task
The primary task is to compare the performance of classifiers in predicting marketing success for bank products. The goal is to optimize the predictive performance of classifiers like K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines, ultimately enhancing a marketing campaign for selling bank products over the telephone. By choosing the most effective classifier, we aim to improve customer targeting, increase conversion rates, and reduce marketing costs.

### Step 5: Engineering Features and Visualization
After understanding the business objective, we proceed to feature engineering and visualization. This step involves preparing the dataset for modeling, which may include encoding categorical variables, scaling numerical features, and transforming the data as necessary. Visualization is used to explore the relationships between features and the target variable, providing insights that can guide model selection.

### Step 6: Train/Test Split
Split the data into training and test sets. The training set is used to build the model, while the test set will evaluate the model's performance on unseen data. This ensures that we can assess how well the model generalizes to new data.

### Step 7: A Baseline Model
Establish a baseline model to make simple predictions based on basic heuristics or assumptions (e.g., predicting the majority class). This serves as a reference point for evaluating the performance of more sophisticated models.

### Step 8: A Simple Model - Logistic Regression
Build and train a Logistic Regression model, a powerful classifier for binary outcomes. This step allows us to evaluate the performance of a simple model before experimenting with more complex ones.

### Step 9: Score the Model
Once we train the model, we score it using various performance metrics (such as accuracy, precision, recall, and F1-score) on both the training and test sets. This helps us understand how well the model is performing and identify areas for improvement.

### Step 10: Model Comparisons
Finally, we compare the performance of multiple classifiers (e.g., KNN, Decision Trees, SVM) to determine which one performs best for this task. We also evaluate computational efficiency (training time and prediction time) to identify models that are both effective and efficient, ensuring optimal performance in real-world applications.

### Step 11: Improving the Model
Once we built and evaluated the initial models, the next step is to improve the model to achieve better performance. 

                           Train Time  Accuracy  Precision    Recall  F1-Score
Logistic Regression          0.247245  0.909444   0.667257  0.403209  0.502667
KNN                          0.015698  0.900583   0.577957  0.459893  0.512210
Decision Tree                0.222385  0.886502   0.500000  0.514439  0.507116
SVM                         15.706445  0.894513   0.603125  0.206417  0.307570
KNN (GridSearchCV)          14.034704  0.907138   0.615489  0.484492  0.542190
SVM (GridSearchCV)         837.827642  0.906167   0.656371  0.363636  0.467997
Logistic Regression (RFE)    0.031380  0.896455   0.654135  0.186096  0.289759


