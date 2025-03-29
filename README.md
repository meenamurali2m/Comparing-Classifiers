# Comparing-Classifiers

Jupyter Notebook - https://github.com/meenamurali2m/Comparing-Classifiers/blob/main/prompt_III_MM.ipynb

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
Import the dataset into our environment, using pandas library. This allows us to manipulate and explore the data, preparing it for analysis and model training.

### Step 3: Understanding the Features
Identify the various features of the dataset, which include customer demographics, previous interactions, and marketing campaign results. We also check for any missing values and identify whether any features need to be converted to different data types (e.g., categorical variables needing encoding or continuous variables requiring scaling).

### Step 4: Understanding the Task
The primary task is to compare the performance of classifiers in predicting marketing success for bank products. The goal is to optimize the predictive performance of classifiers like K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines, ultimately enhancing a marketing campaign for selling bank products over the telephone. By choosing the most effective classifier, we aim to improve customer targeting, increase conversion rates, and reduce marketing costs.

### Step 5: Engineering Features and Visualization
After understanding the business objective, we proceed to feature engineering and visualization. This step involves preparing the dataset for modeling, which includes encoding categorical variables, scaling numerical features, and transforming the data as necessary. Visualization is used to explore the relationships between features and the target variable, providing insights that can guide model selection.

![Image](https://github.com/user-attachments/assets/b7774a1d-ec1d-4e7e-87bd-7bc7453d4d6e)

![Image](https://github.com/user-attachments/assets/3113d693-69b2-4113-b29b-ecaf2d47cda5)



### Step 6: Train/Test Split
Split the data into training and test sets. The training set is used to build the model, while the test set will evaluate the model's performance on unseen data. This ensures that we can assess how well the model generalizes to new data.

### Step 7: A Baseline Model
Establish a baseline model to make simple predictions based on basic heuristics or assumptions (e.g., predicting the majority class). This serves as a reference point for evaluating the performance of more sophisticated models.

### Step 8: A Simple Model - Logistic Regression
Build and train a Logistic Regression model, a powerful classifier for binary outcomes. This step allows us to evaluate the performance of a simple model before experimenting with more complex ones.

### Step 9: Score the Model
Once we train the model, we score it using various performance metrics (such as accuracy, precision, recall, and F1-score) on both the training and test sets. This helps us understand how well the model is performing and identify areas for improvement.

#### Accuracy: 0.9092

### Step 10: Model Comparisons
Finally, we compare the performance of multiple classifiers (e.g., KNN, Decision Trees, SVM) to determine which one performs best for this task. We also evaluate computational efficiency (training time and prediction time) to identify models that are both effective and efficient, ensuring optimal performance in real-world applications.

<img width="364" alt="Image" src="https://github.com/user-attachments/assets/57ad5141-51fa-4891-bc81-71c930081679" />

### Step 11: Improving the Model
Once we built and evaluated the initial models, the next step is to improve the model to achieve better performance. 

<img width="401" alt="Image" src="https://github.com/user-attachments/assets/0664f3de-86da-4452-b8b4-475f486dc7bd" />

![Image](https://github.com/user-attachments/assets/d67dfdfd-7a4f-450e-b9c6-5f0aad3a37d3)

![Image](https://github.com/user-attachments/assets/1ae165fa-184a-4954-ae2a-4ea123d4a5b1)


### Conclusion
#### Key Findings:
##### Model Performance:

We compared Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM), examining their ability to predict customer subscription outcomes. Through cross-validation and performance metrics like accuracy, precision, recall, and F1-score, we identified the models that performed well and those that needed improvement.

##### Feature Engineering:

We explored the importance of feature engineering, which included encoding categorical variables, scaling numerical features, and addressing missing values. This was crucial for ensuring that the data fed into the models was in an optimal format for prediction.

Feature selection techniques like Recursive Feature Elimination (RFE) and regularization were used to remove irrelevant or redundant features, enhancing the models' interpretability and reducing overfitting.

##### Baseline Model and Comparisons:

We established a baseline by using a simple model and compared it to more complex classifiers. The performance improvements from complex models helped us understand how different classifiers handle the given problem.

The KNN, Logistic Regression, Decision Tree, and SVM models were evaluated in terms of training time and test accuracy, providing a clear picture of which models were the most efficient and effective for this problem.

##### Model Tuning and Improvement:

We applied hyperparameter tuning, cross-validation, and ensemble methods like Random Forest and Boosting to further improve model performance.

Techniques like SMOTE were also used to address class imbalances in the target variable, ensuring that the models learned from both the majority and minority classes effectively.

#### Final Thoughts:
The goal was to identify the best-performing model to predict customer subscription to a bank product. By selecting the most efficient classifier, marketing teams can improve customer targeting, resulting in higher conversion rates, reduced marketing costs, and more effective campaigns.

The ability to select the optimal model for predicting success can lead to more personalized marketing strategies, better resource allocation, and ultimately, greater customer satisfaction and revenue for the bank.

The project demonstrated the importance of data preprocessing, feature engineering, model comparison, and hyperparameter tuning in building robust predictive models. While models like SVM and Logistic Regression performed well, additional improvements in feature selection, cross-validation, and ensemble methods can lead to even more accurate predictions. By continuously optimizing these models, we can enhance the predictive capabilities of marketing campaigns and maximize business outcomes.
