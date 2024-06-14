# Customer_Churn_Prediction

Sure! Here is a sample README file for your project:

---

# Customer Churn Prediction

This project aims to predict customer churn for a subscription-based service or business using historical customer data. The dataset includes various features such as usage behavior and customer demographics. Different machine learning algorithms, including Logistic Regression, Random Forests, and Gradient Boosting, are used to predict churn.

## Dataset

The dataset used for this project is `Churn_Modelling.csv`, which contains the following columns:
- `RowNumber`: Row index
- `CustomerId`: Unique identifier for each customer
- `Surname`: Customer's last name
- `CreditScore`: Customer's credit score
- `Geography`: Customer's location
- `Gender`: Customer's gender
- `Age`: Customer's age
- `Tenure`: Number of years the customer has been with the bank
- `Balance`: Customer's account balance
- `NumOfProducts`: Number of products the customer has with the bank
- `HasCrCard`: Whether the customer has a credit card (1=Yes, 0=No)
- `IsActiveMember`: Whether the customer is an active member (1=Yes, 0=No)
- `EstimatedSalary`: Customer's estimated salary
- `Exited`: Whether the customer has left the bank (1=Yes, 0=No)

## Project Structure

The project consists of the following files:
- `Customer_churn_prediction.py`: The main Python script that performs data preprocessing, model training, and evaluation.
- `Churn_Modelling.csv`: The dataset file.

## Requirements

To run this project, you need the following libraries installed:
- pandas
- numpy
- scikit-learn

You can install these libraries using pip:

```sh
pip install pandas numpy scikit-learn
```

## Running the Project

1. Ensure that both `Customer_churn_prediction.py` and `Churn_Modelling.csv` are located in the same directory.
2. Run the Python script:

```sh
python Customer_churn_prediction.py
```

The script will read the dataset, preprocess the data, train the models, and output the results.

## Results

The script outputs the following results for each model (Logistic Regression, Random Forest, and Gradient Boosting):
- Best hyperparameters found through Grid Search
- Accuracy on the test set
- ROC AUC score
- Classification report (precision, recall, f1-score, support)

## Interpretation of Metrics

- **Accuracy**: The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.
- **ROC AUC**: The area under the receiver operating characteristic curve. It provides an aggregate measure of performance across all classification thresholds.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1-Score**: The weighted average of precision and recall.
- **Support**: The number of actual occurrences of the class in the specified dataset.

## Sample Output

The output of the script includes the best parameters, accuracy, ROC AUC, and classification report for each model. For example:

```
Logistic Regression:

Best Parameters: {'model__C': 10}
Accuracy: 0.811
ROC AUC: 0.7788507974811218
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.96      0.89      1607
           1       0.55      0.20      0.29       393

    accuracy                           0.81      2000
   macro avg       0.69      0.58      0.59      2000
weighted avg       0.78      0.81      0.77      2000


Random Forest:

Best Parameters: {'model__max_depth': None, 'model__n_estimators': 200}
Accuracy: 0.8665
ROC AUC: 0.86032086086476
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1607
           1       0.76      0.47      0.58       393

    accuracy                           0.87      2000
   macro avg       0.82      0.72      0.75      2000
weighted avg       0.86      0.87      0.85      2000


Gradient Boosting:

Best Parameters: {'model__learning_rate': 0.1, 'model__n_estimators': 100}
Accuracy: 0.864
ROC AUC: 0.8715709420141842
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1607
           1       0.74      0.47      0.58       393

    accuracy                           0.86      2000
   macro avg       0.81      0.72      0.75      2000
weighted avg       0.85      0.86      0.85      2000
```

## Conclusion

Based on the evaluation metrics, the most favorable model can be determined. In this example, the Gradient Boosting model shows the highest ROC AUC and provides a good balance between precision and recall for predicting customer churn.



---

Feel free to customize this README file according to your project's specific needs and additional details.
