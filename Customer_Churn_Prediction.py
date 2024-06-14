import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

#Drop unnecessary columns
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

#Handle categorical variables using OneHotEncoder
categorical_features = ['Geography', 'Gender']
numerical_features = df.drop(['Exited'] + categorical_features, axis=1).columns.tolist()

#Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

#Define the models to be used
models = {
    'Logistic Regression': LogisticRegression(max_iter=250, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Split the data into training and testing sets
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store the results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Define hyperparameters for grid search
    if model_name == 'Logistic Regression':
        param_grid = {'model__C': [0.1, 1, 10, 100]}
    elif model_name == 'Random Forest':
        param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}
    elif model_name == 'Gradient Boosting':
        param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1, 0.2]}
    
    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Store the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    results[model_name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': classification_report(y_test, y_pred)
    }

    # Print the results
    print(f"{model_name}:\n")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Classification Report:\n{results[model_name]['classification_report']}\n")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

# Plot formatting
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
