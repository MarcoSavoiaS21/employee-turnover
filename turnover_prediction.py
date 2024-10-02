import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def get_data_description():
    HRdata = {
        'Variable': [
            'satisfaction_level', 'last_evaluation', 'number_project', 
            'average_monthly_hours', 'time_spend_company', 'Work_accident', 
            'left', 'promotion_last_5years', 'Department', 'salary'
        ],
        'Description': [
            'Job satisfaction level [0–1]', 
            'Last performance review score [0–1]', 
            'Number of projects', 
            'Average monthly hours worked', 
            'Years at the company', 
            'Work accident experience', 
            'Left the company', 
            'Promoted in last 5 years', 
            'Department', 
            'Salary (low, medium, high)'
        ],
        'Type': [
            'numerical', 'numerical', 'numerical', 
            'numerical', 'numerical', 'categorical', 
            'categorical', 'categorical', 'categorical', 'categorical'
        ]
    }
    description_df = pd.DataFrame(HRdata)
    return description_df


def data_inspection(data):
    # Check for missing values
    print("\nNAN value count:")
    print(data.isnull().sum())

    # Inbalanced distribution of 'promotion_last_5years'
    promotion_counts = data['promotion_last_5years'].value_counts(normalize=True) * 100
    print("\nPromotion last 5 years distribution in percentage:")
    print(promotion_counts)

    # Turnover rate
    turnover_rate = data.left.value_counts() / len(data)
    print("\nTurnover rate:")
    print(turnover_rate)

    return promotion_counts, turnover_rate

def preprocessing(data):

    # Define input variable
    y = data["left"]
    X = data.drop(["left"], axis=1)

    # Map salary categories to numerical values
    X['salary'] = X['salary'].map({'low': 1, 'medium': 2, 'high': 3})

    # Convert 'Department' into dummy variables
    X = pd.get_dummies(X, columns=['Department'], drop_first=True)

    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, X, y


def train_default_model(x_train, y_train, x_test, y_test):
    # Training default model
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    
    # Test data evaluation
    accuracy = clf.score(x_test, y_test)
    print(f"Test Accuracy (Default Model): {accuracy}")
    return clf

def hyperparameter_tuning(x_train, y_train):
    # Hyperparameter tuning grid
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, 
                               scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(x_train, y_train)
    
    # Hyperparameter result
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: ", grid_search.best_score_)

    return grid_search.best_estimator_

def evaluate_model(clf, x_test, y_test):
    # Test data evaluation for tuned model
    accuracy = clf.score(x_test, y_test)
    print(f"Test Accuracy (Tuned Model): {accuracy}")
    
    # Prediction test result
    y_pred = clf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def get_feature_importance(clf, X):
    # Feature importances
    importances = clf.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df)
    
    return feature_importance_df


def train_logistic_regression(x_train, y_train, x_test, y_test):
    # Train the Logistic Regression model
    regressionmodel = LogisticRegression()
    regressionmodel.fit(x_train, y_train)

    # Evaluate accuracy on the test data
    accuracy = regressionmodel.score(x_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Make predictions on the test set
    y_pred = regressionmodel.predict(x_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred, regressionmodel


def extract_logistic_regression_coefficients(regressionmodel, X):
    # Extract coefficients and intercept
    coefficients = regressionmodel.coef_[0]
    intercept = regressionmodel.intercept_[0]

    feature_names = X.columns 
    coef_of_feature = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Coefficient values
    coef_of_feature['Absolute Coefficient'] = coef_of_feature['Coefficient'].abs()
    coef = coef_of_feature.sort_values(by='Absolute Coefficient', ascending=False).drop('Absolute Coefficient', axis=1)
    
    # Print coefficients DataFrame
    print(coef)

    # Print intercept
    print(f"Intercept: {intercept}")

    return coef, intercept

def plot_correlation_heatmap(X, y):
    # Add the target variable
    X['left'] = y  

    # Exclude all 'Department' variables
    columns_to_include = [col for col in X.columns if not col.startswith('Department')]
    filtered_X = X[columns_to_include]

    # Correlation matrix
    correlation_matrix = filtered_X.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Heatmap of All Features (Excluding Department)")
    plt.show()
