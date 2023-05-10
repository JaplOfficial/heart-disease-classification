import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def prepare_dataset():
    # Load the Cleveland dataset (the only processed dataset)
    cleveland_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

    # Load the unprocessed datasets
    hungarian_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data", header=None, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

    switzerland_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data", header=None, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

    va_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data", header=None, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

    # Concatenate all the datasets into one
    all_data = pd.concat([cleveland_data, hungarian_data, switzerland_data, va_data], axis=0)
    
    all_data['target'] = all_data['target'].replace([1, 2, 3, 4], 1)

    # Extract the features and target
    target = all_data['target']
    features = all_data.drop('target', axis=1)

    # Replace any '?' with 0
    features = features.replace('?', 0)

    # Convert to float
    features = features.astype(float)

    # Replace NaN and infinite values with 0
    features = features.replace([np.inf, -np.inf, np.nan], 0)

    # Split the data into training, validation, and testing sets
    X_train, X_valtest, y_train, y_valtest = train_test_split(features, target, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)

    # Normalize the features using StandardScaler based on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)



    return X_train, X_val, X_test, y_train, y_val, y_test
