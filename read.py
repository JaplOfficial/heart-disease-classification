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

    # Extract the features and target
    features = cleveland_data.drop('target', axis=1)
    target = cleveland_data['target']

    # replace any '?' with 0
    features = features.replace('?', 0)

    # convert to float
    features = features.astype(float)

    # replace NaN and infinite values with 0
    features = features.replace([np.inf, -np.inf, np.nan], 0)

    # Normalize the features feature-wise using StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Split the normalized data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_norm, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
