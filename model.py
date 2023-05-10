import numpy as np
import random
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras

# Define the size of each layer.
input_size = 13
hidden_size = 20
output_size = 1

def init_random_svm():
    return SVC(kernel='rbf', C=random.uniform(0, 10), gamma=random.uniform(0, 10))

def init_svm(C=1.0, gamma=1.0):
    return SVC(kernel='rbf', C=C, gamma=gamma)

def svm_rbf_model(X, y, C=1.0, gamma=1.0):
    # Create the SVM with RBF kernel model
    clf = SVC(kernel='rbf', C=C, gamma=gamma)

    # Fit the model on the data
    clf.fit(X, y)

    # Make predictions on the training data
    y_pred = clf.predict(X)

    # Print the results
    print('Accuracy score: {:.3f}'.format(accuracy_score(y, y_pred)))
    print('Precision score: {:.3f}'.format(precision_score(y, y_pred, average='weighted')))
    print('Recall score: {:.3f}'.format(recall_score(y, y_pred, average='weighted')))
    print('F1 score: {:.3f}'.format(f1_score(y, y_pred, average='weighted')))
    print('-----------------------------------')

    return clf, accuracy_score(y, y_pred), f1_score(y, y_pred, average='weighted')

def create_model():
    # Generate the evolved weights and fill the array.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'),
        tf.keras.layers.Dense(output_size, activation='relu')
    ])
    evolved_weights = np.concatenate([w.flatten() for w in model.get_weights()])
    # Compile the model with an optimizer and loss function.
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model

def init_weights():
    # Generate random weights between -1 and 1 for each weight in the network
    evolved_weights = np.random.uniform(-1, 1, size=(input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size)
    return evolved_weights

def set_weights(model, evolved_weights):

    # Upload the evolved weights to the model.
    model.set_weights([
        evolved_weights[:input_size * hidden_size].reshape(input_size, hidden_size),
        evolved_weights[input_size * hidden_size:input_size * hidden_size + hidden_size],
        evolved_weights[input_size * hidden_size + hidden_size:input_size * hidden_size + hidden_size + hidden_size * output_size].reshape(hidden_size, output_size),
        evolved_weights[input_size * hidden_size + hidden_size + hidden_size * output_size:].reshape(output_size,)
    ])

    return model
