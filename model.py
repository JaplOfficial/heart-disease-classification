import numpy as np
import tensorflow as tf

from tensorflow import keras

# Define the size of each layer.
input_size = 13
hidden_size = 20
output_size = 1

def create_model():
    # Generate the evolved weights and fill the array.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'),
        tf.keras.layers.Dense(output_size, activation='relu')
    ])
    evolved_weights = np.concatenate([w.flatten() for w in model.get_weights()])
    # Compile the model with an optimizer and loss function.
    optimizer = keras.optimizers.Adam(lr=0.0001)
    loss = keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
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


model = create_model()
weights = init_weights()
model = set_weights(model, weights)
