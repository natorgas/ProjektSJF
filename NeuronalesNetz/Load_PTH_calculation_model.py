import tensorflow as tf
import numpy as np
import os

def swish(x):
    return x * tf.keras.activations.sigmoid(x)

# Register the custom activation function
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

dir_path = os.path.dirname(os.path.realpath(__file__))
saved_model_dir = os.path.join(dir_path, "PTH_calculation_model.keras")

# Load the model
model = tf.keras.models.load_model(saved_model_dir, custom_objects={'swish': tf.keras.layers.Activation(swish)})

alpha = float(input("Alpha: "))
beta = float(input("Beta: "))

print(f"Prediction for Alpha = {alpha}° and Beta = {beta}°")
input_data = np.array([[alpha, beta]])
pred = float(model.predict(input_data)[0][0])

if pred >= 35:
    print("Prediction PTH: 40s")
    print("-> Angle pair is stable")
elif 30 < pred < 35:
    print("Prediction PTH: 30s")
else:
    print(f"Prediction PTH: {round(pred, 2)}s")
