import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

saved_model_dir = os.path.join(dir_path, "PTH_calculation_model_stable_is_40.keras")

model = tf.keras.models.load_model(saved_model_dir)

alpha = float(input("Alpha: "))
beta = float(input("Beta: "))

print(f"Vorhersage für Alpha = {alpha}° und Beta = {beta}°")
input_data = np.array([[alpha, beta]])
pred = float(model.predict(input_data)[0][0])

if pred >= 35:
    print("Vorhersage PTH: 40s")
    print("-> Winkelpaar ist stabil")

elif 30 < pred < 35:
    print("Vorhersage PTH: 30s")

else:
    print(f"Vorhersage PTH: {round(pred, 2)}s")

