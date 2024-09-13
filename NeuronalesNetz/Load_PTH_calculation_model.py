import tensorflow as tf
import pandas as pd
import numpy as np
import os

def swish(x):
    return x * tf.keras.activations.sigmoid(x)

# Register the custom activation function
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

saved_model_dir = os.path.join(current_dir, "PTH_calculation_model.keras")
training_data_dir = os.path.join(parent_dir, "NeuronalesNetzTrainingsDaten")
alpha_trainig_path = os.path.join(training_data_dir, "ALPHA.txt")
beta_trainig_path = os.path.join(training_data_dir, "BETA.txt")
times_path = os.path.join(training_data_dir, "TIMES_DATA_STABLE_IS_40.txt")

# Load the model
model = tf.keras.models.load_model(saved_model_dir, custom_objects={'swish': tf.keras.layers.Activation(swish)})

alpha = float(input("Alpha: "))
beta = float(input("Beta: "))

print(f"Prediction for Alpha = {alpha}° and Beta = {beta}°")
input_data = np.array([[alpha, beta]])
pred = float(model.predict(input_data)[0][0])
rounded_pred = round(pred, 2)

if pred >= 35:
    print("Prediction PTH: 40s")
    print("-> Angle pair is stable")
elif 30 < pred < 35:
    print("Prediction PTH: 30s")
else:
    print(f"Prediction PTH: {rounded_pred}s")

alpha_data = pd.read_csv(alpha_trainig_path, header=None, names=["Alpha"])
beta_data = pd.read_csv(beta_trainig_path, header=None, names=["Beta"])
times_data = pd.read_csv(times_path, header=None, names=["Times"])

alpha_list = list(alpha_data.iloc[:, 0])
beta_list = list(beta_data.iloc[:, 0])
times_list = list(times_data.iloc[:, 0])

def find_index(alpha, beta):
    first_correspoding_index = alpha_list.index(alpha) # Index of the first element in the list that corresponds to alpha

    for i in range(1441):
        if beta_list[first_correspoding_index + i] == beta:
            actual_index = first_correspoding_index + i
            return actual_index


if alpha in alpha_list and beta in beta_list:
    index = find_index(alpha, beta)

else:
    alpha = round(alpha, 0)
    beta = round(beta, 0)
    index = find_index(alpha, beta)

measured_time = times_list[index]
error = round(abs(pred - measured_time), 2)

print(f"The measured PTH was {measured_time}s. This corresponds to a difference of {error}s.")
