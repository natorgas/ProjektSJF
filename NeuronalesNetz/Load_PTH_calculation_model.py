
import tensorflow as tf
import numpy as np
import os

# Get the directory of the script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Full path to the directory containing the SavedModel
saved_model_dir = os.path.join(dir_path, "PTH_calculation_model_stable_is_40.keras")

# Load the model
model = tf.keras.models.load_model(saved_model_dir)

alpha = np.linspace(-180, 0, 721)
beta = np.linspace(-180, 180, 1441)

# Batch size for writing to files
batch_size = 250

# Reset TensorFlow session after every reset_interval iterations
reset_interval = 250
current_iteration = 0


def write_to_files(predicted_time, stable_a, stable_b, chaotic_a, chaotic_b):
    with open("Predicted_times.txt", "a") as file_predicted_times, \
            open("Predicted_stable_alpha.txt", "a") as file_stable_alpha, \
            open("Predicted_stable_beta.txt", "a") as file_stable_beta, \
            open("Predicted_chaotic_alpha.txt", "a") as file_chaotic_alpha, \
            open("Predicted_chaotic_beta.txt", "a") as file_chaotic_beta:

        if predicted_time >= 35:
            file_stable_alpha.write("%s\n" % stable_a)
            file_stable_beta.write("%s\n" % stable_b)
            predicted_time = 40
        else:
            file_chaotic_alpha.write("%s\n" % chaotic_a)
            file_chaotic_beta.write("%s\n" % chaotic_b)

        if 30 < predicted_time < 35:
            predicted_time = 30

        file_predicted_times.write("%s\n" % predicted_time)


for a in alpha:
    for b in beta:
        print(a, b)
        input_data = np.array([[a, b]])
        pred = model.predict(input_data)[0][0]

        write_to_files(pred, a, b, a, b)

        current_iteration += 1
        if current_iteration % reset_interval == 0:
            tf.keras.backend.clear_session()
            print("TensorFlow session reset at iteration:", current_iteration)
