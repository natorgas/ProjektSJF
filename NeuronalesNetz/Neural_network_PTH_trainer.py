import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

alpha_data = pd.read_csv("ALPHA.txt", header=None, names=["Alpha"])
beta_data = pd.read_csv("BETA.txt", header=None, names=["Beta"])
times_data = pd.read_csv("TIMES_DATA_STABLE_IS_40.txt", header=None, names=["Times"])


def swish(x):
    return x * tf.keras.activations.sigmoid(x)


data = pd.concat([alpha_data, beta_data, times_data], axis=1)

print(alpha_data)

predict = "Times"

X = data.drop(columns=["Times"])
Y = np.array(data[predict])

# Aufteilen der Daten
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = tf.keras.models.Sequential()

# Anzahl Schichten und Neuronen festlegen
model.add(tf.keras.layers.InputLayer(input_shape=(2,)))
model.add(tf.keras.layers.Dense(units=10, activation=swish))
model.add(tf.keras.layers.Dense(units=200, activation=swish))
model.add(tf.keras.layers.Dense(units=100, activation=swish))
model.add(tf.keras.layers.Dense(units=200, activation=swish))
model.add(tf.keras.layers.Dense(units=200, activation=swish))
model.add(tf.keras.layers.Dense(units=40, activation=swish))
model.add(tf.keras.layers.Dense(units=1, activation=swish))

model.compile(optimizer="nadam", loss="mean_absolute_error")

# Modell trainieren
model.fit(x_train, y_train, epochs=5)

# Auswertung
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute Error:", mae)

r2 = r2_score(y_test, y_pred)
print(f"r2 = {r2}")

model.evaluate(x_test, y_test)

model.save("PTH_calculation_model_stable_is_40.keras")