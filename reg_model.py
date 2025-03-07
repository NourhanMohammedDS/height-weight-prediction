import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load dataset
file_path = "wh.csv"

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: File 'wh.csv' not found. Ensure it is uploaded.")

# Prepare input (Height) and output (Weight)
x = data["Height"].values.reshape(-1, 1)  # Reshape to 2D
y = data["Weight"].values

# Create the model
model = Sequential([
    Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.5), loss="mean_squared_error")

# Train the model
model.fit(x, y, epochs=50, verbose=1)

# Model accuracy
yp = model.predict(x)
print("R-squared Score:", mtr.r2_score(y, yp))

# Plot actual vs predicted data
plt.scatter(x, y, label="Actual Data")
plt.plot(x, yp, c="red", label="Predicted Line")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()

# Single prediction
H = float(input("Enter the Height: "))
predicted_weight = model.predict(np.array([[H]]))
print("Predicted Weight:", predicted_weight[0][0])

# Overfitting test
xtr, xts, ytr, yts = train_test_split(x, y, train_size=0.80, random_state=42)

# Train on training set only
model.fit(xtr, ytr, epochs=50, verbose=1)
ytr_p = model.predict(xtr)
yts_p = model.predict(xts)

# Accuracy scores
print("Train R-squared Score:", mtr.r2_score(ytr, ytr_p))
print("Test R-squared Score:", mtr.r2_score(yts, yts_p))

# Save the trained model
model.save('my_model.keras')
print("Model saved successfully!")


