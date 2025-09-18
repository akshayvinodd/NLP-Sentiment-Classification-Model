
# =========================
# Step 1: Import libraries
# =========================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =========================
# Step 2: Load dataset
# =========================
# IMDB dataset comes pre-tokenized into integers
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))

# =========================
# Step 3: Preprocess data
# =========================
# Pad sequences so they all have same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# =========================
# Step 4: Build model
# =========================
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=32, input_length=200),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# =========================
# Step 5: Train model
# =========================
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=512,
    validation_split=0.2,
    verbose=1
)

# =========================
# Step 6: Evaluate
# =========================
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test Accuracy:", round(accuracy, 3))

# =========================
# Step 7: Plot training history
# =========================
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

