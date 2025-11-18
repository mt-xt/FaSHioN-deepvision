import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

BATCH_SIZE = 32 # Example batch size
RANDOM_SEED = 42 # not necessary i think

epoch_input = input("Enter number of epochs to train the model: ")
EPOCHS = int(epoch_input)


# Use the train and test splits provided by fashion-mnist. 
# x = images, y = labels
(x_val, y_val), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Use the last 12000 samples of the training data as a validation set. 
validation_x = x_val[48000:]
validation_y = y_val[48000:]

# Use the first 48000 samples of the training data as the new training set.
train_x = x_val[:48000]
train_y = y_val[:48000]

train_x = train_x.astype("float32")/255.0
validation_x = validation_x.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# Reshape data to add channel dimension 
# 1: grayscale, 3: RGB
train_x = train_x.reshape((48000, 28, 28, 1))
validation_x = validation_x.reshape((12000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

'''Layer Specifications:
    2D convolutional layer, 28 filters, 3x3 window size, ReLU activation
    2x2 max pooling
    2D convolutional layer, 56 filters, 3x3 window size, ReLU activation
    fully-connected layer, 56 nodes, ReLU activation
    fully-connected layer, 10 nodes, softmax activation'''

model = Sequential([
    Conv2D(filters=28, kernel_size=3, activation='relu', input_shape=(28, 28, 1)), # CONVO LAYERS
    MaxPooling2D((2, 2)),
    Conv2D(filters=56, kernel_size=3, activation='relu'), # CONVO LAYERS, PATTERN DETECTION
    Flatten(), # why do we need this?
    Dense(units=56, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(
    optimizer=Adam(), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
# Train for 10 epochs / or input
data = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=32,validation_data=(validation_x, validation_y))
model.summary()
# print(data.history.keys())
# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])

# Evaluate training and validation accuracy at the end of each epoch, and plot them as line plots on the same set of axes.
plt.figure(figsize=(12, 7))
plt.plot(data.history['loss'], label='Training Loss', marker='x')
plt.plot(data.history['accuracy'], label='Validation Accuracy', marker='o')
plt.xticks(range(0, len(data.history['accuracy'])))  # Every epoch
plt.yticks(np.arange(0, 1.0, 0.025))  
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.savefig('imgs/training_validation_accuracy_loss.png')
plt.tight_layout()
plt.show()

# Print the number of trainable parameters in the model
print(f"Number of trainable parameters: {model.count_params()}")

# Evaluate accuracy on the test set.
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Final Test Accuracy: {test_accuracy}")

# TODO Show an example from the test set for each class where the model misclassifies.
