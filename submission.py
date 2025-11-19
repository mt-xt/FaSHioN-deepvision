import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import CUDA
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers # condensed imports smfh
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


BATCH_SIZE = 32 # Example batch size
# RANDOM_SEED = 42  not necessary i think

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

# Begin training

'''Layer Specifications:
    2D convolutional layer, 28 filters, 3x3 window size, ReLU activation
    2x2 max pooling
    2D convolutional layer, 56 filters, 3x3 window size, ReLU activation
    fully-connected layer, 56 nodes, ReLU activation
    fully-connected layer, 10 nodes, softmax activation'''

model = Sequential([
    layers.Conv2D(filters=28, kernel_size=3, activation='relu', input_shape=(28, 28, 1)), # CONVO lAYERS
    # layers.Conv2D(filters=28, kernel_size=3, activation='relu', input_shape=(28, 28, 1)), # delete
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=56, kernel_size=3, activation='relu'), # CONVO lAYERS, PATTERN DETECTION
    # layers.Conv2D(filters=56, kernel_size=3, activation='relu'), # delete
    # layers.MaxPooling2D((2, 2)), # delete
    layers.Flatten(), 
    layers.Dense(units=56, activation='relu'),
    layers.Dense(units=10, activation='softmax')
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

# Print the number of trainable parameters in the model
print(f"Number of trainable parameters: {model.count_params()}")

# Evaluate accuracy on the test set.
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Final Test Accuracy: {test_accuracy}")

# Show an example from the test set for each class where the model misclassifies.
predictions = model.predict(x_test) 
misclassified_indices = []
for i in range(len(y_test)):
    if np.argmax(predictions[i]) != y_test[i]:
        misclassified_indices.append(i)

# Note: Each training and test example is assigned to one of the following labels
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

# Clusters too much on the graph; removed for now
# label = {0: "T-shirt/top", 
#          1: "Trouser", 
#          2: "Pullover", 
#          3: "Dress", 
#          4: "Coat", 
#          5: "Sandal", 
#          6: "Shirt", 
#          7: "Sneaker", 
#          8: "Bag", 
#          9: "Ankle boot"}

displayed_classes = set()
plt.figure(figsize=(5, 5))
count = 0
for idx in misclassified_indices:
    if count == 6:
        break
    true_label = y_test[idx]
    if true_label not in displayed_classes:
        plt.subplot(3, 4, len(displayed_classes) + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Answer: {true_label} \n Pred: {np.argmax(predictions[idx])}') # Corrct Ans and prediction 
        plt.axis('off')
        displayed_classes.add(true_label)
        count += 1
plt.tight_layout()
plt.savefig('imgs/misclassified_examples.png')
plt.show()

# dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])
# Evaluate training and validation accuracy at the end of each epoch, and plot them as line plots on the same set of axes.
plt.figure(figsize=(9, 7))
plt.plot(data.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(data.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.xticks(range(0, len(data.history['accuracy'])))  # Every epoch
plt.yticks(np.arange(0.75, 1.0, 0.025))  
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('imgs/training_validation_accuracy_loss.png')
plt.tight_layout()
plt.show()