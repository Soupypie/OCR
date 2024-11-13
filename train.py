import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow_datasets as tfds

# Load the dataset as tf.data.Dataset
train_data, test_data = tfds.load('emnist', split=['train', 'test'], as_supervised=True)

# Preprocess the data
def preprocess_image(image, label):
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Reshape to add a channel dimension (28, 28, 1)
    image = tf.reshape(image, (28, 28, 1))
    return image, label

train_ds = train_data.map(preprocess_image).batch(64)
test_ds = test_data.map(preprocess_image).batch(64)

# Convert labels to categorical (for multi-class classification)
def preprocess_labels(image, label):
    label = tf.one_hot(label, depth=62)  # EMNIST has 62 classes
    return image, label

train_ds = train_ds.map(preprocess_labels)
test_ds = test_ds.map(preprocess_labels)

# Define a more complex model architecture with Batch Normalization and Dropout
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(62, activation='softmax')  # EMNIST has 62 classes
])

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch % 10 == 0: 
        return lr * 0.5
    else:
        return lr

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    callbacks=[LearningRateScheduler(lr_schedule), early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model in .keras format
model.save('emnist_ocr_model.keras')
