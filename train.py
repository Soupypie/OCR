import numpy as np
import pickle
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import kerastuner as kt
import json
import os

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Load and preprocess the EMNIST dataset
with open('emnist_train.pkl', 'rb') as f:
    emnist_train = pickle.load(f)
with open('emnist_test.pkl', 'rb') as g:
    emnist_test = pickle.load(g)

# Extract data and labels
X_train, y_train = emnist_train['data'], emnist_train['labels']
X_val, y_val = emnist_test['data'], emnist_test['labels']

# Reshape and normalize image data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_val = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to categorical format
y_train = keras.utils.to_categorical(y_train, num_classes=62)
y_val = keras.utils.to_categorical(y_val, num_classes=62)

# Set up data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.25,
    brightness_range=[0.7, 1.3]
)
datagen.fit(X_train)

# Function to build and compile the CNN model for Keras Tuner
def model_builder(hp):
    model = keras.Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # First Convolutional Block with Tuning
    for i in range(hp.Int('conv_blocks', 2, 4)):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32)
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))

    # Dense Layers
    model.add(Flatten())
    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=64)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(62, activation='softmax'))

    # Compile with tunable learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Hyperparameter tuning with Keras Tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='my_dir',
                     project_name='emnist_tuning')

# Early stopping, model checkpoint, and reduce learning rate callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_emnist_model.keras', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model with augmented data and verbose tuning output
tuner.search(datagen.flow(X_train, y_train, batch_size=64),
             epochs=50,
             validation_data=(X_val, y_val),
             callbacks=[early_stopping, model_checkpoint, reduce_lr],
             verbose=1)

# Get and save the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps_dict = {
    "conv_blocks": best_hps.get('conv_blocks'),
    "filters": [best_hps.get(f'filters_{i}') for i in range(best_hps.get('conv_blocks'))],
    "dense_units": best_hps.get('dense_units'),
    "learning_rate": best_hps.get('learning_rate')
}
with open('best_hyperparameters.json', 'w') as json_file:
    json.dump(best_hps_dict, json_file)

# Print the optimal hyperparameters
print(f"""
The hyperparameter search is complete. Optimal configuration:
- Number of convolutional blocks: {best_hps.get('conv_blocks')}
- Filters per convolutional block: {[best_hps.get(f'filters_{i}') for i in range(best_hps.get('conv_blocks'))]}
- Dense layer units: {best_hps.get('dense_units')}
- Learning rate: {best_hps.get('learning_rate')}
""")

# Build and summarize the final model with optimal hyperparameters
model = tuner.hypermodel.build(best_hps)
model.summary()

# Final training with the best model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Save the final trained model
model.save('emnist_model_final.keras')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')

# Function to load and preprocess individual PNG images
def preprocess_png_image(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img_array

# Example of loading a PNG image and making a prediction
def predict_image(image_path, model):
    img_array = preprocess_png_image(image_path)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_label, confidence

# Usage example
# image_path = 'path/to/your/image.png'
# label, confidence = predict_image(image_path, model)
# print(f'Predicted label: {label}, Confidence: {confidence:.2f}')
