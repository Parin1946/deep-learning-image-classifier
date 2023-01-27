
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_simple_cnn(input_shape=(128, 128, 3), num_classes=10):
    """
    Builds a simple Convolutional Neural Network (CNN) for image classification.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: A compiled Keras Sequential model.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation=\'relu\', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation=\'relu\'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation=\'relu\'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flattening and Dense Layers
        Flatten(),
        Dense(512, activation=\'relu\'),
        Dropout(0.5),
        Dense(num_classes, activation=\'softmax") # Output layer with softmax for multi-class classification
    ])

    model.compile(
        optimizer=\'adam\',
        loss=\'categorical_crossentropy\',
        metrics=[\'accuracy"]
    )
    return model

def train_model(model, train_dir, validation_dir, epochs=10, batch_size=32, img_size=(128, 128)):
    """
    Trains the given Keras model using image data generators.

    Args:
        model (tf.keras.Model): The compiled Keras model to train.
        train_dir (str): Path to the training data directory.
        validation_dir (str): Path to the validation data directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (tuple): Target size for input images (height, width).

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode=\'nearest\'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=\'categorical\'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=\'categorical\'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    return history

if __name__ == "__main__":
    # Example usage (dummy directories for demonstration)
    print("Building model...")
    cnn_model = build_simple_cnn()
    cnn_model.summary()

    # In a real scenario, you would have actual image directories
    # train_data_dir = \'/path/to/your/train_data\'
    # validation_data_dir = \'/path/to/your/validation_data\'
    # print("Training model...")
    # history = train_model(cnn_model, train_data_dir, validation_data_dir)
    # print("Model training complete.")

    print("Model definition complete. For training, provide actual data directories.")
