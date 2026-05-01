import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    Flatten,
    Reshape,
    MaxPooling2D,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.optimizers.legacy import Adam


def build_final_optimized_cnn(input_length, classes):
    model = Sequential()

    model.add(tf.keras.layers.GaussianNoise(stddev=0.2, input_shape=(input_length,)))

    channels = 1
    columns = 64
    rows = int(input_length / (columns * channels))

    model.add(Reshape((rows, columns, channels)))

    # Block 1
    model.add(Conv2D(32, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(64, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(64, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

    # Classifier head
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(classes, name="y_pred", activation="softmax"))

    return model


def compile_model(model, learning_rate=0.0005):
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model


def get_class_weights():
    return {
        0: 2.0,  # cough
        1: 1.0,  # non-cough
    }


if __name__ == "__main__":
    # Edge Impulse provides input_length and classes during training.
    input_length = 32000
    classes = 2

    model = build_final_optimized_cnn(input_length, classes)
    model = compile_model(model)

    model.summary()