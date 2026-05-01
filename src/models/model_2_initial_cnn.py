import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    Flatten,
    Reshape,
    MaxPooling2D,
)
from tensorflow.keras.optimizers.legacy import Adam


def build_initial_cnn(input_length, classes):
    model = Sequential()

    model.add(tf.keras.layers.GaussianNoise(stddev=0.45, input_shape=(input_length,)))

    channels = 1
    columns = 64
    rows = int(input_length / (columns * channels))

    model.add(Reshape((rows, columns, channels), input_shape=(input_length,)))

    model.add(
        Conv2D(
            16,
            kernel_size=3,
            kernel_constraint=tf.keras.constraints.MaxNorm(1),
            padding="same",
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

    model.add(
        Conv2D(
            32,
            kernel_size=3,
            kernel_constraint=tf.keras.constraints.MaxNorm(1),
            padding="same",
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

    model.add(Flatten())

    model.add(
        Dense(
            16,
            activation="relu",
            activity_regularizer=tf.keras.regularizers.l1(0.00001),
        )
    )
    model.add(Dropout(0.25))

    model.add(
        Dense(
            2,
            activation="relu",
            activity_regularizer=tf.keras.regularizers.l1(0.00001),
        )
    )

    model.add(Dense(classes, name="y_pred", activation="softmax"))

    return model


def compile_model(model, learning_rate=0.001):
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


if __name__ == "__main__":
    # Edge Impulse provides input_length and classes during training.
    input_length = 32000
    classes = 2

    model = build_initial_cnn(input_length, classes)
    model = compile_model(model)

    model.summary()