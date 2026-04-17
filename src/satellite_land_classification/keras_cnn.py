from __future__ import annotations

from pathlib import Path

from .augmentation import keras_datagen_kwargs
from .data import validate_dataset_dir
from .utils import ensure_dir


def _tf():
    import tensorflow as tf

    return tf


def build_keras_cnn(image_size: tuple[int, int] = (64, 64), channels: int = 3):
    """Build the CNN architecture used in Module 2 Keras."""

    tf = _tf()
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Conv2D(
                32,
                (5, 5),
                activation="relu",
                padding="same",
                strides=(1, 1),
                kernel_initializer=tf.keras.initializers.HeUniform(),
                input_shape=(image_size[0], image_size[1], channels),
            ),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            layers.Conv2D(
                64,
                (5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            layers.Conv2D(
                128,
                (5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            layers.Conv2D(
                256,
                (5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            layers.Conv2D(
                512,
                (5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            layers.Conv2D(
                1024,
                (5, 5),
                activation="relu",
                padding="same",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                64,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(
                128,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(
                256,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(
                512,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(
                1024,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(
                2048,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeUniform(),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def create_keras_generators(
    dataset_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    validation_split: float = 0.2,
):
    """Create training and validation generators matching the notebook settings."""

    tf = _tf()
    root = validate_dataset_dir(dataset_dir)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **keras_datagen_kwargs(validation_split=validation_split)
    )
    train_generator = datagen.flow_from_directory(
        root,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
    )
    validation_generator = datagen.flow_from_directory(
        root,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
    )
    return train_generator, validation_generator


def train_keras_cnn(
    dataset_dir: str | Path,
    output_model_path: str | Path,
    image_size: tuple[int, int] = (64, 64),
    channels: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 3,
    validation_split: float = 0.2,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    checkpoint_monitor: str = "val_accuracy",
    checkpoint_mode: str = "max",
) -> dict[str, list[float]]:
    """Train the Keras CNN and save the best model checkpoint."""

    tf = _tf()
    model_path = Path(output_model_path)
    ensure_dir(model_path.parent)
    train_generator, validation_generator = create_keras_generators(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    model = build_keras_cnn(image_size=image_size, channels=channels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_best_only=True,
        verbose=1,
    )
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb],
        verbose=1,
    )
    return {key: [float(item) for item in values] for key, values in history.history.items()}


def load_keras_model(model_path: str | Path):
    """Load a serialized Keras model."""

    return _tf().keras.models.load_model(model_path)


def predict_with_keras_model(model_path: str | Path, image_path: str | Path) -> dict[str, float]:
    """Predict a single image with a binary Keras model."""

    tf = _tf()
    model = load_keras_model(model_path)
    image = tf.keras.utils.load_img(image_path, target_size=(64, 64))
    array = tf.keras.utils.img_to_array(image) / 255.0
    array = tf.expand_dims(array, axis=0)
    prob = float(model.predict(array, verbose=0).reshape(-1)[0])
    return {"non_agri": 1.0 - prob, "agri": prob}
