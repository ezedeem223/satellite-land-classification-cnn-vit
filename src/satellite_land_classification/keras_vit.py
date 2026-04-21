from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from .augmentation import keras_datagen_kwargs
from .data import validate_dataset_dir
from .features import resolve_feature_layer_name
from .utils import AssetNotFoundError, ensure_dir, require_path


@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
    """Learnable positional embedding used in the Keras hybrid experiment."""

    def __init__(self, num_patches: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    """Transformer encoder block used in the Keras hybrid experiment."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.mha = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x):
        x = self.norm1(x + self.mha(x, x))
        return self.norm2(x + self.mlp(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config


def build_cnn_vit_hybrid(
    cnn_model,
    feature_layer_name: str,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    mlp_dim: int = 2048,
    num_classes: int = 2,
):
    """Build the Keras CNN-ViT hybrid used in the preserved Keras experiments."""

    cnn_model.trainable = False
    features = cnn_model.get_layer(feature_layer_name).output
    height, width, channels = features.shape[1], features.shape[2], features.shape[3]
    x = layers.Reshape((height * width, channels))(features)
    x = AddPositionEmbedding(height * width, channels)(x)
    for _ in range(num_transformer_layers):
        x = TransformerBlock(channels, num_heads=num_heads, mlp_dim=mlp_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(cnn_model.layers[0].input, outputs, name="CNN_ViT_hybrid")


def load_hybrid_model(model_path: str | Path):
    """Load a serialized Keras hybrid model with custom layers."""

    require_path(model_path, "Keras hybrid model")
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AddPositionEmbedding": AddPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        },
    )


def train_keras_cnn_vit(
    dataset_dir: str | Path,
    pretrained_cnn_path: str | Path,
    output_model_path: str | Path,
    feature_layer_name: str | None = None,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    steps_per_epoch: int = 128,
    validation_split: float = 0.2,
    num_transformer_layers: int = 4,
    attention_heads: int = 8,
    mlp_dim: int = 2048,
) -> dict[str, list[float]]:
    """Train the Keras CNN-ViT hybrid using the preserved hybrid setup."""

    root = validate_dataset_dir(dataset_dir)
    cnn_path = Path(pretrained_cnn_path)
    if not cnn_path.exists():
        raise AssetNotFoundError(
            f"Pretrained Keras CNN backbone not found at {cnn_path}. "
            "Train the baseline Keras CNN first or provide a compatible pretrained checkpoint."
        )
    model_path = Path(output_model_path)
    ensure_dir(model_path.parent)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **keras_datagen_kwargs(validation_split=validation_split)
    )
    train_gen = datagen.flow_from_directory(
        root,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        root,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=True,
    )
    cnn_model = tf.keras.models.load_model(cnn_path)
    hybrid_model = build_cnn_vit_hybrid(
        cnn_model=cnn_model,
        feature_layer_name=resolve_feature_layer_name(feature_layer_name),
        num_transformer_layers=num_transformer_layers,
        num_heads=attention_heads,
        mlp_dim=mlp_dim,
        num_classes=train_gen.num_classes,
    )
    hybrid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    history = hybrid_model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint_cb],
        steps_per_epoch=steps_per_epoch,
    )
    return {key: [float(item) for item in values] for key, values in history.history.items()}


def predict_with_keras_hybrid(model_path: str | Path, image_path: str | Path) -> dict[str, float]:
    """Predict a single image with the Keras hybrid model."""

    model = load_hybrid_model(model_path)
    image = tf.keras.utils.load_img(image_path, target_size=(64, 64))
    array = tf.keras.utils.img_to_array(image) / 255.0
    array = tf.expand_dims(array, axis=0)
    probs = model.predict(array, verbose=0)[0]
    return {"non_agri": float(probs[0]), "agri": float(probs[1])}
