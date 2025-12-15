import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
EfficientNetB0 = tf.keras.applications.EfficientNetB0


def set_finetune_layers(base_cnn: tf.keras.Model, fine_tune_layers: int):
    """
    Freeze all layers except the last `fine_tune_layers`.
    Keeps BatchNorm frozen for stability.
    """
    base_cnn.trainable = True

    if not fine_tune_layers or fine_tune_layers <= 0:
        for layer in base_cnn.layers:
            layer.trainable = False
        return

    for layer in base_cnn.layers[:-fine_tune_layers]:
        layer.trainable = False

    for layer in base_cnn.layers[-fine_tune_layers:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)


def build_hybrid_model(
    img_size: int = 224,
    dct_dim: int = 4096,
    fine_tune_layers: int = 20,
    lr: float = 1e-5,
) -> tf.keras.Model:
    img_shape = (img_size, img_size, 3)

    # --- RGB branch ---
    rgb_input = layers.Input(shape=img_shape, name="rgb_input")

    base_cnn = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
    )

    # apply fine-tuning policy
    set_finetune_layers(base_cnn, fine_tune_layers)

    feat_map = base_cnn(rgb_input)
    cnn_feat = layers.GlobalAveragePooling2D(name="rgb_global_pool")(feat_map)

    # --- DCT branch ---
    dct_input = layers.Input(shape=(dct_dim,), name="dct_input")
    freq = layers.Dense(512, activation="relu")(dct_input)
    freq = layers.Dropout(0.3)(freq)
    freq = layers.Dense(512, activation="relu")(freq)
    freq = layers.Dropout(0.3)(freq)

    # --- Fusion ---
    fused = layers.Concatenate(name="fusion")([cnn_feat, freq])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    output = layers.Dense(1, activation="sigmoid", name="output")(fused)

    model = models.Model(
        inputs=[rgb_input, dct_input],
        outputs=output,
        name="Hybrid_CNN_DCT",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
