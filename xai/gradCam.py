import numpy as np
import tensorflow as tf
import cv2

preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def _prepare_rgb(img_01: np.ndarray, img_size: int = 224) -> tf.Tensor:
    """
    img_01: (H,W,3) float in [0,1] in SAME channel order it trained with.
    NOTE: the pipeline uses BGR (cv2), so keep it consistent.
    """
    x = tf.convert_to_tensor(img_01, dtype=tf.float32)
    x = tf.image.resize(x, (img_size, img_size))
    x = preprocess_input(x * 255.0)
    return tf.expand_dims(x, 0)


def _prepare_dct(dct_vec: np.ndarray) -> tf.Tensor:
    x = tf.convert_to_tensor(dct_vec, dtype=tf.float32)
    return tf.expand_dims(x, 0)


def _resolve_connected_conv_tensor(
    model: tf.keras.Model,
    target_layer: str | None,
    backbone_name: str = "efficientnetb0",
) -> tf.Tensor:
    """
    Return a conv tensor that is GUARANTEED to be connected to model.output.

    For the model, the most reliable tensor is the input to rgb_global_pool
    (the feature map produced by EfficientNet that the classifier actually uses).
    """
    fallback = model.get_layer("rgb_global_pool").input  # (None,H,W,C) connected to output

    # If user asked for something top-level and it's 4D, allow it
    if target_layer:
        try:
            t = model.get_layer(target_layer).output
            if len(t.shape) == 4:
                return t
        except Exception:
            pass

    # Nested backbone inner layers (like "top_conv") are NOT connected in the outer graph
    # unless we rebuild the full forward pass from that tensor, so we safely fallback.
    if target_layer and target_layer not in (None, "auto"):
        print(f"[WARN] target_layer='{target_layer}' is inside a nested backbone; using rgb_global_pool input instead.")

    return fallback

    # Build: backbone.input -> inner_layer.output
    inner_model = tf.keras.Model(inputs=backbone.input, outputs=inner_layer.output)

    # Connect to OUTER graph by calling on model RGB input tensor
    conv_tensor = inner_model(model.inputs[0])

    # Safety: must be 4D
    if len(conv_tensor.shape) != 4:
        return fallback

    return conv_tensor


def compute_gradcam(
    model: tf.keras.Model,
    img_01: np.ndarray,
    dct_vec: np.ndarray,
    target_layer: str | None = None,
    img_size: int = 224,
    target: str = "fake",   # "fake" or "real"
) -> np.ndarray:
    """
    Returns heatmap (img_size, img_size) in [0,1]
    """
    conv_tensor = _resolve_connected_conv_tensor(model, target_layer)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_tensor, model.output],
    )

    rgb_in = _prepare_rgb(img_01, img_size)
    dct_in = _prepare_dct(dct_vec)

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model([rgb_in, dct_in], training=False)

        # IMPORTANT: watch conv_out explicitly (it's not a Variable)
        tape.watch(conv_out)

        prob_fake = pred[:, 0]
        score = prob_fake if target.lower() == "fake" else (1.0 - prob_fake)

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError(
            "Gradients are None. Try target_layer=None (fallback) or target_layer='top_conv'."
        )

    # Grad-CAM weights
    weights = tf.reduce_mean(grads, axis=(1, 2))[0]  # (C,)
    cam = tf.reduce_sum(conv_out[0] * weights[tf.newaxis, tf.newaxis, :], axis=-1)

    cam = tf.nn.relu(cam).numpy()
    cam = cv2.resize(cam, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros((img_size, img_size), dtype=np.float32)

    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam.astype(np.float32)


def overlay_heatmap(img_01: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_01 is BGR float 0..1 from the pipeline.
    Returns BGR uint8 for OpenCV, but can convert to RGB for matplotlib display.
    """
    img_bgr = (np.clip(img_01, 0, 1) * 255).astype(np.uint8)
    hm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, hm_color, alpha, 0)
    return blended
