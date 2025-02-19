import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pandas as pd

pt_train = pd.read_csv("miccai2023_nih-cxr-lt_labels_train.csv")
pt_train = pt_train.drop(columns=['Pneumoperitoneum',
                                  'Pneumoperitoneum',
                                  'Pneumomediastinum',
                                  'Subcutaneous Emphysema',
                                  'Tortuous Aorta',
                                  'Calcification of the Aorta'],
                        axis = 0)
pathology_names = list(pt_train.drop(['id','subj_id'], axis = 1).keys())

# should have all patho names
print(pathology_names)



# Load trained model

def load_model():
    model = tf.keras.models.load_model("cxr_model.h5", compile=False)
    return model

model = load_model()


# Grad-CAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes a Grad-CAM heatmap for a single class index (pred_index).
    Returns a 2D array (H, W) in [0,1].
    """
    # Sub-model: final conv layer + output
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient w.r.t. conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    # Weight the conv outputs
    conv_outputs = conv_outputs[0]  # shape (H, W, C)
    pooled_grads = tf.reshape(pooled_grads, [1,1,-1])
    conv_outputs = conv_outputs * pooled_grads

    # Mean across channels
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)

    # ReLU & normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-10)

    # Upsample (e.g. 7x7 -> 224x224)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )
    heatmap = tf.squeeze(heatmap, axis=-1).numpy()
    return heatmap

def overlay_heatmap(
    original_img,
    heatmap,
    alpha=0.4,         # more subtle
    cmap='jet',
    threshold=None,    # no threshold by default
    grayscale=False
):
    """
    Overlays a Grad-CAM heatmap onto the original image with a more subtle blending.
    original_img in [0,255], shape (H, W, 3)
    heatmap in [0,1], shape (H, W).
    """
    # (Optional) threshold
    if threshold is not None:
        heatmap[heatmap < threshold] = 0.0
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    # Grayscale background?
    if grayscale:
        from PIL import ImageOps
        pil_img = keras.preprocessing.image.array_to_img(original_img)
        pil_img = pil_img.convert('L')
        original_img = keras.preprocessing.image.img_to_array(pil_img)
        # shape (H, W, 1) -> replicate to (H, W, 3)
        original_img = np.repeat(original_img, 3, axis=-1)

    # Convert original to [0,1]
    if original_img.max() > 1.0:
        original_img_float = original_img.astype(np.float32) / 255.0
    else:
        original_img_float = original_img.copy()

    # Convert heatmap to color
    heatmap_255 = np.uint8(255 * heatmap)
    colormap = plt.get_cmap(cmap)
    colormap_colors = colormap(np.arange(256))[:, :3]  # (256,3) in [0,1]
    heatmap_rgb = colormap_colors[heatmap_255]         # (H,W,3)

    # Blend
    blended = (1 - alpha) * original_img_float + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return blended


def compute_top3_gradcams(
    model,
    img_path,
    last_conv_layer_name,
    class_names,
    target_size=(224,224),
    alpha=0.4,
    threshold=None,
    cmap='jet',
    grayscale=False
):
    """
    Loads an image, predicts multi-label probabilities,
    and displays Grad-CAM overlays for the top-3 classes
    (plus the original image).

    Args:
        model (tf.keras.Model)
        img_path (str)
        last_conv_layer_name (str)
        class_names (list of str)
        target_size (tuple)
        alpha (float)
        threshold (float or None)
        cmap (str)
        grayscale (bool)
    """
    # 1) Load & preprocess
    pil_img = image.load_img(img_path, target_size=target_size)
    arr_img = image.img_to_array(pil_img)
    arr_img = tf.keras.applications.densenet.preprocess_input(arr_img)
    arr_img = np.expand_dims(arr_img, axis=0)

    # 2) Predictions -> shape (N,)
    preds = model.predict(arr_img)[0]

    # 3) Sort descending, pick top-3
    top_indices = np.argsort(preds)[::-1][:3]
    top_probs = preds[top_indices]
    top_names = [class_names[i] for i in top_indices]

    # 4) Original image for overlay
    original_img = image.img_to_array(pil_img).astype(np.uint8)

    # 5) We'll have 1 + 3 subplots = 4 columns. Make them bigger.
    fig_cols = 4
    fig, axes = plt.subplots(1, fig_cols, figsize=(5*fig_cols, 6))

    # Show original
    axes[0].imshow(original_img)
    axes[0].set_title("Original", fontsize = 27)
    axes[0].axis('off')

    # 6) For each top class
    for i, (cls_idx, cls_prob, cls_name) in enumerate(zip(top_indices, top_probs, top_names), start=1):
        # Grad-CAM
        heatmap = make_gradcam_heatmap(arr_img, model, last_conv_layer_name, pred_index=cls_idx)
        # Overlay
        overlaid = overlay_heatmap(
            original_img,
            heatmap,
            alpha=alpha,
            cmap=cmap,
            threshold=threshold,
            grayscale=grayscale
        )
        axes[i].imshow(overlaid)
        axes[i].set_title(f"{cls_name}: p={cls_prob:.3f}", fontsize = 27)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# UI for streamlit:
st.title("Chest X-Ray Pathology Classifier")
st.write("Upload a chest X-ray image to get predictions from a trained DenseNet model.")

uploaded_file = st.file_uploader("Choose a CXR image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # 4a) Preprocess
    img_array = preprocess_image(uploaded_file)

    # 4b) Predict
    preds = model.predict(img_array)
    # e.g. preds shape: (1, 14) for 14 pathologies
    st.write("Raw model outputs:", preds)

    # 4c) Binarize if you want a yes/no
    threshold = 0.5
    predicted_labels = (preds[0] >= threshold).astype(int)

    # 4d) Show the user which pathologies are predicted
    pathology_names = ["Atelectasis","Cardiomegaly",..., "No Finding"] # same order as training
    results = []
    for i, pathology in enumerate(pathology_names):
        if predicted_labels[i] == 1:
            results.append(pathology)

    st.write("**Detected Pathologies**:", ", ".join(results) if results else "None")

    # 4e) Grad-CAM
    #    Suppose we run Grad-CAM for a certain pathology index = i
    #    You might show multiple heatmaps, one for each predicted pathology, or
    #    let the user pick a pathology from a dropdown, etc.
    last_conv_layer_name = "conv5_block16_concat"  # last conv layer in DenseNet121
    compute_top3_gradcams(
        model=model,
        img_path="../Images/00000001_000.png",
        last_conv_layer_name=last_conv_layer_name,
        class_names=pathology_cols,
        target_size=(224,224),
        alpha=0.4,      # more subtle overlay
        threshold=None, # no threshold
        cmap='jet',
        grayscale=False # keep color in the X-ray
)
    # 4f) Overlay Grad-CAM on original image
    # Convert heatmap to PIL Image
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((224,224))

    # Optionally make a blend
    original = Image.open(uploaded_file).convert("RGB").resize((224,224))
    overlay = Image.blend(original, heatmap.convert("RGB"), alpha=0.4)

    st.write("Grad-CAM Heatmap (overlay for a single pathology):")
    st.image(overlay, use_column_width=True)
