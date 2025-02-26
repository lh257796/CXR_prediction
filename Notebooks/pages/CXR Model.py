import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt

# if local:
# pt_train = pd.read_csv("./miccai2023_nih-cxr-lt_labels_train.csv")
# if codespaces/online?
pt_train = pd.read_csv("Notebooks/miccai2023_nih-cxr-lt_labels_train.csv")


pt_train = pt_train.drop(
    columns=[
        'Pneumoperitoneum',
        'Pneumoperitoneum',
        'Pneumomediastinum',
        'Subcutaneous Emphysema',
        'Tortuous Aorta',
        'Calcification of the Aorta'
    ],
    axis=0
)
pathology_names = list(pt_train.drop(['id', 'subj_id'], axis=1).keys())

def load_model():
    # if local:
    #model = tf.keras.models.load_model("./cxr_model.keras", compile=False)
    # if codespace/online?
    model = tf.keras.models.load_model("Notebooks/cxr_model.keras", compile=False)

    return model

model = load_model()

from tensorflow import keras
from keras.preprocessing import image

# grad-CAM setup, like in notebook

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = tf.reshape(pooled_grads, [1, 1, -1])
    conv_outputs = conv_outputs * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-10)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )
    heatmap = tf.squeeze(heatmap, axis=-1).numpy()
    return heatmap

def overlay_heatmap(
    original_img,
    heatmap,
    alpha=0.4,
    cmap='jet',
    threshold=None,
    grayscale=False
):
    if threshold is not None:
        heatmap[heatmap < threshold] = 0.0
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    if grayscale:
        from PIL import ImageOps
        pil_img = keras.preprocessing.image.array_to_img(original_img)
        pil_img = pil_img.convert('L')
        original_img = keras.preprocessing.image.img_to_array(pil_img)
        original_img = np.repeat(original_img, 3, axis=-1)

    if original_img.max() > 1.0:
        original_img_float = original_img.astype(np.float32) / 255.0
    else:
        original_img_float = original_img.copy()

    heatmap_255 = np.uint8(255 * heatmap)
    colormap = plt.get_cmap(cmap)
    colormap_colors = colormap(np.arange(256))[:, :3]
    heatmap_rgb = colormap_colors[heatmap_255]

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
    Loads an image (from a path), predicts multi-label probabilities,
    and displays Grad-CAM overlays for the top-3 classes
    (plus the original image).
    """
    pil_img = image.load_img(img_path, target_size=target_size)
    arr_img = image.img_to_array(pil_img)
    arr_img = tf.keras.applications.densenet.preprocess_input(arr_img)
    arr_img = np.expand_dims(arr_img, axis=0)

    preds = model.predict(arr_img)[0]
    top_indices = np.argsort(preds)[::-1][:3]
    top_probs = preds[top_indices]
    top_names = [class_names[i] for i in top_indices]
    original_img = image.img_to_array(pil_img).astype(np.uint8)

    fig_cols = 4
    fig, axes = plt.subplots(1, fig_cols, figsize=(2*fig_cols, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    for i, (cls_idx, cls_prob, cls_name) in enumerate(zip(top_indices, top_probs, top_names), start=1):
        heatmap = make_gradcam_heatmap(arr_img, model, last_conv_layer_name, pred_index=cls_idx)
        overlaid = overlay_heatmap(
            original_img,
            heatmap,
            alpha=alpha,
            cmap=cmap,
            threshold=threshold,
            grayscale=grayscale
        )
        axes[i].imshow(overlaid)
        axes[i].set_title(f"{cls_name}: p={cls_prob:.3f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def preprocess_image(uploaded_file, target_size=(224,224)):
    # uploaded file should be (1,224,224,3) for model
    pil_img = Image.open(uploaded_file).convert('RGB')
    pil_img = pil_img.resize(target_size)
    arr_img = image.img_to_array(pil_img)
    arr_img = tf.keras.applications.densenet.preprocess_input(arr_img)
    arr_img = np.expand_dims(arr_img, axis=0)
    return arr_img

# UI/pipeline

st.title("Thoracic Pathology Chest X-Ray Classifier")
st.write("**Note**: This model is for DEMO/RESEARCH USE only!")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image to upload for analysis...",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    preds = model.predict(img_array)  # shape: (1, num_classes)
    threshold = 0.5
    predicted_labels = (preds[0] >= threshold).astype(int)
    results = []
    for i, pathology in enumerate(pathology_names):
        if predicted_labels[i] == 1:
            results.append(pathology)

    if results:
        st.write("**Top Detected Pathologies**:", ", ".join(results))
    else:
        st.write("No pathologies predicted above threshold.")

    # Optional: Display the uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray", width=224)
    temp_image_path = "temp_uploaded_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    last_conv_layer_name = "conv5_block16_concat"  # Densenet121 last conv layer
    st.write("**Grad-CAM for top-3 predicted classes**:")
    compute_top3_gradcams(
        model=model,
        img_path=temp_image_path,
        last_conv_layer_name=last_conv_layer_name,
        class_names=pathology_names,  # be sure this matches the model's output dimension/order
        target_size=(224,224),
        alpha=0.4,
        threshold=None,
        cmap='jet',
        grayscale=False
    )
    st.pyplot(plt.gcf())
    st.write("_The darkest red portions of the heatmap indicate where the model is 'looking' the most when making a predictive classification. Though Grad-CAM heatmaps are a way to introduce some transparency with the way a machine learning model makes its classifications, it doesn't quite fully explain how it comes to this conclusion and only offers a glimpse into the complex underlying mechanisms influencing the CNN. For that reason, these visualizations should be taken with a grain of salt._")
