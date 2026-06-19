from huggingface_hub import hf_hub_download
import tensorflow as tf


def load_model():
    model_path = hf_hub_download(
        repo_id="Aasish202/service_detection",
        filename="models/best_model.keras"
    )

    return tf.keras.models.load_model(model_path)