import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402
from tensorflow.keras.preprocessing.image import load_img  # noqa: E402

# Suppress TensorFlow Python-level logging
tf.get_logger().setLevel('ERROR')


def find_labels(path):
    for _, direct, _ in os.walk((path)):
        labels = direct
        break
    labels.sort()
    labels = {i: name for i, name in enumerate(labels)}
    return labels


def predict(path_model, path_data):
    try:
        model = tf.keras.models.load_model(path_model)
    except ValueError:
        print(f"Error: no model found at {path_model}")
        return

    dataset = tf.keras.utils.image_dataset_from_directory(
        path_data,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(64, 64),
        shuffle=False,
        interpolation="bilinear",
        follow_links=False,
    )

    class_names = dataset.class_names
    print(f"Found classes: {class_names}")

    print("Making predictions on the entire dataset...")
    predictions = model.predict(dataset, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = []
    for _, labels in dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"\nTotal images processed: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy


def predict_image(path_model, path_img):
    try:
        model = tf.keras.models.load_model(path_model)
    except ValueError:
        print(f"Error: no model found at {path_model}")
        exit(1)

    img = load_img(path_img, target_size=(64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    labels = find_labels(os.path.dirname(os.path.dirname(path_img)))
    predictions = model.predict(img, verbose=0)
    predicted_index = np.argmax(predictions)
    print("Predicted label:", labels[predicted_index])
    predicted_label = labels[predicted_index]
    return predicted_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using the model to predict the dataset"
    )

    parser.add_argument(
        "path_model",
        default="model/",
        help="Path to the model",
    )
    parser.add_argument(
        "path_data",
        default="images/",
        help="Path to the dataset or single image",
    )

    args = parser.parse_args()

    if os.path.isdir(args.path_data):
        predict(args.path_model, args.path_data)
    else:
        predict_image(args.path_model, args.path_data)
