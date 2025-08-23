import os
from datetime import datetime
import argparse
from Augmentation import enrichDataset
from dataset_split import split_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras import layers   # noqa: E402
from tensorflow import keras   # noqa: E402


def main(path: str):
    train_path = split_dataset(path, 0.95)[0]
    print(train_path)
    enrichDataset(train_path)

    train_images, validation_images = keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(64, 64),
        seed=42,
        validation_split=0.2,
        subset="both",
        interpolation="bilinear",
        follow_links=False,
    )

    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True,
        min_delta=0,
        mode="auto",
        baseline=None,
        start_from_epoch=5,
    )

    model = keras.models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(8, activation="softmax"))

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]
    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    epochs = 42

    model.fit(
        train_images,
        validation_data=validation_images,
        epochs=epochs,
        callbacks=[callback],
    )
    print("\033[96mModel train is completed!\033[0m")
    print("\033[92mNow evaluating model...")
    model.evaluate(validation_images, verbose=0)
    print(validation_images)
    print("\033[0m")
    if not os.path.exists("submission/model"):
        os.mkdir("submission/model")
    model.save("submission/model/model" +
               datetime.now().strftime("_%m-%d_%H:%M") + ".keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program is to train a model \
            to predict classes of leaves"
    )
    parser.add_argument(
        "path_data",
        default="images/",
        help="The path to data to train with",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path_data):
        main(args.path_data)
    else:
        print("Error: passed path is not a directory")
