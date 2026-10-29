from pathlib import Path
import pickle

import numpy as np
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input
)
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"
FEATURE_FILE = DATA_DIR / "image_features.pkl"


# Load pretrained VGG16
model = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)


def extract_feature(image_path):
    """
    Extract a 512-dimensional feature vector from one image.
    """

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = model.predict(
        image,
        verbose=0
    )

    return feature[0]


def extract_all_features():
    """
    Extract features for every Flickr8k image
    and save them to disk.
    """

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    print(f"Found {len(image_paths)} images.")

    features = {}

    for index, image_path in enumerate(image_paths, start=1):

        image_id = image_path.name

        features[image_id] = extract_feature(
            image_path
        )

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(image_paths)}"
            )

    with open(FEATURE_FILE, "wb") as file:
        pickle.dump(features, file)

    print("\nFeature extraction complete.")
    print(f"Saved to: {FEATURE_FILE}")


if __name__ == "__main__":
    extract_all_features()