from pathlib import Path
import re


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CAPTION_FILE = DATA_DIR / "Flickr8k.token.txt"


def load_captions(filename):
    """
    Load Flickr8k captions.

    Returns:
        dict:
            {
                "image.jpg": [
                    "caption one",
                    "caption two",
                    ...
                ]
            }
    """

    captions = {}

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            image_id, caption = line.split("\t", 1)

            image_id = image_id.split("#")[0]

            caption = caption.lower()

            caption = re.sub(r"[^a-z\s]", "", caption)

            caption = re.sub(r"\s+", " ", caption).strip()

            caption = f"<start> {caption} <end>"

            if image_id not in captions:
                captions[image_id] = []

            captions[image_id].append(caption)

    return captions

def load_image_ids(filename):
    """
    Load image filenames from a Flickr8k split file.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def get_split_captions(captions, image_ids):
    """
    Return captions only for images belonging to a particular split.
    """
    return {
        image_id: captions[image_id]
        for image_id in image_ids
        if image_id in captions
    }

if __name__ == "__main__":
    captions = load_captions(CAPTION_FILE)

    train_file = DATA_DIR / "Flickr_8k.trainImages.txt"
    dev_file = DATA_DIR / "Flickr_8k.devImages.txt"
    test_file = DATA_DIR / "Flickr_8k.testImages.txt"

    train_ids = load_image_ids(train_file)
    dev_ids = load_image_ids(dev_file)
    test_ids = load_image_ids(test_file)

    train_captions = get_split_captions(captions, train_ids)
    dev_captions = get_split_captions(captions, dev_ids)
    test_captions = get_split_captions(captions, test_ids)

    print(f"Total images with captions: {len(captions)}")
    print(f"Training images: {len(train_captions)}")
    print(f"Validation images: {len(dev_captions)}")
    print(f"Test images: {len(test_captions)}")

    print("\nExample training image:")
    example_image = next(iter(train_captions))

    print(example_image)

    for caption in train_captions[example_image]:
        print("-", caption)