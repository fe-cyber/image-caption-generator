from pathlib import Path
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import load_captions, load_image_ids


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CAPTION_FILE = DATA_DIR / "Flickr8k.token.txt"

def load_training_captions():
    """
    Load captions for the training set.
    """
    captions = load_captions(CAPTION_FILE)
    train_file = DATA_DIR / "Flickr_8k.trainImages.txt"
    train_ids = load_image_ids(train_file)
    all_captions = load_captions(CAPTION_FILE)

    train_captions = {
        image_id: all_captions[image_id]
        for image_id in train_ids
        if image_id in all_captions
    }

    return train_captions

def build_tokenizer(train_captions):
    """
    Build a tokenizer from the training captions.
    """
    all_captions = []
    for captions in train_captions.values():
        all_captions.extend(captions)

    tokenizer = Tokenizer(
        oov_token="<unk>"
    )
    tokenizer.fit_on_texts(all_captions)

    return tokenizer

def create_sequences(tokenised_caption):
    """
    Create input-output sequences from a tokenised caption.
    """
    sequences = []
    for i in range(1, len(tokenised_caption)):
        input_seq = tokenised_caption[:i]
        output_word = tokenised_caption[i]
        sequences.append((input_seq, output_word))

    return sequences

def prepare_sequences(sequences, max_length):
    X = pad_sequences(
        [seq[0] for seq in sequences],
        maxlen=max_length,
        padding="pre"
    )

    y = [seq[1] for seq in sequences]

    return X, y

def get_max_caption_length(train_captions, tokenizer):
    max_length = 0

    # find the longest tokenized caption
    for (image_id, captions) in train_captions.items():
        for caption in captions:
            tokenized_caption = tokenizer.texts_to_sequences([caption])[0]

            if len(tokenized_caption) > max_length:
                max_length = len(tokenized_caption)
                print(f"New max length found: {max_length} for caption: {caption}")

    return max_length

def build_training_data(train_captions, tokenizer, image_features, max_length):
    X1, X2, y = [], [], []

    for (image_id, captions) in train_captions.items():
        feature = image_features[image_id]

        for caption in captions:
            tokenized_caption = tokenizer.texts_to_sequences([caption])[0]
            sequences = create_sequences(tokenized_caption)

            for input_seq, output_word in sequences:
                input_seq_padded = pad_sequences(
                    [input_seq],
                    maxlen=max_length,
                    padding="pre"
                )[0]

                X1.append(feature)
                X2.append(input_seq_padded)
                y.append(output_word)

    return (X1, X2), y

def create_dataset(X1, X2, y, batch_size=32):
    """
    Create a TensorFlow dataset from the training data.
    """
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
    dataset = dataset.shuffle(buffer_size=len(y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
              
if __name__ == "__main__":

    # 1. Load training captions
    train_captions = load_training_captions()

    # 2. Build tokenizer
    tokenizer = build_tokenizer(train_captions)

    # 3. Find maximum caption length
    max_length = get_max_caption_length(train_captions, tokenizer)

    # 4. Load image features
    with open("data/image_features.pkl", "rb") as f:
        image_features = pickle.load(f)

    # 5. Build training data
    (X1, X2), y = build_training_data(
        train_captions,
        tokenizer,
        image_features,
        max_length
    )

    # 6. Sanity checks
    print("Maximum caption length:", max_length)
    print("Number of training examples:", len(y))
    print("Image feature shape:", X1[0].shape)
    print("Caption input shape:", X2[0].shape)
    print("Target:", y[0])
    dataset = create_dataset(X1, X2, y)

    for inputs, targets in dataset.take(1):
        image_features, caption_inputs = inputs

        print("Image batch shape:", image_features.shape)
        print("Caption batch shape:", caption_inputs.shape)
        print("Target batch shape:", targets.shape)

    vocab_size = len(tokenizer.word_index) + 1

    print("Vocabulary size:", vocab_size)