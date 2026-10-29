from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from dataset import (
    load_training_captions,
    build_tokenizer,
    get_max_caption_length,
    build_training_data,
    create_dataset
)
import pickle
import json
image_input = Input(shape=(512,))
caption_input = Input(shape=(37,))

image_projection = Dense(256, activation="relu")(image_input)
caption_embedding = Embedding(
    input_dim=7592,
    output_dim=256,
    mask_zero=True
)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)
combined = Concatenate()([image_projection, caption_lstm])
output = Dense(7592, activation="softmax")(combined)
model = Model(
    inputs=[image_input, caption_input],
    outputs=output
)

loss = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=loss
)

if __name__ == "__main__":

    with open("data/image_features.pkl", "rb") as f:
        image_features = pickle.load(f)

    train_captions = load_training_captions()

    tokenizer = build_tokenizer(train_captions)

    max_length = get_max_caption_length(train_captions, tokenizer)

    (X1, X2), y = build_training_data(
        train_captions,
        tokenizer,
        image_features,
        max_length
    )

    dataset = create_dataset(X1, X2, y, batch_size=32)

    for inputs, targets in dataset.take(1):
        image_batch, caption_batch = inputs

        print("Image batch:", image_batch.shape)
        print("Caption batch:", caption_batch.shape)
        print("Target batch:", targets.shape)

        predictions = model((image_batch, caption_batch))

        print("Predictions:", predictions.shape)

    model.summary()

    # Small test set
    history = model.fit(
        dataset,
        epochs=1
    )

    model.save("models/image_caption_model.keras")

    with open("models/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())