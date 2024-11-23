import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pickle

# Step 1: Load preprocessed data
print("Loading data...")
data = pd.read_csv("/content/drive/MyDrive/archive/cornell_movie_dialogue_pairs.csv")
print("Data loaded. Sample:")
print(data.head())

# Step 2: Tokenization
vocab_size = 30000  # Vocabulary size
oov_token = "<OOV>"  # Token for out-of-vocabulary words
max_len = 20  # Maximum sequence length

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(data["input"])  # Fit tokenizer on input sentences

# Debugging: Check tokenizer mappings
print("Sample word-to-index mapping (first 10):")
print({k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:10]})
print("Sample index-to-word mapping (first 10):")
print({k: tokenizer.index_word[k] for k in list(tokenizer.index_word)[:10]})

# Tokenize and pad input sequences
input_sequences = tokenizer.texts_to_sequences(data["input"])  # Convert text to sequences
padded_input = pad_sequences(input_sequences, maxlen=max_len, truncating="post")  # Pad sequences

# Tokenize and pad response sequences
response_sequences = tokenizer.texts_to_sequences(data["response"])
padded_response = pad_sequences(response_sequences, maxlen=max_len, truncating="post")

# Debugging: Inspect a few sequences
print("Example tokenized input sequence:")
print(input_sequences[:5])
print("Example padded input sequence:")
print(padded_input[:5])

# Use only the first 100,000 samples for faster training
print("Reducing dataset to 100,000 samples...")
padded_input = padded_input[:100000]
padded_response = padded_response[:100000]

print(f"Padded input shape: {padded_input.shape}")
print(f"Padded response shape: {padded_response.shape}")

# Debugging: Check for `<OOV>` tokens
oov_count = sum(1 for seq in input_sequences for token in seq if token == tokenizer.word_index[oov_token])
print(f"Number of <OOV> tokens in input sequences: {oov_count}")

# Step 3: Model definition
print("Defining the model...")
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),  # Embedding layer
    LSTM(128, return_sequences=True),  # First LSTM layer
    LSTM(128, return_sequences=True),  # Second LSTM layer
    TimeDistributed(Dense(vocab_size, activation="softmax"))  # TimeDistributed output layer
])

# Step 4: Compile the model
print("Compiling the model...")
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Step 5: Add learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)

# Step 6: Train the model with 2 epochs and batch size 128
print("Starting training...")
model.fit(padded_input, padded_response, epochs=2, batch_size=128, callbacks=[lr_scheduler])

# Step 7: Save the model and tokenizer
print("Saving the model and tokenizer...")
model.save("/content/drive/MyDrive/archive/models/chatbot_model.h5")
with open("/content/drive/MyDrive/archive/models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training complete. Model and tokenizer saved.")