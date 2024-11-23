from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the trained model and tokenizer
model_path = "/content/drive/MyDrive/archive/models/chatbot_model.h5"
tokenizer_path = "/content/drive/MyDrive/archive/models/tokenizer.pkl"

print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully!")

print("Loading tokenizer...")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded successfully!")

# Define chatbot function
def generate_response(input_text):
    max_len = 20
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, truncating="post")
    
    # Predict the output
    prediction = model.predict(padded_sequence)
    
    # Extract the most probable word at each time step
    predicted_word_indices = np.argmax(prediction, axis=-1).flatten()
    
    # Convert word indices to text
    response = tokenizer.sequences_to_texts([predicted_word_indices.tolist()])[0]
    return response

# Interactive chatbot
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = generate_response(user_input)
    print(f"Bot: {response}")
