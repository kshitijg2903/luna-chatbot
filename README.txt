Luna – AI-Powered Chatbot - 
Luna is an AI-powered chatbot developed using deep learning and natural language processing (NLP) techniques. It is trained on the Cornell Movie-Dialogs Corpus, a metadata-rich dataset of fictional conversations extracted from movie scripts. Luna can engage in human-like conversations by leveraging an LSTM-based sequence-to-sequence (seq2seq) model.

Key Features - 
Luna – The Chatbot
Interactive chatbot capable of generating contextually relevant responses.
Trained on 100,000 conversational pairs from the Cornell dataset for efficient computation.
Implements an LSTM-based seq2seq architecture with tokenization, padding, and word embeddings.
Cornell Movie-Dialogs Corpus
Conversations:
220,579 exchanges between 10,292 character pairs.
Movies:
Metadata from 617 unique movie titles.
Characters:
Gender metadata for 3,774 characters.
Position in movie credits for 3,321 characters.
Files in the Repository
1. Chatbot Files
chatbot.py: Python script for interacting with Luna.
train.py: Script to preprocess data and train the chatbot model.
preproccess.py: Data preprocessing pipeline for the Cornell dataset.
models/:
chatbot_model.h5: Trained LSTM model.
tokenizer.pkl: Tokenizer used for text preprocessing.
2. Dataset Files
The Cornell Movie-Dialogs Corpus consists of:

movie_titles_metadata.txt: Metadata for movies.
movie_characters_metadata.txt: Metadata for movie characters.
movie_lines.txt: Text of each utterance.
movie_conversations.txt: Conversation structure (list of utterance IDs).
raw_script_urls.txt: URLs for raw script sources.

1. Preprocessing
Tokenization: Converts text into numerical sequences.
Padding: Ensures uniform input size for the LSTM model.
Embedding: Maps words into dense vectors that represent their semantic meaning.
2. LSTM Seq2Seq Model
Encoder-Decoder architecture:
Encoder compresses the input into a fixed-length context vector.
Decoder generates the output word by word, leveraging the context vector.
3. Training
Loss Function: Sparse categorical cross-entropy.
Optimizer: Adam optimizer for efficient training.

Future Improvements
Train on the full Cornell dataset for richer responses.
Use pre-trained embeddings (e.g., GloVe or Word2Vec).
Deploy Luna as a web application or API.
