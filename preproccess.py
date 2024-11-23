import os
import re
import pandas as pd

lines_file = "/content/drive/MyDrive/archive/movie_lines.txt"
conversations_file = "/content/drive/MyDrive/archive/movie_conversations.txt"


def load_lines(file_path):
    lines = {}
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:  # Ensure proper formatting
                lines[parts[0]] = parts[4]
    return lines

def load_conversations(file_path):
    conversations = []
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 4:  # Ensure proper formatting
                # Convert the string list to a Python list
                line_ids = eval(parts[3])
                conversations.append(line_ids)
    return conversations

# Extract dialogue pairs
def extract_conversations(lines, conversations):
    input_responses = []
    for conv in conversations:
        for i in range(len(conv) - 1):  # Pair consecutive lines
            try:
                input_responses.append((lines[conv[i]], lines[conv[i + 1]]))
            except KeyError as e:
                print(f"Missing line ID: {e}")  # Log missing line IDs
    return input_responses

lines = load_lines(lines_file)
conversations = load_conversations(conversations_file)
dialogue_pairs = extract_conversations(lines, conversations)


df = pd.DataFrame(dialogue_pairs, columns=["input", "response"])
df.to_csv("cornell_movie_dialogue_pairs.csv", index=False)
print("Preprocessing Complete! Dataset saved as cornell_movie_dialogue_pairs.csv.")
