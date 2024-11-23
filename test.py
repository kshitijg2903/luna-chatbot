import pandas as pd
df = pd.read_csv("cornell_movie_dialogue_pairs.csv")
print(f"Total dialogue pairs extracted: {len(df)}")
