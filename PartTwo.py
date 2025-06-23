# 2a - Read and preprocess the data:
import pandas as pd

def preprocess_hansard(path):
    # Read CSV
    df = pd.read_csv(path)
    
    # i. Rename 'Labour (Co-op)' to 'Labour'
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    
    # Keep only top 4 most frequent parties
    top_parties = df["party"].value_counts().nlargest(4).index.tolist()
    df = df[df["party"].isin(top_parties)]
    df = df[df["party"] != "Speaker"]  # Remove 'Speaker'

    # iii. Keep only rows where speech_class == 'Speech'
    df = df[df['speech_class'] == 'Speech']
    # iv. Remove speeches < 1000 chars
    df = df[df['speech'].str.len() >= 1000]
    print(df.shape)
    return df

#