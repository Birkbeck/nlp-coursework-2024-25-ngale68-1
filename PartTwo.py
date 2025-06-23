# 2a - Read and preprocess the data:
import pandas as pd

def preprocess_hansard(path):
    # Read CSV
    df = pd.read_csv(path)
    
    # i. Rename 'Labour (Co-op)' to 'Labour'
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    
    # ii. Keep only top 4 most frequent parties
    top_parties = df["party"].value_counts().nlargest(4).index.tolist()
    df = df[df["party"].isin(top_parties)]
    df = df[df["party"] != "Speaker"]  # Remove 'Speaker'

    # iii. Keep only rows where speech_class == 'Speech'
    df = df[df['speech_class'] == 'Speech']
    
    # iv. Remove speeches < 1000 chars
    df = df[df['speech'].str.len() >= 1000]
    print(df.shape)
    return df


# 2b - Vectorise and split the data:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def vectorise_and_split(df):
    vectoriser = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectoriser.fit_transform(df['speech'])
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=26, stratify=y
    )
    return X_train, X_test, y_train, y_test, vectoriser


if __name__ == "__main__":
    # 2a
    path = r"p2-texts\hansard40000.csv"
    df = preprocess_hansard(path)