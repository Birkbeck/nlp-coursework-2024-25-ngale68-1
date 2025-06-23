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

def vectorise_and_split(df, vectoriser):
    X = vectoriser.fit_transform(df['speech'])
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=26, stratify=y
    )
    return X_train, X_test, y_train, y_test, vectoriser


# 2c - Train a classifier:
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

def classifiers_eval(X_train, X_test, y_train, y_test):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=26)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f1_score(y_test, y_pred_rf, average='macro'))
    print(classification_report(y_test, y_pred_rf))

    # SVM
    svm = SVC(kernel='linear', random_state=26)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f1_score(y_test, y_pred_svm, average='macro'))
    print(classification_report(y_test, y_pred_svm))


# 2e - Custom tokenizer:
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]


if __name__ == "__main__":
    # 2a
    path = r"p2-texts\hansard40000.csv"
    df = preprocess_hansard(path)

    # 2b
    vectoriser = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train, X_test, y_train, y_test, vectoriser = vectorise_and_split(df, vectoriser)

    # 2c
    classifiers_eval(X_train, X_test, y_train, y_test)

    # 2d
    vectoriser_ng = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,3))
    X_train_ng, X_test_ng, y_train_ng, y_test_ng, vectoriser_ng = vectorise_and_split(df, vectoriser_ng)
    classifiers_eval(X_train_ng, X_test_ng, y_train_ng, y_test_ng)

    # 2e
    vectoriser_custom = TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        max_features=3000,
        ngram_range=(1, 3),   # unigrams, bigrams, trigrams
        min_df=3,             # ignore terms in fewer than 3 docs
        max_df=0.8            # ignore terms in more than 80% of docs
    )
    X_train_c, X_test_c, y_train_c, y_test_c, _ = vectorise_and_split(df, vectoriser_custom)
    classifiers_eval(X_train_c, X_test_c, y_train_c, y_test_c)