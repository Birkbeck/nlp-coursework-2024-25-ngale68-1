#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    from nltk.tokenize import sent_tokenize, word_tokenize

    sentences = sent_tokenize(text)
    words = [w for w in word_tokenize(text) if w.isalpha()]  # Filter out punctuation
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syl(w, d) for w in words)

    if num_sentences == 0 or num_words == 0:
        return 0.0

    fkgl = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    return fkgl


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    import re

    word = word.lower()
    if word in d:
        return [len([y for y in x if y[-1].isdigit()]) for x in d[word]][0]  # CMU dict: each pronunciation is a list of phonemes, vowels have digits
    else:
        return len(re.findall(r"[aeiouy]+", word))  # Count vowel clusters as syllables


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    for file in Path(path).glob("*.txt"):
        name = file.stem # Expect filename format: title-author-year.txt
        try:
            title, author, year = name.rsplit("-", 2) # Split filename format: title-author-year
            year = int(year)
        except ValueError:
            continue  # Skip files that do not match the expected format
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        data.append({"title": title, "author": author, "year": year, "text": text})
    df = pd.DataFrame(data, columns=["text", "title", "author", "year"])
    df = df.sort_values("year", ignore_index=True)  # Sort by year, ignoring index
    return df


def flesch_kincaid(df):
    """Returns a dictionary mapping title to Flesch-Kincaid grade level."""
    import nltk
    cmudict = nltk.corpus.cmudict.dict()
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    import os

    os.makedirs(store_path, exist_ok=True) # Ensure output directory exists

    parsed_docs = []
    max_length = nlp.max_length

    for text in df["text"]:
        # If text is too long, split into chunks and parse separately
        if len(text) > max_length:
            docs = []
            for i in range(0, len(text), max_length - 1000):  # leave a margin
                chunk = text[i:i + max_length - 1000]
                docs.append(nlp(chunk))
            # Concatenate all tokens from chunks into a single Doc
            from spacy.tokens import Doc
            all_tokens = []
            for doc in docs:
                all_tokens.extend([t for t in doc])
            # Create a new Doc from all tokens (metadata will be lost)
            parsed_doc = Doc(nlp.vocab, words=[t.text for t in all_tokens])
        else:
            parsed_doc = nlp(text)
        parsed_docs.append(parsed_doc)

    df = df.copy()
    df["parsed"] = parsed_docs

    # Save to pickle
    pickle_path = store_path / out_name
    df.to_pickle(pickle_path)

    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    from nltk import word_tokenize
    import string

    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens if w.isalpha()] # Filter out punctuation and make all tokens lowercase
    if not words:
        return 0.0
    types = set(words)
    ttr = len(types) / len(words)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    # Find subjects of the specified verb (any tense)
    subjects = []
    for tok in doc:
        # Check if token is a verb with the correct lemma
        if tok.pos_ == "VERB" and tok.lemma_.lower() == verb.lower():
            # Find its subject(s)
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(child.text.lower())
    return Counter(subjects).most_common(10)



def object_counts(doc):
    """Extracts the most common syntactic objects in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(object_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

