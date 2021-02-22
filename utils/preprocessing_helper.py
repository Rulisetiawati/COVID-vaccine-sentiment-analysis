import string

# import demoji
import nltk
import re

from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# from emoji.unicode_codes import UNICODE_EMOJI

nltk.download("wordnet")


def preprocessing(text):
    """Exclude punctuations and digits from text."""
    text = process_tweets(text)
    text = text.lower()
    exclude = string.punctuation + string.digits
    for i in exclude:
        text = text.replace(i, "")
    return text


def process_tweets(text):
    """Exclude mentions, urls, and html reference characters in a string using regular expression"""
    text = re.sub("(\@|https:\/\/)\S+", "", text)  # remove mentions and urls
    text = re.sub(r"&[a-z]+;", "", text)  # exclude html reference characters
    return text


def lemmatize(text):
    """Lemmatize tweets by WordNetLemmatizer"""
    lemma_list = []
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    words = word_tokenize(text)

    for word in words:
        lemma = lemmatizer.lemmatize(word, "n")
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, "v")
        lemma_list.append(lemma)

    return " ".join(lemma_list)


# def convert_emoji_to_text(tweet):
#     """Convert emoji into text description in the tweet."""
#     tokens = tweet.split()
#     for i, token in enumerate(tokens):
#         if token in UNICODE_EMOJI:
#             emo_desc = demoji.findall(token)[token]
#             new_rep = "_".join(emo_desc.split(":")[0].split())
#             tokens[i] = new_rep
#     return " ".join(tokens)