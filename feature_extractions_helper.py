import nltk
import re
import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk import sent_tokenize, word_tokenize, pos_tag
from emoji.unicode_codes import UNICODE_EMOJI
from nltk.tokenize import RegexpTokenizer
import string

FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "u", "ur", "urs"}
THIRD_PERSON_PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
}
PUNCTUATION_LIST = list(string.punctuation)

###### Attribution for positive words dictionary: https://gist.github.com/mkulakowski2/4289437
positive_words = []
with open("data/helper_dicts/positive-words.txt") as f:
    for i in range(35):
        f.readline()

    for i in range(35, 2042):
        positive_words.append(f.readline().strip("\n"))


###### Attribution for negative words dictionary: https://gist.github.com/mkulakowski2/4289441
negative_words = []
with open("data/helper_dicts/negative-words.txt") as f:
    for i in range(35):
        f.readline()

    for i in range(35, 2042):
        negative_words.append(f.readline().strip("\n"))

###### This sentiment dictionary for slang words are from SlangSD, made by Liang Wu (Student, ASU), Fred Morstatter (Student, ASU), Huan Liu (Professor, ASU).
###### Link for SlangSD: http://liangwu.me/slangsd/
slang_df = pd.read_csv(
    "data/helper_dicts/SlangSD.txt", delimiter="\t", names=["sentiment"]
)

########################################################
# The verb dictionary used in this code is adapted from https://github.com/monolithpl/verb.forms.dictionary
########################################################
verb_dict = pd.read_csv(
    "data/helper_dicts/verbs-dictionaries.csv",
    delimiter="\t",
    header=None,
    names=["present_simple", "past_simple", "past_participle", "present_participle"],
)
past_simples = verb_dict["past_simple"].to_dict().values()

tokenizer = RegexpTokenizer(r"\w+")


def get_avg_pos_words(text):
    """Calculate the positive words ratio per text."""
    tokens = word_tokenize(text)
    pos_count = 0

    for word in tokens:
        if word in positive_words:
            pos_count += 1

    return pos_count / len(tokens)


def get_avg_neg_words(text):
    """Calculate the negative words ratio per text."""
    tokens = word_tokenize(text)
    neg_count = 0

    for word in tokens:
        if word in negative_words:
            neg_count += 1

    return neg_count / len(tokens)


def count_past_tense(text):
    """Count the number of past tense in the text."""
    counter = 0
    tokens = word_tokenize(text.lower())
    tagged_words = pos_tag(tokens)

    for word, pos in tagged_words:
        if pos[0] == "V":
            if word in past_simples:
                counter += 1

    return counter


def count_future_tense(text):
    """Count the number of future tense in the text."""
    future_form = {"'ll", "will", "wo"}
    counter = 0
    tokens = word_tokenize(text.lower())
    tagged_words = pos_tag(tokens)

    for word, pos in tagged_words:
        if pos == "MD":
            if word in future_form:
                counter += 1
    return counter


def count_first_person_pro(text):
    """Count the number of first-person pronouns in the text."""
    return len(
        re.findall(r"\b({})\b".format("|".join(FIRST_PERSON_PRONOUNS)), text.lower())
    )


def count_second_person_pro(text):
    """Count the number of second-person pronouns in the text."""
    return len(
        re.findall(r"\b({})\b".format("|".join(SECOND_PERSON_PRONOUNS)), text.lower())
    )


def count_third_person_pro(text):
    """Count the number of third-person pronouns in the text."""
    return len(
        re.findall(r"\b({})\b".format("|".join(THIRD_PERSON_PRONOUNS)), text.lower())
    )


def count_coord_conj(text):
    """Count the number of coordinating conjunctions in the text."""
    token_tag_pairs = pos_tag(word_tokenize(text.lower()))
    return len([p[1] for p in token_tag_pairs if p[1] == "CC"])


def count_commas(text):
    """Count the number of commas in the text."""
    counter = 0
    tokens = word_tokenize(text.lower())

    for word in tokens:
        if word == ",":
            counter += 1
    return counter


def count_multi_punc(text, include_dots=True):
    """Count the number of multi punctuation characters in the text."""
    counter = 0

    if include_dots:
        pattern = r"(\!{2,}|\?{2,}|\.{3,})"
    else:
        pattern = r"(\!{2,}|\?{2,})"

    compiled = re.compile(pattern)

    for match in compiled.finditer(text.lower()):
        if match:
            counter += 1

    return counter


def get_avg_slang_sent(text):
    """Calculate the slang ratio per text."""
    slang_sent = 0
    split = text.split(" ")
    tokens = [token.strip("".join(PUNCTUATION_LIST)) for token in split]

    for word in tokens:
        if word in slang_df.index and word not in stopwords.words("english"):
            slang_sent += slang_df.loc[word]["sentiment"]

    return slang_sent / len(tokens)  ## avg vs just raw sum


def count_tags(text):
    """Count the number of common nouns, proper nouns, adverb, and wh- words"""

    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)

    common_noun_count = 0
    proper_noun_count = 0
    adv_count = 0
    wh_count = 0

    for word, pos in tagged_words:
        if pos == "NN" or pos == "NNS":
            common_noun_count += 1
        elif pos == "NNPS" or pos == "NNP":
            proper_noun_count += 1
        elif pos == "RB" or pos == "RBR" or pos == "RBS":
            adv_count += 1
        elif pos == "WP" or pos == "WDT" or pos == "WRB":
            wh_count += 1
    return common_noun_count, proper_noun_count, adv_count, wh_count


def count_cap_words(text):
    """Count the amount of capitalized words in a text"""
    cap_words = 0
    words = word_tokenize(text)
    for word in words:
        if word.isupper():
            cap_words = cap_words + 1
        else:
            cap_words = cap_words
    return cap_words


def avg_len_sent(text):
    """Calculates the average length of sentences, in tokens"""
    token_count = len(text.split())
    sent_count = text.count(". ") + 1
    if sent_count != 0:
        return token_count / sent_count
    else:
        return 0


def avg_len_tokens(text):
    """Calculates the average length of tokens, excluding punctuation, in characters"""
    token_with_no_punc = tokenizer.tokenize(text.lower())
    if len(token_with_no_punc) != 0:
        return len("".join(token_with_no_punc)) / len(token_with_no_punc)
    else:
        return 0


def num_of_sent(text):
    """Counts the number of sentences"""
    return text.count(". ") + 1

def num_slang_acronym(text):
    '''Count the amount of slang acronyms in a text'''
    return len(re.findall(r"\b({})\b".format("|".join(SLANG)), text.lower()))