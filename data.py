import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer

#Import instances and drop unused columns
twitter_instances = pd.read_csv("dataset/gender-classifier-DFE-791531.csv", encoding='latin-1')
twitter_instances.drop(columns=["_unit_id", "_golden", "_unit_state", "_trusted_judgments", "_last_judgment_at", "gender:confidence", "profile_yn", "profile_yn:confidence", "created", "fav_number", "gender_gold", "link_color", "profile_yn_gold", "profileimage", "retweet_count", "sidebar_color", "text"], inplace=True)

#gender is gold label
#description features

#remove rows which don't have a gender label
no_gender = twitter_instances[twitter_instances["gender"].isnull()].index
twitter_instances.drop(no_gender, inplace=True)

#remove rows without description
no_desciption = twitter_instances[twitter_instances["description"].isnull()].index
twitter_instances.drop(no_desciption, inplace=True)

#print(len(twitter_instances))
#The remaining dataset has 16306 instances

#Divide data in training and test sets (division 80/20)
#TODO: Need to split up all separate words and put them in as separate features
training_instances = twitter_instances["description"][:13045] #13045 instances
test_instances = twitter_instances["description"][13045:] #3179 instances

training_labels = twitter_instances["gender"][:13045]
test_labels = twitter_instances["gender"][13045:]


#splits decriptions into tokens; returns list of lists of tokens
def split_description(descriptions):
    tokenized_descriptions = []
    for description in descriptions:
        sentences_nltk = sent_tokenize(description)
        tokens_per_description = []
        for sentence_nltk in sentences_nltk:
            tokens_per_description = word_tokenize(sentence_nltk)
            tokens_per_description = remove_punctuation(tokens_per_description)
            tokens_per_description = apply_lemma(tokens_per_description)
            #maybe don't apply because it could be meaningful
            #tokens_per_description = remove_stopwords(tokens_per_description)
        tokenized_descriptions.append(tokens_per_description)
    return tokenized_descriptions


def remove_stopwords(description):
    english_stopwords = stopwords.words('english')
    set_english_stopwords = set(english_stopwords)
    without_stopwords = []
    for token in description:
        if token.lower() not in set_english_stopwords:
            without_stopwords.append(token)
    return without_stopwords


def remove_punctuation(description):
    without_punctuation = []
    for token in description:
        translation = token.translate({ord(char): '' for char in string.punctuation})
        if translation != '':
            without_punctuation.append(translation)
    return without_punctuation

def apply_lemma(description):
    lemmatized_description = []
    wordnet = WordNetLemmatizer()
    for token in description:
        lemmatized_description.append(wordnet.lemmatize(token))
    return lemmatized_description



print(split_description(training_instances[:10]))