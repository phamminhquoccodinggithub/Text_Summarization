"""
    Contain some functions to preprocess data
"""
import re
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')


def get_all_files(dir_path):
    """
        Return a list of files in folder

        @param dir_path: path of folder
    """
    return sorted(os.listdir(dir_path))


def read_file(file_path):
    """
        Return a list of sentences in file

        @param file_path: file path
    """
    with open(file_path, encoding="utf-8") as f:
        contents = f.readlines()
    return contents


def normalization(file_path):
    """
        Remove all characters except alphabet and space

        @param file_path: file path
    """
    text_normalization = []
    for line in read_file(file_path):
        sentences = []

        if line.endswith('</s>\n'):
            line = line.rstrip('</s>\n')

        sentences.append(line[line.find(">")+1:])

        for sentence in sentences:
            sentence = re.sub("[^\\w\\s]", "", str(sentence))
            sentence = re.sub("[^\\D]", "", sentence)
            text_normalization.append(sentence)
    return text_normalization


def tokenization(file_path):
    """
        Split each sentence into words, remove stop words and empty in list

        @param file_path: file path
    """
    stop_words = set(stopwords.words("english"))
    sentence_tokens = [
        [
            words.lower() for words in sentence.split(' ') \
                if words.lower() not in stop_words and words != ""
        ] for sentence in normalization(file_path)
    ]
    return sentence_tokens


def stemming_words(contents):
    """
        Remove morphological affixes from words, leaving only the word stem.
        Ex: sized -> size, reference -> refer

        @param contents: list of sentences
    """
    stemmer = PorterStemmer()
    return [[stemmer.stem(word) for word in sentence] for sentence in contents]


def lemmatizing_words(contents):
    """
        Grouping together the inflected forms of a word
        Ex: rocks -> rock, better -> good

        @param contents: list of sentences
    """
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in sentence] for sentence in contents]


def get_clean_data(file_path):
    """
        Get final clean data of preprocessing phase

        @param file_path: file path
    """
    contents = tokenization(file_path)
    clean_data = lemmatizing_words(contents)
    return clean_data


# Test
# FILE_PATH  = "DUC_TEXT/train/d061j"
# print(get_clean_data(file_path=FILE_PATH))
