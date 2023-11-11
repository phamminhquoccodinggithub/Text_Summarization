"""
    Contain some functions to preprocess data
"""
import re

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
# nltk.download('wordnet')


def readFile(file_path):
    """
        Return a list of sentences in file

        @param file_path: file path
    """
    doc = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            doc.append(sent_tokenize(line))
    return [sentence for row in doc for sentence in row]

def normalization(file_path):
    """
        Remove all characters except alphabet and space

        @param file_path: file path
    """
    textNormalization = []
    for line in readFile(file_path):
        sentences = []

        if line.endswith('</s>'):
            line = line.rstrip('</s>')

        sentences.append(line[line.find(">")+1:])

        for sentence in sentences:
            sentence = re.sub("[^\\w\\s]", "", str(sentence))
            sentence = re.sub("[^\\D]", "", sentence)
            textNormalization.append(sentence)
    return textNormalization

def tokenization(file_path):
    """
        Split each sentence into words, remove stop words and empty in list

        @param file_path: file path
    """
    stopWords = set(stopwords.words("english"))
    sentenceTokens = [
        [
            words.lower() for words in sentence.split(' ') \
                if words.lower() not in stopWords and words != ""
        ] for sentence in normalization(file_path)
    ]
    return sentenceTokens

def stemmingWords(contents):
    """
        Remove morphological affixes from words, leaving only the word stem.
        Ex: sized -> size, reference -> refer
    """
    stemmer = PorterStemmer()
    return [[stemmer.stem(word) for word in sentence] for sentence in contents]

def lemmatizingWords(contents):
    """
        Grouping together the inflected forms of a word
        Ex: rocks -> rock, better -> good
    """
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in sentence] for sentence in contents]

def getCleanData(file_path):
    """
        Get final clean data of preprocessing phase

        @param file_path: file path
    """
    contents = tokenization(file_path)
    cleanData = stemmingWords(contents)
    cleanData = lemmatizingWords(cleanData)
    return cleanData

# Test
# FILE_PATH  = "DUC_TEXT/train/d061j"
# print(getCleanData(file_path=FILE_PATH))
