import re
from nltk.stem import PorterStemmer

STOP_WORDS = {'for', 't', 'ma', 'here', 'm', "don't", "wasn't", 'too', "isn't", 'with', 'where', 'having', 'or', 'hers',
              'their', 'between', 'are', 'through', 'am', 'each', 'any', 'yours', "hadn't", "you're", 'all', 'were',
              'what', 'being', "wouldn't", 'wouldn', 'there', 'such', 'hadn', 'most', 'do', 'her', 'doesn', 'which',
              'it', 'these', 'off', 'shouldn', "you've", 'that', 'be', 'who', 'at', 'against', 'did', 'has', 'from',
              'the', 'few', 'we', "should've", 'in', 're', "won't", 'should', 'mightn', 'yourself', "shouldn't", 'when',
              "couldn't", "aren't", 'isn', 'was', 'this', 'is', 'just', 'some', "hasn't", 'on', 'above', 'while',
              'then', 'our', 'if', 'him', 'wasn', 'very', 'have', 'shan', 'doing', 'ain', 'by', 'yourselves', 'whom',
              'don', 'herself', 'you', 'nor', 'and', 'now', 'its', 'other', 'than', 'both', 'as', 'weren', 'only',
              "mustn't", 'needn', "that'll", 've', 'can', 'because', 'haven', 'them', 'does', 'until', 'why', 'an',
              'ours', 'll', "weren't", 'his', 'been', 'd', 'me', 'into', 'ourselves', 'theirs', 'my', 'own', "doesn't",
              'how', 'she', 'to', 'myself', 'up', 'down', 'won', "she's", 'o', 'those', 's', "shan't", 'more', 'during',
              "haven't", 'itself', 'i', 'himself', "you'd", 'had', "didn't", "needn't", 'a', 'same', 'out', 'couldn',
              'before', 'mustn', 'no', 'so', 'will', 'your', 'aren', 'about', 'y', 'over', 'after', 'he', 'hasn',
              'didn', 'below', 'but', "mightn't", 'further', 'themselves', 'again', 'not', "you'll", "it's", 'of',
              'under', 'they', 'once', '', "cf", "fibrosis", "cystic", "patient", "found", "non", "relation", "ability",
              "whether"}

PHRASES = {"essential fatty", "pancreatic insufficiency", "hereditary diseases", "reproductive system",
           "genetic counseling", "prescribed therapy", "sweat tests", "pilocarpine iontophoresis",
           "titrimetric analysis", "gibson-cooke", "erroneous results", "insulin secretion", "insulin metabolism",
           "prostaglandin metabolism", "polyamine metabolism", "methylation metabolism", "vitamin a"}

ps = PorterStemmer()


def clean_word(w):
    return ps.stem(re.sub('[^A-Za-z0-9]+', '', w.lower()))


def tokenize(s):
    tokens = re.split("[\s\\\/\-\n]", s)
    for p in PHRASES:
        if p in s.lower():
            tokens.append(s)
    return tokens


def clean_token_list(words):
    words = map(clean_word, words)
    words = list(filter(lambda x: x not in STOP_WORDS, words))
    return words


def keep_vocabulary(words, vocabulary):
    return [w for w in words if w in vocabulary]
