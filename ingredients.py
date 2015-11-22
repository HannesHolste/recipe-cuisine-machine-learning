"""
Parsing for ingredient lines of recipes.
from https://github.com/JoshRosen/cmps140_creative_cooking_assistant
"""
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import pyparsing


LEMMATIZER = WordNetLemmatizer()



def normalize_ingredient_name(ingredient_name):
    """
    Normalizes an ingredient name, removing pluralization.
    >>> normalize_ingredient_name('eggs')
    'egg'
    >>> normalize_ingredient_name('bing cherries')
    'bing cherry'
    """
    words = ingredient_name.lower().strip(' *').split()
    return ' '.join(LEMMATIZER.lemmatize(w) for w in words)
