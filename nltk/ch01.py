# Source: http://www.nltk.org/book/ch01.html
from __future__ import division

# Section 1
# --------------------------------------------------

# 1.2 Getting started with NLTK

import nltk
nltk.download('book')

from nltk.book import *


# 1.3 Searching Text
text1.concordance('monstrous')
text1.concordance('live')

# Similar words, these are like synonym but derived from the context.
text1.similar('monstrous')
text2.similar('monstrous')
text1.similar('live')

# We can also obtain the shared context between multiple phrases.
text2.common_contexts(["monstrous", "very"])

# Dispersion plot, this plot displayes the location of occurence within a text.
from nltk.draw.dispersion import dispersion_plot
dispersion_plot(text4, ["citizens", "democracy",
                        "freedom", "duties", "America"])

# We can also generate text based on the article.
#
# NOTE(Michael): This feature does not work in NLTK 3.0.
# text3.generate()

# 1.4 Counting Volcabulary
len(text3)

# We can also obtain the unique words (or token) in the text.
#
# Esentially, what we are doing is convert the text to a set, which
# contains only unique entries.
sorted(set(text3))

# We can also calculate the number of unique token and lexical richness
len(set(text3))

# First of if we are using Python 2, we can import division where all
# division results are forced to be floats.
len(set(text3)) / len(text3)

# Let's count the occurence and also the observation frequency.
text3.count("smote")
text3.count("smote") / len(text3) * 100


def lexical_diversity(text):
    """Function to calculate the diversity of the text. It is a simple
    statistics such that the number of unique words is divided by the
    number of total words

    """
    return len(set(text)) / len(text)


def percentage(count, total):
    """
    A simple function to calculate percentage
    """
    return count / total * 100


def token_percentage(word, text):
    """This function takes a particular word, and the calculates the
    observation frequency in percentage within a given text.

    """
    word_count = text.count(word)
    text_len = len(text)
    return percentage(word_count, text_len)

# Test the functions
print lexical_diversity(text3)
print lexical_diversity(text5)
print token_percentage('a', text4)

# Seciton 2
# --------------------------------------------------
