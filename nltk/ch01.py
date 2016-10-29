# Source: http://www.nltk.org/book/ch01.html
from __future__ import division

# Section 1: Computing with Language: Texts and Words
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

# Section 2: A Closer Look at Python: Texts as Lists of Words
# --------------------------------------------------

# Lists
#
# Text are represented in a list with each word represented as an
# element within the list.

# we can add the list together
sent1 + sent2

# 2.2 Indexing Lists

# Obtain words from index
text4[173]
text4[1000:1020]

# Obtain index from list
text4.index('awaken')

# Joining words
' '.join(['this', 'is', 'a', 'test'])
'_ '.join(['this', 'is', 'a', 'test'])

# Splitting words
'this is a test'.split()

# Section 3: Computing with Language: Simple Statistics
# --------------------------------------------------

# Frequency distribution
fdist1 = FreqDist(text1)
print(fdist1)

fdist1.most_common(10)
fdist1['whale']

fdist2 = FreqDist(text2)
fdist2.most_common(10)

# Ploting the cumulative count
fdist1.plot(50, cumulative=True)

# Hapaxes, words that only appear once. They provide a better context
# about the article.
fdist1.hapaxes()

# Extract long words which has more than 15 characters
vocab = set(text1)
long_words = [word for word in vocab if len(word) > 15]

# Obtain the frequency distribution of long words and also eliminating
# rare long words
fdist5 = FreqDist(text5)
sorted([word for word in set(text5) if len(word) > 7 and fdist5[word] > 7])

# 3.3 Collocation and Bigram

# Bigrams are two words that occur together.
#
# NOTE (Michael): Somehow the bigrams function can not be found.

# list(bigrams(['more', 'is', 'said', 'than', 'done']))

# Collacations are bigrams which occur frequently.
text4.collocations()

# 3.4 Counting other things

# Count length of words
[len(w) for w in text1]

# Word length frequency
fdist = FreqDist(len(w) for w in text1)

# Most common length
fdist.most_common()
fdist.max()
fdist.freq(2)

# Section 8: Exercise
# --------------------------------------------------

# 1. Try using the Python interpreter as a calculator, and typing
# expressions like 12 / (4 + 1).
12 / (4 + 1)

# 2. Given an alphabet of 26 letters, there are 26 to the power 10, or 26
# ** 10, ten-letter strings we can form. That works out to
# 141167095653376. How many hundred-letter strings are possible?

26 ** 100

# 3. The Python multiplication operation can be applied to lists. What
# happens when you type ['Monty', 'Python'] * 20, or 3 * sent1?

3 * 'abc'

# 4. Review 1 on computing with language. How many words are there in
# text2? How many distinct words are there?

len(text2)
len(set(text2))

# 5. Compare the lexical diversity scores for humor and romance
# fiction in 1.1. Which genre is more lexically diverse?


# 6. Produce a dispersion plot of the four main protagonists in Sense and
# Sensibility: Elinor, Marianne, Edward, and Willoughby. What can you
# observe about the different roles played by the males and females in
# this novel? Can you identify the couples?

dispersion_plot(text2, ['Elinor', 'Marianne', 'Edward', 'Willoughby'])

# 7.Find the collocations in text5.

text5.collocations()

# 8. Consider the following Python expression: len(set(text4)). State
# the purpose of this expression. Describe the two steps involved in
# performing this computation.

# It gives the number of unique words, the set function removes
# duplicated words while the len function makes the count.
