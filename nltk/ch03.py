# Source : http://www.nltk.org/book/ch03.html

from __future__ import division  # Python 2 users only
import nltk
import re
import pprint
from nltk import word_tokenize

# 3.1 Accessing Text from the Web and from Disk

# Download the English translation of Crime and Punishment.
import urllib2
url = "http://www.gutenberg.org/files/2554/2554.txt"
response = urllib2.urlopen(url)
raw = response.read().decode('utf8')

# Tokenise the text. Tokenise means to split the text string into
# words and punctuations and remove unwanted spaces.
tokens = word_tokenize(raw)

# Make it a Text class
text = nltk.Text(tokens)

# The text contain preface and disclaimer. Chapter 1 starts from the
# following.
text[1024:1062]

# Have a look a the collocations
text.collocations()

# The text contains the 'Project Gutenberg' in the collocation as it
# is part of the header. We will subset the text to only the part that
# we are intersted.

starting_string = raw.find("PART I")
ending_string = raw.rfind("End of Project Gutenberg's Crime")
raw = raw[starting_string:ending_string]

# Searching tokenised text
from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a> (<.*>) <man>")

chat = nltk.Text(nps_chat.words())
chat.findall(r"<.*> <.*> <bro>")

chat.findall(r"<l.*>{3,}")

# 3.6 Normalizing Text

raw = """DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.  Supreme executive power derives
from a mandate from the masses, not from some farcical aquatic
ceremony."""
tokens = word_tokenize(raw)

# Stemming is a process to strip the affixes to it's root and
# standardise the words. Therea are different various stemmers each
# with it's own strength and weakness.
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]

# Lemmatisation, it is the same as stemming but also checks whether
# the word is in the dictionary and thus is a lot slower.
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]
