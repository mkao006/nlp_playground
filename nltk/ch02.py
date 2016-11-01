# Source : http://www.nltk.org/book/ch02.html

from __future__ import division
import nltk
from nltk.book import *

# Section 1: Accessing Text Corpora
# --------------------------------------------------

# 1.1 Gutenber Corpus
nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)

# Get concordance, first of all, we need to convert to nltk text class.
emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance('surprize')

# Get some basic statistics of each article.
for fileid in nltk.corpus.gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars / num_words), round(num_words /
                                              num_sents), round(num_words / num_vocab), fileid)

# NOTE (Michael): The function does not calculate the characters and
#                 the setences correctly.
#
# def summarise_text(text):
#     num_chars = len(''.join(text))
#     num_words = len(text)
#     num_sents = len(''.join(text).split('.'))
#     num_vocab = len(set(w.lower() for w in text))
#     print(round(num_chars / num_words),
#           round(num_words / num_sents),
#           round(num_words / num_vocab))

# 1.2 web and chat text

# Up till this point, we have been analysing text which are establish
# literature. Let us explore some informal texts.

from nltk.corpus import webtext

for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')


# Chat text
from nltk.corpus import nps_chat

chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]

# 1.3 Brown corpus
#
# The Brown corpus was the first million word electronic corpus of
# English, created in 1961 by Brown University.

from nltk.corpus import brown
brown.categories()

brown.words(categories='news')
brown.words(fileids=['cg22'])
brown.sents(categories=['news', 'editorial', 'reviews'])

# The brown corpus is a convinient source to study the difference
# between genres, known as stylistics.
#
# Let's have a look at the verbs used.
news_text = brown.words(categories='news')
fdist = FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']


for m in modals:
    print(m + ':', str(fdist[m]))

# We can also create the frequency table conditioned by genre then
# produce a frequency table.
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance',
          'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']

cfd.tabulate(conditions=genres, samples=modals)

# 1.4 Reuters Corpus

# This is a dataset containing articles from Reuters. The advantage of
# this document is that it is already split into test/training sets
# for modelling.

from nltk.corpus import reuters
reuters.fileids()
reuters.categories()

# Unlike the Brown corpus where each article can belong to a single
# corpus, the articles in Reuters can and often has multiple genre.

reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])
reuters.fileids(['barley', 'corn'])

# 1.5 Inaugural Address
from nltk.corpus import inaugural
inaugural.fileids()

# Extract the year from the field ids
[fileid[:4] for fileid in inaugural.fileids()]

cfd_address = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for word in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if word.lower().startswith(target))

cfd_address.plot()

# 1.7 Corpora in Other Languages

# NOTE (Michael): Some of the corporas below require additional
#                 download.
nltk.corpus.cess_esp.words()
nltk.corpus.floresta.words()
nltk.corpus.indian.words('hindi.pos')
nltk.corpus.udhr.fileids()
nltk.corpus.udhr.words('Javanese-Latin1')[11:]

# Shown below is the cumulative percentage of word length.
#
# We can see that ibiobio has the highest acent rate and thus majority
# of the words are short, while on the other hand the Inuktiku has a
# slow acend rate which means the average word length are considrably
# longer.
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
             'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)

# Section 2: Conditional Frequency Distributions

# 2.2 Counting words by genre
from nltk.corpus import brown
cfd_brown = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))

genre_words = [(genre, word)
               for genre in ['news', 'romance']
               for word in brown.words(categories=genre)]

# 2.3 Plotting and tabulate distributions
from nltk.util import bigrams
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
        'and', 'the', 'earth', '.']
list(bigrams(sent))


# 2.4 Generate random text

def generate_model(cfdist, word, num=15):
    """This function takes a frequency distribution and generates a rand
    om text with the specified length.


    Basically, the function takes the most likely bigram and take the
    word with the highest frequency count following the current word.
    """
    for i in range(num):
        print(word)
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

cfd['living']
generate_model(cfd, 'living')

# Section 4: Lexical Resources

# 4.1 Wordlist corpora


def unusual_words(text):
    '''This function takes a text and then return unusual words.

    The function first extracts all the vocab, and the deduct the set
    of words that are deemed as common in the nltk.corpus.words.words
    corpra.

    '''
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
unusual_words(nltk.corpus.nps_chat.words())

# Stop words are high frequency words which bears little meaning or
# context.
from nltk.corpus import stopwords
stopwords.words('english')


def nonstopword_fraction(text):
    '''The function calculates the fraction of the article that are non
    stopwords

    '''
    stopwords = nltk.corpus.stopwords.words('english')
    nonstopwords = [word
                    for word in text
                    if word.lower() not in stopwords]
    return len(nonstopwords) / len(text)

nonstopword_fraction(nltk.corpus.reuters.words())

# Solve word puzzle
#
# 1. Create the distribution of the available characters
#
# 2. The define the obligatory letter
#
# 3. Take all available words in the nltk corpus and loop through the
#    words which satisfies the criteria.
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
words_found = [word for word in wordlist
               if len(word) >= 6
               and obligatory in word
               and nltk.FreqDist(word) <= puzzle_letters]

print(words_found)
len(words_found)

# Find over unisexual names with the names corpus
#
# The corpus contain male and female names, it appears that there are
# more female names than male names.
names = nltk.corpus.names
names.fileids()
male_name = names.words('male.txt')
female_name = names.words('female.txt')

unisexual_names = [name for name in male_name if name in female_name]
print(unisexual_names)
len(unisexual_names)

# Create distribution of the ending characters of female names
#
# From the distribution plot we can see that words ending with 'a' and
# 'e' are more likely to be female names. This may have been a result
# of the latin root.
last_character = [name[-1] for name in male_name]

last_character = [(fileid, word[-1])
                  for fileid in names.fileids()
                  for word in names.words(fileid)]

last_character_dist = nltk.ConditionalFreqDist(last_character)
last_character_dist.plot()

# 4.2 A pronouncing dictionary

# The cmudict contains the pronounciation of each word, each word can
# have multiple pronounciations.
entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[42371:42379]:
    print(entry)

# Find words that sound like Nicks
syllable = ['N', 'IH0', 'K', 'S']
similar_pron = [word for word, pron in entries if pron[-4:] == syllable]
print(similar_pron)

# Find words such that ends with 'n' but pronounced like 'M'
n_like_m = [w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']
print(n_like_m)

# Words that does not start with 'n' but the pronounciation starts
# with 'N'.
sorted(set(w[:2] for w, pron in entries
           if pron[0] == 'N'
           and w[0] != 'n'))

# Instead of the list of tuples, we can access the cmu dictionary as a
# python dictionary
prondict = nltk.corpus.cmudict.dict()
prondict['fire']

# 4.3 Comparative wordlist
#
# The swadesh comparative word list is a list of 200 common words in
# multiple languages.
from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')

# Use the word list to construct a translator
fr2en = swadesh.entries(['fr', 'en'])
translate = dict(fr2en)
translate['chien']
translate['jeter']

# We can also add in extra language by updating our dictionary, german
# and spanish are added.
de2en = swadesh.entries(['de', 'en'])
es2en = swadesh.entries(['es', 'en'])
translate.update(dict(de2en))
translate.update(dict(es2en))

# Spanish for dog
translate['perro']
# German for dog
translate['Hund']

# 4.4 Shoebox and toolbox lexicons
#
# Toolbox previously shoebox is a multipurpose tool for field
# liguistists. We only demonstrate the dictionaries contain in the
# toolbox.
from nltk.corpus import toolbox
toolbox.entries('rotokas.dic')

# Section 5: WordNet
# --------------------------------------------------

# 5.1 Senses and synonym
#
# The word motor car has only one meaning and is identified as
# 'car.n.01' known as 'synset' which is a collection of words (or
# 'lemma') which maps to the noun car.
from nltk.corpus import wordnet as wn
motorcar_synset = wordnet.synsets('motorcar')

# To get actual synonym words we can use use the lemma_names method.
#
# NOTE (Michael): The singular form is used below!
wn.synset('car.n.01').lemma_names()

# We can also obtain the definition.
wn.synset('car.n.01').definition()

# And an example sentence
wn.synset('car.n.01').examples()

# We can also obtain the synsets first then work with the specific
# lemma.
motorcar_synsets = wn.synsets('motorcar')
motorcar_lemmas = wn.synset(motorcar_synset[0].name()).lemmas()
wn.lemma('car.n.01.automobile')
wn.lemma('car.n.01.automobile').synset()
wn.lemma('car.n.01.automobile').name()

# Unlike the word automobile, the word car has multiple synsets.
car_synsets = wn.synsets('car')
synsets_names = [(synset, synset.lemma_names())
                 for synset in car_synsets]

# 5.2 WordNet hierachchy.
#
# The WordNet is arranged in a hierachy fashion. See the source page
# for more information.

# This hierachyallows us to explore the relationship between words
# such as hyponyms. e.g. Spoon is a hyponym of cutleries.

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[0]
print(sorted(lemma.name()
             for synset in types_of_motorcar
             for lemma in synset.lemmas()))

# There are two meaning to 'cutleries', we take the second meaning
# which corresponds to the kitchen cutleries.
#
# As we can see, fork, spoon, spork and table knife are hyponyms of
# kitchen cutleries.
cutlery_synsets = wn.synsets('cutleries')
cutlery_synsets[0].definition()  # tool for cutting
cutlery_synsets[1].definition()  # tableware for cutting/eating food
cutlery_synset = wn.synset(cutlery_synsets[1].name())
types_of_cutleries = cutlery_synset.hyponyms()

# If we can nagivate below the hierachy with hyponyms, we can navigate
# above with hypernyms.
#
# There are multiple paths, we can see it can be a 'physical entity'
# while at the same time a 'wheeled vehicle'.
motorcar.hypernyms()
paths = motorcar.hypernym_paths()
len(paths)
[synset.name() for synset in paths[0]]

# We can get the most general hypernyms or the root
motorcar.root_hypernyms()

# 5.3 More Lexical Relations

# Hypernyms and hyponyms are called lexical relations because they
# relate one synset to another. We can also nagivate to subcomponents
# (meronym) or the set that contains it (holonym). For example,
# 'trunk' is a meronym of 'tree', while 'forest' is a holonym of
# 'tree'. The hierachical relationship is multi-dimensional.
#
# Within meronym, we have part or substance. Trunk is a part meronym,
# while heartwood is a substance meronym.
wn.synset('tree.n.01').part_meronyms()
wn.synset('tree.n.01').substance_meronyms()

# And forest is the holonym
wn.synset('tree.n.01').member_holonyms()

# There is also the relationship known as 'entailment'. For example,
# when you walk, it entails 'step'; when you eat, it entails 'chew'.
wn.synset('walk.v.01').entailments()
wn.synset('eat.v.01').entailments()

# Other than synonym, we can obtain antonyms as well.
wn.lemma('supply.n.02.supply').antonyms()
wn.lemma('rush.v.01.rush').antonyms()


# 5.4 Semantic Similarity

# We can find the lowest common hypernyms between two words. That is,
# when traversing above in the hypernyms direction, what is the lowest
# meeting point.
#
# For example, right whale is a baleen whale within whales while orca
# is a toothed whale in whale. Thus, their lowest common hypernym is
# 'whales'. On the other hand, minke whale is also a baleen whale in
# the whale family, so the lowest common hypernym is 'baleen whale'.
#
# Of course we are not restricted to the biological family, we can
# have the lowest common hypernym between right whale and novel.
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
common_dolphin = wn.synset('common_dolphin.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
right.lowest_common_hypernyms(minke)
right.lowest_common_hypernyms(orca)
right.lowest_common_hypernyms(common_dolphin)
right.lowest_common_hypernyms(tortoise)
right.lowest_common_hypernyms(novel)

# We can also see the specificity with the depth.
wn.synset('baleen_whale.n.01').min_depth()
wn.synset('whale.n.02').min_depth()
wn.synset('vertebrate.n.01').min_depth()
wn.synset('entity.n.01').min_depth()

# Instead of the common hypernyms, we can calculate a score that is
# between 0 and 1 to show the similarities.
#
# Because orca and common dolphine belongs to the dolphine family, and
# thus have identical similarity score.
right.path_similarity(minke)
right.path_similarity(orca)
right.path_similarity(common_dolphin)
right.path_similarity(tortoise)
right.path_similarity(novel)
