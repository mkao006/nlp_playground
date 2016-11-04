# Supervised Learning For Document Classification With Scikit-Learn
#
# Source:
# https://www.quantstart.com/articles/Supervised-Learning-for-Document-Classification-with-Scikit-Learn

# Load the Reuters corpus
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Select topic
topic = ['wheat', 'corn', 'maize', 'rice', 'soybean']
commodity_files = reuters.fileids(topic)

# Split file into training and testing
training_commodity_files = [file
                            for file in commodity_files
                            if 'training' in file]
test_commodity_files = [file
                        for file in commodity_files
                        if 'test' in file]


# Get training data
train_commodity_set = [(categories, reuters.raw(file))
                       for file in training_commodity_files
                       for categories in reuters.categories(file)
                       if categories in topic]

test_commodity_set = [(categories, reuters.raw(file))
                      for file in test_commodity_files
                      for categories in reuters.categories(file)
                      if categories in topic]


def create_tfidf_training_data(docs):
    """Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list.

    The function returns both the class label vector (y) and the
    corpus token/feature matrix (X).

    """
    # Create the training data class labels
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, y

# Split the data into feature and response.
train_x, train_y = create_tfidf_training_data(train_commodity_set)
test_x, test_y = create_tfidf_training_data(test_commodity_set)


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X, y)
    return svm

# Train and obtain the fit.
train_model = train_svm(train_x, train_y)
train_pred = train_model.predict(train_x)

print(train_model.score(train_x, train_y))
print(confusion_matrix(train_pred, train_y))

# NOTE (Michael): To use the training and test set approach, we need
#                 to create the td-idf matrix together so the
#                 dimension is the same. After the matrix has been
#                 calculated, we can then split into test and training
#                 set.
