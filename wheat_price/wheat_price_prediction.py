# NOTE(Michael): As the Reuters corpus does not contain date, and thus
#                we can not correlate it to wheat price.
#
#                To move forward, we would have to build a topic
#                classification on the Reuters corpus then use the
#                model tag the unlabelled data from 'The Reading
#                Machine'.
#
#                Then we can proceed with the price prediction.

import httplib2
from nltk.corpus import reuters
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

# Extract articles that contains topics of interest.
topics = ['wheat', 'rice', 'maize', 'grain']
related_topic_files = reuters.fileids(topics)

articles = [{'categories': reuters.categories(file),
             'article': reuters.raw(file)}
            for file in related_topic_files]

relevant_tags = set()
[relevant_tags.add(category)
 for article in articles
 for category in article['categories']]


def retrieve_sentiment(string):
    ''' Use the google nlp api to calculate the sentiments.
    '''

    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('language', 'v1beta1', credentials=credentials)
    service_request = service.documents().analyzeSentiment(
        body={
            'document': {
                'type': 'PLAIN_TEXT',
                'content': string,
            }
        }
    )
    response = service_request.execute()

    polarity = response['documentSentiment']['polarity']
    magnitude = response['documentSentiment']['magnitude']
    return {'polarity': polarity, 'magnitude': magnitude}

# Create keyword one-hot vector
for article in articles:
    for tag in relevant_tags:
        article[tag + '_keyword'] = int(tag in article['categories'])

# Get sentiment index
for article in articles:
    article.update(retrieve_sentiment(article['article']))
