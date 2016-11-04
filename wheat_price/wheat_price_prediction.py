from nltk.corpus import reuters
import httplib2
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
