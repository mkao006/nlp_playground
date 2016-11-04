import argparse
from googleapiclient import discovery
import httplib2
import json
from oauth2client.client import GoogleCredentials


def get_sentiment(movie_review_file):
    '''Run a sentiment analysis request on text within a passed filename.'''

    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('language', 'v1beta1', credentials=credentials)

    with open(movie_review_file, 'r') as review_file:
        service_request = service.documents().analyzeSentiment(
            body={
                'document': {
                    'type': 'PLAIN_TEXT',
                    'content': review_file.read(),
                }
            }
        )
        response = service_request.execute()

    polarity = response['documentSentiment']['polarity']
    magnitude = response['documentSentiment']['magnitude']

    print('Sentiment: polarity of {} with magnitude of {}'.format(
        polarity, magnitude))
    return 0

get_sentiment('test.txt')

articles = []
with open('/home/mk/Github/EST_projects/TheReadingMachine/data/amis_articles_27_07_2016.jsonl') as f:
    for i, line in enumerate(f):
        if i <= 4:
            articles.append(json.loads(line))
    f.close()

for i, line in enumerate(articles):
    with open('test{0}.txt'.format(i), 'w') as f:
        f.write(line['article'])
        f.close()

for i in range(5):
    get_sentiment('test{0}.txt'.format(i))
