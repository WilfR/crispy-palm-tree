# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

### # Instantiates a client
### client = language.LanguageServiceClient()
###
### # The text to analyze
### text = u'Hello, world! You evil bad bastard! I hate you. I hate everything about you.'
### ### text = u'kiss kiss i love you to the moon and back!'
###
### document = types.Document(
###     content=text,
###     type=enums.Document.Type.PLAIN_TEXT)
###
### # Detects the sentiment of the text
### sentiment = client.analyze_sentiment(document=document).document_sentiment
###
### print('Text: {}'.format(text))
### print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

class ReviewScorer():
    def __init__(self):
        self.client = language.LanguageServiceClient()

    def __del__( self ) :
        pass

    def score_review( self, review ) :
        document = types.Document( content=review, type=enums.Document.Type.PLAIN_TEXT)
        sentiment = self.client.analyze_sentiment(document=document).document_sentiment

        ### print('Text: {}'.format(review))
        ### print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

        if sentiment.score >= 0.0 :
            return 'pos'
        return 'neg'

def Main():
    pass

def score_review( review ) :
    client = language.LanguageServiceClient()

    document = types.Document( content=review, type=enums.Document.Type.PLAIN_TEXT)
    sentiment = self.client.analyze_sentiment(document=document).document_sentiment


    if sentiment.score >= 0.0 :
        return 'pos'
    return 'neg'


if __name__ == '__main__':
    Main()
