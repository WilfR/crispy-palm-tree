import unittest
import ReviewScorer

class ReviewScorerTests( unittest.TestCase ):

    def testPositiveReview( self ) :
        scorer = ReviewScorer.ReviewScorer()
        posReviewFilename = 'C:\\Users\\wilf\\AppData\\Roaming\\nltk_data\\corpora\\movie_reviews\\pos\\cv000_29590.txt'
        with open(posReviewFilename,'r') as f:
            review = f.read()
        prediction = scorer.score_review(review)
        self.assertEqual('pos',prediction)

    def testNegativeReview( self ) :
        scorer = ReviewScorer.ReviewScorer()
        posReviewFilename = 'C:\\Users\\wilf\\AppData\\Roaming\\nltk_data\\corpora\\movie_reviews\\neg\\cv000_29416.txt'
        with open(posReviewFilename,'r') as f:
            review = f.read()
        prediction = scorer.score_review(review)
        self.assertEqual('neg',prediction)


if __name__ == '__main__':
    unittest.main()
