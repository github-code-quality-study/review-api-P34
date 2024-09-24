import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
# sia = SentimentIntensityAnalyzer()
# stop_words = set(stopwords.words('english'))

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

VALID_LOCATIONS = [
     'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California', 'Colorado Springs, Colorado',
    'Denver, Colorado', 'El Cajon, California', 'El Paso, Texas', 'Escondido, California', 'Fresno, California',
    'La Mesa, California', 'Las Vegas, Nevada', 'Los Angeles, California', 'Oceanside, California', 'Phoenix, Arizona',
    'Sacramento, California', 'Salt Lake City, Utah', 'San Diego, California', 'Tucson, Arizona'
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.reviews = pd.read_csv('data/reviews.csv').to_dict('records')
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def analyze_sentiment(self, review_body: str) -> dict:
        sentiment_scores = self.sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":

           
            # Create the response body from the reviews and convert to a JSON byte string
            
            # Write your code here

            params = parse_qs(environ.get('QUERY_STRING', ''))
            location = params.get('location', [None])[0]
            start_date = params.get('start_date', [None])[0]
            end_date = params.get('end_date', [None])[0]

            filtered_reviews = self.reviews

            if location and location in VALID_LOCATIONS:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) >= start_date]

            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) <= end_date]

            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
            
            filtered_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
                params = parse_qs(request_body)

                location = params.get('Location', [None])[0]
                review_body = params.get('ReviewBody', [None])[0]

                if not location or not review_body or location not in VALID_LOCATIONS: 
                    raise ValueError('Invalid Input')

                review = {
                    'ReviewId': str(uuid.uuid4()),
                    'Location': location,
                    'Timestamp': datetime.now().strftime(TIMESTAMP_FORMAT),
                    'ReviewBody': review_body
                }
                review['sentiment'] = self.analyze_sentiment(review_body)
                self.reviews.append(review)

                response_body = json.dumps(review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                response_body = json.dumps({"error": str(e)}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()