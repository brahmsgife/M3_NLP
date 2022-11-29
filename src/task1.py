# Dependencies
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalysis:
    """
    Class of taks1
    """
    def __init__(self, model_name='nlptown/bert-base-multilingual-uncased-sentiment',
                path='../tiny_movie_reviews_dataset.txt'):
        self.path = path
        self.model_name = model_name

    def get_predictions(self):
        """
        Given the model created, it analyzes each review and 
        generates a prediction to see if it's a positive or negative review.
        """
        with open(self.path, 'r') as f:
            reviews = f.readlines()

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        predictions = []
        for i in reviews:
            input = tokenizer(i, return_tensors="pt", max_length=512, padding=True, truncation=True)
            with torch.no_grad():
                output = model(**input).logits
            prediction = output.argmax().item()
            predictions.append(prediction)
        
        results = []
        for i in predictions:
            if i == 1:
                results.append('POSITIVE')
                print('POSITIVE')
            else:
                results.append('NEGATIVE')
                print('NEGATIVE')
        
        return results