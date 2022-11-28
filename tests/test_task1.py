import sys
sys.path.insert(0, '../src')
import warnings
warnings.filterwarnings("ignore")
import unittest
import pytest
from task1 import SentimentAnalysis

results = ['POSITIVE','NEGATIVE','NEGATIVE', 'POSITIVE', 'POSITIVE', 
        'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 
        'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 
        'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE']

def test_SentimentAnalysis_class_get_predictions():
    model = SentimentAnalysis()
    assert model.get_predictions() == results