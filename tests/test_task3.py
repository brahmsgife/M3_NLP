import sys
sys.path.insert(0, '../src')
import warnings
warnings.filterwarnings("ignore")
import unittest
import pytest
from task3 import Translator

scores = {"Google Translate Score": 0.29274903274872854,
          "IBM Translator Score": 0.3080364582441837}

def test_Translator_class_get_scores():
    translator = Translator()
    assert translator.get_scores() == scores