import json
import sys
sys.path.insert(0, '../src')
import warnings
warnings.filterwarnings("ignore")
import unittest
import pytest
from task2 import NER



results_json = {
    "epoch": 3.0,
    "eval_accuracy": 0.9633426529978254,
    "eval_f1": 0.7094017094017093,
    "eval_loss": 0.132642924785614,
    "eval_precision": 0.6587301587301587,
    "eval_recall": 0.7685185185185185,
    "eval_runtime": 15.0375,
    "eval_samples": 92,
    "eval_samples_per_second": 6.118,
    "eval_steps_per_second": 0.798,
    "train_loss": 0.15190548055312214,
    "train_runtime": 1156.6483,
    "train_samples": 543,
    "train_samples_per_second": 1.408,
    "train_steps_per_second": 0.176
}


def test_NER_class_get_training_results():
    ner = NER()
    assert ner.get_training_results() == results_json