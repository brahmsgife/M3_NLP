# Dependencies
import os
import evaluate
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datasets import load_dataset
from accelerate import Accelerator

import wandb

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import Trainer
from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification

from huggingface_hub import notebook_login

from nltk.translate.bleu_score import sentence_bleu
from google.cloud import translate_v2 as translate
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# ------------------------------------------------------- Task 1 ------------------------------------------------------- 
def read_dataset(path):
    """
    Read the dataset called "tiny_movie_reviews_dataset.txt".
    """
    with open(path, 'r') as f:
        reviews = f.readlines()
    return reviews

def create_model(model_name):
    """
    Create a instance of the model "nlptown/bert-base-multilingual-uncased-sentiment" for sentiment analysis.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_predictions(model, tokenizer, dataset):
    """
    Given the model created, it analyzes each review and 
    generates a prediction to see if it's a positive or negative review.
    """
    predictions = []
    for i in dataset:
        input = tokenizer(i, return_tensors="pt", max_length=512, padding=True, truncation=True)
        with torch.no_grad():
            output = model(**input).logits
        prediction = output.argmax().item()
        predictions.append(prediction)
    return predictions

def results(predictions):
    """
    Allows to show the predictions for each review of dataset.
    """
    for i in predictions:
        if i == 1:
            print('POSITIVE')
        else:
            print('NEGATIVE') 

def SentimentAnalysis():
    """
    Main function that allows you to perform sentiment analysis from a set of data.
    """
    reviews = read_dataset(path=r'./tiny_movie_reviews_dataset.txt')
    model, tokenizer = create_model(model_name='nlptown/bert-base-multilingual-uncased-sentiment')
    predictions = get_predictions(model, tokenizer, reviews)
    results(predictions)
    return results


# ------------------------------------------------------- Task 2 ------------------------------------------------------- 
def get_num_labels(raw_datasets):
    """
    Get the number of classes.
    """
    ner_feature = raw_datasets['train'].features['ner_tags']
    label_names = ner_feature.feature.names
    return len(label_names)

def get_label_names(raw_datasets):
    """
    Get the classes.
    """
    ner_feature = raw_datasets['train'].features['ner_tags']
    return ner_feature.feature.names

def create_inputs(raw_datasets, tokenizer):
    """
    Create the inputs for the pretrained model.
    """
    return tokenizer(raw_datasets['train'][0]['tokens'], is_split_into_words=True)

def align_labels_with_tokens(labels, word_ids):
    """
    Modify the labels; aligning and considering the special values (-100).
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token, value=-100
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples):
    """
    Get tokenized inputs.
    """
    tokenized_inputs = tokenizer(examples['tokens'], 
                                 truncation=True, 
                                 is_split_into_words=True,)
    all_labels = examples['ner_tags']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

def get_tokenized_datasets(raw_datasets):
    """
    """
    return raw_datasets.map(tokenize_and_align_labels, batched=True, remove_columns=raw_datasets['train'].column_names,)

def get_small_train_dataset(tokenized_datasets, N_EXAMPLES_TO_TRAIN):
    """
    """
    return tokenized_datasets['train'].shuffle(seed=42).select(range(N_EXAMPLES_TO_TRAIN))

def get_small_eval_dataset(tokenized_datasets, N_EXAMPLES_TO_EVAL):
    """
    """
    return tokenized_datasets['test'].shuffle(seed=42).select(range(N_EXAMPLES_TO_EVAL))

def get_train_dataloader(small_train_dataset, data_collator):
    """
    """
    return DataLoader(small_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=8,)

def get_eval_dataloader(small_eval_dataset, data_collator):
    """
    """
    return DataLoader(small_eval_dataset, collate_fn=data_collator, batch_size=8,)

def compute_metrics(eval_preds):
    """
    Metrics (precision, recall, f1 and accuracy) for evaluation on training.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens -100) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    metric = evaluate.load('seqeval')
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {'precision': all_metrics['overall_precision'],
            'recall': all_metrics['overall_recall'],
            'f1': all_metrics['overall_f1'],
            'accuracy': all_metrics['overall_accuracy'],}

def create_base_model(name_base_model, num_labels, label_names):
    """
    Function that allows loading the pretrained model.
    """
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(name_base_model,
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            num_labels=num_labels,)
    return model

def train_base_model(num_train_epochs, optimizer, train_dataloader, model):
    """
    Function that allows to tune the pre-trained model.
    """
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)
    
    args = TrainingArguments('bert-finetuned-ner',
                            data_seed=42,
                            evaluation_strategy='epoch',
                            save_strategy='epoch',
                            num_train_epochs=num_train_epochs,
                            adam_beta1=0.9,
                            learning_rate=2e-5,
                            weight_decay=0.01,
                            lr_scheduler_type='linear',
                            report_to='wandb',
                            push_to_hub=True,)

    trainer = Trainer(model=model,
                    args=args,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    optimizers=(optimizer,lr_scheduler),
                    train_dataset=small_train_dataset,
                    eval_dataset=small_eval_dataset,)
    
    return trainer.train()

def NER():
    """
    Main function Task2.
    """
    num_labels = get_num_labels(raw_datasets)
    train_dataloader = get_train_dataloader(small_train_dataset, data_collator)
    eval_dataloader = get_eval_dataloader(small_eval_dataset, data_collator)
    
    model = create_base_model(name_base_model, num_labels, label_names)
    
    num_train_epochs = 3 
    optimizer = AdamW(model.parameters(), lr=2e-5) 
    train_base_model(num_train_epochs, optimizer, train_dataloader, model)
    
    return train_base_model

name_dataset = 'ncbi_disease'
raw_datasets = load_dataset(name_dataset)
label_names = get_label_names(raw_datasets)

name_base_model = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(name_base_model) 

N_EXAMPLES_TO_TRAIN = 543 # 10% of original dataset
N_EXAMPLES_TO_EVAL = 92 # 10% of original dataset

tokenized_datasets = get_tokenized_datasets(raw_datasets)
small_train_dataset = get_small_train_dataset(tokenized_datasets, N_EXAMPLES_TO_TRAIN)
small_eval_dataset = get_small_eval_dataset(tokenized_datasets, N_EXAMPLES_TO_EVAL)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# ------------------------------------------------------- Task 3 ------------------------------------------------------- 
def load_spanish_texts(path):
    spanish_texts = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:100]:
            spanish_texts.append(line)
    spanish_texts = [x.replace("\n", "") for x in spanish_texts]
    return spanish_texts


def load_english_texts(path):
    english_texts = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:100]:
            english_texts.append(line)
    english_texts = [x.replace("\n", "") for x in english_texts]
    return english_texts

# KEYS
api_key = 's7rwQuCoVdiBBo2P_hSxfkilAoHfATF9Qy3JMjJ0H0IK'
api_url = 'https://api.au-syd.language-translator.watson.cloud.ibm.com/instances/70d82c08-78ec-464d-9ed6-17a73de5a2ea'
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/HP456787/Documents/ITESM/7MO SEM/InteligenciaArtificialAvanzadaII/M3_NLP/m3-nlp-e9d4ac58420b.json'

# Load Corpus texts
spanish_texts = load_spanish_texts('C:/Users/HP456787/Documents/ITESM/7MO SEM/InteligenciaArtificialAvanzadaII/M3_NLP/es_corpus.txt')
english_texts = load_english_texts('C:/Users/HP456787/Documents/ITESM/7MO SEM/InteligenciaArtificialAvanzadaII/M3_NLP/en_corpus.txt')

# Translators
google_translator = translate.Client()
authenticator = IAMAuthenticator(api_key)
ibm_translator = LanguageTranslatorV3(version='2018-05-01', authenticator=authenticator)
ibm_translator.set_service_url(api_url)


# ------------------------------------------------------- Running Tasks ------------------------------------------------------- 
#----------------------------------------------------------- Task 1 -----------------------------------------------------------
print('-'*69 + 'Task1' + '-'*69)
SentimentAnalysis()

#----------------------------------------------------------- Task 2 -----------------------------------------------------------

print('-'*69 + 'Task2' + '-'*69)
#notebook_login()
run = wandb.init()
NER()

#----------------------------------------------------------- Task 3 -----------------------------------------------------------
print('-'*69 + 'Task3' + '-'*69)
# Implementation
ibm_bleu = []
google_bleu = []

for i in range(len(english_texts)):
  google_result = google_translator.translate(english_texts[i], 'es')
  score_google_bleu = sentence_bleu(spanish_texts[i].split(), google_result['translatedText'].split())
  google_bleu.append(score_google_bleu)
  
  ibm_result = ibm_translator.translate(text=english_texts[i], model_id='en-es')
  score_ibm_bleu = sentence_bleu(spanish_texts[i].split(), ibm_result.result['translations'][0]['translation'].split())
  ibm_bleu.append(score_ibm_bleu)

print('Google Score: ', round((sum(google_bleu)/100),4))
print('IBM Score: ', round((sum(ibm_bleu)/100),4))