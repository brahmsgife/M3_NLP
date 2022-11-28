import os
import dotenv
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from google.cloud import translate_v2 as translate
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# KEYS
api_key = 's7rwQuCoVdiBBo2P_hSxfkilAoHfATF9Qy3JMjJ0H0IK'
api_url = 'https://api.au-syd.language-translator.watson.cloud.ibm.com/instances/70d82c08-78ec-464d-9ed6-17a73de5a2ea'
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/HP456787/Documents/ITESM/7MO SEM/InteligenciaArtificialAvanzadaII/M3_NLP/m3-nlp-e9d4ac58420b.json'

# Load Corpus texts
spanish_texts = []
english_texts = []

with open('C:/Users/HP456787/Documents/ITESM/7MO SEM/InteligenciaArtificialAvanzadaII/M3_NLP/es_corpus.txt', 'r', encoding='utf-8') as f:
  lines = f.readlines()
  for line in lines[:100]:
    spanish_texts.append(line)
    
with open('C:/Users/HP456787/Documents/ITESM/7MO SEM/InteligenciaArtificialAvanzadaII/M3_NLP/en_corpus.txt', 'r', encoding='utf-8') as f:
  lines = f.readlines()
  for line in lines[:100]:
    english_texts.append(line)

spanish_texts = [x.replace("\n", "") for x in spanish_texts]
english_texts = [x.replace("\n", "") for x in english_texts]

# Translators
google_translator = translate.Client()
authenticator = IAMAuthenticator(api_key)
ibm_translator = LanguageTranslatorV3(version='2018-05-01', authenticator=authenticator)
ibm_translator.set_service_url(api_url)


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