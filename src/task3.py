#Dependencies 
import os
import json
import warnings
warnings.filterwarnings("ignore")
from google.cloud import translate_v2 as translate
from nltk.translate.bleu_score import sentence_bleu
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator



api_key = 's7rwQuCoVdiBBo2P_hSxfkilAoHfATF9Qy3JMjJ0H0IK'
api_url = 'https://api.au-syd.language-translator.watson.cloud.ibm.com/instances/70d82c08-78ec-464d-9ed6-17a73de5a2ea'


class Translator():
    """
    Class of task 3
    """
    # Class variables
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'C:\Users\HP456787\Documents\ITESM\7MO SEM\InteligenciaArtificialAvanzadaII\M3_NLP\m3-nlp-e9d4ac58420b.json'


    def __init__(self,path1=r'C:\Users\HP456787\Documents\ITESM\7MO SEM\InteligenciaArtificialAvanzadaII\M3_NLP\es_corpus.txt', 
                path2=r'C:\Users\HP456787\Documents\ITESM\7MO SEM\InteligenciaArtificialAvanzadaII\M3_NLP\en_corpus.txt'):
        self.path1 = path1
        self.path2 = path2

    def setKey(self, api_key):
        self.api_key = api_key

    def setUrl(self, api_url):
        self.api_url = api_url
    
    def get_scores(self):
        """
        Translates one hundred English-Spanish sentences with the help of the Google and IBM API, 
        and also evaluates them with sentence_bleu.
        """
        spanish_texts = []
        english_texts = []

        with open(self.path1, 'r', encoding='utf-8') as f:
            corpus1 = f.readlines()
            for line in corpus1[:100]:
                spanish_texts.append(line)
        
        with open(self.path2, 'r', encoding='utf-8') as f:
            corpus2 = f.readlines()
            for line in corpus2[:100]:
                english_texts.append(line)
        
        spanish_texts = [x.replace("\n", "") for x in spanish_texts]
        english_texts = [x.replace("\n", "") for x in english_texts]

        google_translate = translate.Client()
        authenticator = IAMAuthenticator(api_key)
        ibm_translator = LanguageTranslatorV3(version='2018-05-01', authenticator=authenticator)
        ibm_translator.set_service_url(api_url)

        ibm_bleu = []
        google_bleu = []
        for i in range(len(english_texts)):
            google_result = google_translate.translate(english_texts[i], 'es')
            score_google_bleu = sentence_bleu([spanish_texts[i].split()], 
                                               google_result['translatedText'].split())
            google_bleu.append(score_google_bleu)
            
            ibm_result = ibm_translator.translate(text=english_texts[i], model_id='en-es')
            score_ibm_bleu = sentence_bleu([spanish_texts[i].split()], 
                                            ibm_result.result['translations'][0]['translation'].split())
            ibm_bleu.append(score_ibm_bleu)
        print('Google Translate Score: ', (sum(google_bleu)/100))
        print('IBM Translator Score: ', (sum(ibm_bleu)/100))

        return {"Google Translate Score": (sum(google_bleu)/100),
                "IBM Translator Score": (sum(ibm_bleu)/100)}