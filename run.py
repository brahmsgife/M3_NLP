import wandb
from src.task1 import SentimentAnalysis
from src.task2 import NER
from src.task3 import Translator

print('-'*94)
print('-'*44 + 'Task 1' + '-'*44)
print('-'*94)
model = SentimentAnalysis()
model.get_predictions()


print('-'*94)
print('-'*44 + 'Task 2' + '-'*44)
print('-'*94)
# huggingface-cli login
run = wandb.init()
ner = NER()
ner.get_training_results()


print('-'*94)
print('-'*44 + 'Task 3' + '-'*44)
print('-'*94)
translator = Translator()
translator.get_scores()