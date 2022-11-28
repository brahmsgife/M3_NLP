# Dependencies
import json
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from accelerate import Accelerator
import wandb
import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import notebook_login
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification



# 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
# notebook_login()
# 502bf1bc9209c5bf7b9450f804ae3a6867881f02
# run = wandb.init()

N_EXAMPLES_TO_TRAIN = 543 # 10% of original dataset
N_EXAMPLES_TO_EVAL = 92 # 10% of original dataset


class NER:
    """
    Class of task 2
    """
    def __init__(self, dataset='ncbi_disease', base_model='bert-base-cased',
                metric='seqeval', epochs=3):
        self.dataset = dataset
        self.base_model = base_model
        self.metric = metric
        self.epochs = epochs

    def get_training_results(self):
        """
        Train and evaluate a pre-trained model for token classification 
        given certain arguments and epochs.
        """
        def align_labels_with_tokens(labels, word_ids):
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

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens -100) and convert to labels
            true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
            true_predictions = [[label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                                for prediction, label in zip(predictions, labels)]

            all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
            
            return {'precision': all_metrics['overall_precision'],
                    'recall': all_metrics['overall_recall'],
                    'f1': all_metrics['overall_f1'],
                    'accuracy': all_metrics['overall_accuracy'],}
            

        raw_datasets = load_dataset(self.dataset)
        label_names = raw_datasets['train'].features['ner_tags'].feature.names

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        inputs = tokenizer(raw_datasets['train'][0]['tokens'], is_split_into_words=True)


        tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True,
                                            remove_columns=raw_datasets['train'].column_names,)
        
        small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(N_EXAMPLES_TO_TRAIN))
        small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(N_EXAMPLES_TO_EVAL))

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        metric = evaluate.load(self.metric)

        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}

        model = AutoModelForTokenClassification.from_pretrained(self.base_model,
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            num_labels=3,)
        
        train_dataloader = DataLoader(tokenized_datasets['train'],
                                shuffle=True,
                                collate_fn=data_collator,
                                batch_size=8,)

        eval_dataloader = DataLoader(tokenized_datasets['validation'], 
                                collate_fn=data_collator, 
                                batch_size=8,)
        
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = self.epochs * num_update_steps_per_epoch
        optimizer = AdamW(model.parameters(), lr=2e-5)
        lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

        args = TrainingArguments('bert-finetuned-ner',
                            data_seed=42,
                            evaluation_strategy='epoch',
                            save_strategy='epoch',
                            num_train_epochs=self.epochs,
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
                        #train_dataset=tokenized_datasets['train'],
                        eval_dataset=small_eval_dataset,
                        #eval_dataset=tokenized_datasets['validation'],
                        )
        # Training
        train_result = trainer.train()

        # Compute train results
        metrics = train_result.metrics
        max_train_samples = len(small_train_dataset)
        metrics['train_samples'] = min(max_train_samples, len(small_train_dataset))

        # Save train results
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)

        # Compute evaluation results
        metrics = trainer.evaluate()
        max_val_samples = len(small_eval_dataset)
        metrics['eval_samples'] = min(max_val_samples, len(small_eval_dataset))

        # Save evaluation results
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

        with open(r'C:\Users\HP456787\Documents\ITESM\7MO SEM\InteligenciaArtificialAvanzadaII\M3_NLP\bert-finetuned-ner\all_results.json') as file:
            results_json = json.load(file)

        return results_json