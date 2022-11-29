# M3_NLP
Repository for the Natural Language Processing class of the Advanced Artificial Intelligence concentration.


## Description
In this repository you will find the 3 tasks of the Feedback Moment of Module 3 of the Concentration of Advanced Artificial Intelligence.


### Installing
Clone the repository, create the virtual environment, and make sure to install the requirements.
```
pip install -r requirements.txt
```

### Executing program
Before running the main "run.py" script, if you want to change the keys for task 3, please go to src/task3.py and change it in lines 13, 14 and 22.
Also, make sure to log in with your HuggingFace account:
```
huggingface-cli login
```
Run the main script run.py, it contains everything necessary for the correct execution of the 3 tasks.
```
py run.py
```

Tests:
To run tests, from M3_NLP/tests, call any of the scripts. Here is an example:
```
pytest -v test_task1.py
```

### Recommendations
Please make sure you have your own HuggingFace, WandB, Google-Cloud Translate and IBM Watson Language Translator tokens.

### Notes: 
In case of task 2, the graphs of train set error and test rate during training with the complete dataset, can be seen in this workspace: https://wandb.ai/brahmsgife/M3_NLP/runs/4kgeet58?workspace=user-brahmsgife

## Author
Abraham Gil Félix
