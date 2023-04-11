# finetuning script 

In this project, we extend XLM-Roberta-For-QA to a long document XLM-Roberta-For-QA model. the model supports 4096 tokens as input, which is an improvement over the original model which supports 512 tokens as input. 

We also provide a finetuning script for the model on the NLQuAD dataset.

NLQuAD stands for Non-Factoid Long Question Answering Dataset. 

## Requirements

```bash
pip install -r requirements.txt
```

## Data

```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Yviu4C8kJYh8EpfJGzLjAR5-N_I39Y4D' -O NLQuAD_train.json
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17rXbzbOL71baX5ArBN3wFzC8NAZFPf8G' -O NLQuAD_valid.json
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HhbU52MBI7Uper6CSzCJiDh6eHGrQY4f' -O NLQuAD_eval.json
```

In the dataset script, we use the NLQuaAD json files to create a torch dataset. We use the NLQuAD_train.json file to train the model, and the NLQuAD_valid.json file to validate the model. We use the NLQuAD_eval.json file to evaluate the model.



## Configuration

```bash
cat config.py
```
in the config sc

## Finetuning

```bash
python finetune_nlquad.py
```

We finetune the model for 5 epochs on the NLQuAD dataset. We use the AdamW optimizer with a learning rate of 3e-5 and a batch size of 12. We use a linear scheduler with warmup steps of 1000. We ustilise torch.cuda.amp.GradScaler() to scale the gradients, and torch.cuda.amp.autocast() to enable automatic mixed precision. 

