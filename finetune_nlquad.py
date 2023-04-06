import json
import os
import random

import numpy as np
import torch
from transformers import XLMRobertaTokenizerFast

from config import config
from dataset import make_dataloaders, prepare_features, read_nlquad
from engine import Engine, get_optimizer, get_scheduler
from model import XLMRobertaLongForQuestionAnswering
from processing import calculate_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        config["hub_model_id"], use_auth_token=config["access_token"]
    )

    xlm_roberta = XLMRobertaLongForQuestionAnswering.from_pretrained(
        config["hub_model_id"], use_auth_token=config["access_token"]
    ).to(config["device"])
    config["tokenizer"] = tokenizer

    train_loader, valid_loader, eval_loader = make_dataloaders(config)

    optimizer = get_optimizer(xlm_roberta, config)
    scheduler = get_scheduler(train_loader, optimizer, config)

    engine = Engine(xlm_roberta, optimizer, scheduler, config)

    train_loss, valid_loss = 0, 0 
    for epoch in range(config["epochs"]):
        train_loss = engine.train(train_loader, epoch)
        valid_loss = engine.validate(valid_loader, epoch)

    engine.save_checkpoint(train_loss, valid_loss, 5)

    eval_data = read_nlquad(config["eval_path"])
    eval_dataset = prepare_features(
        eval_data, config["num_evaluation_examples"], mode="eval"
    )
    print(f"Evaluating on {len(eval_dataset)} examples")
    evaluation_predictions = engine.evaluate(eval_loader)
    print("Calculating metrics")
    results = calculate_metrics(eval_data, eval_dataset, evaluation_predictions)

    with open("results.json", "w") as f:
        json.dump(results, f)
