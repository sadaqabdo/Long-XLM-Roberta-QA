import json
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
from transformers import (Trainer, TrainingArguments, XLMRobertaTokenizerFast,
                          default_data_collator)

from config import config
from dataset import interleave, prepare_features, read_nlquad, read_squad2
from model import XLMRobertaLongForQuestionAnswering
from processing import calculate_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # assert torch.cuda.device_count() > 0, "No GPU found"
    # assert config["device"] == "cuda", "GPU not found"
    # assert torch.backends.cudnn.enabled, "CUDNN is not enabled"

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

    # Read NLQuAD data
    print("Reading NLQuAD data")
    train_data = read_nlquad(config["train_path"]).flatten()
    valid_data = read_nlquad(config["valid_path"]).flatten()
    eval_data = read_nlquad(config["eval_path"]).flatten()

    train_dataset = prepare_features(
        train_data, config["num_training_examples"], mode="train"
    )
    valid_dataset = prepare_features(
        valid_data, config["num_validating_examples"], mode="train"
    )
    valid_dataset_for_eval = prepare_features(
        valid_data, config["num_validating_examples"], mode="eval"
    )
    eval_dataset = prepare_features(
        eval_data, config["num_evaluation_examples"], mode="eval"
    )

    # Read SQuAD v2 data
    if config["squad_v2"]:
        print("Reading SQuAD v2 data")
        squad2_train_data, squad2_valid_data = read_squad2(config)
        squad2_train_data = squad2_train_data.flatten()
        squad2_valid_data = squad2_valid_data.flatten()

        squad2_train_dataset = prepare_features(
            squad2_train_data, config["num_training_examples"], mode="train"
        )
        squad2_valid_dataset = prepare_features(
            squad2_valid_data, config["num_validating_examples"], mode="train"
        )
        squad2_valid_dataset_for_eval = prepare_features(
            squad2_valid_data, config["num_validating_examples"], mode="eval"
        )

        print("Interleaving NLQuAD and SQuAD v2 data")
        train_dataset = interleave(squad2_train_dataset, train_dataset, config["seed"])
        valid_dataset = interleave(squad2_valid_dataset, valid_dataset, config["seed"])
        valid_dataset_for_eval = interleave(
            squad2_valid_dataset_for_eval, valid_dataset_for_eval, config["seed"]
        )
        eval_dataset = interleave(
            squad2_valid_dataset_for_eval, eval_dataset, config["seed"]
        )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        optim="adamw_torch",
        learning_rate=config["learning_rate"],
        adam_epsilon=config["epsilon"],
        fp16=config["fp16"],
        max_grad_norm=config["max_grad_norm"],
        seed=config["seed"],
        metric_for_best_model="f1",
        do_train=True,
        do_eval=True,
        do_predict=True,
    )

    trainer = Trainer(
        xlm_roberta,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
        tokenizer=config["tokenizer"],
    )

    print("Training model")
    trainer.train()

    print("Saving model")
    trainer.save_model(config["output_dir"])
    print("Model Saved")

    print("Evaluating model")
    evaluation = trainer.evaluate()
    print(f"Evaluation : {evaluation}")

    print("Evaluating model on Valid Dataset")
    validation_predictions = trainer.predict(valid_dataset_for_eval).predictions

    print("Calculating metrics : \n")
    valid_time = time.time()
    validation_set_res = calculate_metrics(
        valid_data, valid_dataset_for_eval, validation_predictions
    )
    print(f"Results on Validation: {validation_set_res}")

    print("Evaluating model on Eval Dataset")
    evaluation_predictions = trainer.predict(eval_dataset).predictions

    print("Calculating metrics : \n")
    eval_time = time.time()
    evluation_set_res = calculate_metrics(
        eval_data, eval_dataset, evaluation_predictions
    )
    print(f"Results on Evaluation : {evluation_set_res}")

    print(f"Evaluation took: {str(timedelta(seconds=time.time() - eval_time))}")
    print("Saving results")
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation_results": validation_set_res,
                "evaluation_results": evluation_set_res,
            },
            f,
        )

    print("The End.")
    print("Directed By The QA Company.")
