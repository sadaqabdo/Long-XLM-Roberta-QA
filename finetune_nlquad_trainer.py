import json
import logging
import os
import random
import sys
import time

import datasets
import numpy as np
import torch
import transformers
from transformers import (Trainer, TrainingArguments, XLMRobertaTokenizerFast,
                          default_data_collator)

from config import config
from dataset import (cast_dataset_features, interleave, prepare_features,
                     read_nlquad, read_squad2)
from model import XLMRobertaLongForQuestionAnswering
from processing import calculate_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Setup logging
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

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
    logger.info("Reading NLQuAD data")
    nlquad_train_data = read_nlquad(config["train_path"])
    nlquad_valid_data = read_nlquad(config["valid_path"])
    nlquad_eval_data = read_nlquad(config["eval_path"])

    nlquad_train_dataset = prepare_features(
        nlquad_train_data, config["nlquad_num_training_examples"], mode="train"
    )
    nlquad_valid_dataset = prepare_features(
        nlquad_valid_data, config["nlquad_num_validating_examples"], mode="train"
    )
    nlquad_valid_dataset_for_eval = prepare_features(
        nlquad_valid_data, config["nlquad_num_validating_examples"], mode="eval"
    )
    nlquad_eval_dataset = prepare_features(
        nlquad_eval_data, config["nlquad_num_evaluation_examples"], mode="eval"
    )

    # Read SQuAD v2 data
    if config["squad_v2"]:
        logger.info("Reading SQuAD v2 data")
        squad2_train_data = read_squad2("train")
        squad2_valid_data = read_squad2("validation")

        squad2_train_data = cast_dataset_features(squad2_train_data)
        squad2_valid_data = cast_dataset_features(squad2_valid_data)

        train_data = interleave(squad2_train_data, nlquad_train_data, config["seed"])
        valid_data = interleave(squad2_valid_data, nlquad_valid_data, config["seed"])
        eval_data = interleave(squad2_valid_data, nlquad_eval_data, config["seed"])

        squad2_train_dataset = prepare_features(
            squad2_train_data, config["squadv2_num_training_examples"], mode="train"
        )
        squad2_valid_dataset = prepare_features(
            squad2_valid_data, config["squadv2_num_validating_examples"], mode="train"
        )
        squad2_valid_dataset_for_eval = prepare_features(
            squad2_valid_data, config["squadv2_num_validating_examples"], mode="eval"
        )

        logger.info("Interleaving NLQuAD and SQuAD v2 data")
        train_dataset = interleave(
            squad2_train_dataset, nlquad_train_dataset, config["seed"]
        )
        valid_dataset = interleave(
            squad2_valid_dataset, nlquad_valid_dataset, config["seed"]
        )
        valid_dataset_for_eval = interleave(
            squad2_valid_dataset_for_eval, nlquad_valid_dataset_for_eval, config["seed"]
        )
        eval_dataset = interleave(
            squad2_valid_dataset_for_eval, nlquad_eval_dataset, config["seed"]
        )
    else:
        train_data = nlquad_train_data
        valid_data = nlquad_valid_data
        eval_data = nlquad_eval_data

        train_dataset = nlquad_train_dataset
        valid_dataset = nlquad_valid_dataset
        valid_dataset_for_eval = nlquad_valid_dataset_for_eval
        eval_dataset = nlquad_eval_dataset

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
        logging_dir=config["output_dir"],
        save_strategy="no",
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_nan_inf_filter=False,
    )

    trainer = Trainer(
        xlm_roberta,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
        tokenizer=config["tokenizer"],
    )

    logger.info("Training model")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    logger.info("Saving model")
    trainer.save_model(config["output_dir"])
    logger.info("Model Saved")

    logger.info("Evaluating model")
    evaluation = trainer.evaluate()
    logger.info("Evaluation %s", evaluation)

    logger.info("Evaluating model on Valid Dataset")
    validation_predictions = trainer.predict(valid_dataset_for_eval).predictions

    logger.info("Calculating metrics : \n")
    valid_time = time.time()
    validation_set_res = calculate_metrics(
        valid_data, valid_dataset_for_eval, validation_predictions
    )
    logger.info("Results on Validation Set: %s", validation_set_res)

    logger.info("Evaluating model on Eval Dataset")
    evaluation_predictions = trainer.predict(eval_dataset).predictions

    logger.info("Calculating metrics : \n")

    evluation_set_res = calculate_metrics(
        eval_data, eval_dataset, evaluation_predictions
    )
    logger.info("Results on Evaluation Set: %s", evluation_set_res)

    logger.info("#" * 50)
    logger.info("Saving results")
    with open("results_trainer.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation_results": validation_set_res,
                "evaluation_results": evluation_set_res,
            },
            f,
        )

    logger.info("The End.")
    logger.info("Directed By The QA Company.")
