import json
import logging
import os
import random
import sys
import time
from datetime import timedelta

import datasets
import numpy as np
import torch
import transformers
from transformers import XLMRobertaTokenizerFast

from config import config
from dataset import (
    interleave,
    make_dataloaders,
    prepare_features,
    read_nlquad,
    read_squad2,
)
from engine import Engine, get_optimizer, get_scheduler
from model import XLMRobertaLongForQuestionAnswering
from processing import calculate_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger.setLevel(logging.INFO)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()


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

    train_loader, valid_loader, valid_loader_for_eval, eval_loader = make_dataloaders(
        config
    )

    optimizer = get_optimizer(xlm_roberta, config)
    scheduler = get_scheduler(train_loader, optimizer, config)

    engine = Engine(xlm_roberta, optimizer, scheduler, config)

    train_loss, valid_loss = 0, 0
    time_start = time.time()
    for epoch in range(config["epochs"]):
        train_loss = engine.train(train_loader, epoch)
        end_epoch_time = time.time()
        logger.info(
            f"Training Epoch {epoch+1} took: {str(timedelta(seconds=end_epoch_time - time_start))}"
        )
        valid_loss = engine.validate(valid_loader, epoch)
        logger.info(
            f"Validation Epoch {epoch+1} took: {str(timedelta(seconds=time.time() - end_epoch_time))}"
        )

    logger.info(f"All Training took: {str(timedelta(seconds=time.time() - time_start))}")
    engine.save_checkpoint(train_loss, valid_loss, config["epochs"])

    logger.info("#" * 50)
    logger.info("Evaluating model on Valid Dataset")

    nlquad_valid_data = read_nlquad(config["valid_path"])
    squad_valid_data = read_squad2("valid")

    valid_data = interleave(nlquad_valid_data, squad_valid_data, config["seed"])
    valid_dataset = prepare_features(
        valid_data, config["num_validating_examples"] * 2, mode="eval"
    )

    logger.info(f"Evaluating on {len(valid_dataset)} examples from Valid Dataset")
    validation_predictions = engine.evaluate(valid_loader_for_eval)

    logger.info("Calculating metrics for Validation Set : \n")
    valid_time = time.time()
    valid_results = calculate_metrics(valid_data, valid_dataset, validation_predictions)
    logger.info(f"Results on Validation Set: {valid_results}")
    logger.info(f"Validation took: {str(timedelta(seconds=time.time() - valid_time))}")

    logger.info("#" * 50)
    logger.info("Evaluating model on Eval Dataset")
    nlquad_eval_data = read_nlquad(config["eval_path"])
    eval_data = interleave(nlquad_eval_data, squad_valid_data, config["seed"])

    eval_dataset = prepare_features(
        eval_data, config["num_evaluation_examples"] * 2, mode="eval"
    )

    logger.info(f"Evaluating on {len(eval_dataset)} examples from Eval Dataset")
    evaluation_predictions = engine.evaluate(eval_loader)

    logger.info("Calculating metrics for Evaluation Set: \n")
    eval_time = time.time()
    eval_results = calculate_metrics(eval_data, eval_dataset, evaluation_predictions)
    logger.info(f"Results on Evaluation Set: {eval_results}")
    logger.info(f"Evaluation took: {str(timedelta(seconds=time.time() - eval_time))}")

    logger.info("#" * 50)
    logger.info("Saving results to results.json")
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"valid_results": valid_results, "eval_results": eval_results}, f)
