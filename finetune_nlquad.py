from transformers import XLMRobertaTokenizerFast

from config import config
from dataset import make_dataloaders, prepare_features, read_nlquad
from engine import Engine, get_optimizer, get_scheduler
from model import XLMRobertaLongForQuestionAnswering
from processing import calculate_metrics

if __name__ == "__main__":

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

    for epoch in range(config["epochs"]):
        engine.train(train_loader, epoch)
        engine.validate(valid_loader, epoch)
        engine.save_model(epoch)

    eval_data = read_nlquad(config["eval_path"])
    eval_dataset = prepare_features(eval_data, config["num_examples"] / 2, mode="eval")
    evaluation_predictions = engine.evaluate(eval_loader)
    calculate_metrics(eval_data, eval_dataset, evaluation_predictions)
