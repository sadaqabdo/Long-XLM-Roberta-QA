from transformers import XLMRobertaTokenizerFast

from config import config
from dataset import make_dataloaders
from engine import Engine, get_optimizer, get_scheduler
from model import XLMRobertaLongForQuestionAnswering

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

    engine.evaluate(eval_loader)
