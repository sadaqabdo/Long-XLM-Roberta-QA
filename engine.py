import math
import os

import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def get_optimizer(model, config):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
    )
    return optimizer


def get_scheduler(split_dataloader, optimizer, config):
    num_training_steps = math.ceil(len(split_dataloader) / 2) * config["epochs"]
    num_warmup_steps = config["warmup_steps"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps * config["gradient_accumulation_steps"],
        num_training_steps=num_training_steps * config["gradient_accumulation_steps"],
    )

    print(f"Total Training Steps: {num_training_steps}")
    return scheduler


class Engine:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def save_model(self, epoch):
        self.model.eval()
        self.model.save_pretrained(
            os.path.join(self.config["output_dir"], f"epoch_{epoch}")
        )
        self.model.train()

    def train(self, train_dataloader, epoch):
        count = 0
        losses = []
        self.model.zero_grad()
        self.model.train()

        for batch_idx, batch_data in tqdm(enumerate(train_dataloader)):
            batch_data = {k: v.to(self.config["device"]) for k, v in batch_data.items()}

            input_ids, attention_mask, targets_start, targets_end = (
                batch_data["input_ids"],
                batch_data["attention_mask"],
                batch_data["start_positions"],
                batch_data["end_positions"],
            )

            # fp16
            with torch.autocast(
                device_type=self.config["device"], dtype=torch.bfloat16
            ):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=targets_start,
                    end_positions=targets_end,
                )

                loss = output["loss"]
                loss.backward()

            count += input_ids.size(0)

            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.config["max_grad_norm"],
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if (batch_idx % 2 == 0) or (batch_idx + 1) == len(self.train_dataloader):
                print(f"Epoch {epoch+1} \t 'Train Loss: {loss.item()}")

        print("train_loss", sum(losses) / len(losses))

    def validate(self, valid_dataloader, epoch):
        losses = []
        all_start_logits = []
        all_end_logits = []

        self.model.eval()

        for batch_idx, batch_data in tqdm(enumerate(valid_dataloader)):
            batch_data = {k: v.to(self.config["device"]) for k, v in batch_data.items()}

            input_ids, attention_mask, targets_start, targets_end = (
                batch_data["input_ids"],
                batch_data["attention_mask"],
                batch_data["start_positions"],
                batch_data["end_positions"],
            )

            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=targets_start,
                    end_positions=targets_end,
                )
            outputs_start, outputs_end = output.start_logits, output.end_logits
            all_start_logits.append(outputs_start[0].cpu().numpy())
            all_end_logits.append(outputs_end[0].cpu().numpy())

            loss = output.loss

            losses.append(loss.item())

        print("Epoch {} Valid Loss: {: >4.5f}".format(epoch ,sum(losses) / len(losses)))
        return (all_start_logits, all_end_logits)

    def predict(self, eval_dataloader):
        all_start_logits = []
        all_end_logits = []

        self.model.eval()
        for batch_idx, batch_data in tqdm(enumerate(eval_dataloader)):
            batch_data = {k: v.to(self.config["device"]) for k, v in batch_data.items()}

            input_ids, attention_mask = (
                batch_data["input_ids"],
                batch_data["attention_mask"],
            )

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            outputs_start, outputs_end = output.start_logits, output.end_logits
            all_start_logits.append(outputs_start[0].cpu().numpy())
            all_end_logits.append(outputs_end[0].cpu().numpy())

        return (all_start_logits, all_end_logits)
