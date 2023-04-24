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
        eps=config["epsilon"],
    )
    return optimizer


def get_scheduler(split_dataloader, optimizer, config):
    num_training_steps = (
        math.ceil(len(split_dataloader) / config["batch_size"]) * config["epochs"]
    )
    num_warmup_steps = config["warmup_steps"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"Total Training Steps: {num_training_steps}")
    return scheduler


def get_scaler():
    scaler = torch.cuda.amp.GradScaler()
    return scaler


class Engine:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = get_scaler()
        self.config = config

    def save_checkpoint(self, train_loss, valid_loss, epoch):
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])

        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            os.path.join(self.config["output_dir"], f"epoch_{epoch}"),
        )
        self.model.train()

    def train(self, train_dataloader, epoch):
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
            with torch.autocast(device_type=self.config["device"], dtype=torch.float16):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=targets_start,
                    end_positions=targets_end,
                )

                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.config["max_grad_norm"],
            )

            self.scaler.step(self.optimizer)
            old_scaler = self.scaler.get_scale()
            self.scaler.update()
            new_scaler = self.scaler.get_scale()
            if old_scaler > new_scaler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            # if loss is nan, stop training, and print the start and end logits
            if math.isnan(loss.item()) or torch.isnan(loss).any():
                print("Loss is nan, stopping training")
                print("Start Logits: ", output["start_logits"])
                print("End Logits: ", output["end_logits"])
                raise ValueError("Loss is nan")

            if batch_idx % self.config["print_freq"] == 0:
                print(f"Epoch: {epoch+1} \t Batch: {batch_idx} \t Loss: {loss.item()}")

        print("Training Loss: ", sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def validate(self, valid_dataloader, epoch):
        losses = []

        self.model.eval()

        for batch_data in tqdm(valid_dataloader):
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

            loss = output.loss

            losses.append(loss.item())

        print(f"Epoch: {epoch+1} \t Validation Loss: {sum(losses) / len(losses)}")

        return sum(losses) / len(losses)

    def evaluate(self, eval_dataloader):
        all_start_logits = []
        all_end_logits = []

        self.model.eval()
        for batch_data in tqdm(eval_dataloader):
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
