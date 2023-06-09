import json

import datasets
import pandas as pd
from datasets import Sequence, Value, interleave_datasets, load_dataset
from numpy import int32
from torch.utils.data import DataLoader
from tqdm import tqdm

from processing import prepare_train_features, prepare_validation_features


def read_nlquad(path):
    with open(path, "r", encoding="utf-8") as reader:
        split = json.load(reader)

    new_data = list()
    for data in tqdm(split["data"]):
        for paragraph in data["paragraphs"]:
            for qas in paragraph["qas"]:
                assert len(qas["answers"]) == 1, "oops"
                for answer in qas["answers"]:
                    single_tuple = {}
                    single_tuple["id"] = qas["id"]
                    single_tuple["title"] = data["title"]
                    single_tuple["context"] = paragraph["context"]
                    single_tuple["question"] = qas["question"]

                    single_tuple["answers"] = {}
                    single_tuple["answers"]["text"] = [answer["text"]]
                    single_tuple["answers"]["answer_start"] = [
                        int32(answer["answer_start"])
                    ]

                    new_data.append(single_tuple)
    tmpdf = pd.DataFrame(new_data)

    return datasets.Dataset.from_pandas(tmpdf)


def prepare_features(split_data, num_examples, mode="train"):
    if num_examples < 0:
        num_examples = len(split_data)

    if mode == "train":
        split_dataset = (
            split_data.select(range(int(num_examples)))
            .map(
                prepare_train_features,
                remove_columns=split_data.column_names,
                batched=True,
            )
            .with_format("torch")
        )
    else:
        split_dataset = (
            split_data.select(range(int(num_examples)))
            .map(
                prepare_validation_features,
                remove_columns=split_data.column_names,
                batched=True,
            )
            .with_format("torch")
        )
    return split_dataset


def set_loader(split, batch_size, columns_to_remove=None):
    if columns_to_remove is not None:
        split = split.remove_columns(columns_to_remove)

    split_loader = DataLoader(
        split,
        batch_size=int(batch_size),
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )

    return split_loader


def make_dataloaders(config):
    train_data = read_nlquad(config["train_path"])
    valid_data = read_nlquad(config["valid_path"])
    eval_data = read_nlquad(config["eval_path"])

    train_dataset = prepare_features(
        train_data, config["nlquad_num_training_examples"], mode="train"
    )
    valid_dataset = prepare_features(
        valid_data, config["nlquad_num_validating_examples"], mode="train"
    )
    valid_dataset_for_eval = prepare_features(
        valid_data, config["nlquad_num_validating_examples"], mode="eval"
    )
    eval_dataset = prepare_features(
        eval_data, config["nlquad_num_evaluation_examples"], mode="eval"
    )

    if config["squad_v2"]:
        squad2_train_data = read_squad2("train")
        squad2_valid_data = read_squad2("validation")

        squad2_train_dataset = prepare_features(
            squad2_train_data, config["squadv2_num_training_examples"], mode="train"
        )
        squad2_valid_dataset = prepare_features(
            squad2_valid_data, config["squadv2_num_validating_examples"], mode="train"
        )
        squad2_valid_dataset_for_eval = prepare_features(
            squad2_valid_data, config["squadv2_num_validating_examples"], mode="eval"
        )

        train_dataset = interleave(squad2_train_dataset, train_dataset, config["seed"])
        valid_dataset = interleave(squad2_valid_dataset, valid_dataset, config["seed"])
        valid_dataset_for_eval = interleave(
            squad2_valid_dataset_for_eval, valid_dataset_for_eval, config["seed"]
        )
        eval_dataset = interleave(
            squad2_valid_dataset_for_eval, eval_dataset, config["seed"]
        )

    train_loader = set_loader(train_dataset, config["batch_size"])
    valid_loader = set_loader(valid_dataset, config["batch_size"])
    valid_loader_for_eval = set_loader(
        valid_dataset_for_eval,
        config["batch_size"],
        columns_to_remove=["example_id", "offset_mapping"],
    )
    eval_loader = set_loader(
        eval_dataset,
        config["batch_size"],
        columns_to_remove=["example_id", "offset_mapping"],
    )

    return train_loader, valid_loader, valid_loader_for_eval, eval_loader


def read_squad2(split):
    squad2_split = load_dataset("squad_v2", split=split)
    return squad2_split


def cast_dataset_features(data):
    new_features = data.features.copy()
    new_features["answers"] = {
        "answer_start": Sequence(
            feature=Value(dtype="int32", id=None), length=-1, id=None
        ),
        "text": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
    }
    data = data.cast(new_features)
    return data


def interleave(dataset1, dataset2, seed):
    return interleave_datasets([dataset1, dataset2], seed=seed)
