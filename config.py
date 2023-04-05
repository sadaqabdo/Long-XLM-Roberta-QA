import torch

config = {
    "epochs": 4,
    "batch_size": 4,
    "max_length": 4096,
    "doc_stride": 128,
    "learning_rate": 3e-05,
    "max_grad_norm": 10,  # try others
    "optimizer_type": "AdamW",
    "weight_decay": 1e-2,
    "epsilon": 1e-6,
    "fp16": True,
    "warmup_steps": 100,
    "num_examples": 2,
    "gradient_accumulation_steps": 1,
    "push_to_hub": False,
    "train_path": "./data/NLQuAD_train.json",
    "valid_path": "./data/NLQuAD_valid.json",
    "eval_path": "./data/NLQuAD_eval.json",
    "output_dir": "./",
    "pad_on_right": "right",
    "hub_model_id": "sadaqabdo/XLMRobertaLongForQuestionAnswering-base-squad2-512-4096",
    "access_token": "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",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
