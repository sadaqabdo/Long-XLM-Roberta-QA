import collections
import re
import string
from collections import defaultdict

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from config import config

nltk.download("punkt")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_exact_match(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return (
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
        )
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def Jaccard_index(context, gold_answers, prediction):
    text = " ".join(word_tokenize(context)).lower()
    gold_answers = " ".join(word_tokenize(gold_answers[0])).lower()
    prediction = " ".join(word_tokenize(prediction)).lower()
    if prediction == "":
        pred_set = set()
    else:
        pred_start = text.find(prediction)
        pred_end = len(text) - (text[::-1].find(prediction[::-1]))
        pred_set = set(list(range(pred_start, pred_end)))
        if pred_start == -1 or pred_end == -1:
            pred_set = set()

    if gold_answers == "":
        gold_start = 0
        gold_end = 0
        gold_set = set()
    else:
        gold_start = text.find(gold_answers)
        gold_end = len(text) - (text[::-1].find(gold_answers[::-1]))
        gold_set = set(list(range(gold_start, gold_end)))
        if gold_start == -1 or gold_end == -1:
            gold_set = set()

    intersection = gold_set.intersection(pred_set)
    union = gold_set.union(pred_set)

    intersection_list = list(intersection)
    union_list = list(union)

    intersection_list.sort()
    union_list.sort()

    if not intersection_list:
        intersection_word = ""
    else:
        intersection_word = text[intersection_list[0] : intersection_list[-1] + 1]
    if not union_list:
        union_words = ""
    else:
        union_words = text[union_list[0] : union_list[-1] + 1]

    intersection_word_length = len(word_tokenize(intersection_word))
    union_word_length = len(word_tokenize(union_words))

    if intersection_word_length == 0 and union_word_length == 0:
        JI = 1
    else:
        JI = intersection_word_length / union_word_length

    return JI


def postprocess_qa_predictions(
    examples,
    features,
    predictions,
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 1000,
    null_score_diff_threshold: float = 0.0,
):
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features."
        )

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            offset_mapping = features[feature_index]["offset_mapping"]

            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            feature_null_score = start_logits[0] + end_logits[0]

            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:

                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue

                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

    return all_predictions


def calculate_metrics(examples, features, predictions):

    all_predictions = postprocess_qa_predictions(examples, features, predictions)
    examples_df = examples.to_pandas()

    id_to_data = {}
    for index, row in examples_df.iterrows():
        id_to_data[row["id"]] = (row["answer"], row["context"])

    f1 = defaultdict(float)
    precisions = defaultdict(float)
    recalls = defaultdict(float)
    em = defaultdict(float)
    JI = defaultdict(float)

    for id, pred_text in tqdm(all_predictions.items()):
        original_answer, original_context = id_to_data[id]
        f1[id], precisions[id], recalls[id] = compute_f1(original_answer, pred_text)
        em[id] = compute_exact(original_answer, pred_text)
        JI[id] = Jaccard_index(original_context, [original_answer], pred_text)

        if JI[id] == 0:
            JI[id] = Jaccard_index(original_context, [original_answer], pred_text[0:-5])

    f1_np = np.fromiter(f1.values(), dtype=float)
    recall_np = np.fromiter(recalls.values(), dtype=float)
    precision_np = np.fromiter(precisions.values(), dtype=float)
    em_np = np.fromiter(em.values(), dtype=float)
    JI_np = np.fromiter(JI.values(), dtype=float)

    print("\n mean F1 = ", np.mean(f1_np))
    print("mean Recall = ", np.mean(recall_np))
    print("mean Precision = ", np.mean(precision_np))
    print("mean EM = ", np.mean(em_np))
    print("mean Area Intersection over Union or Jaccard Index = ", np.mean(JI_np))

    return {
        "f1": np.mean(f1_np),
        "recall": np.mean(recall_np),
        "precision": np.mean(precision_np),
        "em": np.mean(em_np),
        "JI": np.mean(JI_np),
    }


def prepare_train_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = config["tokenizer"](
        examples["question" if config["pad_on_right"] else "context"],
        examples["context" if config["pad_on_right"] else "question"],
        truncation="only_second" if config["pad_on_right"] else "only_first",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):

        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(config["tokenizer"].cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answer = examples["answer"][sample_index]

        if len(answer) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:

            start_char = examples["answer_start"][sample_index]
            end_char = examples["answer_end"][sample_index]

            token_start_index = 0
            while sequence_ids[token_start_index] != (
                1 if config["pad_on_right"] else 0
            ):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if config["pad_on_right"] else 0):
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = config["tokenizer"](
        examples["question" if config["pad_on_right"] else "context"],
        examples["context" if config["pad_on_right"] else "question"],
        truncation="only_second" if config["pad_on_right"] else "only_first",
        max_length=config["max_length"],
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if config["pad_on_right"] else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples
