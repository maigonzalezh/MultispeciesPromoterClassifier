import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import evaluate
from scipy.special import expit as sigmoid
from enum import Enum
from peft import LoraConfig, TaskType, get_peft_model
from typing import Optional, Dict, Sequence, Tuple, List


class PretrainedModels(Enum):
    DNABERT = "dnabert"
    DNABERT2 = "dnabert2"
    NT_TRANSFORMER = "nt-transformer"


model_paths = {
    "dnabert": "zhihan1996/DNA_bert_3",
    'dnabert2': "zhihan1996/DNABERT-2-117M",
    "nt-transformer": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
}


def return_kmer(seq, K=3):
    kmer_list = []
    for x in range(len(seq) - K + 1):  # move a window of size K across the sequence
        kmer_list.append(seq[x: x + K])

    kmer_seq = " ".join(kmer_list)
    return kmer_seq


def parse_torch_input_data(dataset_df: pd.DataFrame, model: str = PretrainedModels.DNABERT.value):
    sequences = dataset_df['Sequence'].values
    sequences = [seq.upper() for seq in sequences]
    labels = dataset_df['label'].values
    indices = range(len(sequences))

    if model == PretrainedModels.DNABERT.value:
        kmers_sequences = [return_kmer(seq, K=3) for seq in sequences]
        return kmers_sequences, labels, indices

    else:
        return sequences, labels, indices


def load_model(model_name: str, frozen: bool, return_model=True,
               custom_model_path=None, use_lora: bool = False):
    device = None

    if torch.cuda.is_available():
        # for CUDA
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        print("Running the model on CUDA")

    elif torch.backends.mps.is_available():
        # for M1
        device = torch.device("mps")
        print("Running the model on M1 CPU")

    else:
        print("Running the model on CPU")

    model_path = model_paths[model_name] if custom_model_path is None else custom_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=21,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if model_name == PretrainedModels.NT_TRANSFORMER.value:
        tokenizer.eos_token = tokenizer.pad_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=True
    )

    # if model_name == PretrainedModels.NT_TRANSFORMER.value:
    #     lora_target_modules = 'query,value,key,dense'
    #     lora_alpha = 32
    #     lora_r = 8
    #     lora_dropout = 0.05

    #     lora_config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=list(lora_target_modules.split(",")),
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="SEQ_CLS",
    #         inference_mode=False,
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()

    if frozen:
        for param in model.bert.parameters():
            param.requires_grad = False

    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05,
    #     #modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
    # )

    # lora_classifier = get_peft_model(model, peft_config)
    # lora_classifier.print_trainable_parameters()
    # lora_classifier.to(device)

    model.to(device)

    if return_model:
        return model, tokenizer, device


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


roc_auc_score = evaluate.load("roc_auc")
accuracy_score = evaluate.load("accuracy")
f1_score = evaluate.load("f1")
recall = evaluate.load("recall")
precision_score = evaluate.load("precision")


def compute_metrics(eval_preds, model_name: str):

    if model_name == PretrainedModels.DNABERT.value:
        predictions, labels = eval_preds

    elif model_name == PretrainedModels.DNABERT2.value:
        pred_data, labels = eval_preds
        predictions, _ = pred_data

    elif model_name == PretrainedModels.NT_TRANSFORMER.value:
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

    predictions = np.argmax(predictions, axis=1)
    probs = sigmoid(predictions)

    accuracy = accuracy_score.compute(
        predictions=predictions, references=labels)["accuracy"]
    precision = precision_score.compute(
        predictions=predictions, references=labels)["precision"]
    rec = recall.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_score.compute(predictions=predictions, references=labels)["f1"]
    roc_auc = roc_auc_score.compute(
        prediction_scores=probs, references=labels)["roc_auc"]

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

    return metrics
