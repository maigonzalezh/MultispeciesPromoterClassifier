import numpy as np
import sklearn.metrics
import os
import sklearn

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from scripts.utils.bert import PretrainedModels, load_model, TorchDataset

from pathlib import Path
from scipy.special import softmax
from scripts.common.optimizers.base import Optimizer


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "roc_auc": sklearn.metrics.roc_auc_score(
            valid_labels, logits[valid_mask][:, 1]
        ),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


class BERTOptimizer(Optimizer):
    def setup(self, config,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              random_state: int,
              batch_size: int,
              non_promoter_origin: str = None,
              training_name: str = None,
              pretrained_model: str = None,
              max_seq_length: int = 512):
        super().setup(config, X_train, y_train, X_val,
                      y_val, random_state,
                      non_promoter_origin, training_name)
        self.pretrained_model = pretrained_model
        self.seq_max_len = max_seq_length
        self.batch_size = batch_size

    def step(self):
        model, tokenizer, device = load_model(
            model_name=self.pretrained_model, frozen=False)

        SEQ_MAX_LEN = self.seq_max_len

        train_encodings = tokenizer(
            self.X_train,
            max_length=SEQ_MAX_LEN,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        train_dataset = TorchDataset(
            train_encodings["input_ids"], train_encodings["attention_mask"], self.y_train
        )

        val_encodings = tokenizer(
            self.X_val,
            max_length=SEQ_MAX_LEN,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        results_dir = Path("./results/classification/")

        results_dir.mkdir(parents=True, exist_ok=True)

        lr = self.config['lr']

        EPOCHS = 10

        BATCH_SIZE = 32

        val_dataset = TorchDataset(
            val_encodings["input_ids"], val_encodings["attention_mask"], self.y_val
        )

        training_args = TrainingArguments(
            output_dir=results_dir / "checkpoints",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=8,
            logging_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=1,
            save_total_limit=1,
            weight_decay=0.01,
            learning_rate=lr,
            fp16=True,
            seed=self.random_state,
            eval_accumulation_steps=20
        )

        callback = [EarlyStoppingCallback(early_stopping_patience=5)]

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=callback
        )

        trainer.train()

        y_train_pred_obj = trainer.predict(train_dataset)
        y_val_pred_obj = trainer.predict(val_dataset)

        if (self.pretrained_model == PretrainedModels.DNABERT.value) or \
                (self.pretrained_model == PretrainedModels.NT_TRANSFORMER.value):
            y_train_predictions = y_train_pred_obj.predictions
            y_val_predictions = y_val_pred_obj.predictions

        elif self.pretrained_model == PretrainedModels.DNABERT2.value:
            y_train_predictions, _ = y_train_pred_obj.predictions
            y_val_predictions, _ = y_val_pred_obj.predictions

        y_train_pred_score = softmax(y_train_predictions, axis=1)
        y_val_pred_score = softmax(y_val_predictions, axis=1)

        metric_valid, loss_valid, \
            acc_valid, f1_valid = self.compute_overall_metrics(y_train_predictions, y_val_predictions,
                                                               y_train_pred_score, y_val_pred_score)

        self.wandb.log({"mean_roc_auc": metric_valid, "mean_logloss": loss_valid,
                       "mean_accuracy": acc_valid, "mean_f1": f1_valid})

        if (not os.path.exists('../models')):
            os.makedirs('../models')

        model_path = f'/app/scripts/models/{self.non_promoter_origin}-{self.trial_name}'

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        self.wandb.finish()
        return {"mean_roc_auc": metric_valid, "done": True}
