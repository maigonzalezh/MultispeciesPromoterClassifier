from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score, f1_score,
                             recall_score, precision_score, matthews_corrcoef, confusion_matrix)
from scripts.common.models_def.cnn import create_cnn_base
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import Adam

import numpy as np
import wandb as wandb_lib
import pickle
import os

from scripts.common.optimizers.base import TuningOptimizer, Optimizer


class CNNTuningOptimizer(TuningOptimizer):
    def setup(self,
              config,
              X: np.ndarray,
              y: np.ndarray,
              random_state: int,
              k_folds: int,
              non_promoter_origin: str = None,
              training_name: str = None,
              batch_size=32):

        super().setup(config, X, y, random_state, k_folds, non_promoter_origin, training_name)
        self.batch_size = batch_size


class CNNOptimizer(Optimizer):
    def setup(self,
              config,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              random_state: int,
              non_promoter_origin: str = None,
              training_name: str = None,
              batch_size=32):

        super().setup(config, X_train, y_train, X_val, y_val,
                      random_state, non_promoter_origin, training_name)
        self.batch_size = batch_size

class WandbReportingFoldCallback(Callback):
    def __init__(self, fold, wandb):
        super().__init__()
        self.fold = fold
        self.wandb = wandb

    def on_epoch_end(self, epoch, logs=None):
        self.wandb.log({f"fold_{self.fold}_val_loss": logs.get('val_loss'),
                        f"fold_{self.fold}_val_accuracy": logs.get('val_accuracy'),
                        f"fold_{self.fold}_loss": logs.get('loss'),
                        f"fold_{self.fold}_accuracy": logs.get('accuracy'),
                        "epoch": epoch})

class WandbReportingCallback(Callback):
    def __init__(self, wandb):
        super().__init__()
        self.wandb = wandb

    def on_epoch_end(self, epoch, logs=None):
        self.wandb.log({f"val_loss": logs.get('val_loss'),
                        f"val_accuracy": logs.get('val_accuracy'),
                        f"loss": logs.get('loss'),
                        f"accuracy": logs.get('accuracy'),
                        "epoch": epoch})


def get_predictions_by_class(y_pred):
    y_pred_score_pos = y_pred[:, 0]
    y_pred_score_neg = y_pred_score_pos * -1 + 1
    predictions_by_class = np.concatenate(
        (y_pred_score_neg.reshape(-1, 1), y_pred_score_pos.reshape(-1, 1)), axis=1)

    return predictions_by_class


def cnn_config_builder(config):
    dropout = config['dropout']
    n_conv_layers = config['n_conv_layers']
    n_dense_layers = config['n_dense_layers']

    conv_layers = []
    dense_layers = []

    for i in range(n_conv_layers):
        layer = {}
        layer['n_neurons'] = config[f'n_neurons-conv_{i+1}']
        layer['kernel_size'] = config[f'kernel_size-conv_{i+1}']
        layer['l2_regularizer'] = config[f'l2_regularizer-conv_{i+1}']
        layer["l2_bias_regularizer"] = config[f'l2_bias_regularizer-conv_{i+1}']

        conv_layers.append(layer)

    for i in range(n_dense_layers):
        layer = {}
        layer['n_neurons'] = config[f'n_neurons-dense_{i+1}']
        layer['l2_regularizer'] = config[f'l2_regularizer-dense_{i+1}']
        layer["l2_bias_regularizer"] = config[f'l2_bias_regularizer-dense_{i+1}']

        dense_layers.append(layer)

    return dropout, conv_layers, dense_layers


class CNNTuningTrainable(CNNTuningOptimizer):
    def step(self):
        dropout, conv_layers, dense_layers = cnn_config_builder(self.config)
        metric = "auc"

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True,
                              random_state=self.random_state)

        cv_results = {
            f'{metric}_train': [],
            'accuracy_train': [],
            'logloss_train': [],
            'f1_train': [],
            f'{metric}_valid': [],
            'logloss_valid': [],
            'accuracy_valid': [],
            'f1_valid': []
        }

        labels = ['non-promoter', 'promoter']

        all_y_trues = []
        all_y_scores = []

        for index, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
            fold = index + 1
            X_train_fold, X_val_fold = self.X[train_index], self.X[val_index]
            y_train_fold, y_val_fold = self.y[train_index], self.y[val_index]

            model = create_cnn_base(
                input_shape=X_train_fold.shape[1:],
                dropout=dropout,
                conv_layers=conv_layers,
                dense_layers=dense_layers)

            lr = 1e-4

            callbacks = [WandbReportingCallback(self.wandb)]
            epochs_per_fold = 100

            optimizer = Adam(learning_rate=lr)
            callbacks.append(EarlyStopping(patience=10, monitor='val_loss'))

            model.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.fit(X_train_fold,
                      y_train_fold,
                      epochs=epochs_per_fold,
                      batch_size=self.batch_size,
                      validation_data=(X_val_fold, y_val_fold),
                      verbose=2,
                      callbacks=[
                          WandbReportingFoldCallback(fold, self.wandb),
                          EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
                      ])

            y_train_pred_score = model.predict(X_train_fold)
            y_val_pred_score = model.predict(X_val_fold)

            all_y_trues.extend(y_val_fold)
            all_y_scores.extend(y_val_pred_score)

            y_train_pred = np.where(y_train_pred_score > 0.5, 1, 0)
            y_val_pred = np.where(y_val_pred_score > 0.5, 1, 0)

            log_loss_train = log_loss(y_train_fold, y_train_pred_score)
            log_loss_valid = log_loss(y_val_fold, y_val_pred_score)
            roc_auc_score_train = roc_auc_score(
                y_train_fold, y_train_pred_score)
            roc_auc_score_valid = roc_auc_score(y_val_fold, y_val_pred_score)
            accuracy_train = accuracy_score(y_train_fold, y_train_pred)
            accuracy_valid = accuracy_score(y_val_fold, y_val_pred)
            f1_train = f1_score(y_train_fold, y_train_pred)
            f1_valid = f1_score(y_val_fold, y_val_pred)

            cv_results[f'{metric}_train'].append(roc_auc_score_train)
            cv_results['logloss_train'].append(log_loss_train)
            cv_results['accuracy_train'].append(accuracy_train)
            cv_results['f1_train'].append(f1_train)
            cv_results[f'{metric}_valid'].append(roc_auc_score_valid)
            cv_results['logloss_valid'].append(log_loss_valid)
            cv_results['accuracy_valid'].append(accuracy_valid)
            cv_results['f1_valid'].append(f1_valid)

            predictions_by_class = get_predictions_by_class(y_val_pred_score)

            self.wandb.log({f"roc_curve_fold_{fold}": wandb_lib.plot.roc_curve(
                y_val_fold, predictions_by_class, labels=labels)})

        metric_train = np.mean(cv_results[f'{metric}_train'])
        metric_valid = np.mean(cv_results[f'{metric}_valid'])

        loss_train = np.mean(cv_results['logloss_train'])
        loss_valid = np.mean(cv_results['logloss_valid'])

        acc_valid = np.mean(cv_results['accuracy_valid'])
        f1_valid = np.mean(cv_results['f1_valid'])

        all_y_trues_flat = np.array(all_y_trues).flatten()
        all_y_scores_flat = np.array(all_y_scores).reshape(-1, 1)

        all_predictions_by_class = get_predictions_by_class(all_y_scores_flat)

        self.wandb.log({"roc_overall": wandb_lib.plot.roc_curve(
            all_y_trues_flat, all_predictions_by_class, labels=labels)})

        print('Finished!')
        print(f'Train {metric}:{metric_train}')
        print(f'Valid {metric}:{metric_valid}')
        print('Train Loss:{}'.format(loss_train))
        print('Valid Loss:{}'.format(loss_valid))

        self.wandb.log({"mean_roc_auc": metric_valid, "mean_logloss": loss_valid,
                       "mean_accuracy": acc_valid, "mean_f1": f1_valid})

        self.wandb.finish()

        return {"mean_roc_auc": metric_valid, "mean_f1": f1_valid, "done": True}

    def save_checkpoint(self, checkpoint_dir: str):
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        pass


class CNNTrainable(CNNOptimizer):
    def step(self):
        config = self.config

        dropout, conv_layers, dense_layers = cnn_config_builder(
            self.best_params)
        metric = "auc"

        results = {
            f'{metric}_train': [],
            'accuracy_train': [],
            'logloss_train': [],
            'f1_train': [],
            f'{metric}_valid': [],
            'logloss_valid': [],
            'accuracy_valid': [],
            'f1_valid': []
        }

        labels = ['non-promoter', 'promoter']

        model = create_cnn_base(
            input_shape=self.X_train.shape[1:],
            dropout=dropout,
            conv_layers=conv_layers,
            dense_layers=dense_layers)

        lr = config['lr']

        callbacks = [WandbReportingCallback(self.wandb)]
        epochs = 100

        optimizer = Adam(learning_rate=lr)
        callbacks.append(EarlyStopping(patience=10, monitor='val_loss'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=self.batch_size,
                  validation_data=(self.X_val, self.y_val), verbose=2, callbacks=callbacks)

        y_train_pred_score = model.predict(self.X_train)
        y_val_pred_score = model.predict(self.X_val)
        y_train_pred = np.where(y_train_pred_score > 0.5, 1, 0)
        y_val_pred = np.where(y_val_pred_score > 0.5, 1, 0)

        log_loss_train = log_loss(self.y_train, y_train_pred_score)
        log_loss_valid = log_loss(self.y_val, y_val_pred_score)

        roc_auc_score_train = roc_auc_score(self.y_train, y_train_pred_score)
        roc_auc_score_valid = roc_auc_score(self.y_val, y_val_pred_score)

        accuracy_train = accuracy_score(self.y_train, y_train_pred)
        accuracy_valid = accuracy_score(self.y_val, y_val_pred)

        f1_train = f1_score(self.y_train, y_train_pred)
        f1_valid = f1_score(self.y_val, y_val_pred)

        recall_valid = recall_score(self.y_val, y_val_pred)
        precision_valid = precision_score(self.y_val, y_val_pred)
        mcc_valid = matthews_corrcoef(self.y_val, y_val_pred)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(
            self.y_train, y_train_pred).ravel()
        tn_valid, fp_valid, fn_valid, tp_valid = confusion_matrix(
            self.y_val, y_val_pred).ravel()
        specificity_train = tn_train / (tn_train + fp_train)
        specificity_valid = tn_valid / (tn_valid + fp_valid)

        results[f'{metric}_train'].append(roc_auc_score_train)
        results['logloss_train'].append(log_loss_train)
        results['accuracy_train'].append(accuracy_train)
        results['f1_train'].append(f1_train)

        results[f'{metric}_valid'].append(roc_auc_score_valid)
        results['logloss_valid'].append(log_loss_valid)
        results['accuracy_valid'].append(accuracy_valid)
        results['f1_valid'].append(f1_valid)

        predictions_by_class = get_predictions_by_class(y_val_pred_score)
        self.wandb.log({f"roc_curve": wandb_lib.plot.roc_curve(
            self.y_val, predictions_by_class, labels=labels)})

        metric_train = results[f'{metric}_train'][0]
        metric_valid = results[f'{metric}_valid'][0]

        loss_train = results['logloss_train'][0]
        loss_valid = results['logloss_valid'][0]

        acc_valid = results['accuracy_valid'][0]
        f1_valid = results['f1_valid'][0]

        print('Finished!')
        print(f'Train {metric}:{metric_train}')
        print(f'Valid {metric}:{metric_valid}')
        print('Train Loss:{}'.format(loss_train))
        print('Valid Loss:{}'.format(loss_valid))

        if (not os.path.exists('../models')):
            os.makedirs('../models')

        # save model
        model_path = f'/app/scripts/models/{self.non_promoter_origin}-{self.trial_name}.keras'

        if (not os.path.exists('./models')):
            os.makedirs('./models')

        model.save(model_path)

        self.wandb.finish()

        metrics_summary = {
            "ROC AUC": roc_auc_score_valid,
            "Log Loss": log_loss_valid,
            "Accuracy": accuracy_valid,
            "F1 Score": f1_valid,
            "Recall (sensibility)": recall_valid,
            "Precision": precision_valid,
            "MCC": mcc_valid,
            "Specificity": specificity_valid,
        }

        print("Metrics Summary:")
        for metric, value in metrics_summary.items():
            print(f"{metric}: {value}")

        return {"roc_auc": metric_valid, "logloss": loss_valid,
                "accuracy": acc_valid, "f1": f1_valid, "done": True}

    def save_checkpoint(self, checkpoint_dir: str):
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        pass
