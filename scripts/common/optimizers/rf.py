from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score, f1_score,
                             recall_score, precision_score, matthews_corrcoef, confusion_matrix)
from cuml.ensemble import RandomForestClassifier as cuRF

import numpy as np
import wandb as wandb_lib
import pickle
import os

from scripts.common.optimizers.base import TuningOptimizer, Optimizer


class RandomForestTuningTrainable(TuningOptimizer):
    def step(self):
        config = self.config
        m = config.get("m")
        n_estimators = config.get("n_estimators")

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

            model = cuRF(
                # n_jobs=16,
                n_estimators=n_estimators,
                max_features=m,
                n_streams=1,  # for replicability
                verbose=1)

            X_train_fold = X_train_fold.astype(np.float32)
            X_val_fold = X_val_fold.astype(np.float32)

            model.fit(X_train_fold, y_train_fold)

            train_probs = model.predict_proba(X_train_fold)
            val_probs = model.predict_proba(X_val_fold)

            y_train_pred_score = train_probs[:, 1]
            y_val_pred_score = val_probs[:, 1]

            all_y_trues.extend(y_val_fold)
            all_y_scores.extend(val_probs)

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

            self.wandb.log({f"roc_curve_fold_{fold}": wandb_lib.plot.roc_curve(
                y_val_fold, val_probs, labels=labels)})

        metric_train = np.mean(cv_results[f'{metric}_train'])
        metric_valid = np.mean(cv_results[f'{metric}_valid'])

        loss_train = np.mean(cv_results['logloss_train'])
        loss_valid = np.mean(cv_results['logloss_valid'])

        acc_valid = np.mean(cv_results['accuracy_valid'])
        f1_valid = np.mean(cv_results['f1_valid'])

        all_y_trues_flat = np.array(all_y_trues).flatten()
        all_y_scores_flat = np.array(all_y_scores)

        print("all y trues flat", all_y_trues_flat)
        print("all y scores flat", all_y_scores_flat)

        self.wandb.log({"roc_overall": wandb_lib.plot.roc_curve(
            all_y_trues_flat, all_y_scores_flat, labels=labels)})

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


class RandomForestTrainable(Optimizer):
    def step(self):
        config = self.config
        m = config.get("m")
        n_estimators = config.get("n_estimators")

        labels = ['non-promoter', 'promoter']

        model = cuRF(
            n_estimators=n_estimators,
            max_features=m,
            n_streams=1,  # for replicability
            verbose=1)

        results = {
            'auc_train': [],
            'accuracy_train': [],
            'logloss_train': [],
            'f1_train': [],
            'auc_valid': [],
            'logloss_valid': [],
            'accuracy_valid': [],
            'f1_valid': []
        }
        X_train = self.X_train.astype(np.float32)
        X_val = self.X_val.astype(np.float32)

        model.fit(X_train, self.y_train)

        train_probs = model.predict_proba(X_train)
        val_probs = model.predict_proba(X_val)

        y_train_pred_score = train_probs[:, 1]
        y_val_pred_score = val_probs[:, 1]

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

        # Additional metrics
        recall_train = recall_score(self.y_train, y_train_pred)
        recall_valid = recall_score(self.y_val, y_val_pred)
        precision_train = precision_score(self.y_train, y_train_pred)
        precision_valid = precision_score(self.y_val, y_val_pred)
        mcc_train = matthews_corrcoef(self.y_train, y_train_pred)
        mcc_valid = matthews_corrcoef(self.y_val, y_val_pred)

        # Specificity calculation
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(self.y_train, y_train_pred).ravel()
        tn_valid, fp_valid, fn_valid, tp_valid = confusion_matrix(self.y_val, y_val_pred).ravel()
        specificity_train = tn_train / (tn_train + fp_train)
        specificity_valid = tn_valid / (tn_valid + fp_valid)

        results['auc_train'].append(roc_auc_score_train)
        results['logloss_train'].append(log_loss_train)
        results['accuracy_train'].append(accuracy_train)
        results['f1_train'].append(f1_train)

        results['auc_valid'].append(roc_auc_score_valid)
        results['logloss_valid'].append(log_loss_valid)
        results['accuracy_valid'].append(accuracy_valid)
        results['f1_valid'].append(f1_valid)

        self.wandb.log({"roc_overall": wandb_lib.plot.roc_curve(
            self.y_val, val_probs, labels=labels)})

        print('Finished!')

        self.wandb.log({"mean_roc_auc": roc_auc_score_valid, "mean_logloss": log_loss_valid,
                       "mean_accuracy": accuracy_valid, "mean_f1": f1_valid, "mean_recall": recall_valid,
                       "mean_precision": precision_valid, "mean_mcc": mcc_valid})

        self.wandb.finish()

        if (not os.path.exists('../models')):
            os.makedirs('../models')

        # save model
        model_path = f'/app/scripts/models/{self.non_promoter_origin}-{self.trial_name}.pkl'
        pickle.dump(model, open(model_path, "wb"))

        # Print all metrics
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

        return {"mean_roc_auc": roc_auc_score_valid, "mean_f1": f1_valid, "done": True}

    def save_checkpoint(self, checkpoint_dir: str):
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        pass