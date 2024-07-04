import numpy as np
from typing import Dict, Union
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import json
from tabulate import tabulate


def build_metrics(df, save=False, save_path=None):
    unique_species = df['SpeciesName'].unique()

    specie_metrics = []

    for specie in unique_species:
        specie_df = df[df['SpeciesName'] == specie]

        y_true_specie = specie_df['label'].values
        y_pred_score_specie = specie_df['pred_score'].values

        specie_metrics.append({
            "specie_name": specie,
            "metrics": compute_metrics(y_true_specie, y_pred_score_specie)
        })

    metrics = {
        'by_specie': specie_metrics,
        'overall': compute_metrics(df['label'].values, df['pred_score'].values)
    }

    if save:
        with open(save_path, 'w') as f:
            json.dump(metrics, f)

    return metrics


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Union[int, float]]:
    y_pred = np.where(y_score > 0.5, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    total = tn+fp+fn+tp
    roc_auc = roc_auc_score(y_true, y_score)

    # round metrics
    specificity = round(specificity, 4)
    sensitivity = round(sensitivity, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    accuracy = round(accuracy, 4)
    mcc = round(mcc, 4)
    f1 = round(f1, 4)
    roc_auc = round(roc_auc, 4)

    return {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'total': total,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'mcc': mcc,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_true': y_true.tolist(),
        'y_score': y_score.tolist()
    }


def print_metrics_by_specie(metrics):
    table = [['Specie', 'Specificity', 'Sensitivity', 'Precision', 'Recall',
              'Accuracy', 'MCC', 'F1-score', 'ROC AUC', 'TP', 'FP', 'TN', 'FN', "Total"]]

    specie_metrics = metrics['by_specie']
    specie_metrics = sorted(
        specie_metrics, key=lambda x: x['metrics']['total'], reverse=True)

    for specie in specie_metrics:
        metrics = specie['metrics']
        table.append([specie['specie_name'],
                      metrics['specificity'],
                      metrics['sensitivity'],
                      metrics['precision'],
                      metrics['recall'],
                      metrics['accuracy'],
                      metrics['mcc'],
                      metrics['f1'],
                      metrics['roc_auc'],
                      metrics['tp'],
                      metrics['fp'],
                      metrics['tn'],
                      metrics['fn'],
                      metrics['total']])

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

def print_metrics_by_models_species(metrics: list):
    table = [['Modelo', 'Specie', 'Specificity', 'Sensitivity', 'Precision', 'Recall',
              'Accuracy', 'MCC', 'F1-score', 'ROC AUC', 'TP', 'FP', 'TN', 'FN', "Total"]]

    specie_metrics = metrics['by_specie']
    specie_metrics = sorted(
        specie_metrics, key=lambda x: x['metrics']['total'], reverse=True)

    for specie in specie_metrics:
        metrics = specie['metrics']
        table.append([specie['specie_name'],
                      metrics['specificity'],
                      metrics['sensitivity'],
                      metrics['precision'],
                      metrics['recall'],
                      metrics['accuracy'],
                      metrics['mcc'],
                      metrics['f1'],
                      metrics['roc_auc'],
                      metrics['tp'],
                      metrics['fp'],
                      metrics['tn'],
                      metrics['fn'],
                      metrics['total']])

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))