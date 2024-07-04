import argparse
import json
import os

from scripts.common.trainers.cnn import CNNTuningStudy, CNNTrainStudy
from ray import tune


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train models for promoter prediction")

    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")

    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds for cross validation")

    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train test split")

    parser.add_argument("--non_promoter_origin", type=str,
                        default='random', choices=['random', 'cds'])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    param_space = {
        "dropout": tune.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        "n_conv_layers": tune.choice([3]),
        "n_dense_layers": tune.choice([3]),
        "n_neurons-conv_1": tune.choice([128, 256, 512]),
        "kernel_size-conv_1": tune.choice([3, 5, 7]),
        "l2_regularizer-conv_1": tune.choice([0.001, 0.0001]),
        "l2_bias_regularizer-conv_1": tune.choice([0.001, 0.0001]),
        "n_neurons-conv_2": tune.choice([128, 256, 512]),
        "kernel_size-conv_2": tune.choice([3, 5, 7]),
        "l2_regularizer-conv_2": tune.choice([0.001, 0.0001]),
        "l2_bias_regularizer-conv_2": tune.choice([0.001, 0.0001]),
        "n_neurons-conv_3": tune.choice([128, 256, 512]),
        "kernel_size-conv_3": tune.choice([3, 5, 7]),
        "l2_regularizer-conv_3": tune.choice([0.001, 0.0001]),
        "l2_bias_regularizer-conv_3": tune.choice([0.001, 0.0001]),
        "n_neurons-dense_1": tune.choice([128, 256, 512]),
        "l2_regularizer-dense_1": tune.choice([0.001, 0.0001]),
        "l2_bias_regularizer-dense_1": tune.choice([0.001, 0.0001]),
        "n_neurons-dense_2": tune.choice([128, 256, 512]),
        "l2_regularizer-dense_2": tune.choice([0.001, 0.0001]),
        "l2_bias_regularizer-dense_2": tune.choice([0.001, 0.0001]),
        "n_neurons-dense_3": tune.choice([128, 256, 512]),
        "l2_regularizer-dense_3": tune.choice([0.001, 0.0001]),
        "l2_bias_regularizer-dense_3": tune.choice([0.001, 0.0001]),
    }

    common_args = {
        "project_name": "MultispeciesPromoterClassifier",
        "storage_path": "s3://promoter-classifier/",
        "n_samples": 60,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "non_promoter_origin": args.non_promoter_origin,
        'k_folds': args.k_folds,
        'param_space': param_space,
        'batch_size': 64,
    }

    study = CNNTuningStudy(**common_args)
    best_params = study.run_study()

    # save params
    with open(f'/app/scripts/tuning/cnn/best_params/{args.non_promoter_origin}.json', 'w') as f:
        json.dump(best_params, f)

    # best_params = {
    #     'cds': {
    #         'm': 'sqrt',
    #         'n_estimators': 3000
    #     },
    #     'random': {
    #         'm': 'sqrt',
    #         'n_estimators': 5000
    #     }
    # }

    # training_args = {
    #     "project_name": "MultispeciesPromoterClassifier",
    #     "storage_path": "s3://promoter-classifier/",
    #     "random_state": args.random_state,
    #     "test_size": args.test_size,
    #     "non_promoter_origin": args.non_promoter_origin,
    #     "n_samples": 1,
    #     'param_space': {
    #         "m": tune.grid_search([best_params[args.non_promoter_origin]['m']]),
    #         "n_estimators": tune.grid_search([best_params[args.non_promoter_origin]['n_estimators']]),
    #     },
    # }

    # training_study = RandomForestStudy(**training_args)

    # training_study.run_study()


if __name__ == "__main__":
    main()
