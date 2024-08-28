import argparse
from scripts.common.trainers.rf import RandomForestTuningStudy, RandomForestStudy
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

    # common_args = {
    #     "project_name": "MultispeciesPromoterClassifier",
    #     "storage_path": "s3://promoter-classifier/",
    #     "n_samples": 1,
    #     "random_state": args.random_state,
    #     "test_size": args.test_size,
    #     "non_promoter_origin": args.non_promoter_origin,
    #     'k_folds': args.k_folds,
        # 'param_space': {
        #     "m": tune.grid_search(["sqrt", "log2"]),
        #     "n_estimators": tune.grid_search([1000, 2000, 3000, 4000, 5000]),
        # },
    # }

    # study = RandomForestTuningStudy(**common_args)
    # study.run_study()

    best_params = {
        'cds': {
            'm': 'sqrt',
            'n_estimators': 3000
        },
        'random': {
            'm': 'sqrt',
            'n_estimators': 5000
        }
    }

    training_args = {
        "project_name": "MultispeciesPromoterClassifier",
        "storage_path": "s3://promoter-classifier/",
        "random_state": args.random_state,
        "test_size": args.test_size,
        "non_promoter_origin": args.non_promoter_origin,
        "n_samples": 1,
        'param_space': {
            "m": tune.grid_search([best_params[args.non_promoter_origin]['m']]),
            "n_estimators": tune.grid_search([best_params[args.non_promoter_origin]['n_estimators']]),
        },
    }

    training_study = RandomForestStudy(**training_args)

    training_study.run_study()


if __name__ == "__main__":
    main()
