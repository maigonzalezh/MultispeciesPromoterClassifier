import argparse
from scripts.common.trainers.bert import BERTTrainStudy
from ray import tune


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train models for promoter prediction")

    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")

    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train test split")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")

    parser.add_argument("--non_promoter_origin", type=str,
                        default='random', choices=['random', 'cds'])

    parser.add_argument('--model', type=str, default='dnabert2',
                        choices=['dnabert', 'dnabert2', 'nt-transformer'],
                        help='Model to use for training')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    common_args = {
        "project_name": "PromoterClassifier",
        "storage_path": "s3://promoter-classifier/",
        "n_samples": 1,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "non_promoter_origin": args.non_promoter_origin,
    }

    nt_study = BERTTrainStudy(**common_args,
                              param_space={"lr": tune.grid_search(
                                  [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
                              )},
                              pretrained_model='nt-transformer',
                              max_seq_length=128)

    dnabert_study = BERTTrainStudy(**common_args,
                                   param_space={"lr": tune.grid_search(
                                       [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
                                   )},
                                   pretrained_model='dnabert',
                                   max_seq_length=128)

    dnabert2_study = BERTTrainStudy(**common_args,
                                    param_space={"lr": tune.grid_search(
                                        [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
                                    )},
                                    pretrained_model='dnabert2',
                                    max_seq_length=25)

    nt_study.run_study()
    dnabert_study.run_study()
    dnabert2_study.run_study()


if __name__ == "__main__":
    main()
