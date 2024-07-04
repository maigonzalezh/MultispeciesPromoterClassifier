import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.utils.dataset import parse_data
from scripts.common.trainers.base import TrainStudy
from scripts.common.optimizers.cnn import CNNTuningTrainable, CNNTrainable
from ray.tune.search.optuna import OptunaSearch


class CNNTuningStudy(TrainStudy):
    def __init__(self,
                 project_name: str,
                 storage_path: str,
                 n_samples: int,
                 random_state: int,
                 test_size: float,
                 non_promoter_origin: str,
                 param_space: dict,
                 k_folds: int = 5,
                 batch_size: int = 32,
                 ):
        super().__init__(project_name, storage_path, n_samples, random_state,
                         test_size, param_space, non_promoter_origin)
        self.k_folds = k_folds
        self.batch_size = batch_size

    @property
    def optimizer(self):
        return CNNTuningTrainable

    @property
    def dataset(self):
        return pd.read_csv(f"./data/{self.non_promoter_origin}/dataset.csv")

    @property
    def training_name(self):
        return f'CNN-tuning-{self.non_promoter_origin.upper()}'

    @property
    def tune_config_args(self):
        return {
            "num_samples": self.num_samples,
            "search_alg": OptunaSearch(),
            "metric": "mean_f1",
            "mode": "max",
        }

    @property
    def resources(self):
        return {"cpu": 8, "gpu": 1}

    def get_trainable_params(self):
        X, y = parse_data(self.dataset, is2d=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        return {
            "X": X_train,
            "y": y_train,
            "random_state": self.random_state,
            "k_folds": self.k_folds,
            "non_promoter_origin": self.non_promoter_origin,
            "training_name": self.training_name,
            "batch_size": self.batch_size,
        }

class CNNTrainStudy(TrainStudy):
    def __init__(self,
                 project_name: str,
                 storage_path: str,
                 n_samples: int,
                 random_state: int,
                 test_size: float,
                 non_promoter_origin: str,
                 param_space: dict,
                 ):
        super().__init__(project_name, storage_path, n_samples, random_state,
                         test_size, param_space, non_promoter_origin)

    @property
    def optimizer(self):
        return CNNTrainable

    @property
    def dataset(self):
        return pd.read_csv(f"./data/{self.non_promoter_origin}/dataset.csv")

    @property
    def training_name(self):
        return f'CNN-train-{self.non_promoter_origin.upper()}'

    @property
    def tune_config_args(self):
        return {
            "num_samples": self.num_samples,
        }

    @property
    def resources(self):
        return {"cpu": 10, "gpu": 1}

    def get_trainable_params(self):
        X, y = parse_data(self.dataset, is2d=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "random_state": self.random_state,
            "non_promoter_origin": self.non_promoter_origin,
            "training_name": self.training_name,
        }
