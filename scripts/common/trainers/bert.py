import pandas as pd
from scripts.common.optimizers.bert import BERTOptimizer
from scripts.common.trainers.base import TrainStudy
from sklearn.model_selection import train_test_split
from scripts.utils.bert import parse_torch_input_data


class BERTTrainStudy(TrainStudy):
    def __init__(self, project_name: str, storage_path: str,
                 n_samples: int, random_state: int,
                 test_size: float, batch_size: int,
                 non_promoter_origin: str,
                 param_space: dict,
                 pretrained_model: str,
                 max_seq_length: int = 512):
        super().__init__(project_name, storage_path, n_samples, random_state,
                         test_size, batch_size, param_space, non_promoter_origin)
        self.pretrained_model = pretrained_model
        self.max_seq_length = max_seq_length

    @property
    def optimizer(self):
        return BERTOptimizer

    @property
    def dataset(self):
        return pd.read_csv(f"./data/{self.non_promoter_origin}/dataset.csv")

    @property
    def training_name(self):
        return f'{self.pretrained_model.upper()}-train-{self.non_promoter_origin.upper()}'

    @property
    def tune_config_args(self):
        return {
            "num_samples": self.num_samples,
        }

    @property
    def resources(self):
        return {"cpu": 10, "gpu": 1}

    def get_trainable_params(self):
        parsed_sequences, labels, indices = parse_torch_input_data(
            self.dataset, model=self.pretrained_model)

        X_train, X_val, y_train, y_val, \
            index_train, index_test = train_test_split(parsed_sequences,
                                                       labels, indices,
                                                       test_size=self.test_size,
                                                       random_state=self.random_state)
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "non_promoter_origin": self.non_promoter_origin,
            "pretrained_model": self.pretrained_model,
            "training_name": self.training_name,
            "max_seq_length": self.max_seq_length
        }
