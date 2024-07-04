from abc import ABC, abstractmethod
from ray import tune, train


class TrainStudy(ABC):
    def __init__(self,
                 project_name: str,
                 storage_path: str,
                 n_samples: int,
                 random_state: int,
                 test_size: float,
                 param_space: dict,
                 non_promoter_origin: str):
        self.project_name = project_name
        self.storage_path = storage_path
        self.random_state = random_state
        self.test_size = test_size
        self.non_promoter_origin = non_promoter_origin
        self.num_samples = n_samples
        self.param_space = param_space

    @property
    def search_alg(self):
        return None

    @property
    @abstractmethod
    def optimizer(self):
        pass

    @property
    @abstractmethod
    def dataset(self):
        pass

    @property
    @abstractmethod
    def training_name(self):
        pass

    @property
    def storage_training_path(self):
        return f'{self.storage_path}{self.training_name}'

    @property
    @abstractmethod
    def resources(self):
        pass

    @property
    def get_trainable_params(self):
        pass

    @property
    @abstractmethod
    def tune_config_args(self):
        pass

    def run_study(self):
        trainable_params = self.get_trainable_params()
        print(trainable_params)
        print(self.optimizer)
        trainable = tune.with_parameters(
            self.optimizer, **trainable_params)

        trainable_with_resources = tune.with_resources(
            trainable=trainable, resources=self.resources)

        if (tune.Tuner.can_restore(self.storage_training_path)):
            tuner = tune.Tuner.restore(
                self.storage_training_path,
                trainable=trainable_with_resources,
                resume_errored=True)
        else:
            tuner = tune.Tuner(
                trainable_with_resources,
                tune_config=tune.TuneConfig(
                    **self.tune_config_args,
                ),
                run_config=train.RunConfig(
                    name=self.training_name,
                    storage_path=self.storage_path,
                ),
                param_space=self.param_space,
            )

        results = tuner.fit()
        best_params = results.get_best_result().config

        print("Best hyperparameters found were: ", best_params)

        return best_params
