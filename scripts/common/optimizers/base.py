import numpy as np

from ray.air.integrations.wandb import setup_wandb
from ray import tune


class OptimizerBase(tune.Trainable):
    def setup(self, config, random_state: int, non_promoter_origin: str = None, training_name: str = None):
        self.config = config
        self.random_state = random_state
        self.non_promoter_origin = non_promoter_origin
        self.wandb = setup_wandb(config, trial_id=self.trial_id, trial_name=self.trial_name,
                                 group=training_name, project="MultispeciesPromoterClassifier")


class Optimizer(OptimizerBase):
    def setup(self,
              config,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              random_state: int,
              non_promoter_origin: str = None,
              training_name: str = None):
        super().setup(config, random_state, non_promoter_origin, training_name)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val


class TuningOptimizer(OptimizerBase):
    def setup(self,
              config,
              X: np.ndarray,
              y: np.ndarray,
              random_state: int,
              k_folds: int,
              non_promoter_origin: str = None,
              training_name: str = None):
        super().setup(config, random_state, non_promoter_origin, training_name)
        self.X = X
        self.y = y
        self.k_folds = k_folds
