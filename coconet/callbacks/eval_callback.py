import tensorflow as tf

import wandb
from wandb.keras import WandbEvalCallback


# TODO (ayulockin): Modify based on
# https://gist.github.com/ayulockin/659172226c1e8d5cc7c8c2a33ad095a3
# when the data and model pipeline are complete.
class RetinaNetWandbEvalCallback(WandbEvalCallback):
    def __init__(self, args):
        pass

    def add_ground_truth(self, logs):
        pass

    def add_model_predictions(self, epoch, logs):
        pass


def get_evaluation_callback(args):
    return RetinaNetWandbEvalCallback(
        args,
        dataloader,
        is_train=is_train,
    )
