import os
import ml_collections
from ml_collections import config_dict


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "coconet"

    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()

    return configs


def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_name = "RetinaNet"

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()

    # Early stopping
    configs.early_stopper = ml_collections.ConfigDict()
    configs.use_earlystopping = True
    early_stopper.early_patience = 6

    # Reduce LR on plateau
    configs.reduce_lr_on_plateau = ml_collections.ConfigDict()
    configs.use_reduce_lr_on_plateau = False
    reduce_lr_on_plateau.factor = 0.2
    reduce_lr_on_plateau.patience = 3

    # Model checkpointing
    configs.ckpt_callback = ConfigDict()
    ckpt_callback.checkpoint_filepath = "wandb/model_{epoch}"
    ckpt_callback.monitor = "val_loss"
    ckpt_callback.save_best_only = True
    ckpt_callback.save_weights_only = False
    ckpt_callback.initial_value_threshold = 0.0

    # Model evaluation
    configs.eval_viz_callback = ConfigDict()
    eval_viz_callback.viz_num_images = 100

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 15

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
