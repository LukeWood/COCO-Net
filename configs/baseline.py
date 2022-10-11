import os
import ml_collections
from ml_collections import ConfigDict


def get_wandb_configs() -> ConfigDict:
    configs = ConfigDict()
    configs.project = "coconet"

    return configs


def get_dataset_configs() -> ConfigDict:
    configs = ConfigDict()

    return configs


def get_model_configs() -> ConfigDict:
    configs = ConfigDict()
    configs.model_name = "RetinaNet"

    return configs


def get_callback_configs() -> ConfigDict:
    configs = ConfigDict()

    # Early stopping
    configs.early_stopper = ConfigDict()
    configs.use_earlystopping = True
    configs.early_stopper.monitor = "val_loss"
    configs.early_stopper.early_patience = 6
    configs.early_stopper.restore_best_weights = True

    # Reduce LR on plateau
    configs.reduce_lr_on_plateau = ConfigDict()
    configs.use_reduce_lr_on_plateau = True
    configs.reduce_lr_on_plateau.monitor = "val_loss"
    configs.reduce_lr_on_plateau.factor = 0.2
    configs.reduce_lr_on_plateau.patience = 3

    # Model checkpointing
    configs.ckpt_callback = ConfigDict()
    configs.ckpt_callback.checkpoint_filepath = "wandb/model_{epoch}"
    configs.ckpt_callback.monitor = "val_loss"
    configs.ckpt_callback.save_best_only = True
    configs.ckpt_callback.save_weights_only = False
    configs.ckpt_callback.initial_value_threshold = 0.0

    # Model evaluation
    configs.eval_viz_callback = ConfigDict()
    configs.eval_viz_callback.viz_num_images = 100

    # Use TensorBoard synced to W&B
    configs.use_tensorboard = True

    return configs


def get_train_configs() -> ConfigDict:
    configs = ConfigDict()
    configs.epochs = 15

    return configs


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.seed = 1234
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
