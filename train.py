import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger

from coconet import callbacks

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our object detection model.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool(
    "log_eval", False, "Log model prediction, needs --wandb argument as well."
)


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    CALLBACKS = []
    sync_tensorboard = None
    if config.callback_config.use_tensorboard:
        sync_tensorboard = True
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            job_type="train",
            config=config.to_dict(),
            sync_tensorboard=sync_tensorboard
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [WandbMetricsLogger()]

    # Prepare Dataloader

    # Initialize callbacks
    callback_config = config.callback_config
    # Builtin early stopping callback
    if callback_config.use_earlystopping:
        earlystopper = callbacks.get_earlystopper(config)
        CALLBACKS += [earlystopper]
    # Built in callback to reduce learning rate on plateau
    if callback_config.use_reduce_lr_on_plateau:
        reduce_lr_on_plateau = callbacks.get_reduce_lr_on_plateau(config)
        CALLBACKS += [reduce_lr_on_plateau]

    # Initialize Model checkpointing callback
    if FLAGS.log_model:
        # Custom W&B model checkpoint callback
        model_checkpointer = callbacks.get_model_checkpoint_callback(config)
        CALLBACKS += [model_checkpointer]

    # Custom W&B model prediction visualization callback
    if wandb.run is not None:
        if FLAGS.log_eval:
            model_pred_viz = callbacks.get_evaluation_callback(config)
            CALLBACKS += [model_pred_viz]

    if config.callback_config.use_tensorboard:
        CALLBACKS += [tf.keras.callbacks.TensorBoard()]

    # TODO (ayulockin): Remove the print statement
    print(CALLBACKS)

    # Initialize model
    
    
    # Compile the model
    

    # Train the model
    

if __name__ == "__main__":
    app.run(main)
