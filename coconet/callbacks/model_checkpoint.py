import tensorflow as tf

import wandb
import wandb.keras import WandbModelCheckpoint


def get_model_checkpoint_callback(args):
    if wandb.run is not None:
        model_checkpointer = WandbModelCheckpoint(
            filepath=args.callback_config.checkpoint_filepath,
            monitor=args.callback_config.monitor,
            save_best_only=args.callback_config.save_best_only,
            save_weights_only=args.callback_config.save_weights_only,
            initial_value_threshold=args.callback_config.initial_value_threshold,
        )
    else:
        model_checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.callback_config.checkpoint_filepath,
            monitor=args.callback_config.monitor,
            save_best_only=args.callback_config.save_best_only,
            save_weights_only=args.callback_config.save_weights_only,,
            initial_value_threshold=args.callback_config.initial_value_threshold,
        )

    return model_checkpointer
