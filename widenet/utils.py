import tensorflow as tf
import tensorflow.keras as keras
import math

from model_configs import PatchEncoder, Switch, Router


def load_model(path, compile=False, remove_last_n_layers=0):
    '''
    Load a saved model from disk.

    Parameters:

    `path`: path for the saved model.
    
    `compile`: Boolean, whether to compile the model after loading. Default to False.

    `remove_last_n_layers`: integer indicating the number of last layers to remove, usually set to remove prediction heads for fine-tuning.
    '''
    
    loaded_model = keras.models.load_model(path, compile=compile, custom_objects={'PatchEncoder': PatchEncoder, 'Switch': Switch, 'Router': Router})
    if remove_last_n_layers == 0:
        return loaded_model
    else:
        model = keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[-remove_last_n_layers-1].output)
        return model

def get_custom_cos_scheduler(g_base_lr, g_warmup_epochs=1500, g_epochs=10000, g_min_lr=1e-5, g_hold_on_epochs=1.0):
    def custom_cos_scheduler(epoch, lr):
        base_lr = tf.cast(g_base_lr, tf.float32)
        min_lr = tf.cast(g_min_lr, tf.float32)
        warmup_epochs = tf.cast(g_warmup_epochs, tf.float32)
        train_epochs = tf.cast(g_epochs, tf.float32)
        hold_on_epochs = tf.cast(g_hold_on_epochs, tf.float32)

        def warmup():
            temp = tf.cond(
                tf.math.greater_equal(warmup_epochs, epoch),
                lambda: (base_lr-min_lr) * tf.cast(epoch/warmup_epochs, tf.float32) + min_lr,
                lambda: base_lr
            )
            return temp

        def cosdecay():
            temp = tf.cast(1 + tf.math.cos(tf.constant(math.pi) * (epoch-warmup_epochs*hold_on_epochs) / (train_epochs-warmup_epochs*hold_on_epochs)), tf.float32)

            return tf.cast(0.5*base_lr, tf.float32) * temp

        result = tf.cond(
            tf.math.greater_equal(warmup_epochs*hold_on_epochs, epoch),
            lambda: warmup(),
            lambda: cosdecay()
        )

        return result
    
    return custom_cos_scheduler
    


