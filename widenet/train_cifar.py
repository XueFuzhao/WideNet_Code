import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import argparse
from modeling import CONFIGS, create_vit_classifier, create_vit_moe_classifier

# Training settings
parser = argparse.ArgumentParser(description='Tensorflow ViT(MoE) training on cifar',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-dir', default='./tmp/checkpoint',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=3e-3,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.03,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# model details
parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--use_moe", action='store_true', default=False,
                    help='use ViT-MoE model')
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                             "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "ViT-MoE-B"],
                    default="ViT-B_16",
                    help="Which model to use.")


def run_experiment(model, args):
    '''
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    '''

    optimizer = tfa.optimizers.AdamW(
        learning_rate=args.base_lr,
        weight_decay=args.wd
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ]
    )

    checkpoint_filepath = args.checkpoint_dir
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


if __name__ == '__main__':
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    args = parser.parse_args()
    config = CONFIGS[args.model_type]

    # Prepare data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    # Prepare model
    input_shape = (32, 32, 3)
    num_classes = 10
    if args.use_moe:
        model = create_vit_moe_classifier(config, input_shape, img_size= args.img_size, num_classes= num_classes)
    else:
        model = create_vit_classifier(config, input_shape, img_size=args.img_size, num_classes=num_classes)
    model.summary()

    # Train
    #history = run_experiment(model, args)


