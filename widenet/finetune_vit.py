import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import os
import datetime
from tensorflow.keras import layers
import math
from augment import RandAugment
import argparse
import pickle
from modeling import CONFIGS, create_vit_classifier, create_vit_moe_classifier

from utils import load_model


# Training settings
parser = argparse.ArgumentParser(description='Tensorflow ViT(MoE) training on Imagenet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_dir', default='gs://fuzhao/log/',
                    help='log file saved place')
parser.add_argument('--checkpoint-dir', default='./tmp/checkpoint',
                    help='checkpoint file saved place')
parser.add_argument('--tpu', default='',
                    help='TPU name')
parser.add_argument('--data_set', choices=['Imagenet', 'Cifar10', 'Cifar100'],
                    default='Imagenet')
parser.add_argument('--data_dir', default='gs://imagenet2012_eu/imagenet-2012-tfrecord',
                    help='data dir')
parser.add_argument('--pretrained_model_path', default='gs://fuzhao/vit_large',
                    help='data dir')


parser.add_argument('--model_save_path', help='Path for saving the trained model')

parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU instead of TPU')


# Training details
parser.add_argument('--training_option', choices=['Custom', 'Keras'], default='Keras')

parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='GLOBAL input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--eval_every', type=int, default=10,
                    help='evaluation frequency ')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=3e-3,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--decay-steps', type=int, default=100000,
                    help='number of learning rate decay steps')
parser.add_argument('--wd', type=float, default=0.03,
                    help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='label smoothing')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# model details
parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--use_moe", action='store_true', default=False,
                    help='use ViT-MoE model')
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                             "ViT-L_32", "ViT-H_14", "ViT-XH_14", "R50-ViT-B_16", 
                                             "ViT-MoE-B_16", "ViT-MoE-L_16", "ViT-MoE-H_14",
                                            "ViT-MoE-XH_14"],
                    default="ViT-B_16",
                    help="Which model to use.")
#MOE options
parser.add_argument("--num_experts", type=int, default=8,
                    help='num of experts for Token Mixture Layers')
parser.add_argument("--num_masked_experts", type=float, default=0.0,
                    help='num of experts masked')
parser.add_argument("--capacity_factor", type=float, default=1.0,
                    help='capacity factor of mixer of experts')
parser.add_argument('--top_k', type=int, default=1,
                    help='top_k experts are selected')
parser.add_argument("--use_aux_loss", action='store_true', default=False,
                    help='do not use balanced loss')
parser.add_argument("--aux_loss_alpha", type=float, default=1.0,
                    help='do not use balanced loss')
parser.add_argument("--aux_loss_beta", type=float, default=0.001,
                    help='do not use balanced loss')

parser.add_argument('--switch_deepth', type=int, default=1,
                    help='number of layers used when one time switch done')


parser.add_argument("--mixup", type=float, default=0.5,
                    help='MixUp Augumentation parameter')
parser.add_argument("--beta2", type=float, default=0.999,
                    help='beta2 value of optimizer')
parser.add_argument("--opt", choices=["LAMB", "Adam"],
                    default="Adam",
                    help='Optimizer')
parser.add_argument("--inception_style", action='store_true', default=False,
                    help='use Inception-style preprocessing')
parser.add_argument('--hold_on_epochs', type=float, default=1,
                    help='learning rate (default: 0.01)')

parser.add_argument("--use_representation", action='store_true', default=False,
                    help='use use_representation before head')

parser.add_argument("--share_att", action='store_true', default=False,
                    help='share attention weights')

parser.add_argument("--share_ffn", action='store_true', default=False,
                    help='share attention weights')

parser.add_argument('--group_deepth', type=int, default=128,
                    help='number of layers used within one group')



def prepare_img(image, label, num_classes):
    image = tf.image.resize(image, [224, 224])
    label = tf.squeeze(label)
    label = tf.one_hot(label, num_classes)
    return image, label

#strategy = tf.distribute.MultiWorkerMirroredStrategy()
#print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))



def dataset_prepare():
    #ar
    if args.data_set == 'Imagenet':
        num_classes = 1001


        train_file_names = os.path.join(args.data_dir, '*-01024')
        val_file_names = os.path.join(args.data_dir, '*-00128')
        train_files = tf.data.Dataset.list_files(train_file_names)
        val_files = tf.data.Dataset.list_files(val_file_names)

        

        imagenet_raw_train = train_files.interleave(tf.data.TFRecordDataset,
                                                    num_parallel_calls=tf.data.AUTOTUNE,
                                                     deterministic=False)
        imagenet_raw_val = val_files.interleave(tf.data.TFRecordDataset,
                                                 num_parallel_calls=tf.data.AUTOTUNE,
                                                 deterministic=False)
        


        def parse_image(record):
            features = {
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)
            }
            parsed_record = tf.io.parse_single_example(record, features)
            image = tf.io.decode_jpeg(parsed_record['image/encoded'], channels=3)
            label = tf.cast(parsed_record['image/class/label'], tf.int32)
            return image, label

        imagenet_train = imagenet_raw_train.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        imagenet_val = imagenet_raw_val.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = imagenet_train.map(
            lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset = imagenet_val.map(
            lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE
        )


    elif args.data_set == 'Cifar10':

        num_classes = 10
        
        train_file_names = os.path.join(args.data_dir, 'cifar10-train*')
        val_file_names = os.path.join(args.data_dir, 'cifar10-test*')
        train_files = tf.data.Dataset.list_files(train_file_names)
        val_files = tf.data.Dataset.list_files(val_file_names)

        

        cifar10_raw_train = train_files.interleave(tf.data.TFRecordDataset,
                                                    num_parallel_calls=tf.data.AUTOTUNE,
                                                     deterministic=False)
        cifar10_raw_val = val_files.interleave(tf.data.TFRecordDataset,
                                                 num_parallel_calls=tf.data.AUTOTUNE,
                                                 deterministic=False)
        


        def parse_image(record):
            features = {
                'label': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([], tf.string)
            }
            parsed_record = tf.io.parse_single_example(record, features)
            image = tf.io.decode_jpeg(parsed_record['image'], channels=3)
            label = tf.cast(parsed_record['label'], tf.int32)
            return image, label

        cifar10_train = cifar10_raw_train.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        cifar10_val = cifar10_raw_val.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = cifar10_train.map(
            lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset = cifar10_val.map(
            lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE
        )
        
        
    
    
    return train_dataset,test_dataset


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    #ds_one, ds_two= ds
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)




def get_dataset(batch_size, dataset, is_training=True, inception_style=False, use_randaug=True):

    # Only shuffle and repeat the dataset in training. The advantage of having an
    # infinite dataset for training is to avoid the potential last partial batch
    # in each epoch, so that you don't need to think about scaling the gradients
    # based on the actual batch size.

    if is_training:

        if inception_style:

            dataset = dataset.repeat(args.epochs+1)

            def _pp(im,y):
                #im = decoder(data['image'])
                channels = im.shape[-1]
                begin, size, _ = tf.image.sample_distorted_bounding_box(
                    tf.shape(im),
                    tf.zeros([0, 0, 4], tf.float32),
                    area_range=(0.05, 1.0),
                    min_object_covered=0,  # Don't enforce a minimum area.
                    use_image_if_no_bounding_boxes=True
                )
                im = tf.slice(im, begin, size)
                # Unfortunately, the above operation loses the depth-dimension. So we
                # need to restore it the manual way.
                im.set_shape([None, None, 3])
                im = tf.image.resize(im, [224, 224])
                if tf.random.uniform(shape=[]) > 0.5:
                    im = tf.image.flip_left_right(im)
                return tf.cast((tf.cast(im, tf.float32)- 127.5)/ 127.5, tf.float32), y

            if use_randaug:
                randaug = RandAugment(num_layers=2, magnitude=15)
                dataset = dataset.map(lambda x, y: (randaug.distort(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

            dataset = dataset.map(
                lambda im, y: _pp(im,y), num_parallel_calls=tf.data.AUTOTUNE
            )

            #train_ds_mu = dataset.shuffle(50000).batch(batch_size)
            train_ds_one = dataset.shuffle(100).batch(batch_size)
            train_ds_two = dataset.shuffle(100).batch(batch_size)

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
            train_ds_mu = train_ds.map(
                lambda ds_one, ds_two: mix_up(ds_one, ds_two),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            train_ds_out = train_ds_mu.prefetch(buffer_size=tf.data.AUTOTUNE)

            return train_ds_out

        else:
            #dataset = dataset.repeat(args.epochs+1)
            #for the whole training
            dataset = dataset.repeat(num_epochs+1)

            #train_ds_mu = dataset.shuffle(50000).batch(batch_size)
            #dataset = dataset.repeat(300)
            train_ds_one = dataset.shuffle(100).batch(batch_size)
            train_ds_two = dataset.shuffle(100).batch(batch_size)

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
            train_ds_mu = train_ds.map(
                lambda ds_one, ds_two: mix_up(ds_one, ds_two),
                num_parallel_calls=tf.data.AUTOTUNE
            )  

            train_ds_mu = train_ds_mu.prefetch(buffer_size=tf.data.AUTOTUNE)

            random_augmentation = keras.Sequential(
                [
                    #layers.experimental.preprocessing.Normalization(mean=0,variance=1.), 
                    layers.experimental.preprocessing.Resizing(256, 256),
                    layers.experimental.preprocessing.RandomCrop(224, 224),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(factor=0.2),
                    #layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
                    layers.experimental.preprocessing.Normalization(
                        mean=[0.485, 0.456, 0.406],
                        variance=[0.052, 0.050, 0.050]
                    )

                ],
                name="random_augmentation",
            )


            train_ds_mu = train_ds_mu.map(lambda x, y: (random_augmentation(x, training=True), y),
                                          num_parallel_calls=tf.data.AUTOTUNE)

            #train_ds_mu = train_ds_mu.map(lambda x, y: (random_augmentation(x, training=True), y),
            #                              num_parallel_calls=tf.data.AUTOTUNE)
            train_ds_out = train_ds_mu.prefetch(buffer_size=tf.data.AUTOTUNE)
#             train_ds_out = train_ds_mu
            return train_ds_out

    else:
        if inception_style:
            def _pp_val(im,y):
                #im = decoder(data['image'])
                im = tf.image.resize(im, [224, 224])
                return tf.cast((tf.cast(im, tf.float32)- 127.5)/ 127.5, tf.float32),y
            dataset = dataset.map(
                lambda im, y: _pp_val(im,y), num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.batch(batch_size)
            val_ds_mu = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#             val_ds_mu = dataset
            return val_ds_mu

        else:
            dataset = dataset.batch(batch_size)
            #dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            random_augmentation_val = keras.Sequential(
                [
                    layers.experimental.preprocessing.Resizing(256, 256),
                    layers.experimental.preprocessing.CenterCrop(224,224),
                    layers.experimental.preprocessing.Normalization(
                        mean=[0.485, 0.456, 0.406],
                        variance=[0.052, 0.050, 0.050]
                    )
                ],
                name="random_augmentation_val",
            )

            dataset = dataset.map(lambda x_val, y_val: (random_augmentation_val(x_val, training=True), y_val),
                                          num_parallel_calls=tf.data.AUTOTUNE)
            '''normal_layer = layers.experimental.preprocessing.Normalization()
            def normal_fn(x):
                normal_layer.adapt(x)
                return normal_layer(x)
            dataset = dataset.map(lambda x, y: (normal_fn.adapt(x), y),num_parallel_calls=tf.data.AUTOTUNE)'''

            val_ds_mu = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#             val_ds_mu = dataset
            return val_ds_mu




if __name__ == '__main__':
    print('tensorflow version:', tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #tf.enable_v2_behavior()
    args = parser.parse_args()
    config = CONFIGS[args.model_type]
    print(args)
    #print(args.log_dir)
    
    if not args.use_gpu:
        # TPU Set Up
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        service_addr = resolver.get_master().replace(':8470', ':8466')
        #service_addr = resolver.get_master().replace(':8170', ':8166')
        print('SEVRVICE ADDRESS:', service_addr)

        print('tensorflow version:', tf.__version__)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        print('Number of TPU cores: {}'.format(strategy.num_replicas_in_sync))
    else:
        # GPU set-up
        strategy = tf.distribute.MirroredStrategy()
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))
    
    if args.data_set == 'Cifar10':
        num_classes = 10
    else:
        num_classes = 100

    '''with strategy.scope():

        num_classes = 10
        cifar_train, cifar_test = tfds.load(
            name='cifar10', 
            split=['train', 'test'], 
            data_dir='/users/hliu/weifutao/data', 
            shuffle_files=True, 
            as_supervised=True
        )'''



    #train_dataset = cifar_train.map(lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    #test_dataset = cifar_test.map(lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset,test_dataset=dataset_prepare()
    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()
    
    if args.data_set == 'Imagenet':
        train_len = 1281167
        test_len = 50000
    elif args.data_set == 'Cifar10':
        train_len = 50000
        test_len = 10000
    
    
    (x_sample, y_sample) = next(iter(test_dataset))
    x_shape = x_sample.shape
    y_shape = y_sample.shape
    steps_per_epoch = train_len // args.batch_size +1
    steps_per_eval = test_len // args.batch_size 
    
    print(f"train_length:{train_len}-test_length:{test_len}")
    print(f"x_shape: {x_shape} - y_shape: {y_shape}")
    print('Number of classes:', num_classes)
    print('Steps per epoch:', steps_per_epoch)
    print('Data loading finish')


    if args.training_option == 'Keras':

        train_dist_dataset = get_dataset(args.batch_size, train_dataset, is_training=True, 
                                         inception_style=args.inception_style)
        test_dist_dataset = get_dataset(args.batch_size, test_dataset, is_training=False,
                                        inception_style=args.inception_style)
        per_replica_batch_size = args.batch_size // strategy.num_replicas_in_sync
        print('Batch size allocated to each core:', per_replica_batch_size)

        with strategy.scope():
            #if args.data_set == 'Imagenet':
            #    input_shape = (224, 224, 3)
            #elif args.data_set == 'Cifar10':
            #    input_shape = (224, 224, 3)
            input_shape = (args.img_size, args.img_size, 3)

            #model = load_model(path='/users/hliu/weifutao/experiment_layer_norm/vit_base', remove_last_n_layers=3)
            model = load_model(path=args.pretrained_model_path, remove_last_n_layers=3)
        #     model.summary()
            inputs = tf.keras.layers.Input(shape=input_shape)
            features = model(inputs)
            if args.data_set == 'Cifar10':
                output_shape=10
            elif args.data_set == 'Cifar100':
                output_shape=100
            outputs = tf.keras.layers.Dense(output_shape)(features)
            new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            new_model.summary()
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.base_lr, decay_steps=steps_per_epoch*args.epochs),
                momentum=0.9,
                clipnorm=1
            )
            
            def get_lr_metrics(optimizer_f):
                def lr_here(y_true, y_pred):
                    return optimizer_f._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
                return lr_here
            
            lr_metrics = get_lr_metrics(optimizer)

            new_model.compile(
                optimizer=optimizer,
                steps_per_execution=steps_per_epoch,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', lr_metrics]
            )

        new_model.fit(
            train_dist_dataset,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_dist_dataset,
            validation_steps=steps_per_eval,
            validation_freq=args.eval_every,
            verbose=2
        )
