import argparse
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import os
from tensorflow.keras import layers
import math
from augment import RandAugment
from utils import load_model, get_custom_cos_scheduler


parser = argparse.ArgumentParser(description='Tensorflow ViT(MoE) training on Imagenet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Training settings
parser.add_argument('--log_dir', default='gs://fuzhao/log/', help='log file saved place')
parser.add_argument('--checkpoint-dir', default='./tmp/checkpoint', help='checkpoint file saved place')
parser.add_argument('--data_dir', default='gs://imagenet2012_eu/imagenet-2012-tfrecord', help='data dir')
parser.add_argument('--tpu', default='', help='TPU name')
parser.add_argument('--data_set', choices=['Imagenet', 'Cifar10', 'Cifar100'], default='Imagenet')
parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU instead of TPU')
parser.add_argument('--initial_epoch', type=int, default=-1, help='Select the checkpoint that has been trained for this many epoches to resume training from. Defaults to -1, in which case the lastest checkpoint will be used.')
# Training details
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=3e-3, help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5, help='number of warmup epochs')
parser.add_argument('--hold_on_epochs', type=float, default=1, help='learning rate (default: 0.01)')
parser.add_argument("--mixup", type=float, default=0.5, help='MixUp Augumentation parameter')
parser.add_argument('--eval_every', type=int, default=10, help='evaluation frequency ')
parser.add_argument('--save_freq', type=int, default=10, help='saves the model at end of this many epochs')
parser.add_argument('--batch-size', type=int, default=32, help='GLOBAL input batch size for training')
parser.add_argument("--inception_style", action='store_true', default=False, help='use Inception-style preprocessing')
parser.add_argument('--model_save_path', help='Path for saving the trained model')

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)





#@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = 224 * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = 224 * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=224, dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=224, dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, 224)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, 224)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, 224)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, 224)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


#@tf.function
def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, 224, 224
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, 224, 224
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (224 * 224)
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label


#def mix_up(ds, alpha=0.2):
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


def prepare_img(image, label, num_classes):
    if args.data_set == 'Imagenet':
        image = tf.image.resize(image, [224, 224])
    elif args.data_set == 'Cifar10':
        image = tf.image.resize(image, [32, 32])
    label = tf.squeeze(label)
    label = tf.one_hot(label, num_classes)
    return image, label
    #return tf.cast((tf.cast(image, tf.float32)- 127.5)/ 127.5, tf.float32), label
    #if args.inception_style:
    #    return tf.cast((tf.cast(image, tf.float32)- 127.5)/ 127.5, tf.float32), label
    #else:
    #    return tf.cast(image, tf.float32)/255.0, label


def get_dataset(batch_size, dataset, is_training=True,inception_style=False,use_randaug=True):

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
                    use_image_if_no_bounding_boxes=True)
                im = tf.slice(im, begin, size)
                # Unfortunately, the above operation loses the depth-dimension. So we
                # need to restore it the manual way.
                im.set_shape([None, None, 3])
                if args.data_set == 'Imagenet':
                    im = tf.image.resize(im, [224, 224])
                else:
                    im = tf.image.resize(im, [32, 32])
                if tf.random.uniform(shape=[]) > 0.5:
                    im = tf.image.flip_left_right(im)
                return tf.cast((tf.cast(im, tf.float32)- 127.5)/ 127.5, tf.float32),y
            
            if use_randaug:
                randaug = RandAugment(num_layers=2, magnitude=15)
                dataset = dataset.map(lambda x, y: (randaug.distort(x), y),
                                          num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                
            dataset = dataset.map(
                lambda im, y: _pp(im,y), num_parallel_calls=tf.data.AUTOTUNE
            )
            
            
                
            
            #train_ds_mu = dataset.shuffle(50000).batch(batch_size)
            train_ds_one = dataset.shuffle(50000).batch(batch_size)
            train_ds_two = dataset.shuffle(50000).batch(batch_size)

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
            train_ds_mu = train_ds.map(
                lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=args.mixup),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            train_ds_out = train_ds_mu.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return train_ds_out
        
        else:
            
        
            #dataset = dataset.repeat(args.epochs+1)
            #for the whole training
            dataset = dataset.repeat(args.epochs+1)

            #train_ds_mu = dataset.shuffle(50000).batch(batch_size)
            #dataset = dataset.repeat(300)
            train_ds_one = dataset.shuffle(50000).batch(batch_size)
            train_ds_two = dataset.shuffle(50000).batch(batch_size)

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
            train_ds_mu = train_ds.map(
                lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=args.mixup),
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
            return train_ds_out

    else:
        if inception_style:
            def _pp_val(im,y):
                #im = decoder(data['image'])
                if args.data_set == 'Imagenet':
                    im = tf.image.resize(im, [224, 224])
                else:
                    im = tf.image.resize(im, [32, 32])
                return tf.cast((tf.cast(im, tf.float32)- 127.5)/ 127.5, tf.float32),y
            dataset = dataset.map(
                lambda im, y: _pp_val(im,y), num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.batch(batch_size)
            val_ds_mu = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
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
            return val_ds_mu


def dataset_prepare():
    #ar
    if args.data_set == 'Imagenet':
        num_classes = 1000


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
        
        cifar_train, cifar_test = tfds.load(name='cifar10', split=['train', 'test'],
                                            data_dir=args.data_dir,
                                            shuffle_files=True, as_supervised=True)
        train_dataset = cifar_train.map(
            lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset = cifar_test.map(
            lambda image, label: prepare_img(image, label, num_classes), num_parallel_calls=tf.data.AUTOTUNE
        )
        
        
    
    
    return train_dataset,test_dataset

if __name__ == '__main__':
    print('tensorflow version:', tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #tf.enable_v2_behavior()
    args = parser.parse_args()
#     config = CONFIGS[args.model_type]
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


    # Prepare Data
    if args.data_set == 'Imagenet':
        num_classes = 1000
    else:
        num_classes = 10
        # imagenet_train, imagenet_val = tfds.load(name='imagenet2012', split=['train', 'validation'],
        #                                          data_dir=args.data_dir,
        #                                          shuffle_files=True, as_supervised=True)

    train_dataset, test_dataset = dataset_prepare()
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
    steps_per_epoch = train_len // args.batch_size + 1
    steps_per_eval = test_len // args.batch_size 

    print(f"train_length:{train_len}-test_length:{test_len}")
    print(f"x_shape: {x_shape} - y_shape: {y_shape}")
    print('Number of classes:', num_classes)
    print('Steps per epoch:', steps_per_epoch)
    print('Data loading finish')
    

    train_dist_dataset = get_dataset(args.batch_size, train_dataset, is_training=True, 
                                    inception_style=args.inception_style)
    test_dist_dataset = get_dataset(args.batch_size, test_dataset, is_training=False,
                                        inception_style=args.inception_style)
    per_replica_batch_size = args.batch_size // strategy.num_replicas_in_sync
    print('Batch size allocated to each core:', per_replica_batch_size)

    with strategy.scope():
        if args.data_set == 'Imagenet':
            input_shape = (224, 224, 3)
        elif args.data_set == 'Cifar10':
            input_shape = (32, 32, 3)
            
        # load model
        if args.initial_epoch == -1:
            all_checkpoints = os.listdir(args.checkpoint_dir)
            all_checkpoints.sort()
            checkpoint_path = os.path.join(args.checkpoint_dir, all_checkpoints[-1])
#             print(checkpoint_path)
#             print(type(checkpoint_path))
            initial_epoch = int(checkpoint_path.split('_')[-1])
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'ckpt_{args.initial_epoch}')
            initial_epoch = args.initial_epoch
        
        model = load_model(path=checkpoint_path, compile=True)
        model.summary()
    
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt_{epoch:03d}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_freq=args.save_freq * steps_per_epoch
    )
        
    lr_scheduler = get_custom_cos_scheduler(
        g_base_lr=args.base_lr,
        g_warmup_epochs=args.warmup_epochs,
        g_epochs=args.epochs,
        g_min_lr=1e-5,
        g_hold_on_epochs=args.hold_on_epochs
    )
    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        
    callbacks = [
        lr_schedule_callback,
        model_checkpoint_callback            
    ]
    
    model.fit(
        train_dist_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dist_dataset,
        validation_steps=steps_per_eval,
        validation_freq=args.eval_every,
        callbacks=callbacks,
        verbose=2,
        initial_epoch=initial_epoch
    )
    model.save(args.model_save_path)
