import tensorflow as tf
from .dataset_base import DatasetBase
import os


class Cifar10Dataset(DatasetBase):
    """
    Cifar10 dataset.
    """

    def __init__(self, config, is_training):
        """
        Constructor function.
        :param config: Config object
        :param is_training: whether to construct the training subset
        :return:
        """
        super(Cifar10Dataset, self).__init__(config, is_training)
        if is_training:
            self.file_pattern = os.path.join(self.data_dir, 'train.tfrecords')
            self.batch_size = self.batch_size
        else:
            self.file_pattern = os.path.join(self.data_dir, 'test.tfrecords')
            self.batch_size = self.batch_size_eval
        self.dataset_fn = tf.data.TFRecordDataset
        self.buffer_size = 10000
        # self.parse_fn = lambda x: parse_fn(x)
        self.num_samples_of_train = 60000
        self.num_samples_of_val = 10000

    @property
    def train_steps_per_epoch(self):
        return int(self.num_samples_of_train/self.batch_size)

    @property
    def val_steps_per_epoch(self):
        return int(self.num_samples_of_train / self.batch_size_val)

    def parse_fn(self, example_serialized):
        """
        Parse features from the serialized data
        :param example_serialized: serialized data
        :return: image,label
        """
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        features = tf.io.parse_single_example(example_serialized, feature_description)
        image = tf.io.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, dtype='float32') / 255.0
        label = tf.one_hot(tf.cast(features['label'], dtype=tf.int32), 10)
        # label = tf.cast(features['label'], dtype=tf.int32)
        return tf.reshape(image, [32, 32, 3]), label



