# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Resnet-50 on imagenet Learner definition
"""
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from .learner_base import LearnerBase


class Learner(LearnerBase):
    """
    Resnet-50 on imagenet Learner
    """
    def __init__(self, config):
        super().__init__(config)
        self.callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),
            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            hvd.callbacks.LearningRateWarmupCallback(self.learning_rate*hvd.size(), warmup_epochs=5, verbose=0),
            # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
            hvd.callbacks.LearningRateScheduleCallback(self.learning_rate*hvd.size(), start_epoch=5, end_epoch=30,
                                                       multiplier=1.),
            hvd.callbacks.LearningRateScheduleCallback(self.learning_rate*hvd.size(), start_epoch=30, end_epoch=60,
                                                       multiplier=1e-1),
            hvd.callbacks.LearningRateScheduleCallback(self.learning_rate*hvd.size(), start_epoch=60, end_epoch=80,
                                                       multiplier=1e-2),
            hvd.callbacks.LearningRateScheduleCallback(self.learning_rate*hvd.size(), start_epoch=80, multiplier=1e-3),
        ]
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path,
                                                                                  './checkpoint-{epoch}.h5'),
                                                                     period=self.checkpoint_save_period))

    def get_optimizer(self):
        """
        Model compile optimizer
        :return: Return model compile optimizer
        """
        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate*hvd.size(), momentum=0.9)
        opt = hvd.DistributedOptimizer(opt)
        return opt

    def get_losses(self, is_training=True):
        """
        Model compile losses
        :param is_training: is training or not
        :return: Return model compile losses
        """
        softmax_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        if (self.config.get_attribute('scheduler') == 'distill' or self.config.get_attribute('is_distill'))\
                and is_training:
            return None
        else:
            return softmax_loss

    def get_metrics(self, is_training=True):
        """
        Model compile metrics
        :param is_training: is training or not
        :return: Return model compile metrics
        """
        if (self.config.get_attribute('scheduler') == 'distill' or self.config.get_attribute('is_distill')) \
                and is_training:
            return None
        return ['sparse_categorical_accuracy']
