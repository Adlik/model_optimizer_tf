# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CNN1d on iscx_session_all learner definition
"""
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from .learner_base import LearnerBase
from .utils import cosine_multiplier


class Learner(LearnerBase):
    """
    CNN1d on iscx_session_all learner
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
            hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=0),
            hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=50, multiplier=1.),
            hvd.callbacks.LearningRateScheduleCallback(
                start_epoch=50, multiplier=lambda epoch: cosine_multiplier(epoch, total_epoch=self.epochs))

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
        opt = tf.keras.optimizers.SGD(self.learning_rate*hvd.size(), momentum=0.9)
        opt = hvd.DistributedOptimizer(opt)
        return opt

    def get_losses(self, is_training=True):
        """
        Model compile losses
        :param is_training: is training or not
        :return: Return model compile losses
        """
        return 'sparse_categorical_crossentropy'

    def get_metrics(self, is_training=True):
        """
        Model compile metrics
        :param is_training: is training or not
        :return: Return model compile metrics
        """
        if (self.config.get_attribute('scheduler') == 'distill' or self.config.get_attribute('is_distill', False)) \
                and is_training:
            return None
        return ['sparse_categorical_accuracy']
