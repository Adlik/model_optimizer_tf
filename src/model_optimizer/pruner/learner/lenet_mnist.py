# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .learner_base import LearnerBase
import tensorflow as tf
import os
import horovod.tensorflow.keras as hvd


class Learner(LearnerBase):
    def __init__(self, config):
        super(Learner, self).__init__(config)
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
        ]
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path,
                                                                     './checkpoint-{epoch}.h5'),
                                                                     period=self.checkpoint_save_period))

    def get_optimizer(self):
        opt = tf.optimizers.Adam(self.learning_rate*hvd.size())
        opt = hvd.DistributedOptimizer(opt)
        return opt

    def get_losses(self):
        return 'sparse_categorical_crossentropy'

    def get_metrics(self):
            return ['sparse_categorical_accuracy']
