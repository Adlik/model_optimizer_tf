# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Learner callbacks utility
"""
import math
import horovod.tensorflow.keras as hvd


def get_call_backs(lr_schedulers, initial_lr):
    """
    Build learner callbacks from schedulers
    :param lr_schedulers: schedulers dict
    :return: callbacks list
    """
    callbacks = [
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
    for lr_func in lr_schedulers:
        if lr_func['class'] == 'LearningRateWarmupCallback':
            callbacks.append(hvd.callbacks.LearningRateWarmupCallback(initial_lr,
                                                                      warmup_epochs=lr_func['warmup_epochs'],
                                                                      verbose=lr_func['verbose']))
        elif lr_func['class'] == 'LearningRateScheduleCallback':
            end_epoch = None
            if 'end_epoch' in lr_func:
                end_epoch = lr_func['end_epoch']
            callbacks.append(hvd.callbacks.LearningRateScheduleCallback(initial_lr,
                                                                        start_epoch=lr_func['start_epoch'],
                                                                        end_epoch=end_epoch,
                                                                        multiplier=float(lr_func['multiplier'])))
    return callbacks


def cosine_multiplier(epoch, total_epoch=240):
    """
    Applies cosine decay to the learning rate.
    :param epoch: current epoch
    :param total_epoch: total epochs
    :return: decayed learning rate
    """
    if epoch >= total_epoch:
        return 0.5 + 0.5 * math.cos(math.pi * (total_epoch-1) / total_epoch)
    else:
        return 0.5 + 0.5 * math.cos(math.pi * epoch / total_epoch)
