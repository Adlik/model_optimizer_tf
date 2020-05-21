# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler utility
"""
import os
import yaml


def get_config(path):
    """
    Get config from a path
    :param path: file path
    :return:
    """
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def config_get_epochs_to_train(config):
    """
    Get epochs from Config object
    :param config: Config object
    :return:
    """
    scheduler_config = get_scheduler(config)
    if scheduler_config is None:
        return 0, [], None
    return get_epochs_lr_to_train(scheduler_config)


def get_epochs_lr_to_train(scheduler_config):
    """
    Get epochs from scheduler dict
    :param scheduler_config: scheduler dict
    :return:
    """
    epochs = set()
    for item in scheduler_config['prune_schedulers']:
        epochs.update(item['epochs'])
    global_epochs = sorted(epochs)
    start_epoch = 0
    epochs_span = []
    for end_epoch in global_epochs:
        span_epoch = end_epoch - start_epoch
        epochs_span.append(span_epoch)
        start_epoch = end_epoch
    print('epochs_span: {}'.format(epochs_span))
    lr_schedulers = None
    if 'lr_schedulers' in scheduler_config:
        lr_schedulers = scheduler_config['lr_schedulers']
    return global_epochs, epochs_span, lr_schedulers


def get_scheduler(config):
    """
    Get scheduler dict from config
    :param config: Config object
    :return:
    """
    base_dir = os.path.dirname(__file__)
    dataset = config.get_attribute('dataset')
    model = config.get_attribute('model_name')
    scheduler = config.get_attribute('scheduler')
    file_name = config.get_attribute('scheduler_file_name')
    if scheduler == 'train':
        return None
    if file_name is not None:
        return get_config(os.path.join(base_dir, scheduler, file_name))
    else:
        return get_config(os.path.join(base_dir, scheduler, model+'_'+dataset+'.yaml'))
