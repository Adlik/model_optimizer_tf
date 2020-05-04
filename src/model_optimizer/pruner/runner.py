# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .core import get_pruner
from .scheduler.common import config_get_epochs_to_train
from .learner import get_learner


def _prune(config, epoch, learner):
    prune_list = get_pruner(config, epoch)
    learner.print_model_summary()
    for pruner in prune_list:
        model = learner.get_latest_train_model()
        new_model = pruner.prune(model)
        learner.train_models_update(new_model)
        learner.print_model_summary()


def run_scheduler(config):
    learner = get_learner(config)
    global_epochs, epochs_span, lr_schedulers = config_get_epochs_to_train(config)
    cur_epoch = 0
    ln_cur_epoch = learner.cur_epoch
    if ln_cur_epoch > 0 and len(epochs_span) != 0:
        _prune(config, ln_cur_epoch, learner)
    for epoch_span in epochs_span:
        next_epoch = cur_epoch+epoch_span
        initial_epoch = cur_epoch
        if ln_cur_epoch > 0:
            if next_epoch <= ln_cur_epoch:
                cur_epoch += epoch_span
                continue
            else:
                initial_epoch = ln_cur_epoch
                ln_cur_epoch = 0
        learner.build_train()
        learner.train(initial_epoch=initial_epoch, epochs=cur_epoch+epoch_span, lr_schedulers=lr_schedulers)
        cur_epoch += epoch_span
        _prune(config, cur_epoch, learner)
    if cur_epoch < ln_cur_epoch:
        cur_epoch = ln_cur_epoch
    target_epoch = config.get_attribute('epochs')
    if cur_epoch < target_epoch:
        learner.build_train()
        learner.train(initial_epoch=cur_epoch, epochs=target_epoch, lr_schedulers=lr_schedulers)
        learner.save_eval_model()
        learner.build_eval()
        learner.eval()

