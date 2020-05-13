# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Get pruner
"""
from ..scheduler.common import get_scheduler
from .pruner import AutoPruner
from .pruner import SpecifiedLayersPruner


def get_pruner(config, epoch):
    """
    Get pruner list according epoch
    :param config: Config object
    :param epoch: epoch
    :return: pruner list
    """
    scheduler_config = get_scheduler(config)
    pruner_list = []
    pruners = {
        'auto_prune': AutoPruner,
        'specified_layer_prune': SpecifiedLayersPruner
    }
    for scheduler in scheduler_config['prune_schedulers']:
        if epoch in scheduler['epochs']:
            func_name = scheduler['pruner']['func_name']
            pruner_type = scheduler_config['pruners'][func_name]['prune_type']
            if pruner_type in pruners:
                pruner_list.append(pruners[pruner_type](scheduler_config['pruners'][func_name]))
    return pruner_list

