"""
Tests for the model_optimizer.models.get_model method.
"""
import os
# If you did not execute the setup.py, uncomment the following four lines
from model_optimizer.pruner.config import create_config_from_obj as prune_conf_from_obj
from model_optimizer.pruner.models import get_model


def test_get_model_distill():
    """
    test get_model function for distillation
    """
    base_dir = os.path.dirname(__file__)
    request = {
        "dataset": "imagenet",
        "model_name": "resnet_50",
        "data_dir": "",
        "batch_size": 256,
        "batch_size_val": 100,
        "learning_rate": 0.1,
        "epochs": 90,
        "checkpoint_path": os.path.join(base_dir, "./models_ckpt/resnet_50_imagenet_distill"),
        "checkpoint_save_period": 5,  # save a checkpoint every 5 epoch
        "checkpoint_eval_path": os.path.join(base_dir, "./models_eval_ckpt/resnet_50_imagenet_distill"),
        "scheduler": "train",
        "scheduler_file_name": "resnet_50_imagenet_0.3.yaml",
        "classifier_activation": None  # None or "softmax", default is softmax
    }

    config = prune_conf_from_obj(request)
    train_model = get_model(config, (244, 244, 3), is_training=True)
    for layer in train_model.layers:
        if layer.name == "DistillLoss":
            assert False
            break
