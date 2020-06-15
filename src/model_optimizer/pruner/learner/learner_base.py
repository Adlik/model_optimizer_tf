# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LearnerBase class
"""
import abc
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from ..dataset import get_dataset
from ..models import get_model
from .utils import get_call_backs
from ...stat import print_keras_model_summary, print_keras_model_params_flops


class LearnerBase(metaclass=abc.ABCMeta):
    """
    LearnerBase class
    """
    def __init__(self, config):
        self.config = config
        self.checkpoint_path = config.get_attribute('checkpoint_path')
        self.epochs = config.get_attribute('epochs')
        self.checkpoint_save_period = config.get_attribute('checkpoint_save_period')
        self.checkpoint_format = 'checkpoint-{epoch}.h5'
        self.learning_rate = config.get_attribute('learning_rate')
        self.models_train = []
        self.models_eval = []
        self.train_steps_per_epoch = 1
        self.eval_steps_per_epoch = 1
        self.resume_from_epoch = 0
        self.verbose = 1
        self.cur_epoch = 0
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        self.verbose = 1 if hvd.rank() == 0 else 0
        origin_train_model = get_model(config, is_training=True)
        origin_eval_model = get_model(config, is_training=False)
        self.models_train.append(origin_train_model)
        self.models_eval.append(origin_eval_model)
        train_model = tf.keras.models.clone_model(origin_train_model)
        eval_model = tf.keras.models.clone_model(origin_eval_model)
        self.models_train.append(train_model)
        self.models_eval.append(eval_model)
        self.train_dataset, self.eval_dataset = self.build_dataset()
        self.build_train()
        self.build_eval()
        self.load_model()
        self.save_model_path = config.get_attribute('checkpoint_eval_path')
        self.callbacks = []

    @property
    def resume_epoch(self):
        """
        Resume epoch property
        :return: Return resume epoch
        """
        return self.resume_from_epoch

    @abc.abstractmethod
    def get_losses(self):
        """
        Model compile losses
        :return: Return model compile losses
        """
        pass

    @abc.abstractmethod
    def get_optimizer(self):
        """
        Model compile optimizer
        :return: Return model compile optimizer
        """
        pass

    @abc.abstractmethod
    def get_metrics(self):
        """
        Model compile metrics
        :return: Return model compile metrics
        """
        pass

    def build_dataset(self):
        """
        Dataset for train or evaluate
        :return: Return dataset for train or eval
        """
        ds_train = get_dataset(self.config, is_training=True, num_shards=hvd.size(), shard_index=hvd.rank())
        self.train_steps_per_epoch = ds_train.steps_per_epoch
        self.train_steps_per_epoch = self.train_steps_per_epoch // hvd.size()
        train_dataset = ds_train.build()
        ds_eval = get_dataset(self.config, is_training=False)
        self.eval_steps_per_epoch = ds_eval.steps_per_epoch
        eval_dataset = ds_eval.build()
        return train_dataset, eval_dataset

    def build_train(self):
        """
        Model compile for train model
        :return:
        """
        loss = self.get_losses()
        optimizer = self.get_optimizer()
        metrics = self.get_metrics()
        train_model = self.models_train[-1]
        train_model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics,
                            experimental_run_tf_function=False)

    def build_eval(self):
        """
        Model compile for eval model
        :return:
        """
        loss = self.get_losses()
        optimizer = self.get_optimizer()
        metrics = self.get_metrics()
        eval_model = self.models_eval[-1]
        eval_model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics,
                           experimental_run_tf_function=False)

    def train(self, initial_epoch=0, epochs=1, lr_schedulers=None):
        """
        Model train process
        :return:
        """
        train_model = self.models_train[-1]
        if lr_schedulers is not None:
            self.callbacks.clear()
            self.callbacks.extend(get_call_backs(lr_schedulers))
            if hvd.rank() == 0:
                self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path,
                                                                                      './checkpoint-{epoch}.h5'),
                                                                         period=self.checkpoint_save_period))
        train_model.fit(self.train_dataset, initial_epoch=initial_epoch, steps_per_epoch=self.train_steps_per_epoch,
                        epochs=epochs, verbose=self.verbose, callbacks=self.callbacks)
        self.cur_epoch += epochs-initial_epoch

    def eval(self):
        """
        Model eval process, only evaluate on rank 0
        :return:
        """
        if hvd.rank() != 0:
            return
        eval_model = self.models_eval[-1]
        score = eval_model.evaluate(self.eval_dataset, steps=self.eval_steps_per_epoch)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def get_latest_train_model(self):
        """
        Get latest train model
        :return: Return latest train model
        """
        return self.models_train[-1]

    def get_latest_eval_model(self):
        """
        Get latest eval model
        :return: Return latest eval model
        """
        return self.models_eval[-1]

    def get_original_train_model(self):
        """
        Get original train model
        :return: Return original train model
        """
        return self.models_train[0]

    def get_original_eval_model(self):
        """
        Get original eval model
        :return: Return original eval model
        """
        return self.models_eval[0]

    def train_models_update(self, new_model):
        """
        Update last train model
        :return:
        """
        old_model = self.models_train.pop()
        del old_model
        self.models_train.append(new_model)

    def eval_models_update(self, new_model):
        """
        Update last eval model
        :return:
        """
        old_model = self.models_eval.pop()
        del old_model
        self.models_eval.append(new_model)

    def load_model(self):
        """
        Load checkpoint and update cur_epoch resume_from_epoch train_model
        :return:
        """
        self.resume_from_epoch = 0
        for try_epoch in range(self.epochs, 0, -1):
            if os.path.exists(os.path.join(self.checkpoint_path, self.checkpoint_format.format(epoch=try_epoch))):
                self.resume_from_epoch = try_epoch
                break
        if self.resume_from_epoch > 0:
            self.cur_epoch = self.resume_from_epoch
            model = tf.keras.models.load_model(
                os.path.join(self.checkpoint_path,
                             self.checkpoint_format.format(epoch=self.resume_from_epoch)))
            self.train_models_update(model)

    def save_eval_model(self):
        """
        Save evaluate model
        :return:
        """
        if hvd.rank() != 0:
            return
        train_model = self.models_train[-1]
        eval_model = self.models_eval[-1]
        clone_model = tf.keras.models.clone_model(eval_model)
        for i, layer in enumerate(clone_model.layers):
            if 'Conv2D' in str(type(layer)):
                clone_model.layers[i].filters = train_model.get_layer(layer.name).filters
            elif 'Dense' in str(type(layer)):
                clone_model.layers[i].units = train_model.get_layer(layer.name).units
        pruned_eval_model = tf.keras.models.model_from_json(clone_model.to_json())
        pruned_eval_model.set_weights(train_model.get_weights())
        save_model_path = os.path.join(self.save_model_path, 'checkpoint-') + str(self.cur_epoch)+'.h5'
        pruned_eval_model.save(save_model_path)
        self.eval_models_update(pruned_eval_model)

    def print_model_summary(self):
        """
        Print model summary
        :return:
        """
        train_model = self.models_train[-1]
        print_keras_model_summary(train_model, hvd.rank())

    def print_model_params_flops(self):
        """
        Print model params and flops
        :return:
        """
        train_model = self.models_train[-1]
        print_keras_model_params_flops(train_model, hvd.rank())
