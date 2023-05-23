# Copyright 2021 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distilling Loss Layer
"""
import tensorflow as tf


class DistillLossLayer(tf.keras.layers.Layer):
    """
    Layer to compute the loss for distillation.
    the total loss =  the student loss + the distillation loss

    Arguments:
        alpha: a float between [0.0, 1.0]. It corresponds to the importance between the student loss and the
        distillation loss.
        temperature: the temperature of distillation.  Defaults to 10.
        teacher_path: the model path of teacher. The format of the  model is h5.
        name: String, name to use for this layer. Defaults to 'DistillLoss'.

    Call arguments:
      inputs: inputs of the layer. It corresponds to [input, y_true, y_prediction]
    """
    def __init__(self, teacher_path, alpha=1.0, temperature=10, name="DistillLoss",
                 teacher_model_load_func=None, **kwargs):
        """
        :param teacher_path: the model path of teacher. The format of the  model is h5.
        :param alpha: a float between [0.0, 1.0]. It corresponds to the importance between the student loss and the
         distillation loss.
        :param temperature: the temperature of distillation.  Defaults to 10.
        :param name: String, name to use for this layer. Defaults to 'DistillLoss'.
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_path = teacher_path
        self.accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.teacher_model_load_func = teacher_model_load_func
        if self.teacher_model_load_func == "load_tf_ensemble_model":
            from .tf_model_loader import load_tf_ensemble_model
            self.teacher = load_tf_ensemble_model(self.teacher_path)
        else:
            self.teacher = tf.keras.models.load_model(self.teacher_path)

    # pylint: disable=unused-argument
    def call(self, inputs, **kwargs):
        """
        :param inputs: inputs of the layer. It corresponds to [input, y_true, y_prediction]
        :return: the total loss of the  distiller model
        """
        x, y_true, y_pred = inputs
        rtn_loss = None
        if y_true is not None:
            student_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
            self.teacher.trainable = False
            teacher_predictions = self.teacher(x)
            distillation_loss = tf.keras.losses.KLDivergence()(
                tf.nn.softmax(y_pred / self.temperature, axis=1),
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1)
            )
            stu_loss = self.alpha * student_loss
            dis_loss = (1 - self.alpha) * self.temperature * self.temperature * distillation_loss
            rtn_loss = stu_loss + dis_loss

            self.add_loss(rtn_loss)
            self.add_metric(student_loss, aggregation="mean", name="stu_loss")
            self.add_metric(dis_loss, aggregation="mean", name="dis_loss")

            self.add_metric(self.accuracy_fn(y_true, y_pred))
        return rtn_loss

    def get_config(self):
        """
        Implement get_config to enable serialization.
        """
        config = super().get_config()
        config.update({"teacher_path": self.teacher_path})
        config.update({"alpha": self.alpha})
        config.update({"temperature": self.temperature})
        config.update({"teacher_model_load_func": self.teacher_model_load_func})
        return config
