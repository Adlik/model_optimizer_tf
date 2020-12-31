import tensorflow as tf


class DistillLossLayer(tf.keras.layers.Layer):
    def __init__(self,alpha, temperature, teacher_path, name="DistillLoss", **kwargs):
        super(DistillLossLayer, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_path = teacher_path
        self.accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.teacher = tf.keras.models.load_model(self.teacher_path)

    def call(self, inputs, **kwargs):
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
        config = super(DistillLossLayer, self).get_config()
        config.update({"teacher_path": self.teacher_path})
        config.update({"alpha": self.alpha})
        config.update({"temperature": self.temperature})
        return config


