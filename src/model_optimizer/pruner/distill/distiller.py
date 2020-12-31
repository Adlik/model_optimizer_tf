import tensorflow as tf

from .distill_loss import DistillLossLayer


def get_distiller(student_model, scheduler_config):
    input_img = tf.keras.layers.Input(shape=(224, 224, 3), name='image')
    input_lbl = tf.keras.layers.Input((), name="label", dtype='int32')
    student = student_model
    _, logits = student(input_img)
    total_loss = DistillLossLayer(scheduler_config['teacher_path'], scheduler_config['alpha'],
                                  scheduler_config['temperature'], )([input_img, input_lbl, logits])
    distill_model = tf.keras.Model(inputs=[input_img, input_lbl], outputs=[logits, total_loss])

    return distill_model
