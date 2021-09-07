# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert softmax model to logits model used for predict
"""

import argparse
import os
import tensorflow as tf

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--scr_path',
        default=os.path.join(base_dir, "../examples/models_eval_ckpt/cnn1d", "checkpoint-60.h5"),
        help='path of the model whose output is softmax')
    parser.add_argument(
        '--dest_path',
        default=os.path.join(base_dir, "../examples/models_eval_ckpt/cnn1d", "checkpoint-60-logits.h5"),
        help='path of the model whose output is logits')

    args = parser.parse_args()
    loaded_model = tf.keras.models.load_model(args.scr_path)
    loaded_model_json = loaded_model.to_json()
    model_json = loaded_model_json.replace("softmax", "linear")
    model_logits = tf.keras.models.model_from_json(model_json)
    model_logits.set_weights(loaded_model.get_weights())
    model_logits.save(args.dest_path)
