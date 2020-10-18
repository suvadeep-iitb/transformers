# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" A TF 2.0 Adaptive Softmax for Transformer XL model.
"""


import tensorflow as tf

from .modeling_tf_utils import shape_list


class TFAdaptiveSoftmaxMask(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1, keep_order=False, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.n_clusters = len(cutoffs) - 1

        self.div_val = div_val
        self.keep_order = keep_order

        self.out_weights = []
        self.out_biases = []
        self.out_projs = []

    def build(self, input_shape):
        if self.n_clusters > 0:
            self.cluster_weight = self.add_weight(
                shape=(self.n_clusters, self.d_embed), initializer="zeros", \
                    trainable=True, name="cluster_weight"
            )
            self.cluster_bias = self.add_weight(
                shape=(self.n_clusters,), initializer="zeros", trainable=True, \
                    name="cluster_bias"
            )

        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = self.d_embed // (self.div_val ** i)
            self.out_weights.append(
                self.add_weight(
                    shape=(r_idx - l_idx, d_emb_i),
                    initializer="zeros",
                    trainable=True,
                    name="out_weights_._{}".format(i),
                )
            )
            self.out_biases.append(
                self.add_weight(
                    shape=(r_idx - l_idx,),
                    initializer="zeros",
                    trainable=True,
                    name="out_layers_._{}_.bias".
                        format(i)
                )
            )

            if d_emb_i == d_proj and self.div_val == 1:
                self.emb_projs.append(None)
            else:
                self.emb_projs.append(
                    self.add_weight(
                        shape=(d_emb_i, self.d_proj),
                        initializer="zeros",
                        trainable=True,
                        name="out_projs_._{}".format(i),
                    )
                )

    @staticmethod
    def _logit(x, W, b, proj=None):
        y = x
        if x.shape.ndims == 3:
            if proj is not None:
                y = tf.einsum("ibd,ed->ibe", y, proj)
            return tf.einsum("ibd,nd->ibn", y, W) + b
        else:
            if proj is not None:
                y = tf.einsum('id,ed->ie', y, proj)
            return tf.einsum('id,nd->in', y, W) + b

    @staticmethod
    def _gather_logprob(logprob, target):
        lp_size = shape_list(target)
        r = tf.range(lp_size[0])
        c = tf.range(lp_size[1])
        C, R = tf.meshgrid(c, r)
        idx = tf.stack([R, C, target], axis=2)
        return tf.gather_nd(logprob, idx)

    def call(self, hidden, target, return_mean=True, training=False):
        head_logprob = 0
        if self.n_clusters == 0:
            output = self._logit(hidden, self.out_weights[0], self.out_biases[0], self.out_projs[0])
            out = tf.nn.log_softmax(output, axis=-1)
            if target is not None:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
        else:
            hidden_sizes = shape_list(hidden)
            out = []
            loss = tf.zeros(hidden_sizes[:2], dtype=tf.float32)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                if target is not None:
                    mask = (target >= l_idx) & (target < r_idx)
                    cur_target = tf.minimum(target, r_idx-1)
                    cur_target = tf.maximum(cur_target-l_idx, 0)

                cur_W = self.out_weights[i]
                cur_b = self.out_biases[i]
                cur_P = self.out_projs[i]

                if i == 0:
                    cur_W = tf.concat([cur_W, self.cluster_weight], 0)
                    cur_b = tf.concat([cur_b, self.cluster_bias], 0)

                    head_logit = self._logit(hidden, cur_W, cur_b, cur_P)
                    head_logprob = tf.nn.log_softmax(head_logit)
                    out.append(head_logprob[..., :self.cutoffs[0]])
                        
                    if target is not None:
                        cur_loss = self._gather_logprob(head_logprob, cur_target)
                        loss = tf.where(mask, cur_loss, loss)
                else:
                    tail_logit = self._logit(hidden, cur_W, cur_b, cur_P)
                    tail_logprob = tf.nn.log_softmax(tail_logit)

                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    logprob_i = head_logprob[..., cluster_prob_idx, None] + tail_logprob
                    out.append(logprob_i)

                    if target is not None:
                        cur_loss = self._gather_logprob(logprob_i, cur_target)
                        loss = tf.where(mask, cur_loss, loss)

            if target is not None:
                loss = -loss
            out = tf.concat(out, axis=-1)

        if target is not None:
            if return_mean:
                loss = tf.reduce_mean(loss)
            # Add the training-time loss value to the layer using `self.add_loss()`.
            self.add_loss(loss)

            # Log the loss as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.
            self.add_metric(loss, name=self.name, aggregation="mean" if return_mean else "")

        return out
