# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.utils.utils import mask_nans, deco_print
from .loss import Loss

import sys
import numpy as np

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.distances import CosineSimilarity

def dense_to_sparse(dense_tensor, sequence_length):
  indices = tf.where(tf.sequence_mask(sequence_length))
  values = tf.gather_nd(dense_tensor, indices)
  shape = tf.shape(dense_tensor, out_type=tf.int64)
  return tf.SparseTensor(indices, values, shape)


class CTCLoss(Loss):
  """Implementation of the CTC loss."""
  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
        'mask_nan': bool,
    })

  def __init__(self, params, model, name="ctc_loss"):
    """CTC loss constructor.

    See parent class for arguments description.

    Config parameters:

    * **mask_nan** (bool) --- whether to mask nans in the loss output. Defaults
      to True.
    """
    super(CTCLoss, self).__init__(params, model, name)
    self._mask_nan = self.params.get("mask_nan", True)
    # this loss can only operate in full precision
    # if self.params['dtype'] != tf.float32:
    #   deco_print("Warning: defaulting CTC loss to work in float32")
    self.params['dtype'] = tf.float32
    self.mining_func = miners.TripletMarginMiner(margin = 0.5, distance = CosineSimilarity, type_of_triplets = "hard")

  def _compute_loss(self, input_dict):
    """CTC loss graph construction.

    Expects the following inputs::

      input_dict = {

      }

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              "decoder_output": {
                "logits": tensor, shape [batch_size, time length, tgt_vocab_size]
                "src_length": tensor, shape [batch_size]
              },
              "target_tensors": [
                tgt_sequence (shape=[batch_size, time length, num features]),
                tgt_length (shape=[batch_size])
              ]
            }

    Returns:
      averaged CTC loss.
    """
    logits = input_dict['decoder_output']['logits']
    tgt_sequence, tgt_length = input_dict['target_tensors']
    # this loss needs an access to src_length since they
    # might get changed in the encoder
    src_length = input_dict['decoder_output']['src_length']

    # Compute the CTC loss
    total_loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(tgt_sequence, tgt_length),
        inputs=logits,
        sequence_length=src_length,
        ignore_longer_outputs_than_inputs=True,
    )

    if self._mask_nan:
      total_loss = mask_nans(total_loss)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)
    return avg_loss

  def _compute_cosine_distances(self, a, b):
    # x shape is n_a * dim
    # y shape is n_b * dim
    # results shape is n_a * n_b

    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1)
    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance

  def _get_anchor_positive_triplet_mask(self, labels):
      """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
      Args:
          labels: tf.int32 `Tensor` with shape [batch_size]
      Returns:
          mask: tf.bool `Tensor` with shape [batch_size, batch_size]
      """
      # Check that i and j are distinct
      indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
      indices_not_equal = tf.logical_not(indices_equal)

      # Check if labels[i] == labels[j]
      # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
      labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

      # Combine the two masks
      mask = tf.logical_and(indices_not_equal, labels_equal)

      return mask


  def _get_anchor_negative_triplet_mask(self, labels):
      """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
      Args:
          labels: tf.int32 `Tensor` with shape [batch_size]
      Returns:
          mask: tf.bool `Tensor` with shape [batch_size, batch_size]
      """
      # Check if labels[i] != labels[k]
      # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
      labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

      mask = tf.logical_not(labels_equal)

      return mask




  def _compute_triplet_loss(self, output1, labels):
    # print(output1.shape, output2.shape, output3.shape)
    # r = int((output1.shape[1]-20)/2)
    # s = int((output2.shape[1]-20)/2)
    # print(tf.reduce_mean(output1[:,:,:],axis=1).shape)
    embeddings = tf.reduce_mean(output1[:,:,:],axis=1)
    pairwise_dist = self._compute_cosine_distances(embeddings, embeddings)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    # mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)
    pairwise_dist = tf.cast(pairwise_dist, tf.float32)
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    # tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # # For each anchor, get the hardest negative
    # # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    # tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + 0.7, 0.0)

    # # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    # cos_hinge_loss = tf.clip_by_value(- (1 - tf.losses.cosine_distance(tf.reduce_mean(output1[:,:,:],axis=1), tf.reduce_mean(output2[:,:,:], axis=1), axis=-1, reduction=tf.losses.Reduction.MEAN)) \
    #                                    + (1 - tf.losses.cosine_distance(tf.reduce_mean(output1[:,:,:],axis=1), tf.reduce_mean(output3[:,:,:], axis=1), axis=-1, reduction=tf.losses.Reduction.MEAN)) \
    #                                   + 0.5, #margin
    #                                   clip_value_min=0, clip_value_max=tf.float32.max)
    return triplet_loss

  def _compute_dtw_loss(self, output1, output2, output3):
    pass