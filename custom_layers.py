from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

import math
import tensorflow as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.util import docstring as docstring_util
from tensorflow_probability.python.layers.dense_variational import _DenseVariational

class SUCOVELayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      util_rows,
      util_cols,
      util_vals,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(SUCOVELayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.util_rows =  tf.ragged.constant(util_rows, dtype=tf.int64)
    self.util_cols =  tf.ragged.constant(util_cols, dtype=tf.int64)
    self.util_vals =  tf.ragged.constant(util_vals, dtype=tf.float32)

    idx = tf.stack([self.util_rows.values, self.util_cols.values], 1)
    self.util_mat = tf.SparseTensor(idx, self.util_vals.values, [units, units])
    self.num_samples = num_samples
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    with ops.name_scope('M-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)

        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
        outputs = tf.transpose(outputs, [1,0])
        outputs = tf.sparse.sparse_dense_matmul(self.util_mat, outputs)
        outputs = tf.transpose(outputs, [1,0])
        hypothesis = tf.argmax(tf.exp(outputs), axis = 1)
        hypothesis = array_ops.stop_gradient(hypothesis)

    with ops.name_scope('E-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)


        true_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        targets, inputs,
                        self.num_samples, self.units)

        elbo =  _sum_rows(tf.log_sigmoid(true_logits - sampled_logits))


        hrows = tf.gather(self.util_rows, hypothesis)
        hcols = tf.gather(self.util_cols, hypothesis)
        hvals = tf.gather(self.util_vals, hypothesis)
        idb = tf.reshape(tf.range(tf.shape(targets)[0]), [-1,1])
        idb = tf.cast(idb, tf.int64)
        idb = tf.zeros_like(hrows) + idb
        idx = tf.stack([idb.values, hcols.values], 1)
        idx = tf.keras.backend.print_tensor(idx,"idx")

        sparse_logits = _compute_sparse_logits(
                      self.kernel_posterior_tensor,
                      self.bias_posterior_tensor,
                      inputs,
                      hcols.values,
                      idb.values)
        sparse_samples = tf.gather(sampled_logits, idb.values)
        sparse_samples = tf.reshape(sparse_samples, [-1,1])
        sparse_elbo = _sum_rows(tf.log_sigmoid(sparse_logits - sparse_samples))
        sparse_elbo = tf.keras.backend.print_tensor(sparse_elbo, "sparse")
        # idx_probs = hvals.values * sparse_elbo
        # idx_probs_ragged = tf.RaggedTensor.from_row_splits(idx_probs, hrows.row_splits)
        # gain = tf.reduce_sum(idx_probs_ragged, 1)

        lc_loss = elbo  + 0*tf.reduce_sum(sparse_elbo)#+ tf.log(gain)
        lc_loss = -tf.keras.backend.mean(lc_loss)

    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, outputs)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class SUCSoftmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      util_rows,
      util_cols,
      util_vals,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(SUCSoftmax, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.util_rows =  tf.ragged.constant(util_rows, dtype=tf.int64)
    self.util_cols =  tf.ragged.constant(util_cols, dtype=tf.int64)
    self.util_vals =  tf.ragged.constant(util_vals, dtype=tf.float32)

    idx = tf.stack([self.util_rows.values, self.util_cols.values], 1)
    self.util_mat = tf.SparseTensor(idx, self.util_vals.values, [units, units])

    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    with ops.name_scope('M-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)

        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
        outputs = tf.transpose(outputs, [1,0])
        outputs = tf.sparse.sparse_dense_matmul(self.util_mat, outputs)
        outputs = tf.transpose(outputs, [1,0])
        hypothesis = tf.argmax(tf.exp(outputs), axis = 1)
        hypothesis = array_ops.stop_gradient(hypothesis)

    with ops.name_scope('E-step'):
      sampled_values = candidate_sampling_ops.all_candidate_sampler(
                              true_classes = tf.cast(targets, tf.int64),
                              num_true = 1,
                              num_sampled = self.units,
                              unique = True)

      true_logits, sampled_logits = _compute_sampled_logits(
                      self.kernel_posterior_tensor,
                      self.bias_posterior_tensor,
                      targets, inputs, self.units, self.units,
                      sampled_values = sampled_values,
                      remove_accidental_hits = False)

      log_Z = tf.reduce_logsumexp(sampled_logits, 1, keepdims = True)
      softmax_loss = - true_logits + log_Z
      #
      hrows = tf.gather(self.util_rows, hypothesis)
      hcols = tf.gather(self.util_cols, hypothesis)
      hvals = tf.gather(self.util_vals, hypothesis)
      idb = tf.reshape(tf.range(tf.shape(targets)[0]), [-1,1])
      idb = tf.cast(idb, tf.int64)
      idb = tf.zeros_like(hrows) + idb
      idx = tf.stack([idb.values, hcols.values], 1)

      idx_logits = tf.gather_nd(outputs, idx)

      idx_log_Z = tf.gather(log_Z, idb.values)
      idx_probs = hvals.values * tf.exp(idx_logits  - idx_log_Z)
      idx_probs_ragged = tf.RaggedTensor.from_row_splits(idx_probs,
                                                         hrows.row_splits)
      gain = tf.reduce_sum(idx_probs_ragged, 1)

      lc_loss = softmax_loss - tf.log(gain)
      lc_loss = tf.keras.backend.mean(lc_loss)

    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, outputs)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class DUCOVELayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      util_mat,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(DUCOVELayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.num_samples = num_samples
    self.util_mat =  tf.convert_to_tensor(value=util_mat, dtype=tf.float32)
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    targets = math_ops.cast(targets, dtypes.int64)
    targets = array_ops.stop_gradient(targets)
    targets_flat = array_ops.reshape(targets, [-1])

    with ops.name_scope('M-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
        outputs = tf.exp(outputs)
        outputs = tf.matmul(outputs, self.util_mat, transpose_b = True)
        hypothesis = tf.argmax(outputs, axis = 1)
        hypothesis = array_ops.stop_gradient(hypothesis)

    with ops.name_scope('E-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

        sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
             true_classes=targets,
             num_true=1,
             num_sampled=self.num_samples,
             unique=True,
             range_max=self.units)
        sampled, true_expected_count, sampled_expected_count = (
           array_ops.stop_gradient(s) for s in sampled_values)
        sampled = math_ops.cast(sampled, dtypes.int64)

        k_const = (self.units - 1)/(self.num_samples)
        idx = tf.cast(tf.range(tf.shape(targets)[0]), tf.int64)
        target_ids = tf.stack([idx, targets_flat], 1)

        sampled_logits = tf.gather(outputs, sampled, axis=1)
        target_logits = tf.gather_nd(outputs, target_ids)
        target_logits = tf.reshape(target_logits, [-1,1])

        sampled_logits = _remove_accidental_hits(sampled_logits,
                                            targets, sampled,
                                            self.num_samples)


        elbo =  _sum_rows(tf.log_sigmoid(target_logits - sampled_logits))

        lowerbound = tf.reduce_sum(
                    tf.log_sigmoid(outputs[:,None] - sampled_logits[...,None]), 1)
        ar_probs = tf.exp(lowerbound)
        util = tf.nn.embedding_lookup(self.util_mat, hypothesis)
        u_p = util * ar_probs
        gain = _sum_rows(u_p)

        lc_loss =  elbo + tf.log(gain)
        lc_loss = - tf.keras.backend.mean(lc_loss)

    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, outputs)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class DUCARLayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      util_mat,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(DUCARLayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.num_samples = num_samples
    self.util_mat =  tf.convert_to_tensor(value=util_mat, dtype=tf.float32)
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    targets = math_ops.cast(targets, dtypes.int64)
    targets = array_ops.stop_gradient(targets)
    targets_flat = array_ops.reshape(targets, [-1])

    with ops.name_scope('M-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
        outputs = tf.exp(outputs)
        outputs = tf.matmul(outputs, self.util_mat, transpose_b = True)
        hypothesis = tf.argmax(outputs, axis = 1)
        hypothesis = array_ops.stop_gradient(hypothesis)

    with ops.name_scope('E-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

        sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
             true_classes=targets,
             num_true=1,
             num_sampled=self.num_samples,
             unique=True,
             range_max=self.units)
        sampled, true_expected_count, sampled_expected_count = (
           array_ops.stop_gradient(s) for s in sampled_values)
        sampled = math_ops.cast(sampled, dtypes.int64)

        k_const = (self.units - 1)/(self.num_samples)
        idx = tf.cast(tf.range(tf.shape(targets)[0]), tf.int64)
        target_ids = tf.stack([idx, targets_flat], 1)

        # target_logits, sampled_logits = _compute_sampled_logits(
        #                 self.kernel_posterior_tensor,
        #                 self.bias_posterior_tensor,
        #                 targets, inputs,
        #                 self.num_samples, self.units,
        #                 remove_accidental_hits = True)

        sampled_logits = tf.gather(outputs, sampled, axis=1)
        target_logits = tf.gather_nd(outputs, target_ids)
        target_logits = tf.reshape(target_logits, [-1,1])

        sampled_logits = _remove_accidental_hits(sampled_logits,
                                            targets, sampled,
                                            self.num_samples)


        approx = 1. + k_const * _sum_rows(tf.exp(sampled_logits - target_logits))
        eta = array_ops.stop_gradient(approx)
        elbo = 1 - tf.log(eta) - approx/eta

        exp_term = tf.reduce_sum(
                    tf.exp(sampled_logits[...,None] - outputs[:,None]), 1)
        approx = 1. + k_const * exp_term
        eta = array_ops.stop_gradient(approx)
        lowerbound = 1 - tf.log(eta) - approx/eta
        ar_probs = tf.exp(lowerbound)
        util = tf.nn.embedding_lookup(self.util_mat, hypothesis)
        u_p = util * ar_probs
        gain = _sum_rows(u_p)

        lc_loss =  elbo + tf.log(gain)
        lc_loss = - tf.keras.backend.mean(lc_loss)

    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, outputs)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class DUCSoftmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      util_mat,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(DUCSoftmax, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    # self.num_samples = num_samples
    self.util_mat = tf.convert_to_tensor(util_mat, dtype = tf.float32)
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]
    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    with ops.name_scope('M-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)


        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b = True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
        outputs = tf.exp(outputs)
        outputs = tf.matmul(outputs, self.util_mat, transpose_b = True)
        hypothesis = tf.argmax(outputs, axis = 1)
        hypothesis = array_ops.stop_gradient(hypothesis)

    with ops.name_scope('E-step'):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)

        sampled_values = candidate_sampling_ops.all_candidate_sampler(
                                true_classes = tf.cast(targets, tf.int64),
                                num_true = 1,
                                num_sampled = self.units,
                                unique = True)
        true_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        targets, inputs, self.units, self.units,
                        sampled_values = sampled_values,
                        remove_accidental_hits = False)
        #
        log_Z = tf.reduce_logsumexp(sampled_logits, 1, keepdims = True)
        log_softmax = true_logits - log_Z
        #

        softmax_probs = tf.exp(sampled_logits  -  log_Z)
        util = tf.nn.embedding_lookup(self.util_mat, hypothesis)
        gain = _sum_rows(util * softmax_probs)
        lc_loss =  log_softmax + tf.log(gain)
        lc_loss = - tf.keras.backend.mean(lc_loss)
        self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, sampled_logits)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class IUCARLayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(IUCARLayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.num_samples = num_samples
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    targets = math_ops.cast(targets, dtypes.int64)
    targets_flat = array_ops.reshape(targets, [-1])

    # for testing purposes, you'd need access to the entire output
    with ops.name_scope("M-step"):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

        hypothesis = tf.argmax(tf.exp(outputs), axis = 1)
        hypothesis = tf.reshape(hypothesis, [-1, 1])
        hypothesis = array_ops.stop_gradient(hypothesis)

    with ops.name_scope("E-step"):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.nn.bias_add(outputs, m_bias_posterior_tensor)


        k_const = (self.units - 1)/(self.num_samples)

        target_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        targets, inputs,
                        self.num_samples, self.units)

        approx = 1. + k_const * _sum_rows(tf.exp(sampled_logits - target_logits))
        eta = array_ops.stop_gradient(approx)
        lowerbound = 1 - tf.log(eta) - approx/eta

        hypothesis_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        hypothesis, inputs,
                        self.num_samples, self.units)

        approx = 1. + k_const * _sum_rows(tf.exp(sampled_logits - hypothesis_logits))
        eta = array_ops.stop_gradient(approx)
        log_gain = 1 - tf.log(eta) - approx/eta

        lc_loss = - tf.keras.backend.mean(lowerbound + log_gain)
        # lc_loss = -tf.keras.backend.mean(lowerbound)
    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True

    return tf.keras.backend.in_train_phase(target_logits, outputs, training = None)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class IUCOVELayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(IUCOVELayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.num_samples = num_samples
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    targets = math_ops.cast(targets, dtypes.int64)
    targets_flat = array_ops.reshape(targets, [-1])

    with ops.name_scope('M-step'):
    # for testing purposes, you'd need access to the entire output
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

        hypothesis = tf.argmax(tf.exp(outputs), axis = 1)
        hypothesis = tf.reshape(hypothesis, [-1, 1])
        hypothesis = array_ops.stop_gradient(hypothesis)
    with ops.name_scope('E-step'):

        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)

        # k_const = (self.units - 1)/(self.num_samples)
        k_const = 1.
        target_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        targets, inputs,
                        self.num_samples, self.units,
                        num_true = 1)

        lowerbound =  _sum_rows(tf.log_sigmoid(target_logits - sampled_logits))

        hypothesis_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        hypothesis, inputs,
                        self.num_samples, self.units,
                        num_true = 1)

        log_gain =  _sum_rows(tf.log_sigmoid(hypothesis_logits - sampled_logits))

        lc_loss = -k_const * tf.keras.backend.mean(lowerbound + log_gain)

    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True

    return tf.keras.backend.in_train_phase(target_logits, outputs, training = None)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class IUCSoftmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(IUCSoftmax, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2

    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    targets = math_ops.cast(targets, dtypes.int64)
    targets_flat = array_ops.reshape(targets, [-1])

    with ops.name_scope("E-step"):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)

        outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
        outputs = tf.exp(outputs)
        hypothesis = tf.argmax(outputs, axis = 1)
        hypothesis = array_ops.stop_gradient(hypothesis)
        true_classes = tf.stack([targets_flat, hypothesis], 1)

    with ops.name_scope("M-step"):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterior_affine_tensor = None
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
            self.bias_posterior)

        sampled_values = candidate_sampling_ops.all_candidate_sampler(
                                true_classes = true_classes,
                                num_true = 2,
                                num_sampled = self.units,
                                unique = True)

        true_logits, sampled_logits = _compute_sampled_logits(
                        self.kernel_posterior_tensor,
                        self.bias_posterior_tensor,
                        true_classes, inputs, self.units, self.units,
                        num_true = 2,
                        sampled_values = sampled_values,
                        remove_accidental_hits = False)

        target_logits = true_logits[:,0]
        hypothesis_logits = true_logits[:,1]

        log_Z = tf.reduce_logsumexp(sampled_logits, 1, keepdims = True)
        softmax_loss = - target_logits + log_Z

        log_gain = hypothesis_logits -  log_Z
        lc_loss = softmax_loss - log_gain
        lc_loss = tf.keras.backend.mean(lc_loss)
    self.add_loss(lc_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, sampled_logits, training = None)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 3
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class ARLayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(ARLayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.num_samples = num_samples
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2), tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

    # for training purposes, get clever. Return true_logits and sampled_logits
    k_const  =  (self.units - 1)/(self.num_samples)
    true_logits, sampled_logits = _compute_sampled_logits(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets, inputs,
                    self.num_samples, self.units)

    approx = 1. + k_const * _sum_rows(tf.exp(sampled_logits - true_logits))
    eta = array_ops.stop_gradient(approx)
    lowerbound = 1 - tf.log(eta) - approx/eta
    ar_loss = -tf.keras.backend.mean(lowerbound)
    self.add_loss(ar_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True

    return tf.keras.backend.in_train_phase(true_logits, outputs, training = None)


  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class OVELayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(OVELayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    self.num_samples = num_samples
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2), tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]

    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    # for testing purposes, you'd need access to the entire output
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

    # k_const = (self.units - 1)/(self.num_samples)
    true_logits, sampled_logits = _compute_sampled_logits(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets, inputs,
                    self.num_samples, self.units)

    lowerbound =  _sum_rows(tf.log_sigmoid(true_logits - sampled_logits))
    ove_loss = - tf.keras.backend.mean(lowerbound)
    self.add_loss(ove_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True

    return tf.keras.backend.in_train_phase(true_logits, outputs, training = None)


  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

class Softmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(Softmax, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    # self.num_samples = num_samples

    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]
    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)

    sampled_values = candidate_sampling_ops.all_candidate_sampler(
                            true_classes = tf.cast(targets, tf.int64),
                            num_true = 1,
                            num_sampled = self.units,
                            unique = True)

    true_logits, sampled_logits = _compute_sampled_logits(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets, inputs, self.units, self.units,
                    sampled_values = sampled_values,
                    remove_accidental_hits = False)

    log_Z = tf.reduce_logsumexp(sampled_logits, 1, keepdims = False)
    softmax_loss = - true_logits + log_Z
    softmax_loss = tf.keras.backend.mean(softmax_loss)
    self.add_loss(softmax_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(sampled_logits, sampled_logits)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    return tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

class NCELayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
      kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
      **kwargs):
    # pylint: disable=g-doc-args
    """Construct layer.
    Args:
      ${args}
    """
    # pylint: enable=g-doc-args
    super(NCELayer, self).__init__(
        units=units,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        **kwargs)
    # self.num_samples = num_samples
    self.num_samples = num_samples
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 2
    input_shape, target_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)

    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value})]
    # If self.dtype is None, build weights using the default dtype.
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [self.units, in_size], 'kernel_posterior',
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [self.units, in_size], 'kernel_prior',
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], 'bias_posterior',
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], 'bias_prior',
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs_list):
    inputs, targets = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
    self.bias_posterior)
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

    sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
                            true_classes = tf.cast(targets, tf.int64),
                            num_true = 1,
                            num_sampled = self.num_samples,
                            unique = True,
                            range_max = self.units)

    nce_loss = tf.nn.nce_loss(
                self.kernel_posterior_tensor,
                self.bias_posterior_tensor,
                targets,
                inputs,
                self.num_samples,
                self.units,
                num_true=1,
                sampled_values=sampled_values,
                remove_accidental_hits=False,
                partition_strategy='mod',
                name='nce_loss'
            )


    nce_loss = tf.keras.backend.mean(nce_loss)
    self.add_loss(nce_loss)

    if not self._built_kernel_divergence:
      self._apply_divergence(self.kernel_divergence_fn,
                             self.kernel_posterior,
                             self.kernel_prior,
                             self.kernel_posterior_tensor,
                             name='divergence_kernel')
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias_divergence_fn,
                             self.bias_posterior,
                             self.bias_prior,
                             self.bias_posterior_tensor,
                             name='divergence_bias')
      self._built_bias_divergence = True
    return tf.keras.backend.in_train_phase(outputs, outputs)

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) == 2
    assert input_shape[0][-1]
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of `input_shape` must be defined, '
          'but saw: {}'.format(input_shape))
    return input_shape[:-1].concatenate(self.units)

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    return tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

# ********************************************** #
def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=False,
                            remove_accidental_hits=True,
                            partition_strategy="mod",
                            name=None,
                            seed=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.
  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).
  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.
  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
        class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits_v2`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.
  Returns:
    out_logits: `Tensor` object with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits_v2` (sampled softmax).
    out_labels: A Tensor object with the same shape as `out_logits`.
  """

  if isinstance(weights, variables.PartitionedVariable):
    weights = list(weights)
  if not isinstance(weights, list):
    weights = [weights]

  with ops.name_scope(name, "compute_sampled_logits",
                      weights + [biases, inputs, labels]):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = (
        array_ops.stop_gradient(s) for s in sampled_values)
    # pylint: enable=unpacking-non-sequence
    sampled = math_ops.cast(sampled, dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.

    # weights shape is [num_classes, dim]
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)

    # true_w shape is [batch_size * num_true, dim]
    true_w = array_ops.slice(all_w, [0, 0],
                             array_ops.stack(
                                 [array_ops.shape(labels_flat)[0], -1]))

    sampled_w = array_ops.slice(
        all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    all_b = embedding_ops.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)
    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    if remove_accidental_hits:
      acc_hits = candidate_sampling_ops.compute_accidental_hits(
          labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits

      # This is how SparseToDense expects the indices.
      acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = array_ops.reshape(
          math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
      sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                        "sparse_indices")
      # Create sampled_logits_shape = [batch_size, num_sampled]
      sampled_logits_shape = array_ops.concat(
          [array_ops.shape(labels)[:1],
           array_ops.expand_dims(num_sampled, 0)], 0)
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += sparse_ops.sparse_to_dense(
          sparse_indices,
          sampled_logits_shape,
          acc_weights,
          default_value=0.0,
          validate_indices=False)

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    # out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    # out_labels = array_ops.concat([
    #     array_ops.ones_like(true_logits) / num_true,
    #     array_ops.zeros_like(sampled_logits)
    # ], 1)

    return true_logits, sampled_logits

def _remove_accidental_hits(sampled_logits,
                            labels,
                            sampled,
                            num_sampled,
                            num_true = 1):
  if labels.dtype != dtypes.int64:
     labels = math_ops.cast(labels, dtypes.int64)
     labels = array_ops.stop_gradient(labels)


  acc_hits = candidate_sampling_ops.compute_accidental_hits(
        labels, sampled, num_true=num_true)
  acc_indices, acc_ids, acc_weights = acc_hits

  # This is how SparseToDense expects the indices.
  acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
  acc_ids_2d_int32 = array_ops.reshape(
      math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
  sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                    "sparse_indices")
  # Create sampled_logits_shape = [batch_size, num_sampled]
  sampled_logits_shape = array_ops.concat(
      [array_ops.shape(labels)[:1],
       array_ops.expand_dims(num_sampled, 0)], 0)
  if sampled_logits.dtype != acc_weights.dtype:
    acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
  sampled_logits += sparse_ops.sparse_to_dense(
      sparse_indices,
      sampled_logits_shape,
      acc_weights,
      default_value=0.0,
      validate_indices=False)
  return sampled_logits

def _compute_sparse_logits(
            weights,
            biases,
            inputs,
            row_idx,
            inp_idx):

    all_weights = tf.gather(weights, row_idx)
    all_biases = tf.gather(biases, row_idx)
    all_inputs = tf.gather(inputs, inp_idx)

    return _sum_rows(all_weights * all_inputs) + all_biases

def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.stack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])
