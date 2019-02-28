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
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

import math
import tensorflow as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.math import random_rademacher
from tensorflow_probability.python.util import docstring as docstring_util
from tensorflow_probability.python.layers.dense_variational import _DenseVariational


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

    # for testing purposes, you'd need access to the entire output
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
    outputs = tf.nn.softmax(outputs, 1)


    # for training purposes, get clever. Return true_logits and sampled_logits
    k_const  = (self.units - 1)/(self.num_samples)
    true_logits, sampled_logits = _compute_sampled_logits(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets, inputs, self.num_samples, self.units)


    approx = 1. + k_const * _sum_rows(tf.exp(sampled_logits - true_logits))
    eta = array_ops.stop_gradient(approx)
    lowerbound = 1 - tf.log(eta) - (1./eta) * approx

    #
    # lowerbound = mul_factor * _sum_rows(tf.log_sigmoid(true_logits - sampled_logits))
    ar_loss = -tf.keras.backend.mean(lowerbound)

    # softmax_loss = tf.keras.backend.mean(
    #                 tf.keras.backend.sparse_categorical_crossentropy(
    #                     targets, outputs, from_logits = True
    #                 ))
    # # # add as implicit losses
    # true_loss = tf.keras.backend.in_train_phase(ove_loss, softmax_loss)
    self.add_loss(ar_loss)

    # outputs = self._apply_variational_kernel(inputs)
    # outputs = self._apply_variational_bias(outputs)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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

    return tf.keras.backend.in_train_phase(sampled_logits, outputs, training = None)


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


class SLCSoftmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      M,
      loss_mat,
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
    super(SLCSoftmax, self).__init__(
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
    self.M = M
    self.loss_mat =  tf.convert_to_tensor(value=loss_mat, dtype=tf.float32)
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 3
    input_shape, target_shape, hypothesis_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    hypothesis_shape = tf.TensorShape(hypothesis_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value}),
                        tf.keras.layers.InputSpec(min_ndim=1,
                                 axes={-1: hypothesis_shape[-1].value})]

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
    inputs, targets, hypothesis = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)
    hypothesis = tf.convert_to_tensor(value=hypothesis, dtype=self.dtype)

    targets = math_ops.cast(targets, dtypes.int64)
    targets_flat = array_ops.reshape(targets, [-1])

    hypothesis = math_ops.cast(hypothesis, dtypes.int64)
    hypothesis = array_ops.reshape(hypothesis, [-1])

    # for testing purposes, you'd need access to the entire output
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    logits = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
    outputs = tf.nn.softmax(logits)
    softmax_loss = tf.keras.backend.sparse_categorical_crossentropy(
                            targets,
                            outputs,
                            from_logits = False)

    # sampled_values = candidate_sampling_ops.all_candidate_sampler(
    #       true_classes=tf.cast(targets, tf.int64),
    #       num_true=1,
    #       num_sampled=self.units,
    #       unique=True)

    sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
          true_classes=targets,
          num_true=1,
          num_sampled=1,
          unique=True,
          range_max=self.units)

   # the rain in spain stays mainly in the plain
    sampled, true_expected_count, sampled_expected_count  =  (
                array_ops.stop_gradient(s) for s in sampled_values)
    sampled = math_ops.cast(sampled, dtypes.int64)

    # idx0 = tf.range(tf.shape(hypothesis)[0])
    # idx0 = tf.cast(idx0, tf.int64)
    #
    # idx = tf.stack([idx0, hypothesis], 1)
    # idh = tf.stack([hypothesis, targets_flat], 1)
    #
    # h1 = tf.gather_nd(self.loss_mat, idh)
    # l1 = tf.gather_nd(outputs, idx)
    # loss_term = h1 * l1

    loss_mat_sampled = tf.gather(self.loss_mat, sampled, axis = 1)
    outputs_sampled  = tf.gather(outputs, sampled, axis = 1)
    h_losses = tf.nn.embedding_lookup(loss_mat_sampled, hypothesis)
    utility = _sum_rows(h_losses * outputs_sampled)
    log_utility = tf.log(utility)

    # loss_mat_sampled = tf.gather(self.loss_mat, hypothesis, axis = 0)
    # idx0  = tf.cast(tf.range(tf.shape(hypothesis)[0]), tf.int64)
    # idx = tf.stack([hypothesis, idx0], 1)
    # idx1 = tf.stack([idx0, hypothesis], 1)
    # l = tf.gather_nd(loss_mat_sampled, idx1)
    # h = tf.gather_nd(outputs, idx1)
    # risk =  h
    # log_utility = tf.log(risk)


    lc_loss = softmax_loss - log_utility
    self.add_loss(tf.keras.backend.mean(lc_loss))


    # m_term = _sum_rows(self.M * outputs_sampled)
    # utility_term = tf.log(self.M - risk_term)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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
    return outputs

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

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    return tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

class Softmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      M = None,
      loss_mat = None,
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
    self.M = M
    self.loss_mat = loss_mat
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

    # for testing purposes, you'd need access to the entire output
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
    outputs = tf.nn.softmax(outputs)
    softmax_loss = tf.keras.backend.sparse_categorical_crossentropy(
                            targets,
                            outputs,
                            from_logits = False)

    self.add_loss(tf.keras.backend.mean(softmax_loss))

    # outputs = self._apply_variational_kernel(inputs)
    # outputs = self._apply_variational_bias(outputs)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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
    return outputs

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

class LCSoftmax(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      units,
      M,
      loss_mat,
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
    super(LCSoftmax, self).__init__(
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
    self.M = M
    self.loss_mat =  tf.convert_to_tensor(value=loss_mat, dtype=tf.float32)
    self.input_spec = [ tf.keras.layers.InputSpec(min_ndim=2),
                        tf.keras.layers.InputSpec(min_ndim=1),
                        tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 3
    input_shape, target_shape, hypothesis_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    hypothesis_shape = tf.TensorShape(hypothesis_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value}),
                        tf.keras.layers.InputSpec(min_ndim=1,
                                 axes={-1: hypothesis_shape[-1].value})]

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
    inputs, targets, hypothesis = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)

    hypothesis = tf.convert_to_tensor(value=hypothesis, dtype=self.dtype)
    hypothesis = math_ops.cast(hypothesis, dtypes.int64)
    hypothesis = array_ops.reshape(hypothesis, [-1])

    # for testing purposes, you'd need access to the entire output
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)
    outputs = tf.nn.softmax(outputs)
    softmax_loss = tf.keras.backend.sparse_categorical_crossentropy(
                            targets,
                            outputs,
                            from_logits = False)

    h_losses = tf.nn.embedding_lookup(self.loss_mat, hypothesis)
    loss_term = _sum_rows(h_losses * outputs)
    utility_term = tf.log(self.M - loss_term)
    lc_loss = softmax_loss - utility_term

    self.add_loss(tf.keras.backend.mean(lc_loss))

    # outputs = self._apply_variational_kernel(inputs)
    # outputs = self._apply_variational_bias(outputs)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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
    return outputs

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

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    return tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

class LCOVELayer(_DenseVariational):
  # @docstring_util.expand_docstring(args=doc_args)
  def __init__(
      self,
      num_samples,
      units,
      M,
      loss_mat,
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
    super(LCOVELayer, self).__init__(
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
    self.M = M
    self.loss_mat = loss_mat
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2), tf.keras.layers.InputSpec(min_ndim=1),
                            tf.keras.layers.InputSpec(min_ndim=1)]

  def build(self, input_shape_list):
    assert len(input_shape_list) == 3
    input_shape, target_shape, hypothesis_shape = input_shape_list
    input_shape = tf.TensorShape(input_shape)
    target_shape = tf.TensorShape(target_shape)
    hypothesis_shape = tf.TensorShape(hypothesis_shape)
    in_size = input_shape[-1].value
    if in_size is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})
    self.input_spec = [tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: in_size}),
                       tf.keras.layers.InputSpec(min_ndim=1,
                                axes={-1: target_shape[-1].value}),
                        tf.keras.layers.InputSpec(min_ndim=1,
                                 axes={-1: hypothesis_shape[-1].value})]

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
    inputs, targets, hypothesis = inputs_list
    inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
    targets = tf.convert_to_tensor(value=targets, dtype=self.dtype)
    hypothesis = tf.convert_to_tensor(value=hypothesis, dtype=self.dtype)

    # for testing purposes, you'd need access to the entire output
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = tf.nn.bias_add(outputs, self.bias_posterior_tensor)

    # for training purposes, get clever. Return true_logits and sampled_logits
    true_logits, sampled_logits, sampled_idx = _compute_sampled_logits(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets, inputs, self.num_samples, self.units, 1, True)

    lowerbound = tf.reduce_sum(tf.log_sigmoid(true_logits - sampled_logits), 1)

    # utility_dependent_term = tf.log(self.M - tf.reduce_prod)
    # add as implicit losses
    self.add_loss(-tf.keras.backend.mean(lowerbound))

    # outputs = self._apply_variational_kernel(inputs)
    # outputs = self._apply_variational_bias(outputs)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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
    return tf.keras.backend.in_train_phase(sampled_logits, outputs, training = None)


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

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    return tf.matmul(inputs, self.kernel_posterior_tensor, transpose_b=True)

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
    outputs = tf.nn.softmax(outputs, 1)

    # for training purposes, get clever. Return true_logits and sampled_logits
    mul_factor = (self.units - 1)/(self.num_samples)
    true_logits, sampled_logits = _compute_sampled_logits(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets, inputs, self.num_samples, self.units)
    #
    lowerbound = mul_factor * _sum_rows(tf.log_sigmoid(true_logits - sampled_logits))
    ove_loss = -tf.keras.backend.mean(lowerbound)

    # softmax_loss = tf.keras.backend.mean(
    #                 tf.keras.backend.sparse_categorical_crossentropy(
    #                     targets, outputs, from_logits = True
    #                 ))
    # # # add as implicit losses
    # true_loss = tf.keras.backend.in_train_phase(ove_loss, softmax_loss)
    true_loss = ove_loss
    self.add_loss(true_loss)

    # outputs = self._apply_variational_kernel(inputs)
    # outputs = self._apply_variational_bias(outputs)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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

    return tf.keras.backend.in_train_phase(sampled_logits, outputs, training = None)


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

class NCE(_DenseVariational):
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
    super(NCE, self).__init__(
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
    outputs = tf.nn.softmax(outputs, 1)

    nce_loss = tf.nn.sampled_softmax_loss(
                    self.kernel_posterior_tensor,
                    self.bias_posterior_tensor,
                    targets,
                    inputs,
                    self.num_samples,
                    self.units,
                    remove_accidental_hits=True)
    # for training purposes, get clever. Return true_logits and sampled_logits
    # mul_factor = (self.units - 1)/(self.num_samples)
    # true_logits, sampled_logits = _compute_sampled_logits(
    #                 self.kernel_posterior_tensor,
    #                 self.bias_posterior_tensor,
    #                 targets, inputs, self.num_samples, self.units)
    # #
    # lowerbound = mul_factor * _sum_rows(tf.log_sigmoid(true_logits - sampled_logits))
    # ove_loss = -tf.keras.backend.mean(lowerbound)
    #
    # softmax_loss = tf.keras.backend.mean(
    #                 tf.keras.backend.sparse_categorical_crossentropy(
    #                     targets, outputs, from_logits = True
    #                 ))
    # # # add as implicit losses
    # true_loss = tf.keras.backend.in_train_phase(ove_loss, softmax_loss)

    self.add_loss(tf.keras.backend.mean(nce_loss))

    # outputs = self._apply_variational_kernel(inputs)
    # outputs = self._apply_variational_bias(outputs)
    # if self.activation is not None:
    #   outputs = self.activation(outputs)  # pylint: disable=not-callable

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

    return tf.keras.backend.in_train_phase(outputs, outputs, training = None)


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

def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
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
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
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
    if all_w.dtype != inputs.dtype:
      all_w = math_ops.cast(all_w, inputs.dtype)

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
    if all_b.dtype != inputs.dtype:
      all_b = math_ops.cast(all_b, inputs.dtype)
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
      sampled_logits += gen_sparse_ops.sparse_to_dense(
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
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    # out_labels = array_ops.concat([
    #     array_ops.ones_like(true_logits) / num_true,
    #     array_ops.zeros_like(sampled_logits)
    # ], 1)
  return true_logits, sampled_logits

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
