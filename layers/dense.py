from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops as init

import tensorflow as tf

from .utils import reparametrize, local_reparametrize, flipout_dense
from .utils import kl_mixture


class _DenseVariational(base.Layer):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               trainable=True,
               local_reparametrization=False,
               flipout=False,
               seed=None,
               name=None,
               **kwargs):
    super(_DenseVariational, self).__init__(
      trainable=trainable,
      name=name,
      **kwargs
    )
    if local_reparametrization and flipout:
      raise ValueError(
        'Cannot apply both flipout and local '
        'reparametrizations for variance reduction.')
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.local_reparametrization = local_reparametrization
    self.flipout = flipout
    self.seed = seed
    self.input_spec = base.InputSpec(min_ndim=2)

  def _add_bias(self, inputs, loc, scale, sample=True):
    if not self.use_bias:
      return inputs
    bias = reparametrize(loc, scale, sample)
    return tf.nn.bias_add(inputs, bias)

  def _apply_divergence(self, *args, **kwargs):
    raise NotImplementedError()

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
        "The innermost dimension of input_shape must be defined, "
        "but saw: {}".format(input_shape))
    return input_shape[:-1].concatenate(self.units)


class BayesBackpropDense(_DenseVariational):
  def __init__(self, units,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               mu_initializer=init.random_normal_initializer(0., 1e-2),
               rho_initializer=init.random_uniform_initializer(-9., -6.),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs):
    super(BayesBackpropDense, self).__init__(
      units=units,
      activation=activation,
      use_bias=use_bias,
      local_reparametrization=local_reparametrization,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )
    self.clip_std = clip_std
    self.prior_pi = prior_pi
    self.prior_logsigma_1 = prior_logsigma_1
    self.prior_logsigma_2 = prior_logsigma_2
    self.mu_initializer = mu_initializer
    self.rho_initializer = rho_initializer

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    in_size = input_shape.with_rank_at_least(2)[-1].value
    self.input_spec = base.InputSpec(min_ndim=2, axes={-1: in_size})
    self.kernel_mu = self.add_variable('posterior_kernel_mu',
                                       shape=[in_size, self.units],
                                       initializer=self.mu_initializer,
                                       dtype=self.dtype,
                                       trainable=True)
    self.kernel_rho = self.add_variable('posterior_kernel_rho',
                                        shape=[in_size, self.units],
                                        initializer=self.rho_initializer,
                                        dtype=self.dtype,
                                        trainable=True)
    if self.use_bias:
      self.bias_mu = self.add_variable('posterior_bias_mu',
                                       shape=[self.units, ],
                                       initializer=init.zeros_initializer(),
                                       dtype=self.dtype,
                                       trainable=True)
      self.bias_rho = self.add_variable('posterior_bias_rho',
                                        shape=[self.units, ],
                                        initializer=self.rho_initializer)
    else:
      self.bias_mu = None
      self.bias_rho = None
    self.built = True

  def _get_scales(self):
    kernel_sigma = tf.nn.softplus(self.kernel_rho)
    if self.clip_std:
      kernel_sigma = tf.clip_by_value(kernel_sigma, 0.,
                                      self.clip_std)

    if self.use_bias:
      bias_sigma = tf.nn.softplus(self.bias_rho)
      if self.clip_std:
        bias_sigma = tf.clip_by_value(bias_sigma, 0., self.clip_std)
    else:
      bias_sigma = None

    return kernel_sigma, bias_sigma

  def _apply_divergence(self):
    kernel_sigma = tf.nn.softplus(self.kernel_rho)
    kernel = reparametrize(self.kernel_mu, kernel_sigma, True)
    kld = kl_mixture(self.kernel_mu,
                     kernel_sigma,
                     tf.exp(self.prior_logsigma_1),
                     tf.exp(self.prior_logsigma_2),
                     self.prior_pi,
                     kernel)

    if self.use_bias:
      bias_sigma = tf.nn.softplus(self.bias_rho)
      bias = reparametrize(self.bias_mu, bias_sigma, True)
      kld += kl_mixture(self.bias_mu,
                        bias_sigma,
                        tf.exp(self.prior_logsigma_1),
                        tf.exp(self.prior_logsigma_2),
                        self.prior_pi,
                        bias)
    self.add_loss(kld)

  def call(self, inputs, stochastic=True):
    kernel_sigma, bias_sigma = self._get_scales()

    if self.local_reparametrization:
      outputs = local_reparametrize(
        inputs, self.kernel_mu, kernel_sigma, stochastic)
    elif self.flipout:
      outputs = flipout_dense(
        self.units, inputs, self.kernel_mu, kernel_sigma, stochastic, self.seed)
    else:
      kernel = reparametrize(self.kernel_mu, kernel_sigma, stochastic)
      outputs = tf.matmul(inputs, kernel)

    self._add_bias(outputs, self.bias_mu, bias_sigma, stochastic)
    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_divergence()
    return outputs


def bbb_dense(
    inputs, units,
    stochastic=True,
    activation=None,
    use_bias=True,
    clip_std=None,
    prior_pi=0.2,
    prior_logsigma_1=-2.0,
    prior_logsigma_2=-5.0,
    mu_initializer=init.random_normal_initializer(0., 1e-2),
    rho_initializer=init.random_uniform_initializer(-9., -6.),
    local_reparametrization=False,
    flipout=False,
    trainable=True,
    name=None,
    reuse=None):
  layer = BayesBackpropDense(units,
                             activation=activation,
                             use_bias=use_bias,
                             clip_std=clip_std,
                             prior_pi=prior_pi,
                             prior_logsigma_1=prior_logsigma_1,
                             prior_logsigma_2=prior_logsigma_2,
                             mu_initializer=mu_initializer,
                             rho_initializer=rho_initializer,
                             local_reparametrization=local_reparametrization,
                             flipout=flipout,
                             trainable=trainable,
                             name=name,
                             dtype=inputs.dtype.base_dtype,
                             _scope=name,
                             _reuse=reuse
                             )
  return layer.apply(inputs, stochastic)


class GroupNJDense(_DenseVariational):
  def __init__(self, units,
               activation=None,
               use_bias=True,
               clip_std=None,
               mu_initializer=init.random_normal_initializer(0., 1e-2),
               logvar_initializer=init.random_normal_initializer(-9., 1e-2),
               z_mu_initializer=init.random_normal_initializer(1, 1e-2),
               epsilon=1e-8,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs):
    super(GroupNJDense, self).__init__(
      units=units,
      activation=activation,
      use_bias=use_bias,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )
    self.clip_std = clip_std
    self.mu_initializer = mu_initializer
    self.logvar_initializer = logvar_initializer
    self.z_mu_initializer = z_mu_initializer
    self.epsilon = epsilon

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    in_size = input_shape.with_rank_at_least(2)[-1].value
    self.input_spec = base.InputSpec(min_ndim=2, axes={-1: in_size})
    self.kernel_mu = self.add_variable('posterior_kernel_mu',
                                       shape=[in_size, self.units],
                                       initializer=self.mu_initializer,
                                       dtype=self.dtype,
                                       trainable=True)
    self.kernel_logvar = self.add_variable('posterior_kernel_logvar',
                                           shape=[in_size, self.units],
                                           initializer=self.logvar_initializer,
                                           dtype=self.dtype,
                                           trainable=True)

    self.z_mu = self.add_variable('posterior_z_mu',
                                  shape=[in_size, ],
                                  initializer=self.z_mu_initializer,
                                  dtype=self.dtype,
                                  trainable=True)

    self.z_logvar = self.add_variable('posterior_z_logvar',
                                      shape=[in_size, ],
                                      initializer=self.logvar_initializer)

    if self.use_bias:
      self.bias_mu = self.add_variable('posterior_bias_mu',
                                       shape=[self.units, ],
                                       initializer=init.zeros_initializer(),
                                       dtype=self.dtype,
                                       trainable=True)
      self.bias_logvar = self.add_variable('posterior_bias_logvar',
                                           shape=[self.units, ],
                                           initializer=self.logvar_initializer)
    else:
      self.bias_mu = None
      self.bias_rho = None
    self.built = True

  def _get_scales(self):
    kernel_sigma = tf.exp(0.5 * self.kernel_logvar)
    if self.clip_std:
      kernel_sigma = tf.nn.clip_tf.clip_by_value(kernel_sigma, 0.,
                                                 self.clip_std)

    if self.use_bias:
      bias_sigma = tf.exp(0.5 * self.bias_logvar)
      if self.clip_std:
        bias_sigma = tf.nn.clip_tf.clip_by_value(bias_sigma, 0., self.clip_std)
    else:
      bias_sigma = None

    return kernel_sigma, bias_sigma

  def _get_log_dropout_rates(self):
    return self.z_logvar - tf.log(
      tf.square(self.z_mu) + self.epsilon)

  def _get_z_batch_tiled(self, batch_size, sample):
    multiples = [batch_size, 1]
    z_sigma = tf.exp(0.5 * self.z_logvar)
    z_mu_tiled = tf.tile(tf.expand_dims(self.z_mu, 0), multiples)
    z_sigma_tiled = tf.tile(tf.expand_dims(z_sigma, 0), multiples)
    return reparametrize(z_mu_tiled, z_sigma_tiled, sample)

  def _apply_divergence(self):
    k1, k2, k3 = 0.63576, 1.87320, 1.48695
    log_alpha = self._get_log_dropout_rates()
    kld = -tf.reduce_sum(
      k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.nn.softplus(
        -log_alpha) - k1)

    kld += tf.reduce_sum(
      -0.5 * self.kernel_logvar + 0.5 * (tf.exp(self.kernel_logvar) +
                                         tf.square(self.kernel_mu)) - 0.5)

    if self.use_bias:
      kld += tf.reduce_sum(
        -0.5 * self.bias_logvar + 0.5 * (tf.exp(self.bias_logvar) +
                                         tf.square(self.bias_mu)) - 0.5)

    self.add_loss(kld)

  def call(self, inputs, stochastic=True):
    kernel_sigma, bias_sigma = self._get_scales()
    batch_size = tf.shape(inputs)[0]

    if self.flipout:
      z = self._get_z_batch_tiled(1, stochastic)
      kernel_mu = z * self.kernel_mu
      kernel_sigma = z * kernel_sigma
      outputs = flipout_dense(
        self.units, inputs, kernel_mu, kernel_sigma, stochastic)
    else:
      z = self._get_z_batch_tiled(batch_size, stochastic)
      outputs = local_reparametrize(
        inputs * z, self.kernel_mu, kernel_sigma, stochastic)

    self._add_bias(outputs, self.bias_mu, bias_sigma, stochastic)
    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_divergence()
    return outputs

class L0NormDense(base.Layer):
  def __init__(self, units,
               activation=None,
               use_bias=True,
               dropout_rate=0.5,
               temperature=0.6,
               gamma=-0.1,
               zeta=1.1,
               kernel_initializer=init.random_normal_initializer(0., 1e-2),
               bias_initializer=init.zeros_initializer(),
               trainable=True,
               name=None,
               **kwargs
               ):
    super(L0NormDense, self).__init__(
      trainable=trainable,
      name=name,
      **kwargs
    )
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.temperature = temperature
    self.gamma = gamma
    self.zeta = zeta
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.epsilon = 1e-8

    # Construct log_alpha initializer.
    alpha = dropout_rate / (1. - dropout_rate)
    self.log_alpha_initializer = init.random_normal_initializer(alpha, 0.01)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.in_size = input_shape.with_rank_at_least(2)[-1].value
    self.input_spec = base.InputSpec(min_ndim=2, axes={-1: self.in_size})
    self.kernel = self.add_variable('kernel',
                                    shape=[self.in_size, self.units],
                                    dtype=self.dtype,
                                    initializer=self.kernel_initializer)
    self.log_alpha = self.add_variable('log_alpha',
                                       shape=[self.in_size, self.units],
                                       dtype=self.dtype,
                                       initializer=self.log_alpha_initializer)

    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units, ],
                                    dtype=self.dtype,
                                    initializer=self.bias_initializer)

  def _get_z(self, stochastic):
    u = tf.random_uniform(shape=(self.self.in_size, self.units),
                          minval=0, maxval=1,
                          dtype=self.dtype)
    if stochastic:
      conc = tf.nn.sigmoid(
        (tf.log(u) - tf.log(
          1. - u) + self.log_alpha) / self.temperature)
    else:
      conc = tf.nn.sigmoid(self.loga)
    conc = conc * (self.zeta - self.gamma) + self.gamma
    return tf.minimum(1., tf.maximum(conc, 0.))

  def _apply_regularization(self):
    cost = self.log_alpha - self.beta * tf.log(-self.gamma / self.zeta)
    l0_cost = tf.reduce_sum(tf.nn.sigmoid(cost))
    self.add_loss(l0_cost)

  def call(self, inputs, stochastic=True):
    z = self._get_z(stochastic)
    kernel = self.kernel * z
    outputs = tf.matmul(inputs, kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_regularization()
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
        'The innermost dimension of input_shape must be defined, but saw: %s'
        % input_shape)
    return input_shape[:-1].concatenate(self.units)
