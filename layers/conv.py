from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops as init

import tensorflow as tf

from .utils import reparametrize, local_reparametrize_conv, flipout_conv
from .utils import kl_mixture


class _L0NormConv(base.Layer):
  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding="valid",
               data_format="channels_last",
               dilation_rate=1,
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
    super(_L0NormConv, self).__init__(
      trainable=trainable,
      name=name,
      **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, "kernel_size")
    self.strides = utils.normalize_tuple(strides, rank, "strides")
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(
      dilation_rate, rank, "dilation_rate")
    self.activation = activation
    self.use_bias = use_bias
    self.dropout_rate = dropout_rate
    self.temperature = temperature
    self.gamma = gamma
    self.zeta = zeta
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    # Construct log_alpha initializer.
    alpha = dropout_rate / (1. - dropout_rate)
    self.log_alpha_initializer = init.random_normal_initializer(alpha, 0.01)

  def _add_bias(self, inputs):
    if not self.use_bias:
      return inputs

    bias = self.bias
    outputs = inputs
    if self.data_format == "channels_first":
      if self.rank == 1:
        # nn.bias_add does not accept a 1D input tensor.
        bias = tf.reshape(bias, (1, self.filters, 1))
        outputs += bias
      if self.rank == 2:
        outputs = tf.nn.bias_add(outputs,
                                 bias,
                                 data_format="NCHW")
      if self.rank == 3:
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients. To use bias_add, we collapse Z
        # and Y into a single dimension to obtain a 4D input tensor.
        outputs_shape = outputs.shape.as_list()
        outputs_4d = tf.reshape(outputs,
                                [outputs_shape[0], outputs_shape[1],
                                 outputs_shape[2] * outputs_shape[3],
                                 outputs_shape[4]])
        outputs_4d = tf.nn.bias_add(outputs_4d,
                                    bias,
                                    data_format="NCHW")
        outputs = tf.reshape(outputs_4d, outputs_shape)
    else:
      outputs = tf.nn.bias_add(outputs,
                               bias,
                               data_format="NHWC")
    return outputs

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    self.kernel_shape = self.kernel_size + (input_dim, self.filters)
    self.kernel = self.add_variable('kernel',
                                    shape=self.kernel_shape,
                                    dtype=self.dtype,
                                    initializer=self.kernel_initializer)
    self.log_alpha = self.add_variable('log_alpha',
                                       shape=self.kernel_shape,
                                       dtype=self.dtype,
                                       initializer=self.log_alpha_initializer)

    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units, ],
                                    dtype=self.dtype,
                                    initializer=self.bias_initializer)
    self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                     axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
      input_shape,
      filter_shape=self.kernel_mu.get_shape(),
      dilation_rate=self.dilation_rate,
      strides=self.strides,
      padding=self.padding.upper(),
      data_format=utils.convert_data_format(self.data_format,
                                            self.rank + 2))
    self.built = True

  def _get_z(self, stochastic):
    u = tf.random_uniform(shape=self.kernel_shape,
                          minval=0, maxval=1,
                          dtype=self.dtype)
    if stochastic:
      conc = tf.nn.sigmoid(
        (tf.log(u) - tf.log(1. - u) + self.log_alpha) / self.temperature)
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
    outputs = self._convolution_op(inputs, kernel)

    if self.use_bias:
      self._add_bias(outputs)

    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_regularization()
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == "channels_last":
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
          space[i],
          self.kernel_size[i],
          padding=self.padding,
          stride=self.strides[i],
          dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tf.TensorShape([input_shape[0]] + new_space +
                            [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
          space[i],
          self.kernel_size[i],
          padding=self.padding,
          stride=self.strides[i],
          dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tf.TensorShape([input_shape[0], self.filters] +
                            new_space)


class L0NormConv1D(_L0NormConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding="valid",
               data_format="channels_last",
               dilation_rate=1,
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
               **kwargs):
    super(L0NormConv1D, self).__init__(
      rank=1,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      dropout_rate=dropout_rate,
      temperature=temperature,
      gamma=gamma,
      zeta=zeta,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      trainable=trainable,
      name=name,
      **kwargs
    )


class L0NormConv2D(_L0NormConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding="valid",
               data_format="channels_last",
               dilation_rate=1,
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
               **kwargs):
    super(L0NormConv2D, self).__init__(
      rank=2,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      dropout_rate=dropout_rate,
      temperature=temperature,
      gamma=gamma,
      zeta=zeta,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      trainable=trainable,
      name=name,
      **kwargs
    )


class _ConvVariational(base.Layer):
  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding="valid",
               data_format="channels_last",
               dilation_rate=1,
               activation=None,
               use_bias=True,
               trainable=True,
               local_reparametrization=False,
               flipout=False,
               seed=None,
               name=None,
               **kwargs
               ):
    super(_ConvVariational, self).__init__(
      trainable=trainable,
      name=name,
      **kwargs)
    if local_reparametrization and flipout:
      raise ValueError(
        'Cannot apply both flipout and local '
        'reparametrizations for variance reduction.')
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, "kernel_size")
    self.strides = utils.normalize_tuple(strides, rank, "strides")
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(
      dilation_rate, rank, "dilation_rate")
    self.activation = activation
    self.use_bias = use_bias
    self.local_reparametrization = local_reparametrization
    self.flipout = flipout
    self.seed = seed

  def _add_bias(self, inputs, loc, scale, sample=True):
    if not self.use_bias:
      return inputs

    bias = reparametrize(loc, scale, sample)
    outputs = inputs
    if self.data_format == "channels_first":
      if self.rank == 1:
        # nn.bias_add does not accept a 1D input tensor.
        bias = tf.reshape(bias, (1, self.filters, 1))
        outputs += bias
      if self.rank == 2:
        outputs = tf.nn.bias_add(outputs,
                                 bias,
                                 data_format="NCHW")
      if self.rank == 3:
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients. To use bias_add, we collapse Z
        # and Y into a single dimension to obtain a 4D input tensor.
        outputs_shape = outputs.shape.as_list()
        outputs_4d = tf.reshape(outputs,
                                [outputs_shape[0], outputs_shape[1],
                                 outputs_shape[2] * outputs_shape[3],
                                 outputs_shape[4]])
        outputs_4d = tf.nn.bias_add(outputs_4d,
                                    bias,
                                    data_format="NCHW")
        outputs = tf.reshape(outputs_4d, outputs_shape)
    else:
      outputs = tf.nn.bias_add(outputs,
                               bias,
                               data_format="NHWC")
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == "channels_last":
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
          space[i],
          self.kernel_size[i],
          padding=self.padding,
          stride=self.strides[i],
          dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tf.TensorShape([input_shape[0]] + new_space +
                            [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
          space[i],
          self.kernel_size[i],
          padding=self.padding,
          stride=self.strides[i],
          dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tf.TensorShape([input_shape[0], self.filters] +
                            new_space)


class _BayesBackpropConv(_ConvVariational):
  def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(_BayesBackpropConv, self).__init__(
      rank=rank,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      local_reparametrization=local_reparametrization,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs)
    self.clip_std = clip_std
    self.prior_pi = prior_pi
    self.prior_logsigma_1 = prior_logsigma_1
    self.prior_logsigma_2 = prior_logsigma_2
    self.kernel_mu_initializer = kernel_mu_initializer
    self.kernel_rho_initializer = kernel_rho_initializer
    self.bias_mu_initializer = bias_mu_initializer
    self.bias_rho_initializer = bias_rho_initializer
    self.input_spec = base.InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)
    self.kernel_mu = self.add_variable('posterior_kernel_mu',
                                       shape=kernel_shape,
                                       initializer=self.kernel_mu_initializer,
                                       trainable=True,
                                       dtype=self.dtype)
    self.kernel_rho = self.add_variable('posterior_kernel_rho',
                                        shape=kernel_shape,
                                        initializer=self.kernel_rho_initializer,
                                        trainable=True,
                                        dtype=self.dtype)

    if self.use_bias:
      self.bias_mu = self.add_variable('posterior_bias_mu',
                                       shape=[self.filters, ],
                                       initializer=self.bias_mu_initializer,
                                       dtype=self.dtype,
                                       trainable=True)
      self.bias_rho = self.add_variable('posterior_bias_rho',
                                        shape=[self.filters, ],
                                        initializer=self.bias_rho_initializer)
    else:
      self.bias_mu = None
      self.bias_rho = None
    self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                     axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
      input_shape,
      filter_shape=self.kernel_mu.get_shape(),
      dilation_rate=self.dilation_rate,
      strides=self.strides,
      padding=self.padding.upper(),
      data_format=utils.convert_data_format(self.data_format,
                                            self.rank + 2))
    self.built = True

  def _get_scales(self):
    kernel_sigma = tf.nn.softplus(self.kernel_rho)
    if self.clip_std:
      kernel_sigma = tf.nn.clip_ops.clip_by_value(kernel_sigma, 0.,
                                                  self.clip_std)

    if self.use_bias:
      bias_sigma = tf.nn.softplus(self.bias_rho)
      if self.clip_std:
        bias_sigma = tf.nn.clip_ops.clip_by_value(bias_sigma, 0., self.clip_std)
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
      outputs = local_reparametrize_conv(
        inputs, self.kernel_mu, kernel_sigma, stochastic, self._convolution_op)
    elif self.flipout:
      outputs = flipout_conv(self.filters, self.rank, inputs, self.kernel_mu,
                             kernel_sigma, stochastic, self._convolution_op,
                             self.seed)
    else:
      kernel = reparametrize(self.kernel_mu, kernel_sigma, stochastic)
      outputs = self._convolution_op(inputs, kernel)

    self._add_bias(outputs, self.bias_mu, bias_sigma, stochastic)
    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_divergence()
    return outputs


class BayesBackpropConv1D(_BayesBackpropConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(BayesBackpropConv1D, self).__init__(
      rank=1,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      clip_std=clip_std,
      prior_pi=prior_pi,
      prior_logsigma_1=prior_logsigma_1,
      prior_logsigma_2=prior_logsigma_2,
      kernel_mu_initializer=kernel_mu_initializer,
      kernel_rho_initializer=kernel_rho_initializer,
      bias_mu_initializer=bias_mu_initializer,
      bias_rho_initializer=bias_rho_initializer,
      local_reparametrization=local_reparametrization,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )


def bbb_conv1d(inputs,
               filters,
               kernel_size,
               stochastic=True,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               reuse=None):
  layer = BayesBackpropConv1D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    data_format=data_format,
    dilation_rate=dilation_rate,
    activation=activation,
    use_bias=use_bias,
    clip_std=clip_std,
    prior_pi=prior_pi,
    prior_logsigma_1=prior_logsigma_1,
    prior_logsigma_2=prior_logsigma_2,
    kernel_mu_initializer=kernel_mu_initializer,
    kernel_rho_initializer=kernel_rho_initializer,
    bias_mu_initializer=bias_mu_initializer,
    bias_rho_initializer=bias_rho_initializer,
    local_reparametrization=local_reparametrization,
    flipout=flipout,
    trainable=trainable,
    seed=seed,
    name=name,
    dtype=inputs.dtype.base_dtype,
    _reuse=reuse,
    _scope=name)
  return layer.apply(inputs, stochastic)


class BayesBackpropConv2D(_BayesBackpropConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(BayesBackpropConv2D, self).__init__(
      rank=2,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      clip_std=clip_std,
      prior_pi=prior_pi,
      prior_logsigma_1=prior_logsigma_1,
      prior_logsigma_2=prior_logsigma_2,
      kernel_mu_initializer=kernel_mu_initializer,
      kernel_rho_initializer=kernel_rho_initializer,
      bias_mu_initializer=bias_mu_initializer,
      bias_rho_initializer=bias_rho_initializer,
      local_reparametrization=local_reparametrization,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )


def bbb_conv2d(inputs,
               filters,
               kernel_size,
               stochastic=True,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               reuse=None):
  layer = BayesBackpropConv2D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    data_format=data_format,
    dilation_rate=dilation_rate,
    activation=activation,
    use_bias=use_bias,
    clip_std=clip_std,
    prior_pi=prior_pi,
    prior_logsigma_1=prior_logsigma_1,
    prior_logsigma_2=prior_logsigma_2,
    kernel_mu_initializer=kernel_mu_initializer,
    kernel_rho_initializer=kernel_rho_initializer,
    bias_mu_initializer=bias_mu_initializer,
    bias_rho_initializer=bias_rho_initializer,
    local_reparametrization=local_reparametrization,
    flipout=flipout,
    trainable=trainable,
    seed=seed,
    name=name,
    dtype=inputs.dtype.base_dtype,
    _reuse=reuse,
    _scope=name)
  return layer.apply(inputs, stochastic)


class BayesBackpropConv3D(_BayesBackpropConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(BayesBackpropConv3D, self).__init__(
      rank=1,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      clip_std=clip_std,
      prior_pi=prior_pi,
      prior_logsigma_1=prior_logsigma_1,
      prior_logsigma_2=prior_logsigma_2,
      kernel_mu_initializer=kernel_mu_initializer,
      kernel_rho_initializer=kernel_rho_initializer,
      bias_mu_initializer=bias_mu_initializer,
      bias_rho_initializer=bias_rho_initializer,
      local_reparametrization=local_reparametrization,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )


def bbb_conv3d(inputs,
               filters,
               kernel_size,
               stochastic=True,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               prior_pi=0.2,
               prior_logsigma_1=-2.0,
               prior_logsigma_2=-5.0,
               kernel_mu_initializer=init.variance_scaling_initializer(),
               kernel_rho_initializer=init.random_normal_initializer(-9., 1e-3),
               bias_mu_initializer=init.random_normal_initializer(0., 1e-3),
               bias_rho_initializer=init.random_normal_initializer(-9., 1e-4),
               local_reparametrization=False,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               reuse=None):
  layer = BayesBackpropConv3D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    data_format=data_format,
    dilation_rate=dilation_rate,
    activation=activation,
    use_bias=use_bias,
    clip_std=clip_std,
    prior_pi=prior_pi,
    prior_logsigma_1=prior_logsigma_1,
    prior_logsigma_2=prior_logsigma_2,
    kernel_mu_initializer=kernel_mu_initializer,
    kernel_rho_initializer=kernel_rho_initializer,
    bias_mu_initializer=bias_mu_initializer,
    bias_rho_initializer=bias_rho_initializer,
    local_reparametrization=local_reparametrization,
    flipout=flipout,
    trainable=trainable,
    seed=seed,
    name=name,
    dtype=inputs.dtype.base_dtype,
    _reuse=reuse,
    _scope=name)
  return layer.apply(inputs, stochastic)


class _GroupNJConv(_ConvVariational):
  def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               mu_initializer=init.random_normal_initializer(0., 1e-2),
               logvar_initializer=init.random_normal_initializer(-9., 1e-2),
               z_mu_initializer=init.random_normal_initializer(1, 1e-2),
               bias_mu_initializer=init.zeros_initializer(),
               epsilon=1e-8,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(_GroupNJConv, self).__init__(
      rank=rank,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
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
    self.bias_mu_initializer = bias_mu_initializer
    self.epsilon = epsilon

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)
    self.kernel_mu = self.add_variable('posterior_kernel_mu',
                                       shape=kernel_shape,
                                       initializer=self.mu_initializer,
                                       trainable=True,
                                       dtype=self.dtype)

    self.kernel_logvar = self.add_variable('posterior_kernel_logvar',
                                           shape=kernel_shape,
                                           initializer=self.logvar_initializer,
                                           trainable=True,
                                           dtype=self.dtype)

    z_dim = input_dim if self.flipout else self.filters
    self.z_mu = self.add_variable('posterior_z_mu',
                                  shape=[z_dim, ],
                                  initializer=self.z_mu_initializer,
                                  dtype=self.dtype,
                                  trainable=True)

    self.z_logvar = self.add_variable('posterior_z_logvar',
                                      shape=[z_dim, ],
                                      initializer=self.logvar_initializer)

    if self.use_bias:
      self.bias_mu = self.add_variable('posterior_bias_mu',
                                       shape=[self.filters, ],
                                       initializer=self.bias_mu_initializer,
                                       dtype=self.dtype,
                                       trainable=True)
      self.bias_logvar = self.add_variable('posterior_bias_logvar',
                                           shape=[self.filters, ],
                                           initializer=self.logvar_initializer)
    else:
      self.bias_mu = None
      self.bias_logvar = None
    self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                     axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
      input_shape,
      filter_shape=self.kernel_mu.get_shape(),
      dilation_rate=self.dilation_rate,
      strides=self.strides,
      padding=self.padding.upper(),
      data_format=utils.convert_data_format(self.data_format,
                                            self.rank + 2))
    self.built = True

  def _get_scales(self):
    kernel_sigma = tf.exp(0.5 * self.kernel_logvar)
    if self.clip_std:
      kernel_sigma = tf.clip_by_value(kernel_sigma, 0., self.clip_std)

    if self.use_bias:
      bias_sigma = tf.exp(0.5 * self.bias_logvar)
      if self.clip_std:
        bias_sigma = tf.clip_by_value(bias_sigma, 0., self.clip_std)
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
      z = tf.expand_dims(self._get_z_batch_tiled(1, stochastic), -1)
      for _ in range(self.rank):
        z = tf.expand_dims(z, 1)
      kernel_mu = z * self.kernel_mu
      kernel_sigma = z * kernel_sigma
      outputs = flipout_conv(self.filters, self.rank, inputs, kernel_mu,
                             kernel_sigma, stochastic, self._convolution_op,
                             self.seed)
    else:
      outputs_mu = self._convolution_op(inputs, self.kernel_mu)
      output_var = self._convolution_op(
        tf.square(inputs), tf.square(kernel_sigma))
      z = self._get_z_batch_tiled(batch_size, stochastic)
      for _ in range(self.rank):
        z = tf.expand_dims(z, 1)
      outputs = reparametrize(
        outputs_mu * z, z * tf.sqrt(output_var), stochastic)

    self._add_bias(outputs, self.bias_mu, bias_sigma, stochastic)
    if self.activation is not None:
      outputs = self.activation(outputs)
    self._apply_divergence()
    return outputs


class GroupNJConv1D(_GroupNJConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               mu_initializer=init.random_normal_initializer(0., 1e-2),
               logvar_initializer=init.random_normal_initializer(-9., 1e-2),
               z_mu_initializer=init.random_normal_initializer(1, 1e-2),
               bias_mu_initializer=init.zeros_initializer(),
               epsilon=1e-8,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(GroupNJConv1D, self).__init__(
      rank=1,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      clip_std=clip_std,
      mu_initializer=mu_initializer,
      logvar_initializer=logvar_initializer,
      z_mu_initializer=z_mu_initializer,
      bias_mu_initializer=bias_mu_initializer,
      epsilon=epsilon,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )


class GroupNJConv2D(_GroupNJConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               mu_initializer=init.random_normal_initializer(0., 1e-2),
               logvar_initializer=init.random_normal_initializer(-9., 1e-2),
               z_mu_initializer=init.random_normal_initializer(1, 1e-2),
               bias_mu_initializer=init.zeros_initializer(),
               epsilon=1e-8,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(GroupNJConv2D, self).__init__(
      rank=2,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      clip_std=clip_std,
      mu_initializer=mu_initializer,
      logvar_initializer=logvar_initializer,
      z_mu_initializer=z_mu_initializer,
      bias_mu_initializer=bias_mu_initializer,
      epsilon=epsilon,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )


class GroupNJConv3D(_GroupNJConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               clip_std=None,
               mu_initializer=init.random_normal_initializer(0., 1e-2),
               logvar_initializer=init.random_normal_initializer(-9., 1e-2),
               z_mu_initializer=init.random_normal_initializer(1, 1e-2),
               bias_mu_initializer=init.zeros_initializer(),
               epsilon=1e-8,
               flipout=False,
               trainable=True,
               seed=None,
               name=None,
               **kwargs
               ):
    super(GroupNJConv3D, self).__init__(
      rank=3,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      clip_std=clip_std,
      mu_initializer=mu_initializer,
      logvar_initializer=logvar_initializer,
      z_mu_initializer=z_mu_initializer,
      bias_mu_initializer=bias_mu_initializer,
      epsilon=epsilon,
      flipout=flipout,
      trainable=trainable,
      seed=seed,
      name=name,
      **kwargs
    )
