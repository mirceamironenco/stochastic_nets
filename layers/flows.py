import tensorflow as tf
import numpy as np

from .utils import bernoullli, zeros_d
from tensorflow.contrib.layers import variance_scaling_initializer


class PlanarFlow(object):
  """
  https://arxiv.org/pdf/1505.05770.pdf
  """

  def __init__(self, fan_in, n_flows=2, scope='planar_flow'):
    self.fan_in = fan_in
    self.n_flows = n_flows
    self.scope = scope
    self.params = []
    self.build()

  def build(self):
    with tf.variable_scope(self.scope):
      for flow in range(self.n_flows):
        w = tf.get_variable('w_{}'.format(flow), (self.fan_in, 1), tf.float32,
                            tf.random_normal_initializer(0., 1e-3))
        u = tf.get_variable('u_{}'.format(flow), (self.fan_in, 1), tf.float32,
                            tf.random_normal_initializer(0., 1e-3))
        b = tf.get_variable('b_{}'.format(flow),
                            initializer=tf.zeros_initializer())
        self.params.append((w, u, b))

  def apply_flow(self, z):
    logdets = tf.zeros((tf.shape(z)[0],))
    for curr_flow in range(self.n_flows):
      w, u, b = self.params[curr_flow]
      uw = tf.matmul(tf.transpose(w), u)
      muw = -1 + tf.nn.softplus(uw)
      u_hat = u + (muw - uw) * w / tf.reduce_sum(tf.square(w))
      if len(z.get_shape()) == 1:
        zwb = z * w + b
      else:
        zwb = tf.matmul(z, w) + b
      psi = tf.matmul(1 - tf.square(tf.nn.tanh(zwb)), tf.transpose(w))
      psi_u = tf.matmul(psi, u_hat)
      logdets += tf.squeeze(tf.log(tf.abs(1 + psi_u)))
      z += tf.matmul(tf.nn.tanh(zwb), tf.transpose(u_hat))
    return z, logdets


class NVPMasked(object):
  """
  https://arxiv.org/pdf/1605.08803.pdf
  """

  def __init__(self, fan_in, n_flows=2, n_hidden=50,
               hidden_size=10, scope='NVP_flow', nonlin=tf.nn.tanh):
    self.fan_in = fan_in
    self.n_flows = n_flows
    self.n_hidden = n_hidden
    self.hidden_size = hidden_size
    self.nonlin = nonlin
    self.scope = scope
    self.params = []
    self.build()

  def build(self):
    with tf.variable_scope(self.scope):
      for flow in range(self.n_flows):
        W = tf.get_variable('w{}_{}'.format(0, flow),
                            (self.fan_in, self.hidden_size), tf.float32,
                            tf.random_normal_initializer(0., 1e-3))
        b = tf.get_variable('b{}_{}'.format(0, flow),
                            (self.hidden_size,), tf.float32,
                            initializer=tf.zeros_initializer())
        self.params.append([(W, b)])

        for curr_h in range(self.n_hidden):
          Wh = tf.get_variable('wh{}_{}'.format(curr_h + 1, flow),
                               (self.hidden_size, self.hidden_size), tf.float32,
                               tf.random_normal_initializer(0., 1e-2))
          bh = tf.get_variable('bh{}_{}'.format(curr_h + 1, flow),
                               (self.hidden_size,), tf.float32,
                               initializer=tf.zeros_initializer())
          self.params[-1].append((Wh, bh))

        Wo = tf.get_variable('wo_{}_{}'.format(self.n_hidden, flow + 1),
                             (self.hidden_size, self.fan_in), tf.float32,
                             tf.random_normal_initializer(0., 1e-2))
        bo = tf.get_variable('bo_{}_{}'.format(self.n_hidden, flow + 1),
                             shape=(self.fan_in,),
                             initializer=tf.zeros_initializer())

        Ws = tf.get_variable('ws_{}_{}'.format(self.n_hidden, flow + 1),
                             (self.hidden_size, self.fan_in), tf.float32,
                             tf.random_normal_initializer(0., 1e-2))
        bs = tf.get_variable('bs_{}_{}'.format(self.n_hidden, flow + 1),
                             shape=(self.fan_in,),
                             initializer=tf.constant_initializer(2.,
                                                                 tf.float32))
        self.params[-1].append((Wo, bo, Ws, bs))

  def get_mu_sigma(self, x, flow):
    curr_params = self.params[flow]
    h = [x]
    for l in range(len(curr_params[:-1])):
      w, b = curr_params[l][0], curr_params[l][1]
      curr_h = tf.matmul(h[-1], w) + b
      h.append(self.nonlin(curr_h))

    Wo, bo, Ws, bs = curr_params[-1]
    mu = tf.matmul(h[-1], Wo) + bo
    sigma = tf.matmul(h[-1], Ws) + bs
    return mu, sigma

  def apply_flow(self, z, deterministic=False, **kwargs):
    log_det = tf.zeros([tf.shape(z)[0]])
    z_f = z
    for curr_flow in range(self.n_flows):
      # m ~ Bern(0.5)

      mask = .5 if deterministic else bernoullli(tf.shape(z), p=0.5)

      # mu = g(h), sigma = k(h)
      mu, sigma = self.get_mu_sigma(mask * z, curr_flow)

      # sigma = sigmoid(k(h))
      sigma = tf.nn.sigmoid(sigma)

      # log |d zt+1 / d z_t| = (1-m)^T * log_sigma
      log_det += tf.reduce_sum((1 - mask) * tf.log(sigma), axis=1)

      # m * z + (1 - m) * (z * sigma + (1 - sigma) * mu)
      z_f = mask * z + (1. - mask) * (z * sigma + (1. - sigma) * mu)

    return z_f, log_det


class IAFFlow(object):
  def __init__(self, incoming, n_flows=2, n_hidden=0, dim_h=50, name=None,
               scope=None, nonlin=tf.nn.elu, **kwargs):
    self.incoming = incoming
    self.n_flows = n_flows
    self.n_hidden = n_hidden
    self.name = name
    self.dim_h = dim_h
    self.params = []
    self.nonlin = nonlin
    self.scope = scope
    self.build()

  def build_mnn(self, fid, param_list):
    dimin = self.incoming
    with tf.variable_scope(self.scope):
      w = tf.get_variable('w{}_{}_{}'.format(0, self.name, fid),
                          shape=(dimin, self.dim_h),
                          initializer=variance_scaling_initializer())
      b = tf.get_variable('b{}_{}_{}'.format(0, self.name, fid),
                          shape=(self.dim_h,),
                          initializer=tf.constant_initializer(0.))
      mask = self.build_mask(dimin, self.dim_h, False)
      param_list.append([(mask, w, b)])
      for l in range(self.n_hidden):
        wh = tf.get_variable('w{}_{}_{}'.format(l + 1, self.name, fid),
                             shape=(self.dim_h, self.dim_h),
                             initializer=variance_scaling_initializer())
        bh = tf.get_variable('b{}_{}_{}'.format(l + 1, self.name, fid),
                             shape=(self.dim_h,),
                             initializer=tf.constant_initializer(0.))
        mask = self.build_mask(self.dim_h, self.dim_h, False)
        param_list[-1].append((mask, wh, bh))

      wout = tf.get_variable(
        'w{}_{}_{}'.format(self.n_hidden + 1, self.name, fid),
        shape=(self.dim_h, dimin),
        initializer=variance_scaling_initializer())
      bout = tf.get_variable(
        'b{}_{}_{}'.format(self.n_hidden + 1, self.name, fid),
        shape=(dimin,),
        initializer=tf.constant_initializer(0.))
      mask1 = self.build_mask(self.dim_h, dimin, True)
      wout2 = tf.get_variable(
        'w{}_{}_{}_sigma'.format(self.n_hidden + 1, self.name, fid),
        shape=(self.dim_h,
               dimin),
        initializer=variance_scaling_initializer())
      bout2 = tf.get_variable(
        'b{}_{}_{}_sigma'.format(self.n_hidden + 1, self.name, fid),
        shape=(dimin,),
        initializer=tf.constant_initializer(0.))
      mask2 = self.build_mask(self.dim_h, dimin, True)
      param_list[-1].append((mask1, wout, bout, mask2, wout2, bout2))

  def build_mask(self, n_in, n_out, diagonalzeros):
    assert n_in % n_out == 0 or n_out % n_in == 0, 'n_in {}, n_out'

    mask = np.ones((n_in, n_out), dtype=np.float32)
    if n_out >= n_in:
      k = n_out // n_in
      for i in range(n_in):
        mask[i + 1:, i * k:(i + 1) * k] = 0
        if diagonalzeros:
          mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
      k = n_in // n_out
      for i in range(n_out):
        mask[(i + 1) * k:, i:i + 1] = 0
        if diagonalzeros:
          mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask

  def build(self):
    for flow in range(self.n_flows):
      self.build_mnn('muf_{}'.format(flow), self.params)

  def ff(self, x, weights):
    inputs = [x]
    for j in range(len(weights[:-1])):
      h = tf.matmul(inputs[-1], weights[j][0] * weights[j][1]) + weights[j][2]
      inputs.append(self.nonlin(h))
    maskmu, wmu, bmu, masksigma, wsigma, bsigma = weights[-1]
    mean = tf.matmul(inputs[-1], maskmu * wmu) + bmu
    sigma = tf.matmul(inputs[-1], masksigma * wsigma) + bsigma
    return mean, sigma

  def get_output_for(self, z, **kwargs):
    logdets = zeros_d((tf.shape(z)[0],))
    for flow in range(self.n_flows):
      if (flow + 1) % 2 == 0:
        z = tf.reverse(z, [1])
      ggmu, ggsigma = self.ff(z, self.params[flow])

      # gate = tf.nn.softplus(ggsigma)
      # z = ggmu + gate * z

      gate = tf.nn.sigmoid(ggsigma)
      z = gate * z + (1 - gate) * ggmu

      logdets += tf.reduce_sum(tf.log(gate), axis=1)

    return z, logdets
