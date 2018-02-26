import tensorflow as tf

from tensorflow.python.ops.distributions import util as distribution_util


def kl_mixture(mu, sigma, prior_sigma1, prior_sigma2, prior_pi, sample):
  x_flat = tf.reshape(sample, [-1])
  mu = tf.reshape(mu, [-1])
  sigma = tf.reshape(sigma, [-1])
  posterior = tf.distributions.Normal(mu, sigma)
  log_posterior = tf.reduce_sum(posterior.log_prob(x_flat))
  N1 = tf.distributions.Normal(0.0, prior_sigma1)
  N2 = tf.distributions.Normal(0.0, prior_sigma2)
  prior1 = tf.log(prior_pi) + N1.log_prob(x_flat)
  prior2 = tf.log(1.0 - prior_pi) + N2.log_prob(x_flat)
  prior_mix = tf.stack([prior1, prior2])
  log_prior = tf.reduce_sum(tf.reduce_logsumexp(prior_mix, [0]))
  return log_posterior - log_prior


def reparametrize(loc, scale, sample):
  output = loc
  if sample:
    output += scale * tf.random_normal(tf.shape(scale), 0, 1)
  return output


def bernoullli(shape, p=0.5):
  return tf.where(
    tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))


def local_reparametrize(inputs, loc, scale, sample):
  outputs_loc = tf.matmul(inputs, loc)
  outputs_var = tf.matmul(
    tf.square(inputs), tf.square(scale))
  return reparametrize(
    outputs_loc, tf.sqrt(outputs_var), sample)


def local_reparametrize_conv(inputs, loc, scale, sample, conv_op):
  # Note: Gradient will be biased due to weight sharing,
  # as inputs are no longer independent.
  outputs_mu = conv_op(inputs, loc)
  outputs_var = conv_op(
    tf.square(inputs), tf.square(scale))
  return reparametrize(
    outputs_mu, tf.sqrt(outputs_var), sample)


def flipout_dense(units, inputs, loc, scale, sample, seed=None):
  outputs = tf.matmul(inputs, loc)
  if sample:
    kernel_noise = reparametrize(0., scale, sample=True)
    input_shape = tf.shape(inputs)
    batch_shape = input_shape[:-1]

    sign_input = random_sign(input_shape, dtype=inputs.dtype,
                             seed=seed)
    sign_output = random_sign(
      tf.concat([batch_shape,
                 tf.expand_dims(units, 0)], 0),
      dtype=inputs.dtype,
      seed=distribution_util.gen_new_seed(
        seed, salt="dense_flipout"))

    perturbed_inputs = tf.matmul(
      inputs * sign_input, kernel_noise) * sign_output
    outputs += perturbed_inputs
  return outputs


def flipout_conv(filters, rank, inputs, loc, scale, sample, conv_op, seed=None):
  outputs = conv_op(inputs, loc)
  if sample:
    kernel_noise = reparametrize(0., scale, sample=True)
    input_shape = tf.shape(inputs)
    output_shape = tf.shape(outputs)
    batch_shape = tf.expand_dims(input_shape[0], 0)
    channels = input_shape[-1]

    sign_input = random_sign(
      tf.concat([batch_shape,
                 tf.expand_dims(channels, 0)], 0),
      dtype=inputs.dtype,
      seed=seed)
    sign_output = random_sign(
      tf.concat([batch_shape,
                 tf.expand_dims(filters, 0)], 0),
      dtype=inputs.dtype,
      seed=distribution_util.gen_new_seed(
        seed, salt="conv_flipout"))
    for _ in range(rank):
      sign_input = tf.expand_dims(sign_input,
                                  1)
      sign_output = tf.expand_dims(sign_output, 1)

    sign_input = tf.tile(
      sign_input,
      [1] + [input_shape[i + 1] for i in range(rank)] + [1])
    sign_output = tf.tile(
      sign_output,
      [1] + [output_shape[i + 1] for i in range(rank)] + [1])

    perturbed_inputs = conv_op(
      inputs * sign_input, kernel_noise) * sign_output

    outputs += perturbed_inputs
  return outputs


def random_sign(shape, dtype=tf.float32, seed=None):
  """Draw values from {-1, 1} uniformly, i.e., Rademacher distribution."""
  random_bernoulli = tf.random_uniform(shape, minval=0, maxval=2,
                                       dtype=tf.int32,
                                       seed=seed)
  return tf.cast(2 * random_bernoulli - 1, dtype)


def zeros_d(shape):
  if isinstance(shape, (list, tuple)):
    shape = tf.stack(shape)
  return tf.zeros(shape)
