import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from layers.dense import bbb_dense
from layers.conv import bbb_conv2d


def mnist():
  from tensorflow.examples.tutorials.mnist import input_data
  m = input_data.read_data_sets(
    'mnist', one_hot=True, validation_size=10000, reshape=False)
  train, valid, test = m.train, m.validation, m.test
  return (train.images, train.labels), (valid.images, valid.labels), (
    test.images, test.labels)


# Load data and setup graph inputs
(x_train, y_train), _, (x_test, y_test) = mnist()
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.int32, [None, 10])


def make_nn(input):
  # Example of BBB network with stochastic forward passes.
  net = bbb_conv2d(input, 20, 5, stochastic=True, activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net, 2, 2, 'same')
  net = bbb_conv2d(net, 50, 5, stochastic=True, activation=tf.nn.relu)
  net = tf.layers.flatten(tf.layers.max_pooling2d(net, 2, 2, 'same'))
  net = bbb_dense(net, 500, stochastic=True, activation=tf.nn.relu)
  return bbb_dense(net, 10, stochastic=True)


logit = make_nn(x)
pred = tf.nn.softmax(logit)
accuracy = tf.reduce_mean(
  tf.to_float(tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))))

# Minimize the negative ELBO
nll = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
kl = tf.reduce_sum(tf.losses.get_regularization_losses()) / x_train.shape[0]
neg_elbo = nll + kl
optimizer = tf.train.AdamOptimizer()
update_op = optimizer.minimize(neg_elbo)

# Train for 100 epochs
session = tf.Session()
session.run(tf.global_variables_initializer())
ds_idx = np.arange(x_train.shape[0])
for _ in tqdm(range(100), ncols=100):
  np.random.shuffle(ds_idx)
  for _ in range(int(x_train.shape[0] / 128)):
    batch_idx = np.random.choice(ds_idx, 128)
    fd = {x: x_train[batch_idx], y: y_train[batch_idx]}
    session.run(update_op, fd)

# Obtain accuracy with 100 samples in batches of 1000
predictions = np.zeros(shape=(x_test.shape[0], 10))
for sample_i in range(100):
  for batch_i in range(x_test.shape[0] // 1000):
    st, end = 1000 * batch_i, 1000 * (batch_i + 1)
    preds = session.run(pred, {x: x_test[st:end]})
    predictions[st:end] += preds / 100

pred_accuracy = np.mean(
  np.equal(np.argmax(predictions, axis=1), np.argmax(y_test, 1)))

print('Predictive accuracy {}'.format(pred_accuracy))


# Predictive uncertainty on a rotated image of the digit 3
# Select an MNIST digit and rotate it

def rotate(angle):
  test_img = x_test[18]
  img = scipy.ndimage.rotate(test_img.reshape((28, 28)), angle, reshape=False)
  return img.reshape((1, -1))


def predict(img):
  mc_steps = 100
  probs_raw = np.zeros((mc_steps, len(img), 10))
  for i in range(mc_steps):
    probs_raw[i] = session.run(pred, feed_dict={x: img.reshape(1, 28, 28, 1)})
  np_probs = np.mean(probs_raw, 0)
  return np_probs, probs_raw


def plot():
  plt.switch_backend('agg')
  df = pd.DataFrame(columns=['prob', 'out', 'angle'])
  for degree in range(10):
    p, r = predict(rotate(degree * 10))
    for n in range(10):
      new_df = pd.DataFrame(columns=['prob', 'out', 'angle'],
                            data=list(zip(r[:, 0, n], [n + 1] * len(r),
                                          [degree * 10] * len(r))))
      df = pd.concat([df, new_df])

  df['unit'] = list(range(100)) * 10 * 10
  df['Prediction'] = [int(f - 1) for f in df['out']]

  plt.figure(figsize=(15, 10))
  sns.tsplot(df, time='angle', value='prob', condition='Prediction',
             unit='unit',
             err_style="ci_bars", interpolate=False)
  plt.ylim(-0.05, 1.05)
  plt.xticks([10 * f for f in range(10)])
  sns.despine()
  plt.ylabel('Softmax probability')
  plt.xlabel('Rotation angle')
  plt.show()
  plt.savefig('mnist.png')


plot()
