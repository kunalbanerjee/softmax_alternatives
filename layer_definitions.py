#Kunal Banerjee
#28-Mar-2020
import tensorflow as tf
import os

def taylor_exp(x, order):
  x_shape= tf.shape(x)
  temp   = tf.ones(x_shape)
  result = tf.ones(x_shape)
  denom  = tf.ones(x_shape)
  for i in range(1, int(order + 1)):
    temp   = tf.multiply(temp, x)
    iteri  = tf.fill(x_shape, i)
    iteri  = tf.cast(iteri, tf.float32)
    denom  = tf.multiply(denom, iteri)
    result = result + (temp / denom)

  return result


@tf.custom_gradient
def taylor_softmax_op(inputs, grad_type='finite', order=4, axis=-1):
  if order <= 0:
    print("Taylor expansion order ", order, " should be >= 1")
    exit()

  expval = taylor_exp(inputs, order)
  sumval = tf.math.reduce_sum(expval, axis)
  #Broadcast the denominator for each class
  #denom = tf.repeat(sumval, tf.shape(inputs)[axis], axis=axis)
  denom = tf.tile(sumval, multiples=[tf.shape(inputs)[axis]])
  denom = tf.reshape(denom, tf.shape(inputs))
  result = tf.math.divide(expval, denom)
   
  def custom_grad(dy):
    if grad_type == 'finite':
      grad_taylor_exp = taylor_exp(inputs, order-1)
      grad = tf.math.multiply(dy, grad_taylor_exp)
    else:
      grad = tf.math.multiply(dy, result)
    return grad
  
  return result, custom_grad


class TaylorSoftmax(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(TaylorSoftmax, self).__init__(**kwargs)

  def call(self, inputs, order=4, axis=-1):
    if order <= 0:
      print("Taylor expansion order ", order, " should be >= 1")
      exit()

    expval = taylor_exp(inputs, order)
    sumval = tf.math.reduce_sum(expval, axis=axis)
    #Broadcast the denominator for each class
    #denom = tf.repeat(sumval, tf.shape(inputs)[axis], axis=axis)
    denom = tf.tile(sumval, multiples=[tf.shape(inputs)[axis]])
    denom = tf.reshape(denom, tf.shape(inputs))
    result = tf.math.divide(expval, denom)
    return result
    #Expects auto gradient

  def get_config(self):
    config = super(TaylorSoftmax, self).get_config()
    return config


@tf.custom_gradient
def sm_softmax_op(inputs, m=0.6):
  if m < 0:
    print("Soft margin ", m, " should be >= 0")
    exit()

  result = tf.zeros(tf.shape(inputs))
  exponent_tensor = tf.exp(inputs)
  sum_of_exponents = tf.reduce_sum(exponent_tensor)
  temp = tf.fill(tf.shape(inputs), m)
  temp = tf.exp(temp)
  numerator = tf.divide(exponent_tensor, temp)
  denominator = tf.subtract(tf.add(numerator, sum_of_exponents), exponent_tensor)
  result = tf.divide(numerator, denominator)
   
  def custom_grad(dy):
    first_term = tf.multiply(result, dy)
    temp1 = tf.reduce_sum(first_term)
    temp2 = tf.broadcast_to(temp1, tf.shape(result))
    second_term = tf.multiply(temp2, result)
    return first_term - second_term

  return result, custom_grad


@tf.custom_gradient
def sm_taylor_softmax_op(inputs, order=4, m=0.6):
  if m < 0:
    print("Soft margin ", m, " should be >= 0")
    exit()
  if order <= 0:
    print("Taylor expansion order ", order, " should be >= 1")
    exit()

  result = tf.zeros(tf.shape(inputs))
  exponent_tensor = taylor_exp(inputs, order)
  sum_of_exponents = tf.reduce_sum(exponent_tensor)
  temp = tf.fill(tf.shape(inputs), m)
  temp = taylor_exp(temp, order)
  numerator = tf.divide(exponent_tensor, temp)
  denominator = tf.subtract(tf.add(numerator, sum_of_exponents), exponent_tensor)
  result = tf.divide(numerator, denominator)
   
  def custom_grad(dy):
    first_term = tf.multiply(result, dy)
    temp1 = tf.reduce_sum(first_term)
    temp2 = tf.broadcast_to(temp1, tf.shape(result))
    second_term = tf.multiply(temp2, result)
    return first_term - second_term

  return result, custom_grad
