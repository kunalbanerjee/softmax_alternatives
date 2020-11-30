#Kunal Banerjee
#28-Mar-2020
import tensorflow as tf
import layer_definitions as ldef
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Softmax, ZeroPadding2D, Activation, Dropout, BatchNormalization

def create_mnist_model(activation='softmax', order=4, axis=-1, m=0.6):
  l_input   = Input(shape=(28, 28), name='l_input')
  l_reshaped= tf.reshape(l_input, [-1, 28, 28, 1])
  l_conv1   = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv1').__call__(l_reshaped)
  l_conv21  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv21').__call__(l_conv1) 
  l_conv22  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv22').__call__(l_conv21) 
  l_conv23  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv23').__call__(l_conv22) 
  l_pool1   = MaxPooling2D((2, 2),2, name='l_pool1').__call__(l_conv23)
  l_conv31  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv31').__call__(l_pool1)
  l_conv32  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv32').__call__(l_conv31)
  l_conv33  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv33').__call__(l_conv32)
  l_pool2   = MaxPooling2D((2, 2),2, name='l_pool2').__call__(l_conv33)
  l_conv41  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv41').__call__(l_pool2)
  l_conv42  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv42').__call__(l_conv41)
  l_conv43  = Conv2D(64, (3, 3), padding='SAME', activation='relu', name='l_conv43').__call__(l_conv42)
  l_pool3   = MaxPooling2D((2, 2),2, name='l_pool3').__call__(l_conv43)
  l_flatten = Flatten(name='l_flatten').__call__(l_pool3)
  l_dense1  = Dense(256, activation='relu', name='l_dense1').__call__(l_flatten)
  l_dense2  = Dense(10, name='l_dense2').__call__(l_dense1)
  if activation == 'softmax':
    l_last = Softmax().__call__(l_dense2)
  elif activation == 'taylor_softmax':
    l_last = ldef.TaylorSoftmax().__call__(l_dense2, order=order, axis=axis)
  elif activation == 'taylor_softmax_grad_finite':
    l_last = ldef.taylor_softmax_op(l_dense2, 'finite', order=order, axis=axis)
  elif activation == 'taylor_softmax_grad_infinite':
    l_last = ldef.taylor_softmax_op(l_dense2, 'infinite', order=order, axis=axis)
  elif activation == 'sm_softmax':
    l_last = ldef.sm_softmax_op(l_dense2, m=m)
  elif activation == 'sm_taylor_softmax':
    l_last = ldef.sm_taylor_softmax_op(l_dense2, order=order, m=m)
  else:
    print("Unknown activation. Exiting...")
    exit()
  model = Model(inputs=l_input, outputs=l_last)
  return model


def create_cifar10_model(activation='softmax', order=4, axis=-1, m=0.6):
  l_input = Input(shape=(32, 32, 3), name='l_input')
  l_conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same').__call__(l_input)
  l_norm1 = BatchNormalization().__call__(l_conv1)
  l_conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same').__call__(l_norm1)
  l_norm2 = BatchNormalization().__call__(l_conv2)
  l_pool1 = MaxPooling2D((2, 2)).__call__(l_norm2)
  l_drop1 = Dropout(0.2).__call__(l_pool1)
  l_conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same').__call__(l_drop1)
  l_norm3 = BatchNormalization().__call__(l_conv3)
  l_conv4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same').__call__(l_norm3)
  l_norm4 = BatchNormalization().__call__(l_conv4)
  l_pool2 = MaxPooling2D((2, 2)).__call__(l_norm4)
  l_drop2 = Dropout(0.3).__call__(l_pool2)
  l_conv5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same').__call__(l_drop2)
  l_norm5 = BatchNormalization().__call__(l_conv5)
  l_conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same').__call__(l_norm5)
  l_norm6 = BatchNormalization().__call__(l_conv6)
  l_pool3 = MaxPooling2D((2, 2)).__call__(l_norm6)
  l_drop3 = Dropout(0.4).__call__(l_pool3)
  l_flat1 = Flatten().__call__(l_drop3)
  l_dense1= Dense(128, activation='relu', kernel_initializer='he_uniform').__call__(l_flat1)
  l_norm7 = BatchNormalization().__call__(l_dense1)
  l_drop4 = Dropout(0.5).__call__(l_norm7)
  l_dense2= Dense(10).__call__(l_drop4)
  if activation == 'softmax':
    l_last = Softmax().__call__(l_dense2)
  elif activation == 'taylor_softmax':
    l_last = ldef.TaylorSoftmax().__call__(l_dense2, order=order, axis=axis)
  elif activation == 'taylor_softmax_grad_finite':
    l_last = ldef.taylor_softmax_op(l_dense2, 'finite', order=order, axis=axis)
  elif activation == 'taylor_softmax_grad_infinite':
    l_last = ldef.taylor_softmax_op(l_dense2, 'infinite', order=order, axis=axis)
  elif activation == 'sm_softmax':
    l_last = ldef.sm_softmax_op(l_dense2, m=m)
  elif activation == 'sm_taylor_softmax':
    l_last = ldef.sm_taylor_softmax_op(l_dense2, order=order, m=m)
  else:
    print("Unknown activation. Exiting...")
    exit()
  model = Model(inputs=l_input, outputs=l_last)
  return model


def create_cifar100_model(activation='softmax', order=4, axis=-1, m=0.6):
  l_input = Input(shape=(32, 32, 3), name='l_input')
  l_pad   = ZeroPadding2D(4).__call__(l_input)
  l_conv1 = Conv2D(384, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)).__call__(l_pad)
  l_elu1  = Activation('elu').__call__(l_conv1)
  l_pool1 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu1)
  l_drop1 = Dropout(0.5).__call__(l_pool1)
  
  l_conv21= Conv2D(384, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_drop1)
  l_conv22= Conv2D(384, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv21)
  l_conv23= Conv2D(640, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv22)
  l_conv24= Conv2D(640, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv23)
  l_elu2  = Activation('elu').__call__(l_conv24)
  l_pool2 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu2)
  l_drop2 = Dropout(0.5).__call__(l_pool2)
  
  l_conv31= Conv2D(640, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_drop2)
  l_conv32= Conv2D(768, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv31)
  l_conv33= Conv2D(768, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv32)
  l_conv34= Conv2D(768, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv33)
  l_elu3  = Activation('elu').__call__(l_conv34)
  l_pool3 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu3)
  l_drop3 = Dropout(0.5).__call__(l_pool3)
  
  l_conv41= Conv2D(768, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_drop3)
  l_conv42= Conv2D(896, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv41)
  l_conv43= Conv2D(896, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv42)
  l_elu4  = Activation('elu').__call__(l_conv43)
  l_pool4 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu4)
  l_drop4 = Dropout(0.5).__call__(l_pool4)
  
  l_conv51= Conv2D(896, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_drop4)
  l_conv52= Conv2D(1024, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv51)
  l_conv53= Conv2D(1024, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv52)
  l_elu5  = Activation('elu').__call__(l_conv53)
  l_pool5 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu5)
  l_drop5 = Dropout(0.5).__call__(l_pool5)
  
  l_conv61= Conv2D(1024, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_drop5)
  l_conv62= Conv2D(1152, (2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_conv61)
  l_elu6  = Activation('elu').__call__(l_conv62)
  l_pool6 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu6)
  l_drop6 = Dropout(0.5).__call__(l_pool6)
  
  l_conv71= Conv2D(1152, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)).__call__(l_drop6)
  l_elu7  = Activation('elu').__call__(l_conv71)
  l_pool7 = MaxPooling2D(pool_size=(2, 2), padding='same').__call__(l_elu7)
  l_drop7 = Dropout(0.5).__call__(l_pool7)
  
  l_flat  = Flatten().__call__(l_drop7)
  l_dense2= Dense(100).__call__(l_flat)
  if activation == 'softmax':
    l_last = Softmax().__call__(l_dense2)
  elif activation == 'taylor_softmax':
    l_last = ldef.TaylorSoftmax().__call__(l_dense2, order=order, axis=axis)
  elif activation == 'taylor_softmax_grad_finite':
    l_last = ldef.taylor_softmax_op(l_dense2, 'finite', order=order, axis=axis)
  elif activation == 'taylor_softmax_grad_infinite':
    l_last = ldef.taylor_softmax_op(l_dense2, 'infinite', order=order, axis=axis)
  elif activation == 'sm_softmax':
    l_last = ldef.sm_softmax_op(l_dense2, m=m)
  elif activation == 'sm_taylor_softmax':
    l_last = ldef.sm_taylor_softmax_op(l_dense2, order=order, m=m)
  else:
    print("Unknown activation. Exiting...")
    exit()
  model = Model(inputs=l_input, outputs=l_last)
  return model

