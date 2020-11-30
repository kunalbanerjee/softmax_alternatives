#Kunal Banerjee
#04-Mar-2020
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import argparse
import os
import model_definitions as mdef
#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='cifar10', help='Datasets: mnist, cifar10, cifar100')
  parser.add_argument('--activation', type=str, default='softmax', help='Supported activations: softmax, taylor_softmax, sm_softmax, sm_taylor_softmax, taylor_softmax_grad_finite, taylor_softmax_grad_infinite')
  #Currently, we choose best model based on dataset
  #parser.add_argument('--model', type=str, default='create_cifar10_model')
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--savedir', type=str, default=None)
  parser.add_argument('--order', type=int, default=6, help='Taylor series expansion order (ideally, should be even)')
  parser.add_argument('--margin', type=float, default=0.6, help='Margin for SM_Softmax')
  args = parser.parse_args()
  print("Configuration::")
  print("Dataset: ", args.dataset)
  print("Activation: ", args.activation)
  print("Batch_size: ", args.batch_size) 
  print("No. of epochs: ", args.epochs) 
  print("Order (Taylor softmax): ", args.order) 
  print("Margin (SM softmax): ", args.margin)

  if args.dataset == 'mnist':
    dataset = tf.keras.datasets.mnist
    model   = mdef.create_mnist_model(args.activation, args.order, -1, args.margin)
    opt     = 'adam'
  elif args.dataset == 'cifar10':
    dataset = tf.keras.datasets.cifar10
    model   = mdef.create_cifar10_model(args.activation, args.order, -1, args.margin)
    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  elif args.dataset == 'cifar100':
    dataset = tf.keras.datasets.cifar100
    model   = mdef.create_cifar100_model(args.activation, args.order, -1, args.margin)
    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  else:
    print("Unsupported dataset. Exiting...")
    exit()
 
  (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0

  model.summary()
  model.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  if args.savedir is None:
    checkpoint_path = './checkpoints/training_' + args.dataset + '_ckpt_' + args.activation + '_order_' + str(args.order) + '/cp.ckpt'
  else:
    if not os.path.isdir(args.savedir):
      os.makedirs(args.savedir)
      checkpoint_path = args.savedir + '/cp.ckpt'

  print("Weights will be saved in", checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)
  model.fit(train_images, train_labels, epochs=args.epochs, batch_size=args.batch_size, 
                validation_data=(test_images, test_labels),
                callbacks=[cp_callback])
  loss,acc = model.evaluate(test_images, test_labels, verbose=2)
  print("Model, accuracy: {:5.2f}%".format(100*acc))

main()
print("Training with softmax alternatives done!")
