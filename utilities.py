from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

import time

##
# Loss functions
##


def loss_np(y_true=None,y_pred=None,loss_mat=None):
	"""
	numpy function to calculate loss from the loss matrix:
	Inputs:
		y_true: true values (N,D)
		y_pred: predicted values (N,D)
		loss_mat: matrix of loss values for selecting outputs (D,D)
		net_loss: True -> same as loss_K, False -> used in optimal decision
	"""


	N,D = np.shape(y_pred)
	# L = np.matmul(y_pred, loss_mat)
	# if len(y_true) != 0:
	# 	A = np.expand_dims(np.matmul(y_pred,loss_mat),1)
	# 	B = np.expand_dims(y_true.T,0)
	# 	B = np.matmul(A,B.T)
	# 	L = B.reshape((N,))
	# else: # For inferring opimal H you treat y_pred as the true values
	A = np.expand_dims(np.matmul(loss_mat,y_pred.T),0)
	R_d = np.zeros_like(y_pred)
	for d in range(D):
		Z = np.zeros_like(y_pred)
		Z[:,d] = 1
		Z = np.expand_dims(Z,1)
		B = np.matmul(Z,A.T)#     Matrix mul for D=12, N = 60,000 is 5000 x slower than for loop
		R_d[:,d] = B.reshape((N,))
		L = R_d
	return L
	# return 0

##
# Optimal H
##

def optimal_h(y_pred_samples, loss_mat, return_risk = False):
	"""
	Calculate the optimal_h
	Inputs:
		y_pred_samples: predicted values (N,D)
		loss_mat: matrix of loss values for selecting outputs (D,D)
	"""
	T,N,D = np.shape(y_pred_samples)
	R_t = np.zeros((N,D)) # Risk
	for t in range(T):
		R_t += loss_np(y_pred = y_pred_samples[t], loss_mat = loss_mat)
	I = np.eye(D)
	H_x = np.argmin(R_t,axis=-1)
	if not return_risk:
		return H_x
	else:
		return H_x, (1./float(T)) * R_t

def sparse_matrix_to_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def utilities_to_tensors(x, sparse = False):
	pass


def load_mnist(n_samples=-1, square=True, conv=False):
    num_classes = 10

    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if square:
        if conv:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols)

    if n_samples != -1:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
