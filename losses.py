from . import backend as bkd
from . import config
from .backend import tf
import numpy as np
import scipy.io
import pandas as pd
from scipy.io import loadmat
import paddle
from . import my_custom_loss_2
import scipy.sparse as sp
#import tensorflow as tf
from scipy.sparse import coo_matrix
def mean_absolute_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)


def mean_squared_error(y_true, y_pred, input_loads):

    return bkd.reduce_mean(bkd.square(y_true - y_pred))


def softmax_cross_entropy(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def zero(*_):
    # TODO: pytorch
    return tf.constant(0, dtype=config.real(tf))


def DD(y_true, y_pred,input_loads):
    Loss=bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))
    return Loss


mat_data = loadmat('./Data/Load_Nodes.mat')
Load_nodes = np.squeeze(mat_data['Load_nodes'])
mat_data = loadmat('./Data/global_stiffness.mat')
global_stiffness = np.squeeze(mat_data['Stiffness_Matrix'])
Load_nodes = Load_nodes.astype(np.int32)
Load_nodes_zero_based = Load_nodes - 1


def DD_EC(y_true, y_pred,input_loads):
    Pred = (y_pred / 1e6) # Scale Down to original for PI
    Disp = Pred
    Loadd = bkd.from_numpy(-1e-6 * input_loads[0]) # Scale down to have almost equal contribution to DD loss (MegaNewton)
    Disp_f_STNEng = bkd.reshape(Disp[:], (len(input_loads[0]), -1)).T
    Strain_Energy = bkd.matmul(bkd.matmul(Disp_f_STNEng.T, bkd.from_numpy(1e-6*global_stiffness)), Disp_f_STNEng).diagonal() # Scale down to have almost equal contribution to DD loss
    Disp_for_WD = Disp[:,Load_nodes_zero_based, :]
    Work_Done = bkd.from_numpy(np.diagonal(np.dot(Loadd, Disp_for_WD[:, :, 1].T)))
    Work_Done.stop_gradient = False
    Data_LOSS = bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))
    PI_LOSS1 = bkd.reduce_mean(bkd.norm(Strain_Energy - Work_Done))
    Total_loss=1*PI_LOSS1+1*Data_LOSS
    return Total_loss

LOSS_DICT = {
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    "mean absolute percentage error": mean_absolute_percentage_error,
    "MAPE": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "softmax cross entropy": softmax_cross_entropy,
    "zero": zero,
    "DD": DD,
    "DD_EC": DD_EC,
}


def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
