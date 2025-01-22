from . import backend as bkd
from . import config
from .backend import tf
import numpy as np
import scipy.io
import pandas as pd
from scipy.io import loadmat
import tensorflow as tf

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

mat_data = loadmat('./Stiffness_Matrix/NonZero_Stiffness.mat')
Stiff_all2=np.squeeze(mat_data['K_sub'])
Stiff_all2=Stiff_all2*1e-6
mat_data = loadmat('./Stiffness_Matrix/NonZeroColumn.mat')
idxs=np.squeeze(mat_data['NonZeroCols2'])
indices_tf2 = tf.constant(idxs, dtype=tf.int32)
def DD_EC_Master(y_true, y_pred,input_loads):
    Actual = (y_true / 1e6)
    Pred = (y_pred / 1e6)
    Disp = Pred
    zeros_tensor = tf.zeros_like(Disp[:, :, :1])
    modified_tensor = tf.concat([Disp[:, :, :4], zeros_tensor, Disp[:, :, 4:]], axis=2)
    Disp=modified_tensor
    Loadd = bkd.from_numpy(1e-6 * input_loads[0])

    Disp_STNEng_All_Disp = tf.reshape(Disp[:], (len(input_loads[0]), -1))

    Disp_STNEng_reduced = tf.gather(Disp_STNEng_All_Disp, indices_tf2, axis=1)


    Disp_STNEng_reduced_Trans = tf.transpose(Disp_STNEng_reduced)

    Strain_Energy_All = tf.linalg.diag_part(
            tf.matmul(tf.matmul(Disp_STNEng_reduced, Stiff_all2), Disp_STNEng_reduced_Trans))

    Work_Done_all = tf.linalg.diag_part(tf.matmul(Loadd, tf.transpose(Disp[:, 1726:, 1])))

    Work_Done_all.stop_gradient = False
    PI_LOSS1 = bkd.reduce_mean(bkd.norm(Strain_Energy_All - Work_Done_all))
    Data_LOSS = bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))
    Total_loss=1*Data_LOSS+1*PI_LOSS1


    return Total_loss

mat_data = loadmat('./Stiffness_Matrix/LHS_RHS_for_KII_KNN_less.mat')
LHS=np.squeeze(mat_data['LHS'])
RHS=np.squeeze(mat_data['RHS'])
LHS=LHS*1e-6 # Scale down to have almost equal contribution to DD loss (MegaNewton)
RHS=RHS*1e-6 # Scale down to have almost equal contribution to DD loss (MegaNewton)
mat_data = loadmat('./Stiffness_Matrix/Indices_for_FI_FN_less.mat')
idxs=np.squeeze(mat_data['indices'])
indices_tf = tf.constant(idxs, dtype=tf.int32)
idxs2=np.squeeze(mat_data['remaining_indices'])
indices_tf_remain = tf.constant(idxs2, dtype=tf.int32)
mat_data = loadmat('./Stiffness_Matrix/Slave_to_Master_Force_Transformation.mat')
Transf=np.squeeze(mat_data['Transf_New'])
def DD_Schur(y_true, y_pred,input_loads):
    Actual = (y_true / 1e6) # Scale Dow to original for PI
    Pred = (y_pred / 1e6) #Scale Dow to original for PI
    Disp = Pred
    Loadd = bkd.from_numpy(1e-6 * input_loads[0]) # Scale down to have almost equal contribution to DD loss (MegaNewton)
    Master_Load = tf.matmul(Loadd, Transf)
    FI = tf.gather(Master_Load, indices_tf-1, axis=1)
    FN = tf.gather(Master_Load, indices_tf_remain-1, axis=1)
    Data_LOSS = bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))
    Disp_Ui = tf.reshape(Disp[:], (len(input_loads[0]), -1))
    Sum= tf.matmul(LHS,tf.transpose(Disp_Ui)) - tf.transpose(FI) + tf.matmul(RHS, tf.transpose(FN))
    Sum=tf.transpose(Sum)
    New_FD_Loss = bkd.abs(bkd.reduce_mean(Sum))
    New_FD_Loss.stop_gradient = False
    Total_loss = 1*New_FD_Loss+1*Data_LOSS
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
    "DD_EC_Master":DD_EC_Master,
    "DD_Schur":DD_Schur,
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
