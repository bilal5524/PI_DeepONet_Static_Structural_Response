import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from deepxde.backend import tf
import os
import deepxde as dde

print(dde.__version__)
dde.config.disable_xla_jit()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

m = 156
batch_size = 60

seed = 123
tf.keras.backend.clear_session()
try:
    tf.keras.utils.set_random_seed(seed)
except:
    pass
dde.config.set_default_float("float64")


mat_data = scipy.io.loadmat('./Input_load.mat')
u0_all = mat_data['Input_load'].astype(np.float64)



mat_data = scipy.io.loadmat('./Output_as_per_Stiff.mat')
Output_new_two = mat_data['Output_Transform_5'].astype(np.float64)

max_val = np.zeros(9000)
min_val = np.zeros(9000)


s_all = Output_new_two.copy()
mat_data = scipy.io.loadmat('./Input_mesh_as_per_Stiff.mat')
xy_train_testing = mat_data['Input_mesh_Stiff'].astype(np.float64)

for idx, fraction_train in enumerate([0.8]):
    print('fraction_train = ' + str(fraction_train))

    # Train / test split
    N_valid_case = len(u0_all)
    N_train = int(N_valid_case * fraction_train)
    train_case = np.random.choice(N_valid_case, N_train, replace=False)
    test_case = np.setdiff1d(np.arange(N_valid_case), train_case)

    u0_train = u0_all[train_case, ::]
    u0_testing = u0_all[test_case, ::]
    s_train = s_all[train_case, :]
    s_testing = s_all[test_case, :]

    print('u0_train.shape = ', u0_train.shape)
    print('type of u0_train = ', type(u0_train))
    print('u0_testing.shape = ', u0_testing.shape)
    print('s_train.shape = ', s_train.shape)
    print('s_testing.shape = ', s_testing.shape)
    print('xy_train_testing.shape', xy_train_testing.shape)

    x_train = (u0_train, xy_train_testing)
    y_train = 1e6*s_train # Scale UP
    x_test = (u0_testing, xy_train_testing)
    y_test = 1e6*s_testing # Scale UP
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [m, 75, 75, 75, 75, 75, 75], [3, 75, 75, 75, 75, 75, 75], "relu", tf.keras.initializers.GlorotNormal(seed=123)
    )

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(iterations=600, batch_size=batch_size)
    np.save('losshistory_Rotat_All_Tensor' + str(fraction_train) + '.npy', losshistory)

    import time as TT

    st = TT.time()
    y_pred = model.predict(data.test_x)
    y_pred=y_pred/1e6 # Scale Down
    y_test=y_test/1e6 # Scale Down

    duration = TT.time() - st
    print('y_pred.shape =', y_pred.shape)
    print('Prediction took ', duration, ' s')
    print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')
    np.savez_compressed('TestData_Rotat_All_Tensor' + str(fraction_train) + '.npz', a=y_test[:, :, 0], b=y_pred[:, :, 0],
                        c=y_test[:, :, 1], d=y_pred[:, :, 1], e=y_test[:, :, 2], f=y_pred[:, :, 2], g=y_pred[:, :, 3],
                        h=y_test[:, :, 3], i=y_pred[:, :, 4], j=y_test[:, :, 4],#k=y_pred[:, :, 5], l=y_test[:, :, 5],
                        m=u0_testing, n=xy_train_testing)

    error_s = []
    error_s2 = []
    error_s3 = []
    error_s4 = []
    error_s5 = []
#    error_s6 = []


    for i in range(len(y_pred)):
        #        error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
        error_s_tmp = np.linalg.norm(y_test[i, :, 0] - y_pred[i, :, 0]) / np.linalg.norm(y_test[i, :, 0])
        error_s_tmp2 = np.linalg.norm(y_test[i, :, 1] - y_pred[i, :, 1]) / np.linalg.norm(y_test[i, :, 1])
        error_s_tmp3 = np.linalg.norm(y_test[i, :, 2] - y_pred[i, :, 2]) / np.linalg.norm(y_test[i, :, 2])
        error_s_tmp4 = np.linalg.norm(y_test[i, :, 3] - y_pred[i, :, 3]) / np.linalg.norm(y_test[i, :, 3])
        error_s_tmp5 = np.linalg.norm(y_test[i, :, 4] - y_pred[i, :, 4]) / np.linalg.norm(y_test[i, :, 4])
#        error_s_tmp6 = np.linalg.norm(y_test[i, :, 5] - y_pred[i, :, 5]) / np.linalg.norm(y_test[i, :, 5])

        if error_s_tmp > 1:
            error_s_tmp = 1
        if error_s_tmp2 > 1:
            error_s_tmp2 = 1
        if error_s_tmp3 > 1:
            error_s_tmp3 = 1
        if error_s_tmp4 > 1:
            error_s_tmp4 = 1
        if error_s_tmp5 > 1:
            error_s_tmp5 = 1
#        if error_s_tmp6 > 1:
#            error_s_tmp6 = 1


        error_s.append(error_s_tmp)
        error_s2.append(error_s_tmp2)
        error_s3.append(error_s_tmp3)
        error_s4.append(error_s_tmp4)
        error_s5.append(error_s_tmp5)
#        error_s6.append(error_s_tmp6)


    error_s = np.stack(error_s)
    error_s2 = np.stack(error_s2)
    error_s3 = np.stack(error_s3)
    error_s4 = np.stack(error_s4)
    error_s5 = np.stack(error_s5)
#    error_s6 = np.stack(error_s6)

    error_arrays = [error_s, error_s2, error_s3, error_s4, error_s5]
    variable_names = ['error_U1', 'error_U2', 'error_U3', 'error_R1','error_R3']
    output_file_path = f'error_All_Tensor {fraction_train}.txt'
    with open(output_file_path, 'w') as output_file:
        # Loop through error arrays and variable names
        for error, variable_name in zip(error_arrays, variable_names):
            # Write variable name to the output file
            output_file.write(f"{variable_name}:\n")
            # Write errors to the output file
            for err in error:
                output_file.write(f"{err}\n")
            # Write a separator between variables
            output_file.write("\n")
    print("error_U1 = ", error_s)
    print("error_U2 = ", error_s2)
    print("error_U3 = ", error_s3)
    print("error_R1 = ", error_s4)
    print("error_R3 = ", error_s5)
#    print("error_R3 = ", error_s6)


    # Calculate mean and std for all testing data samples
    print('mean of relative L2 error of U1: {:.2e}'.format(error_s.mean()))
    print('std of relative L2 error of U1: {:.2e}'.format(error_s.std()))
    print('mean of relative L2 error of U2: {:.2e}'.format(error_s2.mean()))
    print('std of relative L2 error of U2: {:.2e}'.format(error_s2.std()))
    print('mean of relative L2 error of U3: {:.2e}'.format(error_s3.mean()))
    print('std of relative L2 error of U3: {:.2e}'.format(error_s3.std()))
    print('mean of relative L2 error of R1: {:.2e}'.format(error_s4.mean()))
    print('std of relative L2 error of R1: {:.2e}'.format(error_s4.std()))
    print('mean of relative L2 error of R3: {:.2e}'.format(error_s5.mean()))
    print('std of relative L2 error of R3: {:.2e}'.format(error_s5.std()))
#    print('mean of relative L2 error of R3: {:.2e}'.format(error_s6.mean()))
#    print('std of relative L2 error of R3: {:.2e}'.format(error_s6.std()))


    Value = fraction_train
    plt.figure()
    plt.hist(error_s.flatten(), bins=15, label='U1')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error - DeepONet {Value}')
    plt.legend()
    plt.grid(True)

    #   plt.savefig('Err_hist_DeepONet_U1 ' + str(Value) + '.jpg', dpi=300)

    plt.hist(error_s2.flatten(), bins=15, label='U2')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error - DeepONet {Value}')
    plt.legend()
    plt.grid(True)

    plt.hist(error_s3.flatten(), bins=15, label='U3')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error - DeepONet {Value}')
    plt.legend()
    plt.grid(True)

    plt.hist(error_s4.flatten(), bins=15, label='R1')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error - DeepONet {Value}')
    plt.legend()
    plt.grid(True)

    plt.hist(error_s5.flatten(), bins=15, label='R3')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error - DeepONet {Value}')
    plt.legend()
    plt.grid(True)

#    plt.hist(error_s6.flatten(), bins=15, label='R3')
#    plt.xlabel('Relative L2 Error')
#    plt.ylabel('Occurrence')
#    plt.title(f'Histogram of Relative L2 Error - DeepONet {Value}')
#    plt.legend()
#    plt.grid(True)



    plt.savefig('Err_hist_DeepONet_All_Tensor ' + str(Value) + '.jpg', dpi=300)