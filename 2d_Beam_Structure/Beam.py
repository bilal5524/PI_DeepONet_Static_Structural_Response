import sys
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import scipy.io
import os
from sklearn.preprocessing import StandardScaler
from deepxde.backend import paddle
import paddle


paddle.disable_static()

# Check if PaddlePaddle is compiled with CUDA support
gpu_available = paddle.is_compiled_with_cuda()

print("Num GPUs Available: ", int(gpu_available))

m = 21
batch_size = 20
seed = 123
paddle.disable_static()

# Set a random seed
paddle.seed(seed)

dde.config.set_default_float("float64")

mat_data = scipy.io.loadmat('./Data/Input_load.mat')
u0_all = mat_data['Input_load'].astype(np.float64)


mat_data = scipy.io.loadmat('./Data/Output.mat')
Output_new_two = mat_data['Output'].astype(np.float64)

s_all = Output_new_two.copy()
mat_data = scipy.io.loadmat('./Data/Input_mesh.mat')
xy_train_testing = mat_data['Input_mesh'].astype(np.float64)

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
    y_train = 1e6*s_train # Scale Up for Better Training
    x_test = (u0_testing, xy_train_testing)
    y_test = 1e6*s_testing # Scale Up for Better Training
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [m, 48, 48, 48, 48, 48, 48], [2, 48, 48, 48, 48, 48, 48], "relu", "Glorot normal"
    )
    total_params = sum(p.numel() for p in net.parameters())
    print("Total learnable parameters:", total_params)
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    # Below is number of iteration. To get actual number of epoch= Iterations/(Data*Train_ratio/Batch_size)
    losshistory, train_state = model.train(epochs=243840, batch_size=batch_size,
                                           model_save_path="./mdls/TrainFrac_Rotat1_5Lay" + str(fraction_train))
    np.save('losshistory_Rotat1_5Lay' + str(fraction_train) + '.npy', losshistory)


    import time as TT

    st = TT.time()
    y_pred = model.predict(data.test_x)
    y_pred=y_pred/1e6 # Scale Down for Actual Values
    y_test=y_test/1e6 # Scale Down for Actual Values

    duration = TT.time() - st
    print('y_pred.shape =', y_pred.shape)
    print('Prediction took ', duration, ' s')
    print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')
    np.savez_compressed('TestData_Rotat1_5Lay' + str(fraction_train) + '.npz', a=y_test[:, :, 0], b=y_pred[:, :, 0],
                        c=y_test[:, :, 1], d=y_pred[:, :, 1], e=y_test[:, :, 2], f=y_pred[:, :, 2],
                        g=u0_testing, h=xy_train_testing)

    error_s = []
    error_s2 = []
    error_s3 = []

    for i in range(len(y_pred)):

        error_s_tmp = np.linalg.norm(y_test[i,:,0] - y_pred[i,:,0]) / np.linalg.norm(y_test[i,:,0])
        error_s_tmp2 = np.linalg.norm(y_test[i,:,1] - y_pred[i,:,1]) / np.linalg.norm(y_test[i,:,1])
        error_s_tmp3 = np.linalg.norm(y_test[i,:,2] - y_pred[i,:,2]) / np.linalg.norm(y_test[i,:,2])

        if error_s_tmp > 1:
            error_s_tmp = 1
        if error_s_tmp2 > 1:
            error_s_tmp2 = 1
        if error_s_tmp3 > 1:
            error_s_tmp3 = 1

        error_s.append(error_s_tmp)
        error_s2.append(error_s_tmp2)
        error_s3.append(error_s_tmp3)

    error_s = np.stack(error_s)
    error_s2 = np.stack(error_s2)
    error_s3 = np.stack(error_s3)
    error_arrays = [error_s, error_s2, error_s3]
    variable_names = ['error_U1', 'error_U2', 'error_R3']
    output_file_path = f'error1_5Lay {fraction_train}.txt'
    with open(output_file_path, 'w') as output_file:
        for error, variable_name in zip(error_arrays, variable_names):
            output_file.write(f"{variable_name}:\n")
            for err in error:
                output_file.write(f"{err}\n")
            output_file.write("\n")
    print("error_U1 = ", error_s)
    print("error_U2 = ", error_s2)
    print("error_R3 = ", error_s3)

    # Calculate mean and std for all testing data samples
    print('mean of relative L2 error of U1: {:.2e}'.format(error_s.mean()))
    print('std of relative L2 error of U1: {:.2e}'.format(error_s.std()))
    print('mean of relative L2 error of U2: {:.2e}'.format(error_s2.mean()))
    print('std of relative L2 error of U2: {:.2e}'.format(error_s2.std()))
    print('mean of relative L2 error of R3: {:.2e}'.format(error_s3.mean()))
    print('std of relative L2 error of R3: {:.2e}'.format(error_s3.std()))

    Value=fraction_train
    plt.figure()
    plt.hist(error_s.flatten(), bins=15, label='U1')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error {Value}')
    plt.legend()
    plt.grid(True)

    plt.hist(error_s2.flatten(), bins=15, label='U2')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error {Value}')
    plt.legend()
    plt.grid(True)

    plt.hist(error_s3.flatten(), bins=15, label='R3')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Occurrence')
    plt.title(f'Histogram of Relative L2 Error {Value}')
    plt.legend()
    plt.grid(True)
    plt.savefig('Error_Histogram ' + str(Value) + '.jpg', dpi=300)