import numpy as np
import os
input_data = np.zeros((10000, 32, 64))
target_data = np.zeros((10000, 32, 64))

for i in range(10000):
    print(i)
    if(i<50000):
        npz_path = f"./random/{i}.npz"  # 假设您的数据文件名是 "data_0.npz" 至 "data_999.npz"
        npz = np.load(npz_path)
        data = npz['arr_0']  # 每个.npz文件包含一个名为 'arr_0' 的numpy数组，形状为 (40, 32, 64)
        input_data[i] = data[0]
        target_data[i] = data[-1]
    if (100000>i >=50000):
        npz_path = f"./cantilever-beam/{i}.npz"  # 假设您的数据文件名是 "data_0.npz" 至 "data_999.npz"
        npz = np.load(npz_path)
        data = npz['arr_0']  # 每个.npz文件包含一个名为 'arr_0' 的numpy数组，形状为 (40, 32, 64)
        input_data[i] = data[0]
        target_data[i] = data[-1]
    if (150000>i >=100000):
        npz_path = f"./continuous-beam/{i}.npz"  # 假设您的数据文件名是 "data_0.npz" 至 "data_999.npz"
        npz = np.load(npz_path)
        data = npz['arr_0']  # 每个.npz文件包含一个名为 'arr_0' 的numpy数组，形状为 (40, 32, 64)
        input_data[i] = data[0]
        target_data[i] = data[-1]
    if (200000>i >=150000):
        npz_path = f"./simply-supported-beam/{i}.npz"  # 假设您的数据文件名是 "data_0.npz" 至 "data_999.npz"
        npz = np.load(npz_path)
        data = npz['arr_0']  # 每个.npz文件包含一个名为 'arr_0' 的numpy数组，形状为 (40, 32, 64)
        input_data[i] = data[0]
        target_data[i] = data[-1]
# Save input and target data as .npz files
np.savez('./data/all_input_data_64_128_10000.npz', input=input_data)
np.savez('./data/all_target_data_64_128_10000.npz', target=target_data)
print("数据加载完毕")