# 拓扑优化

## 生成数据集

以下文件需要根据实际情况修改其参数。修改完毕后即可直接运行。

### 1. 随机生成: `random_generate_data.py`

该脚本用于生成随机拓扑优化数据集。

### 2. 悬臂梁数据生成: `cantilever_beam_generate_data.py`

该脚本生成悬臂梁的拓扑优化数据。

### 3. 连续梁数据生成: `continuous_beam_generate_data.py`

该脚本生成连续梁的拓扑优化数据。

### 4. 简支梁数据生成: `simply_supported_beam_generate_data.py`

该脚本生成简支梁的拓扑优化数据。

### 5. 配置文件

该代码定义了多个函数，用于生成不同类型和网格尺寸的拓扑优化配置。每个配置字典包含了各种参数，可以用于拓扑优化算法。

### 6. `prepare.py`

该脚本从指定目录加载 `.npz` 文件，提取并处理其中的拓扑优化数据。生成用于机器学习或数据分析的输入数据和目标数据。将处理好的输入和目标数据分别保存为 `.npz` 文件。

### 7. `prepare_all.py`

该脚本从四种不同类型的拓扑优化数据文件中加载数据，并生成一个包含所有输入和目标数据的统一数据集。输入数据是每个样本的第一个时间步的数据，目标数据是每个样本的最后一个时间步的数据。

## 训练模型

### 1. `training_cantilever.py`

该文件用于训练一个使用悬臂梁拓扑优化数据的模型。它可能包含加载数据、预处理、定义模型和训练模型等步骤。

### 2. `training_continue.py`

该文件用于训练一个使用连续梁拓扑优化数据的模型。它可能包含与 `training_cantilever.py` 类似的步骤，但数据集和模型可能有所不同。

### 3. `training_random.py`

该文件用于训练一个使用随机生成的拓扑优化数据的模型。它可能包含加载随机数据、预处理、定义模型和训练模型等步骤。

### 4. `training_random_noise.py`

该文件用于训练一个使用添加了噪声的随机拓扑优化数据的模型。它可能包含加载数据、添加噪声、预处理、定义模型和训练模型等步骤。

### 5. `training_simply.py`

该文件用于训练一个使用简支梁拓扑优化数据的模型。它可能包含与 `training_cantilever.py` 类似的步骤，但数据集和模型有所不同。

### 6. `training_all.py`

该文件用于训练一个综合模型，使用所有类型的拓扑优化数据（悬臂梁、连续梁、随机数据、简支梁等）。它包含加载多个数据集、预处理、定义模型和训练模型等步骤。

### 7. `Iou.py`

该文件提供计算 IoU 分数的函数以及绘制训练过程中损失和 IoU 分数变化图的函数。

### 8. `output_image.py`

该文件生成并保存模型训练过程中的图像，包括输入数据、目标数据和模型输出数据。

## 测试模型

### 1. `test1.py`

该文件加载训练好的模型，对输入数据进行预测，并计算和可视化 IoU 分数。

如有任何问题或建议，请通过 lishun1693@163.com 联系我。
