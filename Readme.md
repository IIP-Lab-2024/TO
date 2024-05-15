# Topology Optimization

## Generate Dataset

The following files need to have their parameters modified according to the actual situation. Once modified, they can be run directly.

### 1. Random Generation: `random_generate_data.py`

This script is used to generate random topology optimization datasets.

### 2. Cantilever Beam Data Generation: `cantilever_beam_generate_data.py`

This script generates topology optimization data for a cantilever beam.

### 3. Continuous Beam Data Generation: `continuous_beam_generate_data.py`

This script generates topology optimization data for a continuous beam.

### 4. Simply Supported Beam Data Generation: `simply_supported_beam_generate_data.py`

This script generates topology optimization data for a simply supported beam.

### 5. Configuration File

This code defines multiple functions to generate topology optimization configurations for different types and grid sizes. Each configuration dictionary contains various parameters that can be used in the topology optimization algorithm.

### 6. `prepare.py`

This script loads `.npz` files from a specified directory, extracts and processes the topology optimization data. It generates input and target data for machine learning or data analysis. The processed input and target data are saved as `.npz` files.

### 7. `prepare_all.py`

This script loads data from four different types of topology optimization files and generates a unified dataset containing all input and target data. The input data is the first time step of each sample, and the target data is the last time step of each sample.

## Training Model

### 1. `training_cantilever.py`

This file trains a model using cantilever beam topology optimization data. It likely includes steps for loading data, preprocessing, defining the model, and training the model.

### 2. `training_continue.py`

This file trains a model using continuous beam topology optimization data. It likely includes similar steps to `training_cantilever.py`, but with different datasets and possibly different models.

### 3. `training_random.py`

This file trains a model using randomly generated topology optimization data. It likely includes steps for loading random data, preprocessing, defining the model, and training the model.

### 4. `training_random_noise.py`

This file trains a model using randomly generated topology optimization data with added noise. It likely includes steps for loading data, adding noise, preprocessing, defining the model, and training the model.

### 5. `training_simply.py`

This file trains a model using simply supported beam topology optimization data. It likely includes similar steps to `training_cantilever.py`, but with different datasets and possibly different models.

### 6. `training_all.py`

This file trains a comprehensive model using all types of topology optimization data (cantilever beam, continuous beam, random data, simply supported beam, etc.). It includes steps for loading multiple datasets, preprocessing, defining the model, and training the model.

### 7. `Iou.py`

This file provides functions to calculate the IoU score and to plot the loss and IoU score during training.

### 8. `output_image.py`

This file generates and saves images during the model training process, including input data, target data, and model output data.

## Test Model

### 1. `test1.py`

This file loads the trained model, predicts on input data, and calculates and visualizes the IoU scores.

For any questions or issues, please contact me at lishun1693@163.com.

## Citation
lf you find this repo helpful, please cite the following paper:

```
@article{li2024topology,
  title={Topology optimization based on improved DoubleU-Net using four boundary condition datasets},
  author={Li, Shun and Zeng, PeiJian and Lin, Nankai and Lu, Maohua and Lin, Jianghao and Yang, Aimin},
  journal={Engineering Optimization},
  pages={1--19},
  year={2024},
  publisher={Taylor \& Francis}
}
```
