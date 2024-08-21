# DeepDissipativeModel

## Installation
For Pytorch, we recommend installing the appropriate version from the official website `https://pytorch.org/get-started/pytorch-2.0/`　according to your GPU and CUDA environment.

Below is an example in a test environment.
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Next, required libraries are installed as follows:
```
pip install git+https://github.com/rtqichen/torchdiffeq
pip install numba
pip install hydra-core --upgrade
conda install joblib
conda install matplotlib pandas
conda install scikit-learn
```

(Optional) For gpu/parallel experiments
```
pip install gpustat
pip install git+https://github.com/kojima-r/ParallelScriptingTools.git
```

Finally, install this repository
```
cd DeepDissipativeModel
pip install .
```

## Directory structure
```
.
├── ddm             # python source codes for DeepDissipativeModel
├── example_linear2 # small example for linear model
├── example_nlink   # small example for nonlinear model
└── experiments   # This is a directory for reproducing the experiments in the paper.
    ├── linear  # mass-spring-damper benchmark　
    ├── nlink   # n-link pendulum benchmark
    └── flow    # fluid system benchmark
```

## Usage
### Dataset generation 
```
ddm-dataset linear  --num 10  --train_num 90
```
Upon executing the command, a new `dataset/` directory will be created within the directory where the above command is run.
The directory will contain 90 training time-series samples and 10  time-series samples.
The method for generating the data is specified by the first argument.
In this instance, as the method is set to `linear`, the data will be generated based on a linear model.
Please note that during the data generation process, ODE (Ordinary Differential Equation) calculations are performed, which may cause the process to take some time.

### Simple training 
```
ddm-train --config config.yaml --data_train ./dataset/linear.train
```
The `--data_train` option is used to specify the data for training.
Note that `.input.npy` and `.obs.npy` postfix should be omitted from this option.
The necessary files will be automatically loaded internally by the program.

You can define detailed neural network settings in the configuration file (`config.yaml`).
For ease of use, you can copy and use the example configuration file located at `example_linear2/config.yaml`.
Adjust the content as needed to set the desired training conditions.

The training result model and logs are saved under `result` by default.
If this directory already exists, they will be overwritten, so please specify it with the `--result_path` option or change the `result_path` item in config.yaml.

### Hyperparameter optimization
```
ddm-opt --config config.yaml --data_train ./dataset/linear.train
```

This setup allows you to perform hyperparameter searches using Optuna.
To run the hyperparameter search, simply replace `ddm-train` with `ddm-opt` in the commands found in the "Simple Training" section.
The optimization results are saved under `study` by default, and the learning results model and validation scores for each trial are saved.


### Resume training 

If you want to stop training and resume it from the checkpoints, you can use a saved checkpoint and resume training with the `--resume` option.
```
ddm-train --config config.yaml --data_train ./dataset/linear.train --resume ./result/model/model.4930.checkpoint
```
# Dissipative options

To enforce dissipativity in a learning model,
you can set the `dissipative_mode` in the config (or specify it as a command option).
Set `dissipative_mode` to one of the following: `naive`, `stable`, `l2stable`,`conservation`, and `dissipative`.
The dissipativity parameter Q, R, S can also be set in the config.

# Experiments
The specific commands used in the experiments have been made into shell scripts and placed in each directory.


