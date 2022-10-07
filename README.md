# hERG model evaluation software #

Package to easily evaluate models on the supplied hERG blocker dataset.

## Prerequisits

All models were developed on a Linux plattform. While Windows is supported, you need to enable filenames longer than 247 characters for the training experiments to be able to run.
Unfortunately ThunderSVM lacks precompiled Windows binaries. To run these experiments on a Windows computer, the easiest approach is to use Windows Subsystems for Linux 2 with support for GPU (see https://learn.microsoft.com/en-us/windows/wsl/setup/environment).

## Install ##
Anaconda is used to manage python dependencies. This software was developed with Anaconda 4.10.1, but any later version should work as well.

All models except thundersvm can be trained with the same environment. To train models with thundersvm (which depends on cuda 9.0), use the `thundersvm_environment.yml`. For the other environments use `torch_environment.yml`.

Create the different envirnoments by running:
```shell
conda env create -f environment.yml   # For all models except thundersvm
conda env create -f thundersvm_environment.yml   # For thundersvm models
```

This creates environments called `herg-base` and `herg-thundersvm`. Now we have to install this package in both of these environments:

```shell
conda activate herg-base
pip install -e .
conda activate herg-thundersvm
pip install -e .
```
This creates a symlinked version of the python package in this repository in the respective environment.

## Training models ##
Models are traiend using experiment configs using the script `scripts/run_experiment.py`. All training options are determined by a python module 
which acts as a configuration file, example versions are located in `config/experiment_configs`

```shell
scripts/run_experiment.py config/experiment_config/[CONFIG_FILE]
```

By default experiment results are saved to a new `experiments` subdirectory.

### Example: hERG ThunderSVM experiments ###
For example, to train the thundersvm models on the hERG data, first enable the correct conda environment:

```shell
conda activate herg-thundersvm
```
Then run the experiments by supplying the experiment configuration file:

```shell
python scripts/run_experiments.py configs/experiment_configs/herg_ogura_experiment_thundersvm.py
```

The resulting experiments are collected in directories. Each model directory will have a number suffix from 00 to 19. You can find the summarized performance in a timestamped subdirectory following the pattern `experiments/herg_ogura/thundersvm_00/[TIMESTAMP]/resamples/resample_00/evaluation`.

## Evaluation on external test set
To run evaluations on a separate dataset, use the script `scripts/evaluate_on_dataset.py`. The script can be run like so:

```shell
python scripts/evaluate_on_dataset.py dataset/herg_karim_et_al/dataset_spec.py experiments/
```

This will locate all trained models in the directory `experiments` and evaluate them on the dataset specified by the supplied `dataset_spec.py` file. This file tells the framework how it should parse the CSV.


## Summarizing performance

Performance of experiments can be summarized with the `scripts/summarize_performance.py` script. Run it like so:

```shell
python scripts/summarize_performance.py experiments/
```

This will create files in the current directory named like `performance_herg_ogura_test_filtered.csv` for all models the script could find which had evaluation runs on that dataset. If `scripts/evaluate_on_dataset.py` has been run, any resulting evaluation data on another will be automaticall picked up and summarized in a separate file for that dataset.


