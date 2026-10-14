# Masters_IndividualProject_220029768

## Setting-up the environment 

To execute the software code, the Conda environment contained in "environment.yml" is required. This project was developed and evaluated using the 
Conda environment contained in "environment.yml". For this reason, the recommended method for reproducing the software code is to use the provided 
environment.yml file to create a Conda environment. 

The environment.yml file provides a more complete description of the original Conda environment, including the Python version, Conda-managed dependencies, 
Conda packages, package versions, channels, and any pip-installed dependencies included in the environment. Using this file therefore
provides the closest reproduction of the environment in which the experiments were originally conducted.

Nonetheless, a "requirements.txt" file is also provided as an alternative. Although requirements.txt provides the required Python packages and their 
versions, it does not describe the complete original Conda environment in the same way as environment.yml. Consequently, recreating the environment 
from environment.yml is the recommended approach when attempting to reproduce the reported experimental results. The requirements.txt file is provided
primarily for convenience.

## Running the code 

First, run Load_cifar100_vFinal.py. This script will cause the CIFAR-100 data to download if not present yet. 
It also performs imortant data preparation, so it must be executed regardless of whether the CIFAR-100 dataset
has already been downloaded or not. If the CIFAR-100 dataset has been downloaded, running this script will not
download it again.

Then, to train any of the specialised models, run 