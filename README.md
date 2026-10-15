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
primarily for convenience of the marker in case an alternative to yml is required.

## Running the code 
Please note, every intermediate implementation attempt is in a different folder. 
**The final project software code is in folder Final_implementation_attempt** of the repository.

First, run Load_cifar100_vFinal.py. This script will download the CIFAR-100 data if it has not yet been downloaded. 
It also performs essential data preparation, so it must be executed regardless of whether the CIFAR-100 dataset
has already been downloaded or not. If the CIFAR-100 dataset has been downloaded, running this script will not
download it again.

Then, to train any of the specialised models, run their respective scripts. Then, to produce the FedAvg or MoE global model from these specialised models, run 
CORRECTED_Complete_Model_Aggregation_All_Layers_mine_AGGREGATION_OF_CLASSIFIER_vFinal.py for the FedAvg and Alternatives_to_Aggregation_MoE_gating_network_vFinal.py
for the final MoE. However, all models and baselines presented in the Project Report can be found 
in folders final_specialised_models and final_specialised_models_BASELINE. To evaluate these models, run script Evaluate_BASELINE_vFinal.py. This is the case 
for all models except the final MoE model (Alternatives_to_Aggregation_MoE_gating_network_vFinal.py), the MoE baselines 
(Alternatives_to_Aggregation_MoE_TASK-IL_BASELINE_vFinal.py and Alternatives_to_Aggregation_MoE_NO_OOD_BASELINE_vFinal.py) and the final FedAvg model 
(CORRECTED_Complete_Model_Aggregation_All_Layers_mine_AGGREGATION_OF_CLASSIFIER_vFinal.py), which need their respective scripts executed for the models to be evaluated.

Whether to re-run the experiments or to evaluate the models in folders final_specialised_models and final_specialised_models_BASELINE, you will have to change the 
file path whenever the models are loaded into the scripts to the location where the project and relevant folders are located in your machine after downloading the 
project repository.