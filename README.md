# Jensen_Tsallis_Divergence_for_Supervised_Classification_under_Data_Imbalance
 
The official code for the paper: "Jensen_Tsallis_Divergence_for_Supervised_Classification_under_Data_Imbalance" submitted to ECLM 2025

## Environment Setup
Create a conda environment, activate it, and install additional pip packages:

```bash
conda env create -f env_JTD.yml
conda activate JTD
```
## Running Experiments
Please check programs/JSON_parameters to see the pre-configured experiments. Each JSON file contains all the simulations of the experiment, achieved by combining the parameter lists stored inside. The final models are stored in the model/ folder. In this folder, you can find the JSON files to reproduce the parameter tuning and the final simulations of the paper. The best parameters are stored in the programs/best_parameters folder.

For example, to run the MNIST hyperparameter tuning experiment, execute the following command:
```bash
python main.py --job-name MNIST_LeNet_5_hyperparameter_seeking
```
If you want to execute a fast training without evaluate the model at each step, run:
```bash
python main.py --job-name MNIST_LeNet_5_hyperparameter_seeking --fast-training
```

To run the final experiment for CIFAR-10, execute:
```bash
python main.py --job-name CIFAR10_resnet34_final_training
```
