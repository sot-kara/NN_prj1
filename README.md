# NN_prj1
Repository for the 1st project of Neural Networks and Deep learning class

## Dependencies
- cupy
- numpy
- scikit-learn
- pytorch and pytorch-vision
- matplotlib

## Dataset
This project uses CIFAR-100 for the training and evaluation. This dataset should be unziped in a directory called data (it should create a subdirectory called cifar-100-python, but the project takes the root directorty), otherwise change the path in main.py.

## Usage of KNN and CC classifiers
- Run main.py to evaluate the classifiers on the test datasets and get the output in a eval_results.csv file
- Run plot_knn_cc_results.py to produce the plots from the eval_results.csv file
- To produce the dataset with PCA run preprocess/preprocess.py.
- The models themselves are in the directory models


## Project Structure
- models/ has every model that was experimented on during this assignment
- utils/ has every helper function, trainer, experiment runner or other auxilary code used for training, plotting, saving history etc
- results/ has the history data and metadata for each model run, and comparisson_figures every figure used to compare models
- figures/ also holds every figure presented in the report
- Run test_*.py to get the results from each experiment