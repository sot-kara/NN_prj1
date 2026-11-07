# NN_prj1
Repository for the 1st project of Neural Networks and Deep learning class

## Dataset
This project uses CIFAR-100 for the training and evaluation. This dataset should be unziped in a directory called data (it should create a subdirectory called cifar-100-python, but the project takes the root directorty), otherwise change the path in main.py.

## Usage of KNN and CC classifiers
- Run main.py to evaluate the classifiers on the test datasets and get the output in a eval_results.csv file
- Run plot_results.py to produce the plots from the eval_results.csv file
- To produce the dataset with PCA run preprocess/preprocess.py. Make sure the output file is visible by main.py
- The models themselves are in the directory models
