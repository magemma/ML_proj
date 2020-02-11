# ML_proj
Implementation from scratch of a feed-forward fully connected nn

## Folder and files structure

(*) `activation_functions.py` contains the interface and classes modelling the functioning of activation functions

(*) `config.py` configuration file used for grid search

(*) `k_plot_test_clas.py` main file used to train and test the best model selected for classification through validation process and plot training and test plots

(*) `k_plot_test_reg.py` main file used to train and test the best model selected for regression through validation process and plot training and test plots

(*) `loss_functions.py` contains the interface and classes modelling the loss functions

(*) `ML-CUP19-TR.csv` train data for regression task

(*) `monks/` train and test data for classification task

(*) `nn_utilities.py` utility functions for the functioning of the neural network

(*) `optimizer.py` contains the interface and classes modelling the momentum optimization

(*) `report_project.pdf` is the report of the activity held, as neural network development and experiments

(*) `results/` folder containing all the `.csv` files containing all models trained using the grid search parameters and sorted according to the final score

(*) `test_classification.py` main file used to train and test the best k (set to 5) models for classification selected through validation process and plot training and test plots

(*) `test_regression.py` main file used to train and test the best k (set to 5) models for regression selected through validation process and plot training and test plots

(*) `train_classification.py` main file used to train and validate a model for classification task using configuration file for conducting a grid search

(*) `train_regression.py` main file used to train and validate a model for regression task using configuration file for conducting a grid search
