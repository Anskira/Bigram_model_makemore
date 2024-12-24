#  Makemore_bigram_model

## Introduction
In this project I have implemented a Multi Layer Perceptron character level language model to generate new names based on the training dataset. The objective of this project is to explore the basics of machine learning: model training, avoiding overfitting/underfitting, hyperparameter tuning. It also focuses on the process of forward and backward propagation, weight initialization, issues faced in this and how they are tackled with (Kaming Initialization, BaTchNorm layer)

##  Project Description
###  Prerequisites
-  Python 3.7
-  Cudo 10.1 or higher (for GPU support)
-  pip (Python Package Installer)

##  Dataset Overview
The text file 'names.txt' contains more than 32k names, using these names the bigram model is trained.

##  Process
1.  Read the 'names.txt' file and got all the characters from the text
2.  Made a string to integer and integer to string mapping of all the characters
3.  The number of characters(block_size) considered for lookback is 3, created a multidimensional list that will store lists containing 3 previous characters and a targets list that stores the preceding character
4.  Developed the linear, batchnorm and tanh activation layers from scratch instead of using the builtin pytorch layers for better understanding of the flow of data and gradients in a neural network and the operations performed on it
5.  Using the kaiming weight initialization, multiplying the weights by a factor of '5/3' to prevent the over shrinking of values by the tanh activation layer, the embedding dimensions were 10 and the number of neurons in hidden layer were 200
6.  The model was trained for 200000 steps and with a batch size of 32, a decaying learning rate algorithm was used

### Metrics obtained
-  Training Accuracy: 1.9712
-  Validation Accuracy: 2.0715


