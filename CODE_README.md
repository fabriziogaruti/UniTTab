# README FOR CODE UNDERSTANDING

# CODE REVIEW

## GENERAL

* run_all_experiments.sh: bash script that contains the commands to launch from terminal the different experiments
* config.py: a script with a configuration dict
* main.py: main script called to perform training or evaluation
* args.py: script that contains the parser for all the arguments
* misc/utils.py: contains script that divide dataset in splits

## DATASET

* dataset/card.py: is the main script to create a dataset from the transaction csv given in the repo, with the cached argument most of the operation are skipped loading the files saved previously (if it is not the first launch)
* dataset/datacollator.py: is the function that transform the dataset inputs into a batch for the training, it is also the class responsible for masking casual elements and setting the correct labels
* dataset/vocab.py: is the class responsible for creating a unique vocabolary for all the values found in the dataset 

N.B the vocabolary is one for all the fields is not distinct, so its dimension is the sum of all the different values of all the fields
