# Thesis Sanquin
This repo contains all code necessary to run the RL agents, Fixed Policy issuing strategy, and FIFO/MROL policy 
mentioned in the thesis. 

Best is to start in the sample notebook

This GITHUB contains the following files:
- *sample_notebook.py* \
Contains all information needed to run the Agent, Fixed Policies, and FIFO/MROL. \
Shown how results can be exported
- *sanquin_blood.py* \
  Contains the functions that select the demand and supply distribution of blood
- *sanquin_inventory.py* \
  Contains all functions and object regarding invenotry and queue
- *environment.py* \
  Contain the environment of the RL agent
- *visualisations.py* \
  Functions to create visualisation and export Excel
- *data_extract.py* \
  This file contains the code of combining the export of tensorboard and the other eval metrics.
- *RL.py* \
Contains all functions to train and test the RL agent
- *FP1.py* \'
Code to run Fixed Policy 1
- *FP2.py* \
Code to run Fixed Policy 2
- *MROL.py* \
Code to run MROL
- *custom_callback.py* \
Custom call back necesaary for decaying reward while training RL agent



There are two folders:
- *results* \
 results are saved here. Some example results are given. \
 subfolders:
   - combined data\
      This folder contains all the data on the traning process
    - evaluation metrics data\
      Exported by the model, this file contains data on how good the model is. Is combined with data of tensorboard into the combined dataset.
    - fig\
    All figures created are stored in this folder
    - fig_data_excel\
    All information used for the figures is stored in this folder
    - model \
    Trained agents are stored here
- *data*, contains the JSON files for the data (not the real data ofc)


# Packages used
- gym (openai gym)
- stable baselines
- pandas
- numpy
- matplotlib
- pulp
