# Thesis Sanquin
This GITHUB contains the following files:
- sanquin_blood.py
  Contains the functions that select the demand and supply distribution of blood
- sanquin_inventory.py
  Contains all functions and object regarding invenotry and queue
- environment.py
  Contain the environment of the RL agent
- visualisations.py
  Functions to create visualisation and export Excel
- PPO2.py
  The algorithm. If you want to train, use this file.
- test_model.py
  With this file the model can be evaluated. Note that this file must have the exact same environment as the saved model that is imported.
- data_extract.py
  This file contains the code of combining the export of tensorboard and the other eval metrics.


There are two folders:
- results, results are saved here. Some example results are given. Always use the combined data!
    - combined data
      This folder contains all the data on the traning process
    - demo 
      Some demo results
    - evaluation metrics data
      Exported by the model, this file contains data on how good the model is. Is combined with data of tensorboard into the combined dataset. 
- data, contains the JSON files for the data (not the real data ofc)


# Packages
- gym (openai gym)
- stable baselines
- pandas
- numpy
- matplotlib
