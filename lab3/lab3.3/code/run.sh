#!/bin/bash

# Activate the stat214 environment
conda activate env_214  
conda env update -f environment.yaml

# Run the Jupyter notebooks
jupyter nbconvert --to notebook --execute --inplace lab3.3_part1_task1.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.3_part1_task2.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.3_part1_task3_and_part2.ipynb