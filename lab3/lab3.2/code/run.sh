#!/bin/bash

# Activate the stat214 environment
conda activate env_214  
conda env update -f environment.yaml

# Run the Jupyter notebooks
jupyter nbconvert --to notebook --execute --inplace lab3.2_part1.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.2_part2.ipynb
jupyter nbconvert --to latex lab3.2.ipynb