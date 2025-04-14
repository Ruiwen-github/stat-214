#!/bin/bash

# Activate the stat214 environment
conda activate env_214  

# Run the Jupyter notebooks
jupyter nbconvert --to notebook --execute --inplace lab3.1_bow.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.1_word2vec.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.1_glove.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.1_model_bow.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3.1_model_w2v.ipynb
jupyter nbconvert --to latex lab3.1.ipynb