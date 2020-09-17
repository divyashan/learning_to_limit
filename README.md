# Learning to Limit Data for Data Minimization Compliance
data minimization in machine learning

```
# Conda environment:
conda create -n llddm python=3.7.5 anaconda
source activate llddm

# CUDA:
nvcc --version
```
# Downloading the datasets: 

```
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
wget http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/reviews.clean.json.gz
```
Create a ./datasets folder and unpack these datasets into it.

## Reproducing experiments

Produce data required for experiments:
```
# Subsampling earliest portion of data collection
# curve, given .01% initialized data
python subsampling.py early

# Subsampling later portion of data collection
# curve, where  sample increment = 2% of |queryable data|
# and initialized data = 10% of |queryable data|
python subsampling.py later 

# Fitting baselines and our method to 
# (sample size, validation performance) tuples
python curve_fitting_expmt.py
```

All results will be saved to the ./results directory, under configs corresponding to the experiment parameters. Code to reproduce each of the figures in the paper can be found in the ./notebooks folder. 

- Figure 1: 'Performance Curve Plots.ipynb'
- Figure 2: 'Power Law Results.ipynb'
- Figure 3: 'Power Law Results.ipynb'
- Figure 4: 'Power Law Results.ipynb'
- Figure 5: 'Power Law Results.ipynb'
- Figure 6: 'Power law Results (per users).ipynb'


