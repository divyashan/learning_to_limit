# Learning to Limit Data via Scaling Laws: Data Minimization Compliance in Practice

This repository provides the official implementation for the paper "Learning to Limit Data Collection via Scaling Laws: Data Minimization Compliance in Practice".

## Requirements

```
# Conda environment:
conda create -n llddm --file requirements.txt
source activate llddm
```
## Downloading the datasets: 

```
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip -P ./datasets
wget http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/reviews.clean.json.gz -P ./datasets
```

Run each cell in ./notebooks/Preprocessing_MovieLens20M.ipynb and ./notebooks/Preprocessing_GoogleLocal.ipynb to produce train/test splits for the large (MovieLens-L, GoogleLocal-L)  and small (MovieLens-S, GoogleLocal-S) samples used for the paper's experiments.

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
- Table 2: 'Table 2 | Diminishing Returns 
- Figure 2: 'Table 2 + Figure 2 | Diminishing Returns' + todo fill this in
- Table 3: 'Table 3 | Robustness to AFA Algorithms'
- Table 4: 'Table 4 | Robustness to Query Size'
- Figure 3A: 'Figure 3A | Per-User Power Law Curve Results'
- Figure 3B: '
- Figure 3C: 'Figure 3C | AFA Burden in Individual Users'


