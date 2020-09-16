# Subsample indices to produce user performance curves

# hyperparameters: step size, global goal, initialized percentage,

# while you haven't acquired the total # of observations
    # maintain a list of completed users 
    # iterate over users
        # fit performance curve for that user (sample sizes, that user's specific performance)
        # check to see if percentage exceeds the goal percentage
        # if it does, add to list of completed users

# There does have to be a separate run for each goal percentage
goal_pcts = [.7, .8, .9, .95]
g = .9
init_pct = .10
mltiny_config = {'n_runs': 5, 'checks': False, 'init_pct': init_pct,
                 'init_mode': 'uniform', 'batch_size': 21000,
                 'step_size': 5250, 
                 'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 213973, 
                 'g': g} 
