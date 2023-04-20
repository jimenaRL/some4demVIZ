#!/bin/bash

users_limit=1

experiment="france_ideN_8_sources_min_followers_25_sources_min_outdegree_3"
embeddings="/viz/embeddings/$experiment"
logdir="/viz/logs/$experiment"

python viz_ideological.py --embeddings=$embeddings --logdir=$logdir --users_limit=$users_limit
tensorboard --bind_all --port 6006 --logdir $logdir &


experiment="ches2019_antielite_salience_vs_ches2019_immigrate_salience_vs_ches2019_enviro_salience_vs_ches2019_eu_position_vs_ches2019_lrgen"
embeddings="$embeddings/$experiment"
logdir="$logdir/$experiment"
python viz_attitudinal.py --embeddings=$embeddings --logdir=$logdir --users_limit=$users_limit
tensorboard --bind_all --port 6007 --logdir $logdir &


