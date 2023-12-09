#!/bin/bash

# Define the list of models
declare -a models=("maml" "relationnet") # "protonet" "matchingnet"

# Define the pairs of I and J
declare -a pairs=(
    "5 1"
    "5 5"
    "5 10"
    "5 20"
    "5 30"
    "5 40"
    "5 50"
    "5 70"
    "5 90"
    "10 5"
    "20 5"
    "30 5"
    "40 5"
    "50 5"
)

# Iterate over each model
for model in "${models[@]}"; do
    # Then iterate over each pair and echo the command
    for pair in "${pairs[@]}"; do
        W=$(echo $pair | cut -d ' ' -f1)
        S=$(echo $pair | cut -d ' ' -f2)
        ( conda run -n few --no-capture-output --live-stream python run.py exp.name=swissprot method=$model dataset=swissprot n_way=$W n_shot=$S )
    done
done
