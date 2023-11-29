#!/bin/bash

# Define the list of models
declare -a models=("protonet" "matchingnet" "maml")

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
        I=$(echo $pair | cut -d ' ' -f1)
        J=$(echo $pair | cut -d ' ' -f2)
        python run.py exp.name=s2 method=$model dataset=tabula_muris_${I}_${J}
    done
done
