#!/bin/bash

declare -a models=("baseline" "baseline_pp")
declare -a datasets=("swissprot" "tabula_muris")
declare -a pairs=(
    "5 1"
    "5 5"
    "20 5"
)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for pair in "${pairs[@]}"; do
            W=$(echo $pair | cut -d ' ' -f1)
            S=$(echo $pair | cut -d ' ' -f2)
            ( conda run -n few --no-capture-output --live-stream python run.py exp.name=$dataset method=$model dataset=$dataset n_way=$W n_shot=$S )
        done
    done
done