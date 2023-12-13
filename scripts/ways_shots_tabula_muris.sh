#!/bin/bash

declare -a models=("relationnet") # "protonet" "matchingnet" "maml" "baseline" "baseline_pp"
declare -a few_shot_pairs=(
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
declare -a baseline_pairs=(
    "5 1"
    "5 5"
    "20 5"
)

dataset="tabula_muris"
representative_aggregation="sum"    # Default parameter from the original paper
deep_distance_type="fc-conc"        # Default parameter from the original paper
deep_distance_layer_size="[128, 1]" # Best parameter from tuning
backbone_layer_dim="[16]"           # Best parameter from tuning
learning_rate="0.001"               # Best parameter from tuning
backbone_weight_decay="0.001"       # Best parameter from tuning
backbone_dropout="0.0"              # Best parameter from tuning

for model in "${models[@]}"; do
    # Define pairs depending on model
    if [ $model == "baseline" ] || [ $model == "baseline_pp" ]; then
        pairs=("${baseline_pairs[@]}")
    else
        pairs=("${few_shot_pairs[@]}")
    fi

    for pair in "${pairs[@]}"; do
        W=$(echo $pair | cut -d ' ' -f1)
        S=$(echo $pair | cut -d ' ' -f2)
        if [ $model == "relationnet" ]; then
            (
                conda run -n few --no-capture-output --live-stream python run.py \
                    exp.name=run_${dataset} \
                    method=${model} \
                    dataset=${dataset} \
                    n_way=${W} \
                    n_shot=${S} \
                    method.representative_aggregation=${representative_aggregation} \
                    method.deep_distance_type=${deep_distance_type} \
                    method.deep_distance_layer_sizes="${deep_distance_layer_size}" \
                    backbone.layer_dim="${backbone_layer_dim}" \
                    method.learning_rate=${learning_rate} \
                    method.backbone_weight_decay=${backbone_weight_decay} \
                    backbone.dropout=${backbone_dropout}
            )
            continue
        fi

        (
            conda run -n few --no-capture-output --live-stream python run.py \
                exp.name=${dataset} \
                method=${model} \
                dataset=${dataset} \
                n_way=${W} \
                n_shot=${S} 
        )
    done
done
