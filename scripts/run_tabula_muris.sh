#!/bin/bash

declare -a models=("relationnet") # "protonet" "matchingnet" "maml" "baseline" "baseline_pp"
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

dataset="tabula_muris"
representative_aggregation="sum"
deep_distance_type="fc-conc"
deep_distance_layer_size="[128, 1]"
backbone_layer_dim="[16]"
learning_rate="0.001"
backbone_weight_decay="0.001"
backbone_dropout="0.0"

for model in "${models[@]}"; do
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
            echo conda run -n few --no-capture-output --live-stream python run.py \
                exp.name=${dataset} \
                method=${model} \
                dataset=${dataset} \
                n_way=${W} \
                n_shot=${S} 
        )
    done
done
