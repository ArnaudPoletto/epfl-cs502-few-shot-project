#!/bin/bash

declare -a representative_aggregations=("sum" "mean")
declare -a deep_distance_types=("fc-conc" "fc-diff" "cosine" "l1" "euclidean")
declare -a deep_distance_layer_sizes=("[128, 64, 32, 1]" "[128, 1]")
declare -a backbone_layer_dims=("[128, 128, 128, 128]" "[16]")

dataset="swissprot"
model="relationnet"
n_way="5"                       # Default analysis configuration
n_shot="5"                      # Default analysis configuration
learning_rate="0.001"           # Best parameter from tuning
backbone_weight_decay="0.001"   # Best parameter from tuning
backbone_dropout="0.0"          # Best parameter from tuning 

for representative_aggregation in "${representative_aggregations[@]}"; do
    for deep_distance_type in "${deep_distance_types[@]}"; do
        for deep_distance_layer_size in "${deep_distance_layer_sizes[@]}"; do
            for backbone_layer_dim in "${backbone_layer_dims[@]}"; do
                ( 
                    conda run -n few --no-capture-output --live-stream python run.py \
                        exp.name=ablation_${dataset} \
                        method=${model} \
                        dataset=${dataset} \
                        n_way=${n_way} \
                        n_shot=${n_shot} \
                        method.representative_aggregation=${representative_aggregation} \
                        method.deep_distance_type=${deep_distance_type} \
                        method.deep_distance_layer_sizes="${deep_distance_layer_size}" \
                        backbone.layer_dim="${backbone_layer_dim}" \
                        method.learning_rate=${learning_rate} \
                        method.backbone_weight_decay=${backbone_weight_decay} \
                        backbone.dropout=${backbone_dropout}
                )
            done
        done
    done
done

./ablation_swissprot_no_f.sh
