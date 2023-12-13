#!/bin/bash

declare -a deep_distance_layer_sizes=("[128, 64, 32, 1]" "[128, 1]")

dataset="tabula_muris"
model="relationnet"
n_way="5"                       # Default analysis configuration
n_shot="5"                      # Default analysis configuration
aggregation="sum"               # Default parameter from the original paper
deep_distance_type="fc-conc"    # Default parameter from the original paper
backbone_layer_dim="[]"         # Embedding module is removed
learning_rate="0.001"           # Best parameter from tuning
backbone_weight_decay="0.001"   # Best parameter from tuning
backbone_dropout="0.0"          # Best parameter from tuning 

for deep_distance_layer_size in "${deep_distance_layer_sizes[@]}"; do
    ( 
        conda run -n few --no-capture-output --live-stream python run.py \
            exp.name=ablation_${dataset} \
            method=${model} \
            dataset=${dataset} \
            n_way=${n_way} \
            n_shot=${n_shot} \
            method.representative_aggregation=${aggregation} \
            method.deep_distance_type=${deep_distance_type} \
            method.deep_distance_layer_sizes="${deep_distance_layer_size}" \
            backbone.layer_dim="${backbone_layer_dim}" \
            method.learning_rate=${learning_rate} \
            method.backbone_weight_decay=${backbone_weight_decay} \
            backbone.dropout=${backbone_dropout}
    )
done
