#!/bin/bash

dataset="tabula_muris"
model="relationnet"
n_way="5"                                   # Default analysis configuration
n_shot="5"                                  # Default analysis configuration
representative_aggregation="sum"            # Default parameter from the original paper
deep_distance_type="fc-conc"                # Default parameter from the original paper
deep_distance_layer_size="[128, 1]"         # Best parameter from tuning
backbone_layer_dim="[16]"                   # Best parameter from tuning
learning_rate="0.001"                       # Best parameter from tuning
backbone_weight_decay="0.001"               # Best parameter from tuning
backbone_dropout="0.0"                      # Best parameter from tuning

conda run -n few --no-capture-output --live-stream python run.py \
    exp.name=run_default_${dataset} \
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
