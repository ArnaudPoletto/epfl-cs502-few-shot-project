#!/bin/bash

declare -a datasets=("swissprot" "tabula_muris")
declare -a deep_distance_layer_sizes=("[128, 64, 64, 32, 32, 8, 1]" "[128, 64, 32, 1]" "[128, 1]")
declare -a backbone_layer_dims=("[128, 128, 128, 128]" "[64, 64]" "[16]")
declare -a learning_rates=("0.01" "0.001" "0.0001")
declare -a backbone_weight_decays=("0.0" "0.001")
declare -a backbone_dropouts=("0.0" "0.1" "0.2")

for dataset in "${datasets[@]}"; do
    for layer_size in "${deep_distance_layer_sizes[@]}"; do
        for backbone_dim in "${backbone_layer_dims[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for backbone_weight_decay in "${backbone_weight_decays[@]}"; do
                    for backbone_dropout in "${backbone_dropouts[@]}"; do
                        ( 
                            conda run -n few --no-capture-output --live-stream python \
                                run.py exp.name=tuning_${dataset} \
                                method=relationnet \
                                dataset=${dataset} \
                                n_way=5 \
                                n_shot=5 \
                                method.stop_epoch=30 \
                                method.representative_aggregation="sum" \
                                method.deep_distance_type="fc-conc" \
                                method.deep_distance_layer_sizes="${layer_size}" \
                                backbone.layer_dim="${backbone_dim}" \
                                method.learning_rate=${learning_rate} \
                                method.backbone_weight_decay=${backbone_weight_decay} \
                                backbone.dropout=${backbone_dropout} 
                        )
                    done
                done
            done
        done
    done
done