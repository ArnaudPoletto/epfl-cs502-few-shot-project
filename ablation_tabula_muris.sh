#!/bin/bash

# Define the list of representative_aggregations, deep_distance_types, and layer dimensions
declare -a representative_aggregations=("mean", "sum")
declare -a deep_distance_types=("cosine" "euclidean" "f1-conc" "f1-diff" "l1")
declare -a deep_distance_layer_sizes=("[128, 64, 64, 32, 32, 8, 1]" "[128, 1]")
declare -a backbone_layer_dims=("[64, 64]" "[16]")

# Iterate over each representative_aggregation
for aggregation in "${representative_aggregations[@]}"; do
    # Then iterate over each deep_distance_type
    for distance_type in "${deep_distance_types[@]}"; do
        # Then iterate over each deep_distance_layer_size
        for layer_size in "${deep_distance_layer_sizes[@]}"; do
            # Then iterate over each backbone_layer_dim
            for backbone_dim in "${backbone_layer_dims[@]}"; do
                # Run the command with the specified parameters
                ( conda run -n few --no-capture-output --live-stream python run.py exp.name=ablation_tabula_muris method=relationnet dataset=tabula_muris n_way=5 n_shot=5 method.representative_aggregation=$aggregation method.deep_distance_type=$distance_type method.deep_distance_layer_sizes="$layer_size" backbone.layer_dim="$backbone_dim" )
            done
        done
    done
done
