# CS-502 Deep learning in Biomedicine: Enhancing Few-Shot Learning in Biomedicine: Benchmarking Relation Network

<p align="center">
Ozan Güven (297076) - ozan.guven@epfl.ch <br>
Arnaud Poletto (302411) - arnaud.poletto@epfl.ch
</p>

## Description

The report "*Enhancing Few-Shot Learning in Biomedicine: Benchmarking Relation Network*" investigates the application of the Relation Network, a deep learning model designed for few-shot learning scenarios, in the field of biomedicine. The study focuses on customizing and optimizing the Relation Network for biomedical data analysis, specifically comparing its efficacy against other few-shot learning algorithms. This comparison is crucial for understanding the model’s strengths and areas for improvement within biomedical data analysis. The work involves the use of two key datasets: Tabula Muris and SwissProt, to evaluate the Relation Network's performance in diverse biomedical scenarios.

This repository contains the code of the implementations of each method, notably the Relation Network, and used to run the experiments and generate the results presented in the report.

## Repository Contents

The repository is structured as follows:

* [📁 backbones](./backbones): Contains code for the backbones used in the experiments.
* [📁 conf](./conf): Contains configuration files for the datasets and the methods used in the experiments.
    * [📁 dataset](./conf/dataset): Contains configuration files for the datasets, namely [📄 swissprot.yaml](./conf/dataset/swissprot.yaml) and [📄 tabula_muris.yaml](./conf/dataset/tabula_muris.yaml).
    * [📁 method](./conf/method): Contains configuration files for the methods. Notably, [📄 relationnet.yaml](./conf/method/relationnet.yaml), the configuration file for the Relation Network.
* [📁 datasets](./datasets): Contains code for loading the datasets used in the experiments.
* [📁 generated](./generated): Contains generated plots and images.
* [📁 methods](./methods): Contains code for the methods used in the experiments. Notably, [📄 relationnet.py](./methods/relationnet.py) the file implementation of the Relation Network.
* [📁 notebooks](./notebooks): Contains notebooks used to analyze results and generate plots.
    * [📄 find_best_tuning_results.ipynb](./notebooks/find_best_tuning_results.ipynb): Identifies the best hyperparameter tuning results for the Relation Network for each dataset.
    * [📄 ways_shots_analysis.ipynb](./notebooks/ways_shots_analysis.ipynb): Analyzes the performance of the available methods for different number of ways and shots.
* [📁 results](./results): Contains the results of the experiments.
    * **TODO: folders for each experiment?**
* [📁 scripts](./scripts): Contains scripts used to run the experiments.
* [📁 utils](./utils): Contains utility code used in the experiments.
* [📄 run.py](./run.py): Main script used to run the experiments.



## Requirements

## Usage


## Repository Structure

