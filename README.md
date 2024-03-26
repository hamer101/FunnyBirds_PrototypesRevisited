# FunnyBirds_PrototypesRevisited
This code implements the solutions described in "Revisiting FunnyBirds evaluation framework for prototypical parts networks" by Szymon Opłatek, Dawid Rymarczyk and Bartosz Zieliński.

## Setup

The suggested folder structure is:

    your_desired_dir/
    ├── FunnyBirds/
    │   └── ...                                      # Unchanged FunnyBirds dataset 
    ├── FunnyBirdsFramework/
    │   ├── explainers/
    │   │   └── explainer_wrapper.py                 # Modified
    │   ├── models/
    │   │   ├── model_wrapper.py                     # Modified
    │   │   ├── ppnet.py                             # Modified
    │   │   └── ...                                  # All of the remaining FunnyBirdsFramework/models files
    │   ├── evaluate_explainability.py               # Modified
    │   └── ...                                      # All of the remaining FunnyBirdsFramework files
    ├── ProtoPNet/
    │   ├── main_funnybirds_multitarget.py           # Appended
    │   ├── push_funnybirds_multitarget.py           # Appended
    │   ├── settings_funnybirds_multitarget.py       # Appended
    │   ├── train_and_test_funnybirds_multitarget.py # Appended
    │   └── ...                                      # All of the remaining ProtoPNet files
    └── model_selection.toml

The script below, executed from this repos' direcotry, will set up the suggested folder structure (+ external resources) within a desired location:

    #!/bin/bash
    project_dir="your_desired_dir"

    wget -P $project_dir download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
    unzip FunnyBirds.zip
    rm FunnyBirds.zip

    git clone https://github.com/visinf/funnybirds-framework.git $project_dir
    cp -f ./FunnyBirdsFramework/evaluate_explainability.py $project_dir/FunnyBirdsFramework/evaluate_explainability.py
    cp -f ./FunnyBirdsFramework/models/model_wrapper.py $project_dir/FunnyBirdsFramework/models/model_wrapper.py
    cp -f ./FunnyBirdsFramework/models/ppnet.py $project_dir/FunnyBirdsFramework/models/ppnet.py
    cp -f ./FunnyBirdsFramework/explainers/explainer_wrapper.py $project_dir/FunnyBirdsFramework/explainers/explainer_wrapper.py

    git clone https://github.com/cfchen-duke/ProtoPNet.git $project_dir
    cp ./ProtoPNet/main_funnybirds_multitarget.py $project_dir/ProtoPNet/main_funnybirds_multitarget.py
    cp ./ProtoPNet/push_funnybirds_multitarget.py $project_dir/ProtoPNet/push_funnybirds_multitarget.py
    cp ./ProtoPNet/settings_funnybirds_multitarget.py $project_dir/ProtoPNet/settings_funnybirds_multitarget.py
    cp ./ProtoPNet/train_and_test_funnybirds_multitarget.py $project_dir/ProtoPNet/train_and_test_funnybirds_multitarget.py

    cp ./model_selection.toml $project_dir/model_selection.toml

## Run

Firstly, make sure you filled out the `model_selection.toml` file ***that got copied into your project directory.***

To train the ProtoPNet, you have to run the `main_funnybirds_multitarget.py` the same way as specified in (ProtoPNet's repo)[https://github.com/cfchen-duke/ProtoPNet].

To run the evaluation, run the command below (don't forget to properly fill `paths` section of .toml config file with your model's paths). Explainer available names are `SSMExplainer` and `SSMAttriblikePExplainer`. You should specify the number of gpu to be used.

`python your_desired_dir/FunnyBirdsFramework/evaluate_explainability.py --data "your_desired_dir/FunnyBirds/" --model ppnet --explainer ... --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu ... --batch_size 100`

Results will be get outputted directly to your CLI.## Analys