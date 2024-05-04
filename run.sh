#!/bin/bash

echo "Enabling Anaconda commands using: . /c/Anacomda3/etc/profile.d/conda.sh"

. /c/Anaconda3/etc/profile.d/conda.sh

echo "Activating Anaconda base environment using: conda activate"

conda activate

echo "Running all scripts"

echo "Running pre_process.py"

python pre_process.py

echo "pre_process.py complete"


echo "Running final_analysis.py"

python final_analysis.py

echo "final_analysis.py complete"
