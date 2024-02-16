#!/bin/bash

# Loading the required modules
source /etc/profile
module load anaconda/2023a-pytorch
module load nccl/2.18.1-cuda11.8  # Also loads module cuda/11.8 as it is a requirement

# Installing packages needed
mkdir /state/partition1/user/$USER
export TMPDIR=/state/partition1/user/$USER
pip install --user --no-cache-dir -r ~/llama/requirements.txt  # Install requirements.txt
pip3 install --user --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Install pytorch with cuda 11.8. NOTE: When I run it after the rest it doesn't install anything new, maybe I don't need this command at all
#pip install --user --no-cache-dir numpy pandas matplotlib seaborn ipykernel  # Check if I need any of that

#git config --global credential.helper store
#export HUGGINGFACE_TOKEN=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
#huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

#rm -rf /state/partition1/user/$USER