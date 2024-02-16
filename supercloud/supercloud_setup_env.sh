# Loading the required module
mkdir /state/partition1/user/$USER
export TMPDIR=/state/partition1/user/$USER
source /etc/profile
module load anaconda/2023a

pip install --user --no-cache-dir transformers
pip install --user --no-cache-dir bitsandbytes
pip install --user --no-cache-dir torch datasets accelerate
#pip install --user --no-cache-dir random
pip install --user --no-cache-dir pandas
pip install --user --no-cache-dir torch
pip install --user --no-cache-dir huggingface_hub[cli]

git config --global credential.helper store
export HUGGINGFACE_TOKEN=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
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

#rm -rf /state/partition1/user/$USER