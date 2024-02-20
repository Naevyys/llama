#!/bin/bash
source /etc/profile
module load anaconda/2023a
pwd
python3.9 -u batch_prompt_inference.py -f ./test_prompt.json -o ./test_output.json -m ./Llama-2-7b-chat-hf/