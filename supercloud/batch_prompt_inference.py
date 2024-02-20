from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig)
#import bitsandbytes
from accelerate import infer_auto_device_map
from tqdm import tqdm
from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd 
import sys
import re
import argparse
import json
from api_token import API_TOKEN

class LlamaInterface:
    def __init__(self, token, model="meta-llama/Llama-2-7b-hf"):
        '''
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        '''
        
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, token=token, device_map=self.device).to(self.device)
        
    def generate_text(self, prompt, max_length=500):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_tokens_size = len(inputs['input_ids'][0])
        actual_max_length = input_tokens_size+max_length
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=actual_max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0]
            
    def batch_generate(self, prompt_list):
        for prompt in prompt_list:
            generated_text = self.generate_text(prompt["prompt"])
            formatted_generation = generated_text.lstrip(prompt["prompt"])
            formatted_generation = formatted_generation.strip("\n")
            prompt["output"] = formatted_generation

def main():
    api_token = API_TOKEN

    # Create the parser
    parser = argparse.ArgumentParser(description='Process a JSON file.')
    
    # Add the arguments
    parser.add_argument('-f', '--file', help='Input JSON file', required=True)
    parser.add_argument('-o', '--output', help='Output File Location', required=True)
    parser.add_argument('-m', '--model', help='LLM under test for generation', required=False)
    
    # Execute the parse_args() method
    args = parser.parse_args()
    
    # Read and process the JSON file
    input_file_path = args.file
    try:
        with open(input_file_path, 'r') as json_file:
            prompt_data = json.load(json_file)
            # Now data holds the JSON object, you can process it as needed
            #print(prompt_data)  # Just to demonstrate we have loaded the JSON
    except FileNotFoundError:
        print(f"The file {input_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file {input_file_path} is not valid JSON.")
        
    output_file_path = args.output
    
    if args.model:
        model_path = args.model
    else:
        model_path = "meta-llama/Llama-2-7b-hf"
    
    model_uut = LlamaInterface(api_token, model_path)
    
    model_uut.batch_generate(prompt_data)
    print(prompt_data)
    

if __name__ == "__main__":
    main()