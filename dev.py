import torch.multiprocessing as mp
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import os, torch
import fire
from llama import Transformer, ModelArgs
from llama_wrapper.AlteredTransformer import AlteredTransformer


model_parallel_size = None


vocab_size = 5000
batch_size = 1
prompt_length = 20
start_position = 0
    
positional_embedding_alteration_params = {
    "zero": dict(indices=(4, 9)),  # Contexts start at position 4 and finish at position 9
    "median": dict(indices=(4, 9)),  # Contexts start at position 4 and finish at position 9
    "reset": dict(indices=[4, 6, 9]),  # Context 1 starts at 4 and finishes at 5, context 2 starts at 6 and finishes at 8
}

mode = "zero"


if __name__=="__main__":
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    print("Reached after parallel model initialization")
    
    params = ModelArgs(n_layers=3, n_heads=2, vocab_size=vocab_size)

    model = AlteredTransformer(params)
    model.switch_mode(mode, **positional_embedding_alteration_params.get(mode, dict()))
    
    tokens = torch.randint(0, vocab_size, (batch_size, prompt_length))
    
    print(model.forward(tokens, start_position).shape)