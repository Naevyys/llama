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

if __name__=="__main__":
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    print("Reached after parallel model initialization")

    params = ModelArgs(n_layers=3, n_heads=5, vocab_size=5000)

    model = AlteredTransformer(params)