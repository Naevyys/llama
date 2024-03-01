import torch
from llama.model import Transformer, ModelArgs


class AlteredTransformer(Transformer):
    def  __init__(self, params: ModelArgs, debug:bool=False):
        super().__init__(params)
        self.alteration_mode = None
        self.alteration_kwargs = dict()
        self.debug = debug

    def switch_debug(self):
        self.debug = not self.debug
        
    def switch_mode(self, mode:str=None, **kwargs):
        '''
        Switch the model's alteration mode.
        
        @param mode: Which alteration mode to use. Valid values are {None, "median", "reset"}.
        @param kwargs: Any kwargs needed for the specified alteration mode.

        @returns None
        '''
        assert mode is None or mode in {"zero", "median", "reset"}, "Invalid mode provided"
        # Note: I don't think it makes any sense to implement zero-patching to replace rotary positional embedding. But I can always add it later if we want to try it as well.

        error_msg = "Provide {} argument for mode {}."

        # Asserting the validity of additional arguments for each mode
        if mode in {"median", "reset"}:
            indices = kwargs.get("indices", None)
            assert indices is not None, error_msg.format("indices", mode)

            if mode == "reset":
                assert isinstance(indices, list) and (len(indices) > 1), "Indices must be a list of at least two elements."
            if mode in {"zero", "median"}:
                assert isinstance(indices, tuple) and (len(indices) == 2), "Indices must be a tuple of two elements."
            
            previous = 0
            for i in indices:
                assert isinstance(i, int) and i >= 0, "Indices provided must be non-negative integers"
                assert i > previous, "Each index must be greater than the previous one."
                previous = i
        
        self.alteration_mode = mode
        self.alteration_kwargs = kwargs

    def alter_positional_embedding(self, freqs_cis):
        '''
        Alter the positional embedding using the approach specified in self.alteration_mode.

        @param freqs_cis: Frequencies to alter.

        @returns torch.Tensor of the same shape as freqs_cis, which are the new positional embeddings to use.
        '''
        new_freqs_cis = freqs_cis.detach().clone()
        if self.alteration_mode == "zero":
            start, stop = self.alteration_kwargs["indices"]
            new_freqs_cis[start:stop, :] = 0+0j
            if self.debug:
                print("Pos embedding zero patching alteration:", new_freqs_cis[start:stop, :3])
        if self.alteration_mode == "median":
            start, stop = self.alteration_kwargs["indices"]
            median = start + (stop - start) // 2
            new_freqs_cis[start:stop, :] = freqs_cis[median, :]
            if self.debug:
                print("Pos embedding median patching alteration:", new_freqs_cis[start:stop, :3])
        if self.alteration_mode == "reset":
            indices = self.alteration_kwargs["indices"]
            for i in range(len(indices) - 1):
                new_freqs_cis[indices[i]:indices[i+1], :] = freqs_cis[indices[0]:indices[0] + indices[i+1] - indices[i], :]
            if self.debug:
                print("Pos embedding reset patching alteration:", new_freqs_cis[indices[0]:indices[-1], :3])
        
        # Otherwise return the same vector as freqs_cis
        return new_freqs_cis

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):  # Let's overwrite the standard Transformer forward
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        if self.debug:
            print("Pos embedding shape before alteration:", freqs_cis.shape)
        freqs_cis = self.alter_positional_embedding(freqs_cis)  # Modification compared to the forward of the superclass
        if self.debug:
            print("Pos embedding shape after alteration:", freqs_cis.shape)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output