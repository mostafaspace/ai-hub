import torch


class PixArtAlphaTextProjection(torch.nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        # HOTFIX: LTX 2.3 Gemma embeddings natively match the inner dimension
        # and do not use the legacy T5 projection layer. If the layer weights
        # are zero (uninitialized) or shapes mismatch, safely bypass projection.
        if not hasattr(self, "_is_uninitialized"):
            self._is_uninitialized = torch.all(self.linear_1.weight == 0).item()
            
        if self._is_uninitialized or caption.shape[-1] != self.linear_1.in_features:
            return caption

        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
