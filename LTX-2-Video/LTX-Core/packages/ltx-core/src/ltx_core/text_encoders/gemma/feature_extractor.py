import torch

from ltx_core.model.model_protocol import ModelConfigurator


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    """
    Feature extractor module for Gemma models.
    Supports separate video and audio projections (LTX-2.3) or a single shared projection (LTX-2.0).
    """

    def __init__(
        self,
        in_features: int = 3840 * 49,
        video_out_features: int = 3840,
        audio_out_features: int = 3840,
        use_separate_projections: bool = False,
    ) -> None:
        super().__init__()
        self.use_separate_projections = use_separate_projections
        if use_separate_projections:
            self.video_aggregate_embed = torch.nn.Linear(in_features, video_out_features, bias=True)
            self.audio_aggregate_embed = torch.nn.Linear(in_features, audio_out_features, bias=True)
        else:
            self.aggregate_embed = torch.nn.Linear(in_features, video_out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_separate_projections:
            return self.video_aggregate_embed(x), self.audio_aggregate_embed(x)
        return self.aggregate_embed(x)

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], config: dict) -> "GemmaFeaturesExtractorProjLinear":
        transformer_config = config.get("transformer", {})
        # LTX-2.3 22B uses separate projections with different output dimensions
        use_separate = transformer_config.get("caption_proj_before_connector", False)
        video_dim = transformer_config.get("connector_num_attention_heads", 30) * transformer_config.get("connector_attention_head_dim", 128)
        audio_dim = transformer_config.get("audio_connector_num_attention_heads", 30) * transformer_config.get("audio_connector_attention_head_dim", 128)
        
        # If not explicit, check if we are in 2.3 era by connector layer count or other keys
        if not use_separate and "connector_num_layers" in transformer_config:
             use_separate = True
             video_dim = transformer_config.get("connector_num_attention_heads", 32) * 128
             audio_dim = transformer_config.get("audio_connector_num_attention_heads", 32) * 64

        return cls(
            video_out_features=video_dim,
            audio_out_features=audio_dim,
            use_separate_projections=use_separate,
        )
