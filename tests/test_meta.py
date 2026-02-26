import torch
import sys
sys.path.append(r'D:\antigravity\ltx2_temp\packages\ltx-core\src')
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from transformers import Gemma3Config

gc = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
gc.text_config.rope_local_base_freq = 10000.0
gc.text_config.rope_theta = 1000000.0

with torch.device('meta'):
    model = Gemma3ForConditionalGeneration(gc)

target_device = 'cpu'
l_model = model.model.language_model
v_model = model.model.vision_tower.vision_model

v_model.embeddings.register_buffer('position_ids', torch.zeros((1, 10), dtype=torch.long, device=target_device))
l_model.embed_tokens.register_buffer('embed_scale', torch.tensor(1.0, device=target_device))
freqs = torch.tensor([1.0], device=target_device)

l_model.rotary_emb.register_buffer('sliding_attention_inv_freq', freqs)
l_model.rotary_emb.register_buffer('sliding_attention_original_inv_freq', freqs)
l_model.rotary_emb.register_buffer('full_attention_inv_freq', freqs)
l_model.rotary_emb.register_buffer('full_attention_original_inv_freq', freqs)

meta_buffers = [k for k, v in model.named_buffers() if v.is_meta]
print(f'Meta buffers remaining: {meta_buffers}')
