import logging
import os
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.loader.registry import StateDictRegistry, DummyRegistry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()
SKIP_AUDIO_DECODE = os.getenv("LTX_SKIP_AUDIO_DECODE", "1") == "1"
# Final VAE decode is the most crash-prone step on this Windows box. Force it
# to CPU so we trade speed for reliability and can actually finish i2v renders.
VAE_DECODE_DEVICE = "cpu"
REGISTRY_MODE = os.getenv("LTX_REGISTRY_MODE", "dummy").strip().lower()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        # CPU state-dict caching lowers VRAM pressure but can explode Windows page-file
        # usage on this workstation. Prefer the dummy registry unless explicitly overridden.
        self.registry = StateDictRegistry() if REGISTRY_MODE == "state_dict" else DummyRegistry()
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            registry=self.registry,
            quantization=quantization,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.stage_1_model_ledger.text_encoder(device="cpu")
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
        
        # Ensure context embeddings are pushed to CUDA before inference (required if Text Encoder was offloaded to CPU)
        if v_context_p is not None: v_context_p = v_context_p.to(self.device)
        if a_context_p is not None: a_context_p = a_context_p.to(self.device)
        if v_context_n is not None: v_context_n = v_context_n.to(self.device)
        if a_context_n is not None: a_context_n = a_context_n.to(self.device)

        # No need for synchronize if it was on CPU, but doesn't hurt.
        del text_encoder
        cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        
        # Free System RAM by popping the 22B base transformer state dict from registry
        # now that it's already moved to the GPU.
        from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP
        self.registry.pop([self.stage_1_model_ledger.checkpoint_path], LTXV_MODEL_COMFY_RENAMING_MAP)
        cleanup_memory()

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        torch.cuda.synchronize()
        if transformer is not None:
            print(f"[LTX-2] Stage 1 transformer deleted.")
            del transformer
        
        print(f"[LTX-2] Running cleanup_memory after Stage 1 denoising...")
        cleanup_memory()
        print(f"[LTX-2] Cleanup complete. VRAM/RAM freed.")

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        torch.cuda.synchronize()
        print(f"[LTX-2] Running cleanup_memory after upsampling...")
        cleanup_memory()
        print(f"[LTX-2] Cleanup complete. VRAM/RAM freed.")

        # Build Stage 2 components before building Stage 2 Transformer to avoid spikes
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        
        # Now that we have conditionings and upscaled latent, we can delete the encoder
        del video_encoder
        print(f"[LTX-2] Running cleanup_memory after video encoder deletion...")
        cleanup_memory()
        print(f"[LTX-2] Cleanup complete. VRAM/RAM freed.")

        transformer = self.stage_2_model_ledger.transformer()

        # Free System RAM again after refined transformer is on GPU.
        # This is critical to leave room for the large VAE decode.
        from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP
        print(f"[LTX-2] Popping Stage 2 transformer from registry...")
        self.registry.pop([self.stage_2_model_ledger.checkpoint_path], LTXV_MODEL_COMFY_RENAMING_MAP)
        print(f"[LTX-2] Running cleanup_memory after Stage 2 transformer pop...")
        cleanup_memory()
        print(f"[LTX-2] Cleanup complete. VRAM/RAM freed.")

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        torch.cuda.synchronize()
        if transformer is not None:
            print(f"[LTX-2] Stage 2 transformer deleted.")
            del transformer
        print(f"[LTX-2] Running cleanup_memory after Stage 2 denoising...")
        cleanup_memory()
        print(f"[LTX-2] Cleanup complete. VRAM/RAM freed.")

        # Use more aggressive tiling for VAE decode to avoid VRAM spikes at the
        # end of generation. 256px tiles are slower but more robust on this box.
        from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, VAE_DECODER_COMFY_KEYS_FILTER
        custom_tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=128),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24)
        )
        from ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, VOCODER_COMFY_KEYS_FILTER

        decode_device = torch.device("cpu") if VAE_DECODE_DEVICE == "cpu" else torch.device(self.device)
        decode_generator = None if decode_device.type == "cpu" else generator
        video_latent_for_decode = video_state.latent
        del video_state
        if SKIP_AUDIO_DECODE:
            del audio_state
        cleanup_memory()

        print(
            f"[LTX-2] Starting VAE decoding ({num_frames}-frame sequence, "
            f"{custom_tiling.spatial_config.tile_size_in_pixels}px tiling, device={decode_device})..."
        )
        print(f"[LTX-2] Building VAE video decoder...")
        if decode_device.type == "cpu":
            print("[LTX-2] Offloading final video decode to CPU for stability.")
            video_latent_for_decode = video_latent_for_decode.to("cpu", dtype=self.dtype)
            cleanup_memory()
            vae_video_decoder = (
                self.stage_2_model_ledger.vae_decoder_builder.build(device=torch.device("cpu"), dtype=self.dtype)
                .to("cpu")
                .eval()
            )
        else:
            vae_video_decoder = self.stage_2_model_ledger.video_decoder()
        print(f"[LTX-2] Video VAE decoder built. Popping from registry...")
        self.registry.pop([self.stage_2_model_ledger.checkpoint_path], VAE_DECODER_COMFY_KEYS_FILTER)
        
        print(f"[LTX-2] Creating VAE decoding iterator...")
        decoded_video = vae_decode_video(
            video_latent_for_decode, vae_video_decoder, custom_tiling, decode_generator
        )
        print(f"[LTX-2] VAE decoder built and iterator created.")

        if SKIP_AUDIO_DECODE:
            print("[LTX-2] Skipping audio decode to preserve memory for video quality.")
            decoded_audio = torch.zeros((2, 16000), dtype=torch.float32)
        else:
            try:
                print(f"[LTX-2] Building VAE audio decoder...")
                vae_audio_decoder = self.stage_2_model_ledger.audio_decoder()
                print(f"[LTX-2] Audio VAE decoder built. Popping from registry...")
                self.registry.pop([self.stage_2_model_ledger.checkpoint_path], AUDIO_VAE_DECODER_COMFY_KEYS_FILTER)
                
                vocoder = self.stage_2_model_ledger.vocoder()
                self.registry.pop([self.stage_2_model_ledger.checkpoint_path], VOCODER_COMFY_KEYS_FILTER)

                decoded_audio = vae_decode_audio(
                    audio_state.latent, vae_audio_decoder, vocoder
                )
                print(f"[LTX-2] Audio decoded.")
            except Exception as e:
                print(f"[LTX-2] Audio decoding failed/skipped (Incompatible architecture): {e}")
                decoded_audio = torch.zeros((2, 16000), dtype=torch.float32)


        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        quantization=args.quantization,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
