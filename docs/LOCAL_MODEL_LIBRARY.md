# Local Model Library

Updated: 2026-05-26

Local app model root:

```text
D:\VIDEO_GENS\wan-ltx-video-studio\models
```

Source inspected:

```text
D:\IMAGE_GENERATORS\Comfy_UI_Furkan_V61\ComfyUI\models
```

The source library is large and mixed-purpose. Only WAN 2.2 and LTX 2.3 assets that look directly useful for this project were copied into the app model root. Personal/remix/legacy assets and unrelated model families were left in place.

The copied local set is 24 files, about 190.336 GB.

## Copied Assets

### WAN 2.2 Core

```text
models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
models/text_encoders/umt5_xxl_fp16.safetensors
models/vae/wan2.2_vae.safetensors
models/vae/wan_2.1_vae.safetensors
```

### WAN 2.2 GGUF

```text
models/unet/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf
models/unet/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf
models/unet/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf
models/unet/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf
```

### WAN 2.2 Lightning LoRAs

```text
models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors
models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors
models/loras/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors
models/loras/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors
```

### LTX 2.3

```text
models/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v2.safetensors
models/diffusion_models/ltx-2-3-22b-dev_transformer_only_fp8_input_scaled.safetensors
models/text_encoders/ltx-2.3_text_projection_bf16.safetensors
models/vae/LTX23_video_vae_bf16.safetensors
models/vae/LTX23_audio_vae_bf16.safetensors
models/vae/taeltx2_3.safetensors
models/loras/ltx-2.3-22b-distilled-lora-384.safetensors
models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
models/checkpoints/LTX-2.3-distilled-Q6_K.gguf
```

## ComfyUI Visibility Reference

This section is retained only to document how the copied model files appeared to ComfyUI during research. The app should not require ComfyUI folder structure or launch ComfyUI as a backend.

Tracked config:

```text
config/comfy_extra_model_paths.yaml
```

Verified with:

```text
runtimes/comfyui-sec-v89
ComfyUI --extra-model-paths-config config\comfy_extra_model_paths.yaml --highvram --use-sage-attention
```

Visible model folders:

```text
/models/diffusion_models: 5
/models/text_encoders: 3
/models/vae: 5
/models/loras: 5
/models/latent_upscale_models: 1
/models/unet_gguf: 4
/models/clip_gguf: 0
/models/checkpoints: 0
```

ComfyUI-GGUF registers `unet_gguf` and `clip_gguf` virtual folders. The WAN `.gguf` files are visible through `/models/unet_gguf` and `UnetLoaderGGUF`, not through `/models/diffusion_models`.

The copied LTX `.gguf` checkpoint is not visible through `/models/checkpoints` in this runtime. Treat it as present on disk but not yet validated for execution until the LTX workflow adapter is installed and tested.

## App Implications

- The first real generation test should use the direct renderer with the smallest practical WAN profile that validates model loading on the RTX 5090.
- WAN A14B FP8 safetensors are ready for high/low expert workflow testing.
- WAN A14B Q8 GGUF is present for future direct GGUF loader research.
- WAN Lightning turbo mode can be tested with explicit high/low LoRA pairing.
- LTX 2.3 assets are present, but advanced LTX execution still needs a direct runtime path validated.
