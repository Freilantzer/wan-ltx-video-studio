# API Workflows

These JSON files are ComfyUI API-format workflows used for runtime probing and adapter development. They are not the app UX and should not be exposed directly to users.

## Pressure Tests

- `wan22_i2v_lightning_a14b_pressure_test.json`

  WAN 2.2 A14B image-to-video with high-noise and low-noise FP8 experts plus Lightning LoRAs. This is a memory pressure test, not a recommended default preset. On the RTX 5090 32 GB target, the first run with Comfy `--highvram --use-sage-attention` peaked around 32.0 GB VRAM.

  Use it only to validate model wiring, LoRA pairing, and two-phase A14B execution. The app should prefer lower-memory profiles for real users.
