# Publication model artifacts

These directories contain the exact U-Net checkpoints and training records used by the publication figure-generation workflow. The YAML-driven training pipeline can independently retrain the models, but backend-dependent numerical differences mean retrained weights are not expected to match these files bit-for-bit.

SHA-256 checksums for `unet_best.pth`:

- Square-Net: `7ca9e7beadff12a0b4bbafc4116029c257b101d950881e9fd5c6f14a2f32c3b5`
- Hex-Net: `9f03deaf71884a852aaf52250d4f3e0a29f90da2db9519687ab083facc67a3d0`
- Blob-Net: `2fa1cfd6caba4de9a3d5f1d5728d33266397c0b882d78d7d3dd600dadabfd041`
- Dense-random control: `0d7c9838668c496a559df4d54e09280c83f312649a4658eebd1b522d61128ac1`
