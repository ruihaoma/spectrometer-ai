# SpectrumUNetTransformer1D training summary

- script: train_spectrum_unet_transformer_1d.py
- created_at: 2026-05-28T23:43:04
- dataset_dir: data/processed/relative_calib_mixed_v1
- train_size: 64005
- val_size: 8001
- device: cuda
- input_shape: [B, 4, 2501]
- output_shape: [B, 2501]
- model: SpectrumUNetTransformer1D
- total_params: 2211073
- trainable_params: 2211073
- loss: weighted_l1 + 0.1 * grad_l1 + 0.05 * mse
- optimizer: AdamW
- epochs: 350
- batch_size: 64
- learning_rate: 0.0002
- weight_decay: 0.0001
