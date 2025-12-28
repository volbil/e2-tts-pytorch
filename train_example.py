import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from datasets import load_dataset

from e2_tts_pytorch.trainer import HFDataset, E2Trainer


hl_gauss_loss = {
    "min_value": 0.0,
    "max_value": 5.0,
    "num_bins": 32,
    "sigma": 0.5,
    "clamp_to_range": True,
}

duration_predictor = DurationPredictor(
    hl_gauss_loss=hl_gauss_loss,
    transformer=dict(
        # dim=512,
        # depth=6,
        dim=256,
        depth=4,
        heads=4,
        dim_head=64,
    ),
)

e2tts = E2TTS(
    duration_predictor=duration_predictor,
    # transformer=dict(
    #     dim=512,
    #     depth=12,
    # ),
    transformer=dict(
        dim=512,  # Reduce from 1024 if memory issues
        depth=8,  # Reduce from 12-16 for faster training
        heads=8,
        dim_head=64,
        ff_mult=4,
        dropout=0.1,
        # Text conditioning
        dim_text=256,  # Half of dim by default
        text_depth=4,  # Reduce text layers
        # MPS-specific settings
        abs_pos_emb=True,  # Absolute positional embeddings work well
        num_registers=32,  # Keep registers for performance
        # Attention settings - keep these simple for MPS
        attn_kwargs=dict(
            gate_value_heads=True,
            softclamp_logits=True,
        ),
        # Disable complex features that may cause MPS issues
        num_residual_streams=1,  # Disable hyper-connections for stability
        attn_laser=False,  # Disable laser attention
        attn_fourier_embed_input=False,  # Disable fourier embedding
    ),
)

# train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])
train_dataset = HFDataset(load_dataset("patriotyk/filatov_24000")["train"])

trainer = E2Trainer(
    e2tts,
    num_warmup_steps=1000,
    # num_warmup_steps=20000,
    # grad_accumulation_steps=1,
    grad_accumulation_steps=8,
    checkpoint_path="e2tts.pt",
    log_file="e2tts.txt",
    sample_rate=24000,
)

if __name__ == "__main__":
    epochs = 10
    batch_size = 4
    # batch_size = 32

    trainer.train(
        train_dataset,
        epochs,
        batch_size,
        save_step=1000,
        num_workers=4,
    )
