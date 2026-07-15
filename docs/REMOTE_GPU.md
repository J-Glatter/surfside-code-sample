# Renting a GPU for the checkpoints (no local GPU needed)

The validation checkpoints (`docs/CHECKPOINTS.md`) run identically on a rented
Linux CUDA box. Cost reality: SD 1.5 is light — **Checkpoint A/B is ~an hour
on a consumer-card pod (~$1); Checkpoint C with LoRA training ~2–3 hours
(~$2)**. Prices below are mid-2026 reference points from the project handover —
re-verify on the day.

## Picking a provider

| Provider | Fit | Notes |
|---|---|---|
| **RunPod** (community cloud) | **Recommended for checkpoints** | RTX 4090 ~$0.34/hr community / ~$0.69 secure; per-minute billing; SSH + persistent volume support |
| Vast.ai | Cheapest, more variance | Marketplace hardware — fine for smoke tests |
| Lambda | Simple, often sold out | ~$0.50+/hr consumer tiers |
| Modal / fal / Replicate | **Not for this** — serverless per-call | The Phase-5 SaaS shape (handover §16–17), not interactive validation |

Any pod with an RTX 3090/4090, a **PyTorch template** (CUDA torch
preinstalled), SSH access, and ≥30 GB disk (SD weights + kohya + outputs).

**Sizing: don't pay for VRAM.** SD 1.5 uses ~4–6 GB inference, ~8–10 GB
training — a 24 GB 4090 already has 3× headroom, so bigger-VRAM tiers
(5090/48 GB-class) buy nothing here. A pricier card must be >(price ratio)×
faster *on saturated GPU work* to win, and only LoRA training and
selector-driven animation saturate; everything else is model-load/IO overhead
that bills identically per second — favouring the cheaper card. Rough 4090
times: sprite ~2–4 s compute (~15–30 s warm wall), refine round ~2 min, LoRA
training 15–25 min, selector walk cycle (8×100 + ControlNet) ~20–30 min.
Note **serverless ≠ pods**: serverless is the per-request endpoint shape
(Phase 5); checkpoints want an SSH pod, which is also ~half the hourly rate.

## Launch → bootstrap (minutes)

1. Create the pod (PyTorch template), note its SSH command.
2. If the repo is private, mint a fine-grained GitHub PAT (read-only, this
   repo) and use it in the clone URL.
3. On the pod:

```bash
export SPRITEFORGE_REPO=https://<token>@github.com/J-Glatter/surfside-code-sample
git clone "$SPRITEFORGE_REPO" spriteforge && cd spriteforge
bash scripts/remote_gpu_setup.sh          # RUN_SMOKE=1 to include a first render
export ANTHROPIC_API_KEY=sk-ant-...       # optional: enables the LLM director
```

4. Run Checkpoint A/B exactly as written in `docs/CHECKPOINTS.md` (the
   `pip install` step is already done). First generation downloads ~4 GB of
   SD weights — on a datacenter pipe that's a couple of minutes.

## Getting results back

```bash
# from your Mac — copy the outputs down before killing the pod
scp -r -P <pod-port> root@<pod-ip>:/workspace/smoke ./
scp -r -P <pod-port> root@<pod-ip>:/workspace/spriteforge/game.json ./
```

Anything not copied down (or on a persistent volume) dies with the pod —
treat outputs, extracted palettes, and trained LoRAs as the deliverables to
exfiltrate; the environment itself is disposable.

## kohya on the pod (Checkpoint C)

```bash
git clone https://github.com/bmaltais/kohya_ss && cd kohya_ss && ./setup.sh
# then, with a dataset prepped by `spriteforge dataset prep`:
accelerate launch train_network.py --config_file /workspace/knight_ds/kohya_config.toml
```

No GUI needed — the config template `dataset prep` writes is CLI-ready.
Training on a 4090 is faster than the 3080 estimate (~30–45 min → often ~15–25).

## Cost hygiene

- **Stop the pod when you walk away** — the meter is per-minute, and idle
  costs the same as busy. This is the entire failure mode of GPU rental.
- A persistent volume (~$0.10/GB/mo) is worth it only if you'll return within
  days — it keeps the SD weights and venv so re-launch is instant. Otherwise
  re-download on the next pod; it's minutes.
- The three checkpoints end-to-end should total **under $10** even with
  generous fumbling time. If a bill looks bigger, a pod was left running.

## What changes vs the Windows-box plan

Nothing in spriteforge — same commands, same outputs. The Pi/WoL/worker
machinery (`docs/WINDOWS_SETUP.md`, `docs/PI_SETUP.md`) is simply not needed
for a rented box; it stays relevant for the 3080 when it's resurrected. If the
PC stays dead long-term and you want the drop-a-file workflow against a rented
GPU, that's the Phase-5 serverless shape (handover §16) — a different setup,
worth doing only after the checkpoints pass.
