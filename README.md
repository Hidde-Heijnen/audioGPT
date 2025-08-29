# audioGPT

Word-level GPT with optional audio-shadow channel and practical training utilities. This is the repository for the MPhil thesis “Audio Interpretable Transformers,” which explores making transformers listenable by constraining representations to PCM audio. [Thesis PDF](https://www.hiddeh.com/thesis.pdf).

We mainly build on our own CamStories-10k dataset and audio vectors ([CamStories-10k on Hugging Face](https://huggingface.co/datasets/Piros/CamStories-10k)), and originally started from the excellent [nanoGPT](https://github.com/karpathy/nanoGPT) codebase. The codebase has since diverged substantially: the model and trainer are purpose-built in `model.py` and `train.py`, with features for word-level tokenisation, rotary/alt positional encodings, attention variants, and an audio-shadow pathway with auxiliary losses.

## Install

```sh
pip install torch numpy pandas pyarrow fastparquet transformers datasets tiktoken wandb tqdm
```

## What’s inside (aligned with `model.py` and `train.py`)

- **Transformer types**: `normal_transformer`, `shadow_audio`, `attention_only`, `attention_only_shadow`.
- **Attention variants**: `standard`, `projected_full`, `identity_full`.
- **Positional encodings**: `learned`, `zeropad`, `sinusoidal`, `none`, `rope` (configurable `rope_theta`).
- **Stability/mitigations**: optional sink token (`use_sink_token`), off-by-one softmax (`softmax_off_by_one`).
- **Audio-shadow channel**: parallel audio embedding stream (`w_audio`) mixed via attention; optional auxiliary losses: `expected` or `target`.
- **Other knobs**: disable last MLP (`disable_last_mlp`), weight tying, configurable dropout, etc.

See `model.py` and `train.py` for the authoritative, up-to-date implementation.

## Thesis context: Audio Interpretable Transformers

### Core idea: making transformers listenable

- Build interpretability into the representation space: make hidden states directly listenable as audio.
- Replace the token embedding matrix (`W_E`) with a fixed dictionary of word-level PCM audio waveforms; tie the unembedding (`W_U = W_E^T`) so outputs are measured in the same audible basis.
- Key finding: dot-product training alone encourages alignment at the final step but leaves large orthogonal components in the residual stream. These sound like noise yet do not affect logits, so additional constraints/diagnostics are needed beyond locked embeddings.

### CamStories dataset

- Foundation for all experiments: [CamStories-10k](https://huggingface.co/datasets/Piros/CamStories-10k). Built by merging, cleaning and normalizing TinyStories and SimpleStories into a high-quality corpus for Small Language Models (SLMs).
- Fixed 10,000-word uncased vocabulary for word-level modeling; strict normalization (grammar/formatting) and balanced gender-neutral name replacement.
- Per-token PCM audio (8 kHz, 4096 samples) for multimodal and audio-interpretable studies. Audio vectors live in `tokens_audio_10k.parquet` under `4096_vec`.

### Architectural explorations

- Positional encoding: RoPE (rotary) is preferred for listenability; it rotates queries/keys without injecting additive noise into values.
- Attention sinks: when a head should abstain, off-by-one softmax ("softmax₁") avoids mixing irrelevant audio more cleanly than a dedicated sink token; both options exist in `train.py`.
- Transparent attention: removing value/output projections keeps the value path an interpretable mixture of inputs (see `attention_type=identity_full` or `projected_full`). Removing MLP preserves listenability but harms LM performance (see `--disable_last_mlp` or `transformer_type=attention_only`).

### The Shadow Audio Transformer

- Practical diagnostic: run a parallel audio stream alongside a standard text transformer (`transformer_type=shadow_audio` or `attention_only_shadow`).
- Each layer reuses the model's attention weights (optionally with softmax₁) to mix the original token waveforms. The resulting audio makes the model's focus audible: louder components correspond to tokens that most influenced the prediction.

## Data: CamStories-10k and audio vectors

- Primary corpus: [CamStories-10k](https://huggingface.co/datasets/Piros/CamStories-10k). Stories live in `camstories_10000.parquet`; audio vectors (8 kHz, length 4096) live in `tokens_audio_10k.parquet` under column `4096_vec`.
- This repo includes prepared binaries and vocabs under `data/camstories/…` for convenience. To (re)create:

```sh
# Example: create word-level 10k data
python -c "from data.camstories.prepare import create_dataset; create_dataset('camstories_10000')"

# Example: create SimpleStories-tokenised variants
python -c "from data.camstories.prepare import create_camstories_10k_ss"
```

- If you want to use locked token embeddings or the audio-shadow channel, place the audio vectors at:

```
data/audio_embedding/tokens_audio_10k.parquet
```

The training script will auto-detect dimensions from the `4096_vec` column.

## Training

All training happens via `train.py` (read the top of the file for defaults). Common examples:

### 1) Baseline word-level LM (no audio)

```sh
python train.py \
  --dataset=camstories/10000 \
  --batch_size=32 --compile=False
```

### 2) Shadow-audio transformer

```sh
python train.py \
  --dataset=camstories/10000 \
  --transformer_type=shadow_audio \
  --shadow_audio_col=4096_vec \
  --shadow_auxiliary_loss=expected \
  --audio_alignment_lambda=1.0
```

This reads `data/audio_embedding/tokens_audio_10k.parquet`, builds the audio pathway (`w_audio`), mixes via attention, and adds the auxiliary loss.

### 3) Locked token embeddings (use audio vectors as token embeddings)

```sh
python train.py \
  --dataset=camstories/10000 \
  --locked_embeddings=4096_vec
```

`train.py` detects the 4096-dim vectors, overrides `n_embd`, loads them into `wte` and automatically freezes token embeddings (weight tying keeps `lm_head` frozen as well). Optionally combine with the shadow-audio setup.

### Optional knobs (selected)

- Positional encodings: `--posenc_type=rope --rope_theta=10000.0` (or `learned`, `sinusoidal`, `zeropad`, `none`).
- Mitigations: `--use_sink_token=True` or `--softmax_off_by_one=True`.
- Attention variants: `--attention_type=projected_full` or `identity_full`.
- Final block without MLP: `--disable_last_mlp=True`.

## Sampling

```sh
python sample.py --out_dir=out --start="Once upon a time" --num_samples=3 --max_new_tokens=100
```

## Relationship to nanoGPT

This project began as a fork/inspiration and still borrows the training loop style, CLI overrides, and overall ergonomics from [nanoGPT](https://github.com/karpathy/nanoGPT). However, the core model (`model.py`) and trainer (`train.py`) have been reworked to support word-level vocabularies, alternative positional encodings (including RoPE), attention variants, and an audio-shadow channel with auxiliary losses.

## Citation and licence

- Dataset: please cite CamStories-10k as described on its page: [CamStories-10k](https://huggingface.co/datasets/Piros/CamStories-10k).
- Code: see `LICENSE` in this repository. The dataset is released under `cdla-sharing-1.0` per its page.

## Acknowledgements

- CamStories datasets and audio vectors (by me and Paulina Körner): [CamStories-10k](https://huggingface.co/datasets/Piros/CamStories-10k)
- Training code inspiration: [nanoGPT](https://github.com/karpathy/nanoGPT)
