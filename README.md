# Byte-Level Cross-Tokenizer Knowledge Distillation

This repository contains the implementation for the master's thesis *"Exploring Approaches to Distillation for Models Without Shared Vocabularies"* (HSE University, 2026). The project extends the [GOLD (General On-Policy Logit Distillation)](https://arxiv.org/abs/2505.XXXXX) framework with a principled byte-level alignment mechanism, enabling statistically consistent knowledge distillation between language models that use different tokenizers and vocabularies.

## Motivation

Standard knowledge distillation methods require teacher and student models to share a compatible tokenizer, which severely limits which model pairs can be used. Existing cross-tokenizer approaches like ULD and GOLD attempt to bridge this gap, but rely on approximations when aligning multi-token groups: the probability merging formula used in GOLD is formally incorrect for all but the predicted token, and — more critically — compares teacher and student distributions that are conditioned on structurally different contexts. This work identifies these flaws and shows they cannot be patched at the token level, motivating a reduction to the byte level where a shared, universal representation is always available.

## Method

The core contribution is a hybrid training objective that switches between token-level and byte-level distillation depending on the alignment group. For singleton groups (one token on each side), the standard GKD or ULD loss is applied as in the original GOLD pipeline. For multi-token groups — where teacher and student tokenize the same substring differently — both models are converted to byte-level distributions via [ByteSampler](https://arxiv.org/abs/2506.14123), which uses a Valid Covering Tree (VCT) to compute exact byte-level probabilities from any HuggingFace ByteLevel BPE model without modifying weights. At each byte position within a group, both distributions are conditioned on exactly the same byte prefix, making the Jensen-Shannon divergence loss well-defined and free of context mismatch. The implementation also replaces GOLD's O(N²) sequence alignment with a linear-time procedure and introduces an incremental sampler state strategy that keeps the total inference cost at O(|P| + |C|) per training example rather than O(G · (|P| + |C|)).

## Experiments

Experiments were conducted on the [HellaSwag](https://rowanzellers.com/hellaswag/) sentence-completion benchmark using EleutherAI/pythia-70m as the student and HuggingFaceTB/SmolLM2-360M-Instruct as the teacher. The two models share only 57.4% of their vocabulary by Jaccard similarity, creating a realistic cross-tokenizer scenario. Training on 1,000 examples for 5 epochs with the ByteSampler-augmented GOLD objective improved the student's scoring accuracy from 30.0% to 34.0% (+4 percentage points), demonstrating the viability of byte-level alignment as a distillation signal even under tight computational constraints.

## Usage

```bash
git clone https://github.com/eanor/byte-level-distillation
cd byte-level-distillation
pip install -r requirements.txt
```

Training with the byte-level loss:

```bash
python train.py \
  --student EleutherAI/pythia-70m \
  --teacher HuggingFaceTB/SmolLM2-360M-Instruct \
  --dataset hellaswag \
  --train_size 1000 \
  --epochs 5 \
  --lr 5e-5 \
  --loss byte_gold
```

> **Note:** ByteSampler currently supports HuggingFace ByteLevel BPE tokenizers (e.g., GPT-NeoX/Pythia, SmolLM 1/2/3). SentencePiece BPE models such as Llama 1/2 are not supported. The byte-level loss adds significant computational overhead (~4× per step compared to the token-level baseline); full-scale experiments require batched byte-level inference or selective application to high-divergence groups only.

## Citation

```bibtex
@mastersthesis{kuznetsova2026byte,
  author  = {Kuznetsova, Svetlana},
  title   = {Exploring Approaches to Distillation for Models Without Shared Vocabularies},
  year    = {2026}
}
```
