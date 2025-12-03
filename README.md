Part 1 — Core Transformer Architecture

Learn the building blocks of Transformers from scratch.

1.1 Positional embeddings — absolute learned vs. sinusoidal

1.2 Self-attention from first principles — manual computation on tiny examples

1.3 Single attention head in PyTorch

1.4 Multi-head attention — splitting, concatenation, linear projections

1.5 Feed-forward networks (MLP layers) — GELU activation, dimensionality expansion

1.6 Residual connections & LayerNorm

1.7 Stacking into a full Transformer block

Part 2 — Training a Tiny LLM

Step through training a minimal language model.

2.1 Byte-level tokenization

2.2 Dataset batching & shifting for next-token prediction

2.3 Cross-entropy loss & label shifting

2.4 Training loop from scratch (without Trainer API)

2.5 Sampling strategies — temperature, top-k, top-p

2.6 Evaluating loss on validation set

Part 3 — Modernizing the Architecture

Improve training efficiency and model performance.

3.1 RMSNorm — replacing LayerNorm; comparing gradients & convergence

3.2 RoPE (Rotary Positional Embeddings) — theory & implementation

3.3 SwiGLU activations in MLP

3.4 KV cache for faster inference

3.5 Sliding-window attention & attention sink

3.6 Rolling buffer KV cache for streaming generation

Part 4 — Scaling Up

Techniques for scaling to larger models/datasets.

4.1 Switching from byte-level to BPE tokenization

4.2 Gradient accumulation & mixed precision training

4.3 Learning rate schedules & warmup

4.4 Checkpointing & resuming training

4.5 Logging & visualization — TensorBoard / wandb

Part 5 — Mixture-of-Experts (MoE)

Explore expert models for sparsity and efficiency.

5.1 MoE theory — expert routing, gating networks, load balancing

5.2 Implementing MoE layers in PyTorch

5.3 Combining MoE with dense layers for hybrid architectures

Part 6 — Supervised Fine-Tuning (SFT)

Fine-tune models on instruction datasets.

6.1 Instruction dataset formatting — prompt + response

6.2 Causal LM loss with masked labels

6.3 Curriculum learning — short → long prompts

6.4 Evaluating outputs against gold responses

Part 7 — Reward Modeling

Teach models to rank outputs.

7.1 Preference datasets — pairwise rankings

7.2 Reward model architecture — transformer encoder

7.3 Loss functions — Bradley–Terry, margin ranking loss

7.4 Sanity checks — reward shaping, debugging

Part 8 — RLHF with PPO

Optimize the policy using rewards.

8.1 Policy network — base LM + value head

8.2 Reward signal — provided by the reward model

8.3 PPO objective — maximize reward while staying close to SFT policy (KL penalty)

8.4 Training loop — sample prompts → generate completions → score → update policy

8.5 Logging & stability tricks — reward normalization, KL control, gradient clipping
