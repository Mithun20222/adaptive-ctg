# Adaptive Hybrid CTG: BOLT-first decoding with span-level COLD escalation

A research prototype that combines two controlled-text-generation (CTG) decoders on a frozen GPT-2 (124M):

- **BOLT** (Liu et al., 2023): fast left-to-right decoding with tunable logit biases and a straight-through estimator.
- **COLD** (Qin et al., 2022): energy-based Langevin dynamics over soft token logits, which can reason about the whole sequence (including a fixed suffix).

**Hybrid idea:** decode with BOLT by default. A lightweight router watches the remaining constraints, and when a keyword is at risk it pauses, freezes the prefix KV-cache, hands a short span to COLD, commits the result, and returns to BOLT.

> **Status: preliminary.** The main finding is negative: once BOLT's step size is tuned for the constraint density, the hybrid does not beat it. Details and caveats are below.

## Key findings

| Experiment | Result |
|---|---|
| Keyword constraints, 8 keywords, 10 prompts | Hybrid on an under-tuned BOLT adds about +1.4 keywords, but a well-tuned BOLT (lr 45) matches or beats it at lower perplexity and 3-4x the speed |
| Escalation on top of tuned BOLT | 4.70 vs 4.60 keywords (noise), with higher perplexity |
| Suffix infilling (3 cases) | No consistent gain in join quality over greedy or repetition-penalty baselines |
| COLD alone (our simplified version) | Weak: about 1.2/4 on rare keywords, about 5x slower than BOLT |

### Frontier: 8 keywords, L=25, 10 prompts

| Method | Mean keywords met (of 8) | Mean perplexity | Median perplexity |
|---|---|---|---|
| BOLT lr=35 | 3.10 | 47.9 | 54.1 |
| BOLT lr=45 | 4.60 | 88.4 | 59.6 |
| BOLT lr=55 | 4.00 | 218.4 | 149.6 |
| Hybrid (BOLT lr=30) | 4.20 | 143.6 | 128.8 |
| Hybrid (BOLT lr=45) | 4.70 | 118.4 | 72.1 |

COLD is not fully deterministic across runs: the hybrid rows vary by about +/-0.1 keywords and +/-10 perplexity.

### Hybrid vs BOLT at a fixed lr=30 (paired, 30 cases)

Hybrid wins 16, ties 14, loses 0 (mean keywords 3.20 to 3.93, mean perplexity 79 to 123). This comparison is against the under-tuned baseline above, so it should not be read as "hybrid beats BOLT".

### Suffix infilling (join NLL, lower is better)

| Method | Mean over 3 cases |
|---|---|
| BOLT (greedy, untuned) | 4.71 |
| Greedy + repetition penalty | 4.66 |
| Hybrid, lambda_right = 5 / 20 / 50 | 4.56 / 4.76 / 4.65 |

## Repository layout

| File | Role |
|---|---|
| `DualPathLM.py` | Frozen GPT-2 with a discrete path (token ids) and a soft path (`probs @ embedding matrix`), both with KV-cache |
| `EnergyFunctions.py` | Fluency energy (soft NLL) and keyword energy (soft-min over start offsets, clamped at 0) |
| `BoltEngine.py` | Biased decoding with STE; `tune()` optimises the bias, `generate()` decodes |
| `ColdSpanEngine.py` | Langevin dynamics over a k-token soft span with a frozen prefix cache, top-k guided discretisation, optional right-context (suffix) energy |
| `AdaptiveRouter.py` | Horizon-deficit trigger `(L - t) / |remaining keywords| <= tau`, plus a suffix-bridge trigger |
| `HybridEngine.py` | BOLT/COLD handoff (`generate`, `generate_infill`) |
| `metrics.py` | Contiguous keyword check, perplexity, suffix join NLL |
| `main.py` | Three-way comparison (BOLT, COLD, hybrid) |
| `test_*.py` | Unit checks and experiments (see below) |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3 with `torch` and `transformers`. The first run downloads GPT-2 (about 500 MB). Experiments in this repo were run on [fill in hardware].

## Running

```powershell
py main.py                    # BOLT vs COLD vs hybrid, 7 cases, about 10 minutes
py test_dualpath.py           # soft path equals discrete path on one-hot input
py test_energy.py             # energies reduce to NLL on hard tokens
py test_router.py             # router logic
py test_paired.py             # hybrid vs BOLT, 30 paired cases (about 25 minutes)
py test_frontier.py           # keywords vs perplexity frontier (about 20 minutes)
py test_infill_sweep.py       # suffix infilling sweep
```

## How it works

- **Soft tokens.** A soft token is a probability vector `p`; its embedding is `p @ E`. With one-hot `p` this equals the normal lookup, which `test_dualpath.py` verifies.
- **Frozen prefix.** The prefix cache is never modified: each Langevin iteration works on a fresh copy, gradients are taken with `torch.autograd.grad` (so the graph is freed each step), and noise updates happen under `no_grad`.
- **Discretisation.** At each position the candidates are the LM's top-50 tokens plus the optimiser's top-3, and the token with the highest optimised logit is committed. The optimiser's own favourites are included because otherwise a strongly preferred keyword can be filtered out.
- **Router.** Escalates when the remaining tokens per missing keyword fall to `tau` (default 2) and sends one keyword per COLD span, with span length `len(keyword) + 3` capped by the remaining room.

## Differences from the original implementations

Checking the authors' released code showed this prototype is a simplified version of both methods, not a reproduction.

| Aspect | Original repos | This repo |
|---|---|---|
| BOLT bias weight | increases 0 to 1 over the sequence | decreases, `1 - t/L` |
| BOLT bias / optimiser | hidden-state bias, AdamW lr 0.4, up to 100 iterations | vocab-space bias, normalised-gradient step, 5 iterations |
| COLD update | Adam on a perturbation, staged noise, 2000 iterations, batch 32 | plain Langevin step, annealed noise, 150 iterations, batch 1 |
| COLD fluency | left-to-right + right-to-left LM | left-to-right only |
| Model | GPT-2 large / XL | GPT-2 small |

## Limitations

- Small evaluation: 10 short prompts and one ocean-themed keyword pool; cases from the same prompt are correlated.
- Perplexity is measured with the same GPT-2 that generates, and generated text is often repetitive even when constraints are met.
- The hybrid configuration (tau = 2, one keyword per span) was chosen on data that is also part of the evaluation.
- Our COLD is lightly tuned and simplified, so results say little about the original method.
- The infilling baseline is untuned BOLT (greedy decoding).

## Possible next steps

A closer BOLT reproduction (Adam, increasing weight), a stronger COLD (more iterations, right-to-left model), more prompts and independent keyword sets, and letting several keywords share one COLD span with a mutual-exclusion term.

## References

- Liu, X., Khalifa, M., Wang, L. (2023). *BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases.* ACL (short papers).
- Qin, L., Welleck, S., Khashabi, D., Choi, Y. (2022). *COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics.* NeurIPS.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
