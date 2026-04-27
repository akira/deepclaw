---
title: Open-Weight Reasoning Models vs Sonnet and Opus — Best Bang for the Buck
created: 2026-04-26
updated: 2026-04-26
type: comparison
tags: [inference, comparison, pricing, provider, coding, reasoning, open-weight, anthropic]
sources:
  - raw/articles/deepinfra-pricing-snapshot-2026-04-26.md
  - raw/articles/open-weight-benchmarks-notes-2026-04-26.md
  - raw/articles/anthropic-pricing-snapshot-2026-04-26.md
  - concepts/closed-model-api-pricing
---

# Open-Weight Reasoning Models vs Sonnet and Opus — Best Bang for the Buck

## What is being compared
A practical cost-vs-quality comparison between Anthropic's premium closed models (`claude-sonnet-4.6` and `claude-opus-4.7`) and leading hosted open-weight reasoning models, with emphasis on which models challenge Sonnet /Opus capability at a fraction of the cost.

Related pages:
- [[closed-model-api-pricing]]
- [[mini-vs-sonnet-and-other-closed-model-value-comparison]]
- [[open-weight-models-vs-gpt-5-mini-pricing-and-performance]]
- [[inference-provider-model-comparison]]

## Baselines
From the Anthropic pricing snapshot:
- claude-sonnet-4.6 = $3.00 input / $15.00 output per 1M tokens
- claude-opus-4.7 = $5.00 input / $25.00 output per 1M tokens

These sit at the high end of general-purpose inference pricing. The comparison question is: what open-weight reasoning model delivers competitive quality at a fraction of this price?

## Hosted reasoning models — price overview

### Dedicated reasoning / thinking models (on DeepInfra)
| Model | Input | Cached | Output | Context |
|---|---:|---:|---:|---:|
| Qwen3-235B-A22B-Thinking-2507 | $0.23 | $0.20 | $2.30 | 256k |
| DeepSeek-R1-0528 | $0.50 | $0.35 | $2.15 | 160k |
| DeepSeek-R1-Distill-Llama-70B | $0.70 | — | $0.80 | 128k |
| Qwen3-Max-Thinking | $1.20 | $0.24 | $6.00 | 250k |

### Models with high-reasoning modes (via Together / hosted)
| Model | Input | Cached | Output | SWE-bench verified |
|---|---:|---:|---:|---:|
| MiniMax M2.5 | $0.30 | $0.06 | $1.20 | 75.80% |
| GLM-5 | $1.00 | — | $3.20 | 72.80% |
| Kimi K2.5 | $0.50 | $0.10 | $2.80 | 70.80% |

### For reference: strong open-weight non-reasoning models
| Model | Input | Output | SWE-bench verified |
|---|---:|---:|---:|
| DeepSeek-V3.2 | $0.26 | $0.38 | 70.00% |
| Qwen3-Coder-480B-A35B-Turbo | $0.30 | $1.00 | — |

## Direct price comparisons vs claude-sonnet-4.6

### DeepSeek-R1-0528
- Input: 83% cheaper than Sonnet ($0.50 vs $3.00)
- Output: 86% cheaper ($2.15 vs $15.00)
- Interpretation: A dedicated open-weight reasoning model at less than one-fifth of Sonnet's price on both axes. The strongest direct Sonnet challenger for reasoning-heavy workloads.

### Qwen3-235B-A22B-Thinking-2507
- Input: 92% cheaper ($0.23 vs $3.00)
- Output: 85% cheaper ($2.30 vs $15.00)
- Interpretation: Thinking-mode Qwen at a dramatically lower price. Strongest pure token-economics case among dedicated reasoning models.

### MiniMax M2.5 (high reasoning mode)
- Input: 90% cheaper ($0.30 vs $3.00)
- Output: 92% cheaper ($1.20 vs $15.00)
- Interpretation: Highest SWE-bench score in this set (75.80%) at one-tenth the output cost. Strongest benchmark-per-dollar candidate in this