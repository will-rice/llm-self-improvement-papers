---
identifier: arxiv:2603.05433v7
title: "CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"
authors:
  - Hejian Sang
  - Yuanda Xu
  - Zhengze Zhou
  - Ran He
  - Zhipeng Wang
  - Jiachen Sun
published: "2026-03-05T17:54:40+00:00"
url: https://arxiv.org/abs/2603.05433v7
source: arxiv
doi: null
arxiv_id: 2603.05433v7
categories:
  - cs.LG
---

# CRISP: Compressed Reasoning via Iterative Self-Policy Distillation

Hejian Sang ^(†)^(†)thanks: Equal contribution.^(†)^(†)thanks:
Correspondence to hejian@alumni.iastate.edu
Email: [hejian@alumni.iastate.edu](mailto:)    Yuanda Xu¹¹footnotemark:
1 Email: [yuanda@math.princeton.edu](mailto:)    Zhengze
Zhou¹¹footnotemark: 1 Email: [zz433@cornell.edu](mailto:)    Ran
He¹¹footnotemark: 1 Email: [rh2528@columbia.edu](mailto:)    Zhipeng
Wang Email: [zhipeng.wang@alumni.rice.edu](mailto:)    Jiachen Sun
Email: [jiachens@umich.edu](mailto:)

###### Abstract

Reasoning models often generate far more tokens than a task requires,
which raises inference cost and can compound errors. We introduce CRISP
(Compressed Reasoning via Iterative Self-Policy Distillation), an
on-policy self-distillation method that teaches a model to reason more
concisely by distilling its own concise behavior back into itself. The
method uses a single idea: condition the same model on a “be concise”
instruction to obtain teacher logits, then minimize the per-token
reverse KL divergence between the student and this teacher on the
student’s own rollouts. It requires no ground-truth answers, no token
budgets, and no difficulty estimators. The reverse-KL objective is
naturally difficulty-adaptive: it compresses easy problems aggressively
while preserving the reasoning steps that hard problems require. On
Qwen3-14B, CRISP cuts reasoning length by up to 56% on MATH-500 and 38%
on the harder AIME 2024, while improving MATH-500 accuracy by up to 3.3
points over the base model and holding AIME 2024 accuracy within about
one point. This behavior generalizes across model sizes and families:
Qwen3-8B shows the same compression with accuracy preserved, and
DeepSeek-R1-Distill-Llama-8B improves accuracy on all five benchmarks
while shortening its responses. General capabilities are preserved
across all three models. Code is available at
[https://github.com/HJSang/OPSD_Reasoning_Compression](https://github.com/HJSang/OPSD_Reasoning_Compression).

Figure 1: Reasoning compression with preserved accuracy on Qwen3-14B.
Results across three math benchmarks of increasing difficulty under a
30K-token budget (mean@8). a, Average reasoning length: CRISP reduces
length by 56% on MATH-500, 38% on AIME 2024, and 32% on AIME 2025, with
the largest reduction on the easiest benchmark. b, Accuracy: CRISP
improves MATH-500 accuracy (93.0 to 96.3) and holds AIME 2024 within
about one point, with a modest drop on the hardest AIME 2025 under this
aggressive (uniform-instruction) setting.

## 1 Introduction

Large reasoning models generate long chains of intermediate reasoning
before producing an answer. Systems such as OpenAI o1 ([Jaech et al.,
2024](#bib.bib18)), Gemini 2.5 ([Comanici et al., 2025](#bib.bib8)),
DeepSeek-R1 ([Guo et al., 2025](#bib.bib13)), and Qwen3 ([Yang et al.,
2025](#bib.bib35)) produce thousands of tokens of deliberation,
exploring alternatives, checking intermediate steps, and verifying
conclusions. This deliberation helps on hard problems, but the models
also produce long reasoning traces on easy inputs where a short answer
would suffice ([Snell et al., 2024](#bib.bib28); [Muennighoff et al.,
2025](#bib.bib24)). The excess tokens raise inference cost and latency,
and on some problems they degrade accuracy by introducing errors in
otherwise correct solutions.

Several families of methods address this problem
(Appendix [B](#A2 "Appendix B Survey of Reasoning Compression Methods ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
surveys recent work), but each has a limitation. Reinforcement learning
methods add a length penalty to the reward, which requires ground-truth
answers and can reduce the model’s ability to explore ([Aggarwal and
Welleck, 2025](#bib.bib1); [Wan et al., 2026](#bib.bib29); [Liu et al.,
2025](#bib.bib23)). Supervised fine-tuning methods train on externally
curated short traces, which causes distribution shift and
forgetting ([Huang et al., 2025](#bib.bib16); [Shenfeld et al.,
2026](#bib.bib26)). Most methods apply a uniform compression target
rather than adapting to problem difficulty. Prompting methods change
behavior only while the prompt is present and do not persist in the
weights.

We propose CRISP (Compressed Reasoning via Iterative Self-Policy
Distillation), which avoids these limitations with a single mechanism:
instruct the model to be concise, then distill that behavior back into
the model so it persists without the instruction. Given a reasoning
model $`\pi_{\theta}`$, we define:

- •
  Teacher: $`\pi_{\theta}(\cdot\mid x,c)`$, the same model conditioned
  on a conciseness instruction $`c`$ (for example: “Solve concisely,
  avoid unnecessary steps”).
- •
  Student: $`\pi_{\theta}(\cdot\mid x)`$, the same model without the
  compression instruction.

Training generates student rollouts and minimizes the per-token reverse
KL divergence between student and teacher distributions. This on-policy
self-distillation approach requires no ground-truth answers, no reward
engineering, and no difficulty estimation. The compression signal
emerges naturally from the KL objective, adapting automatically to
problem difficulty.

|                                                                                               |            |              |                      |                     |
| --------------------------------------------------------------------------------------------- | ---------- | ------------ | -------------------- | ------------------- |
| Method                                                                                        | On- policy | No GT needed | Difficulty- adaptive | Entropy- preserving |
| RL + length penalty ([Aggarwal and Welleck, 2025](#bib.bib1); [Wan et al., 2026](#bib.bib29)) | ✓          | ✗            | ✗                    | ✗                   |
| SFT on compressed CoT ([Huang et al., 2025](#bib.bib16))                                      | ✗          | ✗            | ✗                    | ✓                   |
| OPCD ([Ye et al., 2026](#bib.bib36))                                                          | ✓          | ✗            | ✗                    | ✓                   |
| DLER ([Liu et al., 2025](#bib.bib23))                                                         | ✓          | ✗            | ✗                    | ✗                   |
| Prompting / pruning ([Xu et al., 2025](#bib.bib34))                                           | —          | ✓            | ✗                    | ✓                   |
| CRISP (ours)                                                                                  | ✓          | ✓            | ✓                    | ✓                   |

Table 1: Comparison of reasoning compression methods. CRISP uniquely
combines on-policy training, no dependence on ground-truth (GT) answers,
difficulty-adaptive compression, and entropy preservation
(Appendix [H](#A8 "Appendix H Entropy Preservation During Training ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
contrasts CRISP with representative methods from each paradigm. CRISP is
the only approach that satisfies all four desiderata.

##### Summary of results.

On Qwen3-8B and Qwen3-14B, CRISP reduces MATH-500 reasoning length by
32% to 57% while preserving accuracy, and it improves Qwen3-14B MATH-500
accuracy by up to 3.3 points
(Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
Compression is difficulty-adaptive: reductions are largest on the easier
MATH-500 and smaller on the harder AIME benchmarks, where accuracy stays
within about one point of the base model. The method readily adopts
different concise behaviors: different conciseness instructions all
yield strong compression with preserved accuracy, showing that the
effect does not depend on a single hand-tuned prompt. The method
transfers across model families: on DeepSeek-R1-Distill-Llama-8B it
improves accuracy on all five benchmarks while shortening responses.
Although CRISP trains only on math, out-of-domain accuracy on
GPQA-Diamond and MMLU is preserved, so compressing math reasoning does
not degrade general capabilities.

## 2 Related Work

##### Reasoning compression via reinforcement learning.

The most direct approach: penalize length in the reward function.
L1 ([Aggarwal and Welleck, 2025](#bib.bib1)) caps token count during
GRPO training. DiPO ([Wan et al., 2026](#bib.bib29)) and DIET ([Chen
et al., 2025](#bib.bib5)) estimate difficulty from rollout pass rates
and set per-problem length targets. Leash ([Li et al.,
2025b](#bib.bib20)) shapes rewards with sigmoid functions; DLER ([Liu
et al., 2025](#bib.bib23)) adds curriculum learning. ThinkPrune ([Hou
et al., 2025](#bib.bib4)) continuously trains long-thinking LLMs using
reinforcement learning (RL) with an additional token-budget constraint
while preserving answer correctness. The catch: all of these require
ground-truth answers. No correct answer, no reward and no way to know if
compression went too far.  [Xu et al. (2026)](#bib.bib2) further notes
that reinforcement learning can induce overconfidence errors, thereby
narrowing the model’s reasoning boundary and reducing generation
diversity.

##### Reasoning compression via supervised fine-tuning.

Another route: curate short reasoning traces, then train on them.
SEER ([Huang et al., 2025](#bib.bib16)) samples many solutions and keeps
the shortest correct ones. TokenSkip ([Xia et al., 2025](#bib.bib33))
learns which tokens to skip. DAP/LiteCoT ([Wu et al., 2025](#bib.bib32))
distills from stronger models; S3-CoT ([Du et al., 2026](#bib.bib10))
steers activations toward brevity. The problem is distribution shift:
the student trains on someone else’s reasoning and forgets its
own ([Shenfeld et al., 2026](#bib.bib26)).

##### Training-free compression.

The lightweight option: change the prompt or the decoder, not the
weights. Chain of Draft ([Xu et al., 2025](#bib.bib34)) asks for minimal
drafts instead of full reasoning. TrimR ([Lin et al., 2025](#bib.bib22))
prunes after the fact. NoWait ([Wang et al., 2025a](#bib.bib30)) and
FlowSteer ([Li et al., 2026](#bib.bib21)) steer decoding toward
conciseness. These methods are easy to deploy but achieve limited
compression, and the effect vanishes when you change the prompt.

##### On-policy self-distillation.

The closest relatives of our work use the model as its own teacher.
OPSD ([Zhao et al., 2026](#bib.bib38)) gives the teacher the
ground-truth answer, achieving 4–8$`\times`$ efficiency over GRPO.
SDPO ([Hübotter et al., 2026](#bib.bib17)) conditions on rich feedback
for dense credit assignment. SDFT ([Shenfeld et al., 2026](#bib.bib26))
shows that on-policy distillation dramatically reduces forgetting
compared to standard SFT, interpreting it as inverse RL. OPCD ([Ye
et al., 2026](#bib.bib36)) distills system-prompt behaviors into
weights. We contribute a new application: using a _conciseness
instruction_ as the privileged context, achieving compression without
any ground-truth supervision.

## 3 Method

### 3.1 Problem Formulation

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjIucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmcgbHR4X2ZpZ3VyZV9wYW5lbCIgaGVpZ2h0PSIxNDIuOTUiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NTMuNTEgMTQyLjk1IiB3aWR0aD0iNDUzLjUxIj48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDE0Mi45NSkgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM5OTREMDA7IiBmaWxsPSIjOTk0RDAwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNS42IEwgMCAxMzcuMzUgQyAwIDE0MC40NCAyLjUxIDE0Mi45NSA1LjYgMTQyLjk1IEwgNDQ3LjkxIDE0Mi45NSBDIDQ1MSAxNDIuOTUgNDUzLjUxIDE0MC40NCA0NTMuNTEgMTM3LjM1IEwgNDUzLjUxIDUuNiBDIDQ1My41MSAyLjUxIDQ1MSAwIDQ0Ny45MSAwIEwgNS42IDAgQyAyLjUxIDAgMCAyLjUxIDAgNS42IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGRkYzRTA7IiBmaWxsPSIjRkZGM0UwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuNjYgNS42IEwgMS42NiAxMDIuMDggTCA0NTEuODUgMTAyLjA4IEwgNDUxLjg1IDUuNiBDIDQ1MS44NSAzLjQyIDQ1MC4wOCAxLjY2IDQ0Ny45MSAxLjY2IEwgNS42IDEuNjYgQyAzLjQyIDEuNjYgMS42NiAzLjQyIDEuNjYgNS42IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGRkYzRTA7IiBmaWxsPSIjRkZGM0UwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuNjYgMTAzLjc0IEwgMS42NiAxMzcuMzUgQyAxLjY2IDEzOS41MiAzLjQyIDE0MS4yOCA1LjYgMTQxLjI4IEwgNDQ3LjkxIDE0MS4yOCBDIDQ1MC4wOCAxNDEuMjggNDUxLjg1IDEzOS41MiA0NTEuODUgMTM3LjM1IEwgNDUxLjg1IDEwMy43NCBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDIxLjM1IDExMS4xMykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDoyOS42OWVtOy0tbHR4LWZvLWhlaWdodDoxLjg5ZW07LS1sdHgtZm8tZGVwdGg6MC4yNWVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIyOS42NyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMjYuMjEpIiB3aWR0aD0iNDEwLjgzIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJTMy5GMi5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MjkuNjllbTsiPgo8c3BhbiBpZD0iUzMuRjIucGljMS4xLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlMzLkYyLnBpYzEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5NEQwMDsiPlN0dWRlbnQgUHJvbXB04oCD4oCKPG1hdGggaWQ9IlMzLkYyLnBpYzEubTEiIGNsYXNzPSJsdHhfbWF0aF91bnBhcnNlZCIgYWx0dGV4dD0iXHBpX3tcdGhldGF9KFxjZG90XG1pZCB4KSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5NEQwMDsiIG1hdGhjb2xvcj0iIzk5NEQwMCI+z4A8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5NEQwMDsiIG1hdGhjb2xvcj0iIzk5NEQwMCI+zrg8L21pPjwvbXN1Yj48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM5OTREMDA7IiBtYXRoY29sb3I9IiM5OTREMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KDwvbW8+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk0RDAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzk5NEQwMCIgcnNwYWNlPSIwZW0iPuKLhTwvbW8+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk0RDAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzk5NEQwMCIgcnNwYWNlPSIwLjE2N2VtIj7iiKM8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5NEQwMDsiIG1hdGhjb2xvcj0iIzk5NEQwMCI+eDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk0RDAwOyIgbWF0aGNvbG9yPSIjOTk0RDAwIiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5ccGlfe1x0aGV0YX0oXGNkb3RcbWlkIHgpPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuMzUgMTYuMjQpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzEuNDJlbTstLWx0eC1mby1oZWlnaHQ6NS4zNWVtOy0tbHR4LWZvLWRlcHRoOjAuMmVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI3Ni44IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3NC4wMykiIHdpZHRoPSI0MzQuNzYiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IlMzLkYyLnBpYzEuMiIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozMS40MmVtOyI+CjxzcGFuIGlkPSJTMy5GMi5waWMxLjIuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzMuRjIucGljMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Tb2x2ZSB0aGUgZm9sbG93aW5nIG1hdGggcHJvYmxlbSBzdGVwIGJ5IHN0ZXAuIFRoZSBsYXN0IGxpbmUgb2YgeW91ciByZXNwb25zZSBzaG91bGQgYmUgb2YgdGhlIGZvcm0gQW5zd2VyOiAkQW5zd2VyICh3aXRob3V0IHF1b3Rlcykgd2hlcmUgJEFuc3dlciBpcyB0aGUgYW5zd2VyIHRvIHRoZSBwcm9ibGVtLgo8YnIgY2xhc3M9Imx0eF9icmVhayIgc3R5bGU9Ii0tbHR4LWJyZWFrLXNwYWNlOjYuMHB0OyI+CkZpbmQgYWxsIHJlYWwgbnVtYmVycyA8bWF0aCBpZD0iUzMuRjIucGljMS5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJ4IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPng8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBzdWNoIHRoYXQgPG1hdGggaWQ9IlMzLkYyLnBpYzEubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ieF57M30tNnheezJ9KzExeC02PTAiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bXJvdz48bXJvdz48bXN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MzwvbW4+PC9tc3VwPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4oiSPC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjY8L21uPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1zdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjwvbXN1cD48L21yb3c+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xMTwvbW4+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjwvbXJvdz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7iiJI8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4wPC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij54XnszfS02eF57Mn0rMTF4LTY9MDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+Lgo8YnIgY2xhc3M9Imx0eF9icmVhayIgc3R5bGU9Ii0tbHR4LWJyZWFrLXNwYWNlOjQuMHB0OyI+ClJlbWVtYmVyIHRvIHB1dCB5b3VyIGFuc3dlciBvbiBpdHMgb3duIGxpbmUgYWZ0ZXIg4oCY4oCYQW5zd2VyOuKAmeKAmS4KPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjIucGljMiIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmcgbHR4X2ZpZ3VyZV9wYW5lbCIgaGVpZ2h0PSIyMjcuNyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ1My41MSAyMjcuNyIgd2lkdGg9IjQ1My41MSI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwyMjcuNykgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiMwMDk5MDA7IiBmaWxsPSIjMDA5OTAwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNS42IEwgMCAyMjIuMSBDIDAgMjI1LjE5IDIuNTEgMjI3LjcgNS42IDIyNy43IEwgNDQ3LjkxIDIyNy43IEMgNDUxIDIyNy43IDQ1My41MSAyMjUuMTkgNDUzLjUxIDIyMi4xIEwgNDUzLjUxIDUuNiBDIDQ1My41MSAyLjUxIDQ1MSAwIDQ0Ny45MSAwIEwgNS42IDAgQyAyLjUxIDAgMCAyLjUxIDAgNS42IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNFOEY1RTk7IiBmaWxsPSIjRThGNUU5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuNjYgNS42IEwgMS42NiAxODYuODMgTCA0NTEuODUgMTg2LjgzIEwgNDUxLjg1IDUuNiBDIDQ1MS44NSAzLjQyIDQ1MC4wOCAxLjY2IDQ0Ny45MSAxLjY2IEwgNS42IDEuNjYgQyAzLjQyIDEuNjYgMS42NiAzLjQyIDEuNjYgNS42IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNFOEY1RTk7IiBmaWxsPSIjRThGNUU5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuNjYgMTg4LjQ5IEwgMS42NiAyMjIuMSBDIDEuNjYgMjI0LjI3IDMuNDIgMjI2LjA0IDUuNiAyMjYuMDQgTCA0NDcuOTEgMjI2LjA0IEMgNDUwLjA4IDIyNi4wNCA0NTEuODUgMjI0LjI3IDQ1MS44NSAyMjIuMSBMIDQ1MS44NSAxODguNDkgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAyMS4zNSAxOTUuODkpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MjkuNjllbTstLWx0eC1mby1oZWlnaHQ6MS44OWVtOy0tbHR4LWZvLWRlcHRoOjAuMjVlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMjkuNjciIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDI2LjIxKSIgd2lkdGg9IjQxMC44MyI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzMuRjIucGljMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjI5LjY5ZW07Ij4KPHNwYW4gaWQ9IlMzLkYyLnBpYzIuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTMy5GMi5waWMyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDk5MDA7Ij5UZWFjaGVyIFByb21wdOKAg+KAijxtYXRoIGlkPSJTMy5GMi5waWMyLm0xIiBjbGFzcz0ibHR4X21hdGhfdW5wYXJzZWQiIGFsdHRleHQ9IlxwaV97XGJhcntcdGhldGF9fShcY2RvdFxtaWQgeHssfVw7YykiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDk5MDA7IiBtYXRoY29sb3I9IiMwMDk5MDAiPs+APC9taT48bW92ZXIgYWNjZW50PSJ0cnVlIj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDk5MDA7IiBtYXRoY29sb3I9IiMwMDk5MDAiPs64PC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPsKvPC9tbz48L21vdmVyPjwvbXN1Yj48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDk5MDA7IiBtYXRoY29sb3I9IiMwMDk5MDAiIHN0cmV0Y2h5PSJmYWxzZSI+KDwvbW8+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA5OTAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzAwOTkwMCIgcnNwYWNlPSIwZW0iPuKLhTwvbW8+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA5OTAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzAwOTkwMCIgcnNwYWNlPSIwLjE2N2VtIj7iiKM8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwOTkwMDsiIG1hdGhjb2xvcj0iIzAwOTkwMCI+eDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA5OTAwOyIgbWF0aGNvbG9yPSIjMDA5OTAwIiByc3BhY2U9IjAuNDQ3ZW0iPiw8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwOTkwMDsiIG1hdGhjb2xvcj0iIzAwOTkwMCI+YzwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA5OTAwOyIgbWF0aGNvbG9yPSIjMDA5OTAwIiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5ccGlfe1xiYXJ7XHRoZXRhfX0oXGNkb3RcbWlkIHh7LH1cO2MpPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuMzUgMTYuMjQpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzEuNDJlbTstLWx0eC1mby1oZWlnaHQ6MTEuNDhlbTstLWx0eC1mby1kZXB0aDowLjJlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTYxLjU1IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNTguNzgpIiB3aWR0aD0iNDM0Ljc2Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJTMy5GMi5waWMyLjIiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzEuNDJlbTsiPgo8c3BhbiBpZD0iUzMuRjIucGljMi4yLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlMzLkYyLnBpYzIuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIGx0eF9mb250X2JvbGQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDY2MDA7Ij5Db25jaXNlbmVzcyBpbnN0cnVjdGlvbiA8bWF0aCBpZD0iUzMuRjIucGljMi5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJjIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA2NjAwOyIgbWF0aGNvbG9yPSIjMDA2NjAwIj5jPC9taT48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPmM8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiAodW5pZm9ybSBjb21wcmVzc2lvbik6PHNwYW4gaWQ9IlMzLkYyLnBpYzIuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gU29sdmUgdGhlIGZvbGxvd2luZyBtYXRoIHByb2JsZW0gY29uY2lzZWx5IGFuZCBjb3JyZWN0bHkuIEJlIGRpcmVjdDogYXZvaWQgdW5uZWNlc3NhcnkgZWxhYm9yYXRpb24sIHJlZHVuZGFudCBzdGVwcywgb3IgcmVzdGF0aW5nIHRoZSBwcm9ibGVtLiBGb2N1cyBvbmx5IG9uIHRoZSBrZXkgcmVhc29uaW5nIHN0ZXBzIG5lZWRlZCB0byByZWFjaCB0aGUgYW5zd2VyLgo8YnIgY2xhc3M9Imx0eF9icmVhayIgc3R5bGU9Ii0tbHR4LWJyZWFrLXNwYWNlOjQuMHB0OyI+ClRoZSBsYXN0IGxpbmUgb2YgeW91ciByZXNwb25zZSBzaG91bGQgYmUgb2YgdGhlIGZvcm0gQW5zd2VyOiAkQW5zd2VyICh3aXRob3V0IHF1b3Rlcykgd2hlcmUgJEFuc3dlciBpcyB0aGUgYW5zd2VyIHRvIHRoZSBwcm9ibGVtLgo8YnIgY2xhc3M9Imx0eF9icmVhayIgc3R5bGU9Ii0tbHR4LWJyZWFrLXNwYWNlOjYuMHB0OyI+CkZpbmQgYWxsIHJlYWwgbnVtYmVycyA8bWF0aCBpZD0iUzMuRjIucGljMi5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJ4IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPng8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBzdWNoIHRoYXQgPG1hdGggaWQ9IlMzLkYyLnBpYzIubTQiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ieF57M30tNnheezJ9KzExeC02PTAiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bXJvdz48bXJvdz48bXN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MzwvbW4+PC9tc3VwPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4oiSPC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjY8L21uPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1zdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjwvbXN1cD48L21yb3c+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xMTwvbW4+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjwvbXJvdz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7iiJI8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4wPC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij54XnszfS02eF57Mn0rMTF4LTY9MDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+Lgo8YnIgY2xhc3M9Imx0eF9icmVhayIgc3R5bGU9Ii0tbHR4LWJyZWFrLXNwYWNlOjQuMHB0OyI+ClJlbWVtYmVyIHRvIHB1dCB5b3VyIGFuc3dlciBvbiBpdHMgb3duIGxpbmUgYWZ0ZXIg4oCY4oCYQW5zd2VyOuKAmeKAmS4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

Figure 2: Prompt example for student and teacher policies. Both policies
share the same model parameters but differ in conditioning context. The
teacher receives only a _conciseness instruction_ $`c`$ prepended to the
problem, here the uniform-compression instruction; no ground-truth
answers or reference solutions are provided. This is the key distinction
from prior self-distillation work ([Shenfeld et al., 2026](#bib.bib26)),
where the teacher receives the ground-truth solution as privileged
information. The student prompt is the original prompt from the DAPO-17K
dataset.

Consider a reasoning model $`\pi_{\theta}`$ that, given input $`x`$,
generates a reasoning trace $`r`$ followed by an answer $`a`$, producing
output $`y=(r,a)`$. The reasoning trace typically appears within
\<think\>$`\ldots`$\</think\> delimiters. We aim to learn parameters
$`\theta^{*}`$ such that the model produces shorter reasoning traces
while maintaining accuracy.

Let $`c`$ denote a conciseness instruction (see
Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Problem Formulation ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
for a concrete example). Modern reasoning models can follow such
instructions via in-context learning, producing shorter reasoning traces
when $`c`$ is prepended to the input. We denote the
conciseness-conditioned model as $`\pi_{\theta}(\cdot\mid x,c)`$
(teacher) and the unconditional model as $`\pi_{\theta}(\cdot\mid x)`$
(student). The teacher and student share parameters $`\theta`$ but
receive different inputs.

### 3.2 Training Objective

CRISP minimizes the per-token reverse KL divergence between the student
and a stop-gradient teacher on student-generated rollouts:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
\mathcal{L}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\;y\sim\pi_{\theta}(\cdot\mid x)}\left[\sum_{t=1}^{|y|}D_{\mathrm{KL}}\Big(\pi_{\theta}(\cdot\mid x,y_{<t})\;\Big\|\;\pi_{\bar{\theta}}(\cdot\mid x,c,y_{<t})\Big)\right],
``` |  | (1) |

where $`\bar{\theta}`$ denotes the teacher weights, which are
periodically synchronized with the student
(Section [3.3](#S3.SS3 "3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
and no gradients flow through the teacher’s forward pass. The
expectation over $`y\sim\pi_{\theta}(\cdot\mid x)`$ makes training
*on-policy*: the student is optimized on its own generation
distribution, which prevents the distribution shift inherent in
off-policy SFT.

##### Why reverse KL?

We use reverse KL,
$`D_{\mathrm{KL}}(\pi_{\theta}\|\pi_{\bar{\theta}})`$, rather than the
forward direction $`D_{\mathrm{KL}}(\pi_{\bar{\theta}}\|\pi_{\theta})`$,
and the choice is important in our iterative setting. Reverse KL weights
each gradient update by the student’s own distribution, so the student
only adjusts in token regions it actually generates. This is
mode-seeking: it removes tokens the concise teacher avoids while leaving
the reasoning steps the teacher still uses, and it keeps updates small
because the student already covers the teacher’s high-probability modes.
Forward KL instead weights updates by the teacher’s distribution,
decoupling the update magnitude from how far the student has drifted.
Since our teacher is a stale copy of the student rather than a fixed
external model, this is unstable in practice
(Appendix [G](#A7 "Appendix G Effect of KL Divergence Direction in CRISP ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")):
on Qwen3-8B, forward KL collapses within the first $`{\sim}100`$ steps,
with accuracy falling to near zero and response length diverging to the
token budget, whereas reverse KL trains stably.

### 3.3 Teacher Parameterization

A natural baseline is a *fully frozen* teacher
($`\bar{\theta}=\theta_{0}`$) as in [Zhao et al. (2026)](#bib.bib38).
While simple and stable, the frozen teacher becomes an increasingly weak
compression target as the student improves: once the student has
internalized the initial conciseness signal, no further compression is
possible because the reference distribution no longer leads the student.

To address this, we adopt a *periodic teacher update* strategy. The
teacher weights are synchronized with the current student weights every
$`M`$ training steps:

|     |                                                               |     |     |
|-----|---------------------------------------------------------------|-----|-----|
|     |
       ``` math
       \bar{\theta}\leftarrow\theta\quad\text{every }M\text{ steps}.
       ```                                                            |     | (2) |

Each refresh creates a new, stronger compression target: the updated
teacher, when conditioned on the conciseness instruction $`c`$, produces
traces that are more concise than the previous teacher’s (since the
student, now serving as the new teacher, has already learned to
compress). This *progressive compression* effect pushes the student to
continuously shorten its reasoning over the course of training, beyond
what a single frozen reference can achieve.

##### Difficulty-adaptive compression.

Compression adapts naturally to problem difficulty: for easy problems,
the concise teacher produces much shorter traces, creating strong KL
signal; for hard problems, even the teacher needs extensive reasoning,
yielding weak signal. We formalize this in
Proposition [1](#Thmproposition1 "Proposition 1 (Difficulty-adaptive compression signal). ‣ A.3 Difficulty-Adaptive Compression ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
and verify it empirically in
Section [5.2](#S5.SS2.SSS0.Px2 "Compression naturally adapts to problem difficulty. ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

### 3.4 Training Algorithm

The complete CRISP training procedure is given in
Algorithm [1](#algorithm1 "In 3.4 Training Algorithm ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

Input: Model $`\pi_{\theta}`$, dataset $`\mathcal{D}=\{x_{i}\}`$,
conciseness instruction $`c`$, learning rate $`\eta`$, teacher update
interval $`M`$

Output: Compressed reasoning model $`\pi_{\theta^{*}}`$

Initialize teacher: $`\bar{\theta}\leftarrow\theta_{0}`$;

for *each training step $`k=1,2,\ldots`$* do

   if *$`k\bmod M=0`$* then

      Update teacher: $`\bar{\theta}\leftarrow\theta`$ ; // periodic
refresh

   end if

   Sample batch $`\{x_{1},\ldots,x_{B}\}\sim\mathcal{D}`$;

   for *each $`x_{i}`$ in batch* do

      Generate student rollout:
$`y_{i}\sim\pi_{\theta}(\cdot\mid x_{i})`$;

      for *each token position $`t=1,\ldots,|y_{i}|`$* do

         Compute student logits:
$`q_{t}\leftarrow\pi_{\theta}(\cdot\mid x_{i},y_{i,<t})`$;

         Compute teacher logits:
$`p_{t}\leftarrow\pi_{\bar{\theta}}(\cdot\mid x_{i},c,y_{i,<t})`$ ; //
no grad

         Compute $`D_{\mathrm{KL}}(q_{t}\|p_{t})`$;

      end for

      $`\mathcal{L}_{i}\leftarrow\sum_{t}D_{\mathrm{KL}}(q_{t}\|p_{t})`$;

   end for

   Update student:
$`\theta\leftarrow\theta-\eta\nabla_{\theta}\frac{1}{B}\sum_{i}\mathcal{L}_{i}`$;

   ; // normalized by $`|y_{i}|`$ in practice

end for

return $`\pi_{\theta^{*}}`$;

Algorithm 1 CRISP: On-Policy Self-Distillation for Concise Reasoning

##### Computational cost and simplicity.

The entire training pipeline requires only standard supervised training
infrastructure: no reward models, no value functions, no advantage
estimation, and no multi-rollout sampling. Each training step requires
two forward passes per rollout token: one for the student (with
gradient) and one for the teacher (without gradient, and cacheable
within each $`M`$-step window). The periodic teacher refresh
(Eq. [2](#S3.E2 "In 3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
is a simple weight copy with negligible cost. This simplicity yields
substantial efficiency gains over RL methods, which require multiple
rollouts per prompt, reward model inference, and complex optimization
(e.g., PPO clipping, GAE).

The per-token KL objective also does not require complete rollouts,
because it supplies a training signal at every position rather than only
at the final answer. Truncated rollouts therefore suffice: our ablation
(Section [5.3.5](#S5.SS3.SSS5 "5.3.5 How Sensitive Is Compression to the Rollout Length? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
shows that CRISP trains robustly across maximum rollout lengths of 1K,
4K, 8K, and 30K tokens, with comparable compression and accuracy. Short
rollouts cut generation cost, the dominant expense in on-policy
training, without degrading the result. This is a further advantage over
outcome-reward RL, which needs full-length completions to compute a
terminal reward.

## 4 Theoretical Analysis

We now summarize key theoretical properties of CRISP that illuminate why
such a simple objective can produce strong compression without the
failure modes of length-penalized RL. In particular, we connect the
per-token loss to sequence-level KL, interpret the update as implicit
reward maximization, and analyze when compression preserves accuracy,
adapts to difficulty, and avoids catastrophic forgetting. Proof sketches
are provided inline; full proofs are deferred to
Appendix [A](#A1 "Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

### 4.1 Training Loss as Sequence-Level KL

The first result connects the practical per-token training objective to
a standard information-theoretic quantity, enabling all subsequent
analysis.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzQuU1MxLnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMTguMyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAxMTguMyIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwxMTguMykgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM0RDRENEQ7IiBmaWxsPSIjNEQ0RDREIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMy40NiBMIDAgMTE0Ljg0IEMgMCAxMTYuNzUgMS41NSAxMTguMyAzLjQ2IDExOC4zIEwgNDczLjkyIDExOC4zIEMgNDc1LjgzIDExOC4zIDQ3Ny4zOCAxMTYuNzUgNDc3LjM4IDExNC44NCBMIDQ3Ny4zOCAzLjQ2IEMgNDc3LjM4IDEuNTUgNDc1LjgzIDAgNDczLjkyIDAgTCAzLjQ2IDAgQyAxLjU1IDAgMCAxLjU1IDAgMy40NiBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjVGNUY1OyIgZmlsbD0iI0Y1RjVGNSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwLjY5IDMuNDYgTCAwLjY5IDExNC44NCBDIDAuNjkgMTE2LjM3IDEuOTMgMTE3LjYxIDMuNDYgMTE3LjYxIEwgNDczLjkyIDExNy42MSBDIDQ3NS40NSAxMTcuNjEgNDc2LjY4IDExNi4zNyA0NzYuNjggMTE0Ljg0IEwgNDc2LjY4IDMuNDYgQyA0NzYuNjggMS45MyA0NzUuNDUgMC42OSA0NzMuOTIgMC42OSBMIDMuNDYgMC42OSBDIDEuOTMgMC42OSAwLjY5IDEuOTMgMC42OSAzLjQ2IFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTEuNTUgMTQuMjQpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzIuODNlbTstLWx0eC1mby1oZWlnaHQ6Ni42OWVtOy0tbHR4LWZvLWRlcHRoOjAuMTllbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iOTUuMjEiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDkyLjUyKSIgd2lkdGg9IjQ1NC4yNyI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNC5TUzEucDIucGljMS4xIiBjbGFzcz0ibHR4X2lubGluZS1sb2dpY2FsLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzIuODNlbTsiPgo8c3BhbiBpZD0iUzQuU1MxLnAyLnBpYzEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iUzQuU1MxLnAyLnBpYzEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzQuU1MxLnAyLnBpYzEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5MZW1tYTwvc3Bhbj48c3BhbiBpZD0iUzQuU1MxLnAyLnBpYzEucDEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+wqBbVHJhaW5pbmcgbG9zcyBlcXVhbHMgc2VxdWVuY2UtbGV2ZWwgS0w7IExlbW1hwqA8L3NwYW4+PGEgaHJlZj0iI1RobWxlbW1hMSIgdGl0bGU9IkxlbW1hIDEgKENoYWluIHJ1bGUgb2YgS0wgZm9yIGF1dG9yZWdyZXNzaXZlIG1vZGVscykuIOKAoyBBLjEgU2VxdWVuY2UtTGV2ZWwgRGl2ZXJnZW5jZSBhbmQgdGhlIFRyYWluaW5nIE9iamVjdGl2ZSDigKMgQXBwZW5kaXggQSBUaGVvcmV0aWNhbCBBbmFseXNpcyDigKMgQ1JJU1A6IENvbXByZXNzZWQgUmVhc29uaW5nIHZpYSBJdGVyYXRpdmUgU2VsZi1Qb2xpY3kgRGlzdGlsbGF0aW9uIiBjbGFzcz0ibHR4X3JlZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij48c3BhbiBjbGFzcz0ibHR4X3RleHQgbHR4X3JlZl90YWciPjE8L3NwYW4+PC9hPjxzcGFuIGlkPSJTNC5TUzEucDIucGljMS5wMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5dLsKgClRoZSBwZXItdG9rZW4gQ1JJU1AgbG9zcyAoRXEuwqA8L3NwYW4+PGEgaHJlZj0iI1MzLkUxIiB0aXRsZT0iSW4gMy4yIFRyYWluaW5nIE9iamVjdGl2ZSDigKMgMyBNZXRob2Qg4oCjIENSSVNQOiBDb21wcmVzc2VkIFJlYXNvbmluZyB2aWEgSXRlcmF0aXZlIFNlbGYtUG9saWN5IERpc3RpbGxhdGlvbiIgY2xhc3M9Imx0eF9yZWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+PHNwYW4gY2xhc3M9Imx0eF90ZXh0IGx0eF9yZWZfdGFnIj4xPC9zcGFuPjwvYT48c3BhbiBpZD0iUzQuU1MxLnAyLnBpYzEucDEuMS40IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+KSBlcXVhbHMgdGhlIHNlcXVlbmNlLWxldmVsIEtMIGRpdmVyZ2VuY2UgPC9zcGFuPjxtYXRoIGlkPSJTNC5TUzEucDIucGljMS5wMS5tMSIgY2xhc3M9Imx0eF9tYXRoX3VucGFyc2VkIiBhbHR0ZXh0PSJEX3tcbWF0aHJte0tMfX0oXHBpX3tcdGhldGF9KFxjZG90XG1pZCB4KVx8XHBpX3tcYmFye1x0aGV0YX19KFxjZG90XG1pZCB4LGMpKSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+RDwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5LTDwvbWk+PC9tc3ViPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs+APC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs64PC9taT48L21zdWI+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMGVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMGVtIj7ii4U8L21vPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMGVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMC4xNjdlbSI+4oijPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjE2N2VtIj7iiKU8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+z4A8L21pPjxtb3ZlciBhY2NlbnQ9InRydWUiPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+zrg8L21pPjxtbz7CrzwvbW8+PC9tb3Zlcj48L21zdWI+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMGVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMGVtIj7ii4U8L21vPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMGVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMC4xNjdlbSI+4oijPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LDwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5jPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KTwvbW8+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+RF97XG1hdGhybXtLTH19KFxwaV97XHRoZXRhfShcY2RvdFxtaWQgeClcfFxwaV97XGJhcntcdGhldGF9fShcY2RvdFxtaWQgeCxjKSk8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTNC5TUzEucDIucGljMS5wMS4xLjUiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gYnkgdGhlIGNoYWluIHJ1bGUgZm9yIGF1dG9yZWdyZXNzaXZlIG1vZGVscy48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)

###### Proof sketch.

By the autoregressive factorization
$`q(y\mid x)=\prod_{t}q(y_{t}\mid x,y_{<t})`$, the log-ratio
$`\log\frac{q(y\mid x)}{p(y\mid x)}`$ decomposes into
$`\sum_{t}\log\frac{q(y_{t}\mid x,y_{<t})}{p(y_{t}\mid x,y_{<t})}`$.
Taking expectations over $`y\sim q`$ yields the per-token KL sum, which
equals the sequence-level KL by definition. ∎

This identification underpins all subsequent results by letting us apply
standard information-theoretic tools (Pinsker’s inequality, the
data-processing inequality) to the per-token loss.

### 4.2 Implicit Reward Interpretation

Following the inverse RL framework of [Shenfeld et al.
(2026)](#bib.bib26), we show that CRISP implicitly maximizes a reward
function that combines task performance with a conciseness preference.

###### Theorem 1 (Implicit reward).

The CRISP objective
(Eq. [1](#S3.E1 "In 3.2 Training Objective ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
is equivalent to maximizing the expected implicit reward:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
r(y_{t},x)=\log\pi_{\bar{\theta}}(y_{t}\mid x,c,y_{<t})-\log\pi_{\theta}(y_{t}\mid x,y_{<t}).
``` |  | (3) |

###### Proof sketch.

Expanding the reverse KL:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot\mid x,y_{<t})\|\pi_{\bar{\theta}}(\cdot\mid x,c,y_{<t})\big)`$ | $`\displaystyle=\mathbb{E}_{y_{t}\sim\pi_{\theta}}\left[\log\frac{\pi_{\theta}(y_{t}\mid x,y_{<t})}{\pi_{\bar{\theta}}(y_{t}\mid x,c,y_{<t})}\right]`$ |  | (4) |
|  |  | $`\displaystyle=-\mathbb{E}_{y_{t}\sim\pi_{\theta}}\big[r(y_{t},x)\big].`$ |  | (5) |

Since $`r(y_{t},x)=\log\pi_{\bar{\theta}}-\log\pi_{\theta}`$ naturally
decomposes into a teacher-favoring term $`\log\pi_{\bar{\theta}}`$ and
an entropy-like term $`-\log\pi_{\theta}`$, minimizing reverse KL can be
interpreted as maximizing an implicit, policy-dependent reward-shaping
objective. This is closely related to maximum-entropy RL, but not
identical to the standard setting with a fixed environment reward. ∎

###### Remark 1.

The implicit reward $`r(y_{t},x)`$ is positive when the concise teacher
assigns higher probability to token $`y_{t}`$ than the student does, and
negative otherwise. Thus, CRISP implicitly rewards concise reasoning
without any explicit length penalty. Within each $`M`$-step window, the
teacher weights $`\bar{\theta}`$ are fixed, serving as an implicit trust
region: the student can only move as far as the teacher’s concise
distribution allows, preventing unbounded policy drift. The periodic
refresh
(Eq. [2](#S3.E2 "In 3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
then shifts this trust region forward, enabling progressive compression
across windows.

### 4.3 Accuracy Preservation

A natural concern is whether compression degrades accuracy. The
following theorem shows that accuracy loss is bounded by two
interpretable quantities.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzQuU1MzLnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMDIuNTgiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NzcuMzggMTAyLjU4IiB3aWR0aD0iNDc3LjM4Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDEwMi41OCkgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM0RDRENEQ7IiBmaWxsPSIjNEQ0RDREIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMy40NiBMIDAgOTkuMTIgQyAwIDEwMS4wMyAxLjU1IDEwMi41OCAzLjQ2IDEwMi41OCBMIDQ3My45MiAxMDIuNTggQyA0NzUuODMgMTAyLjU4IDQ3Ny4zOCAxMDEuMDMgNDc3LjM4IDk5LjEyIEwgNDc3LjM4IDMuNDYgQyA0NzcuMzggMS41NSA0NzUuODMgMCA0NzMuOTIgMCBMIDMuNDYgMCBDIDEuNTUgMCAwIDEuNTUgMCAzLjQ2IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGNUY1RjU7IiBmaWxsPSIjRjVGNUY1IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAuNjkgMy40NiBMIDAuNjkgOTkuMTIgQyAwLjY5IDEwMC42NSAxLjkzIDEwMS44OSAzLjQ2IDEwMS44OSBMIDQ3My45MiAxMDEuODkgQyA0NzUuNDUgMTAxLjg5IDQ3Ni42OCAxMDAuNjUgNDc2LjY4IDk5LjEyIEwgNDc2LjY4IDMuNDYgQyA0NzYuNjggMS45MyA0NzUuNDUgMC42OSA0NzMuOTIgMC42OSBMIDMuNDYgMC42OSBDIDEuOTMgMC42OSAwLjY5IDEuOTMgMC42OSAzLjQ2IFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTEuNTUgMTUuMDEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzIuODNlbTstLWx0eC1mby1oZWlnaHQ6NS40OWVtOy0tbHR4LWZvLWRlcHRoOjAuMjVlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNzkuNDkiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDc2LjAzKSIgd2lkdGg9IjQ1NC4yNyI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNC5TUzMucDIucGljMS4xIiBjbGFzcz0ibHR4X2lubGluZS1sb2dpY2FsLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzIuODNlbTsiPgo8c3BhbiBpZD0iUzQuU1MzLnAyLnBpYzEucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPgo8c3BhbiBpZD0iUzQuU1MzLnAyLnBpYzEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzQuU1MzLnAyLnBpYzEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGVvcmVtPC9zcGFuPjxzcGFuIGlkPSJTNC5TUzMucDIucGljMS5wMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij7CoFtBY2N1cmFjeSBwcmVzZXJ2YXRpb247IFRoZW9yZW3CoDwvc3Bhbj48YSBocmVmPSIjVGhtdGhlb3JlbTIiIHRpdGxlPSJUaGVvcmVtIDIgKEFjY3VyYWN5IHByZXNlcnZhdGlvbikuIOKAoyBBLjIgQWNjdXJhY3kgUHJlc2VydmF0aW9uIHVuZGVyIENvbXByZXNzaW9uIOKAoyBBcHBlbmRpeCBBIFRoZW9yZXRpY2FsIEFuYWx5c2lzIOKAoyBDUklTUDogQ29tcHJlc3NlZCBSZWFzb25pbmcgdmlhIEl0ZXJhdGl2ZSBTZWxmLVBvbGljeSBEaXN0aWxsYXRpb24iIGNsYXNzPSJsdHhfcmVmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjxzcGFuIGNsYXNzPSJsdHhfdGV4dCBsdHhfcmVmX3RhZyI+Mjwvc3Bhbj48L2E+PHNwYW4gaWQ9IlM0LlNTMy5wMi5waWMxLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPl0uwqAKSWYgdHJhaW5pbmcgY29udmVyZ2VzIHRvIGxvc3MgPC9zcGFuPjxtYXRoIGlkPSJTNC5TUzMucDIucGljMS5wMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcZXBzaWxvbl97XG1hdGhybXtLTH19IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7PtTwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5LTDwvbWk+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGVwc2lsb25fe1xtYXRocm17S0x9fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM0LlNTMy5wMi5waWMxLnAxLjEuNCIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiBhbmQgdGhlIGNvbmNpc2UgdGVhY2hlciBwcmVzZXJ2ZXMgYWNjdXJhY3kgdG8gd2l0aGluIDwvc3Bhbj48bWF0aCBpZD0iUzQuU1MzLnAyLnBpYzEucDEubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGVwc2lsb25fe1R9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7PtTwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5UPC9taT48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cZXBzaWxvbl97VH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTNC5TUzMucDIucGljMS5wMS4xLjUiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gb2YgdGhlIGJhc2UgbW9kZWwsIHRoZSBzdHVkZW50IHNhdGlzZmllczo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRTYiIGNsYXNzPSJsdHhfZXF1YXRpb24gbHR4X2Vxbl90YWJsZSI+Cgo8c3Bhbj48c3BhbiBjbGFzcz0ibHR4X2VxdWF0aW9uIGx0eF9lcW5fcm93IGx0eF9hbGlnbl9iYXNlbGluZSI+CjxzcGFuIGNsYXNzPSJsdHhfZXFuX2NlbGwgbHR4X2Vxbl9jZW50ZXJfcGFkbGVmdCI+PC9zcGFuPgo8c3BhbiBjbGFzcz0ibHR4X2Vxbl9jZWxsIGx0eF9hbGlnbl9jZW50ZXIiPjxtYXRoIGlkPSJTNC5FNi5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aHJte0FjY30oXHBpX3tcdGhldGFeeyp9fSlcZ2VxXG1hdGhybXtBY2N9KFxwaV97XGJhcntcdGhldGF9fSktXGVwc2lsb25fe1R9LVxzcXJ0e1xlcHNpbG9uX3tcbWF0aHJte0tMfX0vMn0uIiBkaXNwbGF5PSJibG9jayIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPkFjYzwvbWk+PG1vPuKBoTwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+z4A8L21pPjxtc3VwPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+zrg8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4oiXPC9tbz48L21zdXA+PC9tc3ViPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4omlPC9tbz48bXJvdz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPkFjYzwvbWk+PG1vPuKBoTwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+z4A8L21pPjxtb3ZlciBhY2NlbnQ9InRydWUiPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+zrg8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+wq88L21vPjwvbW92ZXI+PC9tc3ViPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4oiSPC9tbz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs+1PC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPlQ8L21pPjwvbXN1Yj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPuKIkjwvbW8+PG1zcXJ0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bXJvdz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs+1PC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPktMPC9taT48L21zdWI+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4vPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjwvbXJvdz48L21zcXJ0PjwvbXJvdz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCI+LjwvbW8+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhybXtBY2N9KFxwaV97XHRoZXRhXnsqfX0pXGdlcVxtYXRocm17QWNjfShccGlfe1xiYXJ7XHRoZXRhfX0pLVxlcHNpbG9uX3tUfS1cc3FydHtcZXBzaWxvbl97XG1hdGhybXtLTH19LzJ9LjwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPgo8c3BhbiBjbGFzcz0ibHR4X2Vxbl9jZWxsIGx0eF9lcW5fY2VudGVyX3BhZHJpZ2h0Ij48L3NwYW4+CjxzcGFuIHJvd3NwYW49IjEiIGNsYXNzPSJsdHhfZXFuX2NlbGwgbHR4X2Vxbl9lcW5vIGx0eF9hbGlnbl9taWRkbGUgbHR4X2FsaWduX3JpZ2h0Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2VxdWF0aW9uIGx0eF9hbGlnbl9yaWdodCI+KDYpPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

###### Proof sketch.

Apply Pinsker’s inequality to convert the KL bound
$`\epsilon_{\mathrm{KL}}`$ into a total variation bound
$`\sqrt{\epsilon_{\mathrm{KL}}/2}`$ between student and teacher
distributions. Total variation bounds the difference in probability of
any event, in particular the correctness event $`A(x)`$, so the
student’s accuracy is within $`\sqrt{\epsilon_{\mathrm{KL}}/2}`$ of the
teacher’s. Combining with the teacher quality assumption
$`\epsilon_{T}`$ via the triangle inequality yields the result. ∎

The bound decomposes accuracy loss into two independent, interpretable
terms: teacher quality ($`\epsilon_{T}`$) and distillation gap
($`\sqrt{\epsilon_{\mathrm{KL}}/2}`$). When the concise teacher is at
least as accurate as the base model ($`\epsilon_{T}\leq 0`$), the bound
becomes
$`\mathrm{Acc}(\pi_{\theta^{*}})\geq\mathrm{Acc}(\pi_{\bar{\theta}})+|\epsilon_{T}|-\sqrt{\epsilon_{\mathrm{KL}}/2}`$,
so accuracy is preserved, and improves whenever the teacher’s gain
exceeds the distillation gap. This matches
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"):
accuracy holds within about a point on the already-strong Qwen3 models
and improves where the base model has headroom (e.g.
DeepSeek-R1-Distill-Llama-8B).

### 4.4 Difficulty-Adaptive Compression

A key design question for any compression method is how to allocate
budget across problems of varying difficulty. We show that CRISP handles
this *automatically*: the compression signal is provably stronger on
easy problems.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzQuU1M0LnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMTguNDIiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NzcuMzggMTE4LjQyIiB3aWR0aD0iNDc3LjM4Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDExOC40MikgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM0RDRENEQ7IiBmaWxsPSIjNEQ0RDREIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMy40NiBMIDAgMTE0Ljk2IEMgMCAxMTYuODcgMS41NSAxMTguNDIgMy40NiAxMTguNDIgTCA0NzMuOTIgMTE4LjQyIEMgNDc1LjgzIDExOC40MiA0NzcuMzggMTE2Ljg3IDQ3Ny4zOCAxMTQuOTYgTCA0NzcuMzggMy40NiBDIDQ3Ny4zOCAxLjU1IDQ3NS44MyAwIDQ3My45MiAwIEwgMy40NiAwIEMgMS41NSAwIDAgMS41NSAwIDMuNDYgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0Y1RjVGNTsiIGZpbGw9IiNGNUY1RjUiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC42OSAzLjQ2IEwgMC42OSAxMTQuOTYgQyAwLjY5IDExNi40OSAxLjkzIDExNy43MiAzLjQ2IDExNy43MiBMIDQ3My45MiAxMTcuNzIgQyA0NzUuNDUgMTE3LjcyIDQ3Ni42OCAxMTYuNDkgNDc2LjY4IDExNC45NiBMIDQ3Ni42OCAzLjQ2IEMgNDc2LjY4IDEuOTMgNDc1LjQ1IDAuNjkgNDczLjkyIDAuNjkgTCAzLjQ2IDAuNjkgQyAxLjkzIDAuNjkgMC42OSAxLjkzIDAuNjkgMy40NiBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDExLjU1IDE0LjI0KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjMyLjgzZW07LS1sdHgtZm8taGVpZ2h0OjYuNjllbTstLWx0eC1mby1kZXB0aDowLjE5ZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9Ijk1LjMyIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5Mi42MykiIHdpZHRoPSI0NTQuMjciPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iUzQuU1M0LnAyLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjMyLjgzZW07Ij4KPHNwYW4gaWQ9IlM0LlNTNC5wMi5waWMxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IlM0LlNTNC5wMi5waWMxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM0LlNTNC5wMi5waWMxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+UHJvcG9zaXRpb248L3NwYW4+PHNwYW4gaWQ9IlM0LlNTNC5wMi5waWMxLnAxLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPsKgW0RpZmZpY3VsdHktYWRhcHRpdmUgY29tcHJlc3Npb247IFByb3Bvc2l0aW9uwqA8L3NwYW4+PGEgaHJlZj0iI1RobXByb3Bvc2l0aW9uMSIgdGl0bGU9IlByb3Bvc2l0aW9uIDEgKERpZmZpY3VsdHktYWRhcHRpdmUgY29tcHJlc3Npb24gc2lnbmFsKS4g4oCjIEEuMyBEaWZmaWN1bHR5LUFkYXB0aXZlIENvbXByZXNzaW9uIOKAoyBBcHBlbmRpeCBBIFRoZW9yZXRpY2FsIEFuYWx5c2lzIOKAoyBDUklTUDogQ29tcHJlc3NlZCBSZWFzb25pbmcgdmlhIEl0ZXJhdGl2ZSBTZWxmLVBvbGljeSBEaXN0aWxsYXRpb24iIGNsYXNzPSJsdHhfcmVmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPjxzcGFuIGNsYXNzPSJsdHhfdGV4dCBsdHhfcmVmX3RhZyI+MTwvc3Bhbj48L2E+PHNwYW4gaWQ9IlM0LlNTNC5wMi5waWMxLnAxLjEuMyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPl0uwqAKVGhlIHBlci10b2tlbiBjb21wcmVzc2lvbiBzaWduYWwgPC9zcGFuPjxtYXRoIGlkPSJTNC5TUzQucDIucGljMS5wMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJTKHgpIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5TPC9taT48bW8+4oGhPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KDwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KTwvbW8+PC9tcm93PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlMoeCk8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTNC5TUzQucDIucGljMS5wMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gaXMgbm9uLWluY3JlYXNpbmcgaW4gcHJvYmxlbSBkaWZmaWN1bHR5IDwvc3Bhbj48bWF0aCBpZD0iUzQuU1M0LnAyLnBpYzEucDEubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iZCh4KSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ZDwvbWk+PG1vPuKBoTwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+eDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5kKHgpPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iUzQuU1M0LnAyLnBpYzEucDEuMS41IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+OiBlYXN5IHByb2JsZW1zIHJlY2VpdmUgc3Ryb25nIGNvbXByZXNzaW9uIHByZXNzdXJlIHdoaWxlIGhhcmQgcHJvYmxlbXMsIHdob3NlIHJlYXNvbmluZyBzdGVwcyBhcmUgcHJlZG9taW5hbnRseSBlc3NlbnRpYWwsIHJlY2VpdmUgd2VhayBwcmVzc3VyZS48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)

###### Proof sketch.

Decompose the normalized KL into essential and compressible token
contributions:
$`S(x)=\rho(x)\cdot D_{\mathcal{E}}+(1-\rho(x))\cdot D_{\mathcal{C}}`$,
where $`\rho(x)`$ is the fraction of essential tokens, and
$`D_{\mathcal{E}},D_{\mathcal{C}}`$ are the category-level KL
divergences. Since compressible tokens carry strictly larger KL
($`D_{\mathcal{C}}>D_{\mathcal{E}}`$) and the essential fraction
$`\rho(x)`$ is non-decreasing in difficulty,
$`S(x)=D_{\mathcal{C}}-\rho(x)(D_{\mathcal{C}}-D_{\mathcal{E}})`$ is a
decreasing affine function of $`\rho(x)`$, hence non-increasing in
$`d(x)`$. ∎

This formalizes the empirical pattern
(Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
that CRISP compresses the easier MATH-500 more aggressively than the
harder AIME benchmarks: on Qwen3-14B, reductions are up to 56% on
MATH-500 but only 32–38% on AIME, without any explicit difficulty
estimation.

### 4.5 Bounded Forgetting

A central advantage of on-policy self-distillation over off-policy SFT
is controlled divergence from the original model. We formalize this via
the *conciseness gap*.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzQuU1M1LnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMTguNDIiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NzcuMzggMTE4LjQyIiB3aWR0aD0iNDc3LjM4Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDExOC40MikgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM0RDRENEQ7IiBmaWxsPSIjNEQ0RDREIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMy40NiBMIDAgMTE0Ljk2IEMgMCAxMTYuODcgMS41NSAxMTguNDIgMy40NiAxMTguNDIgTCA0NzMuOTIgMTE4LjQyIEMgNDc1LjgzIDExOC40MiA0NzcuMzggMTE2Ljg3IDQ3Ny4zOCAxMTQuOTYgTCA0NzcuMzggMy40NiBDIDQ3Ny4zOCAxLjU1IDQ3NS44MyAwIDQ3My45MiAwIEwgMy40NiAwIEMgMS41NSAwIDAgMS41NSAwIDMuNDYgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0Y1RjVGNTsiIGZpbGw9IiNGNUY1RjUiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC42OSAzLjQ2IEwgMC42OSAxMTQuOTYgQyAwLjY5IDExNi40OSAxLjkzIDExNy43MiAzLjQ2IDExNy43MiBMIDQ3My45MiAxMTcuNzIgQyA0NzUuNDUgMTE3LjcyIDQ3Ni42OCAxMTYuNDkgNDc2LjY4IDExNC45NiBMIDQ3Ni42OCAzLjQ2IEMgNDc2LjY4IDEuOTMgNDc1LjQ1IDAuNjkgNDczLjkyIDAuNjkgTCAzLjQ2IDAuNjkgQyAxLjkzIDAuNjkgMC42OSAxLjkzIDAuNjkgMy40NiBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDExLjU1IDE0LjI0KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjMyLjgzZW07LS1sdHgtZm8taGVpZ2h0OjYuNjllbTstLWx0eC1mby1kZXB0aDowLjE5ZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9Ijk1LjMyIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5Mi42MykiIHdpZHRoPSI0NTQuMjciPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iUzQuU1M1LnAyLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjMyLjgzZW07Ij4KPHNwYW4gaWQ9IlM0LlNTNS5wMi5waWMxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IlM0LlNTNS5wMi5waWMxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM0LlNTNS5wMi5waWMxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+UHJvcG9zaXRpb248L3NwYW4+PHNwYW4gaWQ9IlM0LlNTNS5wMi5waWMxLnAxLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPsKgW0JvdW5kZWQgZm9yZ2V0dGluZzsgUHJvcG9zaXRpb27CoDwvc3Bhbj48YSBocmVmPSIjVGhtcHJvcG9zaXRpb24yIiB0aXRsZT0iUHJvcG9zaXRpb24gMiAoQm91bmRlZCBmb3JnZXR0aW5nIHVuZGVyIG9uLXBvbGljeSBzZWxmLWRpc3RpbGxhdGlvbikuIOKAoyBBLjQgQm91bmRlZCBGb3JnZXR0aW5nIGZyb20gdGhlIEJhc2UgTW9kZWwg4oCjIEFwcGVuZGl4IEEgVGhlb3JldGljYWwgQW5hbHlzaXMg4oCjIENSSVNQOiBDb21wcmVzc2VkIFJlYXNvbmluZyB2aWEgSXRlcmF0aXZlIFNlbGYtUG9saWN5IERpc3RpbGxhdGlvbiIgY2xhc3M9Imx0eF9yZWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+PHNwYW4gY2xhc3M9Imx0eF90ZXh0IGx0eF9yZWZfdGFnIj4yPC9zcGFuPjwvYT48c3BhbiBpZD0iUzQuU1M1LnAyLnBpYzEucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+XS7CoApEaXZlcmdlbmNlIGZyb20gdGhlIGJhc2UgbW9kZWwgaXMgYm91bmRlZCBieTo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzQuRTciIGNsYXNzPSJsdHhfZXF1YXRpb24gbHR4X2Vxbl90YWJsZSI+Cgo8c3Bhbj48c3BhbiBjbGFzcz0ibHR4X2VxdWF0aW9uIGx0eF9lcW5fcm93IGx0eF9hbGlnbl9iYXNlbGluZSI+CjxzcGFuIGNsYXNzPSJsdHhfZXFuX2NlbGwgbHR4X2Vxbl9jZW50ZXJfcGFkbGVmdCI+PC9zcGFuPgo8c3BhbiBjbGFzcz0ibHR4X2Vxbl9jZWxsIGx0eF9hbGlnbl9jZW50ZXIiPjxtYXRoIGlkPSJTNC5FNy5tMSIgY2xhc3M9Imx0eF9tYXRoX3VucGFyc2VkIiBhbHR0ZXh0PSJcbWF0aGJie0V9X3t4fVxiaWdbZF97XG1hdGhybXtUVn19KFxwaV97XHRoZXRhXnsqfX0oXGNkb3RcbWlkIHgpLFwsXHBpX3tcdGhldGFfezB9fShcY2RvdFxtaWQgeCkpXGJpZ11cbGVxXHNxcnR7XGVwc2lsb25fe1xtYXRocm17S0x9fS8yfStcbWF0aGJie0V9X3t4fVtcZ2FtbWEoeCldLCIgZGlzcGxheT0iYmxvY2siIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7wnZS8PC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjwvbXN1Yj48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1heHNpemU9IjEuMjAwZW0iIG1pbnNpemU9IjEuMjAwZW0iPls8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ZDwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5UVjwvbWk+PC9tc3ViPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs+APC9taT48bXN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs64PC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPuKIlzwvbW8+PC9tc3VwPjwvbXN1Yj48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KDwvbW8+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwZW0iPuKLhTwvbW8+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjE2N2VtIj7iiKM8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+eDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMC4zMzdlbSI+LDwvbW8+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7PgDwvbWk+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7OuDwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4wPC9tbj48L21zdWI+PC9tc3ViPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjBlbSIgbWF0aGNvbG9yPSIjMDAwMDAwIiByc3BhY2U9IjBlbSI+4ouFPC9tbz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjBlbSIgbWF0aGNvbG9yPSIjMDAwMDAwIiByc3BhY2U9IjAuMTY3ZW0iPuKIozwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+KTwvbW8+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXhzaXplPSIxLjIwMGVtIiBtaW5zaXplPSIxLjIwMGVtIj5dPC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7iiaQ8L21vPjxtc3FydCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PG1yb3c+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7PtTwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5LTDwvbWk+PC9tc3ViPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48L21yb3c+PC9tc3FydD48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+8J2UvDwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48L21zdWI+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPls8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+zrM8L21pPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPl08L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPiw8L21vPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoYmJ7RX1fe3h9XGJpZ1tkX3tcbWF0aHJte1RWfX0oXHBpX3tcdGhldGFeeyp9fShcY2RvdFxtaWQgeCksXCxccGlfe1x0aGV0YV97MH19KFxjZG90XG1pZCB4KSlcYmlnXVxsZXFcc3FydHtcZXBzaWxvbl97XG1hdGhybXtLTH19LzJ9K1xtYXRoYmJ7RX1fe3h9W1xnYW1tYSh4KV0sPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+CjxzcGFuIGNsYXNzPSJsdHhfZXFuX2NlbGwgbHR4X2Vxbl9jZW50ZXJfcGFkcmlnaHQiPjwvc3Bhbj4KPHNwYW4gcm93c3Bhbj0iMSIgY2xhc3M9Imx0eF9lcW5fY2VsbCBsdHhfZXFuX2Vxbm8gbHR4X2FsaWduX21pZGRsZSBsdHhfYWxpZ25fcmlnaHQiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfZXF1YXRpb24gbHR4X2FsaWduX3JpZ2h0Ij4oNyk8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IlM0LlNTNS5wMi5waWMxLnAxLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM0LlNTNS5wMi5waWMxLnAxLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPndoZXJlIDwvc3Bhbj48bWF0aCBpZD0iUzQuU1M1LnAyLnBpYzEucDEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGdhbW1hKHgpIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7OszwvbWk+PG1vPuKBoTwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+eDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cZ2FtbWEoeCk8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTNC5TUzUucDIucGljMS5wMS4yLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gaXMgdGhlIGNvbmNpc2VuZXNzIGdhcCwgaS5lLiwgdGhlIHRvdGFsIHZhcmlhdGlvbiBiZXR3ZWVuIHRoZSBiYXNlIG1vZGVs4oCZcyBvdXRwdXRzIHdpdGggYW5kIHdpdGhvdXQgdGhlIGNvbmNpc2VuZXNzIGluc3RydWN0aW9uLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

###### Proof sketch.

Apply the triangle inequality for total variation:
$`d_{\mathrm{TV}}(\pi_{\theta^{*}},\pi_{\theta_{0}})\leq d_{\mathrm{TV}}(\pi_{\theta^{*}},\pi_{\theta_{0}}(\cdot\mid c))+d_{\mathrm{TV}}(\pi_{\theta_{0}}(\cdot\mid c),\pi_{\theta_{0}})`$.
The first term is bounded by $`\sqrt{\epsilon_{\mathrm{KL}}/2}`$ via
Pinsker’s inequality on the converged training loss; the second term is
the conciseness gap $`\gamma(x)`$ by definition. ∎

For hard problems where the conciseness instruction has little effect,
$`\gamma(x)\approx 0`$, so forgetting is minimal where it matters most.
This contrasts with off-policy SFT, whose forgetting depends on the full
distribution mismatch between teacher data and the base model, a gap
that can be arbitrarily large.

### 4.6 Compression Reduces Compounding Error

Finally, we provide a probabilistic model explaining the most striking
empirical finding: shorter reasoning traces can *improve* accuracy
rather than degrade it.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzQuU1M2LnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIxMTUuNzMiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NzcuMzggMTE1LjczIiB3aWR0aD0iNDc3LjM4Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDExNS43MykgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM0RDRENEQ7IiBmaWxsPSIjNEQ0RDREIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMy40NiBMIDAgMTEyLjI3IEMgMCAxMTQuMTggMS41NSAxMTUuNzMgMy40NiAxMTUuNzMgTCA0NzMuOTIgMTE1LjczIEMgNDc1LjgzIDExNS43MyA0NzcuMzggMTE0LjE4IDQ3Ny4zOCAxMTIuMjcgTCA0NzcuMzggMy40NiBDIDQ3Ny4zOCAxLjU1IDQ3NS44MyAwIDQ3My45MiAwIEwgMy40NiAwIEMgMS41NSAwIDAgMS41NSAwIDMuNDYgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0Y1RjVGNTsiIGZpbGw9IiNGNUY1RjUiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC42OSAzLjQ2IEwgMC42OSAxMTIuMjcgQyAwLjY5IDExMy43OSAxLjkzIDExNS4wMyAzLjQ2IDExNS4wMyBMIDQ3My45MiAxMTUuMDMgQyA0NzUuNDUgMTE1LjAzIDQ3Ni42OCAxMTMuNzkgNDc2LjY4IDExMi4yNyBMIDQ3Ni42OCAzLjQ2IEMgNDc2LjY4IDEuOTMgNDc1LjQ1IDAuNjkgNDczLjkyIDAuNjkgTCAzLjQ2IDAuNjkgQyAxLjkzIDAuNjkgMC42OSAxLjkzIDAuNjkgMy40NiBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDExLjU1IDExLjU1KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjMyLjgzZW07LS1sdHgtZm8taGVpZ2h0OjYuNjllbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjkyLjYzIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5Mi42MykiIHdpZHRoPSI0NTQuMjciPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iUzQuU1M2LnAyLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjMyLjgzZW07Ij4KPHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij4KPHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+UHJvcG9zaXRpb248L3NwYW4+PHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPsKgW0NvbXByZXNzaW9uIHJlZHVjZXMgY29tcG91bmRpbmcgZXJyb3I7IFByb3Bvc2l0aW9uwqA8L3NwYW4+PGEgaHJlZj0iI1RobXByb3Bvc2l0aW9uMyIgdGl0bGU9IlByb3Bvc2l0aW9uIDMgKFNob3J0ZXIgdHJhY2VzIHJlZHVjZSBlcnJvciBhY2N1bXVsYXRpb24pLiDigKMgQS41IENvbXByZXNzaW9uIFJlZHVjZXMgQ29tcG91bmRpbmcgRXJyb3Ig4oCjIEFwcGVuZGl4IEEgVGhlb3JldGljYWwgQW5hbHlzaXMg4oCjIENSSVNQOiBDb21wcmVzc2VkIFJlYXNvbmluZyB2aWEgSXRlcmF0aXZlIFNlbGYtUG9saWN5IERpc3RpbGxhdGlvbiIgY2xhc3M9Imx0eF9yZWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+PHNwYW4gY2xhc3M9Imx0eF90ZXh0IGx0eF9yZWZfdGFnIj4zPC9zcGFuPjwvYT48c3BhbiBpZD0iUzQuU1M2LnAyLnBpYzEucDEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+XS7CoApVbmRlciBhIG1vZGVsIHdoZXJlIGVhY2ggdG9rZW4gaW5kZXBlbmRlbnRseSBpbnRyb2R1Y2VzIGEgcmVhc29uaW5nIGVycm9yIHdpdGggcHJvYmFiaWxpdHkgPC9zcGFuPjxtYXRoIGlkPSJTNC5TUzYucDIucGljMS5wMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJwX3tcbWF0aHJte2Vycn19IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5wPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmVycjwvbWk+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+cF97XG1hdGhybXtlcnJ9fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxLjEuNCIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiwgY29tcHJlc3NpbmcgZnJvbSA8L3NwYW4+PG1hdGggaWQ9IlM0LlNTNi5wMi5waWMxLnAxLm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IkwiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPkw8L21pPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+TDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxLjEuNSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiB0byA8L3NwYW4+PG1hdGggaWQ9IlM0LlNTNi5wMi5waWMxLnAxLm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxhbHBoYSBMIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7OsTwvbWk+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPkw8L21pPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxhbHBoYSBMPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iUzQuU1M2LnAyLnBpYzEucDEuMS42IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IHRva2VucyB5aWVsZHMgYW4gYWNjdXJhY3kgcmF0aW8gPC9zcGFuPjxtYXRoIGlkPSJTNC5TUzYucDIucGljMS5wMS5tNCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIoMS1wX3tcbWF0aHJte2Vycn19KV57LSgxLVxhbHBoYSlMfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3VwPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4oiSPC9tbz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPnA8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ZXJyPC9taT48L21zdWI+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7iiJI8L21vPjxtcm93Pjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4oiSPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPs6xPC9taT48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+TDwvbWk+PC9tcm93PjwvbXJvdz48L21zdXA+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4oMS1wX3tcbWF0aHJte2Vycn19KV57LSgxLVxhbHBoYSlMfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM0LlNTNi5wMi5waWMxLnAxLjEuNyIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiwgd2hpY2ggZ3Jvd3MgZXhwb25lbnRpYWxseSBpbiB0aGUgbnVtYmVyIG9mIHJlbW92ZWQgdG9rZW5zLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

###### Proof sketch.

Direct computation: the accuracy ratio
$`(1-p_{\mathrm{err}})^{\alpha L}/(1-p_{\mathrm{err}})^{L}=(1-p_{\mathrm{err}})^{-(1-\alpha)L}`$.
Using $`\ln(1-p)\leq-p`$ and $`e^{u}\geq 1+u`$, this is at least
$`1+(1-\alpha)L\cdot p_{\mathrm{err}}`$. On MATH-500 with
$`L\approx 4{,}139`$ and $`\alpha\approx 0.44`$ (Qwen3-14B, 30K budget;
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
even $`p_{\mathrm{err}}=10^{-4}`$ gives a linear lower bound of
$`{\sim}23\%`$ relative accuracy improvement ($`{\sim}26\%`$ from the
exact exponential form). ∎

This provides a *lower* bound on the accuracy benefit of compression: in
practice, reasoning errors are positively correlated (one incorrect step
causes subsequent steps to build on a false premise), amplifying the
gain beyond the independence assumption. Consistent with this,
compression improves accuracy where the base model has room to improve,
most clearly on DeepSeek-R1-Distill-Llama-8B (MATH-500
$`71.3{\to}82.1`$;
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
while preserving accuracy on the already-strong Qwen3 models.

## 5 Experiments

### 5.1 Experimental Setting

##### Models and data.

We evaluate CRISP on Qwen3-8B and Qwen3-14B ([Yang et al.,
2025](#bib.bib35)) and DeepSeek-R1-Distill-Llama-8B ([Guo et al.,
2025](#bib.bib13)), training on $`{\sim}`$13,600 competition-level math
problems from DAPO-Math-17k ([Yu et al., 2025](#bib.bib37)) *without
ground-truth answers*; only problem statements are used to generate
student rollouts. We train for 1 epoch with learning rate
$`1\times 10^{-6}`$, global batch size 32, periodic teacher update
(interval $`M{=}50`$; see ablation in
Section [5.3.3](#S5.SS3.SSS3 "5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
and $`8\times`$ H200 GPUs. Although nominally a full epoch, the
algorithm converges quickly at around $`{\sim}`$100 steps. Each prompt
generates a single student rollout (temperature 1.0) with a maximum
response length of 8,192 tokens. Because CRISP optimizes a per-token KL
objective rather than an outcome-based reward, there is no need to
generate complete responses as in RL methods; partial rollouts already
provide a useful training signal. This phenomenon is also observed by
[Chen et al. (2025b)](#bib.bib6) in the context of knowledge
distillation. Full training and infrastructure details are in
Appendix [E](#A5 "Appendix E Training and Implementation Details ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

##### Conciseness instructions.

The teacher is defined by the conciseness instruction $`c`$ prepended to
the problem. We use two instructions, shown below: the *uniform concise
prompt* (v1), which asks for directness on every problem, and the
*difficulty-aware concise prompt* (v2), which additionally instructs the
model not to over-compress hard problems. Both are prepended only to the
teacher; the student sees the plain problem.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuU1MxLlNTUzAuUHgyLnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIyMTIuMDIiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NzcuMzggMjEyLjAyIiB3aWR0aD0iNDc3LjM4Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDIxMi4wMikgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiMwMDhDMDA7IiBmaWxsPSIjMDA4QzAwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNS4wNCBMIDAgMjA2Ljk3IEMgMCAyMDkuNzYgMi4yNiAyMTIuMDIgNS4wNCAyMTIuMDIgTCA0NzIuMzMgMjEyLjAyIEMgNDc1LjEyIDIxMi4wMiA0NzcuMzggMjA5Ljc2IDQ3Ny4zOCAyMDYuOTcgTCA0NzcuMzggNS4wNCBDIDQ3Ny4zOCAyLjI2IDQ3NS4xMiAwIDQ3Mi4zMyAwIEwgNS4wNCAwIEMgMi4yNiAwIDAgMi4yNiAwIDUuMDQgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0U4RjVFOTsiIGZpbGw9IiNFOEY1RTkiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMS4xMSA1LjA0IEwgMS4xMSAxOTIuMzIgTCA0NzYuMjcgMTkyLjMyIEwgNDc2LjI3IDUuMDQgQyA0NzYuMjcgMi44NyA0NzQuNTEgMS4xMSA0NzIuMzMgMS4xMSBMIDUuMDQgMS4xMSBDIDIuODcgMS4xMSAxLjExIDIuODcgMS4xMSA1LjA0IFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgOS4yIDE5Ny4zNykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozMy4xN2VtOy0tbHR4LWZvLWhlaWdodDowLjY5ZW07LS1sdHgtZm8tZGVwdGg6MGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI5LjYxIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5LjYxKSIgd2lkdGg9IjQ1OC45OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuU1MxLlNTUzAuUHgyLnAyLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozMy4xN2VtOyI+CjxzcGFuIGlkPSJTNS5TUzEuU1NTMC5QeDIucDIucGljMS4xLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LlNTMS5TU1MwLlB4Mi5wMi5waWMxLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkZGRkY7Ij5Db25jaXNlbmVzcyBpbnN0cnVjdGlvbnMgPG1hdGggaWQ9IlM1LlNTMS5TU1MwLlB4Mi5wMi5waWMxLm0xIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9ImMiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkZGRkY7IiBtYXRoY29sb3I9IiNGRkZGRkYiPmM8L21pPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+YzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDkuMiAxMC4yMykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNS44NmVtOy0tbHR4LWZvLWhlaWdodDoxMi42OGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTc3LjgxIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNzUuMzkpIiB3aWR0aD0iNDk2LjIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IlM1LlNTMS5TU1MwLlB4Mi5wMi5waWMxLjIiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzUuODZlbTsiPgo8c3BhbiBpZD0iUzUuU1MxLlNTUzAuUHgyLnAyLnBpYzEuMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5TUzEuU1NTMC5QeDIucDIucGljMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij52MSAodW5pZm9ybSBjb25jaXNlIHByb21wdCk6PHNwYW4gaWQ9IlM1LlNTMS5TU1MwLlB4Mi5wMi5waWMxLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPiBTb2x2ZSB0aGUgZm9sbG93aW5nIG1hdGggcHJvYmxlbSBjb25jaXNlbHkgYW5kIGNvcnJlY3RseS4gQmUgZGlyZWN0OiBhdm9pZCB1bm5lY2Vzc2FyeSBlbGFib3JhdGlvbiwgcmVkdW5kYW50IHN0ZXBzLCBvciByZXN0YXRpbmcgdGhlIHByb2JsZW0uIEZvY3VzIG9ubHkgb24gdGhlIGtleSByZWFzb25pbmcgc3RlcHMgbmVlZGVkIHRvIHJlYWNoIHRoZSBhbnN3ZXIuCjxiciBjbGFzcz0ibHR4X2JyZWFrIiBzdHlsZT0iLS1sdHgtYnJlYWstc3BhY2U6NC4wcHQ7Ij4KPC9zcGFuPnYyIChkaWZmaWN1bHR5LWF3YXJlIGNvbmNpc2UgcHJvbXB0KTo8c3BhbiBpZD0iUzUuU1MxLlNTUzAuUHgyLnAyLnBpYzEuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSI+IFNvbHZlIHRoZSBmb2xsb3dpbmcgbWF0aCBwcm9ibGVtIGNvbmNpc2VseSBhbmQgY29ycmVjdGx5LiBQcmlvcml0aXplIGEgY29ycmVjdCBmaW5hbCBhbnN3ZXIgYWJvdmUgYnJldml0eS4gRm9yIHN0cmFpZ2h0Zm9yd2FyZCBwcm9ibGVtcywgYmUgZGlyZWN0OiBhdm9pZCB1bm5lY2Vzc2FyeSBlbGFib3JhdGlvbiwgcmVkdW5kYW50IHN0ZXBzLCBvciByZXN0YXRpbmcgdGhlIHByb2JsZW0uIEZvciBkaWZmaWN1bHQgb3IgbXVsdGktc3RlcCBwcm9ibGVtcywgZG8gTk9UIG92ZXItY29tcHJlc3M6IGtlZXAgZXZlcnkgcmVhc29uaW5nIHN0ZXAgbmVlZGVkIGZvciBjb3JyZWN0bmVzcywgaW5jbHVkaW5nIGNhc2UgYW5hbHlzaXMsIGVkZ2UgY2FzZXMsIGFuZCBhIGJyaWVmIGNoZWNrIG9mIHRoZSBmaW5hbCBhbnN3ZXIuIE5ldmVyIG9taXQgYSBzdGVwIHRoYXQgdGhlIGFuc3dlciBkZXBlbmRzIG9uLjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

##### Benchmarks.

CRISP is trained only on math prompts, so we separate evaluation into
in-domain and out-of-domain tasks. *In-domain*, we use three
mathematical reasoning benchmarks spanning a wide difficulty range:
MATH-500 ([Hendrycks et al., 2021](#bib.bib15)) (500 problems), AIME
2024 (30 problems), and AIME 2025 (30 problems). *Out-of-domain*, we use
two general-capability benchmarks that the model never trains on:
GPQA-Diamond ([Rein et al., 2023](#bib.bib25)) (graduate-level,
Google-proof science questions) and MMLU ([Hendrycks et al.,
2020](#bib.bib14)) (massive multitask knowledge). The out-of-domain
benchmarks test whether compressing math reasoning degrades general
knowledge and scientific reasoning. We define a *token budget* as the
maximum response length allowed during inference, a practical lever for
controlling serving cost, and report results under a 30,000-token
budget, which effectively eliminates truncation and enables a fair
accuracy comparison.

##### Scoring.

Unless otherwise noted, all math accuracies reported in this paper
(MATH-500, AIME 2024/2025) use the *dual-path* scorer: a response is
correct if *either* an extracted “Answer: $`X`$” line or a literal
“$`\backslash`$boxed{$`\cdot`$}” expression math-verifies against the
gold answer, building on the veRL math grading utility ([Sheng et al.,
2025](#bib.bib27)).¹¹ 1 Dual-path scoring:
[https://github.com/HJSang/CRISP_Reasoning_Compression/blob/main/workspace/src/rewards/dual_path_math_verify.py](https://github.com/HJSang/CRISP_Reasoning_Compression/blob/main/workspace/src/rewards/dual_path_math_verify.py),
built on veRL’s math_dapo grader:
[https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py).
This avoids undercounting models whose native answer format differs from
the boxed convention
(Appendix [D](#A4 "Appendix D Answer-Format Breakdown ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
GPQA-Diamond and MMLU are scored by exact letter match on the
multiple-choice answer.

### 5.2 Main Results

|  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
|  | MATH-500 |  | AIME 2024 |  | AIME 2025 |  | GPQA-D |  | MMLU |  |
| Method | Acc | Red. | Acc | Red. | Acc | Red. | Acc | Red. | Acc | Red. |
| Qwen3-8B |  |  |  |  |  |  |  |  |  |  |
| Base Model | 95.7 | — | 76.2 | — | 70.4 | — | 61.5 | — | 81.9 | — |
| CRISP (v2) | 95.7 | 31.6% | 75.0 | 17.1% | 65.8 | 17.5% | 58.3 | 17.2% | 81.2 | 22.4% |
| CRISP (v1) | 95.7 | 56.9% | 72.9 | 32.9% | 58.8 | 28.4% | 58.5 | 36.2% | 80.9 | 44.7% |
| Qwen3-14B |  |  |  |  |  |  |  |  |  |  |
| Base Model | 93.0 | — | 75.0 | — | 69.2 | — | 62.2 | — | 85.1 | — |
| CRISP (v2) | 95.2 | 34.7% | 75.0 | 19.7% | 67.1 | 16.8% | 62.0 | 20.7% | 83.9 | 22.4% |
| CRISP (v1) | 96.3 | 56.3% | 73.8 | 37.5% | 62.9 | 32.1% | 61.9 | 39.7% | 84.2 | 43.1% |
| DeepSeek-R1-Distill-Llama-8B |  |  |  |  |  |  |  |  |  |  |
| Base Model | 71.3 | — | 33.3 | — | 25.0 | — | 47.0 | — | 71.5 | — |
| CRISP (v2) | 79.8 | 23.2% | 42.1 | $`-`$2.5% | 26.2 | 0.1% | 46.7 | 7.0% | 71.4 | 11.4% |
| CRISP (v1) | 82.1 | 31.6% | 39.2 | 6.3% | 27.1 | 7.1% | 48.3 | 10.2% | 71.7 | 17.6% |

Table 2: CRISP compresses reasoning while preserving accuracy (token
budget = 30K). For each benchmark we report accuracy (Acc, mean@8, %)
and token reduction relative to the base model (Red., %; “—” marks the
base reference, negative values indicate the response grew longer).
CRISP is trained with a periodic teacher update ($`M{=}50`$); v1 and v2
denote the uniform and difficulty-aware conciseness instructions
(Section [5.3.2](#S5.SS3.SSS2 "5.3.2 Which Conciseness Instruction Compresses Best? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
The first three benchmarks are in-domain; GPQA-Diamond and MMLU are
out-of-domain and verify that general capabilities are preserved.
Per-row response lengths and the inference-only “concise prompt”
baselines are in
Appendix [C](#A3 "Appendix C Full Main Results ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"),
Table [5](#A3.T5 "Table 5 ‣ Appendix C Full Main Results ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
presents our main results under the 30,000-token budget, which
eliminates response truncation for a fair accuracy comparison.²² 2 MMLU
is evaluated using the Language Model Evaluation Harness ([Gao et al.,
2021](#bib.bib11)):
[https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
The first three benchmarks (MATH-500, AIME 2024/2025) are in-domain, and
the last two (GPQA-Diamond, MMLU) are out-of-domain: CRISP trains only
on math, so the out-of-domain columns measure whether compression harms
general capabilities. We highlight four findings.

##### CRISP compresses reasoning while preserving accuracy.

Across all three models, CRISP substantially shortens reasoning traces
while keeping accuracy close to the base model, and it improves accuracy
where the base model has room. On the in-domain math benchmarks it
reduces length by up to 57% (MATH-500) with accuracy held within about
one point on MATH-500 and the AIME benchmarks. The accuracy effect
depends on base-model headroom: the already-strong Qwen3-8B and
Qwen3-14B (both above 93% on MATH-500) are preserved, with Qwen3-14B
gaining up to 3.3 points, while DeepSeek-R1-Distill-Llama-8B starts
lower and *improves* on all five benchmarks, by up to 10.8 points on
MATH-500. The concise teacher removes redundant tokens that would
otherwise accumulate errors, so the accuracy benefit is largest where
the base model leaves the most room.

##### Compression naturally adapts to problem difficulty.

Prior RL methods estimate difficulty from rollout pass rates ([Wan
et al., 2026](#bib.bib29); [Chen et al., 2025](#bib.bib5)) or train
separate difficulty classifiers; CRISP requires none of this, as
difficulty adaptation emerges from the KL objective. Using benchmarks as
a difficulty proxy, CRISP compresses the easier MATH-500 by 32% to 57%
but the harder AIME benchmarks by only 5% to 38%. The model compresses
easy problems more than hard ones with no explicit difficulty estimate,
which follows from the per-token KL signal (see
§[3.3](#S3.SS3.SSS0.Px1 "Difficulty-adaptive compression. ‣ 3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

##### Different conciseness instructions are readily adopted.

CRISP works with different concise behaviors rather than a single
hand-tuned prompt. The uniform instruction (v1) compresses more
aggressively (up to 57% on MATH-500) but drops more accuracy on the
hardest benchmark, AIME 2025 (e.g. $`-11.6`$ points on Qwen3-8B). The
difficulty-aware instruction (v2) compresses less (around 32% to 35% on
MATH-500) but protects hard problems, cutting the AIME 2025 drop to
$`-4.6`$ points on Qwen3-8B and $`-2.1`$ on Qwen3-14B. Both give strong
compression with preserved accuracy, so the wording of the instruction
trades compression against accuracy on hard problems without changing
the overall behavior.

##### General knowledge and capabilities are preserved.

Although CRISP trains only on math, out-of-domain accuracy is preserved:
on GPQA-Diamond and MMLU, every model stays within about one point of
its base accuracy, and DeepSeek even improves slightly. Compressing math
reasoning therefore does not degrade general knowledge or scientific
reasoning.

##### Qualitative examples.

Figure [3](#S5.F3 "Figure 3 ‣ Qualitative examples. ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
shows two examples of CRISP compressing reasoning while reaching the
same correct answer as the base model, and honoring the two conciseness
instructions. In each case the base model, the difficulty-aware teacher
(v2), and the uniform teacher (v1) all answer correctly, but v2 is much
shorter than the base and v1 is shorter still, illustrating that v2
keeps slightly more reasoning than v1 (more examples in
Appendix [J](#A10 "Appendix J Answer-Format Consolidation: Qualitative Examples ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

Problem 1 (MATH-500): *Twelve $`1\times 1`$ squares form a $`3\times 4`$
rectangle; two shaded right triangles are drawn. What is the total
shaded area?*  (Correct answer: 10)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjI0NS42NiIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyNDUuNjYiIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjQ1LjY2KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0JGQkZCRjsiIGZpbGw9IiNCRkJGQkYiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMzkuNzUgQyAwIDI0My4wMiAyLjY0IDI0NS42NiA1LjkxIDI0NS42NiBMIDQ3MS40NyAyNDUuNjYgQyA0NzQuNzMgMjQ1LjY2IDQ3Ny4zOCAyNDMuMDIgNDc3LjM4IDIzOS43NSBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjlGOUY5OyIgZmlsbD0iI0Y5RjlGOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE5MS40OSBMIDQ3NS40MSAxOTEuNDkgTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxOTkuODIpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6Mi44OWVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNDIuMzYiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDM5Ljk0KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWMxLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPlF3ZW4zLThCIGJhc2UgKDgsNDIyIHRva2Vucyk8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTAuMDYgOC42NykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDo0MS40NmVtOy0tbHR4LWZvLWhlaWdodDoxMi43M2VtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTc2LjEyIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNzYuMTIpIiB3aWR0aD0iNTczLjY5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzEuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzEucDIiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJTNS5GMy5waWMxLnAyLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzEucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljMS5wMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzEucDIuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMS5wMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzgwODA4MDsiPuKApjxzcGFuIGlkPSJTNS5GMy5waWMxLnAyLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cmUtZGVyaXZlcyB0aGUgPG1hdGggaWQ9IlM1LkYzLnBpYzEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iM1x0aW1lcyA0IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4zPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjIyMmVtIj7DlzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4zXHRpbWVzIDQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBkaW1lbnNpb25zIGZyb20gdGhlIEFzeW1wdG90ZSBjb2RlLCB0aGVuIHJlLXZpc3VhbGl6ZXMgZWFjaCB0cmlhbmdsZSBhbmQgcmVwZWF0ZWRseSBzZWNvbmQtZ3Vlc3NlcyB3aGljaCBzZWdtZW50IGlzIHRoZSBiYXNl4oCmPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWMxLnAyLjMiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzEucDIuMy4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij5bc2V2ZXJhbCByZXN0YXRlbWVudHMgb2YgdGhlIHNhbWUgdHdvIHRyaWFuZ2xlcyBiZWZvcmUgY29tbWl0dGluZ108c3BhbiBpZD0iUzUuRjMucGljMS5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0Ij48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzEucDIuNCIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMS5wMi40LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljMS5wMi40LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljMS5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljMS5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzEucDMuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMS5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRyaWFuZ2xlIDE6IDxtYXRoIGlkPSJTNS5GMy5waWMxLm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilx0ZnJhY3sxfXsyfVxjZG90IDJcY2RvdCA0PTQiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48L21mcmFjPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMC4yMjJlbSIgbWF0aGNvbG9yPSIjMDAwMDAwIiByc3BhY2U9IjAuMjIyZW0iPuKLhTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjIyMmVtIj7ii4U8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NDwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cdGZyYWN7MX17Mn1cY2RvdCAyXGNkb3QgND00PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uIFRyaWFuZ2xlIDI6IDxtYXRoIGlkPSJTNS5GMy5waWMxLm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilx0ZnJhY3sxfXsyfVxjZG90IDNcY2RvdCA0PTYiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48L21mcmFjPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMC4yMjJlbSIgbWF0aGNvbG9yPSIjMDAwMDAwIiByc3BhY2U9IjAuMjIyZW0iPuKLhTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4zPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjIyMmVtIj7ii4U8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NDwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj42PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cdGZyYWN7MX17Mn1cY2RvdCAzXGNkb3QgND02PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uIFRvdGFsIDxtYXRoIGlkPSJTNS5GMy5waWMxLm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ij0xMCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtcGhhbnRvbT48L21waGFudG9tPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xMDwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+PTEwPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzEucDMuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMS5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkFuc3dlcjogPG1hdGggaWQ9IlM1LkYzLnBpYzEubTUiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJvbGRzeW1ib2x7MTB9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyIgbWF0aGNvbG9yPSIjMDA4MDAwIj7wnZ+P8J2fjjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYm9sZHN5bWJvbHsxMH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiA8c3BhbiBpZD0iUzUuRjMucGljMS5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDowLjBwdDsiPuKAhDxzcGFuIGlkPSJTNS5GMy5waWMxLnAzLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7Ij7inJM8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljMiIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjIyOC45NyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyMjguOTciIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjI4Ljk3KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzAwOEMwMDsiIGZpbGw9IiMwMDhDMDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMjMuMDcgQyAwIDIyNi4zMyAyLjY0IDIyOC45NyA1LjkxIDIyOC45NyBMIDQ3MS40NyAyMjguOTcgQyA0NzQuNzMgMjI4Ljk3IDQ3Ny4zOCAyMjYuMzMgNDc3LjM4IDIyMy4wNyBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRThGNUU5OyIgZmlsbD0iI0U4RjVFOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE1OC4yOCBMIDQ3NS40MSAxNTguMjggTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxNjYuNjEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6NC4wOGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTguODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDU2LjQ2KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzIuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWMyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNSSVNQwqB2MiAoMiw0NjYgdG9rLCA3MSUpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTAuMzNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE0Mi45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTQyLjkxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNS5GMy5waWMyLjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJTNS5GMy5waWMyLnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljMi5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWMyLnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzIucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWMyLnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzIucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGUgZ3JpZCBpcyA8bWF0aCBpZD0iUzUuRjMucGljMi5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIzXHRpbWVzIDQiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjM8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMC4yMjJlbSIgbWF0aGNvbG9yPSIjMDAwMDAwIiByc3BhY2U9IjAuMjIyZW0iPsOXPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjNcdGltZXMgNDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LiBUcmlhbmdsZSAxIGhhcyBsZWdzIDxtYXRoIGlkPSJTNS5GMy5waWMyLm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjIiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+MjwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IGFuZCA8bWF0aCBpZD0iUzUuRjMucGljMi5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI0IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjogYXJlYSA8bWF0aCBpZD0iUzUuRjMucGljMi5tNCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcdGZyYWN7MX17Mn1cY2RvdCAyXGNkb3QgND00IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tZnJhYz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjIyMmVtIj7ii4U8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwLjIyMmVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMC4yMjJlbSI+4ouFPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NDwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XHRmcmFjezF9ezJ9XGNkb3QgMlxjZG90IDQ9NDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LiBUcmlhbmdsZSAyIGhhcyBsZWdzIDxtYXRoIGlkPSJTNS5GMy5waWMyLm01IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjMiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjM8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+MzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IGFuZCA8bWF0aCBpZD0iUzUuRjMucGljMi5tNiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI0IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjogYXJlYSA8bWF0aCBpZD0iUzUuRjMucGljMi5tNyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcdGZyYWN7MX17Mn1cY2RvdCAzXGNkb3QgND02IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tZnJhYz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgcnNwYWNlPSIwLjIyMmVtIj7ii4U8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MzwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwLjIyMmVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIHJzcGFjZT0iMC4yMjJlbSI+4ouFPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XHRmcmFjezF9ezJ9XGNkb3QgM1xjZG90IDQ9NjwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LiBUaGUgdHdvIHRyaWFuZ2xlcyBkbyBub3Qgb3ZlcmxhcCAobGVmdCB2cy4gcmlnaHQpLCBzbyB0aGUgYXJlYXMgYWRkLjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWMyLnAyLjMiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzIucDIuMy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDsvdGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzIucDIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzIucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPjxzcGFuIGNsYXNzPSJsdHhfcnVsZSIgc3R5bGU9IndpZHRoOjMzMC41cHQ7aGVpZ2h0OjAuM3B0Oy0tbHR4LWJnLWNvbG9yOmJsYWNrO2Rpc3BsYXk6aW5saW5lLWJsb2NrOyI+wqA8L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzIucDMiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJTNS5GMy5waWMyLnAzLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzIucDMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Ub3RhbCBzaGFkZWQgYXJlYSA8bWF0aCBpZD0iUzUuRjMucGljMi5tOCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI9NCs2PTEwIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1waGFudG9tPjwvbXBoYW50b20+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj42PC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjEwPC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij49NCs2PTEwPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzIucDMuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMi5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkFuc3dlcjogPG1hdGggaWQ9IlM1LkYzLnBpYzIubTkiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJvbGRzeW1ib2x7MTB9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyIgbWF0aGNvbG9yPSIjMDA4MDAwIj7wnZ+P8J2fjjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYm9sZHN5bWJvbHsxMH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiA8c3BhbiBpZD0iUzUuRjMucGljMi5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDowLjBwdDsiPuKAhDxzcGFuIGlkPSJTNS5GMy5waWMyLnAzLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7Ij7inJM8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljMyIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjIyOC45NyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyMjguOTciIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjI4Ljk3KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6Izk5NEQwMDsiIGZpbGw9IiM5OTREMDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMjMuMDcgQyAwIDIyNi4zMyAyLjY0IDIyOC45NyA1LjkxIDIyOC45NyBMIDQ3MS40NyAyMjguOTcgQyA0NzQuNzMgMjI4Ljk3IDQ3Ny4zOCAyMjYuMzMgNDc3LjM4IDIyMy4wNyBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRkZGQkY3OyIgZmlsbD0iI0ZGRkJGNyIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE1OC4yOCBMIDQ3NS40MSAxNTguMjggTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxNjYuNjEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6NC4wOGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTguODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDU2LjQ2KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljMy4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzMuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWMzLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNSSVNQwqB2MSAoMiwwMTggdG9rLCA3NiUpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTAuMzNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE0Mi45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTQyLjkxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNS5GMy5waWMzLjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJTNS5GMy5waWMzLnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljMy5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWMzLnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzMucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWMzLnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzMucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Ub3RhbCByZWN0YW5nbGUgYXJlYSBpcyA8bWF0aCBpZD0iUzUuRjMucGljMy5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIxMiIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTI8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+MTI8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi4gVHdvIHJpZ2h0IHRyaWFuZ2xlczogYmFzZSA8bWF0aCBpZD0iUzUuRjMucGljMy5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIyIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjI8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgaGVpZ2h0IDxtYXRoIGlkPSJTNS5GMy5waWMzLm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjQiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+NDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IGdpdmVzIDxtYXRoIGlkPSJTNS5GMy5waWMzLm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjQiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+NDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+OyBiYXNlIDxtYXRoIGlkPSJTNS5GMy5waWMzLm01IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjMiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjM8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+MzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LCBoZWlnaHQgPG1hdGggaWQ9IlM1LkYzLnBpYzMubTYiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iNCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NDwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij40PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gZ2l2ZXMgPG1hdGggaWQ9IlM1LkYzLnBpYzMubTciIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iNiIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij42PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzMucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMy5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljMy5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljMy5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljMy5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzMucDMuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljMy5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkFyZWFzOiA8bWF0aCBpZD0iUzUuRjMucGljMy5tOCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI0IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBhbmQgPG1hdGggaWQ9IlM1LkYzLnBpYzMubTkiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iNiIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij42PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4sIHN1bSA8bWF0aCBpZD0iUzUuRjMucGljMy5tMTAiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iPTEwIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1waGFudG9tPjwvbXBoYW50b20+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjEwPC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij49MTA8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljMy5wMy4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWMzLnAzLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QW5zd2VyOiA8bWF0aCBpZD0iUzUuRjMucGljMy5tMTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJvbGRzeW1ib2x7MTB9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyIgbWF0aGNvbG9yPSIjMDA4MDAwIj7wnZ+P8J2fjjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYm9sZHN5bWJvbHsxMH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiA8c3BhbiBpZD0iUzUuRjMucGljMy5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDowLjBwdDsiPuKAhDxzcGFuIGlkPSJTNS5GMy5waWMzLnAzLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7Ij7inJM8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

Problem 2 (AIME 2024): *On an $`8\times 8`$ grid, count length-16
lattice paths from the lower-left to the upper-right corner that change
direction exactly four times.*  (Correct answer: 294)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljNCIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjIyOS4wNSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyMjkuMDUiIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjI5LjA1KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0JGQkZCRjsiIGZpbGw9IiNCRkJGQkYiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMjMuMTUgQyAwIDIyNi40MSAyLjY0IDIyOS4wNSA1LjkxIDIyOS4wNSBMIDQ3MS40NyAyMjkuMDUgQyA0NzQuNzMgMjI5LjA1IDQ3Ny4zOCAyMjYuNDEgNDc3LjM4IDIyMy4xNSBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjlGOUY5OyIgZmlsbD0iI0Y5RjlGOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE3NC44OSBMIDQ3NS40MSAxNzQuODkgTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxODMuMjEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6Mi44OWVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNDIuMzYiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDM5Ljk0KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljNC4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzQuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM0LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPlF3ZW4zLThCIGJhc2UgKDYsMDQxIHRva2Vucyk8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTAuMDYgOC42NykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDo0MS40NmVtOy0tbHR4LWZvLWhlaWdodDoxMS41M2VtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTU5LjUxIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNTkuNTEpIiB3aWR0aD0iNTczLjY5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzQuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzQucDIiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJTNS5GMy5waWM0LnAyLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzQucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljNC5wMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzQucDIuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNC5wMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzgwODA4MDsiPuKApjxzcGFuIGlkPSJTNS5GMy5waWM0LnAyLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+cmUtZXN0YWJsaXNoZXMgdGhhdCBhIHBhdGggaXMgPG1hdGggaWQ9IlM1LkYzLnBpYzQubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij44PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gUiBhbmQgPG1hdGggaWQ9IlM1LkYzLnBpYzQubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij44PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gVSBtb3ZlcywgdGhlbiB3b3JrcyB0aHJvdWdoIHRoZSBydW4gc3RydWN0dXJlIHNsb3dseSwgcmUtY2hlY2tpbmcgdGhlIOKAnGZvdXIgZGlyZWN0aW9uIGNoYW5nZXPigJ0gY29uZGl0aW9uIHNldmVyYWwgdGltZXPigKY8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzQucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNC5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljNC5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNC5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNC5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzQucDMuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNC5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlN0YXJ0IFU6IDxtYXRoIGlkPSJTNS5GMy5waWM0Lm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxiaW5vbXs3fXsyfVxiaW5vbXs3fXsxfT0xNDciIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPig8L21vPjxtZnJhYyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxpbmV0aGlja25lc3M9IjBwdCIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjc8L21uPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tZnJhYz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPik8L21vPjwvbXJvdz48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KDwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbGluZXRoaWNrbmVzcz0iMHB0IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xPC9tbj48L21mcmFjPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KTwvbW8+PC9tcm93PjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTQ3PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYmlub217N317Mn1cYmlub217N317MX09MTQ3PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD47IHN0YXJ0IFI6IDxtYXRoIGlkPSJTNS5GMy5waWM0Lm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjE0NyIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTQ3PC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjE0NzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LiBUb3RhbCA8bWF0aCBpZD0iUzUuRjMucGljNC5tNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIyOTQiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI5NDwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4yOTQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNC5wMy4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM0LnAzLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QW5zd2VyOiA8bWF0aCBpZD0iUzUuRjMucGljNC5tNiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcYm9sZHN5bWJvbHsyOTR9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyIgbWF0aGNvbG9yPSIjMDA4MDAwIj7wnZ+Q8J2fl/Cdn5I8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGJvbGRzeW1ib2x7Mjk0fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IDxzcGFuIGlkPSJTNS5GMy5waWM0LnAzLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2lubGluZS1ibG9jayIgc3R5bGU9IndpZHRoOjAuMHB0OyI+4oCEPHNwYW4gaWQ9IlM1LkYzLnBpYzQucDMuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiPuKckzwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljNSIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjI0NS41OCIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyNDUuNTgiIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjQ1LjU4KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzAwOEMwMDsiIGZpbGw9IiMwMDhDMDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMzkuNjcgQyAwIDI0Mi45MyAyLjY0IDI0NS41OCA1LjkxIDI0NS41OCBMIDQ3MS40NyAyNDUuNTggQyA0NzQuNzMgMjQ1LjU4IDQ3Ny4zOCAyNDIuOTMgNDc3LjM4IDIzOS42NyBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRThGNUU5OyIgZmlsbD0iI0U4RjVFOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE3NC44OSBMIDQ3NS40MSAxNzQuODkgTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxODMuMjEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6NC4wOGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTguODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDU2LjQ2KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljNS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzUuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM1LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNSSVNQwqB2MiAoMywyMTMgdG9rLCA0NyUpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTEuNTNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE1OS41MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTU5LjUxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNS5GMy5waWM1LjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJTNS5GMy5waWM1LnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljNS5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM1LnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzUucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM1LnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzUucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5BIHBhdGggaXMgPG1hdGggaWQ9IlM1LkYzLnBpYzUubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij44PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gUiBhbmQgPG1hdGggaWQ9IlM1LkYzLnBpYzUubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij44PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gVS4gRm91ciBkaXJlY3Rpb24gY2hhbmdlcyBtZWFucyBmaXZlIG1vbm90b25lIHJ1bnMsIHNvIHRoZSBtb3ZlcyBzcGxpdCBhcyA8bWF0aCBpZD0iUzUuRjMucGljNS5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIzIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4zPC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjM8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBydW5zIG9mIG9uZSBsZXR0ZXIgYW5kIDxtYXRoIGlkPSJTNS5GMy5waWM1Lm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjIiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+MjwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IG9mIHRoZSBvdGhlci4gRGlzdHJpYnV0ZSA8bWF0aCBpZD0iUzUuRjMucGljNS5tNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI4IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjg8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBV4oCZcyBpbnRvIDxtYXRoIGlkPSJTNS5GMy5waWM1Lm02IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjMiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjM8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+MzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IHJ1bnM6IDxtYXRoIGlkPSJTNS5GMy5waWM1Lm03IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxiaW5vbXs3fXsyfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KDwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbGluZXRoaWNrbmVzcz0iMHB0IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48L21mcmFjPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KTwvbW8+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGJpbm9tezd9ezJ9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD47IDxtYXRoIGlkPSJTNS5GMy5waWM1Lm04IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IjgiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+ODwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IFLigJlzIGludG8gPG1hdGggaWQ9IlM1LkYzLnBpYzUubTkiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iMiIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4yPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gcnVuczogPG1hdGggaWQ9IlM1LkYzLnBpYzUubTEwIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxiaW5vbXs3fXsxfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KDwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbGluZXRoaWNrbmVzcz0iMHB0IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xPC9tbj48L21mcmFjPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KTwvbW8+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGJpbm9tezd9ezF9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uIFR3byBjYXNlcyBieSBzdGFydGluZyBsZXR0ZXIuPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzUucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNS5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljNS5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNS5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNS5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzUucDMuMSIgY2xhc3M9Imx0eF9wIj48bWF0aCBpZD0iUzUuRjMucGljNS5tMTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iMlx0aW1lc1xiaW5vbXs3fXsyfVxiaW5vbXs3fXsxfT0yXGNkb3QgMjFcY2RvdCA3PTI5NCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93Pjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjI8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxzcGFjZT0iMC4yMjJlbSIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSIgcnNwYWNlPSIwLjIyMmVtIj7DlzwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4oPC9tbz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsaW5ldGhpY2tuZXNzPSIwcHQiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+NzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+MjwvbW4+PC9tZnJhYz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPik8L21vPjwvbXJvdz48L21yb3c+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPig8L21vPjxtZnJhYyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGxpbmV0aGlja25lc3M9IjBwdCIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj43PC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4xPC9tbj48L21mcmFjPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KTwvbW8+PC9tcm93PjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj49PC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4yPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iIHJzcGFjZT0iMC4yMjJlbSI+4ouFPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4yMTwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwLjIyMmVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIiByc3BhY2U9IjAuMjIyZW0iPuKLhTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+NzwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjI5NDwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+Mlx0aW1lc1xiaW5vbXs3fXsyfVxiaW5vbXs3fXsxfT0yXGNkb3QgMjFcY2RvdCA3PTI5NDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM1LkYzLnBpYzUucDMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzUucDMuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNS5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkFuc3dlcjogPG1hdGggaWQ9IlM1LkYzLnBpYzUubTEyIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilxib2xkc3ltYm9sezI5NH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7IiBtYXRoY29sb3I9IiMwMDgwMDAiPvCdn5DwnZ+X8J2fkjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYm9sZHN5bWJvbHsyOTR9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gPHNwYW4gaWQ9IlM1LkYzLnBpYzUucDMuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MC4wcHQ7Ij7igIQ8c3BhbiBpZD0iUzUuRjMucGljNS5wMy4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyI+4pyTPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljNiIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjIyOC45NyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyMjguOTciIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjI4Ljk3KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6Izk5NEQwMDsiIGZpbGw9IiM5OTREMDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMjMuMDcgQyAwIDIyNi4zMyAyLjY0IDIyOC45NyA1LjkxIDIyOC45NyBMIDQ3MS40NyAyMjguOTcgQyA0NzQuNzMgMjI4Ljk3IDQ3Ny4zOCAyMjYuMzMgNDc3LjM4IDIyMy4wNyBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRkZGQkY3OyIgZmlsbD0iI0ZGRkJGNyIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE1OC4yOCBMIDQ3NS40MSAxNTguMjggTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxNjYuNjEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6NC4wOGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTguODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDU2LjQ2KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljNi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzYuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM2LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNSSVNQwqB2MSAoMSw3NDMgdG9rLCA3MSUpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTAuMzNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE0Mi45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTQyLjkxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNS5GMy5waWM2LjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJTNS5GMy5waWM2LnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljNi5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM2LnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzYucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM2LnAyLjIiIGNsYXNzPSJsdHhfcCI+PG1hdGggaWQ9IlM1LkYzLnBpYzYubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjg8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+ODwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM1LkYzLnBpYzYucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gUiBhbmQgPG1hdGggaWQ9IlM1LkYzLnBpYzYubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij44PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gVSBtb3ZlczsgZm91ciBkaXJlY3Rpb24gY2hhbmdlcyBnaXZlIGZpdmUgcnVucywgcGFydGl0aW9uZWQgPG1hdGggaWQ9IlM1LkYzLnBpYzYubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iM3srfTIiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjM8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4zeyt9MjwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LiBTZWdtZW50IGNvdW50czogPG1hdGggaWQ9IlM1LkYzLnBpYzYubTQiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJpbm9tezd9ezJ9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4oPC9tbz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsaW5ldGhpY2tuZXNzPSIwcHQiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj43PC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjwvbWZyYWM+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4pPC9tbz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYmlub217N317Mn08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBhbmQgPG1hdGggaWQ9IlM1LkYzLnBpYzYubTUiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJpbm9tezd9ezF9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4oPC9tbz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsaW5ldGhpY2tuZXNzPSIwcHQiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj43PC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjwvbWZyYWM+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4pPC9tbz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYmlub217N317MX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgdGltZXMgdHdvIHN0YXJ0aW5nIGxldHRlcnMuPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzYucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNi5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljNi5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNi5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNi5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzYucDMuMSIgY2xhc3M9Imx0eF9wIj48bWF0aCBpZD0iUzUuRjMucGljNi5tNiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSIyXHRpbWVzIDIxXHRpbWVzIDc9Mjk0IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+MjwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwLjIyMmVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIiByc3BhY2U9IjAuMjIyZW0iPsOXPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4yMTwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbHNwYWNlPSIwLjIyMmVtIiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIiByc3BhY2U9IjAuMjIyZW0iPsOXPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj43PC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+Mjk0PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij4yXHRpbWVzIDIxXHRpbWVzIDc9Mjk0PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iUzUuRjMucGljNi5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNi5wMy4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM2LnAzLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QW5zd2VyOiA8bWF0aCBpZD0iUzUuRjMucGljNi5tNyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcYm9sZHN5bWJvbHsyOTR9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyIgbWF0aGNvbG9yPSIjMDA4MDAwIj7wnZ+Q8J2fl/Cdn5I8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGJvbGRzeW1ib2x7Mjk0fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IDxzcGFuIGlkPSJTNS5GMy5waWM2LnAzLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2lubGluZS1ibG9jayIgc3R5bGU9IndpZHRoOjAuMHB0OyI+4oCEPHNwYW4gaWQ9IlM1LkYzLnBpYzYucDMuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiPuKckzwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

Problem 3 (AIME 2025, format change): *Find $`m+n`$ where
$`\tfrac{m}{n}`$ (in lowest terms) is the sum of all $`k`$ for which
$`|25+20i-z|=5`$ and $`|z-4-k|=|z-3i-k|`$ has exactly one solution
$`z`$.*  (Correct answer: 77)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljNyIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjI0NC44MyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyNDQuODMiIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjQ0LjgzKSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0JGQkZCRjsiIGZpbGw9IiNCRkJGQkYiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMzguOTIgQyAwIDI0Mi4xOCAyLjY0IDI0NC44MyA1LjkxIDI0NC44MyBMIDQ3MS40NyAyNDQuODMgQyA0NzQuNzMgMjQ0LjgzIDQ3Ny4zOCAyNDIuMTggNDc3LjM4IDIzOC45MiBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjlGOUY5OyIgZmlsbD0iI0Y5RjlGOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE5MC42NiBMIDQ3NS40MSAxOTAuNjYgTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxOTguOTkpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6Mi44OWVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNDIuMzYiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDM5Ljk0KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljNy4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzcuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM3LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPlF3ZW4zLThCIGJhc2UgKDQsMjg2IHRva2Vucyk8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTAuMDYgMjcuNykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDo0MS40NmVtOy0tbHR4LWZvLWhlaWdodDoxMS4yOWVtOy0tbHR4LWZvLWRlcHRoOjEuMzhlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTc1LjI5IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNTYuMjYpIiB3aWR0aD0iNTczLjY5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzcuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzcucDIiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJTNS5GMy5waWM3LnAyLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzcucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljNy5wMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzcucDIuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNy5wMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzgwODA4MDsiPuKApjxzcGFuIGlkPSJTNS5GMy5waWM3LnAyLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+dGhlIGNpcmNsZSBpcyB0YW5nZW50IHRvIHRoZSBwZXJwZW5kaWN1bGFyIGJpc2VjdG9yIGxpbmU7IHNvbHZpbmcgdGhlIHRhbmdlbmN5IGNvbmRpdGlvbiBnaXZlcyA8bWF0aCBpZD0iUzUuRjMucGljNy5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJrPVx0ZnJhY3syM317OH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPms8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIzPC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjwvbWZyYWM+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+az1cdGZyYWN7MjN9ezh9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gYW5kIDxtYXRoIGlkPSJTNS5GMy5waWM3Lm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ims9XHRmcmFjezEyM317OH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPms8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjEyMzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48L21mcmFjPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPms9XHRmcmFjezEyM317OH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgc28gdGhlIHN1bSBpcyA8bWF0aCBpZD0iUzUuRjMucGljNy5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcdGZyYWN7NzN9ezR9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjczPC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjQ8L21uPjwvbWZyYWM+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cdGZyYWN7NzN9ezR9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD7igKY8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzcucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljNy5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljNy5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNy5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljNy5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzcucDMuMSIgY2xhc3M9Imx0eF9wIj48bWF0aCBpZD0iUzUuRjMucGljNy5tNCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJtPTczLFwsbj00IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+NzM8L21uPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIiByc3BhY2U9IjAuMzM3ZW0iPiw8L21vPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjQ8L21uPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5tPTczLFwsbj00PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iUzUuRjMucGljNy5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiwgc28gPG1hdGggaWQ9IlM1LkYzLnBpYzcubTUiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ibStuPVxib3hlZHs3N30iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm08L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5uPC9taT48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bWVuY2xvc2Ugbm90YXRpb249ImJveCI+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj43NzwvbW4+PC9tZW5jbG9zZT48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5tK249XGJveGVkezc3fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+Ljwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM3LnAzLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzcucDMuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNGRjAwMDA7Ij5ObyDigJxBbnN3ZXI64oCdIGxpbmU7IHJlc3VsdCBvbmx5IGluc2lkZSA8c3BhbiBpZD0iUzUuRjMucGljNy5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiPlxib3hlZHt9PC9zcGFuPi48c3BhbiBpZD0iUzUuRjMucGljNy5wMy4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiA8c3BhbiBpZD0iUzUuRjMucGljNy5wMy4yLjEuMi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2lubGluZS1ibG9jayIgc3R5bGU9IndpZHRoOjAuMHB0OyI+4oCEPHNwYW4gaWQ9IlM1LkYzLnBpYzcucDMuMi4xLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+4pyXPC9zcGFuPuKAhDxzcGFuIGlkPSJTNS5GMy5waWM3LnAzLjIuMS4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPihzdHJpY3QgZm9ybWF0KTwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljOCIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjIyOC45NyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyMjguOTciIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjI4Ljk3KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzAwOEMwMDsiIGZpbGw9IiMwMDhDMDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMjMuMDcgQyAwIDIyNi4zMyAyLjY0IDIyOC45NyA1LjkxIDIyOC45NyBMIDQ3MS40NyAyMjguOTcgQyA0NzQuNzMgMjI4Ljk3IDQ3Ny4zOCAyMjYuMzMgNDc3LjM4IDIyMy4wNyBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRThGNUU5OyIgZmlsbD0iI0U4RjVFOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE1OC4yOCBMIDQ3NS40MSAxNTguMjggTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxNjYuNjEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6NC4wOGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTguODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDU2LjQ2KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljOC4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzguMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM4LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNSSVNQwqB2MiAoMyw1NzYgdG9rLCAxNyUpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTAuMzNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE0Mi45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTQyLjkxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNS5GMy5waWM4LjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJTNS5GMy5waWM4LnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljOC5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM4LnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzgucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM4LnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzgucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UYW5nZW5jeSBvZiB0aGUgY2lyY2xlIHRvIHRoZSBwZXJwZW5kaWN1bGFyIGJpc2VjdG9yIGdpdmVzIDxtYXRoIGlkPSJTNS5GMy5waWM4Lm0xIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ims9XHRmcmFjezIzfXs4fSxcdGZyYWN7MTIzfXs4fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+azwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjM8L21uPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PC9tZnJhYz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4sPC9tbz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTIzPC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjwvbWZyYWM+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+az1cdGZyYWN7MjN9ezh9LFx0ZnJhY3sxMjN9ezh9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4sIHN1bW1pbmcgdG8gPG1hdGggaWQ9IlM1LkYzLnBpYzgubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXHRmcmFjezczfXs0fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtZnJhYyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj43MzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48L21mcmFjPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XHRmcmFjezczfXs0fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+LCBzbyA8bWF0aCBpZD0iUzUuRjMucGljOC5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJtPTczLG49NCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjczPC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4sPC9tbz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj40PC9tbj48L21yb3c+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+bT03MyxuPTQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljOC5wMi4zIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM4LnAyLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7L3RoaW5rJmd0OzxzcGFuIGlkPSJTNS5GMy5waWM4LnAyLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM4LnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij48c3BhbiBjbGFzcz0ibHR4X3J1bGUiIHN0eWxlPSJ3aWR0aDozMzAuNXB0O2hlaWdodDowLjNwdDstLWx0eC1iZy1jb2xvcjpibGFjaztkaXNwbGF5OmlubGluZS1ibG9jazsiPsKgPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM4LnAzIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljOC5wMy4xIiBjbGFzcz0ibHR4X3AiPjxtYXRoIGlkPSJTNS5GMy5waWM4Lm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Im0rbj03Mys0PTc3IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+KzwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+bjwvbWk+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPj08L21vPjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjczPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4rPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj40PC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+Nzc8L21uPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPm0rbj03Mys0PTc3PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iUzUuRjMucGljOC5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljOC5wMy4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM4LnAzLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QW5zd2VyOiA8bWF0aCBpZD0iUzUuRjMucGljOC5tNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcYm9sZHN5bWJvbHs3N30iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7IiBtYXRoY29sb3I9IiMwMDgwMDAiPvCdn5XwnZ+VPC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxib2xkc3ltYm9sezc3fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IDxzcGFuIGlkPSJTNS5GMy5waWM4LnAzLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2lubGluZS1ibG9jayIgc3R5bGU9IndpZHRoOjAuMHB0OyI+4oCEPHNwYW4gaWQ9IlM1LkYzLnBpYzgucDMuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiPuKckzwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuRjMucGljOSIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjIxMi4zNyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAyMTIuMzciIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMjEyLjM3KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6Izk5NEQwMDsiIGZpbGw9IiM5OTREMDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCAyMDYuNDYgQyAwIDIwOS43MyAyLjY0IDIxMi4zNyA1LjkxIDIxMi4zNyBMIDQ3MS40NyAyMTIuMzcgQyA0NzQuNzMgMjEyLjM3IDQ3Ny4zOCAyMDkuNzMgNDc3LjM4IDIwNi40NiBMIDQ3Ny4zOCA1LjkxIEMgNDc3LjM4IDIuNjQgNDc0LjczIDAgNDcxLjQ3IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRkZGQkY3OyIgZmlsbD0iI0ZGRkJGNyIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDE0MS42OCBMIDQ3NS40MSAxNDEuNjggTCA0NzUuNDEgNS45MSBDIDQ3NS40MSAzLjczIDQ3My42NSAxLjk3IDQ3MS40NyAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxNTAuMDEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzYuNDhlbTstLWx0eC1mby1oZWlnaHQ6NC4wOGVtOy0tbHR4LWZvLWRlcHRoOjAuMThlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTguODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDU2LjQ2KSIgd2lkdGg9IjUwNC43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuRjMucGljOS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM2LjQ4ZW07Ij4KPHNwYW4gaWQ9IlM1LkYzLnBpYzkuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM5LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNSSVNQwqB2MSAoMiw1MDcgdG9rLCA0MiUpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6OS4xM2VtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTI2LjMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEyNi4zKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTNS5GMy5waWM5LjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJTNS5GMy5waWM5LnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iUzUuRjMucGljOS5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5GMy5waWM5LnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IlM1LkYzLnBpYzkucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJTNS5GMy5waWM5LnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IlM1LkYzLnBpYzkucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UaGUgdGFuZ2VuY3kgY29uZGl0aW9uIHlpZWxkcyA8bWF0aCBpZD0iUzUuRjMucGljOS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJrPVx0ZnJhY3syM317OH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPms8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIzPC9tbj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjwvbWZyYWM+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+az1cdGZyYWN7MjN9ezh9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gYW5kIDxtYXRoIGlkPSJTNS5GMy5waWM5Lm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ims9XHRmcmFjezEyM317OH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPms8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjEyMzwvbW4+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48L21mcmFjPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPms9XHRmcmFjezEyM317OH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjsgdGhlaXIgc3VtIGlzIDxtYXRoIGlkPSJTNS5GMy5waWM5Lm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilx0ZnJhY3s3M317NH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NzM8L21uPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NDwvbW4+PC9tZnJhYz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlx0ZnJhY3s3M317NH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgc28gPG1hdGggaWQ9IlM1LkYzLnBpYzkubTQiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ibT03MyxuPTQiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm08L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj43MzwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LDwvbW8+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5uPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NDwvbW4+PC9tcm93PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPm09NzMsbj00PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzkucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljOS5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iUzUuRjMucGljOS5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljOS5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iUzUuRjMucGljOS5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzkucDMuMSIgY2xhc3M9Imx0eF9wIj48bWF0aCBpZD0iUzUuRjMucGljOS5tNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJtK249NzciIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj5tPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4rPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj5uPC9taT48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+Nzc8L21uPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPm0rbj03NzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlM1LkYzLnBpYzkucDMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IlM1LkYzLnBpYzkucDMuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iUzUuRjMucGljOS5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkFuc3dlcjogPG1hdGggaWQ9IlM1LkYzLnBpYzkubTYiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJvbGRzeW1ib2x7Nzd9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyIgbWF0aGNvbG9yPSIjMDA4MDAwIj7wnZ+V8J2flTwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYm9sZHN5bWJvbHs3N308L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiA8c3BhbiBpZD0iUzUuRjMucGljOS5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDowLjBwdDsiPuKAhDxzcGFuIGlkPSJTNS5GMy5waWM5LnAzLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7Ij7inJM8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

Figure 3: CRISP compresses reasoning while preserving the correct
answer, and honors both conciseness instructions. On a MATH-500 problem
(top) and an AIME 2024 problem (middle), the base model, the
difficulty-aware teacher (v2), and the uniform teacher (v1) all reach
the correct answer, but the CRISP responses are far shorter; v1
compresses most while v2 keeps slightly more reasoning (e.g. explicitly
noting the triangles do not overlap, or spelling out the run-partition
argument), consistent with its instruction not to over-compress. Bottom:
an AIME 2025 format-change example, where the base model reports its
result only inside “$`\backslash`$boxed{$`\cdot`$}” with no “Answer:”
line (undercounted by a strict answer-format grader), while both CRISP
variants emit the requested “Answer:” line
(Appendix [D](#A4 "Appendix D Answer-Format Breakdown ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

In addition to math reasoning tasks, we also show our method is very
effective in agentic task like Deep Planning ([Zhang et al.,
2026](#bib.bib3)) in Appendix
[I](#A9 "Appendix I Deep Planning Agentic Task ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

### 5.3 Ablation Study

#### 5.3.1 How Much Compression Is Too Much? A Sweet Spot

Compression increases steadily over training, but accuracy does not stay
flat forever. To locate the useful operating point, we track a single
Qwen3-8B run with the difficulty-aware teacher (v2, $`M{=}50`$) across a
full epoch
(Figure [4](#S5.F4 "Figure 4 ‣ 5.3.1 How Much Compression Is Too Much? A Sweet Spot ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

Figure 4: A compression sweet spot (Qwen3-8B, v2 teacher, full epoch).
*Left:* accuracy over training. *Right:* MATH-500 compression factor
(base length / current length). Compression rises steadily with
training; accuracy is preserved up to about $`2\times`$ compression,
reached near step 220 (shaded band), and then declines, first on the
harder AIME 2024 and later on MATH-500. The step-99 checkpoint used
throughout the paper (dashed line) sits well inside this region at
$`{\sim}1.45\times`$ compression. Pushing compression beyond the sweet
spot trades accuracy for diminishing extra reduction.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuU1MzLlNTUzEucDIucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjExNS43MyIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDQ3Ny4zOCAxMTUuNzMiIHdpZHRoPSI0NzcuMzgiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMTE1LjczKSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzAwMDA4MDsiIGZpbGw9IiMwMDAwODAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCAzLjQ2IEwgMCAxMTIuMjcgQyAwIDExNC4xOCAxLjU1IDExNS43MyAzLjQ2IDExNS43MyBMIDQ3My45MiAxMTUuNzMgQyA0NzUuODMgMTE1LjczIDQ3Ny4zOCAxMTQuMTggNDc3LjM4IDExMi4yNyBMIDQ3Ny4zOCAzLjQ2IEMgNDc3LjM4IDEuNTUgNDc1LjgzIDAgNDczLjkyIDAgTCAzLjQ2IDAgQyAxLjU1IDAgMCAxLjU1IDAgMy40NiBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjJGMkZGOyIgZmlsbD0iI0YyRjJGRiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwLjY5IDMuNDYgTCAwLjY5IDExMi4yNyBDIDAuNjkgMTEzLjc5IDEuOTMgMTE1LjAzIDMuNDYgMTE1LjAzIEwgNDczLjkyIDExNS4wMyBDIDQ3NS40NSAxMTUuMDMgNDc2LjY4IDExMy43OSA0NzYuNjggMTEyLjI3IEwgNDc2LjY4IDMuNDYgQyA0NzYuNjggMS45MyA0NzUuNDUgMC42OSA0NzMuOTIgMC42OSBMIDMuNDYgMC42OSBDIDEuOTMgMC42OSAwLjY5IDEuOTMgMC42OSAzLjQ2IFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTEuNTUgMTEuNTUpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzIuODNlbTstLWx0eC1mby1oZWlnaHQ6Ni42OWVtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iOTIuNjMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDkyLjYzKSIgd2lkdGg9IjQ1NC4yNyI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iUzUuU1MzLlNTUzEucDIucGljMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjMyLjgzZW07Ij4KPHNwYW4gaWQ9IlM1LlNTMy5TU1MxLnAyLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJTNS5TUzMuU1NTMS5wMi5waWMxLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5GaW5kaW5nOiBjb21wcmVzc2luZyBhYm91dCA8bWF0aCBpZD0iUzUuU1MzLlNTUzEucDIucGljMS5tMSIgY2xhc3M9Imx0eF9tYXRoX3VucGFyc2VkIiBhbHR0ZXh0PSIyXHRpbWVzIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCI+w5c8L21vPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjJcdGltZXM8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBpcyB0aGUgc3dlZXQgc3BvdC48c3BhbiBpZD0iUzUuU1MzLlNTUzEucDIucGljMS4xLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfbWVkaXVtIj4gVXAgdG8gcm91Z2hseSA8bWF0aCBpZD0iUzUuU1MzLlNTUzEucDIucGljMS5tMiIgY2xhc3M9Imx0eF9tYXRoX3VucGFyc2VkIiBhbHR0ZXh0PSIyXHRpbWVzIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBsc3BhY2U9IjAuMjIyZW0iIG1hdGhjb2xvcj0iIzAwMDAwMCI+w5c8L21vPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjJcdGltZXM8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiByZWR1Y3Rpb24gaW4gcmVhc29uaW5nIGxlbmd0aCwgYWNjdXJhY3kgaXMgcHJlc2VydmVkIChNQVRILTUwMCBzdGF5cyBuZWFyIDk1JSwgQUlNRSAyMDI0IG5lYXIgaXRzIGJhc2UgbGV2ZWwpLiBGdXJ0aGVyIHRyYWluaW5nIGtlZXBzIHNocmlua2luZyByZXNwb25zZXMsIGJ1dCB0aGUgZXh0cmEgY29tcHJlc3Npb24gaXMgbW9kZXN0IGFuZCBhY2N1cmFjeSBiZWdpbnMgdG8gZHJvcCwgb24gQUlNRSAyMDI0IGZpcnN0IGFuZCBldmVudHVhbGx5IG9uIE1BVEgtNTAwLjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

As training proceeds, MATH-500 length falls from the base $`4{,}823`$
tokens toward $`{\sim}2{,}000`$ (a $`{\sim}2.4\times`$ reduction by the
end of the epoch), while MATH-500 accuracy holds near 95% until about
$`2\times`$ compression and then declines to $`{\sim}93\%`$. AIME 2024
is more sensitive: its accuracy is preserved ($`{\sim}70`$–$`74\%`$)
through the same region but erodes to $`{\sim}57\%`$ as compression is
pushed to the end of the epoch, with only a few additional points of
length reduction gained. The gain past $`2\times`$ is therefore small
and comes at a real accuracy cost, so we stop within the sweet spot: the
step-99 checkpoint used throughout this paper sits well inside this
region at $`{\sim}1.45\times`$ compression, on the conservative side of
the $`2\times`$ boundary, delivering strong compression with fully
preserved accuracy. This also motivates the periodic-refresh and
instruction ablations below, which control *how fast* a run moves along
this curve.

#### 5.3.2 Which Conciseness Instruction Compresses Best?

The teacher’s behavior is shaped entirely by its conciseness instruction
$`c`$, making the wording of $`c`$ a central design choice. We compare
five instructions
(Table [3](#S5.T3 "Table 3 ‣ 5.3.2 Which Conciseness Instruction Compresses Best? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
each expressing “be concise” through a different mechanism: v1 gives a
single uniform “be direct, avoid elaboration” directive; our default v2
adds a difficulty-aware caveat to protect hard problems; v3 names the
text categories that are safe to cut versus must-keep; v4 frames the
audience as an expert; and v5 imposes a terse numbered-list format. All
variants are trained identically (Qwen3-8B, full-vocab reverse-KL,
$`M{=}50`$) and evaluated at step 99.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzUuVDMucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmciIGhlaWdodD0iMjM2LjQ2IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDIzNi40NiIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwyMzYuNDYpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA4QzAwOyIgZmlsbD0iIzAwOEMwMCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuMzIgTCAwIDIzMS4xNCBDIDAgMjM0LjA4IDIuMzggMjM2LjQ2IDUuMzIgMjM2LjQ2IEwgNDcyLjA2IDIzNi40NiBDIDQ3NC45OSAyMzYuNDYgNDc3LjM4IDIzNC4wOCA0NzcuMzggMjMxLjE0IEwgNDc3LjM4IDUuMzIgQyA0NzcuMzggMi4zOCA0NzQuOTkgMCA0NzIuMDYgMCBMIDUuMzIgMCBDIDIuMzggMCAwIDIuMzggMCA1LjMyIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNFOEY1RTk7IiBmaWxsPSIjRThGNUU5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuMzggNS4zMiBMIDEuMzggMjMxLjE0IEMgMS4zOCAyMzMuMzEgMy4xNSAyMzUuMDggNS4zMiAyMzUuMDggTCA0NzIuMDYgMjM1LjA4IEMgNDc0LjIzIDIzNS4wOCA0NzUuOTkgMjMzLjMxIDQ3NS45OSAyMzEuMTQgTCA0NzUuOTkgNS4zMiBDIDQ3NS45OSAzLjE1IDQ3NC4yMyAxLjM4IDQ3Mi4wNiAxLjM4IEwgNS4zMiAxLjM4IEMgMy4xNSAxLjM4IDEuMzggMy4xNSAxLjM4IDUuMzIgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA4LjA5IDExNS4xMikiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDo0Ny44ZW07LS1sdHgtZm8taGVpZ2h0OjguODVlbTstLWx0eC1mby1kZXB0aDo4LjM2ZW07Zm9udC1zaXplOjkuMjVwdDsiIGNsYXNzPSJsdHhfbWluaXBhZ2UiIGhlaWdodD0iMjIwLjI5IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMTMuMjYpIiB3aWR0aD0iMzYuMDNlbSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8dGFibGUgaWQ9IlM1LlQzLnBpYzEuMSIgY2xhc3M9Imx0eF90YWJ1bGFyIGx0eF9hbGlnbl9taWRkbGUiPgo8dHIgaWQ9IlM1LlQzLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3RyIj4KPHRkIGlkPSJTNS5UMy5waWMxLjEuMS4xIiBjbGFzcz0ibHR4X3RkIGx0eF9hbGlnbl9sZWZ0IGx0eF9ib3JkZXJfdHQiIHN0eWxlPSJwYWRkaW5nLXRvcDoxLjE1cHQ7cGFkZGluZy1ib3R0b206MS4xNXB0OyI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5JRDwvc3Bhbj48L3RkPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS4xLjIiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCBsdHhfYm9yZGVyX3R0IiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjEuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NjYuN3B0OyI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuMS4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzUuVDMucGljMS4xLjEuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5NZWNoYW5pc208L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC90ZD4KPHRkIGlkPSJTNS5UMy5waWMxLjEuMS4zIiBjbGFzcz0ibHR4X3RkIGx0eF9ub3BhZF9yIGx0eF9hbGlnbl9sZWZ0IGx0eF9ib3JkZXJfdHQiIHN0eWxlPSJwYWRkaW5nLXRvcDoxLjE1cHQ7cGFkZGluZy1ib3R0b206MS4xNXB0OyI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuMS4zLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjEuMy4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4xLjMuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjkwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+SW5zdHJ1Y3Rpb24gPG1hdGggaWQ9IlM1LlQzLnBpYzEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iYyIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YzwvbWk+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5jPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gKGFicmlkZ2VkKTwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3RkPjwvdHI+Cjx0ciBpZD0iUzUuVDMucGljMS4xLjIiIGNsYXNzPSJsdHhfdHIiPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS4yLjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2JvcmRlcl90IiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjkwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+djE8L3NwYW4+PC90ZD4KPHRkIGlkPSJTNS5UMy5waWMxLjEuMi4yIiBjbGFzcz0ibHR4X3RkIGx0eF9hbGlnbl9sZWZ0IGx0eF9hbGlnbl90b3AgbHR4X2JvcmRlcl90IiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjIuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NjYuN3B0OyI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuMi4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzUuVDMucGljMS4xLjIuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjkwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+VW5pZm9ybTwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3RkPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS4yLjMiIGNsYXNzPSJsdHhfdGQgbHR4X25vcGFkX3IgbHR4X2FsaWduX2xlZnQgbHR4X2JvcmRlcl90IiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjIuMy4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIj4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4yLjMuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuMi4zLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Tb2x2ZSBjb25jaXNlbHkgYW5kIGNvcnJlY3RseS4gQmUgZGlyZWN04oCUYXZvaWQgdW5uZWNlc3NhcnkgZWxhYm9yYXRpb24sIHJlZHVuZGFudCBzdGVwcywgb3IgcmVzdGF0aW5nIHRoZSBwcm9ibGVtOyBmb2N1cyBvbmx5IG9uIHRoZSBrZXkgcmVhc29uaW5nIHN0ZXBzIG5lZWRlZCB0byByZWFjaCB0aGUgYW5zd2VyLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3RkPjwvdHI+Cjx0ciBpZD0iUzUuVDMucGljMS4xLjMiIGNsYXNzPSJsdHhfdHIiPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS4zLjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQiIHN0eWxlPSJwYWRkaW5nLXRvcDoxLjE1cHQ7cGFkZGluZy1ib3R0b206MS4xNXB0OyI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij52Mjwvc3Bhbj48L3RkPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS4zLjIiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctdG9wOjEuMTVwdDtwYWRkaW5nLWJvdHRvbToxLjE1cHQ7Ij4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4zLjIuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjY2LjdwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjMuMi4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4zLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo5MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkRpZmZpY3VsdHktYXdhcmUgKGRlZmF1bHQpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvdGQ+Cjx0ZCBpZD0iUzUuVDMucGljMS4xLjMuMyIgY2xhc3M9Imx0eF90ZCBsdHhfbm9wYWRfciBsdHhfYWxpZ25fbGVmdCIgc3R5bGU9InBhZGRpbmctdG9wOjEuMTVwdDtwYWRkaW5nLWJvdHRvbToxLjE1cHQ7Ij4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS4zLjMuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuMy4zLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzUuVDMucGljMS4xLjMuMy4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjkwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+U29sdmUgY29uY2lzZWx5IGFuZCBjb3JyZWN0bHk7IHByaW9yaXRpemUgYSBjb3JyZWN0IGFuc3dlciBvdmVyIGJyZXZpdHkuIEJlIGRpcmVjdCBvbiBzdHJhaWdodGZvcndhcmQgcHJvYmxlbXM7IGZvciBoYXJkL211bHRpLXN0ZXAgb25lcyBkbyBub3Qgb3Zlci1jb21wcmVzc+KAlGtlZXAgY2FzZSBhbmFseXNpcywgZWRnZSBjYXNlcywgYW5kIGEgZmluYWwgY2hlY2suPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvdGQ+PC90cj4KPHRyIGlkPSJTNS5UMy5waWMxLjEuNCIgY2xhc3M9Imx0eF90ciI+Cjx0ZCBpZD0iUzUuVDMucGljMS4xLjQuMSIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCIgc3R5bGU9InBhZGRpbmctdG9wOjEuMTVwdDtwYWRkaW5nLWJvdHRvbToxLjE1cHQ7Ij48c3BhbiBpZD0iUzUuVDMucGljMS4xLjQuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo5MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPnYzPC9zcGFuPjwvdGQ+Cjx0ZCBpZD0iUzUuVDMucGljMS4xLjQuMiIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjQuMi4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6NjYuN3B0OyI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuNC4yLjEuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iUzUuVDMucGljMS4xLjQuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjkwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+UmVkdW5kYW5jeS10YXJnZXRlZDwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3RkPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS40LjMiIGNsYXNzPSJsdHhfdGQgbHR4X25vcGFkX3IgbHR4X2FsaWduX2xlZnQiIHN0eWxlPSJwYWRkaW5nLXRvcDoxLjE1cHQ7cGFkZGluZy1ib3R0b206MS4xNXB0OyI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuNC4zLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjQuMy4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS40LjMuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo5MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPldyaXRlIG9ubHkgd2hhdCBpcyBuZWVkZWQ6IG9taXQgcmVzdGF0ZW1lbnRzLCBuYXJyYXRpb24sIGFuZCBvYnZpb3VzIGFyaXRobWV0aWM7IGtlZXAgZXZlcnkgbG9naWNhbGx5IG5lY2Vzc2FyeSBzdGVwIChjYXNlIHNwbGl0cywga2V5IGlkZWEsIGEgb25lLWxpbmUgY2hlY2spLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3RkPjwvdHI+Cjx0ciBpZD0iUzUuVDMucGljMS4xLjUiIGNsYXNzPSJsdHhfdHIiPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS41LjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQiIHN0eWxlPSJwYWRkaW5nLXRvcDoxLjE1cHQ7cGFkZGluZy1ib3R0b206MS4xNXB0OyI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS41LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij52NDwvc3Bhbj48L3RkPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS41LjIiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2FsaWduX3RvcCIgc3R5bGU9InBhZGRpbmctdG9wOjEuMTVwdDtwYWRkaW5nLWJvdHRvbToxLjE1cHQ7Ij4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS41LjIuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X2FsaWduX3RvcCIgc3R5bGU9IndpZHRoOjY2LjdwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjUuMi4xLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS41LjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo5MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkV4cGVydC1yZWFkZXI8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC90ZD4KPHRkIGlkPSJTNS5UMy5waWMxLjEuNS4zIiBjbGFzcz0ibHR4X3RkIGx0eF9ub3BhZF9yIGx0eF9hbGlnbl9sZWZ0IiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjUuMy4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIj4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS41LjMuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuNS4zLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Xcml0ZSBmb3IgYW4gZXhwZXJ0IHdobyB3YW50cyBvbmx5IHRoZSBlc3NlbnRpYWwgcmVhc29uaW5nOiBzdGF0ZSB0aGUga2V5IHN0ZXBzIGFuZCBkZWNpc2l2ZSBjb21wdXRhdGlvbiwgc2tpcCBzdGFuZGFyZC1mYWN0IGV4cGxhbmF0aW9ucyBhbmQgbmFycmF0aW9uOyBpbmNsdWRlIHJlcXVpcmVkIGNhc2UgYW5hbHlzaXMvY2hlY2tzLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3RkPjwvdHI+Cjx0ciBpZD0iUzUuVDMucGljMS4xLjYiIGNsYXNzPSJsdHhfdHIiPgo8dGQgaWQ9IlM1LlQzLnBpYzEuMS42LjEiIGNsYXNzPSJsdHhfdGQgbHR4X2FsaWduX2xlZnQgbHR4X2JvcmRlcl9iYiIgc3R5bGU9InBhZGRpbmctdG9wOjEuMTVwdDtwYWRkaW5nLWJvdHRvbToxLjE1cHQ7Ij48c3BhbiBpZD0iUzUuVDMucGljMS4xLjYuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo5MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPnY1PC9zcGFuPjwvdGQ+Cjx0ZCBpZD0iUzUuVDMucGljMS4xLjYuMiIgY2xhc3M9Imx0eF90ZCBsdHhfYWxpZ25fbGVmdCBsdHhfYWxpZ25fdG9wIGx0eF9ib3JkZXJfYmIiIHN0eWxlPSJwYWRkaW5nLXRvcDoxLjE1cHQ7cGFkZGluZy1ib3R0b206MS4xNXB0OyI+CjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuNi4yLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDo2Ni43cHQ7Ij4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS42LjIuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuNi4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5UZXJzZS1zdHJ1Y3R1cmVkPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvdGQ+Cjx0ZCBpZD0iUzUuVDMucGljMS4xLjYuMyIgY2xhc3M9Imx0eF90ZCBsdHhfbm9wYWRfciBsdHhfYWxpZ25fbGVmdCBsdHhfYm9yZGVyX2JiIiBzdHlsZT0icGFkZGluZy10b3A6MS4xNXB0O3BhZGRpbmctYm90dG9tOjEuMTVwdDsiPgo8c3BhbiBpZD0iUzUuVDMucGljMS4xLjYuMy4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfYWxpZ25fdG9wIj4KPHNwYW4gaWQ9IlM1LlQzLnBpYzEuMS42LjMuMS4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJTNS5UMy5waWMxLjEuNi4zLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6OTAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5QcmVzZW50IHRoZSBzb2x1dGlvbiBhcyBhIG1pbmltYWwgbnVtYmVyZWQgbGlzdOKAlGVhY2ggc3RlcCBvbmUgc2hvcnQgbGluZSwgbm8gcHJvc2UsIG5vIHJlc3RhdGVtZW50OyBrZWVwIHJlcXVpcmVkIGNhc2VzIGFuZCBlbmQgd2l0aCBhIG9uZS1saW5lIHZlcmlmaWNhdGlvbi48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC90ZD48L3RyPgo8L3RhYmxlPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)

Table 3: Conciseness-instruction variants. Each teacher prompt $`c`$
expresses “be concise” through a different mechanism. The answer-format
suffix (“The last line…Answer:”) is identical across variants and
omitted here.

Figure 5: Conciseness-instruction ablation: accuracy–compression
trade-off (Qwen3-8B, step 99, 30K budget). Each point is one instruction
(v1–v5); axes are token reduction (%, vs. base) and accuracy change
($`\Delta`$Acc, pp vs. base mean@8) on a, MATH-500 and b, AIME 2024.
Upper-right is better (more compression, less accuracy loss). Base model
matches
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
(95.7% / 4,884 tok MATH-500; 76.2% / 14,229 tok AIME 2024).

Figure [5](#S5.F5 "Figure 5 ‣ 5.3.2 Which Conciseness Instruction Compresses Best? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
reveals a clear accuracy–compression frontier across instructions. The
difficulty-aware v2 default best preserves accuracy ($`+0.0\%`$ on
MATH-500, $`-1.6\%`$ on AIME 2024) but compresses least (31.6% / 17.1%),
because its explicit “do not over-compress on hard problems” caveat
protects deliberation. The uniform v1 instruction sits further along the
frontier: dropping that caveat nearly doubles MATH-500 compression to
57.4% (and AIME to 36.5%) for only a small accuracy cost ($`-0.1\%`$ /
$`-2.7\%`$), showing that the difficulty-aware guardrail trades
substantial compression for a marginal accuracy gain. Pushing harder
still, the redundancy-targeted v3 reaches the highest MATH-500
compression (62.9%) but at a real accuracy cost ($`-1.3\%`$ MATH-500,
$`-5.4\%`$ AIME), as do v4 (expert-reader; strongest AIME compression at
40.2%) and v5 (terse format). Overall, no single instruction dominates:
v2 is the accuracy-safe default, v1 is the best choice when aggressive
compression matters, and the more prescriptive v3–v5 buy extra
compression only by giving up accuracy.

#### 5.3.3 How Sensitive Is Compression to the Teacher Update Interval?

The teacher update interval $`M`$
(Eq. [2](#S3.E2 "In 3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
controls how frequently the teacher weights are synchronized with the
student. A larger $`M`$ provides a more stable distillation target but
limits progressive compression; a smaller $`M`$ pushes compression
further but risks instability when the teacher chases a rapidly-moving
student. We sweep $`M\in\{1,10,30,50,100\}`$ on Qwen3-8B (full-vocab
reverse-KL, all else fixed). Because we evaluate at step 99, $`M{=}100`$
never reaches its first refresh and is therefore effectively a *frozen*
teacher.
Figure [6](#S5.F6 "Figure 6 ‣ 5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
reports accuracy and token reduction at step 99.

Figure 6: Teacher update interval $`M`$ ablation (Qwen3-8B, step 99, 30K
budget). Accuracy (*left*) and token reduction (*right*) vs. $`M`$ on
MATH-500 and AIME 2024, with the base taken from
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").
Smaller $`M`$ compresses more aggressively but, past a point,
destabilizes training; $`M{=}1`$ collapses (MATH-500 response length
*grows*, hence negative reduction). At a step-99 evaluation $`M{=}100`$
has not yet refreshed and is effectively a frozen teacher. We use
$`M{=}50`$ as the default (dotted line).

Figure [6](#S5.F6 "Figure 6 ‣ 5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
reveals a clear compression–stability trade-off governed by $`M`$:

$`M{=}1`$ collapses. Refreshing the teacher after every gradient step
creates a moving-target feedback loop in which the student chases a
teacher that is itself changing in response to the student. Training
then degenerates: MATH-500 accuracy falls to 60.8% (AIME 2024 to 6.7%),
MATH-500 response length grows rather than shrinks (negative reduction),
and AIME compression collapses to near zero. This mirrors the
instability reported by [Shenfeld et al. (2026)](#bib.bib26).

Small but non-trivial $`M`$ ($`10`$–$`30`$) compresses hardest.
$`M{=}10`$ removes 64% of MATH-500 tokens and 46% on AIME 2024, the most
aggressive compression in the sweep, but at a real accuracy cost (AIME
2024 drops to 49.6%). $`M{=}30`$ recovers most of that accuracy (67.9%)
while still compressing 52% and 29%.

The frozen teacher ($`M{=}100`$) preserves accuracy but compresses
least. Since $`M{=}100`$ never refreshes before step 99, it acts as a
single static concise teacher: accuracy stays high (95.1% MATH-500,
77.5% AIME 2024) but compression is only 18% and 11%. That every
refreshing schedule ($`M\leq 50`$) compresses more confirms that
periodic refresh, rather than a one-shot concise teacher, is what drives
progressive compression.

We adopt $`M{=}50`$ as the default: it sits at the knee of this
trade-off, retaining base-level accuracy on MATH-500 (95.7%) and AIME
2024 (75.0%) while still achieving substantial compression (31.6% /
17.1%).

#### 5.3.4 How Sensitive Is Compression to the Rollout Temperature?

CRISP generates student rollouts with sampling temperature $`\tau`$,
which controls the diversity of the trajectories the teacher then
scores. We sweep $`\tau\in\{0.3,0.6,1.0,1.5\}`$ with all else fixed
(Qwen3-8B, full-vocab reverse-KL, $`M{=}50`$, v2 instruction),
evaluating at step 99
(Figure [7](#S5.F7 "Figure 7 ‣ 5.3.4 How Sensitive Is Compression to the Rollout Temperature? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

Figure 7: Rollout temperature ablation (Qwen3-8B, step 99, 30K budget).
Accuracy change ($`\Delta`$Acc vs. base mean@8, *left*) and token
reduction (*right*) vs. the rollout temperature $`\tau`$ on MATH-500 and
AIME 2024. Base model identical to
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").
The curves are nearly flat, showing robustness across $`\tau`$; the
default $`\tau{=}1.0`$ (dotted line) is the best operating point.

CRISP is robust to rollout temperature. Across the full
$`\tau\in[0.3,1.5]`$ range, MATH-500 accuracy stays within $`0.7`$ pp of
the base model and compression varies by less than $`3`$ pp ($`30.8`$ to
$`33.3\%`$). The default $`\tau{=}1.0`$ is the best overall operating
point: it gives the smallest AIME accuracy drop ($`-1.6\%`$) at
compression comparable to the rest of the sweep ($`17.1\%`$). Moderate
sampling diversity exposes the teacher to a representative spread of
student trajectories, without the degenerate sampling that very high or
very low temperatures induce. We therefore keep $`\tau{=}1.0`$ as the
default.

#### 5.3.5 How Sensitive Is Compression to the Rollout Length?

Because CRISP optimizes a per-token KL objective rather than an outcome
reward, it does not require complete rollouts; partial trajectories
already supply a training signal
(Section [5.1](#S5.SS1 "5.1 Experimental Setting ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
We test how short the training rollouts can be by capping the student
response length during training at
$`\{1\text{k},4\text{k},8\text{k},30\text{k}\}`$ tokens, all else fixed
(Qwen3-8B, $`M{=}50`$, v2), and evaluating at step 99 under the common
30K eval budget
(Figure [8](#S5.F8 "Figure 8 ‣ 5.3.5 How Sensitive Is Compression to the Rollout Length? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

Figure 8: Training rollout-length ablation (Qwen3-8B, step 99, 30K eval
budget). Accuracy change ($`\Delta`$Acc vs. base mean@8, *left*) and
token reduction (*right*) vs. the maximum *training* rollout length on
MATH-500 and AIME 2024; all variants are evaluated under the same 30K
budget. Base model identical to
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").
Even a 1k-token cap trains well; the $`8`$k default (dotted line) is the
best operating point, while the longest 30k cap is not better.

The result confirms that short training rollouts suffice. Even a
$`1`$k-token cap, roughly a fifth of the base model’s MATH-500 length,
recovers $`29.8\%`$ MATH-500 compression with no accuracy loss, which
validates the partial-rollout design. The $`8`$k default is the best
setting: it gives the strongest MATH-500 compression ($`31.6\%`$) and
the smallest AIME accuracy drop ($`-1.6\%`$), with AIME compression
($`17.1\%`$) on par with the $`4`$k cap. The longest cap ($`30`$k) is
not better; its AIME accuracy drops most ($`-6.5\%`$) while compressing
least ($`13.8\%`$), because unconstrained-length rollouts let the
student reinforce its own verbosity before the teacher signal takes
effect. This makes CRISP cheaper to train than outcome-reward RL, which
needs full-length completions to compute a terminal reward.

## 6 Limitations and Future Work

We discuss the scope of the current study and natural directions for
future work.

##### Instruction-following as an enabler.

CRISP depends on the base model’s ability to follow the conciseness
instruction, since the teacher signal comes entirely from that
instruction. Models that follow instructions well produce a clear
teacher distribution and compress reliably, as we observe across
Qwen3-8B, Qwen3-14B, and DeepSeek-R1-Distill-Llama-8B. Models with weak
instruction-following would provide a weaker teacher signal.
Characterizing the minimum instruction-following capability needed for
effective self-distillation is a direction for future work.

##### Progressive compression dynamics.

The periodic teacher update
(Eq. [2](#S3.E2 "In 3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
pushes compression beyond a frozen teacher, and the method is robust
across a range of moderate update intervals ($`M\in\{30,50\}`$;
Section [5.3.3](#S5.SS3.SSS3 "5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
The compression signal on harder benchmarks such as AIME is weaker
because the teacher itself requires more extensive reasoning. We view
this as a feature of difficulty-adaptive compression
(Section [3.3](#S3.SS3.SSS0.Px1 "Difficulty-adaptive compression. ‣ 3.3 Teacher Parameterization ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
rather than a limitation.

##### Scope of evaluation.

This paper focuses on mathematical reasoning as a controlled testbed
where accuracy can be verified precisely. The design of CRISP is
domain-agnostic, since it requires only problem prompts and a
conciseness instruction, so it applies to other reasoning domains such
as code generation and scientific question answering where ground-truth
verification is unavailable. The main properties of CRISP (no
ground-truth requirement, difficulty adaptivity, and entropy
preservation) are structural and do not depend on the evaluation domain.
Extending the empirical evaluation to broader reasoning tasks is a
natural next step.

##### Teacher quality characterization.

Our experiments consistently show that the conciseness-conditioned
teacher improves accuracy
(Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
and the theoretical analysis
(Theorem [2](#Thmtheorem2 "Theorem 2 (Accuracy preservation). ‣ A.2 Accuracy Preservation under Compression ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
provides formal bounds on accuracy preservation. A finer-grained
characterization of when and why conciseness instructions improve versus
degrade accuracy across different model families would further
strengthen the understanding of self-distillation dynamics.

## 7 Conclusion

We presented CRISP, an on-policy self-distillation method that
compresses reasoning by distilling a model’s own concise behavior back
into itself. The method uses a conciseness instruction to define a
teacher and minimizes per-token reverse KL to that teacher on the
student’s own rollouts, without ground-truth answers, token budgets, or
difficulty estimators. Across three model families, CRISP shortens
reasoning traces substantially while preserving accuracy, and it
improves accuracy when the base model has room to improve.

Two conclusions follow from these results. First, a large fraction of
the tokens that reasoning models produce are redundant, and removing
them preserves or improves accuracy rather than degrading it. Second,
models already contain a concise reasoning mode that a conciseness
instruction can elicit, and on-policy self-distillation can make that
mode the default without reducing entropy
(Appendix [H](#A8 "Appendix H Entropy Preservation During Training ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
or general capability.

Because CRISP supervises only on a conciseness instruction and the
model’s own rollouts, it applies to domains where ground-truth answers
or reliable verifiers are unavailable, provided the model can follow the
instruction.

## References

- Aggarwal and Welleck \[2025\] P. Aggarwal and S. Welleck. L1:
  Controlling how long a reasoning model thinks with reinforcement
  learning. *arXiv preprint arXiv:2503.04697*, 2025.
- Xu et al. \[2026\] Y. Xu, H. Sang, Z. Zhou, R. He, and Z. Wang.
  Overconfident errors need stronger correction: Asymmetric confidence
  penalties for reinforcement learning. *arXiv preprint
  arXiv:2602.21420*, 2026.
- Zhang et al. \[2026\] Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong
  Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv, and Junyang Lin.
  Deepplanning: Benchmarking long-horizon agentic planning with
  verifiable constraints. *arXiv preprint arXiv:2601.18137*, 2026.
- Hou et al. \[2025\] B. Hou, Y. Zhang, J. Ji, Y. Liu, K. Qian,
  J. Andreas, and S. Chang. Thinkprune: Pruning long chain-of-thought of
  llms via reinforcement learning. *arXiv preprint
  arXiv:2504.01296*, 2025.
- Chen et al. \[2025\] W. Chen, J. Yuan, T. Jin, N. Ding, H. Chen,
  Z. Liu, and M. Sun. The overthinker’s diet: Cutting token calories
  with difficulty-aware training. *arXiv preprint
  arXiv:2505.19217*, 2025.
- Chen et al. \[2025b\] W.-R. Chen, V. Kothapalli, A. Fatahibaarzi,
  H. Sang, S. Tang, Q. Song, Z. Wang, and M. Abdul-Mageed. Distilling
  the essence: Efficient reasoning distillation via sequence truncation.
  *arXiv preprint arXiv:2512.21002*, 2025b.
- Chen et al. \[2024\] X. Chen, J. Xu, T. Liang, Z. He, J. Pang, D. Yu,
  L. Song, Q. Liu, M. Zhou, Z. Zhang, et al. Do not think that much for
  2+ 3=? on the overthinking of o1-like llms. *arXiv preprint
  arXiv:2412.21187*, 2024.
- Comanici et al. \[2025\] G. Comanici, E. Bieber, M. Schaekermann,
  I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang,
  E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced
  reasoning, multimodality, long context, and next generation agentic
  capabilities. *arXiv preprint arXiv:2507.06261*, 2025.
- Cui et al. \[2025\] G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang,
  Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, et al. The entropy mechanism
  of reinforcement learning for reasoning language models. *arXiv
  preprint arXiv:2505.22617*, 2025.
- Du et al. \[2026\] Y. Du, S. Zhao, Y. Gao, D. Zhao, Q. Lin, M. Ma,
  J. Li, Y. Jiang, K. He, Q. Xu, et al. S3-cot: Self-sampled succinct
  reasoning enables efficient chain-of-thought llms. *arXiv preprint
  arXiv:2602.01982*, 2026.
- Gao et al. \[2021\] L. Gao, J. Tow, S. Biderman, S. Black, A. DiPofi,
  C. Foster, L. Golding, J. Hsu, K. McDonell, N. Muennighoff, et al. A
  framework for few-shot language model evaluation, 2021.
- Gu et al. \[2023\] Y. Gu, L. Dong, F. Wei, and M. Huang. Minillm:
  Knowledge distillation of large language models. In *arXiv preprint
  arXiv:2306.08543*, 2023.
- Guo et al. \[2025\] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang,
  Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1:
  Incentivizing reasoning capability in llms via reinforcement learning.
  *arXiv preprint arXiv:2501.12948*, 2025.
- Hendrycks et al. \[2020\] D. Hendrycks, C. Burns, S. Basart, A. Zou,
  M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask
  language understanding. *arXiv preprint arXiv:2009.03300*, 2020.
- Hendrycks et al. \[2021\] D. Hendrycks, C. Burns, S. Kadavath,
  A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring
  mathematical problem solving with the math dataset. *arXiv preprint
  arXiv:2103.03874*, 2021.
- Huang et al. \[2025\] K. Huang, S. Liu, X. Hu, T. Xu, L. Bao, and
  X. Xia. Reasoning efficiently through adaptive chain-of-thought
  compression: A self-optimizing framework. *arXiv preprint
  arXiv:2509.14093*, 2025.
- Hübotter et al. \[2026\] J. Hübotter, F. Lübeck, L. Behric,
  A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K.
  Buening, C. Guestrin, et al. Reinforcement learning via
  self-distillation. *arXiv preprint arXiv:2601.20802*, 2026.
- Jaech et al. \[2024\] A. Jaech, A. Kalai, A. Lerer, A. Richardson,
  A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney,
  et al. Openai o1 system card. *arXiv preprint arXiv:2412.16720*, 2024.
- Li et al. \[2025a\] L. Li, J. Hao, J. K. Liu, Z. Zhou, Y. Miao,
  W. Pang, X. Tan, W. Chu, Z. Wang, S. Pan, et al. The choice of
  divergence: A neglected key to mitigating diversity collapse in
  reinforcement learning with verifiable reward. *arXiv preprint
  arXiv:2509.07430*, 2025a.
- Li et al. \[2025b\] Y. Li, L. Ma, J. Zhang, L. Tang, W. Zhang, and
  G. Luo. Leash: Adaptive length penalty and reward shaping for
  efficient large reasoning model. *arXiv preprint arXiv:2512.21540*,
  2025b.
- Li et al. \[2026\] Y. Li, B. Bergner, Y. Zhao, V. P. Patil, B. Chen,
  and C. Wang. Steering large reasoning models towards concise reasoning
  via flow matching. *arXiv preprint arXiv:2602.05539*, 2026.
- Lin et al. \[2025\] W. Lin, X. Li, Z. Yang, X. Fu, H.-L. Zhen,
  Y. Wang, X. Yu, W. Liu, X. Li, and M. Yuan. Trimr: Verifier-based
  training-free thinking compression for efficient test-time scaling.
  *arXiv preprint arXiv:2505.17155*, 2025.
- Liu et al. \[2025\] S. Liu, X. Dong, X. Lu, S. Diao, M. Liu, M. Chen,
  H. Yin, Y. Wang, K. Cheng, Y. Choi, et al. DLER: Doing length penalty
  right – incentivizing more intelligence per token via reinforcement
  learning. *arXiv preprint arXiv:2510.15110*, 2025.
- Muennighoff et al. \[2025\] N. Muennighoff, Z. Yang, W. Shi, X. L. Li,
  L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and
  T. B. Hashimoto. s1: Simple test-time scaling. In *Proceedings of the
  2025 Conference on Empirical Methods in Natural Language Processing*,
  pages 20286–20332, 2025.
- Rein et al. \[2023\] D. Rein, B. L. Hou, A. C. Stickland, J. Petty,
  R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. Gpqa: A
  graduate-level google-proof q&a benchmark. *arXiv preprint
  arXiv:2311.12022*, 2023.
- Shenfeld et al. \[2026\] I. Shenfeld, M. Damani, J. Hübotter, and
  P. Agrawal. Self-distillation enables continual learning. *arXiv
  preprint arXiv:2601.19897*, 2026.
- Sheng et al. \[2025\] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang,
  R. Zhang, Y. Peng, H. Lin, and C. Wu. Hybridflow: A flexible and
  efficient rlhf framework. In *Proceedings of the Twentieth European
  Conference on Computer Systems*, pages 1279–1297, 2025.
- Snell et al. \[2024\] C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling
  llm test-time compute optimally can be more effective than scaling
  model parameters. *arXiv preprint arXiv:2408.03314*, 2024.
- Wan et al. \[2026\] Q. Wan, Z. Xu, L. Wei, X. Shen, and J. Sun.
  Mitigating overthinking in large reasoning models via difficulty-aware
  reinforcement learning. *arXiv preprint arXiv:2601.21418*, 2026.
- Wang et al. \[2025a\] C. Wang, Y. Feng, D. Chen, Z. Chu, R. Krishna,
  and T. Zhou. Wait, we don’t need to" wait"! removing thinking tokens
  improves reasoning efficiency. *arXiv preprint arXiv:2506.08343*,
  2025a.
- Wang et al. \[2025b\] S. Wang, L. Yu, C. Gao, C. Zheng, S. Liu, R. Lu,
  K. Dang, X. Chen, J. Yang, Z. Zhang, et al. Beyond the 80/20 rule:
  High-entropy minority tokens drive effective reinforcement learning
  for llm reasoning. *arXiv preprint arXiv:2506.01939*, 2025b.
- Wu et al. \[2025\] Y. Wu, J. Shi, B. Wu, J. Zhang, X. Lin, N. Tang,
  and Y. Luo. Concise reasoning, big gains: Pruning long reasoning trace
  with difficulty-aware prompting. *arXiv preprint
  arXiv:2505.19716*, 2025.
- Xia et al. \[2025\] H. Xia, C. T. Leong, W. Wang, Y. Li, and W. Li.
  Tokenskip: Controllable chain-of-thought compression in llms.
  *Proceedings of the 2025 Conference on Empirical Methods in Natural
  Language Processing*, pages 3351–3363, 2025.
- Xu et al. \[2025\] S. Xu, W. Xie, L. Zhao, and P. He. Chain of draft:
  Thinking faster by writing less. *arXiv preprint
  arXiv:2502.18600*, 2025.
- Yang et al. \[2025\] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui,
  B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical
  report. *arXiv preprint arXiv:2505.09388*, 2025.
- Ye et al. \[2026\] T. Ye, L. Dong, X. Wu, S. Huang, and F. Wei.
  On-policy context distillation for language models. *arXiv preprint
  arXiv:2602.12275*, 2026.
- Yu et al. \[2025\] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue,
  W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm
  reinforcement learning system at scale. *arXiv preprint
  arXiv:2503.14476*, 2025.
- Zhao et al. \[2026\] S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang,
  F. Chen, and A. Grover. Self-distilled reasoner: On-policy
  self-distillation for large language models. *arXiv preprint
  arXiv:2601.18734*, 2026.
- Zheng et al. \[2024\] L. Zheng, L. Yin, Z. Xie, C. L. Sun, J. Huang,
  C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, et al.
  Sglang: Efficient execution of structured language model programs.
  *Advances in neural information processing systems*,
  37:62557–62583, 2024.

## Appendix A Theoretical Analysis

We provide a formal analysis of CRISP’s key properties: the connection
between the per-token training loss and sequence-level divergence
(Section [A.1](#A1.SS1 "A.1 Sequence-Level Divergence and the Training Objective ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
accuracy preservation guarantees
(Section [A.2](#A1.SS2 "A.2 Accuracy Preservation under Compression ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
formalization of difficulty-adaptive compression
(Section [A.3](#A1.SS3 "A.3 Difficulty-Adaptive Compression ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
forgetting bounds relative to the base model
(Section [A.4](#A1.SS4 "A.4 Bounded Forgetting from the Base Model ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
and a probabilistic model of how compression reduces compounding errors
(Section [A.5](#A1.SS5 "A.5 Compression Reduces Compounding Error ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

### A.1 Sequence-Level Divergence and the Training Objective

We first establish that the per-token CRISP objective
(Eq. [1](#S3.E1 "In 3.2 Training Objective ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
is equivalent to minimizing the sequence-level KL divergence between
student and teacher. This identification enables the application of
standard information-theoretic tools to the per-token loss.

###### Lemma 1 (Chain rule of KL for autoregressive models).

For autoregressive distributions
$`q(y\mid x)=\prod_{t=1}^{|y|}q(y_{t}\mid x,y_{<t})`$ and
$`p(y\mid x)=\prod_{t=1}^{|y|}p(y_{t}\mid x,y_{<t})`$ over the same
token vocabulary, the sequence-level KL divergence decomposes as:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
D_{\mathrm{KL}}\big(q(\cdot\mid x)\;\big\|\;p(\cdot\mid x)\big)=\mathbb{E}_{y\sim q}\left[\sum_{t=1}^{|y|}D_{\mathrm{KL}}\big(q(\cdot\mid x,y_{<t})\;\big\|\;p(\cdot\mid x,y_{<t})\big)\right].
``` |  | (8) |

###### Proof.

By definition of KL divergence and the autoregressive factorization:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle D_{\mathrm{KL}}(q\|p)`$ | $`\displaystyle=\mathbb{E}_{y\sim q}\!\left[\log\frac{q(y\mid x)}{p(y\mid x)}\right]=\mathbb{E}_{y\sim q}\!\left[\sum_{t=1}^{|y|}\log\frac{q(y_{t}\mid x,y_{<t})}{p(y_{t}\mid x,y_{<t})}\right]`$ |  | (9) |
|  |  | $`\displaystyle=\mathbb{E}_{y\sim q}\!\left[\sum_{t=1}^{|y|}D_{\mathrm{KL}}\big(q(\cdot\mid x,y_{<t})\|p(\cdot\mid x,y_{<t})\big)\right],`$ |  | (10) |

where the second equality uses $`\log\prod_{t}=\sum_{t}\log`$, and the
last step recognizes each summand as a per-token KL divergence evaluated
at the sampled prefix $`y_{<t}`$. ∎

###### Corollary 1.

Identifying $`q=\pi_{\theta}(\cdot\mid x)`$ and
$`p=\pi_{\bar{\theta}}(\cdot\mid x,c)`$, the CRISP training loss
(Eq. [1](#S3.E1 "In 3.2 Training Objective ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
equals the expected sequence-level KL divergence:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\big[D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot\mid x)\;\big\|\;\pi_{\bar{\theta}}(\cdot\mid x,c)\big)\big].
``` |  | (11) |

### A.2 Accuracy Preservation under Compression

We show that if self-distillation converges (the training loss is small)
and the concise teacher preserves accuracy, then the student’s accuracy
is guaranteed to remain close to the base model’s.

###### Definition 1 (Accuracy).

For a problem distribution $`\mathcal{D}`$ with correct-answer sets
$`\{A(x)\}_{x\in\mathcal{D}}`$, the accuracy of policy $`\pi`$ is
$`\mathrm{Acc}(\pi)=\mathbb{E}_{x\sim\mathcal{D}}\big[\pi(A(x)\mid x)\big]`$,
where $`\pi(A(x)\mid x)=\sum_{y\in A(x)}\pi(y\mid x)`$.

###### Theorem 2 (Accuracy preservation).

Let $`\pi_{\theta^{*}}`$ denote the converged student with training loss
$`\mathcal{L}(\theta^{*})\leq\epsilon_{\mathrm{KL}}`$. Suppose the
concise teacher preserves accuracy relative to the base model:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathrm{Acc}\big(\pi_{\bar{\theta}}(\cdot\mid\cdot,c)\big)\geq\mathrm{Acc}(\pi_{\bar{\theta}})-\epsilon_{T}.
``` |  | (12) |

Then the student satisfies:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathrm{Acc}(\pi_{\theta^{*}})\geq\mathrm{Acc}(\pi_{\bar{\theta}})-\epsilon_{T}-\sqrt{\frac{\epsilon_{\mathrm{KL}}}{2}}.
``` |  | (13) |

###### Proof.

By
Corollary [1](#Thmcorollary1 "Corollary 1. ‣ A.1 Sequence-Level Divergence and the Training Objective ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"),
we have
$`\mathbb{E}_{x\sim\mathcal{D}}\big[D_{\mathrm{KL}}(\pi_{\theta^{*}}(\cdot\mid x)\|\pi_{\bar{\theta}}(\cdot\mid x,c))\big]\leq\epsilon_{\mathrm{KL}}`$.

Step 1: KL to total variation. For each problem $`x`$, Pinsker’s
inequality gives:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\bar{\theta}}(\cdot\mid x,c)\big)\leq\sqrt{\tfrac{1}{2}\,D_{\mathrm{KL}}\big(\pi_{\theta^{*}}(\cdot\mid x)\|\pi_{\bar{\theta}}(\cdot\mid x,c)\big)}.
``` |  | (14) |

Taking expectations over $`x\sim\mathcal{D}`$ and applying Jensen’s
inequality (using concavity of $`\sqrt{\cdot}`$):

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbb{E}_{x}\big[d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\bar{\theta}}(\cdot\mid x,c)\big)\big]\leq\sqrt{\tfrac{1}{2}\,\mathbb{E}_{x}\!\big[D_{\mathrm{KL}}\big(\pi_{\theta^{*}}(\cdot\mid x)\|\pi_{\bar{\theta}}(\cdot\mid x,c)\big)\big]}\leq\sqrt{\tfrac{\epsilon_{\mathrm{KL}}}{2}}.
``` |  | (15) |

Step 2: Total variation to accuracy. Since total variation bounds the
difference in probability of any event, in particular the correctness
event $`A(x)`$:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\big|\pi_{\theta^{*}}(A(x)\mid x)-\pi_{\bar{\theta}}(A(x)\mid x,c)\big|\leq d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\bar{\theta}}(\cdot\mid x,c)\big).
``` |  | (16) |

Taking expectations over $`x\sim\mathcal{D}`$ and using
$`|\mathbb{E}[f]|\leq\mathbb{E}[|f|]`$:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\big|\mathrm{Acc}(\pi_{\theta^{*}})-\mathrm{Acc}\big(\pi_{\bar{\theta}}(\cdot\mid\cdot,c)\big)\big|\leq\mathbb{E}_{x}\big[d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\bar{\theta}}(\cdot\mid x,c)\big)\big]\leq\sqrt{\tfrac{\epsilon_{\mathrm{KL}}}{2}}.
``` |  | (17) |

Step 3: Combine with teacher quality. From the assumption
$`\mathrm{Acc}(\pi_{\bar{\theta}}(\cdot\mid\cdot,c))\geq\mathrm{Acc}(\pi_{\bar{\theta}})-\epsilon_{T}`$:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle\mathrm{Acc}(\pi_{\theta^{*}})`$ | $`\displaystyle\geq\mathrm{Acc}\big(\pi_{\bar{\theta}}(\cdot\mid\cdot,c)\big)-\sqrt{\tfrac{\epsilon_{\mathrm{KL}}}{2}}\geq\mathrm{Acc}(\pi_{\bar{\theta}})-\epsilon_{T}-\sqrt{\tfrac{\epsilon_{\mathrm{KL}}}{2}}.\qed`$ |  | (18) |

###### Remark 2 (When the bound is vacuous, and why that is informative).

Theorem [2](#Thmtheorem2 "Theorem 2 (Accuracy preservation). ‣ A.2 Accuracy Preservation under Compression ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
identifies two independent sources of potential accuracy loss: the
teacher accuracy gap $`\epsilon_{T}`$ and the student–teacher divergence
$`\sqrt{\epsilon_{\mathrm{KL}}/2}`$. In our experiments,
$`\epsilon_{T}`$ is consistently *negative* (the concise teacher is more
accurate than the base model), making the bound vacuous in the
traditional sense. This is informative rather than a weakness: it
reveals why CRISP improves accuracy. The bound becomes
$`\mathrm{Acc}(\pi_{\theta^{*}})\geq\mathrm{Acc}(\pi_{\bar{\theta}})+|\epsilon_{T}|-\sqrt{\epsilon_{\mathrm{KL}}/2}`$,
so accuracy improves whenever the teacher’s accuracy gain exceeds the
distillation gap. The theorem is most useful in the regime of aggressive
compression where $`\epsilon_{T}`$ might turn positive; it then
quantifies the worst-case accuracy degradation.

### A.3 Difficulty-Adaptive Compression

We formalize the empirical observation
(Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
that CRISP compresses easy problems aggressively while preserving
reasoning on hard problems.

###### Definition 2 (Essential and compressible tokens).

For problem $`x`$ and student rollout
$`y\sim\pi_{\theta}(\cdot\mid x)`$, classify each token position $`t`$
based on the implicit reward sign
(Theorem [1](#Thmtheorem1 "Theorem 1 (Implicit reward). ‣ 4.2 Implicit Reward Interpretation ‣ 4 Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")):

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | $`\displaystyle\mathcal{E}(x,y)`$ | $`\displaystyle=\big\{t:\pi_{\bar{\theta}}(y_{t}\mid x,c,y_{<t})\geq\pi_{\theta}(y_{t}\mid x,y_{<t})\big\}`$ |  | $`\displaystyle\text{(essential: }r(y_{t},x)\geq 0\text{)},`$ |  | (19) |
|  | $`\displaystyle\mathcal{C}(x,y)`$ | $`\displaystyle=\big\{t:\pi_{\bar{\theta}}(y_{t}\mid x,c,y_{<t})<\pi_{\theta}(y_{t}\mid x,y_{<t})\big\}`$ |  | $`\displaystyle\text{(compressible: }r(y_{t},x)<0\text{)}.`$ |  | (20) |

###### Proposition 1 (Difficulty-adaptive compression signal).

Let $`d(x)\in[0,1]`$ denote problem difficulty, defined as the base
model’s failure rate $`d(x)=1-\pi_{\theta_{0}}(A(x)\mid x)`$. Assume:

- (A1)
  Essential fraction increases with difficulty: The expected fraction of
  essential tokens
  $`\rho(x)\coloneqq\mathbb{E}_{y}[|\mathcal{E}(x,y)|/|y|]`$ is
  non-decreasing in $`d(x)`$.
- (A2)
  Category-level KL is problem-independent: There exist constants
  $`D_{\mathcal{E}},D_{\mathcal{C}}>0`$ such that for all problems
  $`x`$:

  |  |  |  |  |  |
  |----|----|----|----|----|
  |  | $`\displaystyle\mathbb{E}\big[D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot\mid x,y_{<t})\|\pi_{\bar{\theta}}(\cdot\mid x,c,y_{<t})\big)\,\big|\,t\in\mathcal{C}(x,y)\big]`$ | $`\displaystyle=D_{\mathcal{C}},`$ |  | (21) |
  |  | $`\displaystyle\mathbb{E}\big[D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot\mid x,y_{<t})\|\pi_{\bar{\theta}}(\cdot\mid x,c,y_{<t})\big)\,\big|\,t\in\mathcal{E}(x,y)\big]`$ | $`\displaystyle=D_{\mathcal{E}}.`$ |  | (22) |
- (A3)
  Compressible tokens carry strictly larger KL:
  $`D_{\mathcal{C}}>D_{\mathcal{E}}`$.

Then the expected normalized compression signal

|  |  |  |  |
|----|----|----|----|
|  |
``` math
S(x)=\mathbb{E}_{y\sim\pi_{\theta}(\cdot\mid x)}\!\left[\frac{1}{|y|}\sum_{t=1}^{|y|}D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot\mid x,y_{<t})\|\pi_{\bar{\theta}}(\cdot\mid x,c,y_{<t})\big)\right]
``` |  | (23) |

is non-increasing in $`d(x)`$.

###### Proof.

Decompose the normalized KL into essential and compressible
contributions. For a given rollout $`y`$:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\frac{1}{|y|}\sum_{t}D_{\mathrm{KL}}(q_{t}\|p_{t})=\underbrace{\frac{|\mathcal{E}|}{|y|}\cdot\bar{D}_{\mathcal{E}}}_{\text{essential term}}+\underbrace{\frac{|\mathcal{C}|}{|y|}\cdot\bar{D}_{\mathcal{C}}}_{\text{compressible term}},
``` |  | (24) |

where $`q_{t}=\pi_{\theta}(\cdot\mid x,y_{<t})`$,
$`p_{t}=\pi_{\bar{\theta}}(\cdot\mid x,c,y_{<t})`$, and
$`\bar{D}_{\mathcal{E}},\bar{D}_{\mathcal{C}}`$ denote the average
per-token KL on essential and compressible tokens in this rollout,
respectively. Taking expectations over $`y`$ and applying
assumption (A2):

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle S(x)`$ | $`\displaystyle=\rho(x)\cdot D_{\mathcal{E}}+(1-\rho(x))\cdot D_{\mathcal{C}}`$ |  | (25) |
|  |  | $`\displaystyle=D_{\mathcal{C}}-\rho(x)\cdot(D_{\mathcal{C}}-D_{\mathcal{E}}).`$ |  | (26) |

By (A3), $`D_{\mathcal{C}}-D_{\mathcal{E}}>0`$, so $`S(x)`$ is a
strictly decreasing affine function of $`\rho(x)`$. Since $`\rho(x)`$ is
non-decreasing in $`d(x)`$ by (A1), $`S(x)`$ is non-increasing in
$`d(x)`$.

*Quantitatively*, for two problems with difficulties $`d_{1}<d_{2}`$
(hence $`\rho(x_{1})\leq\rho(x_{2})`$ by A1):

|  |  |  |  |
|----|----|----|----|
|  |
``` math
S(x_{1})-S(x_{2})=\big(\rho(x_{2})-\rho(x_{1})\big)\cdot\big(D_{\mathcal{C}}-D_{\mathcal{E}}\big)\geq 0.\qed
``` |  | (27) |

###### Remark 3.

Assumption (A1), that harder problems have a larger fraction of
essential reasoning steps, is the core structural assumption. It is
empirically supported by the monotonic compression–difficulty
relationship in
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"):
on Qwen3-14B, MATH-500 (up to 56% compression) is compressed more than
the harder AIME benchmarks (32–38%). Assumption (A2) posits that the
average KL divergence on essential and compressible tokens depends on
token *category* (essential vs. compressible) rather than on the
specific problem. This is a modeling simplification; in practice,
per-token KL values vary across problems, and the proposition should be
interpreted as holding for the category-averaged quantities.
Assumption (A3) states that compressible tokens exhibit a larger
vocabulary-level KL divergence than essential tokens. Note that this
does not follow directly from the per-token reward sign: the
essential/compressible classification
(Definition [2](#Thmdefinition2 "Definition 2 (Essential and compressible tokens). ‣ A.3 Difficulty-Adaptive Compression ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
is based on the probability of the *sampled* token $`y_{t}`$, whereas
the KL divergence $`D_{\mathrm{KL}}(q_{t}\|p_{t})`$ is an expectation
over the *entire vocabulary* at position $`t`$. Rather, (A3) is a
structural modeling assumption, motivated by the intuition that at
positions where the teacher concentrates mass differently from the
student (compressible positions), the full distributional divergence
tends to be larger than at positions where both distributions agree on
the dominant token (essential positions).

### A.4 Bounded Forgetting from the Base Model

A central advantage of on-policy self-distillation over off-policy SFT
is controlled divergence from the original model. We formalize this
through the *conciseness gap*.

###### Definition 3 (Conciseness gap).

The conciseness gap of the base model $`\pi_{\theta_{0}}`$ under
instruction $`c`$ on input $`x`$ is

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\gamma(x)=d_{\mathrm{TV}}\big(\pi_{\theta_{0}}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x,c)\big).
``` |  | (28) |

###### Proposition 2 (Bounded forgetting under on-policy self-distillation).

Consider the first teacher window where $`\bar{\theta}=\theta_{0}`$
(frozen teacher). If the converged CRISP loss satisfies
$`\mathcal{L}(\theta^{*})\leq\epsilon_{\mathrm{KL}}`$, then:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbb{E}_{x\sim\mathcal{D}}\big[d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x)\big)\big]\leq\sqrt{\frac{\epsilon_{\mathrm{KL}}}{2}}+\mathbb{E}_{x\sim\mathcal{D}}[\gamma(x)].
``` |  | (29) |

For subsequent windows with periodic teacher update, the same bound
holds with $`\theta_{0}`$ replaced by the teacher weights
$`\bar{\theta}`$ at the start of that window. Moreover, $`\gamma(x)`$ is
difficulty-adaptive: for hard problems where the conciseness instruction
has little effect, $`\gamma(x)\approx 0`$, so forgetting is minimal.

###### Proof.

By the triangle inequality for total variation distance:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x)\big)\leq\underbrace{d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x,c)\big)}_{\text{student--teacher gap}}+\underbrace{d_{\mathrm{TV}}\big(\pi_{\theta_{0}}(\cdot\mid x,c),\;\pi_{\theta_{0}}(\cdot\mid x)\big)}_{\gamma(x)}.
``` |  | (30) |

The student–teacher gap is bounded via Pinsker’s inequality. By
Corollary [1](#Thmcorollary1 "Corollary 1. ‣ A.1 Sequence-Level Divergence and the Training Objective ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"),
$`\mathbb{E}_{x}[D_{\mathrm{KL}}(\pi_{\theta^{*}}(\cdot\mid x)\|\pi_{\theta_{0}}(\cdot\mid x,c))]\leq\epsilon_{\mathrm{KL}}`$.
Applying Pinsker to each $`x`$ and Jensen’s inequality over
$`x\sim\mathcal{D}`$:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbb{E}_{x}\big[d_{\mathrm{TV}}\big(\pi_{\theta^{*}}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x,c)\big)\big]\leq\sqrt{\tfrac{\epsilon_{\mathrm{KL}}}{2}}.
``` |  | (31) |

Taking expectations on both sides of the triangle inequality
yields ([29](#A1.E29 "In Proposition 2 (Bounded forgetting under on-policy self-distillation). ‣ A.4 Bounded Forgetting from the Base Model ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).

For the difficulty-adaptive claim: on hard problems, the conciseness
instruction cannot substantially alter the output distribution because
most reasoning steps are essential. Thus
$`\pi_{\theta_{0}}(\cdot\mid x,c)\approx\pi_{\theta_{0}}(\cdot\mid x)`$,
giving $`\gamma(x)\approx 0`$. ∎

###### Remark 4 (Comparison with off-policy SFT).

Standard off-policy SFT minimizes
$`-\mathbb{E}_{(x,y)\sim\mathcal{D}_{T}}[\log\pi_{\theta}(y\mid x)]`$ on
a fixed teacher dataset $`\mathcal{D}_{T}`$. The analogous forgetting
decomposition is:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
d_{\mathrm{TV}}\big(\pi_{\theta_{\mathrm{SFT}}}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x)\big)\leq d_{\mathrm{TV}}\big(\pi_{\theta_{\mathrm{SFT}}}(\cdot\mid x),\;\mathcal{D}_{T}(\cdot\mid x)\big)+d_{\mathrm{TV}}\big(\mathcal{D}_{T}(\cdot\mid x),\;\pi_{\theta_{0}}(\cdot\mid x)\big).
``` |  | (32) |

The second term measures the *distribution mismatch* between the
teacher’s data and the base model’s outputs, which can be substantially
larger than the conciseness gap $`\gamma(x)`$, particularly when the
teacher generates qualitatively different reasoning styles. Crucially,
off-policy SFT drives $`\pi_{\theta_{\mathrm{SFT}}}`$ toward
$`\mathcal{D}_{T}`$ directly, while on-policy distillation generates
data from the student’s own evolving distribution, inherently limiting
divergence from the base model \[[Shenfeld et al., 2026](#bib.bib26)\].

### A.5 Compression Reduces Compounding Error

We provide a simple probabilistic model that explains the most striking
empirical finding: shorter reasoning traces can *improve* accuracy
rather than degrade it.

###### Proposition 3 (Shorter traces reduce error accumulation).

Consider an autoregressive reasoning model where each token position
independently introduces a reasoning error (an incorrect intermediate
step that corrupts the final answer) with probability
$`p_{\mathrm{err}}\in(0,1)`$. For a trace of length $`L`$, the
probability of producing a correct answer is:

|     |                                           |     |      |
|-----|-------------------------------------------|-----|------|
|     |
       ``` math
       \mathrm{Acc}(L)=(1-p_{\mathrm{err}})^{L}.
       ```                                        |     | (33) |

If compression reduces the trace from $`L`$ to $`\alpha L`$ tokens
($`\alpha\in(0,1)`$) without increasing the per-token error rate, the
accuracy ratio satisfies:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\frac{\mathrm{Acc}(\alpha L)}{\mathrm{Acc}(L)}=(1-p_{\mathrm{err}})^{-(1-\alpha)L}\geq 1+(1-\alpha)L\cdot p_{\mathrm{err}}.
``` |  | (34) |

The accuracy improvement grows *exponentially* in the number of removed
tokens $`(1-\alpha)L`$.

###### Proof.

Direct computation gives the exact ratio:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\frac{\mathrm{Acc}(\alpha L)}{\mathrm{Acc}(L)}=\frac{(1-p_{\mathrm{err}})^{\alpha L}}{(1-p_{\mathrm{err}})^{L}}=(1-p_{\mathrm{err}})^{-(1-\alpha)L}.
``` |  | (35) |

Let $`m=(1-\alpha)L>0`$. Since $`\ln(1-p)\leq-p`$ for all $`p\in(0,1)`$
(which follows from the concavity of $`\ln`$ and the tangent line at
$`p=0`$), we have $`-\ln(1-p_\mathrm{err})\geq p_{\mathrm{err}}`$,
giving:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
(1-p_{\mathrm{err}})^{-m}=e^{-m\ln(1-p_\mathrm{err})}\geq e^{m\cdot p_{\mathrm{err}}}\geq 1+m\cdot p_{\mathrm{err}},
``` |  | (36) |

where the final inequality is $`e^{u}\geq 1+u`$ for all $`u\geq 0`$. ∎

###### Remark 5 (Quantitative interpretation).

On MATH-500 (Qwen3-14B), the base model generates $`L\approx 4{,}139`$
tokens and CRISP compresses to $`\alpha\approx 0.44`$, removing
$`(1-\alpha)L\approx 2{,}330`$ tokens. Even with a modest per-token
error rate of $`p_{\mathrm{err}}=10^{-4}`$, the linear lower bound gives
an accuracy ratio $`\geq 1.23`$, a 23% relative improvement. The
exponential term
$`(1-p_{\mathrm{err}})^{-2330}\approx e^{0.23}\approx 1.26`$ provides
the tighter bound. On AIME benchmarks where $`L\approx 13{,}000`$
tokens, the same per-token error rate yields even larger potential gains
per token removed.

###### Remark 6 (Conservative nature of the independence assumption).

The independence assumption is conservative: in practice, reasoning
errors compound because one incorrect intermediate step causes
subsequent steps to build on a false premise, amplifying the error.
Under positively correlated errors, the benefit of removing error-prone
tokens exceeds the independent-error prediction, making
Proposition [3](#Thmproposition3 "Proposition 3 (Shorter traces reduce error accumulation). ‣ A.5 Compression Reduces Compounding Error ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
a *lower bound* on the improvement from compression. This is consistent
with the empirical accuracy gains where the base model has room to
improve (e.g., $`71.3{\to}82.1`$ on MATH-500 for
DeepSeek-R1-Distill-Llama-8B;
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
which exceed what the simple independent-error model predicts.

## Appendix B Survey of Reasoning Compression Methods

Table [4](#A2.T4 "Table 4 ‣ Appendix B Survey of Reasoning Compression Methods ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
summarizes 13 reasoning compression methods along four axes. This survey
motivates the design of CRISP by revealing the pervasive dependence on
ground-truth answers and the rarity of difficulty-adaptive methods.

|                |     |     |     |     |              |
|----------------|-----|-----|-----|-----|--------------|
| Method         | LP  | DD  | CA  | HB  | Approach     |
| L1             | ✓   |     | ✓   | ✓   | RL           |
| DiPO           | ✓   | ✓   | ✓   |     | RL           |
| DIET           | ✓   | ✓   | ✓   |     | RL           |
| DLER           | ✓   | ✓   | ✓   |     | RL           |
| Leash          | ✓   |     | ✓   |     | RL           |
| SEER           |     |     | ✓   |     | SFT          |
| TokenSkip      |     |     | ✓   | ✓   | SFT          |
| S3-CoT         |     |     |     |     | Steering     |
| DAP/LiteCoT    |     | ✓   | ✓   |     | SFT          |
| Chain of Draft |     |     |     |     | Prompt       |
| TrimR          |     |     |     |     | Inference    |
| NoWait         |     |     |     |     | Inference    |
| FlowSteer      |     |     |     |     | Inference    |
| CRISP (Ours)   |     | ✓   |     |     | Self-distill |

Table 4: Survey of 13 reasoning compression methods. LP = Length Penalty
in reward; DD = Difficulty-Dependent; CA = Correct Answer required; HB =
Hard Budget.

##### Comparability.

We do not tabulate the reported accuracy/token-reduction numbers of
these methods side by side, because they are not directly comparable:
with the sole exception of NoWait, no cited method uses any of our base
models (Qwen3-8B/14B, DeepSeek-R1-Distill-Llama-8B), and papers differ
in benchmark splits (full MATH vs. MATH-500, composite AIME sets,
MMLU-Pro vs. MMLU, non-Diamond GPQA) and evaluation protocols. NoWait is
the only baseline evaluated on a CRISP base model (Qwen3-8B); as
reported in its paper, it reduces tokens but loses accuracy on the hard
sets (AIME 2025 $`74.6{\to}60.0`$), whereas CRISP compresses while
preserving accuracy.

## Appendix C Full Main Results

Table [5](#A3.T5 "Table 5 ‣ Appendix C Full Main Results ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
gives the complete version of
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"):
it adds the average response length (Len) for every row and the
inference-only “concise prompt” baselines, where the conciseness
instruction is prepended at inference with no training. The
concise-prompt rows show that prompting alone already shortens
responses, and that CRISP extends this compression further while better
preserving accuracy.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | MATH-500 |  |  | AIME 2024 |  |  | AIME 2025 |  |  | GPQA-D |  |  | MMLU |  |  |
| Method | Acc | Len | Red. | Acc | Len | Red. | Acc | Len | Red. | Acc | Len | Red. | Acc | Len | Red. |
| Qwen3-8B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Base Model | 95.7 | 4,884 | — | 76.2 | 14,229 | — | 70.4 | 16,492 | — | 61.5 | 7,921 | — | 81.9 | 1,633 | — |
| Concise prompt (v2) | 94.2 | 3,833 | 21.5% | 74.6 | 12,837 | 9.8% | 63.7 | 15,622 | 5.3% | 59.5 | 5,490 | 30.7% | 82.8 | 1,195 | 26.8% |
| Concise prompt (v1) | 95.6 | 2,983 | 38.9% | 74.2 | 11,352 | 20.2% | 62.1 | 14,194 | 13.9% | 56.8 | 5,582 | 29.5% | 83.0 | 1,195 | 26.8% |
| CRISP (v2) | 95.7 | 3,339 | 31.6% | 75.0 | 11,799 | 17.1% | 65.8 | 13,605 | 17.5% | 58.3 | 6,558 | 17.2% | 81.2 | 1,267 | 22.4% |
| CRISP (v1) | 95.7 | 2,107 | 56.9% | 72.9 | 9,549 | 32.9% | 58.8 | 11,804 | 28.4% | 58.5 | 5,056 | 36.2% | 80.9 | 903 | 44.7% |
| Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Base Model | 93.0 | 4,139 | — | 75.0 | 13,222 | — | 69.2 | 15,622 | — | 62.2 | 6,185 | — | 85.1 | 1,030 | — |
| Concise prompt (v2) | 94.3 | 3,077 | 25.7% | 73.3 | 11,509 | 13.0% | 71.7 | 14,038 | 10.1% | 60.5 | 4,579 | 26.0% | 84.9 | 801 | 22.2% |
| Concise prompt (v1) | 95.9 | 2,354 | 43.1% | 76.7 | 10,114 | 23.5% | 66.2 | 12,478 | 20.1% | 60.6 | 4,569 | 26.1% | 84.9 | 805 | 21.8% |
| CRISP (v2) | 95.2 | 2,701 | 34.7% | 75.0 | 10,615 | 19.7% | 67.1 | 12,990 | 16.8% | 62.0 | 4,905 | 20.7% | 83.9 | 799 | 22.4% |
| CRISP (v1) | 96.3 | 1,808 | 56.3% | 73.8 | 8,261 | 37.5% | 62.9 | 10,611 | 32.1% | 61.9 | 3,727 | 39.7% | 84.2 | 586 | 43.1% |
| DeepSeek-R1-Distill-Llama-8B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Base Model | 71.3 | 3,313 | — | 33.3 | 11,042 | — | 25.0 | 11,983 | — | 47.0 | 8,609 | — | 71.5 | 1,923 | — |
| Concise prompt (v2) | 79.7 | 2,635 | 20.5% | 42.1 | 10,766 | 2.5% | 28.8 | 11,522 | 3.8% | 46.0 | 7,800 | 9.4% | 73.9 | 1,746 | 9.2% |
| Concise prompt (v1) | 80.8 | 2,482 | 25.1% | 45.0 | 9,920 | 10.2% | 29.2 | 10,803 | 9.8% | 46.5 | 7,732 | 10.2% | 74.1 | 1,746 | 9.2% |
| CRISP (v2) | 79.8 | 2,544 | 23.2% | 42.1 | 11,317 | $`-`$2.5% | 26.2 | 11,975 | 0.1% | 46.7 | 8,006 | 7.0% | 71.4 | 1,703 | 11.4% |
| CRISP (v1) | 82.1 | 2,267 | 31.6% | 39.2 | 10,351 | 6.3% | 27.1 | 11,131 | 7.1% | 48.3 | 7,731 | 10.2% | 71.7 | 1,585 | 17.6% |

Table 5: Full main results (token budget = 30K). Accuracy (Acc, mean@8,
%), average reasoning token length (Len), and token reduction relative
to the base model (Red., %; “—” marks the base reference). “Concise
prompt” uses the conciseness instruction at inference only (no
training); CRISP trains with periodic teacher update ($`M{=}50`$). v1
and v2 denote the uniform and difficulty-aware conciseness instructions.

## Appendix D Answer-Format Breakdown

Because a grader keyed to a single answer format can undercount
accuracy, we score every response three ways
(Table [6](#A4.T6 "Table 6 ‣ Appendix D Answer-Format Breakdown ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")):
*answer-only* (an extracted “Answer: $`X`$” line), *boxed-only* (a
literal “$`\backslash`$boxed{$`\cdot`$}”), and *dual* (correct under
either; the scorer used throughout,
Section [5.1](#S5.SS1 "5.1 Experimental Setting ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
The base model splits correct answers across both formats (MATH-500:
57.7% answer-only, 54.5% boxed-only, 93.0% dual), so no single-format
grader captures its accuracy. After CRISP the model consolidates onto
the “Answer:” format: answer-only accuracy rises to 92.8% while
boxed-only drops to 10.0%, yet dual accuracy still improves to 95.2%.
Compression thus removes the redundant post-\</think\> boxed restatement
rather than correctness. The single-format columns sum to more than dual
because some responses are correct in both formats; the dual scorer
counts this overlap once, i.e.
$`\text{dual}=\text{answer-only}+\text{boxed-only}-\text{both}`$. This
overlap shrinks from 19.2% (base) to 7.5% (CRISP) on MATH-500 as the
formats become disjoint, which explains why a boxed-only grader
understates accuracy for models that answer in the “Answer:” format.

|            |           |             |            |      |
|------------|-----------|-------------|------------|------|
| Model      | Benchmark | Answer-only | Boxed-only | Dual |
| Base Model | MATH-500  | 57.7        | 54.5       | 93.0 |
|            | AIME 2024 | 20.8        | 70.8       | 75.0 |
|            | AIME 2025 | 17.5        | 65.8       | 69.2 |
| CRISP      | MATH-500  | 92.8        | 10.0       | 95.2 |
|            | AIME 2024 | 49.2        | 36.7       | 75.0 |
|            | AIME 2025 | 37.5        | 34.6       | 67.1 |

Table 6: Answer-format breakdown of accuracy (Qwen3-14B, 30K budget,
mean@8 %). Each response is scored three ways: answer-only (“Answer:
$`X`$”), boxed-only (“$`\backslash`$boxed{$`\cdot`$}”), and dual
(either). CRISP consolidates correct answers onto the “Answer:” format,
so answer-only accuracy rises sharply while boxed-only drops, without
losing dual accuracy. The dual column matches
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

## Appendix E Training and Implementation Details

##### Technical setup.

All experiments are conducted on a single node equipped with eight
NVIDIA H200 GPUs. Our implementation is built on top of the verl
library [Sheng et al. \[2025\]](#bib.bib27), which provides a
HybridEngine for efficient actor–rollout–reference model co-location. We
use PyTorch Fully Sharded Data Parallel (FSDP) for distributed training
with parameter and optimizer offloading to CPU, and SGLang [Zheng et al.
\[2024\]](#bib.bib39) for batched rollout generation. Sequence
parallelism (Ulysses, degree 4) is enabled during training to handle
long sequences efficiently, while tensor parallelism (degree 2) is used
for inference. Mixed-precision training is performed in bfloat16, and
gradient checkpointing is enabled to reduce peak memory usage.

##### Training data.

Our training data is derived from DAPO-Math-17k [Yu et al.
\[2025\]](#bib.bib37), a deduplicated set of $`{\sim}17{,}000`$
competition-level math problems. We randomly split the dataset into 80%
training ($`{\sim}13{,}600`$ prompts) and 20% validation
($`{\sim}3{,}400`$ prompts) with a fixed seed for reproducibility across
all configurations. For each problem, we construct a *student prompt*
(the original question) and a *teacher prompt* (the question prepended
with a conciseness instruction).

##### Training procedure.

At each training step, the student model generates a response from the
student prompt via SGLang sampling (temperature 1.0, top-$`p`$ 1.0). We
then perform a single gradient update minimizing the reverse KL
divergence between student and teacher logit distributions over the
student’s own generated tokens. All student rollouts are used for
training regardless of correctness; no filtering is applied. Both
teacher and student forward passes are performed for each micro-batch
with chunked logit processing (chunk size 256 tokens) to bound peak GPU
memory; teacher logits are progressively freed after each chunk.

##### Hyperparameters.

Table [7](#A5.T7 "Table 7 ‣ Hyperparameters. ‣ Appendix E Training and Implementation Details ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
summarizes the full configuration, which is shared across all models and
instruction variants.

|  |  |
|----|----|
| Parameter | Value |
| General |  |
| Models | Qwen3-8B, Qwen3-14B, DeepSeek-R1-Distill-Llama-8B |
| Loss function | Reverse KL: $`\mathrm{KL}(\pi_{\text{student}}\|\pi_{\text{teacher}})`$ |
| Teacher | Periodic update ($`M{=}50`$ steps) |
| Data |  |
| Training prompts | $`{\sim}`$13,600 (from DAPO-Math-17k) |
| Validation prompts | $`{\sim}`$3,400 |
| Max prompt length | 1,024 tokens |
| Max response length | 8,192 tokens (training) |
| Generation (student rollout) |  |
| Inference engine | SGLang |
| Temperature | 1.0 |
| Top-$`p`$ | 1.0 |
| Rollouts per prompt | 1 |
| Max generation tokens | 9,216 |
| Evaluation |  |
| Temperature | 0.6 |
| Top-$`p`$ | 0.95 |
| Top-$`k`$ | 20 |
| Rollouts per prompt | 8 |
| Max generation tokens | 30,000 |
| Eval frequency | Every 20 steps |
| Training |  |
| Optimizer | AdamW |
| Learning rate | $`1\times 10^{-6}`$ (constant) |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 (max norm) |
| Global batch size | 32 |
| Micro-batch size per GPU | 2 |
| Epochs | 1 |
| Precision | bfloat16 |
| Infrastructure |  |
| GPUs | $`8\times`$ NVIDIA H200 |
| Tensor parallelism (inference) | 2 |
| Sequence parallelism (training) | Ulysses, degree 4 |
| FSDP parameter offload | Enabled |
| FSDP optimizer offload | Enabled |
| Gradient checkpointing | Enabled |

Table 7: Hyperparameters for CRISP. The same configuration is used
across all models and instruction variants.

##### Evaluation.

We evaluate every 20 steps on three held-out math benchmarks:
MATH-500 [Hendrycks et al. \[2021\]](#bib.bib15), AIME 2024, and
AIME 2025. For each benchmark, we generate 8 responses per problem with
temperature 0.6, top-$`p=0.95`$, and top-$`k=20`$, and report mean
accuracy (fraction of correct samples averaged over problems) and
average response token count. Correctness is determined by extracting
the final answer and comparing against the ground truth using the
symbolic and numeric equivalence checker from the verl library \[[Sheng
et al., 2025](#bib.bib27)\].³³ 3
[https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py)

## Appendix F Alternative Teacher Parameterizations

The main paper uses a *periodic teacher update*
($`\bar{\theta}\leftarrow\theta`$ every $`M`$ steps) that balances
progressive compression with training stability. Here we discuss the
design space of teacher parameterizations, from the most conservative to
the most aggressive. An empirical comparison across update intervals
$`M\in\{1,10,30,50,100\}`$ is provided in
Section [5.3.3](#S5.SS3.SSS3 "5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").

##### Frozen teacher ($`M=\infty`$).

The simplest variant fixes $`\bar{\theta}=\theta_{0}`$ for the entire
training run. This provides a stable, non-shifting compression target
and allows teacher prefill to be pre-computed once. However, the frozen
teacher becomes an increasingly weak compression oracle as the student
improves, limiting the maximum achievable compression.

##### Periodic teacher (our default, $`M{=}50`$).

Setting a finite update interval $`M`$
(Algorithm [1](#algorithm1 "In 3.4 Training Algorithm ‣ 3 Method ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
enables progressive compression: after each refresh, the updated
teacher, having already internalized compression from the previous
round, produces even more concise traces under instruction $`c`$,
providing a stronger compression signal. The discrete refresh avoids
continuous co-adaptation while still allowing compression to deepen over
training. Our ablation
(Section [5.3.3](#S5.SS3.SSS3 "5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
shows that $`M\in\{30,50\}`$ form a stable plateau, retaining base-level
accuracy while still compressing substantially, confirming robustness to
the exact interval within this range.

##### EMA teacher.

The teacher parameters are an exponential moving average of the student,
updated after each gradient step:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\bar{\theta}\leftarrow\alpha\bar{\theta}+(1-\alpha)\theta,\quad\alpha\in[0.99,0.999].
``` |  | (37) |

This provides a smooth, continuous version of progressive compression.
With moderate decay ($`\alpha=0.995`$), it can yield additional
compression beyond the frozen teacher but requires careful monitoring
for collapse.

##### Stop-gradient concurrent teacher ($`M{=}1`$).

The teacher uses the *same* parameters $`\theta`$ as the student, with
stop-gradient during the backward pass. This provides the most
aggressive progressive compression but carries the highest risk of
*progressive compression collapse*: as the student becomes more concise,
the teacher also becomes more concise, creating a positive feedback loop
that can drive output length toward degenerate short sequences. Our
ablation confirms this prediction: $`M{=}1`$ causes training collapse,
with accuracy falling sharply and response length growing rather than
shrinking
(Figure [6](#S5.F6 "Figure 6 ‣ 5.3.3 How Sensitive Is Compression to the Teacher Update Interval? ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")),
consistent with the moving-target instability identified by [Shenfeld
et al. \[2026\]](#bib.bib26).

## Appendix G Effect of KL Divergence Direction in CRISP

### G.1 Background and Motivation

The CRISP distillation loss aligns the live student $`p_{S}`$ with the
teacher $`p_{T}`$, a periodically frozen snapshot of the student itself.
A natural question is which direction of KL divergence to use:

- •
  Reverse KL (our choice):
  $`\mathrm{KL}(p_{S}\|p_{T})=\sum_{v}p_{S}(v)\bigl[\log p_{S}(v)-\log p_{T}(v)\bigr]`$.
  The gradient w.r.t. student parameters is weighted by the *student’s
  own* distribution $`p_{S}(v)`$: the student updates only in regions
  where it currently generates, providing built-in self-regularization
  against abrupt distribution shifts.
- •
  Forward KL (baseline):
  $`\mathrm{KL}(p_{T}\|p_{S})=\sum_{v}p_{T}(v)\bigl[\log p_{T}(v)-\log p_{S}(v)\bigr]`$.
  The gradient is weighted by the *teacher’s* distribution $`p_{T}(v)`$,
  fully decoupled from the student’s current state.
- •
  Jensen–Shannon divergence (JSD): the symmetric mixture
  $`\tfrac{1}{2}\mathrm{KL}(p_{S}\|m)+\tfrac{1}{2}\mathrm{KL}(p_{T}\|m)`$
  with $`m=\tfrac{1}{2}(p_{S}+p_{T})`$. It interpolates between the two
  directions, so its gradient is partly student-weighted and partly
  teacher-weighted.

Forward KL is often used in offline distillation, where the teacher is a
fixed external model whose distribution is a reliable target to match.
In CRISP the teacher is instead a stale copy of the student, refreshed
every $`M`$ steps. We argue that this makes forward KL ill-suited for
CRISP: because its gradient is weighted by $`p_{T}`$ rather than
$`p_{S}`$, an update can move probability mass into regions the current
student never visits, with a magnitude independent of how far the
student has drifted from the teacher.

### G.2 Experimental Setup

We compare three divergences on Qwen3-8B: reverse KL, forward KL, and
their symmetric combination, the Jensen–Shannon divergence (JSD). All
three use the full-vocab setup of
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
with the teacher refreshed every 50 steps, and differ only in the
divergence. We track validation accuracy (mean@8) and mean response
length on MATH-500 and AIME 2024 every 20 steps.

### G.3 Results

Reverse KL and JSD train stably and hold near-base accuracy with strong
compression, whereas forward KL collapses
(Table [8](#A7.T8 "Table 8 ‣ G.3 Results ‣ Appendix G Effect of KL Divergence Direction in CRISP ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")):
its accuracy falls to single digits and its response length saturates
the 30K-token budget. JSD compresses slightly less than reverse KL, so
we adopt reverse KL as the default.

|                   |          |        |           |        |
|-------------------|----------|--------|-----------|--------|
|                   | MATH-500 |        | AIME 2024 |        |
| Divergence        | Acc      | Len    | Acc       | Len    |
| Reverse KL (ours) | 95.7     | 3,339  | 75.0      | 11,799 |
| JSD               | 94.8     | 3,281  | 71.7      | 11,856 |
| Forward KL        | 5.7      | 29,135 | 0.4       | 30,000 |

Table 8: Divergence direction ablation (Qwen3-8B, step 99, 30K budget).
Accuracy (mean@8, %) and average response length (tokens) on MATH-500
and AIME 2024. Forward KL collapses to degenerate maximum-length
generation.

![Refer to caption](2603.05433v7/figures/kl_comparison.png)

Figure 9: Validation accuracy versus training step on Qwen3-8B for
reverse KL (blue), JSD (green), and forward KL (red), on a, MATH-500 and
b, AIME 2024. Reverse KL and JSD hold near-base accuracy; forward KL
collapses to near zero within the first $`{\sim}100`$ steps.

![Refer to caption](2603.05433v7/figures/kl_tokens.png)

Figure 10: Mean response length versus training step on Qwen3-8B for the
same three divergences, on a, MATH-500 and b, AIME 2024. Reverse KL and
JSD compress responses smoothly, whereas forward KL diverges to the
30K-token cap (degenerate maximum-length generation).

The two failure modes are coupled.
Figure [9](#A7.F9 "Figure 9 ‣ G.3 Results ‣ Appendix G Effect of KL Divergence Direction in CRISP ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
shows the accuracy collapse over the first 100 steps: reverse KL and JSD
stay near base accuracy, while forward KL tracks them briefly and then
falls to single digits on MATH-500 and to near zero on AIME 2024 by
step 100.
Figure [10](#A7.F10 "Figure 10 ‣ G.3 Results ‣ Appendix G Effect of KL Divergence Direction in CRISP ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
shows the cause: forward KL does not compress but instead drives
response length up toward the 30K-token budget on both benchmarks, so
the policy degenerates into maximum-length generation, whereas reverse
KL and JSD shorten responses smoothly over the same interval.

### G.4 Mechanism

The contrast follows from the gradient. Under forward KL,
$`\nabla_{\log p_{S}}\,\mathrm{KL}(p_{T}\|p_{S})=-p_{T}(v)`$, so updates
are weighted by the teacher’s distribution regardless of the student’s
current state. When the teacher is a stale copy of the student, these
updates repeatedly push mass into regions the current student does not
cover, and the drift compounds until the policy runs to maximum length.
Under reverse KL,
$`\nabla_{\theta}\,\mathrm{KL}(p_{S}\|p_{T})=\mathbb{E}_{v\sim p_{S}}\!\left[\nabla_{\theta}\log p_{S}(v)\,\big(\log p_{S}(v)-\log p_{T}(v)+1\big)\right]`$,
so updates are weighted by the student’s own distribution $`p_{S}`$.
Because the student already covers the teacher’s high-probability modes,
each refresh requires only a small adjustment, which keeps training
stable. JSD inherits enough of this student-weighted gradient to remain
stable, but its forward-KL component makes it compress slightly less
than pure reverse KL. We therefore use reverse KL for all CRISP
experiments.

## Appendix H Entropy Preservation During Training

A recurring failure mode of length-penalized RL is *entropy collapse*:
as the policy is pushed toward shorter outputs it also becomes
overconfident, its next-token distribution sharpens, and generation
diversity is lost \[[Cui et al., 2025](#bib.bib9), [Wang et al.,
2025b](#bib.bib31), [Xu et al., 2026](#bib.bib2)\]. Because CRISP also
shortens reasoning, a natural concern is whether it induces the same
collapse. It does not.

We track the student’s policy entropy directly during training: at every
step we compute the mean full-vocab per-token Shannon entropy of the
student’s next-token distribution, averaged over the response tokens of
the on-policy rollout (computed with a tensor-parallel entropy reduction
over the sharded vocabulary; this is a logging-only quantity and does
not enter the loss).
Figure [11](#A8.F11 "Figure 11 ‣ Appendix H Entropy Preservation During Training ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
plots this entropy over the first 99 training steps for Qwen3-8B and
Qwen3-14B under the default recipe (v2 teacher, $`M{=}50`$, reverse KL).

Figure 11: CRISP preserves policy entropy while compressing reasoning.
Mean full-vocab per-token student entropy over training for Qwen3-8B
(solid) and Qwen3-14B (dashed); dotted lines mark each model’s step-0
(base) entropy. Entropy fluctuates within a narrow band around its
starting value and shows no downward trend over the run, so the
compression gains in
Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
do not come at the cost of a collapsed, overconfident policy.

For both models the entropy stays within a narrow band around its
base-model value for the entire run: Qwen3-8B moves from $`0.33`$ at
step 0 to $`0.29`$ at step 99 (range $`{\sim}0.27`$–$`0.34`$), and
Qwen3-14B from $`0.33`$ to $`0.32`$ (range $`{\sim}0.28`$–$`0.34`$).
There is no monotone decline toward zero. This is consistent with the
mode-seeking reverse-KL analysis
(Appendix [G](#A7 "Appendix G Effect of KL Divergence Direction in CRISP ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation"))
and the bounded-forgetting result
(Proposition [2](#Thmproposition2 "Proposition 2 (Bounded forgetting under on-policy self-distillation). ‣ A.4 Bounded Forgetting from the Base Model ‣ Appendix A Theoretical Analysis ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")):
because the teacher is the model’s own concise mode rather than an
external overconfident target, distilling into it removes redundant
tokens without sharpening the retained distribution. CRISP therefore
compresses reasoning *without* the entropy collapse that
length-penalized RL is prone to.

## Appendix I Deep Planning Agentic Task

Beyond mathematical reasoning, we evaluate whether our compressed models
retain their capability on complex, multi-step agentic tasks that
require tool calling, constraint satisfaction, and long-horizon
planning. We use the DeepPlanning benchmark \[[Zhang et al.,
2026](#bib.bib3)\], which evaluates LLM agents across two domains:

##### Travel Planning.

The agent is given a natural-language travel request (e.g., “Plan a
3-day trip to Beijing for two people with a \$2,000 budget”) and must
produce a complete itinerary by issuing tool calls to query flights,
trains, hotels, restaurants, and attractions. The benchmark contains 240
test cases (120 Chinese + 120 English). Evaluation checks both *hard
constraints* (budget, dates, party size) and *commonsense constraints*
(route consistency, business hours, activity diversity) across eight
dimensions. We report the composite score, a weighted average of
commonsense and personalization scores.

##### Shopping Planning.

The agent must fulfill a multi-item shopping request (e.g., “Find
running shoes under \$100, a matching sports watch, and apply any
available coupons”) by navigating a simulated e-commerce environment
with tools for product search, filtering, cart management, and coupon
application. The benchmark contains 150 test cases across three
difficulty levels. We report the match rate, the fraction of expected
products correctly placed in the cart.

##### Setup.

We evaluate Qwen3-14B checkpoints from our training (teacher update
frequency $`M{=}40`$, reverse KL loss) on 100 travel planning samples,
evaluating every 10 steps. Each checkpoint is served via vLLM and paired
with the DeepPlanning function-calling agent, which iteratively invokes
the LLM and executes tool calls until a final plan is produced (up to
400 LLM calls per case). We run the base model (step 0) 10 times to
establish a variance estimate.

##### Results.

Figure
[12](#A9.F12 "Figure 12 ‣ Results. ‣ Appendix I Deep Planning Agentic Task ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")
shows the trade-off between response length and planning quality. CRISP
training progressively compresses average LLM response tokens: shopping
tokens decrease from 12.5k (base) to 6.2k (step 120), a 51% reduction;
travel tokens decrease from 5.3k to 3.1k, a 42% reduction. Despite this
substantial compression, planning accuracy remains largely preserved:
shopping match rate stays within the baseline variance through step 70
(25.8% vs. 24.2% base), and travel composite score is stable across all
checkpoints (17.0–19.1% vs. 17.7% base). This demonstrates that our
reasoning compression method transfers beyond mathematical reasoning to
multi-step agentic planning, reducing token consumption without
degrading the model’s ability to orchestrate complex tool-calling
sequences.

Figure 12: Mean response length on training set travel planning and
validation set shopping planning. *Shopping match rate* measures the
fraction of ground-truth products correctly placed in the agent’s cart
(higher is better). *Travel composite score* is the average of a
commonsense score (route consistency, time feasibility, business hours,
etc., across 8 dimensions) and a personalization score (satisfaction of
user-specified hard constraints such as budget, dates, and party size),
averaged over Chinese and English test sets (higher is better).

## Appendix J Answer-Format Consolidation: Qualitative Examples

The prompt asks for a final line of the form “Answer: $`X`$”, but the
base Qwen3 models often ignore this and instead present the final result
only inside a “$`\backslash`$boxed{$`\cdot`$}” expression after
\</think\>, following their post-training convention. A strict
“Answer:”-format grader scores such responses as wrong even when the
boxed value is correct
(Appendix [D](#A4 "Appendix D Answer-Format Breakdown ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation")).
CRISP removes this failure mode: the compressed models consolidate onto
the requested “Answer:” format. The two examples below show the base
model ending with a boxed value and no “Answer:” line, while both CRISP
variants (v1 and v2) emit the “Answer:” line, in addition to being much
shorter.

Problem 1 (MATH-500, intermediate algebra): *Let $`a,b,c`$ be real with
$`|ax^{2}+bx+c|\leq 1`$ for all $`0\leq x\leq 1`$. Find the largest
possible value of $`|a|+|b|+|c|`$.*  (Correct answer: 17)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEwLkYxMy5waWMxIiBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMjYxLjQzIiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDI2MS40MyIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwyNjEuNDMpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQkZCRkJGOyIgZmlsbD0iI0JGQkZCRiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuOTEgTCAwIDI1NS41MyBDIDAgMjU4Ljc5IDIuNjQgMjYxLjQzIDUuOTEgMjYxLjQzIEwgNDcxLjQ3IDI2MS40MyBDIDQ3NC43MyAyNjEuNDMgNDc3LjM4IDI1OC43OSA0NzcuMzggMjU1LjUzIEwgNDc3LjM4IDUuOTEgQyA0NzcuMzggMi42NCA0NzQuNzMgMCA0NzEuNDcgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGOUY5Rjk7IiBmaWxsPSIjRjlGOUY5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMjA3LjI3IEwgNDc1LjQxIDIwNy4yNyBMIDQ3NS40MSA1LjkxIEMgNDc1LjQxIDMuNzMgNDczLjY1IDEuOTcgNDcxLjQ3IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDIxNS41OSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNi40OGVtOy0tbHR4LWZvLWhlaWdodDoyLjg5ZW07LS1sdHgtZm8tZGVwdGg6MC4xOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0Mi4zNiIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzkuOTQpIiB3aWR0aD0iNTA0Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNi40OGVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+UXdlbjMtOEIgYmFzZSAoMTAsNjc3IHRva2Vucyk8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTAuMDYgMjcuNykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDo0MS40NmVtOy0tbHR4LWZvLWhlaWdodDoxMi40OWVtOy0tbHR4LWZvLWRlcHRoOjEuMzhlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTkxLjg5IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNzIuODYpIiB3aWR0aD0iNTczLjY5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMS4yIiBjbGFzcz0ibHR4X2lubGluZS1sb2dpY2FsLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6NDEuNDZlbTsiPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAyIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAyLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMS5wMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0O3RoaW5rJmd0OzxzcGFuIGlkPSJBMTAuRjEzLnBpYzEucDIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEucDIuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAyLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojODA4MDgwOyI+4oCmPHNwYW4gaWQ9IkExMC5GMTMucGljMS5wMi4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPmV4dGVuc2l2ZSBzZWFyY2ggb3ZlciB0aGUgQ2hlYnlzaGV2IHBvbHlub21pYWwgPG1hdGggaWQ9IkExMC5GMTMucGljMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI4eF57Mn0tOHgrMSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93Pjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tc3VwPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPuKIkjwvbW8+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+eDwvbWk+PC9tcm93PjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+OHheezJ9LTh4KzE8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgd2l0aCByZXBlYXRlZCB2ZXJpZmljYXRpb24gdGhhdCBubyBvdGhlciBjaG9pY2UgZXhjZWVkcyB0aGUgYm91bmTigKY8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljMS5wMi4zIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEucDIuMy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDsvdGhpbmsmZ3Q7PHNwYW4gaWQ9IkExMC5GMTMucGljMS5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij48c3BhbiBjbGFzcz0ibHR4X3J1bGUiIHN0eWxlPSJ3aWR0aDozMzAuNXB0O2hlaWdodDowLjNwdDstLWx0eC1iZy1jb2xvcjpibGFjaztkaXNwbGF5OmlubGluZS1ibG9jazsiPsKgPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEucDMiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzEucDMuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAzLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+VGh1cyB0aGUgbWF4aW11bSBvZiA8bWF0aCBpZD0iQTEwLkYxMy5waWMxLm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9InxhfCt8YnwrfGN8IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmI8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+fDwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5jPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+fDwvbW8+PC9tcm93PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPnxhfCt8YnwrfGN8PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gaXM8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkV4MSIgY2xhc3M9Imx0eF9lcXVhdGlvbiBsdHhfZXFuX3RhYmxlIj4KCjxzcGFuPjxzcGFuIGNsYXNzPSJsdHhfZXF1YXRpb24gbHR4X2Vxbl9yb3cgbHR4X2FsaWduX2Jhc2VsaW5lIj4KPHNwYW4gY2xhc3M9Imx0eF9lcW5fY2VsbCBsdHhfZXFuX2NlbnRlcl9wYWRsZWZ0Ij48L3NwYW4+CjxzcGFuIGNsYXNzPSJsdHhfZXFuX2NlbGwgbHR4X2FsaWduX2NlbnRlciI+PG1hdGggaWQ9IkExMC5FeDEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJveGVkezE3fSIgZGlzcGxheT0iYmxvY2siIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1lbmNsb3NlIG5vdGF0aW9uPSJib3giPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjE3PC9tbj48L21lbmNsb3NlPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGJveGVkezE3fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPgo8c3BhbiBjbGFzcz0ibHR4X2Vxbl9jZWxsIGx0eF9lcW5fY2VudGVyX3BhZHJpZ2h0Ij48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAzLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMS5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPk5vIOKAnEFuc3dlcjrigJ0gbGluZTsgdGhlIHJlc3VsdCBhcHBlYXJzIG9ubHkgaW5zaWRlIDxzcGFuIGlkPSJBMTAuRjEzLnBpYzEucDMuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5cYm94ZWR7fTwvc3Bhbj4uPHNwYW4gaWQ9IkExMC5GMTMucGljMS5wMy4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiA8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAzLjIuMS4yLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MC4wcHQ7Ij7igIQ8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAzLjIuMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPuKclzwvc3Bhbj7igIQ8c3BhbiBpZD0iQTEwLkYxMy5waWMxLnAzLjIuMS4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPihzdHJpY3QgZm9ybWF0KTwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEwLkYxMy5waWMyIiBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMjEyLjM3IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDIxMi4zNyIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwyMTIuMzcpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA4QzAwOyIgZmlsbD0iIzAwOEMwMCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuOTEgTCAwIDIwNi40NiBDIDAgMjA5LjczIDIuNjQgMjEyLjM3IDUuOTEgMjEyLjM3IEwgNDcxLjQ3IDIxMi4zNyBDIDQ3NC43MyAyMTIuMzcgNDc3LjM4IDIwOS43MyA0NzcuMzggMjA2LjQ2IEwgNDc3LjM4IDUuOTEgQyA0NzcuMzggMi42NCA0NzQuNzMgMCA0NzEuNDcgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNFOEY1RTk7IiBmaWxsPSIjRThGNUU5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMTU4LjI4IEwgNDc1LjQxIDE1OC4yOCBMIDQ3NS40MSA1LjkxIEMgNDc1LjQxIDMuNzMgNDczLjY1IDEuOTcgNDcxLjQ3IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDE2Ni42MSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNi40OGVtOy0tbHR4LWZvLWhlaWdodDoyLjg4ZW07LS1sdHgtZm8tZGVwdGg6MC4xOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0Mi4yOCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzkuODUpIiB3aWR0aD0iNTA0Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzIuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNi40OGVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzIuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+Q1JJU1DCoHYyICg3LDg5MyB0b2tlbnMpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTAuMzNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE0Mi45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTQyLjkxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzIuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IkExMC5GMTMucGljMi5wMiIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkExMC5GMTMucGljMi5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzIucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMi5wMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBwb2x5bm9taWFsIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzIubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOHheezJ9LTh4KzEiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1zdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj54PC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjI8L21uPjwvbXN1cD48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7iiJI8L21vPjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjwvbXJvdz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPjh4XnsyfS04eCsxPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gKGEgc2NhbGVkIENoZWJ5c2hldiBwb2x5bm9taWFsKSBtZWV0cyB0aGUgY29uc3RyYWludCBhbmQgbWF4aW1pemVzIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzIubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ifGF8K3xifCt8Y3w9OCs4KzEiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+fDwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5hPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+fDwvbW8+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YjwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmM8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48L21yb3c+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE8L21uPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij58YXwrfGJ8K3xjfD04KzgrMTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+Ljwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzIucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAyLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7L3RoaW5rJmd0OzxzcGFuIGlkPSJBMTAuRjEzLnBpYzIucDIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljMi5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAzIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAzLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMi5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBtYXhpbXVtIGlzIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzIubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iOCs4KzEiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+OCs4KzE8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAzLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMi5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkFuc3dlcjogPG1hdGggaWQ9IkExMC5GMTMucGljMi5tNCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcYm9sZHN5bWJvbHsxN30iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7IiBtYXRoY29sb3I9IiMwMDgwMDAiPvCdn4/wnZ+VPC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxib2xkc3ltYm9sezE3fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IDxzcGFuIGlkPSJBMTAuRjEzLnBpYzIucDMuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MC4wcHQ7Ij7igIQ8c3BhbiBpZD0iQTEwLkYxMy5waWMyLnAzLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7Ij7inJM8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEwLkYxMy5waWMzIiBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMjEyLjM3IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDIxMi4zNyIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwyMTIuMzcpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojOTk0RDAwOyIgZmlsbD0iIzk5NEQwMCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuOTEgTCAwIDIwNi40NiBDIDAgMjA5LjczIDIuNjQgMjEyLjM3IDUuOTEgMjEyLjM3IEwgNDcxLjQ3IDIxMi4zNyBDIDQ3NC43MyAyMTIuMzcgNDc3LjM4IDIwOS43MyA0NzcuMzggMjA2LjQ2IEwgNDc3LjM4IDUuOTEgQyA0NzcuMzggMi42NCA0NzQuNzMgMCA0NzEuNDcgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGRkZCRjc7IiBmaWxsPSIjRkZGQkY3IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMTU4LjI4IEwgNDc1LjQxIDE1OC4yOCBMIDQ3NS40MSA1LjkxIEMgNDc1LjQxIDMuNzMgNDczLjY1IDEuOTcgNDcxLjQ3IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDE2Ni42MSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNi40OGVtOy0tbHR4LWZvLWhlaWdodDoyLjg4ZW07LS1sdHgtZm8tZGVwdGg6MC4xOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0Mi4yOCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzkuODUpIiB3aWR0aD0iNTA0Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNi40OGVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+Q1JJU1DCoHYxICg0LDE0NCB0b2tlbnMpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6MTAuMzNlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjE0Mi45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTQyLjkxKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IkExMC5GMTMucGljMy5wMiIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkExMC5GMTMucGljMy5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljMy5wMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkNoZWJ5c2hldiBjaG9pY2UgPG1hdGggaWQ9IkExMC5GMTMucGljMy5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSI4eF57Mn0tOHgrMSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93Pjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tc3VwPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPuKIkjwvbW8+PG1yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj44PC9tbj48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+eDwvbWk+PC9tcm93PjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+OHheezJ9LTh4KzE8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiBzYXRpc2ZpZXMgPG1hdGggaWQ9IkExMC5GMTMucGljMy5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJ8YXheezJ9K2J4K2N8XGxlcSAxIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YTwvbWk+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjwvbW4+PC9tc3VwPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YjwvbWk+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPng8L21pPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YzwvbWk+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7iiaQ8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+fGF4XnsyfStieCtjfFxsZXEgMTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IG9uIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzMubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iWzAsMV0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+WzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4wPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPiw8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPl08L21vPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlswLDFdPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4sIGdpdmluZyA8bWF0aCBpZD0iQTEwLkYxMy5waWMzLm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9InxhfCt8YnwrfGN8PTgrOCsxPTE3IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+YTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0iZmFsc2UiPnw8L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPmI8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgc3RyZXRjaHk9ImZhbHNlIj58PC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+fDwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5jPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJmYWxzZSI+fDwvbW8+PC9tcm93PjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+ODwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjg8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4xPC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjE3PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij58YXwrfGJ8K3xjfD04KzgrMT0xNzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+Ljwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAyLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7L3RoaW5rJmd0OzxzcGFuIGlkPSJBMTAuRjEzLnBpYzMucDIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljMy5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAzIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAzLjEiIGNsYXNzPSJsdHhfcCI+PG1hdGggaWQ9IkExMC5GMTMucGljMy5tNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJ8YXwrfGJ8K3xjfD04KzgrMT0xNyIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93Pjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF4c2l6ZT0iMC43MDBlbSIgbWluc2l6ZT0iMC43MDBlbSIgc3RyZXRjaHk9InRydWUiPnw8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPmE8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF4c2l6ZT0iMC43MDBlbSIgbWluc2l6ZT0iMC43MDBlbSIgc3RyZXRjaHk9InRydWUiPnw8L21vPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4rPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1heHNpemU9IjAuNzAwZW0iIG1pbnNpemU9IjAuNzAwZW0iIHN0cmV0Y2h5PSJ0cnVlIj58PC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj5iPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1heHNpemU9IjAuNzAwZW0iIG1pbnNpemU9IjAuNzAwZW0iIHN0cmV0Y2h5PSJ0cnVlIj58PC9tbz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+KzwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXhzaXplPSIwLjcwMGVtIiBtaW5zaXplPSIwLjcwMGVtIiBzdHJldGNoeT0idHJ1ZSI+fDwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+YzwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXhzaXplPSIwLjcwMGVtIiBtaW5zaXplPSIwLjcwMGVtIiBzdHJldGNoeT0idHJ1ZSI+fDwvbW8+PC9tcm93PjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj49PC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj44PC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4rPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj44PC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4rPC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4xPC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+MTc8L21uPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPnxhfCt8YnwrfGN8PTgrOCsxPTE3PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAzLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Ljwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzMucDMuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWMzLnAzLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QW5zd2VyOiA8bWF0aCBpZD0iQTEwLkYxMy5waWMzLm02IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilxib2xkc3ltYm9sezE3fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiIG1hdGhjb2xvcj0iIzAwODAwMCI+8J2fj/Cdn5U8L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XGJvbGRzeW1ib2x7MTd9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gPHNwYW4gaWQ9IkExMC5GMTMucGljMy5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDowLjBwdDsiPuKAhDxzcGFuIGlkPSJBMTAuRjEzLnBpYzMucDMuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiPuKckzwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

Problem 2 (AIME 2024, geometry): *Tetrahedron $`ABCD`$ has
$`AB{=}CD{=}\sqrt{41}`$, $`AC{=}BD{=}\sqrt{80}`$,
$`BC{=}AD{=}\sqrt{89}`$. The equal face-distance from the interior point
can be written $`\tfrac{m\sqrt{n}}{p}`$; find $`m{+}n{+}p`$.*  (Correct
answer: 104)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEwLkYxMy5waWM0IiBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMjI4LjIyIiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDIyOC4yMiIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwyMjguMjIpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQkZCRkJGOyIgZmlsbD0iI0JGQkZCRiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuOTEgTCAwIDIyMi4zMiBDIDAgMjI1LjU4IDIuNjQgMjI4LjIyIDUuOTEgMjI4LjIyIEwgNDcxLjQ3IDIyOC4yMiBDIDQ3NC43MyAyMjguMjIgNDc3LjM4IDIyNS41OCA0NzcuMzggMjIyLjMyIEwgNDc3LjM4IDUuOTEgQyA0NzcuMzggMi42NCA0NzQuNzMgMCA0NzEuNDcgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGOUY5Rjk7IiBmaWxsPSIjRjlGOUY5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMTc0LjA2IEwgNDc1LjQxIDE3NC4wNiBMIDQ3NS40MSA1LjkxIEMgNDc1LjQxIDMuNzMgNDczLjY1IDEuOTcgNDcxLjQ3IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDE4Mi4zOCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNi40OGVtOy0tbHR4LWZvLWhlaWdodDoyLjg5ZW07LS1sdHgtZm8tZGVwdGg6MC4xOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0Mi4zNiIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzkuOTQpIiB3aWR0aD0iNTA0Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNi40OGVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+UXdlbjMtOEIgYmFzZSAoNyw5NzEgdG9rZW5zKTwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAyNy43KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjQxLjQ2ZW07LS1sdHgtZm8taGVpZ2h0OjEwLjA5ZW07LS1sdHgtZm8tZGVwdGg6MS4zOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIxNTguNjgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEzOS42NikiIHdpZHRoPSI1NzMuNjkiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iQTEwLkYxMy5waWM0LjIiIGNsYXNzPSJsdHhfaW5saW5lLWxvZ2ljYWwtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDo0MS40NmVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQucDIiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQucDIuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7dGhpbmsmZ3Q7PHNwYW4gaWQ9IkExMC5GMTMucGljNC5wMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNC5wMi4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQucDIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiM4MDgwODA7Ij7igKY8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAyLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Y29tcHV0ZXMgdGhlIHZvbHVtZSBhbmQgdG90YWwgZmFjZSBhcmVhLCBmaW5kcyBpbnJhZGl1cyA8bWF0aCBpZD0iQTEwLkYxMy5waWM0Lm0xIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilx0ZnJhY3syMFxzcXJ0ezIxfX17NjN9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIwPC9tbj48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtc3FydCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yMTwvbW4+PC9tc3FydD48L21yb3c+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj42MzwvbW4+PC9tZnJhYz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlx0ZnJhY3syMFxzcXJ0ezIxfX17NjN9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4sIHRoZW4gPG1hdGggaWQ9IkExMC5GMTMucGljNC5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJtez19MjAsbns9fTIxLHB7PX02MyIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIwPC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4sPC9tbz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yMTwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LDwvbW8+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5wPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjM8L21uPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5tez19MjAsbns9fTIxLHB7PX02MzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+4oCmPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzQucDIuMyIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAyLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiNCM0IzQjM7Ij4mbHQ7L3RoaW5rJmd0OzxzcGFuIGlkPSJBMTAuRjEzLnBpYzQucDIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9zZXJpZiIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNC5wMSIgY2xhc3M9Imx0eF9wYXJhIGx0eF9ub2luZGVudCI+PHNwYW4gY2xhc3M9Imx0eF9ydWxlIiBzdHlsZT0id2lkdGg6MzMwLjVwdDtoZWlnaHQ6MC4zcHQ7LS1sdHgtYmctY29sb3I6YmxhY2s7ZGlzcGxheTppbmxpbmUtYmxvY2s7Ij7CoDwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAzIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAzLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljNC5wMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRodXMgPG1hdGggaWQ9IkExMC5GMTMucGljNC5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJtK24rcD0yMCsyMSs2Mz1cYm94ZWR7MTA0fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5wPC9taT48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIwPC9tbj48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjE8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj42MzwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1lbmNsb3NlIG5vdGF0aW9uPSJib3giPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MTA0PC9tbj48L21lbmNsb3NlPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPm0rbitwPTIwKzIxKzYzPVxib3hlZHsxMDR9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAzLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljNC5wMy4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPk5vIOKAnEFuc3dlcjrigJ0gbGluZTsgcmVzdWx0IG9ubHkgaW5zaWRlIDxzcGFuIGlkPSJBMTAuRjEzLnBpYzQucDMuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIj5cYm94ZWR7fTwvc3Bhbj4uPHNwYW4gaWQ9IkExMC5GMTMucGljNC5wMy4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiA8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAzLjIuMS4yLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfaW5saW5lLWJsb2NrIiBzdHlsZT0id2lkdGg6MC4wcHQ7Ij7igIQ8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAzLjIuMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPuKclzwvc3Bhbj7igIQ8c3BhbiBpZD0iQTEwLkYxMy5waWM0LnAzLjIuMS4yLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPihzdHJpY3QgZm9ybWF0KTwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEwLkYxMy5waWM1IiBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMTk1Ljc3IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDE5NS43NyIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwxOTUuNzcpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA4QzAwOyIgZmlsbD0iIzAwOEMwMCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuOTEgTCAwIDE4OS44NiBDIDAgMTkzLjEyIDIuNjQgMTk1Ljc3IDUuOTEgMTk1Ljc3IEwgNDcxLjQ3IDE5NS43NyBDIDQ3NC43MyAxOTUuNzcgNDc3LjM4IDE5My4xMiA0NzcuMzggMTg5Ljg2IEwgNDc3LjM4IDUuOTEgQyA0NzcuMzggMi42NCA0NzQuNzMgMCA0NzEuNDcgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNFOEY1RTk7IiBmaWxsPSIjRThGNUU5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMTQxLjY4IEwgNDc1LjQxIDE0MS42OCBMIDQ3NS40MSA1LjkxIEMgNDc1LjQxIDMuNzMgNDczLjY1IDEuOTcgNDcxLjQ3IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDE1MC4wMSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNi40OGVtOy0tbHR4LWZvLWhlaWdodDoyLjg4ZW07LS1sdHgtZm8tZGVwdGg6MC4xOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0Mi4yOCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzkuODUpIiB3aWR0aD0iNTA0Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNi40OGVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+Q1JJU1DCoHYyICg1LDg1NCB0b2tlbnMpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6OS4xM2VtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTI2LjMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEyNi4zKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMiIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iQTEwLkYxMy5waWM1LnAyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM1LnAyLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPklucmFkaXVzIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzUubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0icj1cdGZyYWN7MjBcc3FydHsyMX19ezYzfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+cjwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtcm93PjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjA8L21uPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1zcXJ0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIxPC9tbj48L21zcXJ0PjwvbXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjYzPC9tbj48L21mcmFjPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPnI9XHRmcmFjezIwXHNxcnR7MjF9fXs2M308L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiwgc28gPG1hdGggaWQ9IkExMC5GMTMucGljNS5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJtez19MjAsbns9fTIxLHB7PX02MyIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjIwPC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4sPC9tbz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yMTwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LDwvbW8+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5wPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+NjM8L21uPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5tez19MjAsbns9fTIxLHB7PX02MzwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+IGFuZCA8bWF0aCBpZD0iQTEwLkYxMy5waWM1Lm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Im0rbitwPTEwNCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4rPC9tbz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+KzwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5wPC9taT48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjEwNDwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+bStuK3A9MTA0PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMi4zIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUucDIuMy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDsvdGhpbmsmZ3Q7PHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3NlcmlmIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM1LnAxIiBjbGFzcz0ibHR4X3BhcmEgbHR4X25vaW5kZW50Ij48c3BhbiBjbGFzcz0ibHR4X3J1bGUiIHN0eWxlPSJ3aWR0aDozMzAuNXB0O2hlaWdodDowLjNwdDstLWx0eC1iZy1jb2xvcjpibGFjaztkaXNwbGF5OmlubGluZS1ibG9jazsiPsKgPC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUucDMiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUucDMuMSIgY2xhc3M9Imx0eF9wIj48bWF0aCBpZD0iQTEwLkYxMy5waWM1Lm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Im0rbitwPTIwKzIxKzYzPTEwNCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPm08L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPis8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPm48L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPis8L21vPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPnA8L21pPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj49PC9tbz48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4yMDwvbW4+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+KzwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+MjE8L21uPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPis8L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjYzPC9tbj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+MTA0PC9tbj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5tK24rcD0yMCsyMSs2Mz0xMDQ8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUucDMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4uPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMy4yIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzUucDMuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5BbnN3ZXI6IDxtYXRoIGlkPSJBMTAuRjEzLnBpYzUubTUiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXGJvbGRzeW1ib2x7MTA0fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiIG1hdGhjb2xvcj0iIzAwODAwMCI+8J2fj/Cdn47wnZ+SPC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxib2xkc3ltYm9sezEwNH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiA8c3BhbiBpZD0iQTEwLkYxMy5waWM1LnAzLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2lubGluZS1ibG9jayIgc3R5bGU9IndpZHRoOjAuMHB0OyI+4oCEPHNwYW4gaWQ9IkExMC5GMTMucGljNS5wMy4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA4MDAwOyI+4pyTPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+)

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEwLkYxMy5waWM2IiBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMTk1Ljc3IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNDc3LjM4IDE5NS43NyIgd2lkdGg9IjQ3Ny4zOCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwxOTUuNzcpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojOTk0RDAwOyIgZmlsbD0iIzk5NEQwMCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDUuOTEgTCAwIDE4OS44NiBDIDAgMTkzLjEyIDIuNjQgMTk1Ljc3IDUuOTEgMTk1Ljc3IEwgNDcxLjQ3IDE5NS43NyBDIDQ3NC43MyAxOTUuNzcgNDc3LjM4IDE5My4xMiA0NzcuMzggMTg5Ljg2IEwgNDc3LjM4IDUuOTEgQyA0NzcuMzggMi42NCA0NzQuNzMgMCA0NzEuNDcgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGRkZCRjc7IiBmaWxsPSIjRkZGQkY3IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMTQxLjY4IEwgNDc1LjQxIDE0MS42OCBMIDQ3NS40MSA1LjkxIEMgNDc1LjQxIDMuNzMgNDczLjY1IDEuOTcgNDcxLjQ3IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDE1MC4wMSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNi40OGVtOy0tbHR4LWZvLWhlaWdodDoyLjg4ZW07LS1sdHgtZm8tZGVwdGg6MC4xOGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI0Mi4yOCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMzkuODUpIiB3aWR0aD0iNTA0Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNi40OGVtOyI+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+Q1JJU1DCoHYxICg1LDgzMyB0b2tlbnMpPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDguNjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NDEuNDZlbTstLWx0eC1mby1oZWlnaHQ6OS4xM2VtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTI2LjMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEyNi4zKSIgd2lkdGg9IjU3My42OSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYuMiIgY2xhc3M9Imx0eF9pbmxpbmUtbG9naWNhbC1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQxLjQ2ZW07Ij4KPHNwYW4gaWQ9IkExMC5GMTMucGljNi5wMiIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNi5wMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYucDIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7LS1sdHgtZmctY29sb3I6I0IzQjNCMzsiPiZsdDt0aGluayZndDs8c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAyLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAyLjIiIGNsYXNzPSJsdHhfcCI+PG1hdGggaWQ9IkExMC5GMTMucGljNi5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJyPVx0ZnJhY3syMFxzcXJ0ezIxfX17NjN9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+cjwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+PTwvbW8+PG1mcmFjIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj48bXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj4yMDwvbW4+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXNxcnQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjIxPC9tbj48L21zcXJ0PjwvbXJvdz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjcwMGVtIj42MzwvbW4+PC9tZnJhYz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5yPVx0ZnJhY3syMFxzcXJ0ezIxfX17NjN9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAyLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IGluIHRoZSBmb3JtIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzYubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXHRmcmFje21cc3FydHtufX17cH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bTwvbWk+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXNxcnQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+bjwvbWk+PC9tc3FydD48L21yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5wPC9taT48L21mcmFjPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XHRmcmFje21cc3FydHtufX17cH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPiB3aXRoIDxtYXRoIGlkPSJBMTAuRjEzLnBpYzYubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ibXs9fTIwLG57PX0yMSxwez19NjMiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPm08L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+PTwvbW8+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj4yMDwvbW4+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+LDwvbW8+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj5uPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+MjE8L21uPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPiw8L21vPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+cDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj49PC9tbz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPjYzPC9tbj48L21yb3c+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+bXs9fTIwLG57PX0yMSxwez19NjM8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPi48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAyLjMiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExMC5GMTMucGljNi5wMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojQjNCM0IzOyI+Jmx0Oy90aGluayZndDs8c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAyLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfc2VyaWYiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYucDEiIGNsYXNzPSJsdHhfcGFyYSBsdHhfbm9pbmRlbnQiPjxzcGFuIGNsYXNzPSJsdHhfcnVsZSIgc3R5bGU9IndpZHRoOjMzMC41cHQ7aGVpZ2h0OjAuM3B0Oy0tbHR4LWJnLWNvbG9yOmJsYWNrO2Rpc3BsYXk6aW5saW5lLWJsb2NrOyI+wqA8L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNi5wMyIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkExMC5GMTMucGljNi5wMy4xIiBjbGFzcz0ibHR4X3AiPjxtYXRoIGlkPSJBMTAuRjEzLnBpYzYubTQiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0ibStuK3A9MTA0IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+bTwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+KzwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+bjwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+KzwvbW8+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC43MDBlbSI+cDwvbWk+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPj08L21vPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuNzAwZW0iPjEwNDwvbW4+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+bStuK3A9MTA0PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAzLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Ljwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMTAuRjEzLnBpYzYucDMuMiIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTEwLkYxMy5waWM2LnAzLjIuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iZm9udC1zaXplOjcwJTstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QW5zd2VyOiA8bWF0aCBpZD0iQTEwLkYxMy5waWM2Lm01IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9Ilxib2xkc3ltYm9sezEwNH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDgwMDA7IiBtYXRoY29sb3I9IiMwMDgwMDAiPvCdn4/wnZ+O8J2fkjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cYm9sZHN5bWJvbHsxMDR9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD4gPHNwYW4gaWQ9IkExMC5GMTMucGljNi5wMy4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9pbmxpbmUtYmxvY2siIHN0eWxlPSJ3aWR0aDowLjBwdDsiPuKAhDxzcGFuIGlkPSJBMTAuRjEzLnBpYzYucDMuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwODAwMDsiPuKckzwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

Figure 13: CRISP consolidates onto the requested “Answer:” format. On a
MATH-500 problem (top) and an AIME 2024 problem (bottom), the base
Qwen3-8B model presents its final result only inside
“$`\backslash`$boxed{$`\cdot`$}” with no “Answer:” line, which a strict
answer-format grader scores as wrong even though the boxed value is
correct. Both CRISP variants (v1 uniform, v2 difficulty-aware) emit the
requested “Answer:” line and are also substantially shorter. This is the
mechanism behind the answer-only/boxed-only shift in
Appendix [D](#A4 "Appendix D Answer-Format Breakdown ‣ CRISP: Compressed Reasoning via Iterative Self-Policy Distillation").
````
