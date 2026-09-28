---
identifier: arxiv:2608.11573v2
title: Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs
authors:
  - Vu Duc Anh
  - Nhat M. Hoang
  - Do Xuan Long
  - Cong-Duy Nguyen
  - Ponhvoan Srey
  - Luu Anh Tuan
published: "2026-08-12T02:31:02+00:00"
url: https://arxiv.org/abs/2608.11573v2
source: arxiv
doi: null
arxiv_id: 2608.11573v2
categories:
  - cs.AI
  - cs.CL
---

# Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs

Vu Duc Anh Affiliation: Nanyang Technological University, Singapore
Email: <vuducanh001@ntu.edu.sg>    Nhat M. Hoang Affiliation: Nanyang
Technological University, Singapore Email: <hoangmin003@ntu.edu.sg>   
Do Xuan Long Affiliation: National University of Singapore
Affiliation: Institute for Infocomm Research (IR), A\*STAR
Email: <ponhvoan002@ntu.edu.sg>    Cong-Duy Nguyen   Ponhvoan Srey   Luu
Anh Tuan ^(†)^(†)thanks: ˜˜Corresponding author. Affiliation: Nanyang
Technological University, Singapore Affiliation: VinUniversity, Vietnam
Email: <anhtuan.luu@ntu.edu.sg> Email: [xuanlong.do@u.nus.edu](mailto:)
Email: [duy.ntc@vinuni.edu.vn](mailto:)

###### Abstract

Achieving effective self-correction, where models verify and correct
their own mistakes, remains a fundamental challenge for large language
models (LLMs). In this work, we propose Self-Fix Step-DPO (SFS-DPO), a
reinforcement learning based, two-stage framework for step-level
self-verification and self-correction. The first stage strengthens
step-level reasoning via step-level preference optimization, while the
second stage explicitly trains models to self-verify and self-correct.
We further introduce a teacher-assisted variant, SFS-DPO-R, which
incorporates explanatory rationales for error verification to provide
stronger corrective signals. Comprehensive in-domain and out-of-domain
evaluations across multiple LLMs demonstrate that SFS-DPO and SFS-DPO-R
consistently outperform prior step-level training baselines. Our
analysis further reveals improvements in self-correction frequency and
effectiveness, highlighting the importance of strengthening step-level
reasoning for robust performance.

![Refer to caption](2608.11573v2/images/method.png)

Figure 1: Initialization Stage (Stage 1) learns local step preferences
by favoring correct continuations over incorrect ones, without revising
errors. Step-wise Self-Correction (Stage 2) complements this by training
explicit self-corrections: revising incorrect steps toward preferred
continuations before proceeding, converting step-wise preferences into
targeted multi-step reasoning improvements.

## 1 Introduction

Despite recent advances in frontier large language models (LLMs),
smaller LLMs still struggle with complex math reasoning, where early
errors can propagate throughout the entire solution [Lightman et al.
(2024)](#bib.bib1); [Hong et al. (2024)](#bib.bib4); [Chen et al.
(2025)](#bib.bib3). To provide more localized learning signals, recent
studies have shifted toward step-level objectives [Lai et al.
(2024)](#bib.bib23); [Lu et al. (2024)](#bib.bib2); [Xu et al.
(2025)](#bib.bib12); [Pham et al. (2026)](#bib.bib32). However, they
mainly optimize preferences over better reasoning continuations, without
explicitly correcting erroneous steps that were already generated [Kumar
et al. (2024)](#bib.bib7); [Pan et al. (2025)](#bib.bib8).

|                  |         |              |                            |       |        |              |              |                  |
| ---------------- | ------- | ------------ | -------------------------- | ----- | ------ | ------------ | ------------ | ---------------- |
| Method           | Init.   | Optimization | Training Size Init. / Opt. |       | Spont. | Err. Detect. | Error Critic | External Teacher |
| SCoRe            | RL      | RL           | 12K/                       | 12K   | ✗      | ✗            | ✗            | ✗                |
| SuperCorrect     | SFT     | Step RL      | 100K/                      | 10K   | ✔     | ✔           | ✔           | ✔               |
| LEMMA            | \-      | SFT          | -/                         | 88.9K | ✔     | ✔           | ✗            | ✗                |
| S3C-MATH         | \-      | SFT          | -/                         | 927K  | ✔     | ✔           | ✔           | ✔               |
| SPOC             | SFT     | RL           | 860K/                      | \-    | ✔     | ✔           | ✔           | ✗                |
| S²R              | SFT     | RL           | 3.1K/                      | 10K   | ✔     | ✔           | ✔           | ✗                |
| SFS-DPO (ours)   | Step RL | Step RL      | 10K/                       | 8.4K  | ✔     | ✔           | ✗            | ✗                |
| SFS-DPO-R (ours) | Step RL | Step RL      | 10K/                       | 8.4K  | ✔     | ✔           | ✔           | ✔               |

Table 1: Comparison of self-correction methods in LLMs. Our methods
(SFS-DPO and SFS-DPO-R) achieve competitive capability coverage with
significantly less training data than prior approaches.

This limitation motivates a complementary paradigm of _self-correction_,
which introduces self-improving loops that enable models to detect and
correct their own mistakes during inference. This has inspired a
parallel line of research on training LLMs to self-correct ([Kumar et
al., 2024](#bib.bib7); [Yang et al., 2025b](#bib.bib6); [Zhao et al.,
2025](#bib.bib10)). While these methods highlight the promises of
self-correction, they introduce optimization challenges. Specifically,
[Kumar et al. (2024)](#bib.bib7) demonstrate that supervised fine-tuning
(SFT) on correction traces is prone to distribution shift and behavior
collapse, and propose an on-policy reinforcement learning (RL) solution.
To improve the self-correction robustness, [Ma et al.
(2025)](#bib.bib13); [Yang et al. (2025b)](#bib.bib6) further decompose
the optimization process into two phases: first, training the models
using SFT with carefully designed templates, followed by a second RL
phase for self-correction. However, these initialization strategies are
primarily designed to enforce models to follow specific templates that
facilitate the self-correction phase, rather than to explicitly
strengthen step-level reasoning. Explicitly strengthening step-level
reasoning is crucial, since step-wise self-correction involves two
challenging subproblems, error detection and targeted revision, and weak
step-level reasoning can cause their joint learning to produce noisy
signals and compounding errors ([Caruana, 1997](#bib.bib21)).

We introduce Self-Fix Step-DPO (SFS-DPO), a two-stage framework that
trains LLMs to self-verify and self-correct via step-level reinforcement
learning. As shown in
[Figure 1](#S0.F1 "In Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
the initialization stage applies an RL-based step-level preference
optimization to strengthen step-wise reasoning, while the second stage
(Step-wise Self-Correction) trains the models to self-correct via
learning the preferences between a self-corrected continuation and the
continuation produced when an incorrect step is left unaddressed. We
further propose a teacher-assisted variant, SFS-DPO-R, which augments
corrections with explanatory rationales to provide stronger corrective
signals and improve downstream accuracy. Comprehensive in-domain and
out-of-domain evaluations on multiple LLMs show that SFS-DPO and
SFS-DPO-R yield consistent gains over baselines. Our contributions are
threefold:

- •
  We propose an RL-based, step-level self-correction training framework
  with two variants, SFS-DPO and SFS-DPO-R, that improves step-level
  reasoning and enables effective self-correction in LLMs.
- •
  We conduct comprehensive in-domain and out-domain experiments
  demonstrating that our method achieves consistent improvements over
  existing baselines across multiple LLMs.
- •
  We analyze self-correction behavior in our framework, providing
  empirical insights into its frequency, effectiveness, and relationship
  with reasoning performance.

## 2 Related Work

### 2.1 RL for LLM Reasoning

Preference optimization has been widely used to align language models
for reasoning tasks, typically operating at the level of full solutions
or complete responses ([Rafailov et al., 2023](#bib.bib22)). However,
answer-level preferences can be coarse for complex math reasoning, as
incorrect solutions may share long correct prefixes with correct ones,
leading to weak credit assignment. To address this limitation, recent
work has explored learning signals at the level of individual reasoning
steps ([Chen et al., 2024](#bib.bib20); [Lai et al., 2024](#bib.bib23);
[Xu et al., 2025](#bib.bib12); [Lightman et al., 2024](#bib.bib1); [Jin
et al., 2025](#bib.bib14); [Lahlou et al., 2025](#bib.bib5)). Notably,
Step-DPO ([Lai et al., 2024](#bib.bib23)) defines preference comparisons
over next-step continuations conditioned on a correct prefix, explicitly
targeting the first erroneous step in a reasoning trajectory and
providing localized supervision without requiring explicit step-level
labels. In addition, process supervision ([Lightman et al.,
2024](#bib.bib1)) demonstrates that supervising intermediate reasoning
steps with explicit correctness judgments significantly improves
mathematical reasoning by enabling more precise credit assignment. Our
work builds on this line by using step-level preference optimization as
a first training stage for learning step-wise self-correction.

### 2.2 Teaching LLMs to Self-Correct

Self-correction is a critical capability for scaling the reliability of
large language models. As a result, a growing body of work studies how
to equip LLMs to self-correct their own reasoning ([Kumar et al.,
2024](#bib.bib7); [Do et al., 2024](#bib.bib33); [Yang et al.,
2025b](#bib.bib6); [Zeng et al., 2025](#bib.bib9); [Zhao et al.,
2025](#bib.bib10)), see
[Table 1](#S1.T1 "In 1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
Notably, SCoRe ([Kumar et al., 2024](#bib.bib7)) learns intrinsic
self-correction via multi-turn on-policy reinforcement learning on
self-generated data, highlighting distribution shift and behavior
collapse as limitations of offline correction training. SuperCorrect
([Yang et al., 2025b](#bib.bib6)) proposes a two-stage teacher–student
framework that initializes models with supervised fine-tuning on
hierarchical thought templates and then applies preference optimization
using teacher-provided correction traces. S3C-MATH ([Yan et al.,
2025](#bib.bib11)) trains step-level self-correction by inserting
incorrect steps into correct solution traces and supervising models to
detect and fix these errors during generation. SPOC ([Zhao et al.,
2025](#bib.bib10)) induces spontaneous self-correction at inference time
by alternating solution generation and verification using reinforcement
learning. S²R ([Ma et al., 2025](#bib.bib13)) combines supervised
initialization with outcome- and process-level reinforcement learning to
train models that self-verify and iteratively refine their reasoning. In
contrast, our work initializes step-level self-correction with a
step-wise RL-based preference optimization stage, which we empirically
find yields more robust downstream self-correction than from scratch or
supervised fine-tuning.

## 3 Methodology

##### Task Formulation.

We formulate self-correction in language model reasoning as the
generation of a multi-step trajectory with explicit error detection and
repair. Given an input problem $`x`$, a language model $`\pi_{\theta}`$
produces a reasoning trajectory

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       \{s_{j}\}_{j=1}^{M}=(s_{1},\ldots,s_{M},\hat{y}),
       ```                                                |     |

where each step $`s_{j}`$ is generated autoregressively as

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       s_{j}\sim\pi_{\theta}(\cdot\mid x,s_{<j})
       ```                                        |     |

We formulate self-correction by allowing each step $`s_{j}`$ to take one
of three types, $`\{\textsc{solution step}`$, error-detection step,
$`\textsc{fixed step}\}`$, corresponding respectively to a standard
reasoning step, an explicit signal (optionally with reasoning)
indicating that the previous step is incorrect, or a corrected version
of an erroneous step. When an error-detection step is generated, the
model produces a fixed step that replaces the incorrect reasoning in the
trajectory. The final answer $`\hat{y}`$ is obtained from the terminal
corrected trajectory.

##### Motivation.

Step-wise self-correction requires optimizing two objectives
simultaneously: evaluating the correctness of a reasoning step and
generating a corrected alternative. Jointly optimizing these objectives
can produce noisy learning signals when step-level reasoning capability
is insufficient, as errors in step evaluation and step revision may
compound ([Caruana, 1997](#bib.bib21)). Our framework, therefore, adopts
a two-phase training strategy: (1) Step-level Preference Optimization as
a initialization stage that improves step-level reasoning capability by
aligning the model toward correct intermediate reasoning steps; and (2)
Step-wise Self-Correction stage. We hypothesize that modeling preference
signals at the step level will provide a stronger foundation for
subsequent step-wise self-correction, enabling more effective generation
of corrected steps once an incorrect step is identified.

### 3.1 Initialization Stage

In the first stage, we employ a reinforcement learning objective to
initialize the model as a initialization training phase. Specifically,
it aims to maximize the likelihood of a correct solution step
$`s_{k}^{+}`$ while minimizing that of the incorrect solution step
$`s_{k}^{-}`$. The optimization objective is:

|     |                                                                                                                                                     |     |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
|     | $`\displaystyle\mathcal{L}_{\text{Pre}}(\theta)=-\mathbb{E}_{(x,\{s_{i}\}_{i=1}^{k-1},s_{k}^{+},s_{k}^{-})\sim\mathcal{D}}\Biggl[\log\sigma\Bigl(`$ |     |
|     | $`\displaystyle\beta\bigl(\log\pi_{\theta}(s_{k}^{+}\mid x,\{s_{i}\}_{i=1}^{k-1})-`$                                                                |     |
|     | $`\displaystyle\log\pi_{\theta}(s_{k}^{-}\mid x,\{s_{i}\}_{i=1}^{k-1})\bigr)\Bigr)\Biggr].`$                                                        |     |

where $`\pi_{\theta}`$ denotes the policy model being optimized,
$`\beta`$ controls the strength of the preference regularization, and
$`\mathcal{D}`$ is a dataset of step-level preference pairs. This stage
serves as a initialization phase that establishes a strong foundation
for subsequent step-wise self-correction optimization. In this work, we
adopt the step-level preference optimization framework of [Lai et al.
(2024)](#bib.bib23) to implement this initialization stage.

### 3.2 Step-wise Self-Correction

After equipping the model with RL-based step-level supervision signals,
we further enable explicit self-correction capability by supervising the
model to recognize and correct the erroneous reasoning steps. Formally,
given an incorrect reasoning step $`s_{k}^{-}`$ following a correct
trajectory $`\{s_{1},\ldots,s_{k-1}\}`$, we construct a preference
objective that favors a self-corrected continuation $`c_{k}^{+}`$ over
the subsequent incorrect step $`s_{k+1}^{-}`$ that would be generated if
the error remains unaddressed. The resulting loss:

|     |                                           |                                                                                                                                                                                            |     |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- |
|     | $`\displaystyle\mathcal{L_{SC}}(\theta)`$ | $`\displaystyle=-\mathbb{E}_{(x,\{s_{i}\}_{i=1}^{k-1},s_{k}^{-},c_{k}^{+},s_{k+1}^{-})\sim\mathcal{D_{SC}}}\Biggl[\log\sigma`$                                                             |     |
|     |                                           | $`\displaystyle\qquad\Bigl(\beta\log\frac{\pi_{\theta}(c_{k}^{+}\mid x,\{s_{i}\}_{i=1}^{k-1},s_{k}^{-})}{\pi_{\text{ref}}(c_{k}^{+}\mid x,\{s_{i}\}_{i=1}^{k-1},s_{k}^{-})}`$              |     |
|     |                                           | $`\displaystyle\qquad-\beta\log\frac{\pi_{\theta}(s_{k+1}^{-}\mid x,\{s_{i}\}_{i=1}^{k-1},s_{k}^{-})}{\pi_{\text{ref}}(s_{k+1}^{-}\mid x,\{s_{i}\}_{i=1}^{k-1},s_{k}^{-})}\Bigr)\Biggr].`$ |     |

Here, $`\pi_{\text{ref}}`$ denotes a frozen baseline model,
$`s_{k}^{-}`$ denotes an incorrect reasoning step, while $`s_{k+1}^{-}`$
represents the subsequent continuation produced when the error is not
detected. In contrast, $`c_{k}^{+}`$ corresponds to a self-corrected
reasoning step that explicitly addresses the detected error. We
investigate two variants of self-correction supervision that differ in
how the corrected step $`c_{k}^{+}`$ is constructed.

#### 3.2.1 Self-Fix Step-DPO (SFS-DPO)

Under the SFS-DPO setting, the self-corrected step is constructed by
explicitly detecting and fixing an erroneous reasoning step. Formally,
we define:

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       c_{k}^{+}=\{d_{k-1},s_{k}^{+}\},
       ```                               |     |

where $`d_{k-1}`$ is an explicit error-detection signal that flags the
flaw in the incorrect step $`s_{k}^{-}`$, and $`s_{k}^{+}`$ is the
corrected reasoning step that resolves the detected error. This
formulation enables teacher-free self-correction by relying solely on
model-generated detection and correction signals.

#### 3.2.2 Self-Fix Step-DPO with Reasoning (SFS-DPO-R)

SFS-DPO-R further incorporates external teacher supervision to provide
richer corrective signals. Specifically, we leverage a strong teacher
model to generate an explicit explanation of why the previous step is
incorrect. The corrected step is then defined as:

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       c_{k}^{+}=\{d_{k-1},r_{k-1},s_{k}^{+}\},
       ```                                       |     |

where $`r_{k-1}`$ denotes a teacher-generated rationale explaining the
error in $`s_{k}^{-}`$. By augmenting the correction with high-quality
explanatory reasoning, SFS-DPO-R offers stronger supervision at the cost
of additional teacher dependence.

We evaluate both SFS-DPO and SFS-DPO-R to contrast a fully teacher-free
setting with a teacher-assisted variant, thereby quantifying the
trade-off between quality and reliance on external models.

## 4 Main Experiments

### 4.1 Experimental Setup

##### Baselines.

To comprehensively evaluate the effectiveness of our proposed methods,
we conduct experiments on seven widely used open-source LLMs, following
the original Step-DPO experimental setup. Specifically, we use three
backbone LLMs from Step-DPO: DeepSeekMath-7B [Shao et al.
(2024)](#bib.bib25), Qwen2-7B, and Qwen2-7B-Instruct [Yang et al.
(2024a)](#bib.bib26). DeepSeekMath-7B and Qwen2-7B are further
fine-tuned on the MetaMath [Yu et al. (2024)](#bib.bib15) and MMIQC [Liu
et al. (2025)](#bib.bib27) datasets, yielding DeepSeekMath-7B-SFT and
Qwen2-7B-SFT. In addition, we include three stronger recent
instruction-tuned models, Llama-3.1-8B-Instruct [Grattafiori et al.
(2024)](#bib.bib24), Qwen3-8B [Yang et al. (2025a)](#bib.bib31) and
Qwen2.5-14B-Instruct [Yang et al. (2024b)](#bib.bib30), as well as the
recent math-specialized model Qwen2.5-Math-7B-Instruct [Yang et al.
(2024c)](#bib.bib29), to further assess the scalability and robustness
of our methods across both general-purpose and math-oriented reasoning
LLMs.

|                          |                                                   |                                                   |                 |                |                |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------- | --------------- | -------------- | -------------- |
|                          | In-Domain                                         |                                                   | Out-of-Domain   |                |                |
| Model                    | MATH                                              | GSM8K                                             | GK2023          | OCW            | Avg.           |
| DeepSeekMath-7B-SFT      | 51.7                                              | 86.8                                              | 38.2            | 19.1           | 49.0           |
|   + Step-DPO             | 51.8\_((+0.1))                                    | 86.7\_((-0.1))                                    | 43.4\_((+5.2))  | 18.0\_((-1.1)) | 50.0\_((+1.0)) |
|   + SFS-DPO              | 52.2$`{}_{{\color[rgb]{0,0.45,0.25}(+0.5)}}^{*}`$ | 87.6$`{}_{{\color[rgb]{0,0.45,0.25}(+0.8)}}^{*}`$ | 43.9\_((+5.7))  | 23.2\_((+4.1)) | 51.7\_((+2.7)) |
|   + SFS-DPO-R            | 52.2$`{}_{{\color[rgb]{0,0.45,0.25}(+0.5)}}^{*}`$ | 88.0$`{}_{{\color[rgb]{0,0.45,0.25}(+1.2)}}^{*}`$ | 43.9\_((+5.7))  | 25.0\_((+5.9)) | 52.3\_((+3.3)) |
| Qwen2-7B-SFT             | 53.9                                              | 87.6                                              | 46.2            | 15.8           | 50.9           |
|   + Step-DPO             | 55.3\_((+1.4))                                    | 87.6\_((+0.0))                                    | 45.7\_((-0.5))  | 15.8\_((+0.0)) | 51.1\_((+0.2)) |
|   + SFS-DPO              | 55.6$`{}_{{\color[rgb]{0,0.45,0.25}(+1.7)}}^{*}`$ | 87.9\_((+0.3))                                    | 46.0\_((-0.2))  | 23.9\_((+8.1)) | 53.4\_((+2.5)) |
|   + SFS-DPO-R            | 55.4\_((+1.5))                                    | 88.0$`{}_{{\color[rgb]{0,0.45,0.25}(+0.4)}}^{*}`$ | 46.2\_((+0.0))  | 22.8\_((+7.0)) | 53.1\_((+2.2)) |
| Qwen2-7B-Instruct        | 55.7                                              | 85.0                                              | 39.7            | 20.2           | 50.2           |
|   + Step-DPO             | 57.1\_((+1.4))                                    | 86.2\_((+1.2))                                    | 42.9\_((+3.2))  | 21.3\_((+1.1)) | 51.9\_((+1.7)) |
|   + SFS-DPO              | 58.6$`{}_{{\color[rgb]{0,0.45,0.25}(+2.9)}}^{*}`$ | 86.6$`{}_{{\color[rgb]{0,0.45,0.25}(+1.6)}}^{*}`$ | 46.0\_((+6.3))  | 20.6\_((+0.4)) | 53.0\_((+2.8)) |
|   + SFS-DPO-R            | 59.1$`{}_{{\color[rgb]{0,0.45,0.25}(+3.4)}}^{*}`$ | 86.0\_((+1.0))                                    | 44.7\_((+5.0))  | 22.1\_((+1.9)) | 53.0\_((+2.8)) |
| Qwen2.5-Math-7B-Instruct | 83.6                                              | 95.2                                              | 57.4            | 23.9           | 65.0           |
|   + Step-DPO             | 84.6\_((+1.0))                                    | 95.8\_((+0.6))                                    | 67.0\_((+9.6))  | 31.6\_((+7.7)) | 69.8\_((+4.8)) |
|   + SFS-DPO              | 84.7\_((+1.1))                                    | 95.8\_((+0.6))                                    | 68.3\_((+10.9)) | 32.0\_((+8.1)) | 70.2\_((+5.2)) |
|   + SFS-DPO-R            | 85.0$`{}_{{\color[rgb]{0,0.45,0.25}(+1.4)}}^{*}`$ | 96.0\_((+0.8))                                    | 67.8\_((+10.4)) | 32.7\_((+8.8)) | 70.4\_((+5.4)) |
| Qwen3-8B                 | 72.9                                              | 92.8                                              | 58.4            | 31.3           | 63.9           |
|   + Step-DPO             | 73.4\_((+0.5))                                    | 92.8\_((+0.0))                                    | 58.2\_((-0.2))  | 31.3\_((+0.0)) | 63.9\_((+0.0)) |
|   + SFS-DPO              | 73.5\_((+0.6))                                    | 93.1\_((+0.3))                                    | 58.4\_((+0.0))  | 33.1\_((+1.8)) | 64.5\_((+0.6)) |
|   + SFS-DPO-R            | 73.7$`{}_{{\color[rgb]{0,0.45,0.25}(+0.8)}}^{*}`$ | 93.9$`{}_{{\color[rgb]{0,0.45,0.25}(+1.1)}}^{*}`$ | 59.0\_((+0.6))  | 34.2\_((+2.9)) | 65.2\_((+1.3)) |
| Llama-3.1-8B-Instruct    | 50.4                                              | 86.2                                              | 38.4            | 25.0           | 50.0           |
|   + Step-DPO             | 51.8\_((+1.4))                                    | 86.4\_((+0.2))                                    | 41.8\_((+3.4))  | 27.6\_((+2.6)) | 51.9\_((+1.9)) |
|   + SFS-DPO              | 51.0\_((+0.6))                                    | 86.7\_((+0.5))                                    | 42.1\_((+3.7))  | 28.3\_((+3.3)) | 52.0\_((+2.0)) |
|   + SFS-DPO-R            | 51.7\_((+1.3))                                    | 87.1$`{}_{{\color[rgb]{0,0.45,0.25}(+0.9)}}^{*}`$ | 42.6\_((+4.2))  | 27.9\_((+2.9)) | 52.3\_((+2.3)) |
| Qwen2.5-14B-Instruct     | 78.8                                              | 93.8                                              | 65.5            | 37.5           | 68.9           |
|   + Step-DPO             | 79.3\_((+0.5))                                    | 94.1\_((+0.3))                                    | 65.7\_((+0.2))  | 36.0\_((-1.5)) | 68.8\_((-0.1)) |
|   + SFS-DPO              | 79.2\_((+0.4))                                    | 94.5$`{}_{{\color[rgb]{0,0.45,0.25}(+0.7)}}^{*}`$ | 65.7\_((+0.2))  | 38.2\_((+0.7)) | 69.4\_((+0.5)) |
|   + SFS-DPO-R            | 79.4\_((+0.6))                                    | 94.5$`{}_{{\color[rgb]{0,0.45,0.25}(+0.7)}}^{*}`$ | 67.8\_((+2.3))  | 37.7\_((+0.2)) | 69.9\_((+1.0)) |

Table 2: Percentage accuracy on in-domain (MATH, GSM8K) and
out-of-domain (GK2023, OCW) benchmarks with greedy decoding. Numbers in
parentheses denote absolute accuracy change relative to the
corresponding base backbone. For in-domain benchmarks, \* denotes
statistically significant improvements over Step-DPO according to
one-sided McNemar’s test (p \< 0.05).

##### Dataset Construction & Training Setup.

To construct preference learning data for the step-wise self-correction
stage, we first collect samples from the original 10K dataset of [Lai et
al. (2024)](#bib.bib23). For each sample, we then append the rejected
reasoning step to the initial reasoning to form a new reasoning prefix
with an erroneous reasoning step, and use the subsequent rejected
reasoning step as the rejected sample for our dataset. Following [Pan et
al. (2025)](#bib.bib8), to construct the chosen step, we concatenate the
correct reasoning step with a self-correction signal. The same set of
signal phrases, listed in
[Figure 6](#A3.F6 "In Appendix C Data Construction Details ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
is then used to automatically identify self-correction behavior in
generated responses. Through this process, we obtain a resource-free
SFS-DPO dataset containing 8,416 samples. We further leverage GPT-4o
[Hurst et al. (2024)](#bib.bib28) to generate explicit explanations for
incorrect reasoning steps and insert them between the self-correction
signal and the correct reasoning step, resulting in the SFS-DPO-R
dataset. Further details are provided in
[Figure 7](#A3.F7 "In Appendix C Data Construction Details ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").

In the initialization stage, to ensure a fair comparison, we use the
same 10K-sample training dataset from Step-DPO [Lai et al.
(2024)](#bib.bib23), which is collected from the training set of GSM8K
[Cobbe et al. (2021)](#bib.bib16) and MATH [Hendrycks et al.
(2021)](#bib.bib17), as the initial source of step-level reasoning
supervision. All models are first trained for three epochs in the
initialization stage with batch size 4. We then train SFS-DPO and
SFS-DPO-R for four epochs in the self-correction stage using batch size 8. All training stages use the AdamW optimizer with a warmup ratio of
0.02 and learning rate of $`5\times 10^{-7}`$.

##### Benchmarks.

To evaluate our method, we consider both in-domain and out-of-domain
(OOD) benchmarks. In-domain performance is measured on GSM8K [Cobbe et
al. (2021)](#bib.bib16) and MATH [Hendrycks et al. (2021)](#bib.bib17),
with 1,319 and 5,000 test questions, respectively. To assess
generalization, we report results on GaoKao2023 (GK2023) ([Liao et al.,
2024](#bib.bib18)) comprising 385 competition-level math problems from
the 2023 Chinese university entrance exam, and OCWCourses (OCW)
([Lewkowycz et al., 2022](#bib.bib19)) containing 272
undergraduate-level STEM problems requiring multi-step reasoning. All
benchmarks are evaluated using answer accuracy. We define the
self-correction rate metric as the fraction of model-generated solutions
that are correct, conditioned on the model deciding to perform
self-correction. Following [Ma et al. (2025)](#bib.bib13), we further
report Error Recall, the fraction of incorrect reasoning steps that the
model flags via a self-correction signal.

### 4.2 In-Domain Results

Our main experimental results in
[Table 2](#S4.T2 "In Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
demonstrate that explicitly modeling step-level self-correction yields
stronger gains than prior step-wise preference learning. Across seven
backbones and two in-domain datasets, Step-DPO provides limited and
inconsistent improvements. While SFS-DPO and SFS-DPO-R improve over the
base models in all settings, Step-DPO mostly gives gains below 1%, with
performance plateauing or slightly degrading on GSM8K for Qwen2-7B-SFT
and DeepSeekMath-7B-SFT. In contrast, SFS-DPO and SFS-DPO-R learn
explicit self-correction and achieve larger gains than Step-DPO,
outperforming it in 11/14 and 12/14 settings, respectively. This
suggests that self-correction training improves reasoning performance
while teaching models when and how to revise their reasoning.

On the model level, our method yields average improvements of 0.83/0.97%
on the math-specialized backbones, and even larger gains of 0.95/1.23%
on the instruction-tuned backbones. Qwen2-7B-Instruct achieves the most
noticeable gains, with a 3.4% improvement on MATH under SFS-DPO-R.
Improvements are smaller on recent backbones, namely
Qwen2.5-14B-Instruct, Qwen3-8B and Llama-3.1-8B-Instruct. One likely
factor is that these backbones already self-correct more often before
training, leaving less room to introduce new correction behavior;
further analysis is provided in
[Appendix B](#A2 "Appendix B Additional Analysis ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
Overall, these results highlight that the primary advantage of our
method lies not in finer-grained preference learning alone, but in
optimizing preferences over corrected trajectories.

Averaged across seven backbones, SFS-DPO improves accuracy by 1.11% on
MATH and 0.69% on GSM8K, while SFS-DPO-R further increases the gains to
1.36% and 0.87%, respectively. The consistent advantage of SFS-DPO-R
over SFS-DPO indicates that explicit explanatory reasoning about errors
provides additional supervision beyond correction alone. By exposing the
model to why a reasoning step is incorrect, SFS-DPO-R encourages more
accurate error localization and more targeted repairs, leading to
improved downstream accuracy.

### 4.3 Out-Of-Domain Results (OOD)

To assess robustness under distribution shift, we evaluate on the
competition-level GK2023 benchmark and undergraduate OCWCourses (OCW),
as shown in
[Table 2](#S4.T2 "In Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
Step-DPO shows inconsistent OOD behavior, including degradation on
Qwen2-7B-SFT for GK2023 and on the larger Qwen2.5-14B-Instruct for OCW,
while SFS-DPO and SFS-DPO-R consistently maintain or improve accuracy
across all backbones and datasets. Notably, Qwen2-7B-Instruct achieves a
substantial 6.3% improvement on GK2023 under SFS-DPO, while
Qwen2.5-Math-7B-Instruct achieves the largest overall OOD gains,
reaching 10.9% on GK2023 and 8.8% on OCW. These results indicate that
explicit step-level error detection and correction yields correction
strategies that generalize beyond the training distribution, while the
small gap between SFS-DPO and SFS-DPO-R suggests that self-generated
correction signals already capture much of the transferable structure
needed for OOD reasoning.

Figure 2: Self-correction rate of SFS-DPO-R under five Instruct LLMs.
Self-correction rate shows a positive correlation with task accuracy

Figure 3: Distribution of numbers of self-correction steps per solution
under SFS-DPO-R, averaged over six backbone models.

## 5 Discussions

### 5.1 Comparison to Self-Correction Baselines

|           |      |       |         |              |
| --------- | ---- | ----- | ------- | ------------ |
| Model     | MATH | GSM8K | SC Rate | Error Recall |
| LEMMA     | 48.5 | 83.3  | 45.3    | 49.9         |
| S²R       | 48.7 | 84.4  | 46.5    | 54.6         |
| SFS-DPO   | 51.0 | 86.7  | 28.8    | 33.2         |
| SFS-DPO-R | 51.7 | 87.1  | 23.9    | 35.9         |

Table 3: Comparison of prior self-correction methods and ours with
Llama-3.1-8B-Instruct as the backbone.

We compare our framework with prior self-correction supervised
fine-tuning (SFT) baselines. Due to the availability of comparable
baselines, we conduct all experiments using Llama-3.1-8B-Instruct as the
shared backbone. As shown in
[Table 3](#S5.T3 "In 5.1 Comparison to Self-Correction Baselines ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
SFS-DPO and SFS-DPO-R consistently outperform existing baselines on both
MATH and GSM8K while exhibiting lower self-correction rates and lower
Error Recall, indicating that neither correcting more often nor
detecting more errors translates into better performance. This may be
due to SFT-based methods suffering from distribution shift and behavior
collapse [Kumar et al. (2024)](#bib.bib7), which biases the model toward
excessive self-correction: it follows correction templates rather than
selectively revising genuine errors. Further analysis of these
behavioral metrics is discussed in
[Section 5.3](#S5.SS3 "5.3 Self-Correction Rate ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").

### 5.2 The Role of Initialization Stage

|          |                           |             |             |
| -------- | ------------------------- | ----------- | ----------- |
|          | Initialization            | MATH        | GSM8K       |
| SFT      | Step RL (ours)            | 55.6        | 87.9        |
|          | No init.                  | 53.1 (-2.5) | 87.3 (-0.6) |
|          | No init. + Joint Training | 53.3 (-2.2) | 84.1 (-3.8) |
|          | RL                        | 53.3 (-2.2) | 83.6 (-4.3) |
| Instruct | Step RL (ours)            | 58.6        | 86.6        |
|          | No init.                  | 53.1 (-4.1) | 85.6 (-1.0) |
|          | No init. + Joint Training | 56.9 (-1.7) | 86.4 (-0.2) |
|          | RL                        | 55.7 (-0.9) | 86.4 (-0.2) |

Table 4: Comparison of training strategies using Qwen2-7B models.
Results in percentage are reported relative to our SFS-DPO baseline.

[Table 4](#S5.T4 "In 5.2 The Role of Initialization Stage ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
compares our step-level initialization with three alternatives: no
initialization, joint training of reasoning and self-correction ability,
and standard RL initialization. Removing the initialization stage leads
to clear performance degradation, showing that directly optimizing
self-correction signals without a strong reasoning foundation is
insufficient. In addition, joint training is also detrimental,
suggesting that step-wise reasoning and self-correction are better
learned sequentially rather than as a single mixed objective. Standard
RL initialization further underperforms Step RL, indicating that
step-level preference optimization provides a stronger foundation for
later self-correction training. Furthermore, the degradation is
especially pronounced on the more challenging MATH benchmark and in the
SFT setting. Overall, these results highlight the importance of
establishing strong step-level reasoning before optimizing
self-correction behavior.

Figure 4: Distribution of self-correction behavior under SFS-DPO-R,
averaged over six backbone models.

[TABLE]

Table 5: Qualitative reasoning between Qwen2-7B-SFT and
Qwen2-7B-SFT-SFS-DPO-R.

### 5.3 Self-Correction Rate

We study the role of the self-correction (SC) rate in model performance.
As shown in
[Figure 2](#S4.F2 "In 4.3 Out-Of-Domain Results (OOD) ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
the SC rate exhibits a positive correlation with overall performance,
indicating that the self-correction ability of our framework plays an
important role in improving reasoning accuracy. However, a high SC rate
does not necessarily translate into better accuracy. Across MATH and
GSM8K, some self-correction methods achieve high SC rates while still
underperforming in final task accuracy
([Table 3](#S5.T3 "In 5.1 Comparison to Self-Correction Baselines ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")),
suggesting that excessive self-correction may reflect forced correction
behavior rather than effective reasoning improvement. In contrast, our
methods achieve stronger overall performance despite lower SC rates,
indicating that our framework learns to apply self-correction more
selectively and effectively. Rather than maximizing correction
frequency, our method encourages the model to self-correct primarily on
more challenging examples where the original reasoning is likely to
fail. This suggests that effective self-correction requires not only
knowing how to self-correct, but also knowing when not to self-correct.
Overall, these findings indicate that SC rate alone is insufficient as a
standalone indicator of self-correction quality. What ultimately matters
is whether self-correction behavior is positively aligned with task
performance, such that revisions reinforce correct reasoning instead of
introducing unnecessary or spurious changes.

### 5.4 Self-Correction Behavior

[Figure 3](#S4.F3 "In 4.3 Out-Of-Domain Results (OOD) ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
and
[Figure 4](#S5.F4 "In 5.2 The Role of Initialization Stage ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
present a detailed analysis of the self-correction behavior exhibited by
the models on GSM8K and MATH.
[Figure 3](#S4.F3 "In 4.3 Out-Of-Domain Results (OOD) ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
analyzes the distribution of the total number of self-correction steps
in solutions, conditioned on the presence of the self-correction signal.
It shows that most solutions contain only a few self-correction steps,
with single-correction cases being most common and the frequency
decreasing as corrections increase. Cases with more than three
corrections are relatively uncommon, suggesting that the model does not
rely on repeated or forced revisions.
[Figure 4](#S5.F4 "In 5.2 The Role of Initialization Stage ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
further shows that self-correction occurrences broadly follow the
distribution of reasoning lengths rather than concentrating at specific
positions, indicating no strong positional bias in self-correction
behavior toward specific reasoning stages. Overall, these results
suggest that SFS-DPO-R encourages stable and targeted self-correction
behavior during multi-step reasoning, providing a foundation for
learning when to self-correct effectively and selectively.

### 5.5 Case Study

As shown in
[Table 5](#S5.T5 "In 5.2 The Role of Initialization Stage ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
the base Qwen2-7B-SFT model makes an error in its calculation and
repeatedly generates incorrect reasoning without effectively resolving
the mistake. In contrast, SFS-DPO-R enables the model to detect the
erroneous reasoning step, localize the calculation error in the middle
of the solution, and explicitly reason about the source of the mistake.
Additional qualitative examples are provided in the
[Table 13](#A4.T13 "In Appendix D Qualitative Analysis of SFS-DPO-R Self-Correction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").

## 6 Conclusion

In this work, we propose SFS-DPO, a two-stage framework that enables
small LLMs to explicitly detect and correct erroneous reasoning steps
during inference. The framework builds on step-level preference
optimization, with an RL-based initialization to strengthen step-wise
reasoning, followed by targeted training for self-correction. We further
introduce SFS-DPO-R, a teacher-assisted variant that incorporates
explanatory rationales to provide stronger corrective signals.
Comprehensive experiments across multiple model backbones demonstrate
consistent improvements over prior methods on both in-domain and
out-of-distribution benchmarks. These results highlight that explicitly
modeling how errors are identified and fixed, rather than merely
preferring better continuations, is critical for robust math reasoning,
positioning step-wise self-correction as a promising direction for
improving the reliability of large language models.

## Limitations

While our framework represents an advancement in self-correction
capabilities during inference of LLMs, several limitations persist.
Firstly, SFS-DPO-R relies on stronger models’ generated rationales,
introducing additional teacher dependence and potential propagation of
teacher biases or errors. In addition, our evaluation focuses on
mathematical reasoning tasks, where intermediate reasoning steps are
naturally well-defined. The generalization of the proposed method to
more open-ended settings, such as creative writing, has not been fully
explored. Future work will explore scaling this framework to more
complex datasets. Finally, we examined moderately sized LLMs, ranging
from 7 to 14 billion parameters. Experiments with larger and more
capable models could strengthen our claims.

## Acknowledgement

This research is supported by the RIE2025 Industry Alignment Fund –
Industry Collaboration Projects (IAF-ICP)(Award I2301E0026),
administered by A\*STAR, as well as supported by Alibaba Group and NTU
Singapore through Alibaba-NTU Global e-Sustainability CorpLab (ANGEL).
Do Xuan Long is supported by the A\*STAR Computing and Information
Science (ACIS) Scholarship.

## References

- Caruana (1997) R. Caruana Multitask learning. Machine learning 28 (1),
  pp. 41–75. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§3](#S3.SS0.SSS0.Px2.p1.1 "Motivation. ‣ 3 Methodology ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Chen et al. (2024) G. Chen, M. Liao, C. Li, and K. Fan Step-level
  value preference optimization for mathematical reasoning. In Findings
  of the Association for Computational Linguistics: EMNLP 2024, Y.
  Al-Onaizan, M. Bansal, and Y. Chen (Eds.), Miami, Florida, USA,
  pp. 7889–7903. External Links:
  [Link](https://aclanthology.org/2024.findings-emnlp.463/),
  [Document](https://dx.doi.org/10.18653/v1/2024.findings-emnlp.463)
  Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Chen et al. (2025) Q. Chen, L. Qin, J. Liu, D. Peng, J. Guan, P.
  Wang, M. Hu, Y. Zhou, T. Gao, and W. Che Towards reasoning era: a
  survey of long chain-of-thought for reasoning large language models.
  arXiv preprint arXiv:2503.09567. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Cobbe et al. (2021) K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H.
  Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C.
  Hesse, and J. Schulman Training verifiers to solve math word problems.
  External Links: 2110.14168, [Link](https://arxiv.org/abs/2110.14168)
  Cited by:
  [§4.1](#S4.SS1.SSS0.Px2.p2.1 "Dataset Construction & Training Setup. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Do et al. (2024) X. L. Do, D. N. Yen, L. A. Tuan, K. Kawaguchi, M.
  Kan, and N. Chen Multi-expert prompting improves reliability, safety
  and usefulness of large language models. In Proceedings of the 2024
  Conference on Empirical Methods in Natural Language Processing,
  pp. 20370–20401. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Grattafiori et al. (2024) A. Grattafiori, A. Dubey, A. Jauhri, A.
  Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A.
  Vaughan, et al. The llama 3 herd of models. arXiv preprint
  arXiv:2407.21783. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Hendrycks et al. (2021) D. Hendrycks, C. Burns, S. Kadavath, A.
  Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt Measuring
  mathematical problem solving with the MATH dataset. In Thirty-fifth
  Conference on Neural Information Processing Systems Datasets and
  Benchmarks Track (Round 2), External Links:
  [Link](https://openreview.net/forum?id=7Bywt2mQsCe) Cited by:
  [§4.1](#S4.SS1.SSS0.Px2.p2.1 "Dataset Construction & Training Setup. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Hong et al. (2024) J. Hong, N. Lee, and J. Thorne ORPO: monolithic
  preference optimization without reference model. In Proceedings of the
  2024 Conference on Empirical Methods in Natural Language
  Processing, Y. Al-Onaizan, M. Bansal, and Y. Chen (Eds.), Miami,
  Florida, USA, pp. 11170–11189. External Links:
  [Link](https://aclanthology.org/2024.emnlp-main.626/),
  [Document](https://dx.doi.org/10.18653/v1/2024.emnlp-main.626) Cited
  by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Hurst et al. (2024) A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A.
  Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et
  al. Gpt-4o system card. arXiv preprint arXiv:2410.21276. Cited by:
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Dataset Construction & Training Setup. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Jin et al. (2025) Z. Jin, X. Li, Y. Ji, C. Peng, Z. Liu, Q. Shi, Y.
  Yan, S. Wang, F. Peng, and G. Yu ReCUT: balancing reasoning length and
  accuracy in LLMs via stepwise trails and preference optimization. In
  Findings of the Association for Computational Linguistics: EMNLP
  2025, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng
  (Eds.), Suzhou, China, pp. 14269–14282. External Links:
  [Link](https://aclanthology.org/2025.findings-emnlp.770/),
  [Document](https://dx.doi.org/10.18653/v1/2025.findings-emnlp.770),
  ISBN 979-8-89176-335-7 Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Kumar et al. (2024) A. Kumar, V. Zhuang, R. Agarwal, Y. Su, J. D.
  Co-Reyes, A. Singh, K. Baumli, S. Iqbal, C. Bishop, R. Roelofs, et al.
  Training language models to self-correct via reinforcement learning.
  arXiv preprint arXiv:2409.12917. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§1](#S1.p2.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§5.1](#S5.SS1.p1.1 "5.1 Comparison to Self-Correction Baselines ‣ 5 Discussions ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Lahlou et al. (2025) S. Lahlou, A. Abubaker, and H. Hacid Port:
  preference optimization on reasoning traces. In Proceedings of the
  2025 Conference of the Nations of the Americas Chapter of the
  Association for Computational Linguistics: Human Language Technologies
  (Volume 1: Long Papers), pp. 10989–11005. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Lai et al. (2024) X. Lai, Z. Tian, Y. Chen, S. Yang, X. Peng, and J.
  Jia Step-dpo: step-wise preference optimization for long-chain
  reasoning of llms. arXiv preprint arXiv:2406.18629. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§3.1](#S3.SS1.p3.1 "3.1 Initialization Stage ‣ 3 Methodology ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Dataset Construction & Training Setup. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§4.1](#S4.SS1.SSS0.Px2.p2.1 "Dataset Construction & Training Setup. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Lewkowycz et al. (2022) A. Lewkowycz, A. Andreassen, D. Dohan, E.
  Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T.
  Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra Solving
  quantitative reasoning problems with language models. In Proceedings
  of the 36th International Conference on Neural Information Processing
  Systems, NIPS ’22, Red Hook, NY, USA. External Links: ISBN
  9781713871088 Cited by:
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Liao et al. (2024) M. Liao, C. Li, W. Luo, W. Jing, and K. Fan MARIO:
  MAth reasoning with code interpreter output - a reproducible pipeline.
  In Findings of the Association for Computational Linguistics: ACL
  2024, L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand,
  pp. 905–924. External Links:
  [Link](https://aclanthology.org/2024.findings-acl.53/),
  [Document](https://dx.doi.org/10.18653/v1/2024.findings-acl.53) Cited
  by:
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Lightman et al. (2024) H. Lightman, V. Kosaraju, Y. Burda, H.
  Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K.
  Cobbe Let’s verify step by step. In The Twelfth International
  Conference on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=v8L0pN6EOi) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Liu et al. (2025) H. Liu, Y. Zhang, Y. Luo, and A. C. Yao Augmenting
  math word problems via iterative question composing. In Proceedings of
  the AAAI Conference on Artificial Intelligence, Vol. 39,
  pp. 24605–24613. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Lu et al. (2024) Z. Lu, A. Zhou, K. Wang, H. Ren, W. Shi, J. Pan, M.
  Zhan, and H. Li Step-controlled dpo: leveraging stepwise error for
  enhanced mathematical reasoning. arXiv preprint arXiv:2407.00782.
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Ma et al. (2025) R. Ma, P. Wang, C. Liu, X. Liu, J. Chen, B. Zhang, X.
  Zhou, N. Du, and J. Li S$`{}^{2}`$R: teaching LLMs to self-verify and
  self-correct via reinforcement learning. In Proceedings of the 63rd
  Annual Meeting of the Association for Computational Linguistics
  (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T.
  Pilehvar (Eds.), Vienna, Austria, pp. 22632–22654. External Links:
  [Link](https://aclanthology.org/2025.acl-long.1104/),
  [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.1104), ISBN
  979-8-89176-251-0 Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Pan et al. (2025) Z. Pan, Y. Li, H. Lin, Q. Pei, Z. Tang, W. Wu, C.
  Ming, H. V. Zhao, C. He, and L. Wu LEMMA: learning from errors for
  MatheMatical advancement in LLMs. In Findings of the Association for
  Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova,
  and M. T. Pilehvar (Eds.), Vienna, Austria, pp. 11615–11639. External
  Links: [Link](https://aclanthology.org/2025.findings-acl.605/),
  [Document](https://dx.doi.org/10.18653/v1/2025.findings-acl.605), ISBN
  979-8-89176-256-5 Cited by: [Figure
  6](#A3.F6 "In Appendix C Data Construction Details ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [Appendix
  C](#A3.p1.1 "Appendix C Data Construction Details ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Dataset Construction & Training Setup. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Pham et al. (2026) H. Pham, D. Le, and A. T. Luu GRACE: step-level
  benchmark for faithful reasoning over context. arXiv preprint
  arXiv:2606.16151. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Rafailov et al. (2023) R. Rafailov, A. Sharma, E. Mitchell, C. D.
  Manning, S. Ermon, and C. Finn Direct preference optimization: your
  language model is secretly a reward model. Advances in neural
  information processing systems 36, pp. 53728–53741. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Shao et al. (2024) Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H.
  Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: pushing the limits
  of mathematical reasoning in open language models. arXiv preprint
  arXiv:2402.03300. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Xu et al. (2025) H. Xu, X. Mao, F. Li, X. Wu, W. Chen, W. Zhang,
  and A. T. Luu Full-step-DPO: self-supervised preference optimization
  with step-wise rewards for mathematical reasoning. In Findings of the
  Association for Computational Linguistics: ACL 2025, W. Che, J.
  Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,
  pp. 24343–24356. External Links:
  [Link](https://aclanthology.org/2025.findings-acl.1249/),
  [Document](https://dx.doi.org/10.18653/v1/2025.findings-acl.1249),
  ISBN 979-8-89176-256-5 Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM Reasoning ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yan et al. (2025) Y. Yan, J. Jiang, Y. Liu, Y. Cao, X. Xu, M.
  Zhang, X. Cai, and J. Shao Sˆ 3cmath: spontaneous step-level
  self-correction makes large language models better mathematical
  reasoners. In Proceedings of the AAAI Conference on Artificial
  Intelligence, Vol. 39, pp. 25588–25596. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yang et al. (2025a) A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B.
  Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report.
  arXiv preprint arXiv:2505.09388. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yang et al. (2024a) A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C.
  Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J.
  Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J.
  Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M.
  Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S.
  Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X.
  Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y.
  Wan, Y. Chu, Z. Cui, Z. Zhang, and Z. Fan Qwen2 technical report.
  ArXiv abs/2407.10671. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:271212307) Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yang et al. (2024b) A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B.
  Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J.
  Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K.
  Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T.
  Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y.
  Liu, Z. Cui, Z. Zhang, and Z. Qiu Qwen2.5 technical report. CoRR
  abs/2412.15115. External Links:
  [Link](https://doi.org/10.48550/arXiv.2412.15115),
  [Document](https://dx.doi.org/10.48550/ARXIV.2412.15115), 2412.15115
  Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yang et al. (2024c) A. Yang, B. Zhang, B. Hui, B. Gao, B. Yu, C.
  Li, D. Liu, J. Tu, J. Zhou, J. Lin, et al. Qwen2. 5-math technical
  report: toward mathematical expert model via self-improvement. arXiv
  preprint arXiv:2409.12122. Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yang et al. (2025b) L. Yang, Z. Yu, T. Zhang, M. Xu, J. E.
  Gonzalez, B. CUI, and S. YAN SuperCorrect: advancing small LLM
  reasoning with thought template distillation and self-correction. In
  The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=PyjZO7oSw2)
  Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Yu et al. (2024) L. Yu, W. Jiang, H. Shi, J. YU, Z. Liu, Y. Zhang, J.
  Kwok, Z. Li, A. Weller, and W. Liu MetaMath: bootstrap your own
  mathematical questions for large language models. In The Twelfth
  International Conference on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=N8N0hgNDRt) Cited by:
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Main Experiments ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Zeng et al. (2025) Y. Zeng, X. Cui, X. Jin, G. Liu, Z. Sun, D. Li, N.
  Yang, J. Hao, H. Zhang, and J. Wang Evolving llms’ self-refinement
  capability via iterative preference optimization. arXiv preprint
  arXiv:2502.05605. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
- Zhao et al. (2025) X. Zhao, T. Xu, X. Wang, Z. Chen, D. Jin, L.
  Tan, Z. Yu, Z. Zhao, Y. He, S. Wang, et al. Boosting llm reasoning via
  spontaneous self-correction. arXiv preprint arXiv:2506.06923. Cited
  by:
  [§1](#S1.p2.1 "1 Introduction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
  [§2.2](#S2.SS2.p1.1 "2.2 Teaching LLMs to Self-Correct ‣ 2 Related Work ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").

## Appendix A Choice of Statistical Test

For the main benchmark results, where the evaluation outcomes are binary
(correct/incorrect), we adopt McNemar’s test for the significant
testing. This test is particularly suitable for paired binary evaluation
settings, as it directly compares the prediction differences between two
systems on the same evaluation samples. Specifically, McNemar’s test
focuses on discordant pairs, i.e., cases where the prediction outcomes
differ between our method and the Step-DPO baseline. We use a one-sided
exact McNemar’s test under the hypothesis that our method improves over
the baseline and report results with $`p<0.05`$ as statistically
significant.

We only report McNemar’s test results for the main benchmarks due to
their relatively larger evaluation sizes. For smaller OOD benchmarks, we
do not report significance results since the limited number of
evaluation samples makes McNemar’s test statistically underpowered and
less reliable.

## Appendix B Additional Analysis

We present additional analysis of self-correction behavior before and
after applying SFS-DPO and SFS-DPO-R in
[Table 6](#A2.T6 "In Appendix B Additional Analysis ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
Before training, several backbone models, including DeepSeekMath-7B-SFT,
Qwen2-7B-SFT and Qwen2-7B-Instruct show limited self-correction ability.
After training, SFS-DPO and SFS-DPO-R generally improve self-correction
behavior on these models. However, a higher self-correction rate does
not necessarily imply better reasoning performance, as shown by
Llama-3.1-8B-Instruct and Qwen3-8B, where the self-correction rate
decreases after training while the overall reasoning performance
improves. This suggests that effective self-correction depends on
selective and accurate revision rather than frequency alone.

Figure [5](#A2.F5 "Figure 5 ‣ Appendix B Additional Analysis ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")
further analyzes epoch-wise behavior on the two SFT backbones. For
DeepSeekMath-7B-SFT, both self-correction rate and task accuracy peak at
epoch 2, suggesting positive alignment between correction behavior and
reasoning performance. In contrast, Qwen2-7B-SFT shows relatively stable
self-correction rates with only minor accuracy fluctuations, again
indicating that correction effectiveness and selectivity matter more
than raw correction frequency.

|                        |      |         |           |
| ---------------------- | ---- | ------- | --------- |
| Model                  | Base | SFS-DPO | SFS-DPO-R |
| DeepSeekMath-7B-SFT    | 8.1  | 14.2    | 19.1      |
| Qwen2-7B-SFT           | 8.0  | 14.2    | 13.1      |
| Qwen2-7B-Instruct      | 5.3  | 12.5    | 14.9      |
| Qwen2.5-7B-Math-Instr. | 21.0 | 35.7    | 42.1      |
| Qwen3-8B               | 35.8 | 26.9    | 28.8      |
| Llama-3.1-8B-Instruct  | 37.0 | 28.8    | 23.9      |
| Qwen-2.5-14B-Instruct  | 37.5 | 43.8    | 46.6      |

Table 6: Self-correction rates of SFS-DPO and SFS-DPO-R across different
backbone models.

Figure 5: Self-correction rate of SFS-DPO-R under two SFT models.

## Appendix C Data Construction Details

We provide additional details on our data construction process. As shown
in
[Figure 6](#A3.F6 "In Appendix C Data Construction Details ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs"),
we use a set of transition signals adapted from [Pan et al.
(2025)](#bib.bib8) to indicate the beginning of self-correction
behavior. We further extend this list with additional explicit phrases,
such as “The previous step is incorrect.” and “There is a mistake.”, to
encourage clearer error-detection behavior before generating the
corrected reasoning step.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTMuRjYucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIiBoZWlnaHQ9IjUxMy43OSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDY1My4xNSA1MTMuNzkiIHdpZHRoPSI2NTMuMTUiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsNTEzLjc5KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzQwNDA0MDsiIGZpbGw9IiM0MDQwNDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCA1LjkxIEwgMCA1MDcuODggQyAwIDUxMS4xNSAyLjY0IDUxMy43OSA1LjkxIDUxMy43OSBMIDY0Ny4yNCA1MTMuNzkgQyA2NTAuNTEgNTEzLjc5IDY1My4xNSA1MTEuMTUgNjUzLjE1IDUwNy44OCBMIDY1My4xNSA1LjkxIEMgNjUzLjE1IDIuNjQgNjUwLjUxIDAgNjQ3LjI0IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjJGMkYyOyIgZmlsbD0iI0YyRjJGMiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDQ4OS42OCBMIDY1MS4xOCA0ODkuNjggTCA2NTEuMTggNS45MSBDIDY1MS4xOCAzLjczIDY0OS40MiAxLjk3IDY0Ny4yNCAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiA0OTguMjcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzkuNzhlbTstLWx0eC1mby1oZWlnaHQ6MC42OWVtOy0tbHR4LWZvLWRlcHRoOjAuMTllbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTIuMyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOS42MSkiIHdpZHRoPSI1NTAuNDQiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkEzLkY2LnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozOS43OGVtOyI+CjxzcGFuIGlkPSJBMy5GNi5waWMxLjEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuRjYucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+4oCcU2VsZi1GaXggU3RlcC1EUE/igJ0gUHJvbXB0IENvbnN0cnVjdGlvbjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMC4wNiAxMC4wNikiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDo0NS43NWVtOy0tbHR4LWZvLWhlaWdodDozNC4wOGVtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNDcxLjU0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA0NzEuNTQpIiB3aWR0aD0iNjMzLjA1Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMy5GNi5waWMxLjIiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6NDUuNzVlbTsiPgo8c3BhbiBpZD0iQTMuRjYucGljMS4yLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkY2LnBpYzEuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPntRdWVzdGlvbn08c3BhbiBpZD0iQTMuRjYucGljMS4yLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfbWVkaXVtIj4uIExldOKAmXMgdGhpbmsgc3RlcCBieSBzdGVwLjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuRjYucGljMS4yLjIiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkY2LnBpYzEuMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPntBbm5vdGF0ZWQgSW5jb3JyZWN0IFJlYXNvbmluZyBTdGVwc308c3BhbiBpZD0iQTMuRjYucGljMS4yLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQiPjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuRjYucGljMS4yLjMiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkY2LnBpYzEuMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPntDaG9vc2Ugb25lIHRyYW5zaXRpb24gcGhyYXNlIGJlbG93fTxzcGFuIGlkPSJBMy5GNi5waWMxLjIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCI+PC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMSIgY2xhc3M9Imx0eF9pdGVtaXplIj4KPHNwYW4gaWQ9IkEzLkkxLmkxIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkxLmkxLnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEuaTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTEucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+QnV0LCB3YWl0LCBsZXTigJlzIHBhdXNlIGFuZCBleGFtaW5lIHRoaXMgbW9yZSBjYXJlZnVsbHkuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmkyIiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkxLmkyLnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEuaTIucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTIucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+V2FpdCBhIHNlY29uZCwgbGV04oCZcyBlbnN1cmUgdGhpcyBpcyByaWdodC4gQ2FsY3VsYXRpbmcgY2FyZWZ1bGx5Ojwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMS5pMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMS5pMy5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkkxLmkzLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxLmkzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkhtbSwgSSB3YW50IHRvIHZlcmlmeSB0aGlzIGNhbGN1bGF0aW9uLiBMZXTigJlzIGdvIHRocm91Z2ggaXQ6PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmk0IiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkxLmk0LnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEuaTQucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTQucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+V2FpdCwgdGhpcyBkb2VzbuKAmXQgc2VlbSByaWdodC4gTGV04oCZcyBwYXVzZSBhbmQgY29uc2lkZXIgdGhpczo8L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTEuaTUiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTUucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pNS5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JMS5pNS5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5MZXTigJlzIHBhdXNlIGFuZCBjb25zaWRlciB3aGF0IHdlIGtub3cgc28gZmFyLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMS5pNiIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMS5pNi5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkkxLmk2LnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxLmk2LnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoaXMgZGlkbuKAmXQgc2VlbSByaWdodC4gV2FpdCwgbGV04oCZcyBjb3JyZWN0IHRoYXQuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmk3IiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkxLmk3LnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEuaTcucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTcucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+V2FpdCwgc29tZXRoaW5nIHNlZW1zIG9mZi4gTGV04oCZcyBwYXVzZSBhbmQgY29uc2lkZXIgd2hhdCB3ZSBrbm93IHNvIGZhci48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTEuaTgiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTgucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pOC5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JMS5pOC5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5MZXTigJlzIHBhdXNlIGFuZCBjb25zaWRlciBpZiB3ZeKAmXZlIHNldCB1cCBldmVyeXRoaW5nIGNvcnJlY3RseS48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTEuaTkiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTkucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pOS5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JMS5pOS5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5XYWl0IGEgc2Vjb25kLiBJcyBldmVyeXRoaW5nIGNvcnJlY3Q/IExldCBtZSBkb3VibGUtY2hlY2suPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmkxMCIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMS5pMTAucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pMTAucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTEwLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPldhaXQsIG1heWJlIHRoZXJl4oCZcyBzb21ldGhpbmcgd3JvbmcuIExldOKAmXMgcGF1c2UgYW5kIHJlY29uc2lkZXIuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmkxMSIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMS5pMTEucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pMTEucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTExLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSByZXN1bHQgbG9va3Mgc3RyYW5nZSwgaXMgZXZlcnl0aGluZyBjb3JyZWN0PyBMZXQgbWUgZG91YmxlLWNoZWNrLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMS5pMTIiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTEyLnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEuaTEyLnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxLmkxMi5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5Eb2VzIHRoaXMgbWFrZSBzZW5zZT8gTGV04oCZcyByZXRoaW5rIHRoaXMuPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmkxMyIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMS5pMTMucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pMTMucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTEzLnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPkNvdWxkIEkgaGF2ZSBtaXNzZWQgc29tZXRoaW5nPyBMZXTigJlzIHBhdXNlIGFuZCBjb25zaWRlciB3aGF0IHdlIGtub3cgc28gZmFyLjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMy5JMS5pMTQiIGNsYXNzPSJsdHhfaXRlbSIgc3R5bGU9Imxpc3Qtc3R5bGUtdHlwZTpub25lOyI+PHNwYW4gY2xhc3M9Imx0eF90YWcgbHR4X3RhZ19pdGVtIj7igKI8L3NwYW4+IAo8c3BhbiBpZD0iQTMuSTEuaTE0LnAxIiBjbGFzcz0ibHR4X3BhcmEiPgo8c3BhbiBpZD0iQTMuSTEuaTE0LnAxLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkEzLkkxLmkxNC5wMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij5XYWl0IDwvc3Bhbj48bWF0aCBpZD0iQTMuSTEuaTE0LnAxLm0xIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxzdGFyIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIj7ii4Y8L21vPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XHN0YXI8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkkxLmkxNSIgY2xhc3M9Imx0eF9pdGVtIiBzdHlsZT0ibGlzdC1zdHlsZS10eXBlOm5vbmU7Ij48c3BhbiBjbGFzcz0ibHR4X3RhZyBsdHhfdGFnX2l0ZW0iPuKAojwvc3Bhbj4gCjxzcGFuIGlkPSJBMy5JMS5pMTUucDEiIGNsYXNzPSJsdHhfcGFyYSI+CjxzcGFuIGlkPSJBMy5JMS5pMTUucDEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuSTEuaTE1LnAxLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPlRoZSBwcmV2aW91cyBzdGVwIGlzIGluY29ycmVjdC4gPC9zcGFuPjxtYXRoIGlkPSJBMy5JMS5pMTUucDEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXHN0YXIiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiPuKLhjwvbW8+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cc3RhcjwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTMuSTEuaTE2IiBjbGFzcz0ibHR4X2l0ZW0iIHN0eWxlPSJsaXN0LXN0eWxlLXR5cGU6bm9uZTsiPjxzcGFuIGNsYXNzPSJsdHhfdGFnIGx0eF90YWdfaXRlbSI+4oCiPC9zcGFuPiAKPHNwYW4gaWQ9IkEzLkkxLmkxNi5wMSIgY2xhc3M9Imx0eF9wYXJhIj4KPHNwYW4gaWQ9IkEzLkkxLmkxNi5wMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5JMS5pMTYucDEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+VGhlcmUgaXMgYSBtaXN0YWtlLiA8L3NwYW4+PG1hdGggaWQ9IkEzLkkxLmkxNi5wMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcc3RhciIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCI+4ouGPC9tbz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxzdGFyPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj4KPHNwYW4gaWQ9IkEzLkY2LnBpYzEuMi40IiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5GNi5waWMxLjIuNC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij57QW5ub3RhdGVkIENvcnJlY3RlZCBDb250aW51YXRpb259PHNwYW4gaWQ9IkEzLkY2LnBpYzEuMi40LjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSI+IAo8YnIgY2xhc3M9Imx0eF9icmVhayI+PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

Figure 6: Dataset construction for SFSDPO with the set of transition
phrases fetched from [Pan et al. (2025)](#bib.bib8). $`\star`$ denotes
additional phrases.

![Refer to caption](2608.11573v2/SFSDPO_Example.png)

Figure 7: Illustration of the process for constructing the SFS-DPO-R
dataset. Incorrect intermediate reasoning steps are identified and
corrected with self-correction signals and self-correction reasonings.

## Appendix D Qualitative Analysis of SFS-DPO-R Self-Correction

The qualitative examples are provided in
Tables [7](#A4.T7 "Table 7 ‣ Appendix D Qualitative Analysis of SFS-DPO-R Self-Correction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs")–[13](#A4.T13 "Table 13 ‣ Appendix D Qualitative Analysis of SFS-DPO-R Self-Correction ‣ Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs").
We include one comparison with LEMMA and S²R, where these methods
exhibit self-correction behavior but still produce excessive or
insufficiently targeted revisions. The remaining examples focus on the
base model and Step-DPO. The base model often propagates earlier errors
without explicit repair, while Step-DPO can improve local step
preference but still fails to revise already-generated incorrect steps.
In contrast, SFS-DPO-R more reliably localizes erroneous intermediate
steps and applies concise corrections while preserving coherent
reasoning. These examples support that effective self-correction
requires selective and accurate revision, not correction frequency
alone.

[TABLE]

Table 7: Qualitative reasoning between base Qwen2-7B-Instruct and
Qwen2-7B-Instruct-SFS-DPO-R, where the base model produces a plausible
but incorrect solution, while SFS-DPO-R identifies the error source and
revises the reasoning to obtain the correct result.

|          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Problem  | Annie’s soccer team is conducting a passing drill in which each of the 11 members of the soccer team must pass the ball to each of the other members exactly three times. How many times must the ball be passed before the passing drill is over? Please reason step by step, and put your final answer within \boxed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Step-DPO | \## Step 1: Determine the total number of passes each player must make Each of the 11 players must pass the ball to each of the other 10 players exactly three times. This means each player will make $`10*3=30`$ passes in total. \## Step 2: Calculate the total number of passes made by all players Since there are 11 players, and each player makes 30 passes, it might seem like the total number of passes is $`11*30`$. However, this counts each pass twice (once for each player involved in the pass), so we need to divide this number by 2 to get the actual number of passes. \## Step 3: Calculate the total number of passes The total number of passes is $`(11*30)/2=165`$. The final answer is: $`\boxed{165}`$                                                                                                                                                                                                                                                                                          |
| S²R      | … The total number of unique passes is 330 / 2 = 165. Wait, let me recheck my solution. To verify the answer, we can consider the structure of the problem and the nature of the passing drill. The problem involves 11 players, and each player must pass the ball to each of the other 10 players exactly three times. This means that each player will make 10 passes, and since there are 11 players, it might seem like there would be 11 x 10 = 110 passes. However, this counts each pass twice (once for each player involved in the pass), so we need to divide by 2 to get the correct number of unique passes. The correct calculation is (11 players x 10 passes/player) / 2 = 55 unique passes. Since each player must pass the ball 3 times to each of the other 10 players, we need to multiply the number of unique passes by 3 to get the total number of passes. 55 unique passes x 3 = 165 passes.The answer provided is 165, which matches the result of this reasoning. Therefore, the answer is correct |
| LEMMA    | … This means there are 55 unique pairs of players. Now, since each player must pass the ball to each of the other players three times, we need to multiply the number of unique pairs by the number of passes each player makes:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

```math
\text{Total passes}=3\times 55=165.
```

Hmm, let’s verify this calculation. Each of the 11 players is passing to 10 other players, and since each pass is counted three times, we can also express it as:

```math
\text{Total passes}=3\times(11\times 10)=3\times 110=330.
```

Wait a second. This seems inconsistent with our earlier calculation. Let’s clarify: we only need to count the unique pairs once, and then multiply by the number of passes per pair. So, the correct approach is indeed to calculate the total number of unique pairs and multiply by the number of passes per pair:

```math
\text{Total passes}=3\times 55=165.
```

Thus, the total number of times the ball must be passed before the drill is over is indeed $`\boxed{165}`$. |
| SFS-DPO-R | \## … \## Step 2: Calculate the total number of passes made by all players in one round of passing. If each of the 11 players passes to the other 10 players exactly 3 times, it might initially seem like the calculation is $`11`$ players $`*`$ $`10`$ players $`*3`$ passes $`=330`$ passes. However, this counts each pass twice because when one player passes to another, it counts as a pass for both the passer and the receiver. So, we need to divide this number by 2 to get the actual number of unique passes in one round. \## Step 3: Calculate the total number of passes needed for the drill. The calculation from Step 2, $`11*10*3`$, is the correct way to find the total number of passes needed because it correctly accounts for the fact that each of the 11 players must pass to the other 10 players exactly 3 times, without double-counting the passes. So, the total number of passes needed is $`11*10*3`$ = 330. The final answer is: $`\boxed{330}`$ |

Table 8: Qualitative comparison on the MATH dataset between the base
Llama-3.1-8B-Instruct-Step-DPO model and
Llama-3.1-8B-Instruct-SFS-DPO-R. S²R cannot perform correct
self-correction while LEMMA exhibits excessive self-correction by
repeatedly revising turning a correct reasoning path into an incorrect
final answer, while SFS-DPO-R performs a targeted correction and reaches
the correct answer.

[TABLE]

Table 9: Qualitative comparison on the MATH dataset between the base
Llama-3.1-8B-Instruct-Step-DPO model and
Llama-3.1-8B-Instruct-SFS-DPO-R. While Step-DPO arrives at an incorrect
conclusion due to flawed inequality reasoning, SFS-DPO-R identifies the
mistake, revises its reasoning, and recovers the correct maximum value.

[TABLE]

Table 10: Qualitative comparison on the Gaokao2023 dataset between the
base Qwen2-7B-SFT model and Qwen2-7B-SFT-SFS-DPO-R. While the base model
produces a seemingly coherent but incomplete reasoning trajectory,
SFS-DPO-R explicitly identifies the source of error at the step level
and revises the flawed reasoning, ultimately leading to the correct
final answer.

|            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Problem    | A chord of a circle is perpendicular to a radius at the midpoint of the radius. The ratio of the area of the larger of the two regions into which the chord divides the circle to the smaller can be expressed in the form $`\displaystyle{{a\pi+b\sqrt{c}}\over{d\pi-e\sqrt{f}}}`$, where $`a`$, $`b`$, $`c`$, $`d`$, $`e`$, and $`f`$ are positive integers, $`a`$ and $`e`$ are relatively prime, and neither $`c`$ nor $`f`$ is divisible by the square of any prime. Find the remainder when the product $`a\cdot b\cdot c\cdot d\cdot e\cdot f`$ is divided by 1000.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Base Model | Step 1: Draw a diagram. Step 2: Label the radius of the circle $`r`$. Step 3: Draw the chord and the two radii that are perpendicular to the chord. … Step 11: Find the product $`a\cdot b\cdot c\cdot d\cdot e\cdot f`$. Step 12: Find the remainder when the product is divided by 1000. The final answer is $`\boxed{120}`$.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| SFS-DPO-R  | Step 1: Given a circle with radius $`R`$, a chord is perpendicular to a radius at the midpoint of the radius. This means the distance from the center of the circle to the chord is $`\frac{R}{2}`$. The length of the chord can be found using the Pythagorean theorem in the right triangle formed by the radius, half the chord, and the distance from the center to the chord. The length of half the chord is $`\sqrt{R^{2}-\left(\frac{R}{2}\right)^{2}}=\sqrt{R^{2}-\frac{R^{2}}{4}}=\sqrt{\frac{3R^{2}}{4}}=\frac{R\sqrt{3}}{2}`$. Therefore, the length of the chord is $`R\sqrt{3}`$. Step 2: The area of the circle is $`\pi R^{2}`$. To find the area of the smaller region formed by the chord, we need to calculate the area of the circular segment. The angle subtended by the chord at the center of the circle can be found using the cosine rule in the triangle formed by two radii and the chord. The cosine of the angle $`\theta`$ is given by $`\cos\theta=\frac{\left(\frac{R}{2}\right)^{2}+\left(\frac{R}{2}\right)^{2}-R^{2}}{2\cdot\frac{R}{2}\cdot\frac{R}{2}}=\frac{\frac{R^{2}}{4}+\frac{R^{2}}{4}-R^{2}}{\frac{R^{2}}{2}}=\frac{\frac{R^{2}}{2}-R^{2}}{\frac{R^{2}}{2}}=\frac{-\frac{R^{2}}{2}}{\frac{R^{2}}{2}}=-1`$. However, this is incorrect because we need to consider the angle subtended by the chord on the circle, which is $`2\theta=2\cos^{-1}\left(-\frac{1}{2}\right)=2\cdot\frac{2\pi}{3}=\frac{2\pi}{3}`$. Step 2: The area of the sector with angle $`\frac{2\pi}{3}`$ is $`\frac{1}{3}\pi R^{2}`$. The area of the triangle formed by the two radii and the chord is $`\frac{1}{2}R^{2}\sin\left(\frac{2\pi}{3}\right)=\frac{1}{2}R^{2}\cdot\frac{\sqrt{3}}{2}=\frac{R^{2}\sqrt{3}}{4}`$. Therefore, the area of the smaller segment is $`\frac{1}{3}\pi R^{2}-\frac{R^{2}\sqrt{3}}{4}=R^{2}\left(\frac{\pi}{3}-\frac{\sqrt{3}}{4}\right)`$. … Step 6: To find the remainder when the product $`a\cdot b\cdot c\cdot d\cdot e\cdot f`$ is divided by 1000, we identify $`a=8`$, $`b=3`$, $`c=3`$, $`d=4`$, $`e=3`$, and $`f=3`$. The product is $`8\cdot 3\cdot 3\cdot 4\cdot 3\cdot 3=2592`$. The remainder when 2592 is divided by 1000 is $`592`$. Thus, the answer is $`\boxed{592}`$. |

Table 11: Qualitative comparison on the MATH dataset between the base
Qwen2.5-Math-7B-Instruct model and Qwen2.5-Math-7B-Instruct-SFS-DPO-R.
SFS-DPO-R demonstrates explicit self-correction by detecting and
revising an incorrect intermediate reasoning step, ultimately recovering
the correct solution, while the base model produces an incorrect answer
without correction.

[TABLE]

Table 12: Qualitative comparison on the MATH dataset between the base
Qwen2.5-14B-Instruct model and Qwen2.5-14B-Instruct-SFS-DPO-R. SFS-DPO-R
exhibits self-correction by identifying that the initial decomposition
of rectangle ABEF is incomplete, but its revised reasoning remains
internally inconsistent, leading to an incorrect final answer despite
attempting correction.

|           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Problem   | Patrick has three glue sticks that are partially used. One has $`1/6`$ left, the second has $`2/3`$ left, and the third one has $`1/2`$ left. If a glue stick is 12 millimeters long originally, what is the total length of the glue sticks that are not used?                                                                                                                                                                                                                                                                                                                                                                                            |
| Step-DPO  | Step 1: Interpret the given fractions. The response incorrectly treats the fractions left as if they should be converted into fractions already used. It therefore claims that the first, second, and third glue sticks have $`5/6`$, $`1/3`$, and $`1/2`$ used, respectively. Step 2: Calculate the lengths. It calculates $`12\left(\frac{5}{6}\right)=10`$ mm, $`12\left(\frac{1}{3}\right)=4`$ mm, and $`12\left(\frac{1}{2}\right)=6`$ mm. Step 3: Return the result. The total used length is $`10+4+6=20`$ mm. The final answer is $`\boxed{20}`$.                                                                                                  |
| SFS-DPO-R | Step 1: Interpret “not used.” The response initially considers computing the unused length using the complement of a used fraction. But wait—the question says that the glue sticks have $`1/6`$, $`2/3`$, and $`1/2`$ left. Therefore, the fractions left are already the unused portions; they should not be converted into their complements. Step 2: Correct the calculation. The unused lengths are $`12\left(\frac{1}{6}\right)=2`$ mm, $`12\left(\frac{2}{3}\right)=8`$ mm, and $`12\left(\frac{1}{2}\right)=6`$ mm. Step 3: Return the corrected result. Thus, the total unused length is $`2+8+6=16`$ mm, and the final answer is $`\boxed{16}`$. |

Table 13: Qualitative comparison on GSM8K between Qwen3-8B-Step-DPO and
Qwen3-8B-SFS-DPO-R. Step-DPO mistakes the fractions left for fractions
used and returns incorrect answer, whereas SFS-DPO-R identifies the
interpretation error and correctly returns groundtruth answer.
