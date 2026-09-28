---
identifier: arxiv:2602.05145v2
title: "TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"
authors:
  - Jiyoung Park
  - Hankyu Jang
  - Changseok Song
  - Wookeun Jung
published: "2026-02-05T00:06:12+00:00"
url: https://arxiv.org/abs/2602.05145v2
source: arxiv
doi: null
arxiv_id: 2602.05145v2
categories:
  - cs.AI
  - cs.LG
---

# TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference

Jiyoung Park, Hankyu Jang, Changseok Song, and Wookeun Jung^(\*)
^(†)^(†)thanks: ^(\*)Corresponding author. Affiliation: Moreh, Inc.  
Seoul, South Korea  
{jiyoung.park, hankyu.jang, changseok.song, wookeun.jung}@moreh.io

###### Abstract

Speculative decoding can substantially accelerate LLM inference, but
realizing its benefits in practice is challenging due to evolving
workloads. We present TIDE (Temporal Incremental Draft Engine), a
serving-engine-native framework that integrates online draft adaptation
directly into high-performance LLM inference systems. TIDE reuses target
model’s intermediate hidden states generated during inference as
training signals for draft adaptation, thereby avoiding additional
target model computation and serving-time overhead. It employs adaptive
runtime control to activate speculation and draft model training only
when beneficial. TIDE exploits heterogeneous clusters by mapping
inference and training to appropriate GPU classes. Across diverse
real-world workloads, TIDE achieves up to 1.66× throughput over
no-speculation baselines while recovering performance on misaligned
workloads where static draft models degrade throughput. TIDE also
reduces training time by up to 3.02× and storage requirements by 24×
compared to existing draft training approaches, and improves system
throughput by up to 1.22× on heterogeneous GPU clusters.

###### Index Terms: 

speculative decoding, online draft model adaptation, heterogeneous GPU
computing, natural language processing, machine learning, distributed
computing.

## I Introduction

Large language models (LLMs) increasingly achieve state-of-the-art
performance on reasoning-intensive tasks, such as mathematics, code
generation, and scientific problem solving, by scaling test-time
computation—for example, by generating longer reasoning traces or
sampling multiple candidate solutions \[[1](#bib.bib5),
[2](#bib.bib6)\]. As these approaches increase the amount of
autoregressive generation per request, decoding efficiency has become a
central bottleneck for deploying reasoning-oriented LLMs in production
systems \[[3](#bib.bib7)\].

Speculative decoding is one of the most effective techniques for
accelerating LLM inference. By allowing a lightweight draft model to
propose multiple draft tokens that are then verified in parallel by a
target model, speculative decoding can significantly improve throughput
and latency \[[4](#bib.bib8), [5](#bib.bib9)\]. However, these gains
depend on _draft–target alignment_—the degree to which the draft model’s
next-token predictions approximate those of the target model
\[[6](#bib.bib38), [7](#bib.bib45)\]. When alignment is poor, the target
model rejects most drafted tokens, and speculative decoding can even
degrade overall serving performance (see
[Fig. 1](#S1.F1 "In I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")).
Indeed, while major inference service providers support speculative
decoding, the feature is typically disabled by default, with providers
cautioning that it does not guarantee speedups and may even increase
latency or reduce throughput depending on the workload and model
configuration \[[8](#bib.bib1), [9](#bib.bib2), [10](#bib.bib3)\].

Fig. 1: Normalized throughput of speculative decoding with a static
draft model versus TIDE across four non-English Alpaca datasets. The
static draft model, trained on English data, degrades throughput below
the no-speculation baseline on all workloads. TIDE recovers and exceeds
baseline throughput by continuously adapting the draft model to the
current workload.

A fundamental challenge is that draft–target alignment is inherently
workload-dependent. In production LLM services, inference workloads
evolve continuously as user behavior changes, application logic is
updated, and prompt templates are modified. While workload distributions
are globally non-stationary, prior studies show that they exhibit strong
short-term temporal locality, with recent inference history remaining
predictive of near-future requests \[[11](#bib.bib43), [12](#bib.bib42),
[13](#bib.bib44), [14](#bib.bib16)\]. This suggests that alignment can,
in principle, be preserved by continuously and rapidly adapting to
recent inference behavior.

Recent work has explored this opportunity by adapting draft models
online using inference-time signals, for example via online distillation
from target model corrections or logits \[[15](#bib.bib10)\]. While
these approaches demonstrate that alignment can be recovered under
distribution shift, they primarily focus on the learning algorithm
itself. It remains an open question whether online draft training can be
integrated into high-performance inference engines while delivering
sustained end-to-end throughput improvements \[[6](#bib.bib38),
[16](#bib.bib15)\].

In practice, addressing this question requires careful coordination
between online draft model training and inference serving. Online
draft-model training must minimize interference with latency-critical
inference, operate under realistic resource constraints, and be
triggered only when beneficial. Because the performance impact of
speculative decoding varies across workload phases, indiscriminate use
of speculative decoding or draft model training is often unnecessary and
can even be counterproductive \[[17](#bib.bib22), [18](#bib.bib17),
[19](#bib.bib18)\]. Effective deployment therefore requires dynamic
runtime control over _when_ to speculate and _when_ to train, based
solely on signals observable during inference serving.

To address these challenges, we introduce Temporal Incremental Draft
Engine (TIDE), a serving-engine-native framework for adaptive
speculative decoding under evolving workloads. Rather than treating
draft adaptation as an isolated learning problem, TIDE jointly manages
training signal collection, draft model updates, and speculative
decoding decisions entirely within the inference serving engine.

TIDE exploits short-term temporal locality in workload distributions by
incrementally training the draft model on recent inference data, while
dynamically controlling when speculative decoding and draft model
training are beneficial. Crucially, since draft model training requires
the target model’s intermediate hidden states for the corresponding
input tokens as supervision signals, a naïve approach would need to
reload the large target model and recompute forward passes solely for
training—an operation that would dominate training cost given the target
model’s scale. TIDE avoids this recomputation by reusing intermediate
hidden states already produced during target model inference. To deliver
these training signals to the separate draft-model training engine, TIDE
copies them from GPU memory to a host buffer and writes the buffered
data to shared storage. It performs both operations asynchronously and
overlaps them with subsequent target-model computation, keeping
training-signal extraction off the serving critical path and allowing
the draft model to adapt fast enough to keep pace with shifting workload
distributions.

This separation also enables efficient deployment on heterogeneous
hardware. Heterogeneous accelerators are receiving increasing attention
in LLM serving \[[20](#bib.bib35), [21](#bib.bib36), [22](#bib.bib37)\].
As an additional industry example, NVIDIA’s Vera Rubin platform pairs
GPUs with specialized low-latency processors (Groq 3 LPX) to serve
different phases of inference on purpose-built hardware
\[[23](#bib.bib4)\]. TIDE embraces this direction by enabling inference
and draft model training to be assigned to different device types
according to their computational characteristics. In our evaluation, we
assign inference to NVIDIA H100 GPUs, which are optimized for
low-latency serving, and dedicate AMD Instinct MI250 GPUs to draft model
adaptation. Compared to a baseline that allocates both GPU types to
inference with a static draft model, TIDE achieves higher overall system
throughput through continuous draft–target alignment improvement. As
heterogeneous clusters become the industry standard rather than the
exception, frameworks like TIDE that effectively leverage diverse
hardware for both inference serving and draft model training are
essential for maximizing system-wide efficiency.

In summary, our main contributions are:

- •
  We propose TIDE, a serving-engine-native framework for adaptive
  speculative decoding that incrementally maintains draft–target
  alignment under inference workloads that are globally non-stationary
  yet exhibit strong short-term temporal locality.
- •
  We generate training signals by reusing intermediate hidden states
  computed during inference and asynchronously delivering them to the
  draft-model training engine through shared storage, overlapping the
  required data movement with subsequent target-model computation to
  keep it off the serving critical path.
- •
  We introduce adaptive runtime control mechanisms that determine when
  to speculate and when to train, enabling speculative decoding only
  when it yields throughput gains and triggering draft model training
  only when alignment degradation is detected.
- •
  We demonstrate effective heterogeneous GPU utilization by decoupling
  inference and training, assigning each workload to the hardware best
  suited for it: latency-sensitive inference on NVIDIA H100 GPUs and
  throughput-oriented training on AMD MI250 GPUs.
- •
  We implement a complete TIDE prototype and show consistent
  system-level throughput improvements across diverse real-world
  workload patterns.

## II Background

Fig. 2: Overview of TIDE architecture and workflow.

### II-A Speculative Decoding in Dynamic Workloads

Large language models (LLMs) generate text autoregressively: each token
is produced sequentially, with every generation step depending on all
previously generated tokens. Because a single decoding step must read
the entire model’s parameters from GPU memory yet performs relatively
little arithmetic, LLM decoding is typically memory-bandwidth bound
\[[24](#bib.bib26), [4](#bib.bib8)\].

Speculative decoding exploits this underutilization. As illustrated in
[Fig. 3](#S3.F3 "In III-A System Overview ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
a smaller draft model $`q`$ autoregressively generates $`\gamma`$
candidate tokens, which the target model $`p`$ then verifies in parallel
in a single forward pass. Because the target model’s decoding step is
memory-bound, verifying a small batch of candidates costs roughly the
same wall-clock time as generating a single token, yielding a speedup
whenever the draft model’s predictions are sufficiently accurate.
Leviathan et al. \[[4](#bib.bib8)\] derived the theoretical speedup of
speculative decoding relative to vanilla autoregressive decoding:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
Theoretical\ Speedup=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(c\gamma+1)}
``` |  | (1) |

where $`\alpha`$ is the token acceptance rate—the probability that a
candidate token drawn from the draft model is also consistent with the
target model’s output distribution, $`\gamma`$ is the number of
candidate tokens, and $`c`$ is the ratio between draft and target model
latency.

This formula reveals three critical conditions for performance
improvements: (1) high acceptance rate requiring strong draft–target
alignment, (2) memory-bound regime where verification latency equals
single-token decoding latency, and (3) minimal draft overhead achievable
through lightweight draft model architectures. Note that condition (2)
weakens as the serving batch size grows: with more concurrent requests,
verification becomes compute-bound and processing extra candidate tokens
adds measurable latency, reducing the benefit of speculative decoding.

Prior work addresses these conditions individually. Online training
approaches \[[15](#bib.bib10), [25](#bib.bib11), [26](#bib.bib12),
[27](#bib.bib13)\] improve draft–target alignment through continuous
fine-tuning using inference-time signals. This is motivated by the
observation that the acceptance rate is inherently workload-dependent:
as the domain or pattern of user requests shifts over time, a previously
well-aligned draft model may become inaccurate, degrading speculative
decoding performance. Adaptive systems \[[17](#bib.bib22),
[18](#bib.bib17), [19](#bib.bib18), [28](#bib.bib19), [29](#bib.bib20),
[30](#bib.bib21)\] focus on runtime control by dynamically adjusting the
number of speculative tokens or disabling speculative decoding based on
system load and acceptance rates. Lightweight draft
architectures \[[31](#bib.bib23), [32](#bib.bib14), [33](#bib.bib24),
[34](#bib.bib25)\] reduce draft model overhead by simplifying model
structure or reusing intermediate representations.

However, these techniques have only been evaluated individually under
controlled experimental settings, without demonstrating their benefits
when integrated into a complete serving system. In particular, there is
limited investigation into how these techniques should be applied in
practice to improve end-to-end throughput in real LLM serving
environments.

### II-B EAGLE: Hidden-State-Based Draft Models

Recent draft model architectures such as EAGLE \[[32](#bib.bib14)\]
achieve minimal overhead by reusing the target model’s own internal
representations. In a Transformer-based LLM, each input token is
transformed through a stack of decoder layers, producing hidden
states—high-dimensional vectors that encode increasingly abstract
semantic information at each layer. The EAGLE-family draft model
consists of only a single decoder layer and a lightweight output
projection (LM head), and takes concatenated hidden states from multiple
decoder layers (e.g., early, middle, and late layers) of the target
model as input. Because these hidden states are already computed during
normal inference, they can be directly reused as both the draft model’s
input and its training signal, requiring no additional target model
computation.

### II-C Heterogeneous Systems for LLM Inference

Modern datacenters inevitably consist of heterogeneous GPU clusters, as
new accelerator generations are adopted incrementally over time. Prior
work has primarily exploited such heterogeneity for inference-only
execution, either by assigning different models to different GPUs based
on coarse-grained criteria (e.g., model size or latency sensitivity)
\[[35](#bib.bib30), [36](#bib.bib31)\], or by distributing different
inference phases of a single model across heterogeneous devices, such as
disaggregated prefill and decoding \[[37](#bib.bib32), [38](#bib.bib33),
[39](#bib.bib34)\]. In these systems, heterogeneity is handled via
static or periodically updated placement policies derived from offline
profiling.

In contrast, TIDE considers heterogeneity in a setting where inference
is interleaved with online draft model adaptation. Rather than
optimizing inference alone, TIDE decouples inference serving and draft
training and assigns them to different GPU classes based on their
respective performance and cost characteristics. This enables
heterogeneous clusters to efficiently support adaptive speculative
decoding, a dimension not addressed by prior heterogeneous LLM serving
systems.

## III TIDE Architecture

### III-A System Overview

Fig. 3: Overview of speculative decoding with $`\gamma=4`$. The draft
model autoregressively generates four candidate tokens
$`\tilde{t}_{1},\ldots,\tilde{t}_{4}`$ from a prefix token $`t_{0}`$.
The target model verifies all candidates in a single parallel forward
pass, producing $`t_{1},\ldots,t_{5}`$. In this example, the first three
draft tokens match the target model’s outputs ($`\tilde{t}_{i}=t_{i}`$
for $`i=1,2,3`$; green), while $`\tilde{t}_{4}\neq t_{4}`$ causes
rejection from the fourth token onward (red). Had $`\tilde{t}_{4}`$ also
been accepted, $`t_{5}`$ would be obtained as a bonus token at no
additional cost.

Fig. 4: Detailed TIDE architecture and workflow. The system monitors
acceptance length to adaptively enable/disable speculative decoding and
selectively trigger training signal collection based on workload
changes.

As illustrated in
[Fig. 2](#S2.F2 "In II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
TIDE consists of the Inference Serving Engine with its Adaptive Drafter,
Acceptance Length Monitor, and Training Signal Extractor, and the Draft
Model Training Engine, working collaboratively to maintain high
inference throughput while continuously adapting the draft model to
evolving workloads.

Inference Serving Engine. The Inference Serving Engine handles incoming
requests using speculative decoding with adaptive control. During each
inference iteration, the draft model generates candidate tokens, which
are then verified by the target model in parallel. This verification
process produces two critical outputs: per-request acceptance length,
the number of draft tokens accepted by the target model for each
request, and draft model training signals, consisting of target model’s
intermediate hidden states generated during prefill and verification
phases for input and output tokens.

Based on these outputs, two decision components operate as illustrated
in
[Fig. 4](#S3.F4 "In III-A System Overview ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"):
The Adaptive Drafter monitors batch size and acceptance length to
determine whether to enable or disable speculative decoding, ensuring it
is only applied when beneficial for performance
(Section [IV-A](#S4.SS1 "IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")).
The Training Signal Extractor decides whether to store training signals
in shared storage, minimizing storage and training overhead. It halts
extraction when the acceptance length improvement plateaus, indicating
the draft model has sufficiently adapted to the current workload
distribution, and resumes extraction upon detecting distribution shift
(Section [IV-B](#S4.SS2 "IV-B Selective Draft Model Training ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")).

Draft Model Training Engine. The Draft Model Training Engine operates
asynchronously on separate GPU resources independent from the Inference
Serving Engine. It continuously monitors shared storage for accumulated
training signals. When the signals exceed a predefined threshold, it
triggers a training cycle, loading the current draft model checkpoint
and fine-tuning it using the accumulated training signals. After
training converges, the updated draft model is evaluated and deployed to
the Inference Serving Engine only if it demonstrates improved acceptance
length.

### III-B Serving-Time Training Signal Extraction

Fig. 5: TIDE’s asynchronous adaptation pipeline. Device-to-host transfer
and buffered storage writes overlap with subsequent GPU verification
(top), keeping training-signal extraction off the serving critical path,
while draft model training proceeds concurrently on separate resources
(bottom).

TIDE employs EAGLE-3 \[[32](#bib.bib14)\], a state-of-the-art
speculative decoding technique, to build a draft model consisting of a
single decoder layer and an LM head. The EAGLE-3 draft model predicts
the next token based on the target model’s intermediate hidden states,
rather than directly from the input prompt as in conventional LLMs.
Specifically, EAGLE-3 uses concatenated hidden states from decoder
layers at low, middle, and high positions of the target model, leading
to richer semantic information for next token predictions. These hidden
states are computed during the target model’s prefill for input prompts
and decoding (or verification if speculative decoding is enabled)
phases, making them natural byproducts of normal inference operations
that can be directly used as training signals for the EAGLE-3 draft
model.

To leverage this opportunity for efficient training signal collection,
we implement intermediate hidden state extraction on top of
SGLang \[[40](#bib.bib27)\] and vLLM \[[24](#bib.bib26)\], making our
approach applicable to any transformer-based architecture. Extraction
adds two serving-side operations: (1) the hidden states for accepted
tokens are concatenated and copied from device memory to a pre-allocated
host buffer, and (2) the buffer is flushed to shared storage when full.
As illustrated in
[Fig. 5](#S3.F5 "In III-B Serving-Time Training Signal Extraction ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
both operations execute asynchronously and overlap with the next
verification step’s compute kernels, keeping training-signal extraction
off the serving critical path despite consuming memory, interconnect,
and storage bandwidth. Draft model training likewise consumes separate
compute resources but proceeds asynchronously from inference serving.

### III-C Online Draft Model Adaptation

Traditional draft model training approaches \[[41](#bib.bib28),
[7](#bib.bib45)\] require running target model inference again during
the training process to generate training data. This leads to two
significant limitations: (1) substantially increased training time as
analyzed in
[Section V-D](#S5.SS4 "V-D Training Efficiency Comparison ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
and (2) excessive GPU memory consumption from loading both the target
and draft models simultaneously, constraining the resources available
for training.

TIDE addresses these limitations by directly utilizing training signals
extracted from the inference serving engine, eliminating redundant
computation. The target model’s intermediate hidden states—used as
training data for the draft model—are already generated during inference
serving, meaning only the compact draft model (a single decoder layer
with an LM head) needs to be loaded onto the device for training,
regardless of the target model’s size. This enables TIDE to scale to
arbitrarily large target models without increasing training resource
requirements.

As illustrated in
[Fig. 5](#S3.F5 "In III-B Serving-Time Training Signal Extraction ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
draft model training proceeds asynchronously from inference serving. The
draft model’s compact architecture, combined with multi-device
parallelization using PyTorch FSDP \[[42](#bib.bib29)\], enables
real-time adaptation even with relatively modest resource allocation
dedicated to training.

## IV Adaptive Control and Runtime Optimization

### IV-A Adaptive Speculative Decoding Control

Fig. 6: Ratio of verification latency $`T(b(\gamma+1))`$ for
$`\gamma=3`$ candidate tokens to single-token decoding latency $`T(b)`$
across different batch sizes. If decoding is completely memory-bound,
this ratio would be 1.0 (ideal case shown by dashed line). The ratio
exhibits substantial variance depending on both batch size and model
architecture. Since the ratio exceeds 1.0 in most configurations, the
effective speedup from speculative decoding is reduced accordingly,
underscoring the need for adaptive control that selectively enables
speculation only when conditions are favorable.

Speculative decoding fundamentally relies on the memory-bound nature of
standard autoregressive decoding to achieve speedup. Even when
draft–target alignment is strong and draft model overhead is minimal,
performance gains can be limited by batch size. As concurrent requests
increase, target model decoding shifts from memory-bound to
compute-bound, increasing verification overhead and diminishing the
benefits of speculative decoding.

This section presents a performance model that estimates the speedup of
speculative decoding as a function of batch size, enabling TIDE to
adaptively apply speculative decoding only when it provides measurable
benefits. We derive a practical speedup formula that accounts for how
batch size affects speculative decoding performance.

In each speculation step, the draft model generates $`\gamma`$ candidate
tokens, of which an average fraction $`\alpha`$ is accepted during
target model verification. The expected acceptance length
is \[[4](#bib.bib8), [5](#bib.bib9)\]:

|     |                                              |     |     |
|-----|----------------------------------------------|-----|-----|
|     |
       ``` math
       E[\ell]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}
       ```                                           |     | (2) |

Let $`D(n)`$ and $`T(n)`$ denote the latency for the draft model and the
target model to decode $`n`$ tokens in parallel, respectively. When
processing $`b`$ concurrent requests in a batch, the per-token latency
with speculative decoding becomes:

|     |                                                  |     |     |
|-----|--------------------------------------------------|-----|-----|
|     |
       ``` math
       SD(b)=\frac{\gamma D(b)+T(b(\gamma+1))}{E[\ell]}
       ```                                               |     | (3) |

where $`\gamma D(b)`$ represents the latency of generating $`\gamma`$
candidate tokens by the draft model, and $`T(b(\gamma+1))`$ represents
the latency of verifying these $`\gamma`$ tokens plus one bonus token
from the previous step by the target model.

The speedup relative to standard autoregressive decoding is:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle Speedup`$ | $`\displaystyle=\frac{T(b)}{SD(b)}`$ |  |  |
|  |  | $`\displaystyle=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)\left(\frac{D(b)}{T(b)}\gamma+\frac{T(b(\gamma+1))}{T(b)}\right)}`$ |  | (4) |

The theoretical speedup formula in
Equation [1](#S2.E1 "Equation 1 ‣ II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
is derived by substituting $`c=D(b)/T(b)`$ and approximating
$`T(b(\gamma+1))/T(b)\approx 1`$, which holds only in the memory-bound
regime. However, this assumption often fails to predict actual
performance. As batch size increases and the standard decoding becomes
compute-bound, the ratio $`\beta(b)=T(b(\gamma+1))/T(b)`$ grows
significantly (see
[Fig. 6](#S4.F6 "In IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")).

Moreover, the coefficient $`c=D(b)/T(b)`$ is not static in practice.
Modern speculative decoding architectures such as EAGLE-3
\[[32](#bib.bib14)\] consist of a single decoder layer and an LM head,
making the draft model’s computational cost negligible compared to the
target model. However, in implementations such as vLLM
\[[24](#bib.bib26)\] and SGLang \[[40](#bib.bib27)\], the draft model
latency is dominated by CPU overhead including kernel launch overhead,
which is independent of batch size. Therefore, we can approximate
$`D(b)\approx D_{0}`$ as a static value, while $`T(b)`$ varies with
batch size, causing $`c(b)=D_{0}/T(b)`$ to decrease as batch size
increases.

Incorporating these observations, we obtain a simplified practical
speedup formula:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
Practical~Speedup=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)\left(c(b)\gamma+\beta(b)\right)}
``` |  | (5) |

Given $`T(n)`$ and $`D_{0}`$, we can determine the minimum acceptance
rate required for speculative decoding to provide performance gains at a
given batch size. Since $`T(n)`$ and $`D_{0}`$ vary depending on the
model architecture, parallelization strategy, and system environment,
they must be empirically profiled. TIDE’s Adaptive Drafter profiles
$`T(n)`$ and $`D_{0}`$ during system initialization by measuring
latencies across different batch sizes. These profiled values are used
to estimate speedup in real-time according to
Equation [5](#S4.E5 "Equation 5 ‣ IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
enabling TIDE to adaptively enable or disable speculative decoding based
on current batch size and acceptance rates, thereby maintaining peak
inference throughput.

 Input:
$`\lambda_{\text{short}},\lambda_{\text{long}},\epsilon,N_{\text{init}},N_{\text{threshold}}`$

 $`\text{collection\_enabled}\leftarrow\text{True}`$,
$`M_{\text{draft}}\leftarrow`$ initial draft model

 

 Measure acceptance rates
$`\{\alpha_{1},\ldots,\alpha_{N_{\text{init}}}\}`$ from first
$`N_{\text{init}}`$ requests

 $`\bar{\alpha}_{\text{short}},\bar{\alpha}_{\text{long}}\leftarrow\frac{1}{N_{\text{init}}}\sum_{i=1}^{N_{\text{init}}}\alpha_{i}`$

 

 while serving requests do

   Extract target-model intermediate hidden states $`h`$ and measure
acceptance rate $`\alpha`$

   $`\bar{\alpha}_{\text{short}}\leftarrow\lambda_{\text{short}}\bar{\alpha}_{\text{short}}+(1-\lambda_{\text{short}})\alpha`$

   $`\bar{\alpha}_{\text{long}}\leftarrow\lambda_{\text{long}}\bar{\alpha}_{\text{long}}+(1-\lambda_{\text{long}})\alpha`$

   

   if
$`\bar{\alpha}_{\text{short}}<\bar{\alpha}_{\text{long}}-\epsilon`$ then

    $`\text{collection\_enabled}\leftarrow\text{True}`$

   end if

   

   if collection_enabled then

    Store $`(h,\alpha)`$

   end if

   

   if $`|\text{stored samples}|\geq N_{\text{threshold}}`$ then

    Split into $`D_{\text{train}},D_{\text{eval}}`$

    $`M_{\text{new}}\leftarrow\text{train}(M_{\text{draft}},D_{\text{train}})`$

    $`\bar{\alpha}_{\text{train}}\leftarrow`$ average $`\alpha`$ in
$`D_{\text{train}}`$

    $`\bar{\alpha}_{\text{eval}}\leftarrow\text{eval}(M_{\text{new}},D_{\text{eval}})`$

    if $`\bar{\alpha}_{\text{eval}}>\bar{\alpha}_{\text{train}}`$ then

     $`M_{\text{draft}}\leftarrow M_{\text{new}}`$

    else if $`\bar{\alpha}_{\text{eval}}<\bar{\alpha}_{\text{train}}`$
then

     $`\text{collection\_enabled}\leftarrow\text{False}`$

    end if

   end if

 end while

Algorithm 1 Selective Training Control

### IV-B Selective Draft Model Training

Fig. 7: Accept length evolution during draft model training across four
datasets using gpt-oss-120b as the target model. Accept length measures
the average number of tokens accepted per speculative decoding step.
Each time step corresponds to 30 seconds of training time.

Training on every incoming request would waste computational resources
and could lead to overfitting on the observed distribution. TIDE employs
a selective training strategy that dynamically enables or disables
training signal collection based on whether the draft model training is
effective, ensuring efficient adaptation while avoiding unnecessary
computation once training yields diminishing returns.

As shown in
[Fig. 7](#S4.F7 "In IV-B Selective Draft Model Training ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
accept length exhibits a characteristic saturation pattern during draft
model training. This saturation behavior motivates TIDE’s selective
training strategy, which adaptively pauses training when further updates
yield minimal acceptance improvements, reducing computational overhead
while maintaining draft model quality.

The key challenge is detecting when draft model training is no longer
effective—when training on the collected data does not improve
acceptance rates. TIDE addresses this by monitoring distribution shifts
through acceptance rate patterns and evaluating training effectiveness
through train-eval performance comparison.

Algorithm [1](#alg1 "Algorithm 1 ‣ IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
presents the selective training control mechanism. The system maintains
two moving averages of acceptance rates at different timescales: a
short-term average ($`\bar{\alpha}_{\text{short}}`$) that responds
quickly to recent changes, and a long-term average
($`\bar{\alpha}_{\text{long}}`$) that captures overall trends. Both
averages are computed using exponential moving average (EMA):

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\bar{\alpha}_{t}=\lambda\cdot\bar{\alpha}_{t-1}+(1-\lambda)\cdot\alpha_{t}
``` |  | (6) |

where $`\lambda`$ is the decay rate controlling the average’s
responsiveness.

This mechanism exhibits the desired self-regulating behavior. Initially,
hidden state collection is enabled so that the draft model can adapt to
the initial workload. Once sufficient data is collected, the system
fine-tunes the draft model. If the fine-tuned model achieves higher
acceptance rates than the previous model, the updated model is deployed.
Otherwise, hidden state collection is disabled until a distribution
shift is detected, at which point collection is automatically
re-enabled.

## V Evaluation

### V-A Experimental Setup

Implementation. TIDE’s inference serving engine is built on
SGLang \[[40](#bib.bib27)\], while the draft model training engine is
based on SpecForge \[[41](#bib.bib28)\].

Hardware. We conduct inference experiments on a single node with up to
eight NVIDIA H100 GPUs and draft model training on a separate node with
four AMD Instinct MI250 GPUs. The nodes use a VAST shared-storage
cluster to exchange hidden states and updated draft model checkpoints.
The storage is mounted over NFSv3-RDMA through a 200 Gb/s InfiniBand
link.

Target Models. We evaluate TIDE on four target models: gpt-oss-120b
\[[43](#bib.bib39)\], Qwen3-235B-A22B \[[44](#bib.bib40)\],
Llama-4-Scout-17B-16E \[[45](#bib.bib46)\], and Llama-3.3-70B-Instruct
\[[46](#bib.bib41)\]. These models span different architectures and
parameter scales, providing a comprehensive evaluation across diverse
model characteristics.

Draft Model Configuration. We fix the number of candidate tokens to 3
across all experiments, as this empirically provides the best
performance. The draft models use the same vocabulary as their
corresponding target models. Since the target models are large enough to
require tensor parallelism for efficient serving, the vocabulary size
overhead in the draft models is negligible.

Datasets. We evaluate TIDE on diverse datasets spanning multiple
domains: ShareGPT (conversational) \[[47](#bib.bib56)\], Science
(scientific text) \[[48](#bib.bib48), [49](#bib.bib49),
[50](#bib.bib47)\], NuminaMath (mathematical reasoning)
\[[51](#bib.bib50)\], and EvolCodeAlpaca (code generation)
\[[52](#bib.bib51)\]. Additionally, we evaluate on multilingual Alpaca
datasets (Korean, Arabic, Chinese, French) \[[53](#bib.bib52),
[54](#bib.bib53), [55](#bib.bib54), [56](#bib.bib55)\] to assess TIDE’s
robustness to distribution shift, as vocabulary differences between
languages represent the most significant source of distribution shift in
our experiments.

Fig. 8: Throughput evolution over time across four datasets during
inference serving using gpt-oss-120b as the target model. Each time step
corresponds to 30 seconds of training time.

Fig. 9: Normalized throughput comparison across four datasets at batch
sizes 1 and 64. All values are normalized to the no-speculation
baseline. At batch size 1, where speculative decoding is most effective,
TIDE achieves up to 1.66$`\times`$ throughput over no speculation. At
batch size 64, TIDE still achieves up to 1.31$`\times`$ improvement.

### V-B Throughput Improvement over Time

[Fig. 8](#S5.F8 "In V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
shows the throughput evolution of TIDE during inference serving across
four datasets. For Science, NuminaMath, and EvolCodeAlpaca, throughput
consistently improves over time as the draft model is incrementally
adapted using inference-time training signals. As shown in
[Fig. 9](#S5.F9 "In V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
the improvement is most pronounced in latency-sensitive scenarios (batch
size 1): TIDE achieves 1.46$`\times`$, 1.66$`\times`$, and
1.52$`\times`$ throughput over the no-speculation baseline,
respectively, compared to 1.29$`\times`$, 1.59$`\times`$, and
1.36$`\times`$ for the static draft model. This is because at low batch
sizes, GPU compute is underutilized during standard autoregressive
decoding, allowing draft token verification to proceed with minimal
overhead—and each accepted draft token directly reduces end-to-end
latency. At batch size 64, where the GPU is more saturated, TIDE still
achieves 1.31$`\times`$, 1.30$`\times`$, and 1.26$`\times`$ throughput
over the baseline, compared to 1.14$`\times`$, 1.18$`\times`$, and
1.17$`\times`$ for the static draft model.

For ShareGPT, the static draft model already achieves 1.30$`\times`$ and
1.14$`\times`$ throughput over no speculation at batch sizes 1 and 64,
respectively, and TIDE’s online adaptation provides further gains to
1.42$`\times`$ and 1.16$`\times`$. This reflects a fundamental ceiling
of speculative decoding on high-entropy conversational workloads
\[[14](#bib.bib16), [11](#bib.bib43)\], not a limitation of TIDE: any
draft model—static or adaptive—faces the same constraint. Notably,
TIDE’s adaptive control remains valuable in such cases by automatically
reducing unnecessary training when further adaptation yields diminishing
returns.

In contrast, TIDE’s impact is most pronounced when draft–target
alignment is poor. As shown in
[Fig. 1](#S1.F1 "In I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
when a static draft model trained on English data is applied to
non-English workloads, speculative decoding degrades throughput to
0.70–0.86$`\times`$ of the no-speculation baseline. TIDE recovers
performance to 1.08–1.11$`\times`$, demonstrating its ability to adapt
to workloads that static approaches cannot serve effectively. Taken
together, these results illustrate that speculative decoding does not
uniformly accelerate all workloads—its effectiveness varies
significantly across domains, ranging from modest gains on high-entropy
conversational data to substantial speedups on structured domains such
as mathematics and code. Pre-training a draft model to cover all
possible workloads is impractical: production workloads span an
open-ended set of domains, and even if broad coverage were achievable at
training time, the distribution of incoming requests shifts over time as
user behavior and application demands evolve. This workload-dependent
and temporally non-stationary nature of speculative decoding
effectiveness highlights the need for runtime adaptation logic that
determines when to enable speculative decoding and when to trigger draft
model training.

### V-C Adaptive Speculative Decoding Control

Fig. 10: Comparison of practical and actual speedup across batch sizes.
Actual speedup is measured using four H100 GPUs.

Fig. 11: Throughput over time for TIDE-default and TIDE-adaptive under
distribution shifts. Sequential language transitions (Korean → Arabic →
Chinese → French) using Alpaca datasets demonstrate adaptive
performance. Both experiments process an identical total workload, while
TIDE-adaptive finishes at an earlier timestep.

Fig. 12: End-to-end throughput of two allocations using the same
hardware budget (8$`\times`$ H100 + 4$`\times`$ MI250), normalized to
All-Inference. All-Inference assigns both GPU nodes to standard
inference with speculative decoding disabled; TIDE uses the H100 node
for speculative decoding with the target and draft models, while the
MI250 node concurrently trains the draft model on the same live
workload. Parentheses indicate the resulting speculative-decoding
speedup ($`s`$) on the H100 serving node.

We validate our analytical model for predicting speculative decoding
speedup and use it to implement adaptive control mechanisms.
[Fig. 10](#S5.F10 "In V-C Adaptive Speculative Decoding Control ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
compares practical speedup predicted by our analytical model
(Equation [5](#S4.E5 "Equation 5 ‣ IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"))
against actual measured speedup across different batch sizes. Across all
four target models, the practical speedup predicted by our analytical
model closely tracks the actual measured speedup within 9% error,
validating the accuracy of our model for adaptive control decisions.

Using this model, we calculate the minimum acceptance length required
for performance improvement at each batch size. We evaluate two
configurations: (1) TIDE-default, which always enables speculative
decoding regardless of batch size and acceptance length, and (2)
TIDE-adaptive, which dynamically enables or disables speculative
decoding based on whether acceptance length exceeds the calculated
threshold.
[Fig. 11](#S5.F11 "In V-C Adaptive Speculative Decoding Control ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
compares throughput over time for both configurations. TIDE-adaptive
mitigates throughput degradation during distribution shifts by
adaptively disabling speculative decoding when acceptance length is
insufficient, while TIDE-default experiences more severe performance
drops.

Fig. 13: Draft model training accuracy comparison between SpecForge
offline and TIDE across four datasets using a global batch size of 16 on
four MI250 GPUs. Accuracy measures the top-1 token prediction match rate
between the draft model and the target model.

### V-D Training Efficiency Comparison

We compare TIDE against two state-of-the-art training approaches for
EAGLE-3 draft models in SpecForge: offline training and online training.

SpecForge Offline Training. This approach assumes a pre-collected
dataset of input prompts paired with their corresponding target
model-generated outputs. It first concatenates each input–output pair
into a single sequence and runs prefill through the target model to
extract intermediate hidden states, which are then stored to disk. Since
draft model training typically requires multiple passes over the
dataset, storing hidden states avoids reloading the target model and
recomputing prefill at every epoch. However, this comes at the cost of
substantial storage as shown in
Table [I](#S5.T1 "Table I ‣ V-D Training Efficiency Comparison ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").

SpecForge Online Training. To reduce disk usage, this approach
regenerates hidden states on-demand at every epoch instead of storing
them. However, this requires reloading the target model and recomputing
prefill for every pass over the dataset, resulting in significantly
slower training time as shown in
Table [II](#S5.T2 "Table II ‣ V-D Training Efficiency Comparison ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").

In contrast to both approaches, TIDE extracts hidden states directly
from the target model’s inference step, eliminating the dedicated
prefill phase that accounts for 40–67% of total training time in
SpecForge methods
(Table [II](#S5.T2 "Table II ‣ V-D Training Efficiency Comparison ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")).
Like SpecForge offline, TIDE stores hidden states to disk, but only
maintains a small buffer that is consumed and discarded after each
training cycle rather than retaining the entire dataset. This
drastically reduces storage requirements—by up to 24$`\times`$ compared
to offline training (e.g., 4.66 TB to 0.19 TB for gpt-oss-120b) as shown
in
Table [I](#S5.T1 "Table I ‣ V-D Training Efficiency Comparison ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")—while
also enabling faster adaptation: as shown in
Table [II](#S5.T2 "Table II ‣ V-D Training Efficiency Comparison ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
TIDE is 1.67$`\times`$ faster than SpecForge offline and 3.02$`\times`$
faster than SpecForge online on the ShareGPT dataset with gpt-oss-120b.
Crucially, these efficiency gains do not sacrifice training quality:
[Fig. 13](#S5.F13 "In V-C Adaptive Speculative Decoding Control ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
confirms that TIDE achieves comparable final accuracy to both SpecForge
methods across four datasets, while enabling continuous adaptation
during live serving that existing approaches cannot provide.

|                        |                   |         |
|------------------------|-------------------|---------|
| Target Model           | SpecForge offline | TIDE    |
| gpt-oss-120b           | 4.66 TB           | 0.19 TB |
| Qwen3-235B-A22B        | 6.63 TB           | 0.27 TB |
| Llama-4-Scout-17B-16E  | 8.29 TB           | 0.34 TB |
| Llama-3.3-70B-Instruct | 13.26 TB          | 0.55 TB |

TABLE I: Storage requirements for hidden states across different target
models. SpecForge offline stores all hidden states for the entire
dataset, while TIDE only maintains a small buffer for active training
batches.

|                   |         |        |         |                |
|-------------------|---------|--------|---------|----------------|
| Method            | Prefill | Train  | Total   | Speedup        |
| SpecForge offline | 6.16 h  | 9.16 h | 15.32 h | 1.00$`\times`$ |
| SpecForge online  | 18.48 h | 9.16 h | 27.64 h | 0.55$`\times`$ |
| TIDE              | -       | 9.16 h | 9.16 h  | 1.67$`\times`$ |

TABLE II: Training time comparison for gpt-oss-120b on ShareGPT (100k)
dataset. TIDE eliminates prefill overhead by reusing hidden states from
inference serving.

### V-E Shared-Storage I/O Scalability

We further quantify whether storage writes can keep pace under
multi-node contention. For gpt-oss-120b, EAGLE-3 stores BF16 hidden
states from three target-model layers per token. Given a hidden size of
2,880 and 2 bytes per BF16 element, each token produces
$`2{,}880\times 3\times 2=17{,}280`$ bytes, or approximately 16.9 KiB,
of training signals. Each inference serving instance uses tensor
parallelism across four GPUs (TP$`=4`$) and generates approximately
6,000–7,000 tokens/s, resulting in 99–116 MiB/s of write traffic.

Our VAST storage sustains 564 MB/s with one writer and 1.47 GB/s with
eight concurrent serving instances across four nodes. The latter
corresponds to approximately 184 MB/s per instance, providing
1.6–1.9$`\times`$ headroom over the observed demand. The aggregate
traffic also uses only a small fraction of the 200 Gb/s InfiniBand
link’s approximately 25 GB/s capacity. Although these measurements
characterize storage capacity rather than per-operation latency, they
demonstrate that shared-storage I/O is not a bottleneck at the evaluated
multi-node scale.

### V-F Heterogeneous GPU Allocation

Fig. 14: Per-GPU throughput comparison for inference and draft model
training, normalized to MI250 baseline. Inference throughput measured on
gpt-oss-120b using SGLang with tensor parallelism. Training throughput
was measured using PyTorch with FSDP on single nodes equipped with 4
MI250, 8 MI300X, and 8 H100 GPUs, respectively.

As modern datacenters incrementally adopt new accelerator generations,
heterogeneous GPU clusters are inevitable. We evaluate how TIDE
leverages this heterogeneity by profiling inference and training
throughput across GPU types.
[Fig. 14](#S5.F14 "In V-F Heterogeneous GPU Allocation ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
shows that the performance gap across GPU generations is significantly
larger for inference than for training: compared to MI250, H100 achieves
6.76$`\times`$ higher inference throughput but only 2.44$`\times`$
higher training throughput, with MI300X showing a similar pattern
(4.42$`\times`$ vs. 1.77$`\times`$). This asymmetry means that
dedicating high-end GPUs to inference—where they provide the greatest
advantage—and assigning older GPUs to draft model training—where the
relative performance penalty is small—maximizes overall system
efficiency. Importantly, since TIDE fully decouples inference and
training onto separate nodes that communicate only through shared
storage for hidden states and model checkpoints, this heterogeneous
deployment requires no cross-vendor driver compatibility, no GPU-to-GPU
interconnect, and no modifications to existing datacenter
infrastructure.

To quantify the benefits of this approach, we compare two allocation
strategies under the same hardware budget of a single H100 node with 8
GPUs and a single MI250 node with 4 GPUs. The All-Inference baseline
assigns both the H100 and MI250 nodes to standard target-model inference
with speculative decoding disabled. TIDE instead uses the H100 node for
speculative decoding with the target and draft models, while assigning
the MI250 node to asynchronous draft model training.

[Fig. 12](#S5.F12 "In V-C Adaptive Speculative Decoding Control ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
shows that, once draft model training has converged, TIDE achieves
1.08–1.22$`\times`$ throughput improvement over the all-inference
baseline. Although TIDE uses the MI250 node for training instead of
inference, it still achieves higher total throughput. This shows that
the throughput gained from speculative decoding on the H100 node exceeds
the throughput that the MI250 node would provide through direct
inference. The improvement correlates with the speculative decoding
speedup achieved through draft model training, ranging from $`s=1.15`$
(ShareGPT, 1.08$`\times`$ throughput) to $`s=1.30`$ (Science,
1.22$`\times`$ throughput). These variations reflect differences in
output distribution characteristics and draft model learning difficulty
across datasets. For instance, Science dataset’s more structured output
enables better draft model learning, resulting in higher acceptance
rates and greater speedup. This result demonstrates that TIDE’s benefits
vary with dataset characteristics and highlights the importance of
considering workload properties when deploying heterogeneous training
strategies.

|                |                              |                               |
|----------------|------------------------------|-------------------------------|
| Dataset        |  Total Input Length (M)      |  Total Output Length (M)      |
| ShareGPT       | 106.18                       | 512.65                        |
| Science        | 7.98                         | 286.15                        |
| EvolCodeAlpaca | 50.77                        | 491.84                        |
| NuminaMath     | 21.07                        | 234.18                        |

TABLE III: Dataset sizes and aggregate input and output sequence
lengths. Lengths are reported in millions of tokens.

Prefill/decode disaggregation provides another possible allocation, with
the MI250 node serving as a prefill worker and the H100 node performing
decoding. However, the workloads in
[Fig. 12](#S5.F12 "In V-C Adaptive Speculative Decoding Control ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference")
are decode-heavy. As shown in
[Table III](#S5.T3 "In V-F Heterogeneous GPU Allocation ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
output sequences are longer than input sequences for all four datasets;
the geometric mean of the per-dataset output-to-input length ratios is
11.7$`\times`$. Direct profiling on H100 also shows that prefill
accounts for at most 1% of end-to-end inference time. Assigning the
MI250 node exclusively to prefill would therefore leave it substantially
underutilized for the decode-heavy workloads evaluated here.

## VI Discussion

While TIDE demonstrates consistent throughput improvements across
diverse workloads, several limitations merit discussion.

### VI-A Draft Architecture Dependence

TIDE’s training signal extraction is tightly coupled to the EAGLE-3
draft architecture, which predicts next tokens from the target model’s
intermediate hidden states at selected decoder layers. This design
choice avoids additional target-model computation for training-signal
generation but also limits generality: draft architectures that do not
consume target model hidden states as input—such as independent draft
models trained via standard distillation \[[15](#bib.bib10)\]—cannot
directly benefit from TIDE’s training signal extraction pipeline.
Extending TIDE to support alternative draft architectures would require
rethinking how training signals are collected and what intermediate
representations are reused, which we leave as future work.

### VI-B Adaptation Latency Under Rapid Distribution Shifts

TIDE detects distribution shifts by comparing short-term and long-term
exponential moving averages of acceptance rates. While this mechanism
effectively identifies gradual workload transitions, it introduces
inherent latency in responding to abrupt distribution changes. Upon
detecting a shift, TIDE must still collect sufficient training signals,
transfer them to the training engine, fine-tune the draft model, and
deploy the updated checkpoint. If the workload shifts faster than this
adaptation cycle, the temporal locality window may close before the
updated draft model is deployed.

TIDE’s adaptive control mechanism provides a short-term mitigation: upon
detecting a distribution shift, the controller can promptly disable
speculative decoding, ensuring that inference throughput does not
degrade below the baseline of standard autoregressive generation. While
this prevents performance regression, it is a conservative fallback that
forgoes the potential benefits of speculative decoding during the
adaptation period. A more fundamental solution would be to maintain a
pool of domain-specialized draft model checkpoints, allowing the system
to switch to a previously trained checkpoint upon detecting a familiar
distribution rather than retraining from scratch. This approach would
substantially reduce adaptation latency for recurring workloads, which
we discuss further in
[Section VI-C](#S6.SS3 "VI-C Catastrophic Forgetting Under Recurring Workloads ‣ VI Discussion ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").

### VI-C Catastrophic Forgetting Under Recurring Workloads

TIDE continuously fine-tunes the draft model on recent inference data to
exploit short-term temporal locality. However, this incremental
adaptation inherently favors the current workload distribution and may
degrade performance on previously learned patterns. We have not
evaluated how such degradation accumulates over repeated workload
transitions, leaving long-term behavior under recurring workloads as an
open question. In production environments where workloads cycle through
recurring patterns—for example, code generation during business hours
followed by conversational queries in the evening—the draft model may
repeatedly lose and relearn the same distributions.

A natural extension is to maintain a pool of domain-specialized draft
model checkpoints, where each checkpoint is associated with a detected
workload distribution. Upon a distribution shift, the system can select
the closest matching checkpoint from the pool and resume fine-tuning
from it, rather than adapting a misaligned model from scratch.
Importantly, TIDE’s core contribution—runtime training signal extraction
from target model inference—is orthogonal to the draft model management
strategy. The same serving-time training signal extraction pipeline can
serve multiple draft models simultaneously, and the adaptive control
mechanisms can be extended to route requests to the most appropriate
checkpoint. We leave the design of such a multi-draft-model management
strategy as future work.

## VII Conclusion

We present TIDE, a serving-engine-native framework that continuously
adapts draft models at runtime by reusing intermediate hidden states
from target model inference without additional target-model computation
and with no additional serving-time overhead for training-signal
extraction. Combined with adaptive control that dynamically toggles
speculative decoding and training, and a decoupled architecture that
leverages heterogeneous GPU clusters, TIDE consistently improves
throughput across diverse workloads—especially in latency-sensitive
scenarios and on out-of-distribution workloads where static draft models
fail. Future work includes extending TIDE with a pool of
domain-specialized draft model checkpoints to mitigate catastrophic
forgetting under recurring workloads, and generalizing the training
signal extraction pipeline to draft architectures beyond EAGLE-3.

## VIII Acknowledgment

This work was supported by Institute of Information & Communications
Technology Planning & Evaluation (IITP) grant funded by the Korea
government (MSIT) (No. RS-2025-02304554, Efficient and Scalable
Framework for AI Heterogeneous Cluster Systems).

## References

- \[1\] C. Snell, J. Lee, K. Xu, and A. Kumar (2024) Scaling LLM
  test-time compute optimally can be more effective than scaling model
  parameters. Note: arXiv:2408.03314 External Links:
  [Link](https://doi.org/10.48550/arXiv.2408.03314) Cited by:
  [§I](#S1.p1.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[2\] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H.
  Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and T.
  Hashimoto (2025) S1: simple test-time scaling. Note: arXiv:2501.19393
  External Links: [Link](https://doi.org/10.48550/arXiv.2501.19393)
  Cited by:
  [§I](#S1.p1.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[3\] M. Davies, N. Crago, K. Sankaralingam, and C. Kozyrakis (2025)
  LIMINAL: exploring the frontiers of LLM decode performance. Note:
  arXiv:2507.14397 External Links:
  [Link](https://doi.org/10.48550/arXiv.2507.14397) Cited by:
  [§I](#S1.p1.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[4\] Y. Leviathan, M. Kalman, and Y. Matias (2023) Fast inference
  from transformers via speculative decoding. Note: arXiv:2211.17192
  External Links: [Link](https://doi.org/10.48550/arXiv.2211.17192)
  Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-A](#S2.SS1.p1.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-A](#S2.SS1.p2.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§IV-A](#S4.SS1.p3.1 "IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[5\] C. Chen, S. Borgeaud, G. Irving, J. Lespiau, L. Sifre, and J.
  Jumper (2023) Accelerating large language model decoding with
  speculative sampling. Note: arXiv:2302.01318 External Links:
  [Link](https://doi.org/10.48550/arXiv.2302.01318) Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§IV-A](#S4.SS1.p3.1 "IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[6\] Y. Zhou, K. Lyu, A. S. Rawat, A. K. Menon, A. Rostamizadeh, S.
  Kumar, J. Kagy, and R. Agarwal (2024) DistillSpec: improving
  speculative decoding via knowledge distillation. Note:
  arXiv:2310.08461 External Links:
  [Link](https://doi.org/10.48550/arXiv.2310.08461) Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§I](#S1.p4.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[7\] F. Hong, R. Raju, J. L. Li, B. Li, U. Thakker, A.
  Ravichandran, S. Jain, and C. Hu (2025) Training domain draft models
  for speculative decoding: best practices and insights. Note:
  arXiv:2503.07807 External Links:
  [Link](https://doi.org/10.48550/arXiv.2503.07807) Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§III-C](#S3.SS3.p1.1 "III-C Online Draft Model Adaptation ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[8\] Fireworks AI (2024)Speculative decoding(Website) Note: Accessed:
  2026-04-08 External Links:
  [Link](https://docs.fireworks.ai/deployments/speculative-decoding)
  Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[9\] Together AI (2025)Dedicated inference(Website) Note: Accessed:
  2026-04-08 External Links:
  [Link](https://docs.together.ai/docs/dedicated-inference) Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[10\] Baseten (2024)How we built production-ready speculative
  decoding with TensorRT-LLM(Website) Note: Accessed: 2026-04-08
  External Links:
  [Link](https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/)
  Cited by:
  [§I](#S1.p2.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[11\] Y. Wang, Y. Chen, Z. Li, X. Kang, Z. Tang, X. He, R. Guo, X.
  Wang, Q. Wang, A. C. Zhou, and X. Chu (2024) BurstGPT: a real-world
  workload dataset to optimize LLM serving systems. Note:
  arXiv:2401.17644 External Links:
  [Link](https://doi.org/10.48550/arXiv.2401.17644) Cited by:
  [§I](#S1.p3.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§V-B](#S5.SS2.p2.1 "V-B Throughput Improvement over Time ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[12\] I. Gim, G. Chen, S. Lee, N. Sarda, A. Khandelwal, and L.
  Zhong (2024) Prompt cache: modular attention reuse for low-latency
  inference. Note: arXiv:2311.04934 External Links:
  [Link](https://doi.org/10.48550/arXiv.2311.04934) Cited by:
  [§I](#S1.p3.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[13\] L. Zheng, W. Chiang, Y. Sheng, T. Li, S. Zhuang, Z. Wu, Y.
  Zhuang, Z. Li, Z. Lin, E. P. Xing, J. E. Gonzalez, I. Stoica, and H.
  Zhang (2024) LMSYS-Chat-1M: a large-scale real-world LLM conversation
  dataset. Note: arXiv:2309.11998 External Links:
  [Link](https://doi.org/10.48550/arXiv.2309.11998) Cited by:
  [§I](#S1.p3.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[14\] Y. Xiang, X. Li, K. Qian, W. Yu, E. Zhai, and X. Jin (2025)
  ServeGen: workload characterization and generation of large language
  model serving in production. Note: arXiv:2505.09999 External Links:
  [Link](https://doi.org/10.48550/arXiv.2505.09999) Cited by:
  [§I](#S1.p3.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§V-B](#S5.SS2.p2.1 "V-B Throughput Improvement over Time ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[15\] X. Liu, L. Hu, P. Bailis, A. Cheung, Z. Deng, I. Stoica, and H.
  Zhang (2024) Online speculative decoding. Note: arXiv:2310.07177
  External Links: [Link](https://doi.org/10.48550/arXiv.2310.07177)
  Cited by:
  [§I](#S1.p4.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§VI-A](#S6.SS1.p1.1 "VI-A Draft Architecture Dependence ‣ VI Discussion ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[16\] M. Yan, S. Agarwal, and S. Venkataraman (2025) Decoding
  speculative decoding. Note: arXiv:2402.01528 External Links:
  [Link](https://doi.org/10.48550/arXiv.2402.01528) Cited by:
  [§I](#S1.p4.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[17\] K. Huang, H. Wu, Z. Shi, H. Zou, M. Yu, and Q. Shi (2025)
  AdaSpec: adaptive speculative decoding for fast, SLO-aware large
  language model serving. In Proceedings of the 2025 ACM Symposium on
  Cloud Computing, pp. 361–374. External Links:
  [Link](https://doi.org/10.1145/3772052.3772239) Cited by:
  [§I](#S1.p5.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[18\] K. Huang, X. Guo, and M. Wang (2025) SpecDec++: boosting
  speculative decoding via adaptive candidate lengths. Note:
  arXiv:2405.19715 External Links:
  [Link](https://doi.org/10.48550/arXiv.2405.19715) Cited by:
  [§I](#S1.p5.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[19\] Y. Hou, F. Zhang, C. Du, X. Zhang, J. Pan, T. Pang, C.
  Du, V. Y. F. Tan, and Z. Yang (2025) BanditSpec: adaptive speculative
  decoding via bandit algorithms. Note: arXiv:2505.15141 External Links:
  [Link](https://doi.org/10.48550/arXiv.2505.15141) Cited by:
  [§I](#S1.p5.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[20\] Y. Mei, Y. Zhuang, X. Miao, J. Yang, Z. Jia, and R.
  Vinayak (2025) Helix: serving large language models over heterogeneous
  GPUs and network via max-flow. In Proceedings of the 30th ACM
  International Conference on Architectural Support for Programming
  Languages and Operating Systems, External Links:
  [Link](https://doi.org/10.1145/3669940.3707215) Cited by:
  [§I](#S1.p8.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[21\] T. Griggs, X. Liu, J. Yu, D. Kim, W. Chiang, A. Cheung, and I.
  Stoica (2024) Mélange: cost efficient large language model serving by
  exploiting GPU heterogeneity. Note: arXiv:2404.14527 External Links:
  [Link](https://doi.org/10.48550/arXiv.2404.14527) Cited by:
  [§I](#S1.p8.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[22\] Y. Zhang, H. Shen, R. Yang, D. Tian, Y. Luo, M. Zhang, L.
  Li, C. Hu, T. Wo, C. Song, and J. Ouyang (2025) Cauchy: a
  cost-efficient LLM serving system through adaptive heterogeneous
  deployment. In Proceedings of the 2025 ACM Symposium on Cloud
  Computing, pp. 881–893. External Links:
  [Link](https://doi.org/10.1145/3772052.3772264) Cited by:
  [§I](#S1.p8.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[23\] K. Aubrey and F. Ghodsian (2026)Inside NVIDIA Groq 3 LPX: the
  low-latency inference accelerator for the NVIDIA Vera Rubin
  platform(Website) Note: Accessed: 2026-08-29 External Links:
  [Link](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)
  Cited by:
  [§I](#S1.p8.1 "I Introduction ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[24\] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E.
  Gonzalez, H. Zhang, and I. Stoica (2023) Efficient memory management
  for large language model serving with PagedAttention. In Proceedings
  of the 29th Symposium on Operating Systems Principles, pp. 611–626.
  External Links: [Link](https://doi.org/10.1145/3600006.3613165) Cited
  by:
  [§II-A](#S2.SS1.p1.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§III-B](#S3.SS2.p2.1 "III-B Serving-Time Training Signal Extraction ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§IV-A](#S4.SS1.p11.1 "IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[25\] R. K. Ramakrishnan, Z. Yuan, S. Zhuo, C. Feng, Y. Lin, C. Su,
  and X. Zhang (2025) OmniDraft: a cross-vocabulary, online adaptive
  drafter for on-device speculative decoding. Note: arXiv:2507.02659
  External Links: [Link](https://doi.org/10.48550/arXiv.2507.02659)
  Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[26\] Y. Qian, H. Wu, Y. Fu, H. Zhang, and P. Zhao (2026) When drafts
  evolve: speculative decoding meets online learning. Note:
  arXiv:2603.12617 External Links:
  [Link](https://doi.org/10.48550/arXiv.2603.12617) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[27\] A. Kumar, S. Sanghavi, and P. Das (2026) Test-time speculation.
  Note: arXiv:2605.09329 External Links:
  [Link](https://doi.org/10.48550/arXiv.2605.09329) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[28\] X. Liu, J. Park, L. Hu, W. Kwon, Z. Li, C. Zhang, K. Du, X.
  Mo, K. You, A. Cheung, Z. Deng, I. Stoica, and H. Zhang (2024)
  TurboSpec: closed-loop speculation control system for optimizing LLM
  serving goodput. Note: arXiv:2406.14066 External Links:
  [Link](https://doi.org/10.48550/arXiv.2406.14066) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[29\] Z. Zhang, J. Xu, T. Liang, X. Chen, Z. He, R. Wang, and Z.
  Tu (2025) Draft model knows when to stop: self-verification
  speculative decoding for long-form generation. In Proceedings of the
  2025 Conference on Empirical Methods in Natural Language Processing,
  pp. 16685–16697. External Links:
  [Link](https://aclanthology.org/2025.emnlp-main.844/),
  [Document](https://dx.doi.org/10.18653/v1/2025.emnlp-main.844) Cited
  by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[30\] S. Agrawal, W. Jeon, and M. Lee (2024) AdaEDL: early draft
  stopping for speculative decoding of large language models via an
  entropy-based lower bound on token acceptance probability. In
  Proceedings of The 4th NeurIPS Efficient Natural Language and Speech
  Processing Workshop, Proceedings of Machine Learning Research, Vol.
  262, pp. 355–369. External Links:
  [Link](https://proceedings.mlr.press/v262/agrawal24a.html) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[31\] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T.
  Dao (2024) Medusa: simple LLM inference acceleration framework with
  multiple decoding heads. Note: arXiv:2401.10774 External Links:
  [Link](https://doi.org/10.48550/arXiv.2401.10774) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[32\] Y. Li, F. Wei, C. Zhang, and H. Zhang (2025) EAGLE-3: scaling
  up inference acceleration of large language models via training-time
  test. Note: arXiv:2503.01840 External Links:
  [Link](https://doi.org/10.48550/arXiv.2503.01840) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§II-B](#S2.SS2.p1.1 "II-B EAGLE: Hidden-State-Based Draft Models ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§III-B](#S3.SS2.p1.1 "III-B Serving-Time Training Signal Extraction ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§IV-A](#S4.SS1.p11.1 "IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[33\] Z. Ankner, R. Parthasarathy, A. Nrusimha, C. Rinard, J.
  Ragan-Kelley, and W. Brandon (2024) Hydra: sequentially-dependent
  draft heads for Medusa decoding. In Proceedings of the First
  Conference on Language Modeling, Note: arXiv:2402.05109 External
  Links: [Link](https://doi.org/10.48550/arXiv.2402.05109) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[34\] A. Zhang, C. Wang, Y. Wang, X. Zhang, and Y. Cheng (2024)
  Recurrent drafter for fast speculative decoding in large language
  models. Note: arXiv:2403.09919 External Links:
  [Link](https://doi.org/10.48550/arXiv.2403.09919) Cited by:
  [§II-A](#S2.SS1.p4.1 "II-A Speculative Decoding in Dynamic Workloads ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[35\] J. Duan, R. Lu, H. Duanmu, X. Li, X. Zhang, D. Lin, I. Stoica,
  and H. Zhang (2024) MuxServe: flexible spatial-temporal multiplexing
  for multiple LLM serving. Note: arXiv:2404.02015 External Links:
  [Link](https://doi.org/10.48550/arXiv.2404.02015) Cited by:
  [§II-C](#S2.SS3.p1.1 "II-C Heterogeneous Systems for LLM Inference ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[36\] Y. Xiang et al. (2025) Aegaeon: effective GPU pooling for
  concurrent LLM serving on the market. In Proceedings of the 31st ACM
  Symposium on Operating Systems Principles (SOSP), External Links:
  [Link](https://doi.org/10.1145/3731569.3764815) Cited by:
  [§II-C](#S2.SS3.p1.1 "II-C Heterogeneous Systems for LLM Inference ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[37\] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin,
  and H. Zhang (2024) DistServe: disaggregating prefill and decoding for
  goodput-optimized large language model serving. Note: arXiv:2401.09670
  External Links: [Link](https://doi.org/10.48550/arXiv.2401.09670)
  Cited by:
  [§II-C](#S2.SS3.p1.1 "II-C Heterogeneous Systems for LLM Inference ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[38\] Y. Jiang, R. Yan, and B. Yuan (2025) HexGen-2: disaggregated
  generative inference of LLMs in heterogeneous environment. In
  International Conference on Learning Representations, External Links:
  [Link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0b941a1e5fbce23fe46b049999d04ed0-Abstract-Conference.html)
  Cited by:
  [§II-C](#S2.SS3.p1.1 "II-C Heterogeneous Systems for LLM Inference ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[39\] X. Chen, R. Shi, L. Zhao, L. Wang, X. Jin, Y. Chen, and H.
  Sun (2025) Disaggregated prefill and decoding inference system for
  large language model serving on multi-vendor GPUs. Note:
  arXiv:2509.17542 External Links:
  [Link](https://doi.org/10.48550/arXiv.2509.17542) Cited by:
  [§II-C](#S2.SS3.p1.1 "II-C Heterogeneous Systems for LLM Inference ‣ II Background ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[40\] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S.
  Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y.
  Sheng (2024) SGLang: efficient execution of structured language model
  programs. Note: arXiv:2312.07104 External Links:
  [Link](https://doi.org/10.48550/arXiv.2312.07104) Cited by:
  [§III-B](#S3.SS2.p2.1 "III-B Serving-Time Training Signal Extraction ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§IV-A](#S4.SS1.p11.1 "IV-A Adaptive Speculative Decoding Control ‣ IV Adaptive Control and Runtime Optimization ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§V-A](#S5.SS1.p1.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[41\] S. Li, Y. Zhu, C. Wang, F. Yin, S. Shi, Y. Wang, Y. Zhang, Y.
  Huang, H. Zheng, and Y. Zhang (2025)SpecForge: train speculative
  decoding models effortlessly(Website) GitHub. External Links:
  [Link](https://github.com/sgl-project/specforge) Cited by:
  [§III-C](#S3.SS3.p1.1 "III-C Online Draft Model Adaptation ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference"),
  [§V-A](#S5.SS1.p1.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[42\] Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. Xu, L.
  Wright, H. Shojanazeri, M. Ott, S. Shleifer, A. Desmaison, C.
  Balioglu, P. Damania, B. Nguyen, G. Chauhan, Y. Hao, A. Mathews,
  and S. Li (2023) PyTorch FSDP: experiences on scaling fully sharded
  data parallel. Proceedings of the VLDB Endowment 16 (12),
  pp. 3848–3860. External Links:
  [Link](https://doi.org/10.14778/3611540.3611569) Cited by:
  [§III-C](#S3.SS3.p3.1 "III-C Online Draft Model Adaptation ‣ III TIDE Architecture ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[43\] OpenAI (2025) Gpt-oss-120b & gpt-oss-20b model card. Note:
  arXiv:2508.10925 External Links:
  [Link](https://doi.org/10.48550/arXiv.2508.10925) Cited by:
  [§V-A](#S5.SS1.p3.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[44\] Qwen Team (2025) Qwen3 technical report. Note: arXiv:2505.09388
  External Links: [Link](https://doi.org/10.48550/arXiv.2505.09388)
  Cited by:
  [§V-A](#S5.SS1.p3.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[45\] Meta AI (2025)Llama 4 scout 17b-16e(Website) Note: Accessed:
  2026-04-08 External Links:
  [Link](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) Cited
  by:
  [§V-A](#S5.SS1.p3.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[46\] A. Grattafiori, A. Dubey, A. Jauhri, et al. (2024) The llama 3
  herd of models. Note: arXiv:2407.21783 External Links:
  [Link](https://doi.org/10.48550/arXiv.2407.21783) Cited by:
  [§V-A](#S5.SS1.p3.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[47\] Aeala (2023)ShareGPT vicuna unfiltered(Website) External Links:
  [Link](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[48\] CAMEL-AI.org (2023)Biology dataset(Website) External Links:
  [Link](https://huggingface.co/datasets/camel-ai/biology) Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[49\] CAMEL-AI.org (2023)Chemistry dataset(Website) External Links:
  [Link](https://huggingface.co/datasets/camel-ai/chemistry) Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[50\] CAMEL-AI.org (2023)Physics dataset(Website) External Links:
  [Link](https://huggingface.co/datasets/camel-ai/physics) Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[51\] AI Mathematical Olympiad (2024)NuminaMath-cot(Website) External
  Links: [Link](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[52\] theblackcat102 (2023)Evol-codealpaca-v1(Website) External
  Links:
  [Link](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[53\] FreedomIntelligence (2023)Alpaca-gpt4-korean(Website) External
  Links:
  [Link](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-korean)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[54\] FreedomIntelligence (2023)Alpaca-gpt4-arabic(Website) External
  Links:
  [Link](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-arabic)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[55\] FreedomIntelligence (2023)Alpaca-gpt4-chinese(Website) External
  Links:
  [Link](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-chinese)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
- \[56\] FreedomIntelligence (2023)Alpaca-gpt4-french(Website) External
  Links:
  [Link](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-french)
  Cited by:
  [§V-A](#S5.SS1.p5.1 "V-A Experimental Setup ‣ V Evaluation ‣ TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference").
````
