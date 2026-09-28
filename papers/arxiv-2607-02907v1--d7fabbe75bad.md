---
identifier: arxiv:2607.02907v1
title: "ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"
authors:
  - Peiming Li
  - Yifan Wang
  - Xiaotian Zhang
  - Zhiyuan Hu
  - Shiyu Li
  - Zheng Wei
  - Yang Tang
published: "2026-07-03T03:14:11+00:00"
url: https://arxiv.org/abs/2607.02907v1
source: arxiv
doi: null
arxiv_id: 2607.02907v1
categories:
  - cs.CL
  - cs.CV
---

# ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space

Peiming Li^(\*) Affiliation: Basic Algorithm Center, PCG, Tencent
E-mail [{peimingli, wyattyfwang, shyuli, hemingwei,
ethanntang}@tencent.com](mailto:%7Bpeimingli,%20wyattyfwang,%20shyuli,%20hemingwei,%20ethanntang%7D@tencent.com)
   Yifan Wang^(\*) Affiliation: Basic Algorithm Center, PCG, Tencent
E-mail [{peimingli, wyattyfwang, shyuli, hemingwei,
ethanntang}@tencent.com](mailto:%7Bpeimingli,%20wyattyfwang,%20shyuli,%20hemingwei,%20ethanntang%7D@tencent.com)
   Xiaotian Zhang Affiliation: Zhejiang University
E-mail <xiaotian.24@intl.zju.edu.cn>    Zhiyuan Hu Affiliation: Peking
University E-mail <zhiyuanhu@stu.pku.edu.cn>    Shiyu Li
Affiliation: Basic Algorithm Center, PCG, Tencent E-mail [{peimingli,
wyattyfwang, shyuli, hemingwei,
ethanntang}@tencent.com](mailto:%7Bpeimingli,%20wyattyfwang,%20shyuli,%20hemingwei,%20ethanntang%7D@tencent.com)
   Zheng Wei^(†) Affiliation: Basic Algorithm Center, PCG, Tencent
E-mail [{peimingli, wyattyfwang, shyuli, hemingwei,
ethanntang}@tencent.com](mailto:%7Bpeimingli,%20wyattyfwang,%20shyuli,%20hemingwei,%20ethanntang%7D@tencent.com)
   Yang Tang^(‡†) Affiliation: Basic Algorithm Center, PCG, Tencent
E-mail [{peimingli, wyattyfwang, shyuli, hemingwei,
ethanntang}@tencent.com](mailto:%7Bpeimingli,%20wyattyfwang,%20shyuli,%20hemingwei,%20ethanntang%7D@tencent.com)

###### Abstract

Multimodal Large Language Models (MLLMs) have achieved remarkable
progress but still struggle with complex visual reasoning tasks
requiring multi-step perception and logical deduction. While explicit
visual generation incurs prohibitive computational costs, existing
latent approaches often rely on external experts or lack rigorous
cognitive logic. In this paper, we introduce ProLaViT (Progressive
Latent Visual Thought), a framework empowering MLLMs to perform
structured visual derivation in the continuous latent space. Unlike
works dependent on heterogeneous external models, ProLaViT leverages an
endogenous self-distillation mechanism, utilizing the model’s own visual
encoder to supervise latent thoughts. To facilitate this, we construct a
scalable programmatic synthesis pipeline enabling the model to
internalize algorithmic precision without inference-time tools. We
design two reasoning paradigms: (1) Coarse-to-Fine Causal Chain for
spatial tasks, guiding attention from global context to local targets.
(2) Dialectical Reasoning Chain for logical tasks, incorporating
counterfactual thinking for verification. Furthermore, we propose a
Distance-Weighted Diversity Loss to impose topology-aware constraints,
preventing feature degeneration by enforcing semantic distinctiveness.
Extensive experiments demonstrate that ProLaViT outperforms baselines on
vision-centric benchmarks, achieving superior accuracy and
interpretability with high efficiency.

###### Keywords: 

Multimodal Large Language Models Latent Visual Reasoning Progressive
Visual Derivation

^(†)^(†)footnotetext: ^(\*)Equal contribution.  ^(†)Corresponding
author.  ^(‡)Project Lead.

![Refer to caption](2607.02907v1/figure1_cropped.png)

Figure 1: Comparison of multimodal reasoning paradigms. Top: Textual CoT
suffers from a modality gap, failing to accurately ground spatial
relationships (e.g., mistaking the wood flooring for the cabinet).
Middle: One-step Latent Prediction attempts to identify the target
region in a single leap but suffers from precision loss, often attending
to irrelevant background areas (hallucination). Bottom: ProLaViT (Ours)
decomposes the task into a progressive latent chain (Locate $`\to`$
Focus $`\to`$ Isolate), enabling precise target isolation and robust
reasoning.

## 1 Introduction

The advent of Multimodal Large Language Models
(MLLMs) \[[17](#bib.bib16), [1](#bib.bib3), [20](#bib.bib17)\] has
revolutionized the intersection of vision and language, enabling systems
to describe images and answer open-ended questions with unprecedented
fluency. Nevertheless, despite these achievements, current MLLMs
frequently exhibit limitations when addressing visual reasoning tasks
that necessitate multi-step spatial perception, geometric analysis, or
logical deduction \[[29](#bib.bib6), [7](#bib.bib5), [28](#bib.bib20),
[19](#bib.bib19)\]. The core challenge lies in the reasoning mechanism,
specifically in how to effectively bridge the gap between high-level
linguistic semantics and low-level visual features.

Prevalent approaches rely on Textual Chain-of-Thought
(CoT) \[[31](#bib.bib1), [38](#bib.bib2)\] to decompose complex tasks.
However, as illustrated in
Fig. [1](#S0.F1 "Figure 1 ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
reasoning solely in text suffers from a fundamental modality gap. Text
is inherently abstract and often fails to capture fine-grained spatial
nuances. For instance, when asked to identify the furniture beneath a
television, the model fails to ground the spatial relationship,
hallucinating “wood flooring” instead of the correct “cabinet”. To
address this, recent works have explored incorporating visual
information into the reasoning chain. Explicit visual generation methods
such as Anole \[[3](#bib.bib23)\], Bagel \[[4](#bib.bib22)\],
Zebra-CoT \[[15](#bib.bib24)\] and ThinkMorph \[[9](#bib.bib9)\]
generate intermediate pixel-level images to aid reasoning. While
effective, the computational cost of iterative image generation is
prohibitive for real-time applications. Conversely, implicit latent
reasoning approaches \[[16](#bib.bib15), [36](#bib.bib8),
[10](#bib.bib14), [23](#bib.bib10), [6](#bib.bib25), [30](#bib.bib26)\]
attempt to model reasoning within the continuous feature space. However,
as illustrated in
Fig. [1](#S0.F1 "Figure 1 ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
these approaches typically adhere to a One-step Latent Prediction
paradigm. Attempting to directly localize a specific visual region such
as a helper image crop within a single step often exceeds the
representational capacity of the model. This limitation frequently
results in significant precision loss where the model focuses on
irrelevant background elements like the floor rather than the intended
target, ultimately leading to fragile representations and erroneous
predictions.

In this work, we argue that effective visual reasoning requires neither
expensive pixel generation nor unstructured latent guessing, but rather
Progressive Visual Derivation within a structured latent space. We
introduce ProLaViT, a novel framework that decomposes complex visual
tasks into a structured chain of latent thoughts. Instead of jumping to
a conclusion, it mimics human cognitive processes, following a Locate
$`\to`$ Focus $`\to`$ Isolate causal chain to progressively refine
visual attention, ensuring precise object isolation and robust
reasoning.

A key challenge in training such multi-step latent models is the absence
of ground truth for intermediate images. Unlike prior works that rely on
external vision experts such as \[[35](#bib.bib12), [13](#bib.bib11),
[21](#bib.bib13)\], ProLaViT employs an Endogenous Self-Distillation
mechanism. This approach effectively avoids the domain gaps and pipeline
complexity often introduced by external dependencies. We leverage the
MLLM’s own frozen vision encoder to extract features from a sequence of
auxiliary images. Crucially, these auxiliary sequences are generated via
a scalable programmatic synthesis pipeline. By training on these
rigorous, code-generated visual trajectories, ProLaViT effectively
internalizes algorithmic precision, learning to perform latent
operations in its latent space without needing external tools during
inference.

Furthermore, a common pathology in sequential latent reasoning is latent
collapse, where the representations of subsequent steps become
indistinguishable, degrading the reasoning chain into redundancy. To
mitigate this, we propose a novel Distance-Weighted Diversity Loss. This
objective function explicitly constrains the topology of the latent
space, penalizing cosine similarity between reasoning steps. Crucially,
the penalty is weighted by the causal distance in the chain, forcing the
latent state of a segmentation step to be significantly distinct from
the initial global view step, thereby ensuring that each step
contributes unique and progressive information to the inference process.
Our main contributions are summarized as follows:

- •
  We propose ProLaViT, a framework that enables MLLMs to perform
  Progressive Visual Derivation in the latent space, overcoming the
  efficiency bottlenecks of explicit generation and the precision
  limitations of unstructured latent methods.
- •
  We introduce an Endogenous Self-Distillation strategy supported by a
  programmatic data synthesis pipeline, allowing the model to
  internalize rigorous algorithmic logic without external vision
  experts.
- •
  We design a Distance-Weighted Diversity Loss to prevent latent
  collapse, enforcing semantic distinctiveness between reasoning steps
  proportional to their causal distance.
- •
  Extensive experiments demonstrate that ProLaViT achieves SOTA
  performance, offering a superior balance of accuracy,
  interpretability, and efficiency.

## 2 Related Work

### 2.1 Multimodal Chain-of-Thought Reasoning

Chain-of-Thought (CoT) prompting \[[31](#bib.bib1)\] enhances LLM
reasoning by decomposing problems into logical steps. Multimodal
CoT \[[38](#bib.bib2), [11](#bib.bib27), [12](#bib.bib28)\] extends this
to vision-language tasks, and works like Qwen-VL \[[1](#bib.bib3)\] and
LLaVA-CoT \[[34](#bib.bib4)\] demonstrate that reasoning data improves
performance. However, these methods rely on projecting visual
information into discrete text. Recent studies \[[7](#bib.bib5),
[29](#bib.bib6)\] indicate this creates a modality gap, as abstract text
often fails to capture fine-grained spatial nuances. This leads to
hallucinations where reasoning is linguistically coherent yet visually
inaccurate. In contrast, ProLaViT performs reasoning directly in the
continuous latent space, preserving visual fidelity.

### 2.2 Visual-Augmented Reasoning with External Tools

To bridge the modality gap, researchers have augmented MLLMs with
external tools \[[32](#bib.bib30), [25](#bib.bib31), [27](#bib.bib32),
[37](#bib.bib33), [18](#bib.bib34)\] or explicit visual
generation \[[9](#bib.bib9), [3](#bib.bib23), [5](#bib.bib35),
[8](#bib.bib36), [14](#bib.bib37)\]. Specifically, methods like
CoVT \[[23](#bib.bib10), [24](#bib.bib7)\] employ experts such as
SAM \[[13](#bib.bib11)\] or DepthAnything \[[35](#bib.bib12)\] for
supervision. However, these approaches face two critical limitations:
(1) Prohibitive Cost due to pixel-level generation or external inference
latency; and (2) Pipeline Complexity arising from heterogeneous
dependencies. In contrast, ProLaViT adopts an endogenous
self-distillation mechanism, leveraging the MLLM’s own frozen encoder to
supervise latent thoughts, eliminating external overhead.

### 2.3 Latent Visual Reasoning

Recent works explore reasoning within continuous latent spaces to
balance efficiency and expressiveness. While language-centric
methods \[[10](#bib.bib14), [26](#bib.bib29)\] introduce continuous
tokens, they lack spatial grounding. Visual adaptations like
Mirage \[[36](#bib.bib8)\] and LVR \[[16](#bib.bib15)\] map visual
semantics into LLMs but typically adopt an unstructured or single-step
paradigm. As our analysis reveals, this often leads to latent collapse,
where intermediate representations degenerate into redundancy. ProLaViT
advances this direction by introducing Structured Progressive
Derivation. Instead of black-box transitions, we enforce a rigorous
causal chain (Locate $`\to`$ Focus $`\to`$ Isolate) constrained by a
Distance-Weighted Diversity Loss, ensuring that latent thoughts remain
topologically distinct and logically progressive.

## 3 Method

### 3.1 Overview

![Refer to caption](2607.02907v1/figure2_cropped.png)

Figure 2: Overview of ProLaViT. The framework generates progressive
latent thoughts (green tokens) supervised by an Endogenous
Self-Distillation mechanism (Right), where latent states are aligned
with visual features from synthesized auxiliary images via
$`\mathcal{L}_{MSE}`$. To prevent representation degeneration, a
Distance-Weighted Diversity Loss ($`\mathcal{L}_{div}`$, Left) penalizes
excessive similarity between reasoning steps, enforcing topological
distinctiveness in the latent space.

![Refer to caption](2607.02907v1/figure3.png)

Figure 3: Illustration of the two reasoning paradigms in ProLaViT. Top:
The Coarse-to-Fine Causal Chain for spatial tasks progressively narrows
attention from global context to specific targets (Locate $`\to`$ Focus
$`\to`$ Isolate). Bottom: The Dialectical Reasoning Chain for logical
tasks (e.g., jigsaw puzzles) employs a trial-and-error approach
(Hypothesize $`\to`$ Critique $`\to`$ Verify) to validate visual
consistency.

Fig. [2](#S3.F2 "Figure 2 ‣ 3.1 Overview ‣ 3 Method ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space")
illustrates the overall architecture of ProLaViT. Given an input image
$`I`$ and a textual query $`Q`$, ProLaViT performs visual reasoning
through a progressive latent thought chain rather than a single-step
prediction. The framework consists of three core components: (1) A
Progressive Latent Visual Thought module that decomposes complex visual
tasks into a structured sequence of $`K`$ intermediate reasoning steps.
(2) An Endogenous Self-Distillation mechanism that leverages the model’s
own vision encoder to provide supervision signals. (3) A
Distance-Weighted Diversity Loss that prevents latent collapse by
enforcing topology-aware constraints on the reasoning chain. Formally,
let $`\mathcal{M}=(\mathcal{V},\mathcal{L})`$ denote a Multimodal Large
Language Model with vision encoder $`\mathcal{V}`$ and language model
$`\mathcal{L}`$. Given an input image $`I`$ and query $`Q`$, ProLaViT
generates a sequence of latent thoughts
$`\mathcal{Z}=\{z_{1},z_{2},\ldots,z_{K}\}`$ that progressively refine
the visual understanding before producing the final answer $`A`$.

### 3.2 Progressive Latent Visual Thought

Unlike prior methods that attempt to predict complex visual cues in a
single step, ProLaViT decomposes visual reasoning into a structured
chain of latent thoughts, as illustrated in
Fig. [3](#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Method ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
We design two distinct reasoning paradigms tailored for different task
types.

Coarse-to-Fine Causal Chain. For spatial tasks such as visual search and
object localization, we adopt a Locate $`\rightarrow`$ Focus
$`\rightarrow`$ Isolate paradigm
(Fig. [3](#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Method ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
Top). Each reasoning step progressively narrows the attention scope:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
z_{k}=\mathcal{T}_{k}(z_{k-1},I,c_{k}),\quad k\in\{1,2,\ldots,K\},
``` |  | (1) |

where $`\mathcal{T}_{k}`$ denotes the $`k`$-th reasoning transformation,
and $`c_{k}`$ represents the task-specific context. Specifically, we
implement four canonical operations that instantiate this paradigm:

- •
  Grid View ($`z_{1}`$): Establishes global context with spatial grid
  overlay (Preparation).
- •
  Bounding Box ($`z_{2}`$): Performs the Locate step via region
  localization.
- •
  Crop ($`z_{3}`$): Executes the Focus step for detailed inspection of
  the target region.
- •
  Segmentation ($`z_{4}`$): Completes the Isolate step with fine-grained
  object boundary delineation.

Dialectical Reasoning Chain. For logical tasks such as jigsaw puzzle
assembly, we employ a Hypothesize $`\rightarrow`$ Critique
$`\rightarrow`$ Verify loop that incorporates counterfactual thinking
(Fig. [3](#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Method ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
Bottom). This paradigm enables the model to evaluate multiple hypotheses
before committing to a solution. Specifically, we instantiate this loop
through four distinct visual operations:

- •
  Edge Enhancement ($`z_{1}`$): Highlights high-frequency boundary
  features to identify potential mating surfaces, forming the initial
  Hypothesis.
- •
  Directional Guidance ($`z_{2}`$): Visualizes the proposed movement
  vector using arrow overlays to suggest the assembly orientation.
- •
  Counterfactual Simulation ($`z_{3}`$): Generates a plausible but
  incorrect assembly state (e.g., misalignment) to trigger the Critique
  mechanism, teaching the model to distinguish subtle errors.
- •
  Solution Verification ($`z_{4}`$): Presents the correctly assembled
  state to Verify the logical consistency and visual coherence of the
  final solution.

Token Representation. Each reasoning step is represented by a set of
learnable tokens in the LLM’s embedding space. Specifically, we allocate
$`N_{\text{tokens}}`$ special tokens (denoted as \<\|vit_pad_latent\|\>)
per reasoning step. For $`K=4`$ steps with $`N_{\text{tokens}}=4`$
tokens each, the total latent thought sequence contains
$`K\times N_{\text{tokens}}=16`$ tokens. These tokens are embedded in
the input sequence and trained to capture the intermediate visual
states:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{H}_{\text{latent}}=\text{LLM}(\mathbf{E}_{\text{input}}\oplus\mathbf{E}_{\text{thought}}),
``` |  | (2) |

where
$`\mathbf{E}_{\text{thought}}\in\mathbb{R}^{(K\cdot N_{\text{tokens}})\times d}`$
represents the latent thought embeddings, and $`\oplus`$ denotes
concatenation.

### 3.3 Endogenous Self-Distillation Mechanism

A key challenge in training latent reasoning models is the lack of
ground truth for intermediate mental images. Prior works rely on
external vision experts (e.g., SAM, DINO), which introduces domain gaps
and pipeline complexity. Instead, ProLaViT employs an Endogenous
Self-Distillation strategy that leverages the MLLM’s own frozen vision
encoder as the teacher.

Programmatic Synthesis Pipeline. To generate supervision signals, we
construct a scalable programmatic pipeline that produces rigorous visual
trajectories. For each training sample, we programmatically generate
$`K`$ auxiliary images
$`\{I_{1}^{\text{aux}},I_{2}^{\text{aux}},\ldots,I_{K}^{\text{aux}}\}`$
corresponding to each reasoning step (e.g., applying crop(I, bbox) or
segment(I, target)). These auxiliary images are generated automatically
using geometric transformations and are guaranteed to be algorithmically
precise. Specific implementation details are provided in the
Supplementary Materials.

Teacher Feature Extraction. The frozen vision encoder $`\mathcal{V}`$
extracts features from each auxiliary image:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{F}_{k}^{\text{teacher}}=\mathcal{V}(I_{k}^{\text{aux}}),\quad k\in\{1,\ldots,K\},
``` |  | (3) |

where $`\mathbf{F}_{k}^{\text{teacher}}\in\mathbb{R}^{L\times d}`$ with
$`L`$ denoting the number of visual tokens per image. Crucially,
gradients are not backpropagated through $`\mathcal{V}`$, ensuring the
vision encoder remains frozen during training.

Cross-Attention Knowledge Distillation. To align the LLM-generated
latent thoughts with the teacher features, we employ a single
cross-attention over the full sequence. Let
$`\mathbf{H}\in\mathbb{R}^{N\times d}`$ denote the hidden states of all
reasoning-step tokens (e.g., $`N=K\times n_{\text{tok}}`$ with
$`n_{\text{tok}}`$ tokens per step). We project and normalize:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{H}^{\text{proj}}=\text{Normalize}\bigl(\text{Linear}(\mathbf{H})\bigr).
``` |  | (4) |

A learnable query matrix $`\mathbf{Q}\in\mathbb{R}^{L\times d}`$ (one
per image length $`L`$) is expanded to length $`K\cdot L`$ (e.g., via
linear interpolation) so that the number of query positions matches the
concatenated teacher sequence. The aligned representation is computed in
one cross-attention:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle\mathbf{\hat{F}}`$ | $`\displaystyle=\text{CrossAttn}\bigl(\mathbf{Q}_{\text{expand}},\mathbf{H}^{\text{proj}},\mathbf{H}^{\text{proj}}\bigr),\quad\mathbf{\hat{F}}\in\mathbb{R}^{K\cdot L\times d},`$ |  | (5) |

where $`\mathbf{Q}_{\text{expand}}\in\mathbb{R}^{K\cdot L\times d}`$.
Key and value are both $`\mathbf{H}^{\text{proj}}`$, thus every query
position (across all $`K`$ steps) attends to the same compressed latent
tokens. We split $`\mathbf{\hat{F}}`$ into $`K`$ segments of length
$`L`$ and denote the $`k`$-th segment by
$`\mathbf{\hat{F}}_{k}\in\mathbb{R}^{L\times d}`$. The self-distillation
loss is:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{distill}}=\frac{1}{K}\sum_{k=1}^{K}\bigl\|\mathbf{\hat{F}}_{k}-\mathbf{F}_{k}^{\text{teacher}}\bigr\|_{2}^{2}.
``` |  | (6) |

### 3.4 Distance-Weighted Diversity Loss

In our experiments, we observe a limitation in the sequential reasoning
of ProLaViT, where representations of subsequent reasoning steps
gradually become indistinguishable. This phenomenon is highlighted by
the token similarity matrix in
Fig. [4](#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
Such convergence risks degrading the progressive reasoning chain into
redundancy, a failure mode we term latent collapse. To mitigate the
aforementioned degradation of the reasoning chain, we propose
introducing a diversity loss during training. However, standard
diversity losses often prove overly strict, as they indiscriminately
penalize any positive similarity. Given that visual features naturally
share common patterns (e.g., edges and textures), pursuing complete
orthogonality is therefore unrealistic. Inspired by metric learning, we
propose a margin-based diversity loss allowing reasonable similarity
while penalizing representation collapse:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{div}}^{\text{margin}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\max(0,\cos(\mathbf{g}_{i},\mathbf{g}_{j})-\tau),
``` |  | (7) |

where $`\mathbf{g}_{k}`$ denotes the centroid of the $`k`$-th reasoning
step’s tokens, $`\mathcal{P}`$ is the set of all step pairs, and
$`\tau\in[0,1]`$ is a learnable margin threshold. A key insight is that
reasoning steps with larger causal distance should be more distinct. To
capture this, we introduce a distance-weighted penalty:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
w_{ij}=\epsilon+(1-\epsilon)\cdot\left(\frac{d(i,j)}{d_{\max}}\right)^{\alpha},
``` |  | (8) |

where $`d(i,j)=|i-j|`$ is the causal distance, $`d_{\max}=K-1`$,
$`\epsilon=0.1`$ is the minimum weight, and $`\alpha\geq 1`$ is a
learnable parameter controlling steepness, parameterized as
$`\alpha=1+\text{softplus}(\theta_{\alpha})`$. The final loss is:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{div}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}w_{ij}\cdot\max(0,\cos(\mathbf{g}_{i},\mathbf{g}_{j})-\tau).
``` |  | (9) |

This formulation respects the hierarchical structure of the reasoning
chain while effectively preventing representation collapse.

### 3.5 Training Objective and Strategy

The overall training objective combines language modeling with latent
supervision:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}=\mathcal{L}_{\text{LM}}+\lambda_{\text{distill}}\mathcal{L}_{\text{distill}}+\lambda_{\text{div}}\mathcal{L}_{\text{div}},
``` |  | (10) |

where $`\mathcal{L}_{\text{LM}}`$ is the standard cross-entropy loss. We
set $`\lambda_{\text{distill}}=1.0`$ and $`\lambda_{\text{div}}=0.2`$.

To ensure the stable emergence of progressive thoughts and prevent
representation collapse, we employ a multi-stage curriculum learning
strategy:

- •
  Phase 1: Latent Anchor Initialization. We freeze the vision encoder
  and LLM backbone, training only the embeddings of the special thought
  tokens. This phase acts as a “warm-up,” initializing the latent tokens
  to a neutral state within the semantic space of the LLM, preventing
  early optimization instability.
- •
  Phase 2: Endogenous Knowledge Distillation. We activate the
  Self-Distillation mechanism ($`\mathcal{L}_{\text{distill}}`$). Here,
  the model learns to align its generated latent thoughts with the
  target visual features extracted by its own frozen encoder. Crucially,
  we do not yet enforce the diversity constraint, allowing the model to
  focus solely on accurate feature reconstruction.
- •
  Phase 3: Structured Chain Evolution. We introduce the
  Distance-Weighted Diversity Loss ($`\mathcal{L}_{\text{div}}`$)
  alongside the distillation loss. This is the critical phase where the
  reasoning chain transforms from a repetitive sequence into a
  structured, progressive derivation. The model is forced to
  differentiate adjacent reasoning steps (e.g., separating Locate from
  Focus), establishing the causal topology in the latent space.
- •
  Phase 4: Global Capability Integration. Finally, we perform joint
  training with general VQA data. This prevents catastrophic forgetting
  of general instruction-following abilities while consolidating the
  learned visual reasoning patterns into the model’s global behavior.

## 4 Experiment

### 4.1 Experimental Setup

We implement ProLaViT based on Qwen2.5-VL-7B-Instruct. The latent
thought uses $`K=4`$ reasoning steps with $`N_{\text{tokens}}=4`$ tokens
per step. Training is performed on 2 NVIDIA H20 GPUs. For Diversity
Loss, $`\lambda_{\text{div}}=0.2`$, margin $`\tau`$ initialized to 0.8.

Benchmarks. We evaluate ProLaViT on a comprehensive suite of
perception-intensive visual reasoning benchmarks that span multiple
dimensions of visual understanding: MMVP \[[29](#bib.bib6)\],
VisPuzzle \[[9](#bib.bib9)\], VStar \[[33](#bib.bib18)\],
ChartQA \[[19](#bib.bib19)\], BL
INK \[[7](#bib.bib5)\], CV-Bench \[[28](#bib.bib20)\].

|  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
| Method | MMVP | VisPuzzle | VStar | ChartQA | BLINK | CV-Bench | BLINK-J | Avg. |
| Base Model |  |  |  |  |  |  |  |  |
| Qwen2.5-VL-Instruct | 77.33 | 34.75 | 76.44 | 78.45 | 54.49 | 73.61 | 59.33 | 66.00 |
| Text-Space Reasoning |  |  |  |  |  |  |  |  |
| SFT | 77.00 | 67.50 | 78.01 | 76.74 | 55.60 | 77.36 | 49.33 | 69.86 |
| CoT SFT | 77.66 | 69.75 | 75.91 | 75.83 | 56.54 | 64.55 | 68.66 | 69.18 |
| Latent-Space Reasoning |  |  |  |  |  |  |  |  |
| CoVT | 78.33 | 35.75 | 79.05 | 24.99 | 57.49 | 75.00 | 66.00 | 61.45 |
| LVR | 77.30 | 35.00 | 76.43 | 64.86 | 53.02 | 76.55 | 57.33 | 64.63 |
| One-step Latent Pred. | 79.33 | 72.00 | 78.01 | 63.02 | 56.28 | 76.10 | 66.00 | 70.85 |
| Ours (Progressive Latent Visual Thought) |  |  |  |  |  |  |  |  |
| ProLaViT (Base) | 78.00 | 74.00 | 79.05 | 76.18 | 57.49 | 77.24 | 71.33 | 73.82 |
| ProLaViT + $`\mathcal{L}_{\text{div}}^{\text{margin}}`$ | 78.33 | 75.25 | 79.05 | 75.66 | 56.33 | 78.33 | 67.33 | 73.58 |
| ProLaViT + $`\mathcal{L}_{\text{div}}`$ (Full) | 79.00 | 74.25 | 80.11 | 78.99 | 57.23 | 77.66 | 76.00 | 75.11 |

Table 1: Performance comparison on visual reasoning benchmarks. ProLaViT
(Full) denotes our model trained with the proposed Distance-Weighted
Diversity Loss. All values are reported in accuracy (%). Best results
are in bold, second best are underlined.

Baselines. We compare ProLaViT against the following methods:

- •
  Qwen2.5-VL-Instruct \[[2](#bib.bib21)\]: The base MLLM without any
  fine-tuning, serving as the foundation model.
- •
  SFT: Standard supervised fine-tuning on the same training data without
  latent reasoning.
- •
  CoT SFT: Fine-tuning with explicit textual Chain-of-Thought reasoning
  annotations.
- •
  CoVT \[[23](#bib.bib10)\]: Chain-of-Visual-Thought that distills
  knowledge from external lightweight vision experts.
- •
  LVR \[[16](#bib.bib15)\]: Latent Visual Reasoning that enables
  reasoning in the visual embedding space.
- •
  One-step Latent Prediction: An ablated variant that predicts all
  visual features in a single step without progressive decomposition.

### 4.2 Quantitative Results

Tab. [1](#S4.T1 "Table 1 ‣ 4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space")
summarizes the performance comparison across diverse visual reasoning
benchmarks. Our proposed framework, ProLaViT (equipped with the full
Distance-Weighted Diversity Loss), consistently outperforms baselines,
achieving the highest average accuracy of 75.11%.

Analysis. We highlight four key observations from the results:

1.  1.
    Structured Decomposition vs. Monolithic Prediction. Comparing
    ProLaViT with One-step Latent Prediction, we observe significant
    gains, particularly on complex tasks like VisPuzzle (+2.25%) and
    BLINK-Jigsaw (+10.00%). This confirms that monolithic latent
    prediction often hits a performance ceiling due to capacity
    overload. In contrast, our Locate $`\rightarrow`$ Focus
    $`\rightarrow`$ Isolate causal chain allows the model to
    incrementally refine its attention, leading to more precise visual
    grounding.
2.  2.
    Bridging the Modality Gap in Spatial Reasoning. While CoT SFT
    improves performance on logical puzzles (VisPuzzle: 69.75%), it
    suffers a sharp decline on perception-intensive benchmarks
    (CV-Bench: 64.55%). This reflects the inherent limitation of text in
    capturing fine-grained spatial coordinates. ProLaViT overcomes this
    by reasoning in the continuous latent space, maintaining strong
    performance across both logical and perceptual tasks without the
    information loss associated with text projection.
3.  3.
    Robustness via Endogenous Alignment. A critical failure mode of
    prior methods is revealed in the ChartQA benchmark. CoVT, which
    relies on external vision experts (e.g., SAM, DepthAnything),
    collapses to 24.99% accuracy, likely due to the domain gap between
    natural images (on which experts are trained) and abstract charts.
    ProLaViT avoids this pitfall by employing Endogenous
    Self-Distillation. By leveraging the MLLM’s own vision encoder, our
    method ensures domain consistency, achieving a robust 78.99% on
    ChartQA.
4.  4.
    Efficacy of Dialectical Reasoning. On the BLINK-Jigsaw benchmark,
    which demands rigorous logical verification, ProLaViT achieves
    76.00%, a remarkable +16.67% improvement over the base model. This
    substantial margin validates the effectiveness of our Hypothesize
    $`\rightarrow`$ Critique $`\rightarrow`$ Verify paradigm,
    demonstrating that structured latent thoughts can successfully
    simulate trial-and-error processes for complex problem-solving.

### 4.3 Qualitative Analysis

To intuitively understand how ProLaViT structures its reasoning process,
we visualize the cosine similarity matrices of the generated latent
tokens in
Fig. [4](#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").

![Refer to caption](2607.02907v1/figure4_cropped.png)

Figure 4: Visualization of latent thought similarity. We plot the cosine
similarity heatmaps between the token representations of different
reasoning steps ($`z_{1}`$ to $`z_{4}`$). (a) No Diversity Loss: The
chain suffers from latent collapse, where subsequent steps
($`z_{2}`$-$`z_{4}`$) become indistinguishable. (b) Margin-Based
Formulation: Enforces distinctiveness equally across all pairs,
potentially disrupting the semantic continuity of adjacent steps. (c)
Distance-Weighted Loss (Ours): Respects the hierarchical structure,
where adjacent steps retain moderate similarity for context, while
distant steps are enforced to be distinct.

Latent Collapse without Constraints. As shown in
Fig. [4](#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space")(a),
in the absence of diversity constraints, the model exhibits severe
latent collapse. The representations for steps $`z_{2}`$ through
$`z_{4}`$ share excessive similarity, effectively degenerating the
progressive chain into redundancy. This explains why the baseline model
fails to refine its attention, it is essentially stuck in the initial
state.

Margin-Based vs. Distance-Weighted Constraints. Applying the
Margin-Based Formulation
(Fig. [4](#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space")(b))
successfully reduces global similarity, but it indiscriminately
penalizes adjacent steps. This disrupts the natural pattern sharing
required for causal reasoning. In contrast, our Distance-Weighted
Diversity Loss
(Fig. [4](#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space")(c))
induces a structure consistent with our hypothesis: adjacent steps
maintain moderate similarity to preserve context, while distant steps
(e.g., $`z_{1}`$ vs. $`z_{4}`$) are forced to be distinct. This confirms
that ProLaViT learns a non-redundant, progressive derivation path where
steps with larger causal distance contribute unique perceptual
information.

### 4.4 Ablation Study

To validate the effectiveness of our core design choices, we conduct
comprehensive ablation studies focusing on two key aspects: the
necessity of progressive decomposition and the impact of the
distance-weighted diversity constraint.

Effect of Progressive Decomposition. We first validate the structural
advantage of our multi-step derivation over monolithic prediction. As
shown in
Tab. [2](#S4.T2 "Table 2 ‣ 4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
the One-step Latent Prediction baseline, which compresses all visual
cues into a single latent state, suffers from a significant performance
bottleneck. In contrast, ProLaViT’s progressive approach yields
substantial gains, particularly on tasks requiring logical depth such as
ChartQA (+15.97%) and BLINK-Jigsaw (+10.00%). This empirical evidence
confirms that complex visual reasoning overloads the capacity of a
single latent state, whereas our structured chain (Locate $`\to`$ Focus
$`\to`$ Isolate) effectively distributes the cognitive load, allowing
for incremental and precise visual grounding.

Effect of Distance-Weighted Diversity Loss. Next, we isolate the impact
of our diversity constraints. We first evaluate the Margin-Based
Formulation ($`\mathcal{L}_{\text{div}}^{\text{margin}}`$) without
distance weighting. While this formulation mitigates latent collapse by
penalizing similarity above the threshold $`\tau`$, it treats all
reasoning step pairs equally. As observed in
Tab. [3](#S4.T3 "Table 3 ‣ 4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
this rigid constraint leads to a performance drop on BLINK-Jigsaw
(-4.00%), likely because it disrupts the natural feature sharing (e.g.,
edges, textures) between adjacent steps. In contrast, our full
Distance-Weighted Diversity Loss ($`\mathcal{L}_{\text{div}}`$) achieves
the best overall performance. By incorporating the causal distance
weight $`w_{ij}`$, it respects the hierarchical structure of the
reasoning chain, it assigns lower penalty weights to adjacent steps to
preserve context, while enforcing that steps with larger causal distance
are more distinct. This design effectively prevents representation
collapse while maintaining semantic continuity, as evidenced by the
robust improvements on ChartQA (+2.81%) and BLINK-Jigsaw (+4.67%).

| Structure              | VisPuzzle | ChartQA | BLINK-J | CV-Bench | Avg.  |
|------------------------|-----------|---------|---------|----------|-------|
| One-step Latent Pred.  | 72.00     | 63.02   | 66.00   | 76.10    | 69.28 |
| ProLaViT (Progressive) | 74.25     | 78.99   | 76.00   | 77.66    | 76.73 |
| Improvement            | +2.25     | +15.97  | +10.00  | +1.56    | +7.45 |

Table 2: Ablation on reasoning structure: One-step Prediction vs.
Progressive Latent Visual Thought (Ours). The progressive decomposition
yields consistent gains, verifying that complex visual reasoning
requires multi-step derivation.

| Configuration | VisPuzzle | VStar | ChartQA | BLINK-J | Avg. |
|----|----|----|----|----|----|
| ProLaViT (No Diversity Loss) | 74.00 | 79.05 | 76.18 | 71.33 | 75.14 |
| + $`\mathcal{L}_{\text{div}}^{\text{margin}}`$ | 75.25 | 79.05 | 75.66 | 67.33 | 74.32 |
| + $`\mathcal{L}_{\text{div}}`$ (Full) | 74.25 | 80.11 | 78.99 | 76.00 | 77.34 |

Table 3: Ablation on diversity constraints.
$`\mathcal{L}_{\text{div}}^{\text{margin}}`$: Margin-Based Formulation
(without distance weights). $`\mathcal{L}_{\text{div}}`$:
Distance-Weighted Diversity Loss (Ours).

Latent Reasoning vs. Text-Trace Supervision. To disentangle the benefit
of latent-space reasoning from the programmatic training data, we
convert the same synthesized trajectories into textual traces (Crop
\[x1,y1,x2,y2\]$`\to`$Seg obj) and fine-tune the base model.
Text-Trajectory SFT gains only $`+0.9\%`$ over CoT SFT (73.41 vs. 72.54
avg.), while ProLaViT gains $`+4.8\%`$ (77.34 avg.). The net $`+3.9\%`$
arises purely from reasoning in continuous latent space, since
text-tokenized coordinates discard sub-pixel appearance cues that
ProLaViT’s latent chain preserves.

Effect of Endogenous vs. Exogenous Distillation. Finally, we examine the
necessity of our Endogenous Self-Distillation mechanism. A common
alternative is to use external vision models to supervise the latent
thoughts. To test this, we replace our native teacher (the Qwen2.5-VL
vision encoder) with two representative external models: (1)
DINOv2-Large \[[21](#bib.bib13)\] (ViT-L/14), a leading discriminative
model; (2) SDXL-VAE \[[22](#bib.bib38)\], a generative autoencoder used
in diffusion models. As shown in
Table [4](#S4.T4 "Table 4 ‣ 4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
our endogenous approach consistently outperforms both external variants.
While DINOv2 performs well on basic perception tasks, it falls behind on
fine-grained benchmarks like VStar. This suggests that aligning latent
thoughts with a different feature space creates an alignment gap,
whereas the native encoder offers naturally aligned supervision.
Notably, SDXL-VAE shows a significant performance drop on VisPuzzle,
despite good results on MMVP. This highlights a key limitation: VAE
latent spaces are optimized for reconstructing pixels and textures,
often sacrificing precise spatial logic. Consequently, structural
details are lost during reasoning. These results confirm that Endogenous
Self-Distillation is not only more efficient but also provides the most
robust supervision for visual reasoning.

| Teacher Model         | Type           | MMVP  | VisPuzzle | VStar | BLINK-J | Avg.  |
|-----------------------|----------------|-------|-----------|-------|---------|-------|
| DINOv2 (ViT-L/14)     | Discriminative | 78.00 | 73.75     | 78.53 | 72.00   | 75.57 |
| SDXL-VAE              | Generative     | 78.66 | 45.50     | 75.91 | 72.00   | 68.02 |
| ProLaViT (Native ViT) | Endogenous     | 79.00 | 74.25     | 80.11 | 76.00   | 77.34 |

Table 4: Ablation on teacher selection: Endogenous (Native Encoder) vs.
Exogenous (External Experts). While external experts perform adequately
on basic perception (MMVP), they struggle with complex spatial reasoning
(VisPuzzle), confirming the superiority of endogenous alignment.

| Ordering        | VisPuzzle | VStar | ChartQA | BLINK-J | CV-Bench | Avg.  |
|-----------------|-----------|-------|---------|---------|----------|-------|
| Canonical Order | 74.25     | 80.11 | 78.99   | 76.00   | 77.66    | 77.40 |
| Random Order    | 68.00     | 74.50 | 73.50   | 70.66   | 69.14    | 71.16 |
| Drop            | -6.25     | -5.61 | -5.49   | -5.34   | -8.52    | -6.24 |

Table 5: Ablation on reasoning step ordering. Random permutation
validates the importance of the canonical coarse-to-fine progression.

Effect of Reasoning Step Ordering. To validate that the causal ordering
of reasoning steps is critical, we randomly permute the order of the
four steps while keeping all other components fixed. As shown in
Tab. [5](#S4.T5 "Table 5 ‣ 4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
the Canonical Order ($`z_{1}\to z_{2}\to z_{3}\to z_{4}`$) achieves
superior performance (77.40% average), while Random Order suffers a
substantial drop (-6.24% average), particularly on spatial tasks like
VisPuzzle (-6.25%) and CV-Bench (-8.52%). These findings provide strong
evidence that the progressive causal structure is a fundamental
requirement for effective visual reasoning.

Efficiency: Latent Reasoning vs. Explicit Tool-Use. A natural question
is whether latent reasoning is preferable to explicitly invoking visual
tools at each step. We fine-tune Qwen2.5-VL on identical traces with
explicit tool tokens that invoke real operations at inference. As shown
in
Tab. [6](#S4.T6 "Table 6 ‣ 4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
explicit tool-use re-invokes the ViT encoder every step, inducing
$`3.4\times`$ latency and $`+117\%`$ TFLOPs. In contrast, ProLaViT
performs a single ViT pass and reasons entirely in latent space
($`1.21\times`$ latency, $`+12\%`$ TFLOPs) while achieving higher
accuracy (VStar 80.11 vs. 79.60, BLINK-J 76.00 vs. 74.00), since latent
reasoning avoids the feature loss from iterative image re-encoding.
Training requires only $`{\sim}`$28 GPU-hours on 2$`\times`$H20,
compared to over 120h for generative-CoT methods \[[9](#bib.bib9)\].

| Method                | VStar | BLINK-J | Latency        | TFLOPs |
|-----------------------|-------|---------|----------------|--------|
| Qwen2.5-VL (Base)     | 76.44 | 59.33   | 1.00$`\times`$ | 1.00   |
| Explicit Tool-Use SFT | 79.60 | 74.00   | 3.40$`\times`$ | 2.17   |
| ProLaViT (Ours)       | 80.11 | 76.00   | 1.21$`\times`$ | 1.12   |

Table 6: Explicit Tool-Use vs. Latent Distillation.

## 5 Conclusion

In this paper, we presented ProLaViT, a novel framework empowering
Multimodal Large Language Models to perform visual reasoning via
progressive latent derivation. By transitioning reasoning from discrete
text to a structured continuous latent space, ProLaViT effectively
bridges the modality gap that limits traditional Chain-of-Thought
approaches. Central to our contribution is the Endogenous
Self-Distillation mechanism, which enables the model to internalize
rigorous algorithmic logic from programmatically synthesized data,
thereby eliminating reliance on computationally expensive external
vision experts. Furthermore, our proposed Distance-Weighted Diversity
Loss mitigates the critical issue of latent collapse, enforcing
topological distinctiveness across the reasoning chain to ensure robust,
non-redundant visual thoughts. Extensive experiments on
perception-intensive benchmarks, including visual search and jigsaw
assembly, demonstrate that ProLaViT achieves state-of-the-art
performance with superior inference efficiency. While this work focuses
on spatial and logical reasoning in static images, the principle of
structured latent derivation holds significant potential for broader
domains. Future work will extend ProLaViT to temporal video reasoning
and complex geometric transformations, paving the way for more
generalizable and interpretable multimodal intelligence.

## References

- \[1\] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C.
  Zhou, and J. Zhou (2023) Qwen-vl: a versatile vision-language model
  for understanding, localization, text reading, and beyond. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[2\] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, and S. Song (2025)
  Qwen2.5-vl technical report. External Links: 2502.13923 Cited by:
  [§A](#S1a.p1.1 "A Implementation Details ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[3\] E. Chern, J. Su, Y. Ma, and P. Liu (2024) Anole: an open,
  autoregressive, native large multimodal models for interleaved
  image-text generation. arXiv preprint arXiv:2407.06135. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[4\] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W.
  Yu, X. Nie, Z. Song, et al. (2025) Emerging properties in unified
  multimodal pretraining. arXiv preprint arXiv:2505.14683. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[5\] R. Dong, C. Han, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J.
  Sun, H. Zhou, H. Wei, X. Kong, X. Zhang, K. Ma, and L. Yi (2024)
  DreamLLM: synergistic multimodal comprehension and creation. In The
  Twelfth International Conference on Learning Representations, Cited
  by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[6\] S. Dong, S. Wang, X. Liu, C. Li, H. Hou, and Z. Wei (2025)
  Interleaved latent visual reasoning with selective perceptual
  modeling. arXiv preprint arXiv:2512.05665. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[7\] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A.
  Smith, W. Ma, and R. Krishna (2024) BLINK: multimodal large language
  models can see but not perceive. ArXiv abs/2404.12390. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[8\] Y. Ge, S. Zhao, Z. Zeng, Y. Ge, C. Li, X. Wang, and Y.
  Shan (2023) Making llama see and draw with seed tokenizer. arXiv
  preprint arXiv:2310.01218. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[9\] J. Gu, Y. Hao, H. W. Wang, L. Li, M. Q. Shieh, Y. Choi, R.
  Krishna, and Y. Cheng (2025) ThinkMorph: emergent properties in
  multimodal interleaved chain-of-thought reasoning. External Links:
  2510.27492 Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.4](#S4.SS4.p7.1 "4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[10\] S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. E. Weston,
  and Y. Tian (2024) Training large language models to reason in a
  continuous latent space. ArXiv abs/2412.06769. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.3](#S2.SS3.p1.1 "2.3 Latent Visual Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[11\] Y. Hao, J. Gu, H. W. Wang, L. Li, Z. Yang, L. Wang, and Y.
  Cheng (2025) Can mllms reason in multimodality? emma: an enhanced
  multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444. Cited
  by:
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[12\] D. Jiang, R. Zhang, Z. Guo, Y. Li, Y. Qi, X. Chen, L. Wang, J.
  Jin, C. Guo, S. Yan, et al. (2025) Mme-cot: benchmarking
  chain-of-thought in large multimodal models for reasoning quality,
  robustness, and efficiency. arXiv preprint arXiv:2502.09621. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[13\] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L.
  Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W. Lo, P. Dollár,
  and R. B. Girshick (2023) Segment anything. 2023 IEEE/CVF
  International Conference on Computer Vision (ICCV), pp. 3992–4003.
  Cited by:
  [§1](#S1.p4.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[14\] J. Y. Koh, D. Fried, and R. Salakhutdinov (2023) Generating
  images with multimodal language models. External Links: 2305.17216
  Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[15\] A. Li, C. Wang, D. Fu, K. Yue, Z. Cai, W. B. Zhu, O. Liu, P.
  Guo, W. Neiswanger, F. Huang, et al. (2025) Zebra-cot: a dataset for
  interleaved vision language reasoning. arXiv preprint
  arXiv:2507.16746. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[16\] B. Li, X. Sun, J. Liu, Z. Wang, J. Wu, X. Yu, H. Chen, E.
  Barsoum, M. Chen, and Z. Liu (2025) Latent visual reasoning. ArXiv
  abs/2509.24251. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.3](#S2.SS3.p1.1 "2.3 Latent Visual Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [5th
  item](#S4.I1.i5.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[17\] H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023) Visual instruction
  tuning. ArXiv abs/2304.08485. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[18\] P. Lu, B. Peng, H. Cheng, M. Galley, K. Chang, Y. N. Wu, S.
  Zhu, and J. Gao (2023) Chameleon: plug-and-play compositional
  reasoning with large language models. ArXiv abs/2304.09842. Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[19\] A. Masry, D. X. Long, J. Q. Tan, S. R. Joty, and E.
  Hoque (2022) ChartQA: a benchmark for question answering about charts
  with visual and logical reasoning. ArXiv abs/2203.10244. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[20\] OpenAI, J. Achiam, S. Adler, S. Agarwal, and L. Ahmad (2024)
  GPT-4 technical report. External Links: 2303.08774 Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[21\] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V.
  Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M.
  Assran, N. Ballas, W. Galuba, R. Howes, P. (. Huang, S. Li, I.
  Misra, M. G. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J.
  Mairal, P. Labatut, A. Joulin, and P. Bojanowski (2023) DINOv2:
  learning robust visual features without supervision. ArXiv
  abs/2304.07193. Cited by:
  [§1](#S1.p4.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.4](#S4.SS4.p5.1 "4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[22\] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J.
  Müller, J. Penna, and R. Rombach (2023) SDXL: improving latent
  diffusion models for high-resolution image synthesis. External Links:
  2307.01952 Cited by:
  [§4.4](#S4.SS4.p5.1 "4.4 Ablation Study ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[23\] Y. Qin, B. Wei, J. Ge, K. Kallidromitis, S. Fu, T. Darrell,
  and X. Wang (2025) Chain-of-visual-thought: teaching vlms to see and
  think better with continuous visual tokens. External Links: 2511.19418
  Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [4th
  item](#S4.I1.i4.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[24\] H. Shao, S. Qian, H. Xiao, G. Song, Z. Zong, L. Wang, Y. Liu,
  and H. Li (2024) Visual cot: advancing multi-modal language models
  with a comprehensive dataset and benchmark for chain-of-thought
  reasoning. Advances in Neural Information Processing Systems 37. Cited
  by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[25\] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang (2023)
  HuggingGPT: solving ai tasks with chatgpt and its friends in hugging
  face. External Links: 2303.17580 Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[26\] Z. Shen, H. Yan, L. Zhang, Z. Hu, Y. Du, and Y. He (2025) Codi:
  compressing chain-of-thought into continuous space via
  self-distillation. arXiv preprint arXiv:2502.21074. Cited by:
  [§2.3](#S2.SS3.p1.1 "2.3 Latent Visual Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[27\] D. Surís, S. Menon, and C. Vondrick (2023) ViperGPT: visual
  inference via python execution for reasoning. External Links:
  2303.08128 Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[28\] C. Team (2025) CVBench: a benchmark for cross-video multimodal
  reasoning. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[29\] S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, and S. Xie (2024)
  Eyes wide shut? exploring the visual shortcomings of multimodal llms.
  2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition
  (CVPR), pp. 9568–9578. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[30\] Q. Wang, Y. Shi, Y. Wang, Y. Zhang, P. Wan, K. Gai, X. Ying,
  and Y. Wang (2025) Monet: reasoning in latent visual space beyond
  images and language. arXiv preprint arXiv:2511.21395. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[31\] J. Wei, X. Wang, D. Schuurmans, M. Bosma, E. H. Chi, F. Xia, Q.
  Le, and D. Zhou (2022) Chain of thought prompting elicits reasoning in
  large language models. ArXiv abs/2201.11903. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[32\] C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan (2023)
  Visual chatgpt: talking, drawing and editing with visual foundation
  models. External Links: 2303.04671 Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[33\] P. Wu and S. Xie (2023) V\*: guided visual search as a core
  mechanism in multimodal llms. 2024 IEEE/CVF Conference on Computer
  Vision and Pattern Recognition (CVPR), pp. 13084–13094. Cited by:
  [§4.1](#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiment ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[34\] G. Xu, P. Jin, Z. Wu, H. Li, Y. Song, L. Sun, and L.
  Yuan (2025) LLaVA-cot: let vision language models reason step-by-step.
  In Proceedings of the IEEE/CVF International Conference on Computer
  Vision (ICCV), pp. 2087–2098. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[35\] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao (2024)
  Depth anything: unleashing the power of large-scale unlabeled data.
  2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition
  (CVPR), pp. 10371–10381. Cited by:
  [§1](#S1.p4.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[36\] Z. Yang, X. Yu, D. Chen, M. Shen, and C. Gan (2025) Machine
  mental imagery: empower multimodal reasoning with latent visual
  tokens. ArXiv abs/2506.17218. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.3](#S2.SS3.p1.1 "2.3 Latent Visual Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[37\] Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z.
  Liu, C. Liu, M. Zeng, and L. Wang (2023) MM-react: prompting chatgpt
  for multimodal reasoning and action. External Links: 2303.11381 Cited
  by:
  [§2.2](#S2.SS2.p1.1 "2.2 Visual-Augmented Reasoning with External Tools ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").
- \[38\] Z. Zhang, A. Zhang, M. Li, H. Zhao, G. Karypis, and A.
  Smola (2024) Multimodal chain-of-thought reasoning in language models.
  External Links: 2302.00923 Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space"),
  [§2.1](#S2.SS1.p1.1 "2.1 Multimodal Chain-of-Thought Reasoning ‣ 2 Related Work ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").

ProLaViT: Learning Progressive Latent Visual
Thoughts in Structured Latent Space
Supplementary Material

## A Implementation Details

Base Model and Parameter-Efficient Tuning. We adopt
Qwen2.5-VL-7B-Instruct \[[2](#bib.bib21)\] as our base model. To enable
efficient adaptation, we apply Low-Rank Adaptation (LoRA) with a rank of
16, a scaling factor $`\alpha=32`$, and a dropout rate of 0.01. LoRA is
applied to all transformer layers within the language model backbone.
However, we explicitly exclude the embedding layers and visual branch
modules from LoRA updates to preserve their pre-trained representations.
During training, both the vision encoder and the language model backbone
remain frozen, only the LoRA adapters and a lightweight projection layer
are updated.

Visual Anchor Configuration. In our main experiments, we utilize the
native vision encoder of Qwen2.5-VL as the sole source for visual
anchors. This encoder produces compact anchor token sequences that serve
as visual chain-of-thought checkpoints during inference. To encourage
the generation of diverse and non-redundant visual latent states across
reasoning steps, we impose a margin-based diversity loss on the anchor
token representations with a weight of $`\lambda_{\text{div}}=0.2`$ and
a margin of $`\delta=0.8`$.

Two-Round Training Pipeline. We employ a two-round curriculum training
strategy:

- •
  Round 1 (6000 steps). Starting from the pre-trained
  Qwen2.5-VL-7B-Instruct, we progressively train the model through three
  internal stages:

  - –
    Stage 0 – VQA Warm-up (steps 0–1500). Anchor pad tokens are appended
    to the input to familiarize the model with the anchor token
    vocabulary, trained via a standard VQA cross-entropy objective.
  - –
    Stage 1 – Feature Alignment (steps 1500–4000). The model is trained
    to explicitly generate the anchor token sequence, supervised by
    reconstructed visual features from the anchor encoder.
  - –
    Stage 2 – CoT Training (steps 4000–6000). The model learns to emit
    anchor tokens in a \<think\>...\</think\>\<answer\>...\</answer\>
    format, interleaving visual latent reasoning with the final textual
    response.

  Upon completion of Round 1, the LoRA weights are merged back into the
  backbone to produce an intermediate checkpoint.
- •
  Round 2 (10000 steps). Building upon the merged Round-1 model, we
  conduct a joint training phase with restructured stage boundaries
  (Stage 1: steps 0–3000; Stage 2: steps 3000–6000; VQA-Only: steps
  6000–8000; Extended: steps 8000–10000). In the subsequent VQA-Only
  sub-stage, the model randomly alternates between the anchor-augmented
  CoT format and the plain VQA format to enhance generalization
  robustness. Finally, the LoRA weights are merged to yield the
  production model.

Optimization Hyperparameters. All experiments are conducted on
$`2\times`$ NVIDIA H20 GPUs utilizing DeepSpeed ZeRO-3 parallelism. We
employ the AdamW optimizer with a cosine learning rate schedule, a
warm-up ratio of 0.05, and a weight decay of 0.1. The primary learning
rate is set to $`5\times 10^{-5}`$, while a reduced rate of
$`1\times 10^{-5}`$ is applied to the visual projection layer. The
per-device batch size is set to 2, with gradient accumulation steps
computed automatically to match the target global batch size. Detailed
hyperparameters are summarized in
Tab. [7](#S1.T7 "Table 7 ‣ A Implementation Details ‣ ProLaViT: Learning Progressive Latent Visual Thoughts in Structured Latent Space").

|  |  |  |
|----|----|----|
| Hyperparameter | Round 1 | Round 2 |
| Maximum Steps | 6000 | 10000 |
| Learning Rate | $`5\times 10^{-5}`$ |  |
| Projection LR | $`1\times 10^{-5}`$ |  |
| Weight Decay | 0.1 |  |
| Warm-up Ratio | 0.05 |  |
| LR Schedule | Cosine |  |
| LoRA Rank / $`\alpha`$ | 16 / 32 |  |
| LoRA Dropout | 0.01 |  |
| Diversity Loss Weight $`\lambda_{\text{div}}`$ | 0.2 |  |
| Diversity Margin $`\delta`$ | 0.8 |  |
| Per-device Batch Size | 2 |  |
| Parallelism | DeepSpeed ZeRO-3 |  |
| GPUs | 2 $`\times`$ NVIDIA H20 |  |

Table 7: Hyperparameters for the two-round training pipeline.

## B Training Dataset Construction

### B.1 Overview.

To enable the model to internalize algorithmic visual reasoning without
reliance on external tools at inference time, we construct a scalable
programmatic synthesis pipeline that converts single-image VQA samples
into multi-view, multi-operation Chain-of-Thought (CoT) training
instances. Our training data is drawn from three task domains: Chart
Refocus, Jigsaw Assembly, and Visual Search.

### B.2 Step 1: Structured CoT Generation.

Each raw sample consists of a single input image, a natural language
question, and a weakly annotated answer. To produce operation-grounded
reasoning chains, we prompt a large multimodal model (Gemini) to analyze
each sample and output a structured multi-modal CoT that specifies: (i)
the key visual regions relevant to each reasoning step (e.g., chart
sub-regions, puzzle pieces, or target objects); (ii) a sequence of
candidate visual operations to be applied to those regions; and (iii)
the natural language rationale linking each operation to the
corresponding reasoning conclusion.

The supported operations include region cropping, bounding-box
annotation, semi-transparent highlighting, spatial grid overlay,
directional arrow and path drawing, and text labeling. For Jigsaw
Assembly, domain-specific operations such as edge highlighting and
counterfactual wrong-assembly illustration are additionally included.
Operations are further annotated with spatial extents (expressed as
either absolute pixel coordinates or relative image percentages), color
specifications, and sequential dependencies that define the order in
which operations should be applied.

### B.3 Step 2: Programmatic Image Synthesis.

The structured descriptions from Step 1 are used as input to a
deterministic rendering module that translates textual operation
specifications into concrete image transformations. The module first
parses the operation sequence, resolving spatial coordinates and
dependency relationships to determine whether each operation should be
applied to the original image or to the output of a preceding operation.
It then executes each operation in order, compositing the results to
produce a chain of intermediate images. Operations that are independent
of one another are applied in parallel to the original image, while
chained operations (e.g., applying a grid overlay followed by regional
highlighting) are executed sequentially, with each step taking the
previous result as its input.

For each training sample, this process yields a set of reasoning
images—up to four per instance—comprising the original problem image and
one or more intermediate transformed views. These images serve as
explicit visual checkpoints that anchor each reasoning step to a
concrete perceptual state, forming the multi-view input consumed by our
model during training.

### B.4 Step 3: Metadata Alignment and Quality Filtering.

After image synthesis, each training instance is assembled with the
following components: the problem image and its associated reasoning
images, the natural language question, the complete multi-step reasoning
chain produced in Step 1, and the final answer. We apply quality
filtering to remove samples in which the structured CoT generation fails
to produce valid output, no executable visual operation can be
identified, or one or more of the synthesized images is missing or
corrupted. The retained samples from all three task domains are merged
into a unified training set.

### B.5 Task Diversity.

The three task domains contribute complementary reasoning patterns.
Chart Refocus requires precise quantitative reading and region-level
re-grounding, exercising operations that localize and annotate specific
chart areas. Jigsaw Assembly demands structural spatial reasoning and
counterfactual analysis of part configurations, relying on operations
that expose assembly errors and directional cues. Visual Search focuses
on target–distractor discrimination in cluttered scenes, utilizing
operations that highlight salient regions and trace spatial search
paths. Training on this diverse mixture exposes the model to a wide
spectrum of visual operation types and reasoning styles, promoting
generalization of the learned visual chain-of-thought mechanism across
tasks.
````
