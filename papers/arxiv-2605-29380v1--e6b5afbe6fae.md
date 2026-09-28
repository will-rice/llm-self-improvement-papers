---
identifier: arxiv:2605.29380v1
title: "TRACER: Persistent Regularization for Robust Multimodal Finetuning"
authors:
  - Hesam Asadollahzadeh
  - Feng Liu
  - Christopher Leckie
  - Sarah M. Erfani
published: "2026-05-28T05:34:23+00:00"
url: https://arxiv.org/abs/2605.29380v1
source: arxiv
doi: null
arxiv_id: 2605.29380v1
categories:
  - cs.AI
  - cs.CV
  - cs.LG
---

# TRACER: Persistent Regularization for Robust Multimodal Finetuning

Hesam Asadollahzadeh Affiliation: School of Computing and Information
Systems (CIS), Faculty of Engineering and IT (FEIT), University of
Melbourne, Australia Correspondence to:
<h.asadollahzadeh@unimelb.edu.au>    Feng Liu Affiliation: School of
Computing and Information Systems (CIS), Faculty of Engineering and IT
(FEIT), University of Melbourne, Australia    Christopher Leckie
Affiliation: School of Computing and Information Systems (CIS), Faculty
of Engineering and IT (FEIT), University of Melbourne, Australia   
Sarah M. Erfani Affiliation: School of Computing and Information Systems
(CIS), Faculty of Engineering and IT (FEIT), University of Melbourne,
Australia

###### Abstract

Mainstream strategies for finetuning pretrained multimodal models often
degrade out-of-distribution (OOD) robustness, a phenomenon known as
catastrophic forgetting. In this paper, we develop a theoretical
framework for multimodal contrastive finetuning, yielding closed-form
solutions and a geometric decomposition for each strategy. This
framework shows that self-distillation is more effective than other
regularization approaches to retain the knowledge of the pretrained
model. Our analysis reveals a largely overlooked limitation: standard
Exponential Moving Average (EMA) teachers, widely used in robust
finetuning, suffer from collapse. To solve this, we prove that a
Weighted Moving Average (WMA) teacher maintains a persistent
regularizing force over finite horizons and yields bias-free convergence
in the task subspace while preserving orthogonal knowledge. These
insights motivate TRACER (Trajectory-Robust Anchoring for Contrastive
Encoder Regularization), which combines contrastive learning with
WMA-guided multi-perspective distillation. Extensive experiments on CLIP
finetuning demonstrate consistent OOD accuracy and calibration gains
across three backbone architectures, and comprehensive ablations confirm
that TRACER is both principled and robust to hyperparameter choices.
Code is available at
[https://github.com/HesamAsad/TRACER](https://github.com/HesamAsad/TRACER).

###### Keywords: 

Multimodal contrastive learning, finetuning, robustness

## 1 Introduction

Pretrained models such as CLIP ([Radford et al., 2021](#bib.bib12)) have
revolutionized machine learning through their remarkable zero-shot
transfer and adaptive capabilities. These models derive their robustness
from large-scale multimodal pretraining ([Fang et al.,
2022](#bib.bib91); [Xu et al., 2024b](#bib.bib92)), enabling diverse
applications from visual recognition ([Shen et al., 2022b](#bib.bib93);
[Zhang et al., 2022b](#bib.bib94)) to generative modeling ([Betker et
al., 2023](#bib.bib95); [Pi et al., 2024](#bib.bib96)) and serving as
backbones for large multimodal models ([Alayrac et al.,
2022](#bib.bib97); [Liu et al., 2023](#bib.bib98); [Zhu et al.,
2024](#bib.bib99)).

Despite these successes, adapting these pretrained models to downstream
tasks via finetuning presents a fundamental challenge: while finetuning
improves in-distribution (ID) performance, it often degrades
out-of-distribution (OOD) robustness ([Radford et al.,
2021](#bib.bib12)). This trade-off manifests as catastrophic forgetting
of pretrained knowledge ([Wortsman et al., 2022b](#bib.bib20)), where
models sacrifice their general-purpose representations to optimize for
task-specific patterns, potentially overfitting to spurious correlations
in the finetuning data.

Several empirical strategies have emerged to mitigate this trade-off.
For example, LP-FT ([Kumar et al., 2022](#bib.bib64)) addresses the
problem of randomly initialized heads distorting pretrained features by
first learning a linear probe on frozen features before full finetuning.
FLYP ([Goyal et al., 2023](#bib.bib28)) extends this idea by reusing
CLIP’s pretrained text encoder as the classification head, maintaining
consistency with the pretraining objective. Post-hoc methods like
WiSE-FT ([Wortsman et al., 2022b](#bib.bib20)) and Model Stock ([Jang et
al., 2024](#bib.bib71)) perform weight averaging between pretrained and
finetuned models to recover lost robustness. Regularization-based
approaches, including $`L_{2}`$–SP ([Li et al., 2018](#bib.bib16)) and
self-distillation with dynamic teachers ([Oh et al., 2024](#bib.bib17)),
introduce constraints to preserve pretrained knowledge. However, most
dynamic-teacher methods rely on an Exponential Moving Average (EMA),
whose regularizing influence provably weakens as the teacher approaches
the student. This limitation is often overlooked in the
robust-finetuning literature on catastrophic forgetting, yet it is
precisely the reason why the OOD robustness is most fragile. Crucially,
robust finetuning depends on maintaining sufficient regularization
strength throughout training; when that strength decays, OOD robustness
erodes even as ID accuracy improves. This motivates _trajectory
regularization_: keeping the teacher anchored to the optimization path
so that it continues to exert a meaningful restoring force.

Moreover, despite the proliferation of these methods, a theoretical
understanding of _what_ changes during contrastive finetuning and
_where_ forgetting occurs remains elusive. We address this gap by
developing a theoretical framework that reveals the geometric structure
of how different finetuning strategies modify pretrained
representations. We find that the linearized contrastive finetuning
objective can be reformulated as a matrix least-squares problem through
what we call the _contrastive target matrix_. This reformulation enables
closed-form solutions for common finetuning strategies, exposing their
fundamentally different geometric behaviors.

Our theoretical insights lead to the design of TRACER (Trajectory-Robust
Anchoring for Contrastive Encoder Regularization), a practical
finetuning method that implements our geometric principles. As shown in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
TRACER combines contrastive learning with dynamic self-distillation,
yielding strong results on ImageNet and its distribution shifts. Across
multiple CLIP architectures, TRACER consistently improves the ID-OOD
trade-off. We validate these findings through extensive ablation studies
spanning distillation components, regularization strength, teacher
update schedules, and kernel shape, demonstrating both the robustness of
our method to hyperparameter choices and contributions of each design
element.

![Refer to caption](2605.29380v1/figures/figure_v6.jpg)

Figure 1: Overview of TRACER. The base contrastive objective is combined
with a dynamic self-distillation loss from a Weighted Moving Average
(WMA) teacher to preserve orthogonal pretrained knowledge while
adaptively mixing within the task subspace. $`\theta^{0}_{\text{CLIP}}`$
represents the initial pretrained CLIP model. $`\theta^{t}`$ denotes the
student model at time $`t`$, with its image and text encoder
($`\mathcal{E}_{\text{Image}}`$ and $`\mathcal{E}_{\text{Text}}`$) being
trained. The student receives gradient updates from
$`\mathcal{L}_{\text{MMCL}}`$. The WMA teacher model $`\psi^{t}`$ is
updated from the student’s parameters. The teacher then provides a
teaching signal $`\mathcal{L}_{\text{SD-WMA}}`$ to regularize the
student. This interplay allows TRACER to adapt to new tasks while
preserving pretrained knowledge. The complete training procedure is
detailed in
Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix D TRACER Algorithm ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

In summary, our work makes the following main contributions: (i) We
introduce the _contrastive target matrix_ reformulation of linearized
contrastive loss, turning the objective into a least-squares problem and
enabling closed-form solutions for standard finetuning and
regularization strategies; (ii) We derive a geometric decomposition that
separates task-subspace mixing from orthogonal preservation, explaining
when and where forgetting occurs and providing a principled basis for
dynamic teachers; (iii) We identify a largely overlooked limitation of
standard teachers in robust finetuning, the inherent collapse of the EMA
teacher–student learning signal, and show how _trajectory
regularization_ with a weighted moving-average (WMA) teacher preserves a
meaningful regularization signal over finite horizons, enabling
bias-free task-subspace convergence; we instantiate these principles in
TRACER with consistent OOD gains on CLIP finetuning, supported by
comprehensive ablations across four axes
(§[B](#A2 "Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).

## 2 Related Work

Contrastive language–image pretraining ([Radford et al.,
2021](#bib.bib12); [Jia et al., 2021](#bib.bib53); [Ilharco et al.,
2021](#bib.bib82); [Zhai et al., 2023](#bib.bib57)) enables strong
zero-shot transfer but naive finetuning can harm OOD robustness ([Taori
et al., 2020](#bib.bib63); [Wortsman et al., 2022b](#bib.bib20)). Robust
finetuning explores weight interpolation/averaging ([Wortsman et al.,
2022b](#bib.bib20); [Wortsman et al., 2022a](#bib.bib52); [Jang et al.,
2024](#bib.bib71)), weight- or output-space regularization ([Li et al.,
2018](#bib.bib16); [Li and Hoiem, 2018](#bib.bib35)), and contrastive
variants aligned to text prompts or energies ([Goyal et al.,
2023](#bib.bib28); [Mao et al., 2024](#bib.bib75); [Nam et al.,
2024](#bib.bib77); [Shu et al., 2023](#bib.bib76)). CaRot ([Oh et al.,
2024](#bib.bib17)) couples contrastive training with new regularizers to
jointly improve OOD accuracy and calibration.

Self-distillation and dynamic teachers stabilize learning and preserve
knowledge ([Hinton et al., 2015](#bib.bib2); [Zhang et al.,
2019](#bib.bib10); [Mobahi et al., 2020](#bib.bib5); [Laine and Aila,
2017](#bib.bib47); [Tarvainen and Valpola, 2017](#bib.bib48)).
Momentum/EMA teachers are effective yet can introduce persistent bias
toward initialization, and their teacher–student gap collapses as
training converges, reducing regularization exactly when OOD robustness
is most vulnerable. This critical flaw is rarely made explicit in the
robust finetuning literature. Our _WMA_ teacher generalizes EMA by
weighting the entire trajectory on normalized time, enabling
endpoint-aware curricula (e.g., arcsine/Beta kernels) and _trajectory
regularization_ that preserves a meaningful teacher gap over finite
horizons. As we prove, this yields bias-free task-subspace convergence.
Our theory complements linearized analyses of supervised and contrastive
learning ([Ji et al., 2023](#bib.bib11); [Tian, 2022](#bib.bib13);
[Nakada et al., 2023](#bib.bib14); [Xue et al., 2024](#bib.bib15); [Hao
et al., 2025](#bib.bib105)) and explains forgetting via an explicit
geometric decomposition. An extended literature review appears in
§[A](#A1 "Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

## 3 Theoretical Analysis

To address the ID-OOD trade-off, where finetuning improves
in-distribution accuracy at the cost of out-of-distribution robustness,
we develop a theoretical framework that reveals the underlying dynamics
of this phenomenon.

### 3.1 Problem Setting and Preliminaries

Finetuning task. We consider robust finetuning of a pretrained
vision–language model on paired image–text data
$`\{(\mathbf{x}_{I}^{i},\mathbf{x}_{T}^{i})\}_{i=1}^{n}`$ drawn from a
downstream task. The goal is to adapt the model so that in-distribution
accuracy improves on this task while pretrained, broadly transferable
representations are preserved for out-of-distribution generalization.

Linearized image/text encoders. Following linearized analyses widely
used in the theory of contrastive learning ([Ji et al.,
2023](#bib.bib11); [Tian, 2022](#bib.bib13); [Nakada et al.,
2023](#bib.bib14); [Xue et al., 2024](#bib.bib15)), we model the image
and text encoders as linear projections,
$`g_{I}(\mathbf{x})=\mathbf{W}_{I}\mathbf{x}`$ and
$`g_{T}(\mathbf{x})=\mathbf{W}_{T}\mathbf{x}`$. The image encoder
$`\mathbf{W}_{I}`$ is adapted from a pretrained state
$`\mathbf{W}_{I}^{0}`$, while the text encoder $`\mathbf{W}_{T}`$ is
frozen at its pretrained state $`\mathbf{W}_{T}^{0}`$. We collect the
$`n`$ image and text features of a batch into matrices
$`\mathbf{X}_{I}\in\mathbb{R}^{d_{I}\times n}`$ and
$`\mathbf{X}_{T}\in\mathbb{R}^{d_{T}\times n}`$, where $`d_{I},d_{T}`$
are the input feature dimensions and $`p`$ is the shared embedding
dimension.

Original MMCL objective. The linearized multimodal contrastive learning
(MMCL) loss is a standard analytic surrogate of the symmetric InfoNCE
objective (the full derivation appears in
§[C.1](#A3.SS1 "C.1 Derivation of ℒ\_"CL" ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")):

|     |                                                                            |                                                                                               |     |     |
| --- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\mathcal{L}_{\text{MMCL}}(\mathbf{W}_{I},\mathbf{W}_{T})=`$ | $`\displaystyle\frac{1}{n(n-1)}\!\Big[\sum_{i\neq j}\!s_{ij}-(n{-}1)\!\sum_{i}\!s_{ii}\Big]`$ |     |     |
|     |                                                                            | $`\displaystyle+R(\mathbf{W}_{I},\mathbf{W}_{T}),`$                                           |     | (1) |

where
$`s_{ij}=(\mathbf{W}_{I}\mathbf{x}_{I}^{i})^{\top}(\mathbf{W}_{T}\mathbf{x}_{T}^{j})`$
is the image–text similarity, and $`R(\cdot)`$ is a cross-Frobenius
term. This loss balances pulling matched pairs together against pushing
unmatched pairs apart. As shown
in §[C.2](#A3.SS2 "C.2 Reformulation of the Least-Squares Objective ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
when $`\mathbf{W}_{T}=\mathbf{W}_{T}^{0}`$ is frozen,
optimizing equation [1](#S3.Ex1 "In 3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
over $`\mathbf{W}_{I}`$ is equivalent (up to constants and a
data-dependent quadratic term that arises naturally in the optimization)
to a matrix least-squares problem driven by the contrastive target
matrix introduced below.

Notation. We use $`\mathbf{I}_{n}\in\mathbb{R}^{n\times n}`$ for the
identity matrix and
$`\mathbf{J}_{n}=\mathbf{1}_{n}\mathbf{1}_{n}^{\top}\in\mathbb{R}^{n\times n}`$
for the all-ones matrix, where $`\mathbf{1}_{n}`$ denotes the
$`n`$-dimensional all-ones vector. The matrix
$`n\mathbf{I}_{n}-\mathbf{J}_{n}`$ acts as a centered contrastive
operator on text features: it preserves the matched (diagonal)
directions while subtracting the batch-mean direction, yielding a
per-column “attract-paired/repel-others” signal. We let
$`\mathcal{P}_{I}\coloneqq\mathbf{X}_{I}(\mathbf{X}_{I}^{\top}\mathbf{X}_{I})^{+}\mathbf{X}_{I}^{\top}`$
denote the orthogonal projector onto $`\mathrm{range}(\mathbf{X}_{I})`$,
the _task subspace_ spanned by the finetuning image features, and
$`\mathbf{I}-\mathcal{P}_{I}`$ the projector onto its orthogonal
complement. Throughout, $`(\cdot)^{+}`$ is the Moore–Penrose
pseudoinverse and $`\left\|\cdot\right\|_{\text{F}}`$ denotes the
Frobenius norm.

### 3.2 Loss Reformulation via the Contrastive Target Matrix

We now rewrite the MMCL
objective equation [1](#S3.Ex1 "In 3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
in a form that exposes closed-form solutions via a single algebraic
construct.

###### Definition 3.1 (Contrastive Target Matrix).

Given the frozen text encoder $`\mathbf{W}_{T}^{0}`$ and finetuning
texts $`\mathbf{X}_{T}`$, we define the _contrastive target matrix_ as

|     |     |     |
| --- | --- | --- |
|     |

````math
\mathbf{Y}_{\text{FT}}\;\coloneqq\;\mathbf{W}_{T}^{0}\mathbf{X}_{T}(n\mathbf{I}_{n}-\mathbf{J}_{n})\;\in\;\mathbb{R}^{p\times n}.
``` |  |

Each column $`\mathbf{y}_{i}`$ is constructed to attract the image
embedding $`\mathbf{x}_{I}^{i}`$ towards its paired text
$`\mathbf{x}_{T}^{i}`$ and repel it from the remaining texts in the
batch (detailed in
§[C.1](#A3.SS1 "C.1 Derivation of ℒ_"CL" ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).

What is $`\mathbf{Y}_{\text{FT}}`$, and why is it “fixed”? The matrix
$`\mathbf{Y}_{\text{FT}}`$ depends only on the frozen text encoder
$`\mathbf{W}_{T}^{0}`$ and the finetuning text features
$`\mathbf{X}_{T}`$; it does *not* depend on the trainable image weights
$`\mathbf{W}_{I}`$. Consequently, $`\mathbf{Y}_{\text{FT}}`$ stays
*fixed throughout finetuning* and plays exactly the role of a target in
a supervised regression problem: it is the centered contrastive signal
that $`\mathbf{W}_{I}\mathbf{X}_{I}`$ should match. In analogy with
linear regression, $`(\mathbf{X}_{I},\mathbf{Y}_{\text{FT}})`$ form
(inputs, targets) and the linearized MMCL loss reduces to a matrix
least-squares problem with $`\mathbf{Y}_{\text{FT}}`$ as the regression
target. The recurring factor $`(n\mathbf{I}_{n}-\mathbf{J}_{n})`$ in
$`\mathbf{Y}_{\text{FT}}`$ implements the contrastive centering with
$`\mathbf{I}_{n}`$ and $`\mathbf{J}_{n}`$ as defined above.

Using $`\mathbf{Y}_{\text{FT}}`$, the linearized MMCL
objective equation [1](#S3.Ex1 "In 3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
reduces (up to constants and a data-dependent quadratic term) to:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\min_{\mathbf{W}_{I}}\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}.
``` |  | (2) |

This formulation is crucial because it enables closed-form solutions for
various finetuning strategies under gradient descent, offering insights
into their behavior (see
§[C.2](#A3.SS2 "C.2 Reformulation of the Least-Squares Objective ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
for the formal trace-to-least-squares equivalence and
§[C.3](#A3.SS3 "C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
for the full derivations and proofs). Using this reformulation, we
analyze how different finetuning strategies mitigate forgetting by
preserving or adapting pretrained knowledge, and reveal the geometric
structure of updates.

### 3.3 Closed-Form Solutions

This subsection follows the setting of [Yang et al.
(2024b)](#bib.bib106) and provides closed-form solutions for various
finetuning strategies following our reformulation
(Equation [2](#S3.E2 "Equation 2 ‣ 3.2 Loss Reformulation via the Contrastive Target Matrix ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")),
revealing a *geometric decomposition*: finetuning involves (i)
preserving pretrained knowledge in directions *orthogonal* to the
finetuning data, and (ii) adapting or mixing knowledge *within* the
task-relevant subspace. We first present the closed-form solutions
below.

###### Theorem 3.2 (Unified Framework for Contrastive Finetuning Solutions).

Let
$`\mathcal{P}_{I}\coloneqq\mathbf{X}_{I}(\mathbf{X}_{I}^{\top}\mathbf{X}_{I})^{+}\mathbf{X}_{I}^{\top}`$
be the orthogonal projector onto $`\mathrm{range}(\mathbf{X}_{I})`$.
Gradient descent initialized at $`\mathbf{W}_{I}^{0}`$ on the objective
$`\mathcal{L}(\mathbf{W}_{I})=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}+\mathcal{R}(\mathbf{W}_{I})`$
converges to the following solutions:

1.  1.
    Direct Finetuning ($`\mathcal{R}(\mathbf{W}_{I})=0`$):

    |  |  |  |
    |----|----|----|
    |  |
    ``` math
    \mathbf{W}_{\text{FT}}=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})+\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}
    ``` |  |
2.  2.
    $`L_{2}`$ Regularization (L2-SP ([Li et al., 2018](#bib.bib16))),
    with
    $`\mathcal{R}(\mathbf{W}_{I})=\frac{\lambda}{2}\left\|\mathbf{W}_{I}-\mathbf{W}_{I}^{0}\right\|_{\text{F}}^{2}`$:

    |  |  |  |
    |----|----|----|
    |  |
    ``` math
    \mathbf{W}_{L_{2}}=(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0})(\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{I})^{-1}
    ``` |  |
3.  3.
    Static Self-Distillation (SD ([Furlanello et al.,
    2018](#bib.bib18))), with
    $`\mathcal{R}(\mathbf{W}_{I})=\frac{\lambda}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{W}_{I}^{0}\mathbf{X}_{I}\right\|_{\text{F}}^{2}`$:

    |  |  |  |
    |----|----|----|
    |  |
    ``` math
    \mathbf{W}_{SD}=\mathbf{W}_{I}^{0}\Big(\mathbf{I}-\frac{1}{1+\lambda}\mathcal{P}_{I}\Big)+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}
    ``` |  |

Here, ⁺ denotes the Moore-Penrose pseudoinverse and $`\lambda>0`$ is the
regularization parameter.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjIuc2YxLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIyOTkuMjEiIG92ZXJmbG93PSJ2aXNpYmxlIiBzdHlsZT0idmVydGljYWwtYWxpZ246LTE0OS42MXB4IiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCAyOTkuMjEgMjk5LjIxIiB3aWR0aD0iMjk5LjIxIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDI5OS4yMSkgbWF0cml4KDEgMCAwIC0xIDAgMCkgdHJhbnNsYXRlKDE0OS42MSwwKSB0cmFuc2xhdGUoMCwxNDkuNjEpIj48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xMTguMTEgLTE0OS42MSBMIC0xMTguMTEgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgLTExOC4xMSBMIDE0OS42MSAtMTE4LjExIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC03OC43NCAtMTQ5LjYxIEwgLTc4Ljc0IDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIC03OC43NCBMIDE0OS42MSAtNzguNzQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0VDRUNFQzstLWx0eC1maWxsLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmctY29sb3I6I0VDRUNFQzsiIGNvbG9yPSIjRUNFQ0VDIiBmaWxsPSIjRUNFQ0VDIiBzdHJva2U9IiNFQ0VDRUMiIHN0cm9rZS13aWR0aD0iMC4zcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTM5LjM3IC0xNDkuNjEgTCAtMzkuMzcgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgLTM5LjM3IEwgMTQ5LjYxIC0zOS4zNyIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAzOS4zNyAtMTQ5LjYxIEwgMzkuMzcgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgMzkuMzcgTCAxNDkuNjEgMzkuMzciIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0VDRUNFQzstLWx0eC1maWxsLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmctY29sb3I6I0VDRUNFQzsiIGNvbG9yPSIjRUNFQ0VDIiBmaWxsPSIjRUNFQ0VDIiBzdHJva2U9IiNFQ0VDRUMiIHN0cm9rZS13aWR0aD0iMC4zcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzguNzQgLTE0OS42MSBMIDc4Ljc0IDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIDc4Ljc0IEwgMTQ5LjYxIDc4Ljc0IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDExOC4xMSAtMTQ5LjYxIEwgMTE4LjExIDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIDExOC4xMSBMIDE0OS42MSAxMTguMTEiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiPjxnIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQjNCM0IzOyIgY29sb3I9IiNCM0IzQjMiIHN0cm9rZS13aWR0aD0iMC42cHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0IzQjNCMzstLWx0eC1maWxsLWNvbG9yOiNCM0IzQjM7IiBmaWxsPSIjQjNCM0IzIiBzdHJva2U9IiNCM0IzQjMiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTE0OS42MSAwIEwgMTQ4Ljc4IDAiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0IzQjNCMzstLWx0eC1maWxsLWNvbG9yOiNCM0IzQjM7IiBmaWxsPSIjQjNCM0IzIiBzdHJva2U9IiNCM0IzQjMiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTQ5LjE5IDApIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0zLjIxIDMuODIgQyAtMi42MiAxLjUzIC0xLjMyIDAuNDUgMCAwIEMgLTEuMzIgLTAuNDUgLTIuNjIgLTEuNTMgLTMuMjEgLTMuODIiIC8+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0IzQjNCMzsiIGNvbG9yPSIjQjNCM0IzIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCM0IzQjM7LS1sdHgtZmlsbC1jb2xvcjojQjNCM0IzOyIgZmlsbD0iI0IzQjNCMyIgc3Ryb2tlPSIjQjNCM0IzIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgLTE0OS42MSBMIDAgMTQ4Ljc4IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCM0IzQjM7LS1sdHgtZmlsbC1jb2xvcjojQjNCM0IzOyIgZmlsbD0iI0IzQjNCMyIgc3Ryb2tlPSIjQjNCM0IzIiBzdHJva2UtZGFzaGFycmF5PSJub25lIiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgdHJhbnNmb3JtPSJtYXRyaXgoMC4wIDEuMCAtMS4wIDAuMCAwIDE0OS4xOSkiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTMuMjEgMy44MiBDIC0yLjYyIDEuNTMgLTEuMzIgMC40NSAwIDAgQyAtMS4zMiAtMC40NSAtMi42MiAtMS41MyAtMy4yMSAtMy44MiIgLz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6Izk5RTFDOTstLWx0eC1maWxsLWNvbG9yOiM5OUUxQzk7LS1sdHgtZmctY29sb3I6Izk5RTFDOTsiIGNvbG9yPSIjOTlFMUM5IiBmaWxsPSIjOTlFMUM5IiBzdHJva2U9IiM5OUUxQzkiIHN0cm9rZS13aWR0aD0iMi4ycHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTExOS4zMyAtNjguOSBMIDExOS4zMyA2OC45IiAvPjwvZz48ZyBzdHJva2Utd2lkdGg9IjAuNHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFOEU4RTg7LS1sdHgtZmlsbC1jb2xvcjojRThFOEU4Oy0tbHR4LWZnLWNvbG9yOiNFOEU4RTg7IiBjb2xvcj0iI0U4RThFOCIgZmlsbD0iI0U4RThFOCIgc3Ryb2tlPSIjRThFOEU4Ij48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCAwIE0gMTAyLjM2IDAgQyAxMDIuMzYgNTYuNTMgNTYuNTMgMTAyLjM2IDAgMTAyLjM2IEMgLTU2LjUzIDEwMi4zNiAtMTAyLjM2IDU2LjUzIC0xMDIuMzYgMCBDIC0xMDIuMzYgLTU2LjUzIC01Ni41MyAtMTAyLjM2IDAgLTEwMi4zNiBDIDU2LjUzIC0xMDIuMzYgMTAyLjM2IC01Ni41MyAxMDIuMzYgMCBaIE0gMCAwIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNDMUMxQzE7LS1sdHgtZmlsbC1jb2xvcjojQzFDMUMxOy0tbHR4LWZnLWNvbG9yOiNDMUMxQzE7IiBjb2xvcj0iI0MxQzFDMSIgZmlsbD0iI0MxQzFDMSIgc3Ryb2tlPSIjQzFDMUMxIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBNIDEwMi4zNiAwIEMgMTAyLjM2IDU2LjUzIDU2LjUzIDEwMi4zNiAwIDEwMi4zNiBDIC01Ni41MyAxMDIuMzYgLTEwMi4zNiA1Ni41MyAtMTAyLjM2IDAgQyAtMTAyLjM2IC01Ni41MyAtNTYuNTMgLTEwMi4zNiAwIC0xMDIuMzYgQyA1Ni41MyAtMTAyLjM2IDEwMi4zNiAtNTYuNTMgMTAyLjM2IDAgWiBNIDAgMCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojQ0NGMEU0Oy0tbHR4LWZpbGwtY29sb3I6I0NDRjBFNDstLWx0eC1mZy1jb2xvcjojQ0NGMEU0OyIgY29sb3I9IiNDQ0YwRTQiIGZpbGw9IiNDQ0YwRTQiIHN0cm9rZT0iI0NDRjBFNCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMCBNIDEwMi4yOSA1OS4wNiBDIDk1Ljc2IDcwLjM1IDQ0LjY4IDUzLjA3IC0xMS44MSAyMC40NiBDIC02OC4zIC0xMi4xNiAtMTA4LjgxIC00Ny43NiAtMTAyLjI5IC01OS4wNiBDIC05NS43NiAtNzAuMzUgLTQ0LjY4IC01My4wNyAxMS44MSAtMjAuNDYgQyA2OC4zIDEyLjE2IDEwOC44MSA0Ny43NiAxMDIuMjkgNTkuMDYgWiBNIDAgMCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNjZEMkFFOy0tbHR4LWZpbGwtY29sb3I6IzY2RDJBRTstLWx0eC1mZy1jb2xvcjojNjZEMkFFOyIgY29sb3I9IiM2NkQyQUUiIGZpbGw9IiM2NkQyQUUiIHN0cm9rZT0iIzY2RDJBRSIgc3Ryb2tlLXdpZHRoPSIwLjdwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTSAxMDIuMjkgNTkuMDYgQyA5NS43NiA3MC4zNSA0NC42OCA1My4wNyAtMTEuODEgMjAuNDYgQyAtNjguMyAtMTIuMTYgLTEwOC44MSAtNDcuNzYgLTEwMi4yOSAtNTkuMDYgQyAtOTUuNzYgLTcwLjM1IC00NC42OCAtNTMuMDcgMTEuODEgLTIwLjQ2IEMgNjguMyAxMi4xNiAxMDguODEgNDcuNzYgMTAyLjI5IDU5LjA2IFogTSAwIDAiIC8+PC9nPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDAgTSAyLjM2IDAgQyAyLjM2IDEuMyAxLjMgMi4zNiAwIDIuMzYgQyAtMS4zIDIuMzYgLTIuMzYgMS4zIC0yLjM2IDAgQyAtMi4zNiAtMS4zIC0xLjMgLTIuMzYgMCAtMi4zNiBDIDEuMyAtMi4zNiAyLjM2IC0xLjMgMi4zNiAwIFogTSAwIDAiIC8+PGcgc3Ryb2tlLXdpZHRoPSIxLjhwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCA2MS45MyA3NS42OSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgwLjYzMzI0IDAuNzczOTYgLTAuNzczOTYgMC42MzMyNCA2MS45MyA3NS42OSkiPjxwYXRoIGQ9Ik0gOS4xMyAwIEMgOC4wMSAwLjI4IDMuMDggMS44NyAwIDMuNTkgTCAwIC0zLjU5IEMgMy4wOCAtMS44NyA4LjAxIC0wLjI4IDkuMTMgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIGZpbGwtb3BhY2l0eT0iMC43IiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjAuOHB0LDIuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1vcGFjaXR5PSIwLjciIHN0cm9rZS13aWR0aD0iMC44cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzAuODcgODYuNjEgTCA5MC42NiA1Mi4zNCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIGZpbGwtb3BhY2l0eT0iMC43IiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjAuOHB0LDIuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1vcGFjaXR5PSIwLjciIHN0cm9rZS13aWR0aD0iMC44cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzAuODcgODYuNjEgTCAtMTkuNzkgMzQuMjciIC8+PC9nPjxnIHN0cm9rZS13aWR0aD0iMS4ycHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjMuMHB0LDMuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gMCAwIEwgODEuMzEgNDYuOTQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMC44NjYwMSAwLjUgLTAuNSAwLjg2NjAxIDgxLjMxIDQ2Ljk0KSI+PHBhdGggZD0iTSA3LjQ3IDAgQyA2LjU1IDAuMjMgMi41MiAxLjUxIDAgMi45MSBMIDAgLTIuOTEgQyAyLjUyIC0xLjUxIDYuNTUgLTAuMjMgNy40NyAwIFoiIC8+PC9nPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjAuNyIgc3Ryb2tlLW9wYWNpdHk9IjAuNyIgc3Ryb2tlLXdpZHRoPSIwLjlwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIHN0cm9rZT0iIzREQTBGRiIgc3Ryb2tlLWRhc2hhcnJheT0iMy4wcHQsMi4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCAtMTUuMjIgMjYuMzciIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuNTAwMDIgMC44NjYwMSAtMC44NjYwMSAtMC41MDAwMiAtMTUuMjIgMjYuMzcpIj48cGF0aCBkPSJNIDYuNjQgMCBDIDUuODMgMC4yIDIuMjQgMS4zNCAwIDIuNTcgTCAwIC0yLjU3IEMgMi4yNCAtMS4zNCA1LjgzIC0wLjIgNi42NCAwIFoiIC8+PC9nPjwvZz48ZyBzdHJva2Utd2lkdGg9IjEuOHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmlsbC1jb2xvcjojMDBDODY0Oy0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBjb2xvcj0iIzAwQzg2NCIgZmlsbD0iIzAwQzg2NCIgc3Ryb2tlPSIjMDBDODY0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIC00OS4xNSAtMjguMzgiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwQzg2NDstLWx0eC1maWxsLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIGNvbG9yPSIjMDBDODY0IiBmaWxsPSIjMDBDODY0IiBzdHJva2U9IiMwMEM4NjQiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuODY2MDMgLTAuNSAwLjUgLTAuODY2MDMgLTQ5LjE1IC0yOC4zOCkiPjxwYXRoIGQ9Ik0gOS4xMyAwIEMgOC4wMSAwLjI4IDMuMDggMS44NyAwIDMuNTkgTCAwIC0zLjU5IEMgMy4wOCAtMS44NyA4LjAxIC0wLjI4IDkuMTMgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZBQkFCOy0tbHR4LWZpbGwtY29sb3I6I0ZGQUJBQjstLWx0eC1mZy1jb2xvcjojRkZBQkFCOyIgY29sb3I9IiNGRkFCQUIiIGZpbGw9IiNGRkFCQUIiIHN0cm9rZT0iI0ZGQUJBQiIgc3Ryb2tlLWRhc2hhcnJheT0iMy4wcHQsMy4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTkuNzkgMzQuMjcgTCAtODEuMTYgLTEuMTYiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzY2REVBMjstLWx0eC1maWxsLWNvbG9yOiM2NkRFQTI7LS1sdHgtZmctY29sb3I6IzY2REVBMjsiIGNvbG9yPSIjNjZERUEyIiBmaWxsPSIjNjZERUEyIiBzdHJva2U9IiM2NkRFQTIiIHN0cm9rZS1kYXNoYXJyYXk9IjMuMHB0LDMuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS13aWR0aD0iMC42cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTgxLjE2IC0xLjE2IEwgLTYxLjM3IC0zNS40MyIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojOTk5OTk5Oy0tbHR4LWZpbGwtY29sb3I6Izk5OTk5OTstLWx0eC1mZy1jb2xvcjojOTk5OTk5OyIgY29sb3I9IiM5OTk5OTkiIGZpbGw9IiM5OTk5OTkiIHN0cm9rZT0iIzk5OTk5OSIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEyOC4zMyAtMTQuMTQpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDoxLjE2ZW07LS1sdHgtZm8taGVpZ2h0OjAuNDRlbTstLWx0eC1mby1kZXB0aDowLjE1ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI5LjY0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3LjE1KSIgd2lkdGg9IjE4Ljk0Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMS5waWMxLm0xIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IndfezF9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk5OTk5OyIgbWF0aGNvbG9yPSIjOTk5OTk5IiBtYXRoc2l6ZT0iMS4yMDBlbSI+dzwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk5OTk5OyIgbWF0aGNvbG9yPSIjOTk5OTk5IiBtYXRoc2l6ZT0iMS4yMDBlbSI+MTwvbW4+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+d197MX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiM5OTk5OTk7LS1sdHgtZmlsbC1jb2xvcjojOTk5OTk5Oy0tbHR4LWZnLWNvbG9yOiM5OTk5OTk7IiBjb2xvcj0iIzk5OTk5OSIgZmlsbD0iIzk5OTk5OSIgc3Ryb2tlPSIjOTk5OTk5IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgLTIxLjI4IDEzNS40NykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjEuMTZlbTstLWx0eC1mby1oZWlnaHQ6MC40NGVtOy0tbHR4LWZvLWRlcHRoOjAuMTVlbTtmb250LXNpemU6MTEuNzVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjkuNjQiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuMTUpIiB3aWR0aD0iMTguOTQiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YxLnBpYzEubTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0id197Mn0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM5OTk5OTk7IiBtYXRoY29sb3I9IiM5OTk5OTkiIG1hdGhzaXplPSIxLjIwMGVtIj53PC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM5OTk5OTk7IiBtYXRoY29sb3I9IiM5OTk5OTkiIG1hdGhzaXplPSIxLjIwMGVtIj4yPC9tbj48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij53X3syfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwQjQ3ODstLWx0eC1maWxsLWNvbG9yOiMwMEI0Nzg7LS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIGNvbG9yPSIjMDBCNDc4IiBmaWxsPSIjMDBCNDc4IiBzdHJva2U9IiMwMEI0NzgiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA3OS4xNiA2Ny43MSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjQuMzNlbTstLWx0eC1mby1oZWlnaHQ6MC44NWVtOy0tbHR4LWZvLWRlcHRoOjAuMjVlbTtmb250LXNpemU6MTBwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjE1LjIxIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMS43NSkiIHdpZHRoPSI1OS44OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjEucGljMS5tMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aHJte3NwYW59KFxtYXRoYmZ7WH1fe0l9XntcdG9wfSkiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBtYXRoY29sb3I9IiMwMEI0NzgiPnNwYW48L21pPjxtbz7igaE8L21vPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIG1hdGhjb2xvcj0iIzAwQjQ3OCIgc3RyZXRjaHk9ImZhbHNlIj4oPC9tbz48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBtYXRoY29sb3I9IiMwMEI0NzgiPvCdkJc8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIG1hdGhjb2xvcj0iIzAwQjQ3OCI+STwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgbWF0aGNvbG9yPSIjMDBCNDc4Ij7iiqQ8L21vPjwvbXN1YnN1cD48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBtYXRoY29sb3I9IiMwMEI0NzgiIHN0cmV0Y2h5PSJmYWxzZSI+KTwvbW8+PC9tcm93PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRocm17c3Bhbn0oXG1hdGhiZntYfV97SX1ee1x0b3B9KTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzY0NjQ2NDstLWx0eC1maWxsLWNvbG9yOiM2NDY0NjQ7LS1sdHgtZmctY29sb3I6IzY0NjQ2NDsiIGNvbG9yPSIjNjQ2NDY0IiBmaWxsPSIjNjQ2NDY0IiBzdHJva2U9IiM2NDY0NjQiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA1My4wNCAtNjEuMykiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjQuMzdlbTstLWx0eC1mby1oZWlnaHQ6MC42NGVtOy0tbHR4LWZvLWRlcHRoOjAuMjZlbTtmb250LXNpemU6OC41cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMC42NCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNy41NikiIHdpZHRoPSI1MS40MSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjEucGljMS5tNCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGNhbHtEfV97XHRleHR7cHJldHJhaW59fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzY0NjQ2NDsiIGNsYXNzPSJsdHhfZm9udF9tYXRoY2FsaWdyYXBoaWMiIG1hdGhjb2xvcj0iIzY0NjQ2NCIgbWF0aHNpemU9IjAuODAwZW0iPvCdkp88L21pPjxtdGV4dCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzY0NjQ2NDsiIGNsYXNzPSJsdHhfbWF0aHZhcmlhbnRfYm9sZCIgbWF0aGNvbG9yPSIjNjQ2NDY0IiBtYXRoc2l6ZT0iMC44MDBlbSI+cHJldHJhaW48L210ZXh0PjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoY2Fse0R9X3tcdGV4dHtwcmV0cmFpbn19PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDBCNDc4Oy0tbHR4LWZpbGwtY29sb3I6IzAwQjQ3ODstLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgY29sb3I9IiMwMEI0NzgiIGZpbGw9IiMwMEI0NzgiIHN0cm9rZT0iIzAwQjQ3OCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC0xMTguOCAtNDIuNCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjIuMTNlbTstLWx0eC1mby1oZWlnaHQ6MC42NGVtOy0tbHR4LWZvLWRlcHRoOjAuMTNlbTtmb250LXNpemU6OC41cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI5LjA3IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3LjU2KSIgd2lkdGg9IjI1LjAxIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMS5waWMxLm01IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoY2Fse0R9X3tcdGV4dHtGVH19IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgY2xhc3M9Imx0eF9mb250X21hdGhjYWxpZ3JhcGhpYyIgbWF0aGNvbG9yPSIjMDBCNDc4IiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2SnzwvbWk+PG10ZXh0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgY2xhc3M9Imx0eF9tYXRodmFyaWFudF9ib2xkIiBtYXRoY29sb3I9IiMwMEI0NzgiIG1hdGhzaXplPSIwLjgwMGVtIj5GVDwvbXRleHQ+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhjYWx7RH1fe1x0ZXh0e0ZUfX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOy0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBjb2xvcj0iIzAwMDAwMCIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgLTkuMyAtMTUuMzgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDowLjU4ZW07LS1sdHgtZm8taGVpZ2h0OjAuNjFlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjguNXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iNy4xMyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNy4xMykiIHdpZHRoPSI2Ljc4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMS5waWMxLm02IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7MH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZ+OPC9tbj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoYmZ7MH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDc4RkY7LS1sdHgtZmlsbC1jb2xvcjojMDA3OEZGOy0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBjb2xvcj0iIzAwNzhGRiIgZmlsbD0iIzAwNzhGRiIgc3Ryb2tlPSIjMDA3OEZGIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNjUuNjIgODguOTgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDoxLjYxZW07LS1sdHgtZm8taGVpZ2h0OjAuODNlbTstLWx0eC1mby1kZXB0aDowLjE1ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxNi4wMSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTMuNTIpIiB3aWR0aD0iMjYuMjUiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YxLnBpYzEubTciIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhiZntXfV97SX1eezB9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWJzdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMS4yMDBlbSI+8J2QljwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMS4yMDBlbSI+STwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMS4yMDBlbSI+MDwvbW4+PC9tc3Vic3VwPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97SX1eezB9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDBDODY0Oy0tbHR4LWZpbGwtY29sb3I6IzAwQzg2NDstLWx0eC1mZy1jb2xvcjojMDBDODY0OyIgY29sb3I9IiMwMEM4NjQiIGZpbGw9IiMwMEM4NjQiIHN0cm9rZT0iIzAwQzg2NCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC04OC42NiAtNDcuNzgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDoyLjM5ZW07LS1sdHgtZm8taGVpZ2h0OjAuN2VtOy0tbHR4LWZvLWRlcHRoOjAuMTVlbTtmb250LXNpemU6MTEuNzVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjEzLjkyIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMS40MykiIHdpZHRoPSIzOC44MyI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjEucGljMS5tOCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJme1d9X3tcdGV4dHtGVH19Xntcc3Rhcn0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBtYXRoY29sb3I9IiMwMEM4NjQiIG1hdGhzaXplPSIxLjIwMGVtIj7wnZCWPC9taT48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iIzAwQzg2NCIgbWF0aHNpemU9IjEuMjAwZW0iPkZUPC9tdGV4dD48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBtYXRoY29sb3I9IiMwMEM4NjQiIG1hdGhzaXplPSIxLjIwMGVtIj7ii4Y8L21vPjwvbXN1YnN1cD48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoYmZ7V31fe1x0ZXh0e0ZUfX1ee1xzdGFyfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwNzhGRjstLWx0eC1maWxsLWNvbG9yOiMwMDc4RkY7LS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIGNvbG9yPSIjMDA3OEZGIiBmaWxsPSIjMDA3OEZGIiBzdHJva2U9IiMwMDc4RkYiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA4OS4yNyA0MC43NikiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjIuOTFlbTstLWx0eC1mby1oZWlnaHQ6MC43NmVtOy0tbHR4LWZvLWRlcHRoOjAuMTNlbTtmb250LXNpemU6OC41cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMC4zOCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOC45KSIgd2lkdGg9IjM0LjI2Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMS5waWMxLm05IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe0l9XnswfVxtYXRoY2Fse1B9X3tJfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtc3Vic3VwPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPvCdkJY8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPkk8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPjA8L21uPjwvbXN1YnN1cD48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIGNsYXNzPSJsdHhfZm9udF9tYXRoY2FsaWdyYXBoaWMiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPvCdkqs8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPkk8L21pPjwvbXN1Yj48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tJfV57MH1cbWF0aGNhbHtQfV97SX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDc4RkY7LS1sdHgtZmlsbC1jb2xvcjojMDA3OEZGOy0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBjb2xvcj0iIzAwNzhGRiIgZmlsbD0iIzAwNzhGRiIgc3Ryb2tlPSIjMDA3OEZGIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgLTcxLjU4IDM5LjA4KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6NC40NmVtOy0tbHR4LWZvLWhlaWdodDowLjc2ZW07LS1sdHgtZm8tZGVwdGg6MC4yNGVtO2ZvbnQtc2l6ZTo4LjVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjExLjY3IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA4LjkpIiB3aWR0aD0iNTIuNDEiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YxLnBpYzEubTEwIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe0l9XnswfShcbWF0aGJme0l9LVxtYXRoY2Fse1B9X3tJfSkiIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZCWPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj5JPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj4wPC9tbj48L21zdWJzdXA+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1heHNpemU9IjAuODAwZW0iIG1pbnNpemU9IjAuODAwZW0iPig8L21vPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPvCdkIg8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPuKIkjwvbW8+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY2xhc3M9Imx0eF9mb250X21hdGhjYWxpZ3JhcGhpYyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2SqzwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+STwvbWk+PC9tc3ViPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1heHNpemU9IjAuODAwZW0iIG1pbnNpemU9IjAuODAwZW0iPik8L21vPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tJfV57MH0oXG1hdGhiZntJfS1cbWF0aGNhbHtQfV97SX0pPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBjb2xvcj0iI0ZGMkQyRCIgc3Ryb2tlLXdpZHRoPSIyLjBwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkYyRDJEOy0tbHR4LWZpbGwtY29sb3I6I0ZGMkQyRDsiIGZpbGw9IiNGRjJEMkQiIHN0cm9rZT0iI0ZGMkQyRCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCAtNjUuOTQgLTAuOTQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0ZGMkQyRDstLWx0eC1maWxsLWNvbG9yOiNGRjJEMkQ7IiBmaWxsPSIjRkYyRDJEIiBzdHJva2U9IiNGRjJEMkQiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuOTk5OSAtMC4wMTQyOCAwLjAxNDI4IC0wLjk5OTkgLTY1Ljk0IC0wLjk0KSI+PHBhdGggZD0iTSA5LjY5IDAgQyA4LjUgMC4zIDMuMjcgMS45OCAwIDMuODIgTCAwIC0zLjgyIEMgMy4yNyAtMS45OCA4LjUgLTAuMyA5LjY5IDAgWiIgLz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0ZGMkQyRDstLWx0eC1maWxsLWNvbG9yOiNGRjJEMkQ7LS1sdHgtZmctY29sb3I6I0ZGMkQyRDsiIGNvbG9yPSIjRkYyRDJEIiBmaWxsPSIjRkYyRDJEIiBzdHJva2U9IiNGRjJEMkQiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtOTYuNjQgLTE3LjQyKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6Mi4zOWVtOy0tbHR4LWZvLWhlaWdodDowLjdlbTstLWx0eC1mby1kZXB0aDowLjE1ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMy44OCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTEuMzkpIiB3aWR0aD0iMzguODMiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YxLnBpYzEubTExIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe1x0ZXh0e0ZUfX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1hdGhzaXplPSIxLjIwMGVtIj7wnZCWPC9taT48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iI0ZGMkQyRCIgbWF0aHNpemU9IjEuMjAwZW0iPkZUPC9tdGV4dD48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tcdGV4dHtGVH19PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkYyRDJEOy0tbHR4LWZpbGwtY29sb3I6I0ZGMkQyRDstLWx0eC1mZy1jb2xvcjojRkYyRDJEOyIgY29sb3I9IiNGRjJEMkQiIGZpbGw9IiNGRjJEMkQiIHN0cm9rZT0iI0ZGMkQyRCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC02My43NyAtMTI2KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjkuNDFlbTstLWx0eC1mby1oZWlnaHQ6MC43MWVtOy0tbHR4LWZvLWRlcHRoOjAuNzFlbTtmb250LXNpemU6OS44cHQ7IiBoZWlnaHQ9IjE5LjM0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5LjY5KSIgd2lkdGg9IjEyNy41NCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjEucGljMS5tMTIiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhiZntXfV97XHRleHR7RlR9fT1cdW5kZXJicmFjZXtcbWF0aGJme1d9X3tJfV57MH0oXG1hdGhiZntJfS1cbWF0aGNhbHtQfV97SX0pfV97XHRleHR7cHJlc2VydmV9fStcdW5kZXJicmFjZXtcbWF0aGJme1d9X3tcdGV4dHtGVH19Xntcc3Rhcn19X3tcdGV4dHtyZXBsYWNlfX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXN1Yj48bWkgbWF0aHNpemU9IjAuODAwZW0iPvCdkJY8L21pPjxtdGV4dCBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhzaXplPSIwLjgwMGVtIj5GVDwvbXRleHQ+PC9tc3ViPjxtbyBtYXRoc2l6ZT0iMC44MDBlbSI+PTwvbW8+PG1yb3c+PG11bmRlcj48bXVuZGVyIGFjY2VudHVuZGVyPSJ0cnVlIj48bXJvdz48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZCWPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1hdGhzaXplPSIwLjgwMGVtIj5JPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1hdGhzaXplPSIwLjgwMGVtIj4wPC9tbj48L21zdWJzdXA+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1heHNpemU9IjAuODAwZW0iIG1pbnNpemU9IjAuODAwZW0iPig8L21vPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMkQyRDsiIG1hdGhjb2xvcj0iI0ZGMkQyRCIgbWF0aHNpemU9IjAuODAwZW0iPvCdkIg8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMkQyRDsiIG1hdGhjb2xvcj0iI0ZGMkQyRCIgbWF0aHNpemU9IjAuODAwZW0iPuKIkjwvbW8+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYyRDJEOyIgY2xhc3M9Imx0eF9mb250X21hdGhjYWxpZ3JhcGhpYyIgbWF0aGNvbG9yPSIjRkYyRDJEIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2SqzwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYyRDJEOyIgbWF0aGNvbG9yPSIjRkYyRDJEIiBtYXRoc2l6ZT0iMC44MDBlbSI+STwvbWk+PC9tc3ViPjwvbXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1heHNpemU9IjAuODAwZW0iIG1pbnNpemU9IjAuODAwZW0iPik8L21vPjwvbXJvdz48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBzdHJldGNoeT0idHJ1ZSI+4o+fPC9tbz48L211bmRlcj48bXRleHQgY2xhc3M9Imx0eF9tYXRodmFyaWFudF9ib2xkIiBtYXRoc2l6ZT0iMC44MDBlbSI+cHJlc2VydmU8L210ZXh0PjwvbXVuZGVyPjxtbyBtYXRoc2l6ZT0iMC44MDBlbSI+KzwvbW8+PG11bmRlcj48bXVuZGVyIGFjY2VudHVuZGVyPSJ0cnVlIj48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZCWPC9taT48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iI0ZGMkQyRCIgbWF0aHNpemU9IjAuODAwZW0iPkZUPC9tdGV4dD48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjJEMkQ7IiBtYXRoY29sb3I9IiNGRjJEMkQiIG1hdGhzaXplPSIwLjgwMGVtIj7ii4Y8L21vPjwvbXN1YnN1cD48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7IiBtYXRoY29sb3I9IiMwMDAwMDAiIHN0cmV0Y2h5PSJ0cnVlIj7ij588L21vPjwvbXVuZGVyPjxtdGV4dCBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhzaXplPSIwLjgwMGVtIj5yZXBsYWNlPC9tdGV4dD48L211bmRlcj48L21yb3c+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97XHRleHR7RlR9fT1cdW5kZXJicmFjZXtcbWF0aGJme1d9X3tJfV57MH0oXG1hdGhiZntJfS1cbWF0aGNhbHtQfV97SX0pfV97XHRleHR7cHJlc2VydmV9fStcdW5kZXJicmFjZXtcbWF0aGJme1d9X3tcdGV4dHtGVH19Xntcc3Rhcn19X3tcdGV4dHtyZXBsYWNlfX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTMy5GMi5zZjEucGljMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo4MCU7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9nPjwvZz48L3N2Zz4=)

(a) Direct FT: Preserves orthogonal, replaces parallel component.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjIuc2YyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIyOTkuMjEiIG92ZXJmbG93PSJ2aXNpYmxlIiBzdHlsZT0idmVydGljYWwtYWxpZ246LTE0OS42MXB4IiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCAyOTkuMjEgMjk5LjIxIiB3aWR0aD0iMjk5LjIxIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDI5OS4yMSkgbWF0cml4KDEgMCAwIC0xIDAgMCkgdHJhbnNsYXRlKDE0OS42MSwwKSB0cmFuc2xhdGUoMCwxNDkuNjEpIj48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xMTguMTEgLTE0OS42MSBMIC0xMTguMTEgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgLTExOC4xMSBMIDE0OS42MSAtMTE4LjExIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC03OC43NCAtMTQ5LjYxIEwgLTc4Ljc0IDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIC03OC43NCBMIDE0OS42MSAtNzguNzQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0VDRUNFQzstLWx0eC1maWxsLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmctY29sb3I6I0VDRUNFQzsiIGNvbG9yPSIjRUNFQ0VDIiBmaWxsPSIjRUNFQ0VDIiBzdHJva2U9IiNFQ0VDRUMiIHN0cm9rZS13aWR0aD0iMC4zcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTM5LjM3IC0xNDkuNjEgTCAtMzkuMzcgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgLTM5LjM3IEwgMTQ5LjYxIC0zOS4zNyIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAzOS4zNyAtMTQ5LjYxIEwgMzkuMzcgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgMzkuMzcgTCAxNDkuNjEgMzkuMzciIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0VDRUNFQzstLWx0eC1maWxsLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmctY29sb3I6I0VDRUNFQzsiIGNvbG9yPSIjRUNFQ0VDIiBmaWxsPSIjRUNFQ0VDIiBzdHJva2U9IiNFQ0VDRUMiIHN0cm9rZS13aWR0aD0iMC4zcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzguNzQgLTE0OS42MSBMIDc4Ljc0IDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIDc4Ljc0IEwgMTQ5LjYxIDc4Ljc0IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDExOC4xMSAtMTQ5LjYxIEwgMTE4LjExIDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIDExOC4xMSBMIDE0OS42MSAxMTguMTEiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiPjxnIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQjNCM0IzOyIgY29sb3I9IiNCM0IzQjMiIHN0cm9rZS13aWR0aD0iMC42cHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0IzQjNCMzstLWx0eC1maWxsLWNvbG9yOiNCM0IzQjM7IiBmaWxsPSIjQjNCM0IzIiBzdHJva2U9IiNCM0IzQjMiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTE0OS42MSAwIEwgMTQ4Ljc4IDAiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0IzQjNCMzstLWx0eC1maWxsLWNvbG9yOiNCM0IzQjM7IiBmaWxsPSIjQjNCM0IzIiBzdHJva2U9IiNCM0IzQjMiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTQ5LjE5IDApIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0zLjIxIDMuODIgQyAtMi42MiAxLjUzIC0xLjMyIDAuNDUgMCAwIEMgLTEuMzIgLTAuNDUgLTIuNjIgLTEuNTMgLTMuMjEgLTMuODIiIC8+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0IzQjNCMzsiIGNvbG9yPSIjQjNCM0IzIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCM0IzQjM7LS1sdHgtZmlsbC1jb2xvcjojQjNCM0IzOyIgZmlsbD0iI0IzQjNCMyIgc3Ryb2tlPSIjQjNCM0IzIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgLTE0OS42MSBMIDAgMTQ4Ljc4IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCM0IzQjM7LS1sdHgtZmlsbC1jb2xvcjojQjNCM0IzOyIgZmlsbD0iI0IzQjNCMyIgc3Ryb2tlPSIjQjNCM0IzIiBzdHJva2UtZGFzaGFycmF5PSJub25lIiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgdHJhbnNmb3JtPSJtYXRyaXgoMC4wIDEuMCAtMS4wIDAuMCAwIDE0OS4xOSkiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTMuMjEgMy44MiBDIC0yLjYyIDEuNTMgLTEuMzIgMC40NSAwIDAgQyAtMS4zMiAtMC40NSAtMi42MiAtMS41MyAtMy4yMSAtMy44MiIgLz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6Izk5RTFDOTstLWx0eC1maWxsLWNvbG9yOiM5OUUxQzk7LS1sdHgtZmctY29sb3I6Izk5RTFDOTsiIGNvbG9yPSIjOTlFMUM5IiBmaWxsPSIjOTlFMUM5IiBzdHJva2U9IiM5OUUxQzkiIHN0cm9rZS13aWR0aD0iMi4ycHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTExOS4zMyAtNjguOSBMIDExOS4zMyA2OC45IiAvPjwvZz48ZyBzdHJva2Utd2lkdGg9IjAuNHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFOEU4RTg7LS1sdHgtZmlsbC1jb2xvcjojRThFOEU4Oy0tbHR4LWZnLWNvbG9yOiNFOEU4RTg7IiBjb2xvcj0iI0U4RThFOCIgZmlsbD0iI0U4RThFOCIgc3Ryb2tlPSIjRThFOEU4Ij48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCAwIE0gMTAyLjM2IDAgQyAxMDIuMzYgNTYuNTMgNTYuNTMgMTAyLjM2IDAgMTAyLjM2IEMgLTU2LjUzIDEwMi4zNiAtMTAyLjM2IDU2LjUzIC0xMDIuMzYgMCBDIC0xMDIuMzYgLTU2LjUzIC01Ni41MyAtMTAyLjM2IDAgLTEwMi4zNiBDIDU2LjUzIC0xMDIuMzYgMTAyLjM2IC01Ni41MyAxMDIuMzYgMCBaIE0gMCAwIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNDMUMxQzE7LS1sdHgtZmlsbC1jb2xvcjojQzFDMUMxOy0tbHR4LWZnLWNvbG9yOiNDMUMxQzE7IiBjb2xvcj0iI0MxQzFDMSIgZmlsbD0iI0MxQzFDMSIgc3Ryb2tlPSIjQzFDMUMxIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBNIDEwMi4zNiAwIEMgMTAyLjM2IDU2LjUzIDU2LjUzIDEwMi4zNiAwIDEwMi4zNiBDIC01Ni41MyAxMDIuMzYgLTEwMi4zNiA1Ni41MyAtMTAyLjM2IDAgQyAtMTAyLjM2IC01Ni41MyAtNTYuNTMgLTEwMi4zNiAwIC0xMDIuMzYgQyA1Ni41MyAtMTAyLjM2IDEwMi4zNiAtNTYuNTMgMTAyLjM2IDAgWiBNIDAgMCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojQ0NGMEU0Oy0tbHR4LWZpbGwtY29sb3I6I0NDRjBFNDstLWx0eC1mZy1jb2xvcjojQ0NGMEU0OyIgY29sb3I9IiNDQ0YwRTQiIGZpbGw9IiNDQ0YwRTQiIHN0cm9rZT0iI0NDRjBFNCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMCBNIDEwMi4yOSA1OS4wNiBDIDk1Ljc2IDcwLjM1IDQ0LjY4IDUzLjA3IC0xMS44MSAyMC40NiBDIC02OC4zIC0xMi4xNiAtMTA4LjgxIC00Ny43NiAtMTAyLjI5IC01OS4wNiBDIC05NS43NiAtNzAuMzUgLTQ0LjY4IC01My4wNyAxMS44MSAtMjAuNDYgQyA2OC4zIDEyLjE2IDEwOC44MSA0Ny43NiAxMDIuMjkgNTkuMDYgWiBNIDAgMCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNjZEMkFFOy0tbHR4LWZpbGwtY29sb3I6IzY2RDJBRTstLWx0eC1mZy1jb2xvcjojNjZEMkFFOyIgY29sb3I9IiM2NkQyQUUiIGZpbGw9IiM2NkQyQUUiIHN0cm9rZT0iIzY2RDJBRSIgc3Ryb2tlLXdpZHRoPSIwLjdwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTSAxMDIuMjkgNTkuMDYgQyA5NS43NiA3MC4zNSA0NC42OCA1My4wNyAtMTEuODEgMjAuNDYgQyAtNjguMyAtMTIuMTYgLTEwOC44MSAtNDcuNzYgLTEwMi4yOSAtNTkuMDYgQyAtOTUuNzYgLTcwLjM1IC00NC42OCAtNTMuMDcgMTEuODEgLTIwLjQ2IEMgNjguMyAxMi4xNiAxMDguODEgNDcuNzYgMTAyLjI5IDU5LjA2IFogTSAwIDAiIC8+PC9nPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDAgTSAyLjM2IDAgQyAyLjM2IDEuMyAxLjMgMi4zNiAwIDIuMzYgQyAtMS4zIDIuMzYgLTIuMzYgMS4zIC0yLjM2IDAgQyAtMi4zNiAtMS4zIC0xLjMgLTIuMzYgMCAtMi4zNiBDIDEuMyAtMi4zNiAyLjM2IC0xLjMgMi4zNiAwIFogTSAwIDAiIC8+PGcgc3Ryb2tlLXdpZHRoPSIxLjhwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCA2MS45MyA3NS42OSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgwLjYzMzI0IDAuNzczOTYgLTAuNzczOTYgMC42MzMyNCA2MS45MyA3NS42OSkiPjxwYXRoIGQ9Ik0gOS4xMyAwIEMgOC4wMSAwLjI4IDMuMDggMS44NyAwIDMuNTkgTCAwIC0zLjU5IEMgMy4wOCAtMS44NyA4LjAxIC0wLjI4IDkuMTMgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIGZpbGwtb3BhY2l0eT0iMC43IiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjAuOHB0LDIuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1vcGFjaXR5PSIwLjciIHN0cm9rZS13aWR0aD0iMC44cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzAuODcgODYuNjEgTCA5MC42NiA1Mi4zNCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIGZpbGwtb3BhY2l0eT0iMC43IiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjAuOHB0LDIuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1vcGFjaXR5PSIwLjciIHN0cm9rZS13aWR0aD0iMC44cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzAuODcgODYuNjEgTCAtMTkuNzkgMzQuMjciIC8+PC9nPjxnIHN0cm9rZS13aWR0aD0iMS4ycHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjMuMHB0LDMuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gMCAwIEwgODEuMzEgNDYuOTQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMC44NjYwMSAwLjUgLTAuNSAwLjg2NjAxIDgxLjMxIDQ2Ljk0KSI+PHBhdGggZD0iTSA3LjQ3IDAgQyA2LjU1IDAuMjMgMi41MiAxLjUxIDAgMi45MSBMIDAgLTIuOTEgQyAyLjUyIC0xLjUxIDYuNTUgLTAuMjMgNy40NyAwIFoiIC8+PC9nPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjAuNyIgc3Ryb2tlLW9wYWNpdHk9IjAuNyIgc3Ryb2tlLXdpZHRoPSIwLjlwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIHN0cm9rZT0iIzREQTBGRiIgc3Ryb2tlLWRhc2hhcnJheT0iMy4wcHQsMi4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCAtMTUuMjIgMjYuMzciIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuNTAwMDIgMC44NjYwMSAtMC44NjYwMSAtMC41MDAwMiAtMTUuMjIgMjYuMzcpIj48cGF0aCBkPSJNIDYuNjQgMCBDIDUuODMgMC4yIDIuMjQgMS4zNCAwIDIuNTcgTCAwIC0yLjU3IEMgMi4yNCAtMS4zNCA1LjgzIC0wLjIgNi42NCAwIFoiIC8+PC9nPjwvZz48ZyBzdHJva2Utd2lkdGg9IjEuOHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmlsbC1jb2xvcjojMDBDODY0Oy0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBjb2xvcj0iIzAwQzg2NCIgZmlsbD0iIzAwQzg2NCIgc3Ryb2tlPSIjMDBDODY0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIC00OS4xNSAtMjguMzgiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwQzg2NDstLWx0eC1maWxsLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIGNvbG9yPSIjMDBDODY0IiBmaWxsPSIjMDBDODY0IiBzdHJva2U9IiMwMEM4NjQiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuODY2MDMgLTAuNSAwLjUgLTAuODY2MDMgLTQ5LjE1IC0yOC4zOCkiPjxwYXRoIGQ9Ik0gOS4xMyAwIEMgOC4wMSAwLjI4IDMuMDggMS44NyAwIDMuNTkgTCAwIC0zLjU5IEMgMy4wOCAtMS44NyA4LjAxIC0wLjI4IDkuMTMgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRTVBNkYzOy0tbHR4LWZpbGwtY29sb3I6I0U1QTZGMzstLWx0eC1mZy1jb2xvcjojRTVBNkYzOyIgY29sb3I9IiNFNUE2RjMiIGZpbGw9IiNFNUE2RjMiIHN0cm9rZT0iI0U1QTZGMyIgc3Ryb2tlLWRhc2hhcnJheT0iMC42cHQsMS4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCAtNTAuNzYgMTYuNCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRTVBNkYzOy0tbHR4LWZpbGwtY29sb3I6I0U1QTZGMzstLWx0eC1mZy1jb2xvcjojRTVBNkYzOyIgY29sb3I9IiNFNUE2RjMiIGZpbGw9IiNFNUE2RjMiIHN0cm9rZT0iI0U1QTZGMyIgc3Ryb2tlLWRhc2hhcnJheT0iMC42cHQsMS4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSA3MC44NyA4Ni42MSBMIC01MC43NiAxNi40IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFNUE2RjM7LS1sdHgtZmlsbC1jb2xvcjojRTVBNkYzOy0tbHR4LWZnLWNvbG9yOiNFNUE2RjM7IiBjb2xvcj0iI0U1QTZGMyIgZmlsbD0iI0U1QTZGMyIgc3Ryb2tlPSIjRTVBNkYzIiBzdHJva2UtZGFzaGFycmF5PSIwLjZwdCwxLjBwdCIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC04MS4xNiAtMS4xNiBMIC01MC43NiAxNi40IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiM5OTk5OTk7LS1sdHgtZmlsbC1jb2xvcjojOTk5OTk5Oy0tbHR4LWZnLWNvbG9yOiM5OTk5OTk7IiBjb2xvcj0iIzk5OTk5OSIgZmlsbD0iIzk5OTk5OSIgc3Ryb2tlPSIjOTk5OTk5IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTI4LjMzIC0xNC4xNCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjEuMTZlbTstLWx0eC1mby1oZWlnaHQ6MC40NGVtOy0tbHR4LWZvLWRlcHRoOjAuMTVlbTtmb250LXNpemU6MTEuNzVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjkuNjQiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuMTUpIiB3aWR0aD0iMTguOTQiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0id197MX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM5OTk5OTk7IiBtYXRoY29sb3I9IiM5OTk5OTkiIG1hdGhzaXplPSIxLjIwMGVtIj53PC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM5OTk5OTk7IiBtYXRoY29sb3I9IiM5OTk5OTkiIG1hdGhzaXplPSIxLjIwMGVtIj4xPC9tbj48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij53X3sxfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6Izk5OTk5OTstLWx0eC1maWxsLWNvbG9yOiM5OTk5OTk7LS1sdHgtZmctY29sb3I6Izk5OTk5OTsiIGNvbG9yPSIjOTk5OTk5IiBmaWxsPSIjOTk5OTk5IiBzdHJva2U9IiM5OTk5OTkiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtMjEuMjggMTM1LjQ3KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6MS4xNmVtOy0tbHR4LWZvLWhlaWdodDowLjQ0ZW07LS1sdHgtZm8tZGVwdGg6MC4xNWVtO2ZvbnQtc2l6ZToxMS43NXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iOS42NCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNy4xNSkiIHdpZHRoPSIxOC45NCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjIucGljMS5tMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJ3X3syfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5OTk5OTsiIG1hdGhjb2xvcj0iIzk5OTk5OSIgbWF0aHNpemU9IjEuMjAwZW0iPnc8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5OTk5OTsiIG1hdGhjb2xvcj0iIzk5OTk5OSIgbWF0aHNpemU9IjEuMjAwZW0iPjI8L21uPjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPndfezJ9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDBCNDc4Oy0tbHR4LWZpbGwtY29sb3I6IzAwQjQ3ODstLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgY29sb3I9IiMwMEI0NzgiIGZpbGw9IiMwMEI0NzgiIHN0cm9rZT0iIzAwQjQ3OCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDc5LjE2IDY3LjcxKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6NC4zM2VtOy0tbHR4LWZvLWhlaWdodDowLjg1ZW07LS1sdHgtZm8tZGVwdGg6MC4yNWVtO2ZvbnQtc2l6ZToxMHB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTUuMjEiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDExLjc1KSIgd2lkdGg9IjU5Ljg4Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMi5waWMxLm0zIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRocm17c3Bhbn0oXG1hdGhiZntYfV97SX1ee1x0b3B9KSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIG1hdGhjb2xvcj0iIzAwQjQ3OCI+c3BhbjwvbWk+PG1vPuKBoTwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgbWF0aGNvbG9yPSIjMDBCNDc4IiBzdHJldGNoeT0iZmFsc2UiPig8L21vPjxtc3Vic3VwPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIG1hdGhjb2xvcj0iIzAwQjQ3OCI+8J2QlzwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgbWF0aGNvbG9yPSIjMDBCNDc4Ij5JPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBtYXRoY29sb3I9IiMwMEI0NzgiPuKKpDwvbW8+PC9tc3Vic3VwPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIG1hdGhjb2xvcj0iIzAwQjQ3OCIgc3RyZXRjaHk9ImZhbHNlIj4pPC9tbz48L21yb3c+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhybXtzcGFufShcbWF0aGJme1h9X3tJfV57XHRvcH0pPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNjQ2NDY0Oy0tbHR4LWZpbGwtY29sb3I6IzY0NjQ2NDstLWx0eC1mZy1jb2xvcjojNjQ2NDY0OyIgY29sb3I9IiM2NDY0NjQiIGZpbGw9IiM2NDY0NjQiIHN0cm9rZT0iIzY0NjQ2NCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDUzLjA0IC02MS4zKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6NC4zN2VtOy0tbHR4LWZvLWhlaWdodDowLjY0ZW07LS1sdHgtZm8tZGVwdGg6MC4yNmVtO2ZvbnQtc2l6ZTo4LjVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjEwLjY0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3LjU2KSIgd2lkdGg9IjUxLjQxIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMi5waWMxLm00IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoY2Fse0R9X3tcdGV4dHtwcmV0cmFpbn19IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojNjQ2NDY0OyIgY2xhc3M9Imx0eF9mb250X21hdGhjYWxpZ3JhcGhpYyIgbWF0aGNvbG9yPSIjNjQ2NDY0IiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2SnzwvbWk+PG10ZXh0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojNjQ2NDY0OyIgY2xhc3M9Imx0eF9tYXRodmFyaWFudF9ib2xkIiBtYXRoY29sb3I9IiM2NDY0NjQiIG1hdGhzaXplPSIwLjgwMGVtIj5wcmV0cmFpbjwvbXRleHQ+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhjYWx7RH1fe1x0ZXh0e3ByZXRyYWlufX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMEI0Nzg7LS1sdHgtZmlsbC1jb2xvcjojMDBCNDc4Oy0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBjb2xvcj0iIzAwQjQ3OCIgZmlsbD0iIzAwQjQ3OCIgc3Ryb2tlPSIjMDBCNDc4IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgLTExOC44IC00Mi40KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6Mi4xM2VtOy0tbHR4LWZvLWhlaWdodDowLjY0ZW07LS1sdHgtZm8tZGVwdGg6MC4xM2VtO2ZvbnQtc2l6ZTo4LjVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjkuMDciIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuNTYpIiB3aWR0aD0iMjUuMDEiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTUiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhjYWx7RH1fe1x0ZXh0e0ZUfX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBjbGFzcz0ibHR4X2ZvbnRfbWF0aGNhbGlncmFwaGljIiBtYXRoY29sb3I9IiMwMEI0NzgiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZKfPC9taT48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iIzAwQjQ3OCIgbWF0aHNpemU9IjAuODAwZW0iPkZUPC9tdGV4dD48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGNhbHtEfV97XHRleHR7RlR9fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIGNvbG9yPSIjMDAwMDAwIiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtOS4zIC0xNS4zOCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjAuNThlbTstLWx0eC1mby1oZWlnaHQ6MC42MWVtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6OC41cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI3LjEzIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3LjEzKSIgd2lkdGg9IjYuNzgiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTYiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhiZnswfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiIG1hdGhjb2xvcj0iIzAwMDAwMCIgbWF0aHNpemU9IjAuODAwZW0iPvCdn448L21uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZnswfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwNzhGRjstLWx0eC1maWxsLWNvbG9yOiMwMDc4RkY7LS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIGNvbG9yPSIjMDA3OEZGIiBmaWxsPSIjMDA3OEZGIiBzdHJva2U9IiMwMDc4RkYiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA2NS42MiA4OC45OCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjEuNjFlbTstLWx0eC1mby1oZWlnaHQ6MC44M2VtOy0tbHR4LWZvLWRlcHRoOjAuMTVlbTtmb250LXNpemU6MTEuNzVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjE2LjAxIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMy41MikiIHdpZHRoPSIyNi4yNSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjIucGljMS5tNyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJme1d9X3tJfV57MH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIxLjIwMGVtIj7wnZCWPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIxLjIwMGVtIj5JPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIxLjIwMGVtIj4wPC9tbj48L21zdWJzdXA+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tJfV57MH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmlsbC1jb2xvcjojMDBDODY0Oy0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBjb2xvcj0iIzAwQzg2NCIgZmlsbD0iIzAwQzg2NCIgc3Ryb2tlPSIjMDBDODY0IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgLTg4LjY2IC00Ny43OCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjIuMzllbTstLWx0eC1mby1oZWlnaHQ6MC43ZW07LS1sdHgtZm8tZGVwdGg6MC4xNWVtO2ZvbnQtc2l6ZToxMS43NXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTMuOTIiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDExLjQzKSIgd2lkdGg9IjM4LjgzIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMi5waWMxLm04IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe1x0ZXh0e0ZUfX1ee1xzdGFyfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3Vic3VwPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIG1hdGhjb2xvcj0iIzAwQzg2NCIgbWF0aHNpemU9IjEuMjAwZW0iPvCdkJY8L21pPjxtdGV4dCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIGNsYXNzPSJsdHhfbWF0aHZhcmlhbnRfYm9sZCIgbWF0aGNvbG9yPSIjMDBDODY0IiBtYXRoc2l6ZT0iMS4yMDBlbSI+RlQ8L210ZXh0PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIG1hdGhjb2xvcj0iIzAwQzg2NCIgbWF0aHNpemU9IjEuMjAwZW0iPuKLhjwvbW8+PC9tc3Vic3VwPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97XHRleHR7RlR9fV57XHN0YXJ9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDg5LjI3IDQwLjc2KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6Mi45MWVtOy0tbHR4LWZvLWhlaWdodDowLjc2ZW07LS1sdHgtZm8tZGVwdGg6MC4xM2VtO2ZvbnQtc2l6ZTo4LjVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjEwLjM4IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA4LjkpIiB3aWR0aD0iMzQuMjYiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTkiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhiZntXfV97SX1eezB9XG1hdGhjYWx7UH1fe0l9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1zdWJzdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2QljwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+STwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+MDwvbW4+PC9tc3Vic3VwPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY2xhc3M9Imx0eF9mb250X21hdGhjYWxpZ3JhcGhpYyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2SqzwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+STwvbWk+PC9tc3ViPjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoYmZ7V31fe0l9XnswfVxtYXRoY2Fse1B9X3tJfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwNzhGRjstLWx0eC1maWxsLWNvbG9yOiMwMDc4RkY7LS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIGNvbG9yPSIjMDA3OEZGIiBmaWxsPSIjMDA3OEZGIiBzdHJva2U9IiMwMDc4RkYiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtNzEuNTggMzkuMDgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDo0LjQ2ZW07LS1sdHgtZm8taGVpZ2h0OjAuNzZlbTstLWx0eC1mby1kZXB0aDowLjI0ZW07Zm9udC1zaXplOjguNXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTEuNjciIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDguOSkiIHdpZHRoPSI1Mi40MSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjIucGljMS5tMTAiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhiZntXfV97SX1eezB9KFxtYXRoYmZ7SX0tXG1hdGhjYWx7UH1fe0l9KSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93Pjxtc3Vic3VwPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPvCdkJY8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPkk8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPjA8L21uPjwvbXN1YnN1cD48bW8gbHNwYWNlPSIwZW0iIHJzcGFjZT0iMGVtIj7igIs8L21vPjxtcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF4c2l6ZT0iMC44MDBlbSIgbWluc2l6ZT0iMC44MDBlbSI+KDwvbW8+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2QiDwvbWk+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+4oiSPC9tbz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBjbGFzcz0ibHR4X2ZvbnRfbWF0aGNhbGlncmFwaGljIiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZKrPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj5JPC9taT48L21zdWI+PC9tcm93PjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF4c2l6ZT0iMC44MDBlbSIgbWluc2l6ZT0iMC44MDBlbSI+KTwvbW8+PC9tcm93PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoYmZ7V31fe0l9XnswfShcbWF0aGJme0l9LVxtYXRoY2Fse1B9X3tJfSk8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0I0MDBEQzsiIGNvbG9yPSIjQjQwMERDIiBzdHJva2Utd2lkdGg9IjIuMHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCNDAwREM7LS1sdHgtZmlsbC1jb2xvcjojQjQwMERDOyIgZmlsbD0iI0I0MDBEQyIgc3Ryb2tlPSIjQjQwMERDIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIC0zNi4yNyAxMS43MiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojQjQwMERDOy0tbHR4LWZpbGwtY29sb3I6I0I0MDBEQzsiIGZpbGw9IiNCNDAwREMiIHN0cm9rZT0iI0I0MDBEQyIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgtMC45NTE1OCAwLjMwNzQgLTAuMzA3NCAtMC45NTE1OCAtMzYuMjcgMTEuNzIpIj48cGF0aCBkPSJNIDkuNjkgMCBDIDguNSAwLjMgMy4yNyAxLjk4IDAgMy44MiBMIDAgLTMuODIgQyAzLjI3IC0xLjk4IDguNSAtMC4zIDkuNjkgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojQjQwMERDOy0tbHR4LWZpbGwtY29sb3I6I0I0MDBEQzstLWx0eC1mZy1jb2xvcjojQjQwMERDOyIgY29sb3I9IiNCNDAwREMiIGZpbGw9IiNCNDAwREMiIHN0cm9rZT0iI0I0MDBEQyIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC04My41OCAxNi42MSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjIuMWVtOy0tbHR4LWZvLWhlaWdodDowLjdlbTstLWx0eC1mby1kZXB0aDowLjI0ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxNS4zNCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTEuMzkpIiB3aWR0aD0iMzQuMTUiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTExIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe0xfezJ9fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0I0MDBEQzsiIG1hdGhjb2xvcj0iI0I0MDBEQyIgbWF0aHNpemU9IjEuMjAwZW0iPvCdkJY8L21pPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0I0MDBEQzsiIG1hdGhjb2xvcj0iI0I0MDBEQyIgbWF0aHNpemU9IjEuMjAwZW0iPkw8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0I0MDBEQzsiIG1hdGhjb2xvcj0iI0I0MDBEQyIgbWF0aHNpemU9IjEuMjAwZW0iPjI8L21uPjwvbXN1Yj48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tMX3syfX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGQUJBQjsiIGNvbG9yPSIjRkZBQkFCIiBzdHJva2Utd2lkdGg9IjEuNHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNGRkFCQUI7LS1sdHgtZmlsbC1jb2xvcjojRkZBQkFCOyIgZmlsbD0iI0ZGQUJBQiIgc3Ryb2tlPSIjRkZBQkFCIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIC02OS4yNiAtMC45OSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZBQkFCOy0tbHR4LWZpbGwtY29sb3I6I0ZGQUJBQjsiIGZpbGw9IiNGRkFCQUIiIHN0cm9rZT0iI0ZGQUJBQiIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgtMC45OTk5IC0wLjAxNDI4IDAuMDE0MjggLTAuOTk5OSAtNjkuMjYgLTAuOTkpIj48cGF0aCBkPSJNIDguMDMgMCBDIDcuMDQgMC4yNCAyLjcxIDEuNjMgMCAzLjE0IEwgMCAtMy4xNCBDIDIuNzEgLTEuNjMgNy4wNCAtMC4yNCA4LjAzIDAgWiIgLz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0ZGQUJBQjstLWx0eC1maWxsLWNvbG9yOiNGRkFCQUI7LS1sdHgtZmctY29sb3I6I0ZGQUJBQjsiIGNvbG9yPSIjRkZBQkFCIiBmaWxsPSIjRkZBQkFCIiBzdHJva2U9IiNGRkFCQUIiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtOTYuNjQgLTE3LjQyKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6Mi4zOWVtOy0tbHR4LWZvLWhlaWdodDowLjdlbTstLWx0eC1mby1kZXB0aDowLjE1ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMy44OCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTEuMzkpIiB3aWR0aD0iMzguODMiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTEyIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe1x0ZXh0e0ZUfX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkFCQUI7IiBtYXRoY29sb3I9IiNGRkFCQUIiIG1hdGhzaXplPSIxLjIwMGVtIj7wnZCWPC9taT48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkFCQUI7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iI0ZGQUJBQiIgbWF0aHNpemU9IjEuMjAwZW0iPkZUPC9tdGV4dD48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tcdGV4dHtGVH19PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojQjQwMERDOy0tbHR4LWZpbGwtY29sb3I6I0I0MDBEQzstLWx0eC1mZy1jb2xvcjojQjQwMERDOyIgY29sb3I9IiNCNDAwREMiIGZpbGw9IiNCNDAwREMiIHN0cm9rZT0iI0I0MDBEQyIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC0xMTUuOTkgLTEzNC42MikiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAyMi4xMikiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA4LjkpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAxMS4wNyAwKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjExLjc4ZW07LS1sdHgtZm8taGVpZ2h0OjAuNTdlbTstLWx0eC1mby1kZXB0aDowLjE2ZW07Zm9udC1zaXplOjkuOHB0OyIgaGVpZ2h0PSI5Ljg0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3LjY5KSIgd2lkdGg9IjE1OS43MiI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTMy5GMi5zZjIucGljMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo4MCU7Ij5JbnRlcnBvbGF0ZXMgYmV0d2VlbiA8L3NwYW4+PG1hdGggaWQ9IlMzLkYyLnNmMi5waWMxLm0xMyIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJme1d9X3tJfV57MH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1YnN1cD48bWkgbWF0aHNpemU9IjAuODAwZW0iPvCdkJY8L21pPjxtaSBtYXRoc2l6ZT0iMC44MDBlbSI+STwvbWk+PG1uIG1hdGhzaXplPSIwLjgwMGVtIj4wPC9tbj48L21zdWJzdXA+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tJfV57MH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTMy5GMi5zZjIucGljMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo4MCU7Ij4gYW5kIDwvc3Bhbj48bWF0aCBpZD0iUzMuRjIuc2YyLnBpYzEubTE0IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe1x0ZXh0e0ZUfX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgbWF0aHNpemU9IjAuODAwZW0iPvCdkJY8L21pPjxtdGV4dCBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhzaXplPSIwLjgwMGVtIj5GVDwvbXRleHQ+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97XHRleHR7RlR9fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfcm93IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDAgMTkuNykiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9jb2wgbHR4X25vcGFkX3IiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDoxNS44ZW07LS1sdHgtZm8taGVpZ2h0OjAuNTllbTstLWx0eC1mby1kZXB0aDowLjE2ZW07Zm9udC1zaXplOjEwLjY1cHQ7IiBoZWlnaHQ9IjExLjA3IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA4LjY1KSIgd2lkdGg9IjIzMi44Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IlMzLkYyLnNmMi5waWMxLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjkwJTsiPnJpZGdlIHNocmlua2FnZSBhZmZlY3RzIGFsbCBkaXJlY3Rpb25zCjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvZz48L2c+PC9nPjwvZz48L2c+PC9zdmc+)

(b) L2-SP: Blends all directions, no structure preservation.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjIuc2YzLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIyOTkuMjEiIG92ZXJmbG93PSJ2aXNpYmxlIiBzdHlsZT0idmVydGljYWwtYWxpZ246LTE0OS42MXB4IiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCAyOTkuMjEgMjk5LjIxIiB3aWR0aD0iMjk5LjIxIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDI5OS4yMSkgbWF0cml4KDEgMCAwIC0xIDAgMCkgdHJhbnNsYXRlKDE0OS42MSwwKSB0cmFuc2xhdGUoMCwxNDkuNjEpIj48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xMTguMTEgLTE0OS42MSBMIC0xMTguMTEgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgLTExOC4xMSBMIDE0OS42MSAtMTE4LjExIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC03OC43NCAtMTQ5LjYxIEwgLTc4Ljc0IDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIC03OC43NCBMIDE0OS42MSAtNzguNzQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0VDRUNFQzstLWx0eC1maWxsLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmctY29sb3I6I0VDRUNFQzsiIGNvbG9yPSIjRUNFQ0VDIiBmaWxsPSIjRUNFQ0VDIiBzdHJva2U9IiNFQ0VDRUMiIHN0cm9rZS13aWR0aD0iMC4zcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTM5LjM3IC0xNDkuNjEgTCAtMzkuMzcgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgLTM5LjM3IEwgMTQ5LjYxIC0zOS4zNyIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAzOS4zNyAtMTQ5LjYxIEwgMzkuMzcgMTQ5LjYxIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0xNDkuNjEgMzkuMzcgTCAxNDkuNjEgMzkuMzciIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0VDRUNFQzstLWx0eC1maWxsLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmctY29sb3I6I0VDRUNFQzsiIGNvbG9yPSIjRUNFQ0VDIiBmaWxsPSIjRUNFQ0VDIiBzdHJva2U9IiNFQ0VDRUMiIHN0cm9rZS13aWR0aD0iMC4zcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzguNzQgLTE0OS42MSBMIDc4Ljc0IDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIDc4Ljc0IEwgMTQ5LjYxIDc4Ljc0IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFQ0VDRUM7LS1sdHgtZmlsbC1jb2xvcjojRUNFQ0VDOy0tbHR4LWZnLWNvbG9yOiNFQ0VDRUM7IiBjb2xvcj0iI0VDRUNFQyIgZmlsbD0iI0VDRUNFQyIgc3Ryb2tlPSIjRUNFQ0VDIiBzdHJva2Utd2lkdGg9IjAuM3B0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDExOC4xMSAtMTQ5LjYxIEwgMTE4LjExIDE0OS42MSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRUNFQ0VDOy0tbHR4LWZpbGwtY29sb3I6I0VDRUNFQzstLWx0eC1mZy1jb2xvcjojRUNFQ0VDOyIgY29sb3I9IiNFQ0VDRUMiIGZpbGw9IiNFQ0VDRUMiIHN0cm9rZT0iI0VDRUNFQyIgc3Ryb2tlLXdpZHRoPSIwLjNwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTQ5LjYxIDExOC4xMSBMIDE0OS42MSAxMTguMTEiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiPjxnIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQjNCM0IzOyIgY29sb3I9IiNCM0IzQjMiIHN0cm9rZS13aWR0aD0iMC42cHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0IzQjNCMzstLWx0eC1maWxsLWNvbG9yOiNCM0IzQjM7IiBmaWxsPSIjQjNCM0IzIiBzdHJva2U9IiNCM0IzQjMiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTE0OS42MSAwIEwgMTQ4Ljc4IDAiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0IzQjNCMzstLWx0eC1maWxsLWNvbG9yOiNCM0IzQjM7IiBmaWxsPSIjQjNCM0IzIiBzdHJva2U9IiNCM0IzQjMiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTQ5LjE5IDApIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIC0zLjIxIDMuODIgQyAtMi42MiAxLjUzIC0xLjMyIDAuNDUgMCAwIEMgLTEuMzIgLTAuNDUgLTIuNjIgLTEuNTMgLTMuMjEgLTMuODIiIC8+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0IzQjNCMzsiIGNvbG9yPSIjQjNCM0IzIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCM0IzQjM7LS1sdHgtZmlsbC1jb2xvcjojQjNCM0IzOyIgZmlsbD0iI0IzQjNCMyIgc3Ryb2tlPSIjQjNCM0IzIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgLTE0OS42MSBMIDAgMTQ4Ljc4IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNCM0IzQjM7LS1sdHgtZmlsbC1jb2xvcjojQjNCM0IzOyIgZmlsbD0iI0IzQjNCMyIgc3Ryb2tlPSIjQjNCM0IzIiBzdHJva2UtZGFzaGFycmF5PSJub25lIiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgdHJhbnNmb3JtPSJtYXRyaXgoMC4wIDEuMCAtMS4wIDAuMCAwIDE0OS4xOSkiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTMuMjEgMy44MiBDIC0yLjYyIDEuNTMgLTEuMzIgMC40NSAwIDAgQyAtMS4zMiAtMC40NSAtMi42MiAtMS41MyAtMy4yMSAtMy44MiIgLz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6Izk5RTFDOTstLWx0eC1maWxsLWNvbG9yOiM5OUUxQzk7LS1sdHgtZmctY29sb3I6Izk5RTFDOTsiIGNvbG9yPSIjOTlFMUM5IiBmaWxsPSIjOTlFMUM5IiBzdHJva2U9IiM5OUUxQzkiIHN0cm9rZS13aWR0aD0iMi4ycHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gLTExOS4zMyAtNjguOSBMIDExOS4zMyA2OC45IiAvPjwvZz48ZyBzdHJva2Utd2lkdGg9IjAuNHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNFOEU4RTg7LS1sdHgtZmlsbC1jb2xvcjojRThFOEU4Oy0tbHR4LWZnLWNvbG9yOiNFOEU4RTg7IiBjb2xvcj0iI0U4RThFOCIgZmlsbD0iI0U4RThFOCIgc3Ryb2tlPSIjRThFOEU4Ij48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMCAwIE0gMTAyLjM2IDAgQyAxMDIuMzYgNTYuNTMgNTYuNTMgMTAyLjM2IDAgMTAyLjM2IEMgLTU2LjUzIDEwMi4zNiAtMTAyLjM2IDU2LjUzIC0xMDIuMzYgMCBDIC0xMDIuMzYgLTU2LjUzIC01Ni41MyAtMTAyLjM2IDAgLTEwMi4zNiBDIDU2LjUzIC0xMDIuMzYgMTAyLjM2IC01Ni41MyAxMDIuMzYgMCBaIE0gMCAwIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNDMUMxQzE7LS1sdHgtZmlsbC1jb2xvcjojQzFDMUMxOy0tbHR4LWZnLWNvbG9yOiNDMUMxQzE7IiBjb2xvcj0iI0MxQzFDMSIgZmlsbD0iI0MxQzFDMSIgc3Ryb2tlPSIjQzFDMUMxIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBNIDEwMi4zNiAwIEMgMTAyLjM2IDU2LjUzIDU2LjUzIDEwMi4zNiAwIDEwMi4zNiBDIC01Ni41MyAxMDIuMzYgLTEwMi4zNiA1Ni41MyAtMTAyLjM2IDAgQyAtMTAyLjM2IC01Ni41MyAtNTYuNTMgLTEwMi4zNiAwIC0xMDIuMzYgQyA1Ni41MyAtMTAyLjM2IDEwMi4zNiAtNTYuNTMgMTAyLjM2IDAgWiBNIDAgMCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojQ0NGMEU0Oy0tbHR4LWZpbGwtY29sb3I6I0NDRjBFNDstLWx0eC1mZy1jb2xvcjojQ0NGMEU0OyIgY29sb3I9IiNDQ0YwRTQiIGZpbGw9IiNDQ0YwRTQiIHN0cm9rZT0iI0NDRjBFNCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgMCBNIDEwMi4yOSA1OS4wNiBDIDk1Ljc2IDcwLjM1IDQ0LjY4IDUzLjA3IC0xMS44MSAyMC40NiBDIC02OC4zIC0xMi4xNiAtMTA4LjgxIC00Ny43NiAtMTAyLjI5IC01OS4wNiBDIC05NS43NiAtNzAuMzUgLTQ0LjY4IC01My4wNyAxMS44MSAtMjAuNDYgQyA2OC4zIDEyLjE2IDEwOC44MSA0Ny43NiAxMDIuMjkgNTkuMDYgWiBNIDAgMCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNjZEMkFFOy0tbHR4LWZpbGwtY29sb3I6IzY2RDJBRTstLWx0eC1mZy1jb2xvcjojNjZEMkFFOyIgY29sb3I9IiM2NkQyQUUiIGZpbGw9IiM2NkQyQUUiIHN0cm9rZT0iIzY2RDJBRSIgc3Ryb2tlLXdpZHRoPSIwLjdwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTSAxMDIuMjkgNTkuMDYgQyA5NS43NiA3MC4zNSA0NC42OCA1My4wNyAtMTEuODEgMjAuNDYgQyAtNjguMyAtMTIuMTYgLTEwOC44MSAtNDcuNzYgLTEwMi4yOSAtNTkuMDYgQyAtOTUuNzYgLTcwLjM1IC00NC42OCAtNTMuMDcgMTEuODEgLTIwLjQ2IEMgNjguMyAxMi4xNiAxMDguODEgNDcuNzYgMTAyLjI5IDU5LjA2IFogTSAwIDAiIC8+PC9nPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDAgTSAyLjM2IDAgQyAyLjM2IDEuMyAxLjMgMi4zNiAwIDIuMzYgQyAtMS4zIDIuMzYgLTIuMzYgMS4zIC0yLjM2IDAgQyAtMi4zNiAtMS4zIC0xLjMgLTIuMzYgMCAtMi4zNiBDIDEuMyAtMi4zNiAyLjM2IC0xLjMgMi4zNiAwIFogTSAwIDAiIC8+PGcgc3Ryb2tlLXdpZHRoPSIxLjhwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCA2MS45MyA3NS42OSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgwLjYzMzI0IDAuNzczOTYgLTAuNzczOTYgMC42MzMyNCA2MS45MyA3NS42OSkiPjxwYXRoIGQ9Ik0gOS4xMyAwIEMgOC4wMSAwLjI4IDMuMDggMS44NyAwIDMuNTkgTCAwIC0zLjU5IEMgMy4wOCAtMS44NyA4LjAxIC0wLjI4IDkuMTMgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIGZpbGwtb3BhY2l0eT0iMC43IiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjAuOHB0LDIuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1vcGFjaXR5PSIwLjciIHN0cm9rZS13aWR0aD0iMC44cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzAuODcgODYuNjEgTCA5MC42NiA1Mi4zNCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIGZpbGwtb3BhY2l0eT0iMC43IiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjAuOHB0LDIuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1vcGFjaXR5PSIwLjciIHN0cm9rZS13aWR0aD0iMC44cHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gNzAuODcgODYuNjEgTCAtMTkuNzkgMzQuMjciIC8+PC9nPjxnIHN0cm9rZS13aWR0aD0iMS4ycHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9IjMuMHB0LDMuMHB0IiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gMCAwIEwgODEuMzEgNDYuOTQiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMC44NjYwMSAwLjUgLTAuNSAwLjg2NjAxIDgxLjMxIDQ2Ljk0KSI+PHBhdGggZD0iTSA3LjQ3IDAgQyA2LjU1IDAuMjMgMi41MiAxLjUxIDAgMi45MSBMIDAgLTIuOTEgQyAyLjUyIC0xLjUxIDYuNTUgLTAuMjMgNy40NyAwIFoiIC8+PC9nPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjAuNyIgc3Ryb2tlLW9wYWNpdHk9IjAuNyIgc3Ryb2tlLXdpZHRoPSIwLjlwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojNERBMEZGOy0tbHR4LWZpbGwtY29sb3I6IzREQTBGRjstLWx0eC1mZy1jb2xvcjojNERBMEZGOyIgY29sb3I9IiM0REEwRkYiIGZpbGw9IiM0REEwRkYiIHN0cm9rZT0iIzREQTBGRiIgc3Ryb2tlLWRhc2hhcnJheT0iMy4wcHQsMi4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCAtMTUuMjIgMjYuMzciIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzREQTBGRjstLWx0eC1maWxsLWNvbG9yOiM0REEwRkY7LS1sdHgtZmctY29sb3I6IzREQTBGRjsiIGNvbG9yPSIjNERBMEZGIiBmaWxsPSIjNERBMEZGIiBzdHJva2U9IiM0REEwRkYiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuNTAwMDIgMC44NjYwMSAtMC44NjYwMSAtMC41MDAwMiAtMTUuMjIgMjYuMzcpIj48cGF0aCBkPSJNIDYuNjQgMCBDIDUuODMgMC4yIDIuMjQgMS4zNCAwIDIuNTcgTCAwIC0yLjU3IEMgMi4yNCAtMS4zNCA1LjgzIC0wLjIgNi42NCAwIFoiIC8+PC9nPjwvZz48ZyBzdHJva2Utd2lkdGg9IjEuOHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmlsbC1jb2xvcjojMDBDODY0Oy0tbHR4LWZnLWNvbG9yOiMwMEM4NjQ7IiBjb2xvcj0iIzAwQzg2NCIgZmlsbD0iIzAwQzg2NCIgc3Ryb2tlPSIjMDBDODY0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIC00OS4xNSAtMjguMzgiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwQzg2NDstLWx0eC1maWxsLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIGNvbG9yPSIjMDBDODY0IiBmaWxsPSIjMDBDODY0IiBzdHJva2U9IiMwMEM4NjQiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuODY2MDMgLTAuNSAwLjUgLTAuODY2MDMgLTQ5LjE1IC0yOC4zOCkiPjxwYXRoIGQ9Ik0gOS4xMyAwIEMgOC4wMSAwLjI4IDMuMDggMS44NyAwIDMuNTkgTCAwIC0zLjU5IEMgMy4wOCAtMS44NyA4LjAxIC0wLjI4IDkuMTMgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZDQjgwOy0tbHR4LWZpbGwtY29sb3I6I0ZGQ0I4MDstLWx0eC1mZy1jb2xvcjojRkZDQjgwOyIgY29sb3I9IiNGRkNCODAiIGZpbGw9IiNGRkNCODAiIHN0cm9rZT0iI0ZGQ0I4MCIgc3Ryb2tlLWRhc2hhcnJheT0iMC42cHQsMS4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCA5MC42NiA1Mi4zNCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZDQjgwOy0tbHR4LWZpbGwtY29sb3I6I0ZGQ0I4MDstLWx0eC1mZy1jb2xvcjojRkZDQjgwOyIgY29sb3I9IiNGRkNCODAiIGZpbGw9IiNGRkNCODAiIHN0cm9rZT0iI0ZGQ0I4MCIgc3Ryb2tlLWRhc2hhcnJheT0iMC42cHQsMS4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAwIDAgTCAtNjEuMzcgLTM1LjQzIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGQUIzMzsiIGNvbG9yPSIjRkZBQjMzIiBzdHJva2Utd2lkdGg9IjAuNnB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNGRkFCMzM7LS1sdHgtZmlsbC1jb2xvcjojRkZBQjMzOyIgZmlsbD0iI0ZGQUIzMyIgc3Ryb2tlPSIjRkZBQjMzIiBzdHJva2UtZGFzaGFycmF5PSIzLjBwdCwzLjBwdCIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0Ij48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIDguMTcgNC43MiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZBQjMzOy0tbHR4LWZpbGwtY29sb3I6I0ZGQUIzMzsiIGZpbGw9IiNGRkFCMzMiIHN0cm9rZT0iI0ZGQUIzMyIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NjAzIDAuNSAtMC41IDAuODY2MDMgOC4xNyA0LjcyKSI+PHBhdGggZD0iTSA1LjgxIDAgQyA1LjEgMC4xNyAxLjk2IDEuMTYgMCAyLjIzIEwgMCAtMi4yMyBDIDEuOTYgLTEuMTYgNS4xIC0wLjE3IDUuODEgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZENTk5Oy0tbHR4LWZpbGwtY29sb3I6I0ZGRDU5OTstLWx0eC1mZy1jb2xvcjojRkZENTk5OyIgY29sb3I9IiNGRkQ1OTkiIGZpbGw9IiNGRkQ1OTkiIHN0cm9rZT0iI0ZGRDU5OSIgc3Ryb2tlLWRhc2hhcnJheT0iMy4wcHQsMy4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAtMTkuNzkgMzQuMjcgTCAtNS4xNSA0Mi43MyIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkZENTk5Oy0tbHR4LWZpbGwtY29sb3I6I0ZGRDU5OTstLWx0eC1mZy1jb2xvcjojRkZENTk5OyIgY29sb3I9IiNGRkQ1OTkiIGZpbGw9IiNGRkQ1OTkiIHN0cm9rZT0iI0ZGRDU5OSIgc3Ryb2tlLWRhc2hhcnJheT0iMy4wcHQsMy4wcHQiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLXdpZHRoPSIwLjZwdCI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAxNC42NCA4LjQ1IEwgLTUuMTUgNDIuNzMiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6Izk5OTk5OTstLWx0eC1maWxsLWNvbG9yOiM5OTk5OTk7LS1sdHgtZmctY29sb3I6Izk5OTk5OTsiIGNvbG9yPSIjOTk5OTk5IiBmaWxsPSIjOTk5OTk5IiBzdHJva2U9IiM5OTk5OTkiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxMjguMzMgLTE0LjE0KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6MS4xNmVtOy0tbHR4LWZvLWhlaWdodDowLjQ0ZW07LS1sdHgtZm8tZGVwdGg6MC4xNWVtO2ZvbnQtc2l6ZToxMS43NXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iOS42NCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNy4xNSkiIHdpZHRoPSIxOC45NCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjMucGljMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJ3X3sxfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5OTk5OTsiIG1hdGhjb2xvcj0iIzk5OTk5OSIgbWF0aHNpemU9IjEuMjAwZW0iPnc8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6Izk5OTk5OTsiIG1hdGhjb2xvcj0iIzk5OTk5OSIgbWF0aHNpemU9IjEuMjAwZW0iPjE8L21uPjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPndfezF9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojOTk5OTk5Oy0tbHR4LWZpbGwtY29sb3I6Izk5OTk5OTstLWx0eC1mZy1jb2xvcjojOTk5OTk5OyIgY29sb3I9IiM5OTk5OTkiIGZpbGw9IiM5OTk5OTkiIHN0cm9rZT0iIzk5OTk5OSIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC0yMS4yOCAxMzUuNDcpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDoxLjE2ZW07LS1sdHgtZm8taGVpZ2h0OjAuNDRlbTstLWx0eC1mby1kZXB0aDowLjE1ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI5LjY0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA3LjE1KSIgd2lkdGg9IjE4Ljk0Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMy5waWMxLm0yIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IndfezJ9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk5OTk5OyIgbWF0aGNvbG9yPSIjOTk5OTk5IiBtYXRoc2l6ZT0iMS4yMDBlbSI+dzwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojOTk5OTk5OyIgbWF0aGNvbG9yPSIjOTk5OTk5IiBtYXRoc2l6ZT0iMS4yMDBlbSI+MjwvbW4+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+d197Mn08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMEI0Nzg7LS1sdHgtZmlsbC1jb2xvcjojMDBCNDc4Oy0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBjb2xvcj0iIzAwQjQ3OCIgZmlsbD0iIzAwQjQ3OCIgc3Ryb2tlPSIjMDBCNDc4IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNzkuMTYgNjcuNzEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDo0LjMzZW07LS1sdHgtZm8taGVpZ2h0OjAuODVlbTstLWx0eC1mby1kZXB0aDowLjI1ZW07Zm9udC1zaXplOjEwcHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxNS4yMSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTEuNzUpIiB3aWR0aD0iNTkuODgiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YzLnBpYzEubTMiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhybXtzcGFufShcbWF0aGJme1h9X3tJfV57XHRvcH0pIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgbWF0aGNvbG9yPSIjMDBCNDc4Ij5zcGFuPC9taT48bW8+4oGhPC9tbz48bXJvdz48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBtYXRoY29sb3I9IiMwMEI0NzgiIHN0cmV0Y2h5PSJmYWxzZSI+KDwvbW8+PG1zdWJzdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgbWF0aGNvbG9yPSIjMDBCNDc4Ij7wnZCXPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMEI0Nzg7IiBtYXRoY29sb3I9IiMwMEI0NzgiPkk8L21pPjxtbyBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIG1hdGhjb2xvcj0iIzAwQjQ3OCI+4oqkPC9tbz48L21zdWJzdXA+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBCNDc4OyIgbWF0aGNvbG9yPSIjMDBCNDc4IiBzdHJldGNoeT0iZmFsc2UiPik8L21vPjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aHJte3NwYW59KFxtYXRoYmZ7WH1fe0l9XntcdG9wfSk8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiM2NDY0NjQ7LS1sdHgtZmlsbC1jb2xvcjojNjQ2NDY0Oy0tbHR4LWZnLWNvbG9yOiM2NDY0NjQ7IiBjb2xvcj0iIzY0NjQ2NCIgZmlsbD0iIzY0NjQ2NCIgc3Ryb2tlPSIjNjQ2NDY0IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNTMuMDQgLTYxLjMpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDo0LjM3ZW07LS1sdHgtZm8taGVpZ2h0OjAuNjRlbTstLWx0eC1mby1kZXB0aDowLjI2ZW07Zm9udC1zaXplOjguNXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTAuNjQiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuNTYpIiB3aWR0aD0iNTEuNDEiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YzLnBpYzEubTQiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhjYWx7RH1fe1x0ZXh0e3ByZXRyYWlufX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM2NDY0NjQ7IiBjbGFzcz0ibHR4X2ZvbnRfbWF0aGNhbGlncmFwaGljIiBtYXRoY29sb3I9IiM2NDY0NjQiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZKfPC9taT48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiM2NDY0NjQ7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iIzY0NjQ2NCIgbWF0aHNpemU9IjAuODAwZW0iPnByZXRyYWluPC9tdGV4dD48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGNhbHtEfV97XHRleHR7cHJldHJhaW59fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwQjQ3ODstLWx0eC1maWxsLWNvbG9yOiMwMEI0Nzg7LS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIGNvbG9yPSIjMDBCNDc4IiBmaWxsPSIjMDBCNDc4IiBzdHJva2U9IiMwMEI0NzgiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtMTE4LjggLTQyLjQpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDoyLjEzZW07LS1sdHgtZm8taGVpZ2h0OjAuNjRlbTstLWx0eC1mby1kZXB0aDowLjEzZW07Zm9udC1zaXplOjguNXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iOS4wNyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNy41NikiIHdpZHRoPSIyNS4wMSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjMucGljMS5tNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGNhbHtEfV97XHRleHR7RlR9fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIGNsYXNzPSJsdHhfZm9udF9tYXRoY2FsaWdyYXBoaWMiIG1hdGhjb2xvcj0iIzAwQjQ3OCIgbWF0aHNpemU9IjAuODAwZW0iPvCdkp88L21pPjxtdGV4dCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwQjQ3ODsiIGNsYXNzPSJsdHhfbWF0aHZhcmlhbnRfYm9sZCIgbWF0aGNvbG9yPSIjMDBCNDc4IiBtYXRoc2l6ZT0iMC44MDBlbSI+RlQ8L210ZXh0PjwvbXN1Yj48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoY2Fse0R9X3tcdGV4dHtGVH19PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDstLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgY29sb3I9IiMwMDAwMDAiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC05LjMgLTE1LjM4KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6MC41OGVtOy0tbHR4LWZvLWhlaWdodDowLjYxZW07LS1sdHgtZm8tZGVwdGg6MGVtO2ZvbnQtc2l6ZTo4LjVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjcuMTMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuMTMpIiB3aWR0aD0iNi43OCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjMucGljMS5tNiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJmezB9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyIgbWF0aGNvbG9yPSIjMDAwMDAwIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2fjjwvbW4+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJmezB9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDY1LjYyIDg4Ljk4KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6MS42MWVtOy0tbHR4LWZvLWhlaWdodDowLjgzZW07LS1sdHgtZm8tZGVwdGg6MC4xNWVtO2ZvbnQtc2l6ZToxMS43NXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTYuMDEiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEzLjUyKSIgd2lkdGg9IjI2LjI1Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMy5waWMxLm03IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxtYXRoYmZ7V31fe0l9XnswfSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3Vic3VwPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjEuMjAwZW0iPvCdkJY8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjEuMjAwZW0iPkk8L21pPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjEuMjAwZW0iPjA8L21uPjwvbXN1YnN1cD48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxtYXRoYmZ7V31fe0l9XnswfTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwQzg2NDstLWx0eC1maWxsLWNvbG9yOiMwMEM4NjQ7LS1sdHgtZmctY29sb3I6IzAwQzg2NDsiIGNvbG9yPSIjMDBDODY0IiBmaWxsPSIjMDBDODY0IiBzdHJva2U9IiMwMEM4NjQiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtODguNjYgLTQ3Ljc4KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6Mi4zOWVtOy0tbHR4LWZvLWhlaWdodDowLjdlbTstLWx0eC1mby1kZXB0aDowLjE1ZW07Zm9udC1zaXplOjExLjc1cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMy45MiIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTEuNDMpIiB3aWR0aD0iMzguODMiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YzLnBpYzEubTgiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iXG1hdGhiZntXfV97XHRleHR7RlR9fV57XHN0YXJ9IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWJzdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBDODY0OyIgbWF0aGNvbG9yPSIjMDBDODY0IiBtYXRoc2l6ZT0iMS4yMDBlbSI+8J2QljwvbWk+PG10ZXh0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBDODY0OyIgY2xhc3M9Imx0eF9tYXRodmFyaWFudF9ib2xkIiBtYXRoY29sb3I9IiMwMEM4NjQiIG1hdGhzaXplPSIxLjIwMGVtIj5GVDwvbXRleHQ+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDBDODY0OyIgbWF0aGNvbG9yPSIjMDBDODY0IiBtYXRoc2l6ZT0iMS4yMDBlbSI+4ouGPC9tbz48L21zdWJzdXA+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbWF0aGJme1d9X3tcdGV4dHtGVH19Xntcc3Rhcn08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDc4RkY7LS1sdHgtZmlsbC1jb2xvcjojMDA3OEZGOy0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBjb2xvcj0iIzAwNzhGRiIgZmlsbD0iIzAwNzhGRiIgc3Ryb2tlPSIjMDA3OEZGIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgODkuMjcgNDAuNzYpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDstLWx0eC1mby13aWR0aDoyLjkxZW07LS1sdHgtZm8taGVpZ2h0OjAuNzZlbTstLWx0eC1mby1kZXB0aDowLjEzZW07Zm9udC1zaXplOjguNXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTAuMzgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDguOSkiIHdpZHRoPSIzNC4yNiI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxtYXRoIGlkPSJTMy5GMi5zZjMucGljMS5tOSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJme1d9X3tJfV57MH1cbWF0aGNhbHtQfV97SX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bXN1YnN1cD48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZCWPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj5JPC9taT48bW4gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj4wPC9tbj48L21zdWJzdXA+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXN1Yj48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBjbGFzcz0ibHR4X2ZvbnRfbWF0aGNhbGlncmFwaGljIiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZKrPC9taT48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj5JPC9taT48L21zdWI+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97SX1eezB9XG1hdGhjYWx7UH1fe0l9PC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDA3OEZGOy0tbHR4LWZpbGwtY29sb3I6IzAwNzhGRjstLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgY29sb3I9IiMwMDc4RkYiIGZpbGw9IiMwMDc4RkYiIHN0cm9rZT0iIzAwNzhGRiIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC03MS41OCAzOS4wOCkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjQuNDZlbTstLWx0eC1mby1oZWlnaHQ6MC43NmVtOy0tbHR4LWZvLWRlcHRoOjAuMjRlbTtmb250LXNpemU6OC41cHQ7IiBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMS42NyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOC45KSIgd2lkdGg9IjUyLjQxIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMy5waWMxLm0xMCIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJme1d9X3tJfV57MH0oXG1hdGhiZntJfS1cbWF0aGNhbHtQfV97SX0pIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1yb3c+PG1zdWJzdXA+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+8J2QljwvbWk+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+STwvbWk+PG1uIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXRoc2l6ZT0iMC44MDBlbSI+MDwvbW4+PC9tc3Vic3VwPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG1yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXhzaXplPSIwLjgwMGVtIiBtaW5zaXplPSIwLjgwMGVtIj4oPC9tbz48bXJvdz48bWkgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj7wnZCIPC9taT48bW8gc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDc4RkY7IiBtYXRoY29sb3I9IiMwMDc4RkYiIG1hdGhzaXplPSIwLjgwMGVtIj7iiJI8L21vPjxtc3ViPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIGNsYXNzPSJsdHhfZm9udF9tYXRoY2FsaWdyYXBoaWMiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPvCdkqs8L21pPjxtaSBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwNzhGRjsiIG1hdGhjb2xvcj0iIzAwNzhGRiIgbWF0aHNpemU9IjAuODAwZW0iPkk8L21pPjwvbXN1Yj48L21yb3c+PG1vIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDA3OEZGOyIgbWF0aGNvbG9yPSIjMDA3OEZGIiBtYXhzaXplPSIwLjgwMGVtIiBtaW5zaXplPSIwLjgwMGVtIj4pPC9tbz48L21yb3c+PC9tcm93Pjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97SX1eezB9KFxtYXRoYmZ7SX0tXG1hdGhjYWx7UH1fe0l9KTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0ZGOTYwMDstLWx0eC1maWxsLWNvbG9yOiNGRjk2MDA7LS1sdHgtZmctY29sb3I6I0ZGOTYwMDsiIGNvbG9yPSIjRkY5NjAwIiBmaWxsPSIjRkY5NjAwIiBzdHJva2U9IiNGRjk2MDAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtMC4wNyAtNy4yKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7LS1sdHgtZm8td2lkdGg6My4xN2VtOy0tbHR4LWZvLWhlaWdodDowLjY1ZW07LS1sdHgtZm8tZGVwdGg6MGVtO2ZvbnQtc2l6ZTo4LjVwdDsiIGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjcuNjkiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuNjkpIiB3aWR0aD0iMzcuMjkiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48bWF0aCBpZD0iUzMuRjIuc2YzLnBpYzEubTExIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxmcmFjezF9ezJ9XHRleHR7LW1peH0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXJvdz48bWZyYWMgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjk2MDA7IiBtYXRoY29sb3I9IiNGRjk2MDAiPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGOTYwMDsiIG1hdGhjb2xvcj0iI0ZGOTYwMCIgbWF0aHNpemU9IjAuODAwZW0iPjE8L21uPjxtbiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGOTYwMDsiIG1hdGhjb2xvcj0iI0ZGOTYwMCIgbWF0aHNpemU9IjAuODAwZW0iPjI8L21uPjwvbWZyYWM+PG1vIGxzcGFjZT0iMGVtIiByc3BhY2U9IjBlbSI+4oCLPC9tbz48bXRleHQgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjk2MDA7IiBjbGFzcz0ibHR4X21hdGh2YXJpYW50X2JvbGQiIG1hdGhjb2xvcj0iI0ZGOTYwMCIgbWF0aHNpemU9IjAuODAwZW0iPi1taXg8L210ZXh0PjwvbXJvdz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxmcmFjezF9ezJ9XHRleHR7LW1peH08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGOTYwMDsiIGNvbG9yPSIjRkY5NjAwIiBzdHJva2Utd2lkdGg9IjIuMHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNGRjk2MDA7LS1sdHgtZmlsbC1jb2xvcjojRkY5NjAwOyIgZmlsbD0iI0ZGOTYwMCIgc3Ryb2tlPSIjRkY5NjAwIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDAgMCBMIC0zLjMzIDI3LjYyIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiNGRjk2MDA7LS1sdHgtZmlsbC1jb2xvcjojRkY5NjAwOyIgZmlsbD0iI0ZGOTYwMCIgc3Ryb2tlPSIjRkY5NjAwIiBzdHJva2UtZGFzaGFycmF5PSJub25lIiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIHRyYW5zZm9ybT0ibWF0cml4KC0wLjExOTYxIDAuOTkyODEgLTAuOTkyODEgLTAuMTE5NjEgLTMuMzMgMjcuNjIpIj48cGF0aCBkPSJNIDkuNjkgMCBDIDguNSAwLjMgMy4yNyAxLjk4IDAgMy44MiBMIDAgLTMuODIgQyAzLjI3IC0xLjk4IDguNSAtMC4zIDkuNjkgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojRkY5NjAwOy0tbHR4LWZpbGwtY29sb3I6I0ZGOTYwMDstLWx0eC1mZy1jb2xvcjojRkY5NjAwOyIgY29sb3I9IiNGRjk2MDAiIGZpbGw9IiNGRjk2MDAiIHN0cm9rZT0iI0ZGOTYwMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC0xNi42NyA0Ni4xNSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOy0tbHR4LWZvLXdpZHRoOjIuMzllbTstLWx0eC1mby1oZWlnaHQ6MC43ZW07LS1sdHgtZm8tZGVwdGg6MC4xNWVtO2ZvbnQtc2l6ZToxMS43NXB0OyIgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMTMuODgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDExLjM5KSIgd2lkdGg9IjM4Ljc5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMy5waWMxLm0xMiIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbWF0aGJme1d9X3tcdGV4dHtTRH19IiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1zdWI+PG1pIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkY5NjAwOyIgbWF0aGNvbG9yPSIjRkY5NjAwIiBtYXRoc2l6ZT0iMS4yMDBlbSI+8J2QljwvbWk+PG10ZXh0IHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkY5NjAwOyIgY2xhc3M9Imx0eF9tYXRodmFyaWFudF9ib2xkIiBtYXRoY29sb3I9IiNGRjk2MDAiIG1hdGhzaXplPSIxLjIwMGVtIj5TRDwvbXRleHQ+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XG1hdGhiZntXfV97XHRleHR7U0R9fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6I0ZGOTYwMDstLWx0eC1maWxsLWNvbG9yOiNGRjk2MDA7LS1sdHgtZmctY29sb3I6I0ZGOTYwMDsiIGNvbG9yPSIjRkY5NjAwIiBmaWxsPSIjRkY5NjAwIiBzdHJva2U9IiNGRjk2MDAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAtODguOTIgLTEzNC43MikiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAyMC4wNSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA4LjMpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MTIuMDdlbTstLWx0eC1mby1oZWlnaHQ6MC41N2VtOy0tbHR4LWZvLWRlcHRoOjAuMTNlbTtmb250LXNpemU6OS44cHQ7IiBoZWlnaHQ9IjkuNDQiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuNjkpIiB3aWR0aD0iMTYzLjY1Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IlMzLkYyLnNmMy5waWMxLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjgwJTsiPlByZXNlcnZlIDwvc3Bhbj48bWF0aCBpZD0iUzMuRjIuc2YzLnBpYzEubTEzIiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxwZXJwIiBkaXNwbGF5PSJpbmxpbmUiIGludGVudD0iOmxpdGVyYWwiPjxzZW1hbnRpY3M+PG1vIG1hdGhzaXplPSIwLjgwMGVtIj7in4I8L21vPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+XHBlcnA8L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjxzcGFuIGlkPSJTMy5GMi5zZjMucGljMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9ImZvbnQtc2l6ZTo4MCU7Ij4gKyBjb252ZXggbWl4IGluIDwvc3Bhbj48bWF0aCBpZD0iUzMuRjIuc2YzLnBpYzEubTE0IiBjbGFzcz0ibHR4X01hdGgiIGFsdHRleHQ9IlxwYXJhbGxlbCIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtbyBtYXRoc2l6ZT0iMC44MDBlbSI+4oilPC9tbz48YW5ub3RhdGlvbiBlbmNvZGluZz0iYXBwbGljYXRpb24veC10ZXgiPlxwYXJhbGxlbDwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfcm93IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDAgMTguNzYpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAxMy43NiAwKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjExLjA5ZW07LS1sdHgtZm8taGVpZ2h0OjAuNTdlbTstLWx0eC1mby1kZXB0aDowLjFlbTtmb250LXNpemU6OS44cHQ7IiBoZWlnaHQ9IjguOTgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDcuNjkpIiB3aWR0aD0iMTUwLjMyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PG1hdGggaWQ9IlMzLkYyLnNmMy5waWMxLm0xNSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJcbGFtYmRhPTFcUmlnaHRhcnJvd1xmcmFje1xsYW1iZGF9ezF7K31cbGFtYmRhfT1cZnJhY3sxfXsxeyt9XGxhbWJkYX09XGZyYWN7MX17Mn1cdGV4dHstbWl4fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtcm93PjxtaSBtYXRoc2l6ZT0iMC44MDBlbSI+zrs8L21pPjxtbyBtYXRoc2l6ZT0iMC44MDBlbSI+PTwvbW8+PG1uIG1hdGhzaXplPSIwLjgwMGVtIj4xPC9tbj48bW8gbWF0aHNpemU9IjAuODAwZW0iIHN0cmV0Y2h5PSJmYWxzZSI+4oeSPC9tbz48bWZyYWM+PG1pIG1hdGhzaXplPSIwLjgwMGVtIj7OuzwvbWk+PG1yb3c+PG1uIG1hdGhzaXplPSIwLjgwMGVtIj4xPC9tbj48bW8gbWF0aHNpemU9IjAuODAwZW0iPis8L21vPjxtaSBtYXRoc2l6ZT0iMC44MDBlbSI+zrs8L21pPjwvbXJvdz48L21mcmFjPjxtbyBtYXRoc2l6ZT0iMC44MDBlbSI+PTwvbW8+PG1mcmFjPjxtbiBtYXRoc2l6ZT0iMC44MDBlbSI+MTwvbW4+PG1yb3c+PG1uIG1hdGhzaXplPSIwLjgwMGVtIj4xPC9tbj48bW8gbWF0aHNpemU9IjAuODAwZW0iPis8L21vPjxtaSBtYXRoc2l6ZT0iMC44MDBlbSI+zrs8L21pPjwvbXJvdz48L21mcmFjPjxtbyBtYXRoc2l6ZT0iMC44MDBlbSI+PTwvbW8+PG1yb3c+PG1mcmFjPjxtbiBtYXRoc2l6ZT0iMC44MDBlbSI+MTwvbW4+PG1uIG1hdGhzaXplPSIwLjgwMGVtIj4yPC9tbj48L21mcmFjPjxtbyBsc3BhY2U9IjBlbSIgcnNwYWNlPSIwZW0iPuKAizwvbW8+PG10ZXh0IGNsYXNzPSJsdHhfbWF0aHZhcmlhbnRfYm9sZCIgbWF0aHNpemU9IjAuODAwZW0iPi1taXg8L210ZXh0PjwvbXJvdz48L21yb3c+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5cbGFtYmRhPTFcUmlnaHRhcnJvd1xmcmFje1xsYW1iZGF9ezF7K31cbGFtYmRhfT1cZnJhY3sxfXsxeyt9XGxhbWJkYX09XGZyYWN7MX17Mn1cdGV4dHstbWl4fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PHNwYW4gaWQ9IlMzLkYyLnNmMy5waWMxLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iZm9udC1zaXplOjgwJTsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L2c+PC9nPjwvZz48L2c+PC9nPjwvc3ZnPg==)

(c) SD: Preserves orthogonal, mixes parallel components.

Figure 2: Geometric interpretation of finetuning strategies in 2D weight
space. The green line represents
$`\mathrm{span}(\mathbf{X}_{I}^{\top})`$, the subspace where finetuning
data concentrates. Starting from pretrained weights
$`\mathbf{W}_{I}^{0}`$ (blue), each method combines the orthogonal
component $`\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})`$ and the new
task solution
$`\mathbf{W}_{\text{FT}}^{\star}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$
(green) differently: (a) Direct FT preserves the orthogonal component
and replaces the parallel component entirely; (b) L2-SP creates a global
blend without clean structural decomposition; (c) Static
Self-Distillation preserves the orthogonal component and forms a convex
combination of the parallel components (shown with $`\lambda=1`$ giving
equal weighting).

Geometric Interpretation: As visualized in
Figure [2](#S3.F2 "Figure 2 ‣ 3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
Direct Finetuning discards pretrained knowledge within the finetuning
data subspace, replacing it with the new task solution, while preserving
orthogonal components. $`L_{2}`$ regularization shrinks the entire
solution towards the pretrained weights, leading to a complex,
non-surgical blend.

Self-Distillation achieves a nuanced trade-off: it preserves pretrained
knowledge in the subspace orthogonal to the finetuning data
($`\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})`$), and within the
task-relevant subspace, it computes a convex combination of the
projected pretrained weights and the optimal solution for the new task.
This enables control over knowledge retention and adaptation (further
details in
§[C.4](#A3.SS4 "C.4 Geometric Interpretation of Solutions ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).

### 3.4 Dynamic Self-Distillation with a WMA Teacher

Why static SD is biased. Restricted to the task subspace, the static-SD
solution in
Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2 (Unified Framework for Contrastive Finetuning Solutions). ‣ 3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
is the convex combination
$`\mathbf{W}_{SD}\mathcal{P}_{I}=\tfrac{\lambda}{1+\lambda}\mathbf{W}_{I}^{0}\mathcal{P}_{I}+\tfrac{1}{1+\lambda}\mathbf{W}^{\star}_{\text{FT}}`$,
where
$`\mathbf{W}^{\star}_{\text{FT}}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$
is the minimum-norm task solution. Because the anchor is *fixed* at
$`\mathbf{W}_{I}^{0}`$, for any finite $`\lambda>0`$ the solution stays
*offset* from $`\mathbf{W}^{\star}_{\text{FT}}`$ in the task subspace by
exactly
$`\tfrac{\lambda}{1+\lambda}(\mathbf{W}_{I}^{0}\mathcal{P}_{I}-\mathbf{W}^{\star}_{\text{FT}})`$,
an anchor bias that cannot be removed by tuning $`\lambda`$ alone
(smaller $`\lambda`$ reduces the bias but weakens orthogonal
preservation; larger $`\lambda`$ does the reverse). Geometrically
(Figure [2](#S3.F2 "Figure 2 ‣ 3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")),
static SD lies on the segment between
$`\mathbf{W}_{I}^{0}\mathcal{P}_{I}`$ and
$`\mathbf{W}^{\star}_{\text{FT}}`$ rather than reaching the task
optimum.

Intuition: static SD vs. EMA vs. WMA. The teacher choice controls
*where* the regularizer is anchored along the trajectory: *Static SD*
fixes the anchor at the start ($`\mathbf{W}_{I}^{0}`$) and is biased
toward initialization; an *EMA teacher* stays near the current student
state, so its regularizing signal vanishes as training converges
(precisely when OOD robustness is most fragile); a *WMA teacher* stays
near a *trajectory-weighted consensus* of the optimization path: it
remembers the start but adapts over time, and with a suitable kernel
retains meaningful mass at *both* ends of training. WMA thus addresses
two limitations at once: (i) the static-SD anchor bias and (ii) the EMA
collapse of the teacher–student gap. We instantiate it as a *dynamic*
teacher that evolves as a WMA of the student’s trajectory, illustrated
at the system level in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
with the formal definition below.

###### Definition 3.3 (WMA Teacher).

The WMA teacher $`\mathbf{W}_{\text{Teacher}}^{t}`$ averages student
states $`\mathbf{W}_{I}^{k}`$ over time $`k=0,\dots,t`$, weighted by a
kernel $`\kappa(\tau_{k})`$ on normalized time
$`\tau_{k}=\frac{k+c_{1}}{T+c_{2}}\in(0,1)`$. The offsets
$`c_{1},c_{2}>0`$ keep $`\tau_{k}`$ strictly inside $`(0,1)`$ (avoiding
$`\tau_{0}=0`$ and $`\tau_{T}=1`$), which is required for kernels such
as $`\mathrm{Beta}(0.5,0.5)`$ that diverge at the endpoints; we use
$`c_{1}=0.5,\ c_{2}=1`$ in all experiments. The online recursion is:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\omega_{t}\;`$ | $`\displaystyle=\;\frac{\kappa(\tau_{t})}{\sum_{j=0}^{t}\kappa(\tau_{j})},`$ |  |
|  | $`\displaystyle\mathbf{W}_{\text{Teacher}}^{t}\;`$ | $`\displaystyle=\;(1-\omega_{t})\,\mathbf{W}_{\text{Teacher}}^{t-1}+\omega_{t}\,\mathbf{W}_{I}^{t},`$ |  |
|  | $`\displaystyle\mathbf{W}_{\text{Teacher}}^{0}\;`$ | $`\displaystyle=\;\mathbf{W}_{I}^{0}.`$ |  |

Persistent Regularization and Bias-Free Convergence: Unlike an
Exponential Moving Average (EMA) teacher, whose regularizing influence
vanishes as it converges to the student, the WMA teacher (especially
with a U-shaped kernel like Beta(0.5,0.5)) maintains a persistent
regularizing force over finite training horizons (see
§[C.5.2](#A3.SS5.SSS2 "C.5.2 The Persistent Regularizer of the WMA Teacher ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
This force continuously pulls the student towards its robust pretrained
initialization. We prove that this adaptive anchoring achieves bias-free
convergence to the task-optimal solution within the finetuning subspace:

###### Theorem 3.4 (Bias-Free Convergence in the Task Subspace).

Let
$`\mathbf{W}^{\star}_{\text{FT}}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$
denote the *minimum-norm task solution* (i.e., the
minimum-Frobenius-norm minimizer
of equation [2](#S3.E2 "Equation 2 ‣ 3.2 Loss Reformulation via the Contrastive Target Matrix ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
The WMA teacher’s projection onto the task subspace converges to
$`\mathbf{W}^{\star}_{\text{FT}}\mathcal{P}_{I}`$, and consequently, the
student’s projection also converges to
$`\mathbf{W}^{\star}_{\text{FT}}\mathcal{P}_{I}`$.

We use the name *minimum-norm task solution* for
$`\mathbf{W}^{\star}_{\text{FT}}`$ to distinguish it from the Direct FT
solution in
Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2 (Unified Framework for Contrastive Finetuning Solutions). ‣ 3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
which additionally retains the orthogonal component
$`\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})`$:
$`\mathbf{W}^{\star}_{\text{FT}}`$ is the task-subspace target a
preservation-aware method should reach *within*
$`\mathrm{range}(\mathbf{X}_{I})`$. The theorem shows that dynamic
teachers eliminate the static anchor bias while preserving orthogonal
knowledge (formal proof in
§[C.5](#A3.SS5 "C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).

## 4 Methodology: TRACER

Guided by these geometric insights and the proven benefits of a WMA
teacher in the above section, we propose TRACER, a novel finetuning
method for multimodal models. TRACER combines the standard symmetric
InfoNCE loss with dynamic self-distillation guided by a WMA teacher, as
illustrated in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

The total training objective for TRACER is:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{TRACER}}=\mathcal{L}_{\text{MMCL}}+\lambda_{\text{SD}}\,\mathcal{L}_{\text{SD-WMA}}.
``` |  | (3) |

##### Multi-Modal Contrastive Loss ($`\mathcal{L}_{\text{MMCL}}`$):

This is the primary finetuning loss, typically a symmetric InfoNCE
objective. In our implementation, we also include a cross-Frobenius
regularizer to prevent embedding collapse (standard CLIP finetuning
recipe). This component drives the student model to learn new
task-specific alignments.

Figure 3: Toy Experiment. We compare a pretrained model against four
finetuning methods on a finetuning task. (a) Performance on the original
MNIST and new Colored MNIST task. All finetuning methods successfully
learn the new task. Direct FT and L2 Reg suffer severe performance
degradation (catastrophic forgetting). (b) Catastrophic forgetting rate,
quantified as the percentage drop in accuracy on the original task.
Self-distillation methods are more effective at preserving knowledge.
(c) The performance trade-off between the original task ($`x`$-axis) and
the spurious task ($`y`$-axis). Distillation methods achieve a much
better trade-off, retaining high original task accuracy while mastering
the new task.

##### Dynamic Self-Distillation Loss ($`\mathcal{L}_{\text{SD-WMA}}`$):

This is the core mechanism for robust knowledge preservation and
adaptive mixing. It ensures the student retains generalizable features
by learning from an evolving teacher model. As detailed in
§[C.6](#A3.SS6 "C.6 Distillation Loss Definitions in TRACER ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
$`\mathcal{L}_{\text{SD-WMA}}`$ is a composite distillation loss that
includes several perspectives: (i) Feature Distillation (FD):Directly
aligns student and teacher embeddings. (ii) Contrastive Relational
Distillation (CRD):Matches batch-wise similarity distributions between
student and teacher. (iii) Interactive Contrastive Learning
(ICL):Encourages student-teacher cross-modal alignment. (iv) Cross
Knowledge Distillation (Cross-KD):Aligns cross-modal logits to transfer
relational structure. This multi-perspective approach operationalizes
the theoretical insight of preserving distinct aspects of pretrained
knowledge.

Weighted Moving Average (WMA) Teacher: The teacher model is a central
component of TRACER. Unlike an EMA teacher, which gradually collapses
onto the student, our WMA teacher is a weighted average of the *entire*
student trajectory up to time $`t`$, using a carefully chosen weighting
kernel (e.g., a Beta kernel with $`\beta_{1}=\beta_{2}=0.5`$ as shown in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
and detailed in
§[C.5.1](#A3.SS5.SSS1 "C.5.1 WMA vs. EMA Teachers ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
This ensures that early pretrained states retain a non-trivial
contribution to the teacher over finite horizons, aligning with our
trajectory-regularization motivation. This persistent regularization
provides a continuous restoring force, preventing the student from
over-specializing on spurious correlations in the finetuning data.

## 5 Experiments

This section evaluates TRACER on ImageNet and natural distribution
shifts, including a controlled toy study to validate theoretical
predictions. We conduct comprehensive ablations across four axes
(distillation components, strength, teacher update frequency, and
Beta-kernel shape); extended protocols are in
§[B](#A2 "Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
loss definitions in
§[C.6](#A3.SS6 "C.6 Distillation Loss Definitions in TRACER ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
and teacher details in
§[C.5.1](#A3.SS5.SSS1 "C.5.1 WMA vs. EMA Teachers ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

### 5.1 Synthetic Experiment

We design a controlled toy experiment with spurious correlations
([Arjovsky et al., 2019](#bib.bib29)) to validate our theory. The
behaviors of Direct Finetuning, L2 Regularization, and Self-Distillation
in a non-linear architecture align with our closed-form predictions.

#### 5.1.1 Experimental Setup

##### Datasets.

We use two variants of the MNIST dataset ([LeCun et al.,
1998](#bib.bib30); [Deng, 2012](#bib.bib33)). (i) Original Pretraining
Task: We create a multimodal version of MNIST, where each grayscale
digit image is paired with a simple text description (e.g., an image of
a ‘7’ is paired with the text “the digit 7”). The model is pretrained on
this dataset to learn robust, general-purpose representations for digit
recognition. (ii) Finetuning Task: We create a dataset to introduce a
spurious correlation. Images of digits 0-4 are colored red with 95%
probability, while digits 5-9 are colored blue with 95% probability.
This setup forces the model during finetuning to learn an
easy-to-exploit but non-causal feature (color) to solve the new task,
creating a direct conflict with the original digit recognition
knowledge.

##### Model Architecture.

We employ lightweight, non-linear models to show that our theory extends
beyond the linear case. The architecture consists of a ‘LightViT’
([Dosovitskiy et al., 2021](#bib.bib31)) image encoder and a
‘LightTextTransformer’ ([Vaswani et al., 2017](#bib.bib32)) text
encoder. Both models project their inputs into a shared 128-dimensional
embedding space, where a standard InfoNCE contrastive loss is applied
during pretraining.

##### Pretraining.

The model is pretrained on MNIST using a contrastive objective for 10
epochs, achieving high accuracy on digit recognition but poor
performance on the color-based task.

##### Finetuning Strategies.

We finetune the pretrained image encoder on the ColoredMNIST ([Arjovsky
et al., 2019](#bib.bib29); [Zhang et al., 2022a](#bib.bib34)) task for
10 epochs while keeping the text encoder frozen, mirroring our
theoretical setup. We compare the following methods: (i) Pretrained:The
baseline model without any finetuning. (ii) Direct Finetuning:The image
encoder is finetuned on the new task, as analyzed in
§[C.3](#A3.SS3 "C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
(iii) $`L_{2}`$Regularization:We add a penalty term
$`\frac{\lambda}{2}\left\|\mathbf{W}_{I}-\mathbf{W}_{I}^{0}\right\|_{\text{F}}^{2}`$
to the finetuning loss, corresponding to our analysis of $`L_{2}`$
regularization. (iv) Static Distillation:We use the initial pretrained
model $`\mathbf{W}_{I}^{0}`$ as a fixed teacher and add a distillation
loss term to the finetuning objective, as analyzed for
$`\mathbf{W}_{SD}`$. (v) Dynamic Distillation:We use a teacher model
whose weights are a moving average of the student’s weights,
corresponding to our analysis of the WMA teacher.

#### 5.1.2 Results and Discussion

##### Analysis of Forgetting.

As predicted by our theory, Direct Finetuning exhibits severe
catastrophic forgetting. It achieves near-perfect accuracy (98.5%) on
the new color-based task by overwriting its original knowledge, causing
its performance on the original MNIST test set to degrade from 96.8% to
59.0%, a forgetting rate of 37.9%. $`L_{2}`$ Regularization offers an
improvement, but still forgets 13.6% of the original task’s performance.
In contrast, both Static and Dynamic Distillation demonstrate resilience
to forgetting. They also master the new task but retain a larger portion
of the original knowledge, with forgetting rates of only 1.8% and 0.1%,
respectively. This result empirically supports our geometric
interpretation: by interpolating between old and new knowledge within
the task-relevant subspace while preserving knowledge in the orthogonal
subspace, self-distillation methods achieve a better balance.

##### The Performance Trade-off.

The scatter plot
in Figure [3](#S4.F3 "Figure 3 ‣ Multi-Modal Contrastive Loss (ℒ_"MMCL"): ‣ 4 Methodology: TRACER ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")(c)
visualizes this trade-off: distillation methods achieve a better Pareto
frontier, with Dynamic Distillation finding a slightly better solution
than its static counterpart, aligning with our theoretical analysis.

### 5.2 Main ImageNet Results and Ablations

We report the main ImageNet results and ablations below; detailed
per-backbone tables follow.

|  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| Method | IN | IN-V2 | IN-R | IN-A | IN-S | ObjNet | Avg. |
| ZS | 68.33 | 61.93 | 77.71 | 49.95 | 48.26 | 54.17 | 58.39 |
| LP-FT | 82.44_(±0.08) | 72.74_(±0.18) | 72.81_(±0.22) | 49.28_(±0.31) | 50.31_(±0.15) | 54.42_(±0.14) | 59.91_(±0.18) |
| FLYP | 82.72_(±0.09) | 72.76_(±0.21) | 71.32_(±0.25) | 48.49_(±0.35) | 49.87_(±0.18) | 54.83_(±0.16) | 59.45_(±0.20) |
| Lipsum-FT | 83.32_(±0.05) | 73.57_(±0.12) | 75.93_(±0.14) | 49.87_(±0.28) | 51.43_(±0.12) | 54.35_(±0.11) | 61.03_(±0.14) |
| CaRot | 83.15_(±0.06) | 74.08_(±0.14) | 77.74_(±0.16) | 51.57_(±0.24) | 52.68_(±0.13) | 56.63_(±0.12) | 62.54_(±0.14) |
| TRACER | 82.76_(±0.07) | 74.14_(±0.15) | 79.33_(±0.18) | 54.92_(±0.26) | 53.69_(±0.14) | 58.26_(±0.13) | 64.07_(±0.15) |

Table 1: ImageNet accuracy. We report accuracy ($`\uparrow`$) on
ImageNet and its distribution shift variants by finetuning CLIP ViT-B/16
with six methods. All values are averaged over three seeds with standard
deviations shown as subscripts. In each column, the best value is bold
and the second-best is underlined.

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| Method | IN$`\uparrow`$ | IN-V2$`\uparrow`$ | IN-R$`\uparrow`$ | IN-A$`\uparrow`$ | IN-S$`\uparrow`$ | Avg. shifts$`\uparrow`$ |
| ZS | 68.33 | 61.93 | 77.71 | 49.95 | 48.26 | 59.46 |
| Direct FT | 82.83_(±0.10) | 72.57_(±0.28) | 68.53_(±0.32) | 39.23_(±0.35) | 47.97_(±0.22) | 57.08_(±0.24) |
| L2-SP ([Li et al., 2018](#bib.bib16)) | 82.87_(±0.09) | 72.63_(±0.22) | 68.77_(±0.24) | 39.73_(±0.28) | 48.23_(±0.15) | 57.34_(±0.18) |
| Static SD ([Hinton et al., 2015](#bib.bib2)) | 82.07_(±0.08) | 73.13_(±0.26) | 72.87_(±0.18) | 42.33_(±0.38) | 49.87_(±0.21) | 59.55_(±0.22) |
| LP-FT ([Kumar et al., 2022](#bib.bib64)) | 82.14_(±0.08) | 72.09_(±0.20) | 70.44_(±0.22) | 46.32_(±0.30) | 48.65_(±0.16) | 59.38_(±0.18) |
| FLYP ([Goyal et al., 2023](#bib.bib28)) | 82.72_(±0.09) | 72.76_(±0.24) | 71.32_(±0.26) | 48.49_(±0.34) | 49.87_(±0.19) | 60.61_(±0.21) |
| CAR-FT ([Mao et al., 2024](#bib.bib75)) | 83.27_(±0.06) | 74.03_(±0.18) | 75.37_(±0.28) | 49.53_(±0.24) | 52.97_(±0.20) | 62.98_(±0.18) |
| Lipsum-FT ([Nam et al., 2024](#bib.bib77)) | 83.33_(±0.05) | 73.57_(±0.12) | 75.93_(±0.14) | 49.87_(±0.28) | 51.43_(±0.12) | 62.70_(±0.14) |
| Model Stock ([Jang et al., 2024](#bib.bib71)) | 84.07_(±0.07) | 74.83_(±0.16) | 71.77_(±0.20) | 51.23_(±0.30) | 51.77_(±0.17) | 62.40_(±0.18) |
| ARF ([Han et al., 2024](#bib.bib90)) | 82.73_(±0.08) | 72.77_(±0.19) | 75.63_(±0.22) | 50.27_(±0.28) | 51.83_(±0.16) | 62.63_(±0.17) |
| CaRot ([Oh et al., 2024](#bib.bib17)) | 83.16_(±0.06) | 74.08_(±0.14) | 77.74_(±0.16) | 51.57_(±0.22) | 52.74_(±0.13) | 64.03_(±0.14) |
| TRACER | 82.76_(±0.07) | 74.12_(±0.15) | 79.30_(±0.18) | 54.72_(±0.24) | 53.69_(±0.14) | 65.46_(±0.15) |

Table 2: ImageNet Accuracy. (except ObjectNet) with additional
baselines. All values are averaged over three seeds.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | RN50 |  |  |  | ViT-B/16 |  |  |  | ViT-L/14 |  |  |  |
| Method | ID Acc.$`\uparrow`$ | ID ECE$`\downarrow`$ | OOD Acc.$`\uparrow`$ | OOD ECE$`\downarrow`$ | ID Acc.$`\uparrow`$ | ID ECE$`\downarrow`$ | OOD Acc.$`\uparrow`$ | OOD ECE$`\downarrow`$ | ID Acc.$`\uparrow`$ | ID ECE$`\downarrow`$ | OOD Acc.$`\uparrow`$ | OOD ECE$`\downarrow`$ |
| LP-FT | 76.25 | 0.1042 | 41.62 | 0.3274 | 82.44 | 0.051 | 59.91 | 0.147 | 84.74 | 0.1056 | 64.11 | 0.2521 |
| FLYP | 76.16 | 0.0516 | 42.70 | 0.2127 | 82.72 | 0.064 | 59.45 | 0.184 | 86.19 | 0.0729 | 71.44 | 0.1470 |
| CaRot | 76.12 | 0.0471 | 42.71 | 0.2109 | 83.15 | 0.047 | 62.54 | 0.079 | 86.95 | 0.0349 | 74.13 | 0.0737 |
| TRACER | 76.48 | 0.0470 | 42.73 | 0.1807 | 82.76 | 0.045 | 64.07 | 0.073 | 86.27 | 0.0507 | 75.32 | 0.0732 |

Table 3: ImageNet accuracy and ECE across backbones. We provide
summarized results on CLIP RN50, ViT-B/16, and ViT-L/14, averaged over
three seeds. The best and the second-best in each column are bold and
underlined, respectively. OOD columns are highlighted to emphasize
robustness. (See
Table [1](#S5.T1 "Table 1 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
and Appendix
Table [5](#A2.T5 "Table 5 ‣ Additional Experimental Results. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
for ViT-B/16, and
Table [6](#A2.T6 "Table 6 ‣ Additional Experimental Results. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
and
[7](#A2.T7 "Table 7 ‣ Additional Experimental Results. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
for details.)

#### 5.2.1 Experimental Setup

##### Objective.

Our experiments are designed to validate our theoretical claims and
assess TRACER, as a practical implementation of our framework, on robust
finetuning. We focus on evaluating both accuracy and calibration under
distribution shifts.

##### Datasets and Evaluation.

We use ImageNet-1K (IN) ([Deng et al., 2009](#bib.bib102); [Russakovsky
et al., 2015](#bib.bib100)) as our in-distribution (ID) downstream task.
To measure OOD robustness, we evaluate all finetuned models on a
standard suite of five distribution shift datasets: ImageNet-V2
(IN-V2) ([Recht et al., 2019](#bib.bib62)), ImageNet-Rendition
(IN-R) ([Hendrycks et al., 2021a](#bib.bib60)), ImageNet-Adversarial
(IN-A) ([Hendrycks et al., 2021b](#bib.bib101)), ImageNet-Sketch
(IN-S) ([Wang et al., 2019](#bib.bib61)), and ObjectNet ([Barbu et al.,
2019](#bib.bib59)). We report the average performance across these five
datasets as “Avg. shifts” or “OOD”.

Baselines. We compare TRACER against a comprehensive set of baselines,
including zero-shot (ZS ([Radford et al., 2021](#bib.bib12))), linear
probing then finetuning (LP-FT ([Kumar et al., 2022](#bib.bib64))),
finetune-like-you-pretrain (FLYP ([Goyal et al., 2023](#bib.bib28))),
Lipsum-FT ([Nam et al., 2024](#bib.bib77)), and the recent robust
finetuning method CaRot ([Oh et al., 2024](#bib.bib17)).

Metrics. We report top-1 accuracy and Expected Calibration Error (ECE),
which measures the gap between predicted confidence and empirical
accuracy (lower is better). We average across the five OOD datasets to
summarize robustness.

Implementation Details and Experimental Setup. We finetune CLIP variants
on ImageNet-1K (IN) and evaluate on five OOD datasets: IN-V2, IN-R,
IN-A, IN-S, and ObjectNet, following [Taori et al. (2020)](#bib.bib63).
For all methods, we finetune for 10 epochs using the AdamW optimizer
with a learning rate of $`1\times 10^{-5}`$ and a weight decay of
$`0.01`$. The batch size is set to 224 for ViT-L/14 and 512 for ViT-B/16
and ResNet50. All experiments are run over three random seeds and we
report mean and standard deviation. For TRACER, the WMA teacher uses a
$`\text{Beta}(0.5,0.5)`$ weighting kernel and combines symmetric InfoNCE
with the composite SD loss
(§[C.6](#A3.SS6 "C.6 Distillation Loss Definitions in TRACER ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).

#### 5.2.2 Results and Analysis

##### OOD accuracy and calibration on ViT-B/16.

As shown in
Table [1](#S5.T1 "Table 1 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
TRACER achieves strong OOD accuracy on ViT-B/16, particularly on the
most challenging shifts (ObjectNet and IN-A). In Appendix
Table [5](#A2.T5 "Table 5 ‣ Additional Experimental Results. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
TRACER achieves the lowest average OOD ECE, indicating probabilistic
reliability under shift. With additional baselines
(Table [2](#S5.T2 "Table 2 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")),
TRACER remains among the top OOD performers while staying competitive on
IN. Additionally, the cross-backbone experiments
(Table [3](#S5.T3 "Table 3 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
show similar trends for RN50 and ViT-L/14.

##### OOD degradation as the empirical signature of catastrophic forgetting.

We treat *OOD accuracy degradation* as the primary measurable symptom of
catastrophic forgetting under finetuning: when a model retains less
pretrained, broadly transferable structure, the loss shows up most
clearly on inputs that differ from the finetuning distribution. Three
results in our experiments support this view: (i) Direct FT drops *below
the zero-shot baseline* on average OOD accuracy
(Table [2](#S5.T2 "Table 2 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"):
57.08% vs. 59.46% ZS); (ii) the toy experiment
(§[5.1](#S5.SS1 "5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
measures explicit forgetting rates of 37.9% for Direct FT vs. 0.1% for
dynamic SD; and (iii) the CKA/SVCCA analysis
(Figure [4](#S5.F4 "Figure 4 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
shows that the deeper layers of Direct FT drift away from the pretrained
representation, while TRACER keeps similarity $`>0.97`$ across all
layers. Together, these justify reading the OOD column of
Tables [1](#S5.T1 "Table 1 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")–[3](#S5.T3 "Table 3 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
as a quantitative measure of how much pretrained knowledge each method
preserves.

##### Static SD vs. TRACER.

Table [2](#S5.T2 "Table 2 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
isolates the contribution of the trajectory-regularized teacher on
ViT-B/16. A static self-distillation anchor at $`\mathbf{W}_{I}^{0}`$
already recovers a large portion of OOD accuracy over Direct FT (Avg.
shifts $`57.08\%\!\to\!59.55\%`$), confirming that distillation-based
preservation is necessary. Static SD still leaves a substantial gap to
TRACER ($`59.55\%\!\to\!65.46\%`$, +5.9), largest where the pretrained
representation is most informative and the static-SD anchor bias is most
punishing: IN-R $`72.87\%\!\to\!\mathbf{79.30\%}`$ (+6.4), IN-A
$`42.33\%\!\to\!\mathbf{54.72\%}`$ (+12.4). These are exactly the
renditions/adversarial shifts where
Theorem [3.4](#S3.Thmtheorem4 "Theorem 3.4 (Bias-Free Convergence in the Task Subspace). ‣ 3.4 Dynamic Self-Distillation with a WMA Teacher ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
predicts the WMA teacher’s benefit: it removes the static-SD
task-subspace bias while preserving the orthogonal directions that
matter for unseen styles and natural adversarial examples.

Ablation Studies. Beyond the primary results, we conducted comprehensive
ablation studies across *four axes* (detailed in
§[B](#A2 "Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
to thoroughly validate TRACER’s design choices. (1) Multi-perspective
distillation
(Table [8](#A2.T8 "Table 8 ‣ Ablation 1: Multi-perspective distillation. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")):
(a) *FD and CRD are the strongest single components*. (b) *The four
components are complementary*: every pair including FD or CRD beats its
singletons. (c) *All four together is best overall*: the full TRACER
setting achieves the highest Avg. All; FD stabilizes features, CRD
preserves relational structure, ICL enriches mutual information, and
Cross-KD blends relational and interactive cues. (2) Distillation
strength
(Table [9](#A2.T9 "Table 9 ‣ Ablation 2: Distillation strength 𝜆_"SD". ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")):
Sweeping $`\lambda_{\text{SD}}`$ from 0.1 to 10.0 reveals that moderate
values ($`\approx`$1.0–2.0) achieve the best ID-OOD trade-off, while
higher values improve calibration at the cost of ID accuracy. (3)
Teacher update frequency
(Table [10](#A2.T10 "Table 10 ‣ Ablation 3: Teacher update frequency. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")):
TRACER maintains stable OOD accuracy ($`\sim`$64.0–64.2%) across update
frequencies from every step to every 2500 steps, eliminating the need
for brittle scheduling required by CaRot ([Oh et al.,
2024](#bib.bib17)). (4) Beta-kernel shape
(Table [11](#A2.T11 "Table 11 ‣ Ablation 4: Beta kernel shape. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")):
Arcsine-like weighting ($`\text{Beta}(0.5,0.5)`$) proves most effective.
These extensive ablations confirm that TRACER’s design is both
principled and robust.

Empirical Validation of Geometric Preservation. To verify our
theoretical claim that TRACER preserves knowledge in the orthogonal
subspace, we conduct a layer-wise representational similarity analysis
using Centered Kernel Alignment (CKA) ([Kornblith et al.,
2019](#bib.bib103)) on the CLIP ViT-B/16 image encoder
(Figure [4](#S5.F4 "Figure 4 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
We extract feature maps from every layer (Patch Embeddings, Transformer
Blocks 0–11, and the Final Projection) on the ImageNet validation set
and compare finetuned models against the pretrained model using CKA and
SVCCA ([Raghu et al., 2017](#bib.bib104)). As shown in
Figure [4](#S5.F4 "Figure 4 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
Direct FT exhibits a drop in similarity in the deeper layers (Blocks
6–11) relative to the pretrained model. This confirms that catastrophic
forgetting manifests as a Feature Distortion of high-level semantic
representations. In contrast, TRACER maintains near-perfect similarity
($`>0.97`$) across all layers. This provides empirical evidence for our
geometric interpretation: TRACER successfully anchors the optimization
to the pretrained geometry, performing surgical updates that adapt to
the task without overwriting robust feature extractors.

Figure 4: Layer-wise Representational Similarity. We compare the
internal representations of the Pretrained model against Direct FT and
TRACER using CKA (left) and SVCCA (right) across all layers of the CLIP
ViT-B/16 image encoder. TRACER (gold) preserves the geometric structure
of the pretrained knowledge significantly better than Direct FT (pink),
particularly in deeper layers.

Computational Efficiency and Complexity. TRACER also offers efficiency
advantages over CaRot ([Oh et al., 2024](#bib.bib17)), whose spectral
regularization scales as $`\mathcal{O}(d^{3})`$. TRACER’s distillation
operates on batch similarity matrices ($`\mathcal{O}(B^{2})`$,
$`B\ll d`$), and the WMA update matches standard EMA cost
($`\mathcal{O}(P)`$). As shown in
Table [4](#S5.T4 "Table 4 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
this reduces training time per epoch compared to CaRot.

|  |  |  |  |  |
|----|----|----|----|----|
| Method | Cost | Time / Epoch | Overhead | Avg. OOD Acc. |
| Direct FT | $`\mathcal{O}(P)`$ | $`\sim`$ 16 min | $`1.00\times`$ | 57.08% |
| CaRot ([Oh et al., 2024](#bib.bib17)) | $`\mathcal{O}(B^{2}+d^{3})`$ | $`\sim`$ 29 min | $`1.81\times`$ | 62.54% |
| TRACER (Ours) | $`\mathcal{O}(B^{2})`$ | $`\sim`$ 22 min | $`\mathbf{1.38\times}`$ | 64.07% |

Table 4: Computational Efficiency. Computational efficiency comparison
per epoch on ImageNet-1K using CLIP ViT-B/16 on an NVIDIA H100 GPU.

Teacher Dynamics and Regularization Strength. Our theory posits that the
WMA teacher in TRACER provides a more persistent regularizing signal
than the EMA teacher used in methods like CaRot. To validate this, we
track the KL divergence between teacher and student throughout training
on ImageNet. As shown in Figure
[5](#S5.F5 "Figure 5 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
for the EMA teacher, the KL decays steadily, indicating that the teacher
is rapidly collapsing onto the student and its regularizing influence is
diminishing. In contrast, the WMA teacher maintains a higher and more
stable KL throughout the entire training process. This sustained
divergence supports the view that the WMA teacher provides a persistent
“restoring force,” as predicted by our analysis in
§[C.5.2](#A3.SS5.SSS2 "C.5.2 The Persistent Regularizer of the WMA Teacher ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
This helps prevent the student from converging to a narrow task-specific
minimum and reflects the persistent regularization strength required for
robust generalization.

![Refer to caption](2605.29380v1/teacher_student_kl_v2.png)

![Refer to caption](2605.29380v1/teacher_entropy_v2.png)

![Refer to caption](2605.29380v1/teacher_confidence_v2.png)

Figure 5: Teacher–Student Knowledge Gap During Training. Compared to the
EMA teacher (blue), which shows rapidly vanishing KL divergence and thus
a weakening regularization signal (left), the WMA teacher (orange)
sustains a higher and more stable KL gap. This stability is supported by
higher teacher entropy (middle) and moderated confidence (right),
preventing overfitting. Together, these trends confirm that WMA provides
a stronger and more persistent self-distillation signal than EMA.

Algorithmic Simplicity. While EMA-based methods often require complex,
sparse update schedules (e.g., updating only every 500 steps with linear
warmup and careful momentum tuning) to prevent collapse, TRACER is
robust to update frequency. As shown in
Table [10](#A2.T10 "Table 10 ‣ Ablation 3: Teacher update frequency. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
TRACER maintains consistent performance ($`\sim 64.0-64.2\%`$ OOD
accuracy) whether the teacher is updated every step or every 2500 steps,
reducing the need for brittle hyperparameter tuning.
Figure [6](#A2.F6 "Figure 6 ‣ Additional Experimental Results. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
illustrates this failure mode in prior methods: without careful tuning,
the EMA teacher in CaRot collapses immediately *when updated at every
step*, whereas TRACER’s WMA teacher remains stable even under a dense
update schedule.

## 6 Conclusion

We proved that TRACER’s trajectory-averaging WMA teacher, unlike its EMA
counterpart, maintains a persistent regularizing force over finite
training horizons. This force continuously anchors the model to its
robust pretrained initialization, helping prevent overfitting and
improving out-of-distribution performance. Our extensive ablation
studies confirm that TRACER’s design is both principled and robust to
hyperparameter choices across distillation strength, update frequency,
and kernel shape. Our work bridges the geometry of finetuning with the
practical design of robust methods, and these principles motivate future
extensions to parameter-efficient methods, continual learning,
random-feature analyses of the same trajectory-regularization idea, and
larger vision–language backbones.

Limitations and Scope. Our empirical evaluation focuses on CLIP-style
vision–language backbones and standard vision robustness benchmarks. We
do not yet provide experiments on broader modalities or multimodal LLM
settings, and our theoretical analysis is developed in the linearized
image/text-encoder regime; we discuss concrete extensions to
random-feature settings, to larger VLMs, and the relationship to
prompt-based adaptation in
§[F](#A6 "Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

## Impact Statement

This work develops methods for improving the robustness and calibration
of finetuned multimodal models under distribution shift. Our primary
motivation is enhancing the reliability of foundation models as they are
deployed in real-world applications where inputs may differ from
training data. Robust and calibrated models are particularly valuable in
safety-critical domains such as medical imaging and autonomous systems,
where overconfident predictions under shift can lead to harmful
outcomes. Our method is purely defensive in nature: it mitigates
catastrophic forgetting and improves generalization. We see no obvious
pathways by which this work accelerates harmful capabilities.

## Acknowledgements

This research was conducted by the ARC Centre of Excellence for
Automated Decision-Making and Society (CE200100005), and funded by the
Australian Government through the Australian Research Council. This
research was supported by The University of Melbourne’s Research
Computing Services and the Petascale Campus Initiative. We would like to
thank Navid Akhavan Attar, Aryan Yazdan Parast, and Hugo Lyons Keenan
for valuable discussions and feedback.

## Conflict of Interest Disclosure

The authors declare no financial conflicts of interest.

## References

- Alayrac et al. (2022) J. Alayrac, J. Donahue, P. Luc, A. Miech, I.
  Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R.
  Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M.
  Monteiro, J. L. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S.
  Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman,
  and K. Simonyan Flamingo: a visual language model for few-shot
  learning. In Advances in Neural Information Processing Systems 35:
  Annual Conference on Neural Information Processing Systems 2022,
  NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022,
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Aljundi et al. (2019) R. Aljundi, M. Lin, B. Goujaud, and Y. Bengio
  Gradient based sample selection for online continual learning. In
  Advances in Neural Information Processing Systems 32: Annual
  Conference on Neural Information Processing Systems 2019, NeurIPS
  2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 11816–11825.
  Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Arjovsky et al. (2019) M. Arjovsky, L. Bottou, I. Gulrajani, and D.
  Lopez-Paz Invariant risk minimization. CoRR abs/1907.02893. External
  Links: [Link](http://arxiv.org/abs/1907.02893), 1907.02893 Cited by:
  [§5.1.1](#S5.SS1.SSS1.Px4.p1.1 "Finetuning Strategies. ‣ 5.1.1 Experimental Setup ‣ 5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.1](#S5.SS1.p1.1 "5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Barbu et al. (2019) A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D.
  Gutfreund, J. Tenenbaum, and B. Katz ObjectNet: A large-scale
  bias-controlled dataset for pushing the limits of object recognition
  models. In Advances in Neural Information Processing Systems 32:
  Annual Conference on Neural Information Processing Systems 2019,
  NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada,
  pp. 9448–9458. Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Betker et al. (2023) J. Betker, G. Goh, L. Jing, T. Brooks, J.
  Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al. Improving
  image generation with better captions. Computer Science. https://cdn.
  openai. com/papers/dall-e-3. pdf 2 (3), pp. 8. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Caron et al. (2021) M. Caron, H. Touvron, I. Misra, H. Jégou, J.
  Mairal, P. Bojanowski, and A. Joulin Emerging properties in
  self-supervised vision transformers. In 2021 IEEE/CVF International
  Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada,
  October 10-17, 2021, pp. 9630–9640. External Links:
  [Link](https://doi.org/10.1109/ICCV48922.2021.00951),
  [Document](https://dx.doi.org/10.1109/ICCV48922.2021.00951) Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Deng et al. (2009) J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L.
  Fei-Fei ImageNet: A large-scale hierarchical image database. In 2009
  IEEE Computer Society Conference on Computer Vision and Pattern
  Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA,
  pp. 248–255. External Links:
  [Link](https://doi.org/10.1109/CVPR.2009.5206848),
  [Document](https://dx.doi.org/10.1109/CVPR.2009.5206848) Cited by:
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Deng (2012) L. Deng The MNIST database of handwritten digit images for
  machine learning research \[best of the web\]. IEEE Signal Process.
  Mag. 29 (6), pp. 141–142. External Links:
  [Link](https://doi.org/10.1109/MSP.2012.2211477),
  [Document](https://dx.doi.org/10.1109/MSP.2012.2211477) Cited by:
  [§5.1.1](#S5.SS1.SSS1.Px1.p1.1 "Datasets. ‣ 5.1.1 Experimental Setup ‣ 5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Desai and Johnson (2021) K. Desai and J. Johnson VirTex: learning
  visual representations from textual annotations. In IEEE Conference on
  Computer Vision and Pattern Recognition, CVPR 2021, virtual, June
  19-25, 2021, pp. 11162–11173. External Links:
  [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Desai%5C_VirTex%5C_Learning%5C_Visual%5C_Representations%5C_From%5C_Textual%5C_Annotations%5C_CVPR%5C_2021%5C_paper.html),
  [Document](https://dx.doi.org/10.1109/CVPR46437.2021.01101) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Dosovitskiy et al. (2021) A. Dosovitskiy, L. Beyer, A. Kolesnikov, D.
  Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G.
  Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby An image is worth
  16x16 words: transformers for image recognition at scale. In 9th
  International Conference on Learning Representations, ICLR 2021,
  Virtual Event, Austria, May 3-7, 2021, External Links:
  [Link](https://openreview.net/forum?id=YicbFdNTTy) Cited by:
  [§5.1.1](#S5.SS1.SSS1.Px2.p1.1 "Model Architecture. ‣ 5.1.1 Experimental Setup ‣ 5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Fang et al. (2022) A. Fang, G. Ilharco, M. Wortsman, Y. Wan, V.
  Shankar, A. Dave, and L. Schmidt Data determines distributional
  robustness in contrastive language image pre-training (CLIP). In
  International Conference on Machine Learning, ICML 2022, 17-23 July
  2022, Baltimore, Maryland, USA, Proceedings of Machine Learning
  Research, Vol. 162, pp. 6216–6234. External Links:
  [Link](https://proceedings.mlr.press/v162/fang22a.html) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Fang et al. (2024) A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. T.
  Toshev, and V. Shankar Data filtering networks. In The Twelfth
  International Conference on Learning Representations, ICLR 2024,
  Vienna, Austria, May 7-11, 2024, External Links:
  [Link](https://openreview.net/forum?id=KAk6ngZ09F) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Fang et al. (2023) Y. Fang, W. Wang, B. Xie, Q. Sun, L. Wu, X.
  Wang, T. Huang, X. Wang, and Y. Cao EVA: exploring the limits of
  masked visual representation learning at scale. In IEEE/CVF Conference
  on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC,
  Canada, June 17-24, 2023, pp. 19358–19369. External Links:
  [Link](https://doi.org/10.1109/CVPR52729.2023.01855),
  [Document](https://dx.doi.org/10.1109/CVPR52729.2023.01855) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Fang et al. (2021) Z. Fang, J. Wang, X. Hu, L. Wang, Y. Yang, and Z.
  Liu Compressing visual-linguistic model via knowledge distillation. In
  2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021,
  Montreal, QC, Canada, October 10-17, 2021, pp. 1408–1418. External
  Links: [Link](https://doi.org/10.1109/ICCV48922.2021.00146),
  [Document](https://dx.doi.org/10.1109/ICCV48922.2021.00146) Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Frankle and Carbin (2019) J. Frankle and M. Carbin The lottery ticket
  hypothesis: finding sparse, trainable neural networks. In 7th
  International Conference on Learning Representations, ICLR 2019, New
  Orleans, LA, USA, May 6-9, 2019, External Links:
  [Link](https://openreview.net/forum?id=rJl-b3RcF7) Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- French (1999) R. M. French Catastrophic forgetting in connectionist
  networks. Trends in cognitive sciences 3 (4), pp. 128–135. Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Furlanello et al. (2018) T. Furlanello, Z. C. Lipton, M. Tschannen, L.
  Itti, and A. Anandkumar Born-again neural networks. In Proceedings of
  the 35th International Conference on Machine Learning, ICML 2018,
  Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, Proceedings of
  Machine Learning Research, Vol. 80, pp. 1602–1611. External Links:
  [Link](http://proceedings.mlr.press/v80/furlanello18a.html) Cited by:
  [Theorem
  C.2](#A3.Thmapptheorem2.p1.3.1.4.1.2.1.2.1 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [item 3](#S3.I1.i3.p1.1 "In Theorem 3.2 (Unified Framework for Contrastive Finetuning Solutions). ‣ 3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Garrido et al. (2023) Q. Garrido, Y. Chen, A. Bardes, L. Najman,
  and Y. LeCun On the duality between contrastive and non-contrastive
  self-supervised learning. In The Eleventh International Conference on
  Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023,
  External Links: [Link](https://openreview.net/forum?id=kDEL91Dufpa)
  Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Goyal et al. (2023) S. Goyal, A. Kumar, S. Garg, Z. Kolter, and A.
  Raghunathan Finetune like you pretrain: improved finetuning of
  zero-shot vision models. In IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24,
  2023, pp. 19338–19347. External Links:
  [Link](https://doi.org/10.1109/CVPR52729.2023.01853),
  [Document](https://dx.doi.org/10.1109/CVPR52729.2023.01853) Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p3.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p2.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.7.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Grill et al. (2020) J. Grill, F. Strub, F. Altché, C. Tallec, P. H.
  Richemond, E. Buchatskaya, C. Doersch, B. Á. Pires, Z. Guo, M. G.
  Azar, B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko Bootstrap your
  own latent - A new approach to self-supervised learning. In Advances
  in Neural Information Processing Systems 33: Annual Conference on
  Neural Information Processing Systems 2020, NeurIPS 2020, December
  6-12, 2020, virtual, Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Han et al. (2024) J. Han, Z. Lin, Z. Sun, Y. Gao, K. Yan, S. Ding, Y.
  Gao, and G. Xia Anchor-based robust finetuning of vision-language
  models. In IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024,
  pp. 26909–26918. External Links:
  [Link](https://doi.org/10.1109/CVPR52733.2024.02542),
  [Document](https://dx.doi.org/10.1109/CVPR52733.2024.02542) Cited by:
  [Table
  2](#S5.T2.6.1.11.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Hao et al. (2025) Y. Hao, X. Pan, H. Zhang, C. Ye, R. Pan, and T.
  Zhang Understanding overadaptation in supervised fine-tuning: the role
  of ensemble methods. In Forty-second International Conference on
  Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025,
  External Links: [Link](https://openreview.net/forum?id=1xsW6tvMb3)
  Cited by:
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- HaoChen and Ma (2023) J. Z. HaoChen and T. Ma A theoretical study of
  inductive biases in contrastive learning. In The Eleventh
  International Conference on Learning Representations, ICLR 2023,
  Kigali, Rwanda, May 1-5, 2023, External Links:
  [Link](https://openreview.net/forum?id=AuEgNlEAmed) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- HaoChen et al. (2022) J. Z. HaoChen, C. Wei, A. Kumar, and T. Ma
  Beyond separability: analyzing the linear transferability of
  contrastive representations to related subpopulations. In Advances in
  Neural Information Processing Systems 35: Annual Conference on Neural
  Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA,
  USA, November 28 - December 9, 2022, Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- He et al. (2020) K. He, H. Fan, Y. Wu, S. Xie, and R. B. Girshick
  Momentum contrast for unsupervised visual representation learning. In
  2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
  CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pp. 9726–9735. External
  Links: [Link](https://doi.org/10.1109/CVPR42600.2020.00975),
  [Document](https://dx.doi.org/10.1109/CVPR42600.2020.00975) Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Hendrycks et al. (2021a) D. Hendrycks, S. Basart, N. Mu, S.
  Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M.
  Guo, D. Song, J. Steinhardt, and J. Gilmer The many faces of
  robustness: A critical analysis of out-of-distribution generalization.
  In 2021 IEEE/CVF International Conference on Computer Vision, ICCV
  2021, Montreal, QC, Canada, October 10-17, 2021, pp. 8320–8329.
  External Links: [Link](https://doi.org/10.1109/ICCV48922.2021.00823),
  [Document](https://dx.doi.org/10.1109/ICCV48922.2021.00823) Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Hendrycks and Dietterich (2019) D. Hendrycks and T. G. Dietterich
  Benchmarking neural network robustness to common corruptions and
  perturbations. In 7th International Conference on Learning
  Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019,
  External Links: [Link](https://openreview.net/forum?id=HJz6tiCqYm)
  Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Hendrycks et al. (2021b) D. Hendrycks, K. Zhao, S. Basart, J.
  Steinhardt, and D. Song Natural adversarial examples. In IEEE
  Conference on Computer Vision and Pattern Recognition, CVPR 2021,
  virtual, June 19-25, 2021, pp. 15262–15271. External Links:
  [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Hendrycks%5C_Natural%5C_Adversarial%5C_Examples%5C_CVPR%5C_2021%5C_paper.html),
  [Document](https://dx.doi.org/10.1109/CVPR46437.2021.01501) Cited by:
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Hinton et al. (2015) G. E. Hinton, O. Vinyals, and J. Dean Distilling
  the knowledge in a neural network. CoRR abs/1503.02531. External
  Links: [Link](http://arxiv.org/abs/1503.02531), 1503.02531 Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.5.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Houlsby et al. (2019) N. Houlsby, A. Giurgiu, S. Jastrzebski, B.
  Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly
  Parameter-efficient transfer learning for NLP. In Proceedings of the
  36th International Conference on Machine Learning, ICML 2019, 9-15
  June 2019, Long Beach, California, USA, Proceedings of Machine
  Learning Research, Vol. 97, pp. 2790–2799. External Links:
  [Link](http://proceedings.mlr.press/v97/houlsby19a.html) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§F.2](#A6.SS2.p1.1 "F.2 Relation to PEFT and prompt-based adaptation ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Hu et al. (2022) E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S.
  Wang, L. Wang, and W. Chen LoRA: low-rank adaptation of large language
  models. In The Tenth International Conference on Learning
  Representations, ICLR 2022, Virtual Event, April 25-29, 2022, External
  Links: [Link](https://openreview.net/forum?id=nZeVKeeFYf9) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§F.2](#A6.SS2.p1.1 "F.2 Relation to PEFT and prompt-based adaptation ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Ilharco et al. (2021) OpenCLIP Note: If you use this software, please
  cite it as below. External Links:
  [Document](https://dx.doi.org/10.5281/zenodo.5143773),
  [Link](https://doi.org/10.5281/zenodo.5143773) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Izmailov et al. (2018) P. Izmailov, D. Podoprikhin, T. Garipov, D. P.
  Vetrov, and A. G. Wilson Averaging weights leads to wider optima and
  better generalization. In Proceedings of the Thirty-Fourth Conference
  on Uncertainty in Artificial Intelligence, UAI 2018, Monterey,
  California, USA, August 6-10, 2018, pp. 876–885. External Links:
  [Link](http://auai.org/uai2018/proceedings/papers/313.pdf) Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [4th
  item](#A3.I5.i4.p1.1 "In Key differences. ‣ C.5.1 WMA vs. EMA Teachers ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Jacot et al. (2018) A. Jacot, F. Gabriel, and C. Hongler Neural
  tangent kernel: convergence and generalization in neural networks. In
  Advances in Neural Information Processing Systems, S. Bengio, H.
  Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett
  (Eds.), Vol. 31, pp. . External Links:
  [Link](https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf)
  Cited by:
  [§F.3](#A6.SS3.p1.1 "F.3 Future work: NTK and random-feature extensions of the theory ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Jang et al. (2024) D. Jang, S. Yun, and D. Han Model stock: all we
  need is just a few fine-tuned models. In Computer Vision - ECCV 2024 -
  18th European Conference, Milan, Italy, September 29-October 4, 2024,
  Proceedings, Part XLIV, Lecture Notes in Computer Science, Vol. 15102,
  pp. 207–223. External Links:
  [Link](https://doi.org/10.1007/978-3-031-72784-9%5C_12),
  [Document](https://dx.doi.org/10.1007/978-3-031-72784-9%5F12) Cited
  by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p3.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.10.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Ji et al. (2023) W. Ji, Z. Deng, R. Nakada, J. Zou, and L. Zhang The
  power of contrast for feature learning: A theoretical analysis.  J.
  Mach. Learn. Res. 24, pp. 330:1–330:78. External Links:
  [Link](http://jmlr.org/papers/v24/21-1501.html) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.1](#A3.SS1.p1.1 "C.1 Derivation of ℒ_"CL" ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§F.3](#A6.SS3.p1.1 "F.3 Future work: NTK and random-feature extensions of the theory ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§3.1](#S3.SS1.p2.1 "3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Jia et al. (2021) C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H.
  Pham, Q. V. Le, Y. Sung, Z. Li, and T. Duerig Scaling up visual and
  vision-language representation learning with noisy text supervision.
  In Proceedings of the 38th International Conference on Machine
  Learning, ICML 2021, 18-24 July 2021, Virtual Event, Proceedings of
  Machine Learning Research, Vol. 139, pp. 4904–4916. External Links:
  [Link](http://proceedings.mlr.press/v139/jia21b.html) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Jia et al. (2022) M. Jia, L. Tang, B. Chen, C. Cardie, S. J.
  Belongie, B. Hariharan, and S. Lim Visual prompt tuning. In Computer
  Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel,
  October 23-27, 2022, Proceedings, Part XXXIII, S. Avidan, G. J.
  Brostow, M. Cissé, G. M. Farinella, and T. Hassner (Eds.), Lecture
  Notes in Computer Science, pp. 709–727. External Links:
  [Link](https://doi.org/10.1007/978-3-031-19827-4%5C_41),
  [Document](https://dx.doi.org/10.1007/978-3-031-19827-4%5F41) Cited
  by:
  [§F.2](#A6.SS2.p1.1 "F.2 Relation to PEFT and prompt-based adaptation ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Jung et al. (2020) S. Jung, H. Ahn, S. Cha, and T. Moon Continual
  learning with node-importance based adaptive group sparse
  regularization. In Advances in Neural Information Processing Systems
  33: Annual Conference on Neural Information Processing Systems 2020,
  NeurIPS 2020, December 6-12, 2020, virtual, Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Kirkpatrick et al. (2017) J. Kirkpatrick, R. Pascanu, N.
  Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J.
  Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic
  forgetting in neural networks. Proceedings of the national academy of
  sciences 114 (13), pp. 3521–3526. Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Kornblith et al. (2019) S. Kornblith, M. Norouzi, H. Lee, and G. E.
  Hinton Similarity of neural network representations revisited. In
  Proceedings of the 36th International Conference on Machine Learning,
  ICML 2019, 9-15 June 2019, Long Beach, California, USA, Proceedings of
  Machine Learning Research, Vol. 97, pp. 3519–3529. External Links:
  [Link](http://proceedings.mlr.press/v97/kornblith19a.html) Cited by:
  [§5.2.2](#S5.SS2.SSS2.Px3.p3.1 "Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Kumar et al. (2022) A. Kumar, A. Raghunathan, R. M. Jones, T. Ma,
  and P. Liang Fine-tuning can distort pretrained features and
  underperform out-of-distribution. In The Tenth International
  Conference on Learning Representations, ICLR 2022, Virtual Event,
  April 25-29, 2022, External Links:
  [Link](https://openreview.net/forum?id=UYneFzXSJWh) Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p3.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p2.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.6.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Laine and Aila (2017) S. Laine and T. Aila Temporal ensembling for
  semi-supervised learning. In 5th International Conference on Learning
  Representations, ICLR 2017, Toulon, France, April 24-26, 2017,
  Conference Track Proceedings, External Links:
  [Link](https://openreview.net/forum?id=BJ6oOfqge) Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- LeCun et al. (1998) Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner
  Gradient-based learning applied to document recognition. Proc. IEEE 86
  (11), pp. 2278–2324. External Links:
  [Link](https://doi.org/10.1109/5.726791),
  [Document](https://dx.doi.org/10.1109/5.726791) Cited by:
  [§5.1.1](#S5.SS1.SSS1.Px1.p1.1 "Datasets. ‣ 5.1.1 Experimental Setup ‣ 5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li and Liang (2021) X. L. Li and P. Liang Prefix-tuning: optimizing
  continuous prompts for generation. In Proceedings of the 59th Annual
  Meeting of the Association for Computational Linguistics and the 11th
  International Joint Conference on Natural Language Processing,
  ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6,
  2021, pp. 4582–4597. External Links:
  [Link](https://doi.org/10.18653/v1/2021.acl-long.353),
  [Document](https://dx.doi.org/10.18653/V1/2021.ACL-LONG.353) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li et al. (2023a) X. Li, Z. Wang, and C. Xie CLIPA-v2: scaling CLIP
  training with 81.1% zero-shot imagenet accuracy within a \$10, 000
  budget; an extra \$4, 000 unlocks 81.8% accuracy. CoRR abs/2306.15658.
  External Links: [Link](https://doi.org/10.48550/arXiv.2306.15658),
  [Document](https://dx.doi.org/10.48550/ARXIV.2306.15658), 2306.15658
  Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li et al. (2023b) X. Li, Y. Fang, M. Liu, Z. Ling, Z. Tu, and H. Su
  Distilling large vision-language model with out-of-distribution
  generalizability. In IEEE/CVF International Conference on Computer
  Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 2492–2503.
  External Links: [Link](https://doi.org/10.1109/ICCV51070.2023.00236),
  [Document](https://dx.doi.org/10.1109/ICCV51070.2023.00236) Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li et al. (2018) X. Li, Y. Grandvalet, and F. Davoine Explicit
  inductive bias for transfer learning with convolutional networks. In
  Proceedings of the 35th International Conference on Machine Learning,
  ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018,
  Proceedings of Machine Learning Research, Vol. 80, pp. 2830–2839.
  External Links: [Link](http://proceedings.mlr.press/v80/li18a.html)
  Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Theorem
  C.2](#A3.Thmapptheorem2.p1.3.1.3.1.2.1.2.1 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p3.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [item 2](#S3.I1.i2.p1.1 "In Theorem 3.2 (Unified Framework for Contrastive Finetuning Solutions). ‣ 3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.4.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li et al. (2023c) Y. Li, H. Fan, R. Hu, C. Feichtenhofer, and K. He
  Scaling language-image pre-training via masking. In IEEE/CVF
  Conference on Computer Vision and Pattern Recognition, CVPR 2023,
  Vancouver, BC, Canada, June 17-24, 2023, pp. 23390–23400. External
  Links: [Link](https://doi.org/10.1109/CVPR52729.2023.02240),
  [Document](https://dx.doi.org/10.1109/CVPR52729.2023.02240) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li et al. (2024) Z. Li, X. Li, X. Fu, X. Zhang, W. Wang, S. Chen,
  and J. Yang PromptKD: unsupervised prompt distillation for
  vision-language models. In IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024,
  pp. 26607–26616. External Links:
  [Link](https://doi.org/10.1109/CVPR52733.2024.02513),
  [Document](https://dx.doi.org/10.1109/CVPR52733.2024.02513) Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Li and Hoiem (2018) Z. Li and D. Hoiem Learning without forgetting.
  IEEE Trans. Pattern Anal. Mach. Intell. 40 (12), pp. 2935–2947.
  External Links: [Link](https://doi.org/10.1109/TPAMI.2017.2773081),
  [Document](https://dx.doi.org/10.1109/TPAMI.2017.2773081) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Liang et al. (2023) C. Liang, J. Yu, M. Yang, M. Brown, Y. Cui, T.
  Zhao, B. Gong, and T. Zhou Module-wise adaptive distillation for
  multimodality foundation models. In Advances in Neural Information
  Processing Systems 36: Annual Conference on Neural Information
  Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December
  10 - 16, 2023, Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Liu et al. (2023) H. Liu, C. Li, Q. Wu, and Y. J. Lee Visual
  instruction tuning. In Advances in Neural Information Processing
  Systems 36: Annual Conference on Neural Information Processing Systems
  2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023,
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Liu et al. (2022) H. Liu, J. Z. HaoChen, A. Gaidon, and T. Ma
  Self-supervised learning is more robust to dataset imbalance. In The
  Tenth International Conference on Learning Representations, ICLR 2022,
  Virtual Event, April 25-29, 2022, External Links:
  [Link](https://openreview.net/forum?id=4AZz9osqrar) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Mao et al. (2024) X. Mao, Y. Chen, X. Jia, R. Zhang, H. Xue, and Z. Li
  Context-aware robust fine-tuning. Int. J. Comput. Vis. 132 (5),
  pp. 1685–1700. External Links:
  [Link](https://doi.org/10.1007/s11263-023-01951-2),
  [Document](https://dx.doi.org/10.1007/S11263-023-01951-2) Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.8.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- McCloskey and Cohen (1989) M. McCloskey and N. J. Cohen Catastrophic
  interference in connectionist networks: the sequential learning
  problem. Psychology of learning and motivation 24, pp. 109–165. Cited
  by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Mobahi et al. (2020) H. Mobahi, M. Farajtabar, and P. L. Bartlett
  Self-distillation amplifies regularization in hilbert space. In
  Advances in Neural Information Processing Systems 33: Annual
  Conference on Neural Information Processing Systems 2020, NeurIPS
  2020, December 6-12, 2020, virtual, Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Nakada et al. (2023) R. Nakada, H. I. Gulluk, Z. Deng, W. Ji, J. Zou,
  and L. Zhang Understanding multimodal contrastive learning and
  incorporating unpaired data. In International Conference on Artificial
  Intelligence and Statistics, 25-27 April 2023, Palau de Congressos,
  Valencia, Spain, Proceedings of Machine Learning Research, Vol. 206,
  pp. 4348–4380. External Links:
  [Link](https://proceedings.mlr.press/v206/nakada23a.html) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.1](#A3.SS1.p1.1 "C.1 Derivation of ℒ_"CL" ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§F.3](#A6.SS3.p1.1 "F.3 Future work: NTK and random-feature extensions of the theory ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§3.1](#S3.SS1.p2.1 "3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Nam et al. (2024) G. Nam, B. Heo, and J. Lee Lipsum-ft: robust
  fine-tuning of zero-shot models using random text guidance. In The
  Twelfth International Conference on Learning Representations, ICLR
  2024, Vienna, Austria, May 7-11, 2024, External Links:
  [Link](https://openreview.net/forum?id=2JF8mJRJ7M) Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p2.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.9.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Oh et al. (2025) C. Oh, Y. Li, K. Song, S. Yun, and D. Han DaWin:
  training-free dynamic weight interpolation for robust adaptation. In
  The Thirteenth International Conference on Learning Representations,
  ICLR 2025, Singapore, April 24-28, 2025, External Links:
  [Link](https://openreview.net/forum?id=L8e7tBf4pP) Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Oh et al. (2024) C. Oh, H. Lim, M. Kim, D. Han, S. Yun, J. Choo, A.
  Hauptmann, Z. Cheng, and K. Song Towards calibrated robust fine-tuning
  of vision-language models. In Advances in Neural Information
  Processing Systems 38: Annual Conference on Neural Information
  Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December
  10 - 15, 2024, Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p3.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p2.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.2](#S5.SS2.SSS2.Px3.p2.1 "Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.2](#S5.SS2.SSS2.Px3.p4.1 "Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  2](#S5.T2.6.1.12.1 "In 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [Table
  4](#S5.T4.8.1.3.1 "In Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Pi et al. (2024) R. Pi, L. Yao, J. Han, X. Liang, W. Zhang, and H. Xu
  Ins-detclip: aligning detection model to follow human-language
  instruction. In The Twelfth International Conference on Learning
  Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, External
  Links: [Link](https://openreview.net/forum?id=M0MF4t3hE9) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Radford et al. (2021) A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G.
  Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G.
  Krueger, and I. Sutskever Learning transferable visual models from
  natural language supervision. In Proceedings of the 38th International
  Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual
  Event, Proceedings of Machine Learning Research, Vol. 139,
  pp. 8748–8763. External Links:
  [Link](http://proceedings.mlr.press/v139/radford21a.html) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p2.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p2.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Raghu et al. (2017) M. Raghu, J. Gilmer, J. Yosinski, and J.
  Sohl-Dickstein SVCCA: singular vector canonical correlation analysis
  for deep learning dynamics and interpretability. In Advances in Neural
  Information Processing Systems 30: Annual Conference on Neural
  Information Processing Systems 2017, December 4-9, 2017, Long Beach,
  CA, USA, pp. 6076–6085. Cited by:
  [§5.2.2](#S5.SS2.SSS2.Px3.p3.1 "Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Rebuffi et al. (2017) S. Rebuffi, A. Kolesnikov, G. Sperl, and C. H.
  Lampert ICaRL: incremental classifier and representation learning. In
  2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR
  2017, Honolulu, HI, USA, July 21-26, 2017, pp. 5533–5542. External
  Links: [Link](https://doi.org/10.1109/CVPR.2017.587),
  [Document](https://dx.doi.org/10.1109/CVPR.2017.587) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Recht et al. (2019) B. Recht, R. Roelofs, L. Schmidt, and V. Shankar
  Do imagenet classifiers generalize to imagenet?. In Proceedings of the
  36th International Conference on Machine Learning, ICML 2019, 9-15
  June 2019, Long Beach, California, USA, Proceedings of Machine
  Learning Research, Vol. 97, pp. 5389–5400. External Links:
  [Link](http://proceedings.mlr.press/v97/recht19a.html) Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Robins (1995) A. Robins Catastrophic forgetting, rehearsal and
  pseudorehearsal. Connection Science 7 (2), pp. 123–146. Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Russakovsky et al. (2015) O. Russakovsky, J. Deng, H. Su, J.
  Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. S.
  Bernstein, A. C. Berg, and L. Fei-Fei ImageNet large scale visual
  recognition challenge. Int. J. Comput. Vis. 115 (3), pp. 211–252.
  External Links: [Link](https://doi.org/10.1007/s11263-015-0816-y),
  [Document](https://dx.doi.org/10.1007/S11263-015-0816-Y) Cited by:
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Rusu et al. (2016) A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H.
  Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell
  Progressive neural networks. CoRR abs/1606.04671. External Links:
  [Link](http://arxiv.org/abs/1606.04671), 1606.04671 Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Sariyildiz et al. (2020) M. B. Sariyildiz, J. Perez, and D. Larlus
  Learning visual representations with caption annotations. In Computer
  Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August
  23-28, 2020, Proceedings, Part VIII, Lecture Notes in Computer
  Science, Vol. 12353, pp. 153–170. External Links:
  [Link](https://doi.org/10.1007/978-3-030-58598-3%5C_10),
  [Document](https://dx.doi.org/10.1007/978-3-030-58598-3%5F10) Cited
  by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Saunshi et al. (2019) N. Saunshi, O. Plevrakis, S. Arora, M. Khodak,
  and H. Khandeparkar A theoretical analysis of contrastive unsupervised
  representation learning. In Proceedings of the 36th International
  Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach,
  California, USA, Proceedings of Machine Learning Research, Vol. 97,
  pp. 5628–5637. External Links:
  [Link](http://proceedings.mlr.press/v97/saunshi19a.html) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Schuhmann et al. (2022) C. Schuhmann, R. Beaumont, R. Vencu, C.
  Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M.
  Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R.
  Kaczmarczyk, and J. Jitsev LAION-5B: an open large-scale dataset for
  training next generation image-text models. In Advances in Neural
  Information Processing Systems 35: Annual Conference on Neural
  Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA,
  USA, November 28 - December 9, 2022, Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Shen et al. (2022a) K. Shen, R. M. Jones, A. Kumar, S. M. Xie, J. Z.
  HaoChen, T. Ma, and P. Liang Connect, not collapse: explaining
  contrastive learning for unsupervised domain adaptation. In
  International Conference on Machine Learning, ICML 2022, 17-23 July
  2022, Baltimore, Maryland, USA, Proceedings of Machine Learning
  Research, Vol. 162, pp. 19847–19878. External Links:
  [Link](https://proceedings.mlr.press/v162/shen22d.html) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Shen et al. (2022b) S. Shen, L. H. Li, H. Tan, M. Bansal, A.
  Rohrbach, K. Chang, Z. Yao, and K. Keutzer How much can CLIP benefit
  vision-and-language tasks?. In The Tenth International Conference on
  Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022,
  External Links: [Link](https://openreview.net/forum?id=zf%5C_Ll3HZWgy)
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Shu et al. (2023) Y. Shu, X. Guo, J. Wu, X. Wang, J. Wang, and M. Long
  CLIPood: generalizing CLIP to out-of-distributions. In International
  Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu,
  Hawaii, USA, Proceedings of Machine Learning Research, Vol. 202,
  pp. 31716–31731. External Links:
  [Link](https://proceedings.mlr.press/v202/shu23a.html) Cited by:
  [§A.4](#A1.SS4.p2.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Sun et al. (2023) Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao
  EVA-CLIP: improved training techniques for CLIP at scale. CoRR
  abs/2303.15389. External Links:
  [Link](https://doi.org/10.48550/arXiv.2303.15389),
  [Document](https://dx.doi.org/10.48550/ARXIV.2303.15389), 2303.15389
  Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Taori et al. (2020) R. Taori, A. Dave, V. Shankar, N. Carlini, B.
  Recht, and L. Schmidt Measuring robustness to natural distribution
  shifts in image classification. In Advances in Neural Information
  Processing Systems 33: Annual Conference on Neural Information
  Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual,
  Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p4.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Tarvainen and Valpola (2017) A. Tarvainen and H. Valpola Mean teachers
  are better role models: weight-averaged consistency targets improve
  semi-supervised deep learning results. In Advances in Neural
  Information Processing Systems 30: Annual Conference on Neural
  Information Processing Systems 2017, December 4-9, 2017, Long Beach,
  CA, USA, pp. 1195–1204. Cited by:
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Tian et al. (2023a) J. Tian, X. Dai, C. Ma, Z. He, Y. Liu, and Z. Kira
  Trainable projected gradient method for robust fine-tuning. In
  IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
  2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 7836–7845. External
  Links: [Link](https://doi.org/10.1109/CVPR52729.2023.00757),
  [Document](https://dx.doi.org/10.1109/CVPR52729.2023.00757) Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Tian et al. (2023b) J. Tian, Y. Liu, J. S. Smith, and Z. Kira Fast
  trainable projection for robust fine-tuning. In Advances in Neural
  Information Processing Systems 36: Annual Conference on Neural
  Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA,
  USA, December 10 - 16, 2023, Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Tian (2022) Y. Tian Understanding deep contrastive learning via
  coordinate-wise optimization. In Advances in Neural Information
  Processing Systems 35: Annual Conference on Neural Information
  Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November
  28 - December 9, 2022, Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.1](#A3.SS1.p1.1 "C.1 Derivation of ℒ_"CL" ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§F.3](#A6.SS3.p1.1 "F.3 Future work: NTK and random-feature extensions of the theory ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§3.1](#S3.SS1.p2.1 "3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Titsias et al. (2020) M. K. Titsias, J. Schwarz, A. G. de G.
  Matthews, R. Pascanu, and Y. W. Teh Functional regularisation for
  continual learning with gaussian processes. In 8th International
  Conference on Learning Representations, ICLR 2020, Addis Ababa,
  Ethiopia, April 26-30, 2020, External Links:
  [Link](https://openreview.net/forum?id=HkxCzeHFDB) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Tschannen et al. (2025) M. Tschannen, A. A. Gritsenko, X. Wang, M. F.
  Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y.
  Xia, B. Mustafa, O. J. Hénaff, J. Harmsen, A. Steiner, and X. Zhai
  SigLIP 2: multilingual vision-language encoders with improved semantic
  understanding, localization, and dense features. CoRR abs/2502.14786.
  External Links: [Link](https://doi.org/10.48550/arXiv.2502.14786),
  [Document](https://dx.doi.org/10.48550/ARXIV.2502.14786), 2502.14786
  Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Vaswani et al. (2017) A. Vaswani, N. Shazeer, N. Parmar, J.
  Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin
  Attention is all you need. In Advances in Neural Information
  Processing Systems 30: Annual Conference on Neural Information
  Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA,
  pp. 5998–6008. Cited by:
  [§5.1.1](#S5.SS1.SSS1.Px2.p1.1 "Model Architecture. ‣ 5.1.1 Experimental Setup ‣ 5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wang et al. (2022a) F. Wang, D. Zhou, H. Ye, and D. Zhan FOSTER:
  feature boosting and compression for class-incremental learning. In
  Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv,
  Israel, October 23-27, 2022, Proceedings, Part XXV, Lecture Notes in
  Computer Science, Vol. 13685, pp. 398–414. External Links:
  [Link](https://doi.org/10.1007/978-3-031-19806-9%5C_23),
  [Document](https://dx.doi.org/10.1007/978-3-031-19806-9%5F23) Cited
  by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wang et al. (2019) H. Wang, S. Ge, Z. C. Lipton, and E. P. Xing
  Learning robust global representations by penalizing local predictive
  power. In Advances in Neural Information Processing Systems 32: Annual
  Conference on Neural Information Processing Systems 2019, NeurIPS
  2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 10506–10518.
  Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§5.2.1](#S5.SS2.SSS1.Px2.p1.1 "Datasets and Evaluation. ‣ 5.2.1 Experimental Setup ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wang and Isola (2020) T. Wang and P. Isola Understanding contrastive
  representation learning through alignment and uniformity on the
  hypersphere. In Proceedings of the 37th International Conference on
  Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event,
  Proceedings of Machine Learning Research, Vol. 119, pp. 9929–9939.
  External Links: [Link](http://proceedings.mlr.press/v119/wang20k.html)
  Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wang et al. (2022b) Z. Wang, N. Codella, Y. Chen, L. Zhou, X. Dai, B.
  Xiao, J. Yang, H. You, K. Chang, S. Chang, and L. Yuan Multimodal
  adaptive distillation for leveraging unimodal encoders for
  vision-language tasks. CoRR abs/2204.10496. External Links:
  [Link](https://doi.org/10.48550/arXiv.2204.10496),
  [Document](https://dx.doi.org/10.48550/ARXIV.2204.10496), 2204.10496
  Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wortsman et al. (2022a) M. Wortsman, G. Ilharco, S. Y. Gadre, R.
  Roelofs, R. G. Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y.
  Carmon, S. Kornblith, and L. Schmidt Model soups: averaging weights of
  multiple fine-tuned models improves accuracy without increasing
  inference time. In International Conference on Machine Learning, ICML
  2022, 17-23 July 2022, Baltimore, Maryland, USA, Proceedings of
  Machine Learning Research, Vol. 162, pp. 23965–23998. External Links:
  [Link](https://proceedings.mlr.press/v162/wortsman22a.html) Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wortsman et al. (2022b) M. Wortsman, G. Ilharco, J. W. Kim, M. Li, S.
  Kornblith, R. Roelofs, R. G. Lopes, H. Hajishirzi, A. Farhadi, H.
  Namkoong, and L. Schmidt Robust fine-tuning of zero-shot models. In
  IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
  2022, New Orleans, LA, USA, June 18-24, 2022, pp. 7949–7961. External
  Links: [Link](https://doi.org/10.1109/CVPR52688.2022.00780),
  [Document](https://dx.doi.org/10.1109/CVPR52688.2022.00780) Cited by:
  [§A.4](#A1.SS4.p1.1 "A.4 Robust Finetuning of CLIP ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§A.6](#A1.SS6.p1.1 "A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p2.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§1](#S1.p3.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Wu et al. (2023) K. Wu, H. Peng, Z. Zhou, B. Xiao, M. Liu, L. Yuan, H.
  Xuan, M. Valenzuela, X. S. Chen, X. Wang, H. Chao, and H. Hu TinyCLIP:
  CLIP distillation via affinity mimicking and weight inheritance. In
  IEEE/CVF International Conference on Computer Vision, ICCV 2023,
  Paris, France, October 1-6, 2023, pp. 21913–21923. External Links:
  [Link](https://doi.org/10.1109/ICCV51070.2023.02008),
  [Document](https://dx.doi.org/10.1109/ICCV51070.2023.02008) Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Xu et al. (2024a) H. Xu, S. Xie, X. E. Tan, P. Huang, R. Howes, V.
  Sharma, S. Li, G. Ghosh, L. Zettlemoyer, and C. Feichtenhofer
  Demystifying CLIP data. In The Twelfth International Conference on
  Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024,
  External Links: [Link](https://openreview.net/forum?id=5BCFlnfE1g)
  Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Xu et al. (2024b) H. Xu, S. Xie, X. E. Tan, P. Huang, R. Howes, V.
  Sharma, S. Li, G. Ghosh, L. Zettlemoyer, and C. Feichtenhofer
  Demystifying CLIP data. In The Twelfth International Conference on
  Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024,
  External Links: [Link](https://openreview.net/forum?id=5BCFlnfE1g)
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Xue et al. (2023) Y. Xue, S. Joshi, E. Gan, P. Chen, and B.
  Mirzasoleiman Which features are learnt by contrastive learning? on
  the role of simplicity bias in class collapse and feature suppression.
  In International Conference on Machine Learning, ICML 2023, 23-29 July
  2023, Honolulu, Hawaii, USA, Proceedings of Machine Learning Research,
  Vol. 202, pp. 38938–38970. External Links:
  [Link](https://proceedings.mlr.press/v202/xue23d.html) Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Xue et al. (2024) Y. Xue, S. Joshi, D. Nguyen, and B. Mirzasoleiman
  Understanding the robustness of multi-modal contrastive learning to
  distribution shift. In The Twelfth International Conference on
  Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024,
  External Links: [Link](https://openreview.net/forum?id=rtl4XnJYBh)
  Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Theory of Contrastive Learning ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.1](#A3.SS1.p1.1 "C.1 Derivation of ℒ_"CL" ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.7](#A3.SS7.SSS0.Px1.p1.1 "Preservation of Cross-Class Knowledge. ‣ C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.7](#A3.SS7.SSS0.Px2.p1.1 "Informed Smoothing via Pretrained Similarities. ‣ C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.7](#A3.SS7.SSS0.Px3.p2.1 "Robustness Through Feature Independence. ‣ C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.7](#A3.SS7.SSS0.Px3.p3.1 "Robustness Through Feature Independence. ‣ C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.7](#A3.SS7.p1.1 "C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§F.3](#A6.SS3.p1.1 "F.3 Future work: NTK and random-feature extensions of the theory ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§3.1](#S3.SS1.p2.1 "3.1 Problem Setting and Preliminaries ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Yan et al. (2021) S. Yan, J. Xie, and X. He DER: dynamically
  expandable representation for class incremental learning. In IEEE
  Conference on Computer Vision and Pattern Recognition, CVPR 2021,
  virtual, June 19-25, 2021, pp. 3014–3023. External Links:
  [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Yan%5C_DER%5C_Dynamically%5C_Expandable%5C_Representation%5C_for%5C_Class%5C_Incremental%5C_Learning%5C_CVPR%5C_2021%5C_paper.html),
  [Document](https://dx.doi.org/10.1109/CVPR46437.2021.00303) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Yang et al. (2024a) C. Yang, Z. An, L. Huang, J. Bi, X. Yu, H.
  Yang, B. Diao, and Y. Xu CLIP-KD: an empirical study of CLIP model
  distillation. In IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024,
  pp. 15952–15962. External Links:
  [Link](https://doi.org/10.1109/CVPR52733.2024.01510),
  [Document](https://dx.doi.org/10.1109/CVPR52733.2024.01510) Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Yang et al. (2024b) Z. Yang, A. Zhang, S. Wiseman, X. Kong, K. Ye,
  and D. Yin Memory retaining finetuning via distillation. In NeurIPS
  2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles
  and Scalability, External Links:
  [Link](https://openreview.net/forum?id=NEYvQUr6em) Cited by:
  [§3.3](#S3.SS3.p1.1 "3.3 Closed-Form Solutions ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Yu et al. (2022) J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M.
  Seyedhosseini, and Y. Wu CoCa: contrastive captioners are image-text
  foundation models. Trans. Mach. Learn. Res. 2022. External Links:
  [Link](https://openreview.net/forum?id=Ee277P3AYC) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Yuan et al. (2021) X. Yuan, Z. Lin, J. Kuen, J. Zhang, Y. Wang, M.
  Maire, A. Kale, and B. Faieta Multimodal contrastive training for
  visual representation learning. In IEEE Conference on Computer Vision
  and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021,
  pp. 6995–7004. External Links:
  [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Yuan%5C_Multimodal%5C_Contrastive%5C_Training%5C_for%5C_Visual%5C_Representation%5C_Learning%5C_CVPR%5C_2021%5C_paper.html),
  [Document](https://dx.doi.org/10.1109/CVPR46437.2021.00692) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zenke et al. (2017) F. Zenke, B. Poole, and S. Ganguli Continual
  learning through synaptic intelligence. In Proceedings of the 34th
  International Conference on Machine Learning, ICML 2017, Sydney, NSW,
  Australia, 6-11 August 2017, Proceedings of Machine Learning Research,
  Vol. 70, pp. 3987–3995. External Links:
  [Link](http://proceedings.mlr.press/v70/zenke17a.html) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Finetuning, Forgetting, and Regularization ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhai et al. (2023) X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer
  Sigmoid loss for language image pre-training. In IEEE/CVF
  International Conference on Computer Vision, ICCV 2023, Paris, France,
  October 1-6, 2023, pp. 11941–11952. External Links:
  [Link](https://doi.org/10.1109/ICCV51070.2023.01100),
  [Document](https://dx.doi.org/10.1109/ICCV51070.2023.01100) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p1.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhang et al. (2019) L. Zhang, J. Song, A. Gao, J. Chen, C. Bao, and K.
  Ma Be your own teacher: improve the performance of convolutional
  neural networks via self distillation. In 2019 IEEE/CVF International
  Conference on Computer Vision, ICCV 2019, Seoul, Korea (South),
  October 27 - November 2, 2019, pp. 3712–3721. External Links:
  [Link](https://doi.org/10.1109/ICCV.2019.00381),
  [Document](https://dx.doi.org/10.1109/ICCV.2019.00381) Cited by:
  [§A.5](#A1.SS5.p1.1 "A.5 Knowledge Distillation and Self-Distillation ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§2](#S2.p2.1 "2 Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhang et al. (2022a) M. Zhang, N. S. Sohoni, H. R. Zhang, C. Finn,
  and C. Ré Correct-n-contrast: a contrastive approach for improving
  robustness to spurious correlations. In International Conference on
  Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland,
  USA, Proceedings of Machine Learning Research, Vol. 162,
  pp. 26484–26516. External Links:
  [Link](https://proceedings.mlr.press/v162/zhang22z.html) Cited by:
  [§5.1.1](#S5.SS1.SSS1.Px4.p1.1 "Finetuning Strategies. ‣ 5.1.1 Experimental Setup ‣ 5.1 Synthetic Experiment ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhang et al. (2022b) R. Zhang, Z. Guo, W. Zhang, K. Li, X. Miao, B.
  Cui, Y. Qiao, P. Gao, and H. Li PointCLIP: point cloud understanding
  by CLIP. In IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022,
  pp. 8542–8552. External Links:
  [Link](https://doi.org/10.1109/CVPR52688.2022.00836),
  [Document](https://dx.doi.org/10.1109/CVPR52688.2022.00836) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhang et al. (2022c) Y. Zhang, H. Jiang, Y. Miura, C. D. Manning,
  and C. P. Langlotz Contrastive learning of medical visual
  representations from paired images and text. In Proceedings of the
  Machine Learning for Healthcare Conference, MLHC 2022, 5-6 August
  2022, Durham, NC, USA, Proceedings of Machine Learning Research, Vol.
  182, pp. 2–25. External Links:
  [Link](https://proceedings.mlr.press/v182/zhang22a.html) Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Contrastive Language-Image Pretraining ‣ Appendix A Additional Related Work ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhang and Sabuncu (2020) Z. Zhang and M. R. Sabuncu Self-distillation
  as instance-specific label smoothing. In Advances in Neural
  Information Processing Systems 33: Annual Conference on Neural
  Information Processing Systems 2020, NeurIPS 2020, December 6-12,
  2020, virtual, Cited by:
  [§C.7](#A3.SS7.SSS0.Px2.p1.1 "Informed Smoothing via Pretrained Similarities. ‣ C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
  [§C.7](#A3.SS7.p2.1 "C.7 Connection to Robustness via Inter-Class Feature Sharing ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhou et al. (2022) K. Zhou, J. Yang, C. C. Loy, and Z. Liu Learning to
  prompt for vision-language models. International Journal of Computer
  Vision (IJCV). Cited by:
  [§F.2](#A6.SS2.p1.1 "F.2 Relation to PEFT and prompt-based adaptation ‣ Appendix F Extended Discussion ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
- Zhu et al. (2024) D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny
  MiniGPT-4: enhancing vision-language understanding with advanced large
  language models. In The Twelfth International Conference on Learning
  Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, External
  Links: [Link](https://openreview.net/forum?id=1tZbq88f27) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

## Appendix

## Appendix A Additional Related Work

### A.1 Contrastive Language-Image Pretraining

Initial advancements in contrastive learning between vision and language
modalities were made by Virtex ([Desai and Johnson, 2021](#bib.bib79)),
ICMLM ([Sariyildiz et al., 2020](#bib.bib80)), and ConVIRT ([Zhang et
al., 2022c](#bib.bib81)). These early approaches laid the groundwork for
later models like CLIP ([Radford et al., 2021](#bib.bib12); [Ilharco et
al., 2021](#bib.bib82)) and ALIGN ([Jia et al., 2021](#bib.bib53)),
which scaled contrastive techniques to larger datasets and model
architectures. Subsequent work explores improved cross-modal interaction
and training recipes ([Yuan et al., 2021](#bib.bib54); [Yu et al.,
2022](#bib.bib55); [Fang et al., 2023](#bib.bib56)). Following these,
several open-weight contrastive models have been introduced to improve
CLIP’s performance and robustness ([Sun et al., 2023](#bib.bib85); [Zhai
et al., 2023](#bib.bib57); [Li et al., 2023a](#bib.bib86); [Fang et al.,
2024](#bib.bib88); [Xu et al., 2024a](#bib.bib87); [Schuhmann et al.,
2022](#bib.bib83)). For example, SigLIP ([Zhai et al.,
2023](#bib.bib57); [Tschannen et al., 2025](#bib.bib84)) modifies the
contrastive loss by using a sigmoid function instead of softmax, and
FLIP ([Li et al., 2023c](#bib.bib65)) integrates masking strategies to
accelerate training.

### A.2 Theory of Contrastive Learning

A rich theoretical literature analyzes contrastive learning from first
principles, characterizing when and why contrastive objectives recover
useful features and class structure ([Saunshi et al.,
2019](#bib.bib36)). The alignment–uniformity lens formalizes how pulling
positives together while spreading embeddings uniformly on the sphere
drives representation quality ([Wang and Isola, 2020](#bib.bib37)). For
tractability, many works study linearized or simplified contrastive
losses that replace log-exp with linear functions and show that their
gradients align with those of standard objectives up to reweighting,
enabling closed-form analysis and geometric insight ([Ji et al.,
2023](#bib.bib11); [Tian, 2022](#bib.bib13); [Nakada et al.,
2023](#bib.bib14); [Xue et al., 2024](#bib.bib15)). This linearized
viewpoint has proven effective in theoretical analyses across
self-supervised contrastive learning (CL) ([Ji et al.,
2023](#bib.bib11); [HaoChen et al., 2022](#bib.bib22); [HaoChen and Ma,
2023](#bib.bib23); [Shen et al., 2022a](#bib.bib24)), multimodal
contrastive learning (MMCL) ([Nakada et al., 2023](#bib.bib14)),
non-contrastive methods ([Liu et al., 2022](#bib.bib25)), and supervised
CL ([Xue et al., 2023](#bib.bib26)). Complementing these results,
large-scale empirical studies suggest that many design choices of
popular losses (e.g., log-exp, cosine similarity) are not essential for
effective representation learning ([Garrido et al., 2023](#bib.bib27)).

### A.3 Finetuning, Forgetting, and Regularization

Catastrophic forgetting, adapting to new data at the expense of prior
knowledge, has long been recognized as a central challenge in sequential
and transfer learning ([McCloskey and Cohen, 1989](#bib.bib4); [French,
1999](#bib.bib1)). Mitigation strategies include: (i) regularization,
which constrains parameter updates via importance penalties or output
consistency ([Kirkpatrick et al., 2017](#bib.bib3); [Zenke et al.,
2017](#bib.bib9); [Li and Hoiem, 2018](#bib.bib35)); (ii) replay, which
mixes current data with stored or synthesized memories ([Robins,
1995](#bib.bib7); [Rebuffi et al., 2017](#bib.bib6); [Aljundi et al.,
2019](#bib.bib38)); and (iii) architectural growth, which expands
capacity and distills across modules ([Rusu et al., 2016](#bib.bib8);
[Yan et al., 2021](#bib.bib39); [Wang et al., 2022a](#bib.bib40)). L2-SP
([Li et al., 2018](#bib.bib16)) tethers the solution to the pretrained
initialization via weight-space regularization, while output-space
regularizers distill prior behaviors during adaptation ([Li and Hoiem,
2018](#bib.bib35)). Additionally, parameter-efficient finetuning methods
such as adapters ([Houlsby et al., 2019](#bib.bib66)) and prefix tuning
([Li and Liang, 2021](#bib.bib67)) enable task adaptation without full
model updates, thus mitigating forgetting. Among these, Low-Rank
Adaptation (LoRA) ([Hu et al., 2022](#bib.bib68)) has gained prominence
for finetuning large language models by injecting trainable low-rank
matrices into existing weights, achieving competitive performance with
reduced parameter updates and minimal forgetting. Further work explores
functional regularization ([Titsias et al., 2020](#bib.bib69)) and
knowledge-preserving contrastive losses ([Jung et al.,
2020](#bib.bib70)) to encourage feature stability. As model sizes grow,
scalable and minimally invasive adaptation techniques, balancing
plasticity and stability, remain critical to continual and transfer
learning paradigms.

### A.4 Robust Finetuning of CLIP

Robustness evaluates how well models maintain performance under
distribution shifts, which can include synthetic corruptions ([Hendrycks
and Dietterich, 2019](#bib.bib58)) as well as real-world variations in
viewpoint, style, and time ([Barbu et al., 2019](#bib.bib59); [Hendrycks
et al., 2021a](#bib.bib60); [Wang et al., 2019](#bib.bib61); [Recht et
al., 2019](#bib.bib62)). A standard protocol for evaluating CLIP-like
models, proposed by ([Taori et al., 2020](#bib.bib63)), involves
finetuning on ImageNet and measuring transfer performance on a suite of
realistic OOD sets (ImageNet-V2, -A, -R, -Sketch, and ObjectNet), which
is now standard practice. This evaluation highlights a central
challenge: naive finetuning methods like Linear Probing (LP), which only
trains a classification head, or Direct Full finetuning, which updates
all parameters, often create a trade-off between in-distribution (ID)
performance and OOD robustness. To address this, a diverse array of
robust finetuning techniques has been developed. A prominent line of
work involves post-hoc averaging or interpolating model weights. For
instance, WiSE-FT ([Wortsman et al., 2022b](#bib.bib20)) averages the
weights of the zero-shot and a fully finetuned model, while Model
Soup ([Wortsman et al., 2022a](#bib.bib52)) averages the weights of
multiple models found through a hyperparameter search. This concept is
extended by Model Stock ([Jang et al., 2024](#bib.bib71)), which
efficiently builds and averages a diverse set of minimally adapted
models. Other post-hoc methods include TPGM ([Tian et al.,
2023a](#bib.bib72)) and its efficient successor Fast TPGM ([Tian et al.,
2023b](#bib.bib73)), which project finetuned weights back towards the
initial weights, and DaWin ([Oh et al., 2025](#bib.bib74)), which
introduces a training-free, dynamic interpolation where the mixing
coefficient is decided on a per-sample basis using predictive entropy.

Beyond post-hoc modifications, many methods introduce regularization
during the finetuning process itself. These can constrain the model in
weight-space, such as L2-SP ([Li et al., 2018](#bib.bib16)) which
penalizes weight deviation, or by maintaining an EMA of model parameters
to find smoother, more robust solutions. Others operate in the
output-space, where Knowledge Distillation (KD) ([Hinton et al.,
2015](#bib.bib2)) aligns the student’s predictions with the robust
zero-shot teacher. A particularly relevant strategy for Vision-Language
Models is using the text modality for guidance. This includes continuing
contrastive learning with supervised image-text pairs as in
Finetune-Like-You-Pretrain (FLYP) ([Goyal et al., 2023](#bib.bib28)),
aligning with fixed context-specific prompts in CAR-FT ([Mao et al.,
2024](#bib.bib75)), regularizing the model’s energy function using
random texts to preserve broad semantic alignment in Lipsum-FT ([Nam et
al., 2024](#bib.bib77)), or improving discrimination with both positive
and negative prompts as in CLIPood ([Shu et al., 2023](#bib.bib76)).
Alternative strategies modify the training pipeline, such as the
two-stage LP-FT approach ([Kumar et al., 2022](#bib.bib64)) which first
finds a good head via linear probing before full finetuning. More
advanced methods like CaRot ([Oh et al., 2024](#bib.bib17)) aim to
simultaneously improve OOD accuracy and confidence calibration through a
principled combination of contrastive learning and novel regularization
terms.

### A.5 Knowledge Distillation and Self-Distillation

Knowledge Distillation (KD) was initially introduced for compression,
where a smaller student learns from a larger teacher’s outputs ([Hinton
et al., 2015](#bib.bib2)). The same principle underpins continual and
transfer learning, where a pretrained model guides finetuning to
preserve capabilities, often termed Learning without Forgetting (LwF)
([Li and Hoiem, 2018](#bib.bib35)). Self-distillation (SD) is a special
case where the model learns from its own initial state ([Zhang et al.,
2019](#bib.bib10); [Mobahi et al., 2020](#bib.bib5)). Beyond
single-modality SD, multimodal KD aligns internal signals and outputs to
preserve cross-modal structure ([Fang et al., 2021](#bib.bib41); [Wang
et al., 2022b](#bib.bib42); [Li et al., 2023b](#bib.bib43); [Liang et
al., 2023](#bib.bib44); [Li et al., 2024](#bib.bib45)), with recent work
demonstrating effective CLIP distillation via affinity matching and
weight inheritance ([Wu et al., 2023](#bib.bib46); [Yang et al.,
2024a](#bib.bib78)).

### A.6 Dynamic Teachers, Weight Averaging, and Mode Connectivity

Temporal ensembling and EMA teachers stabilize training and improve
targets ([Laine and Aila, 2017](#bib.bib47); [Tarvainen and Valpola,
2017](#bib.bib48)), and they underpin momentum-encoder methods in
self-supervised learning ([He et al., 2020](#bib.bib49); [Grill et al.,
2020](#bib.bib50); [Caron et al., 2021](#bib.bib51)). Separately, model
averaging and linear mode connectivity suggest that interpolations and
averages often lie in flat, low-loss regions and improve robustness
([Izmailov et al., 2018](#bib.bib21); [Frankle and Carbin,
2019](#bib.bib19)). Wise-FT leverages interpolation between pretrained
and finetuned weights to strengthen OOD performance ([Wortsman et al.,
2022b](#bib.bib20); [Wortsman et al., 2022a](#bib.bib52)).

## Appendix B Additional Experiments and Ablations

##### Experimental Protocol.

We use the same seeds, and hyperparameter configurations as in the main
experiments, varying only the stated factor per ablation.

##### Additional Experimental Results.

To further demonstrate the generalizability of our method, we present
results using the CLIP RN50 and ViT-L/14 backbones. A summary of these
experiments, including ViT-B/16 for comparison, is provided in
Table [3](#S5.T3 "Table 3 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
with detailed results reported below.

|               |       |       |       |       |       |        |       |
|---------------|-------|-------|-------|-------|-------|--------|-------|
| Method        | IN    | IN-V2 | IN-R  | IN-A  | IN-S  | ObjNet | Avg.  |
| ZS            | 0.057 | 0.055 | 0.054 | 0.097 | 0.085 | 0.078  | 0.074 |
| LP-FT         | 0.051 | 0.089 | 0.061 | 0.205 | 0.166 | 0.212  | 0.147 |
| FLYP          | 0.064 | 0.117 | 0.097 | 0.244 | 0.220 | 0.238  | 0.184 |
| Lipsum-FT     | 0.038 | 0.052 | 0.043 | 0.129 | 0.102 | 0.132  | 0.091 |
| CaRot         | 0.047 | 0.037 | 0.058 | 0.124 | 0.070 | 0.108  | 0.079 |
| TRACER (Ours) | 0.045 | 0.039 | 0.041 | 0.104 | 0.078 | 0.103  | 0.073 |

Table 5: ImageNet calibration (ECE) on ViT-B/16. We report ECE
($`\downarrow`$) on ImageNet and its distribution shift variants. In
each column, the best value is bold and the second-best is underlined.

|                   |        |        |        |        |        |           |             |
|-------------------|--------|--------|--------|--------|--------|-----------|-------------|
| Acc.$`\uparrow`$  |        |        |        |        |        |           |             |
| Method            | IN     | IN-V2  | IN-R   | IN-A   | IN-S   | ObjectNet | Avg. shifts |
| ZS                | 59.83  | 52.90  | 60.72  | 23.25  | 35.45  | 40.27     | 42.52       |
| FT                | 76.21  | 64.87  | 50.66  | 18.11  | 33.90  | 42.32     | 41.97       |
| LP-FT             | 76.25  | 64.48  | 49.55  | 18.60  | 33.33  | 42.13     | 41.62       |
| FLYP              | 76.16  | 65.10  | 51.55  | 20.08  | 34.24  | 42.53     | 42.70       |
| CaRot             | 76.12  | 65.36  | 52.16  | 19.32  | 34.05  | 42.67     | 42.71       |
| TRACER (Ours)     | 76.48  | 65.58  | 51.54  | 19.52  | 34.34  | 42.66     | 42.73       |
| ECE$`\downarrow`$ |        |        |        |        |        |           |             |
| ZS                | 0.0624 | 0.0559 | 0.0530 | 0.2048 | 0.0740 | 0.0899    | 0.0955      |
| FT                | 0.0983 | 0.1623 | 0.1860 | 0.4692 | 0.2824 | 0.3023    | 0.2804      |
| LP-FT             | 0.1042 | 0.1759 | 0.2709 | 0.5184 | 0.3520 | 0.3197    | 0.3274      |
| FLYP              | 0.0516 | 0.0872 | 0.1439 | 0.3872 | 0.2021 | 0.2432    | 0.2127      |
| CaRot             | 0.0471 | 0.0601 | 0.0948 | 0.3435 | 0.3435 | 0.2127    | 0.2109      |
| TRACER (Ours)     | 0.0470 | 0.0564 | 0.1176 | 0.3456 | 0.1741 | 0.2097    | 0.1807      |

Table 6: ImageNet results on CLIP ResNet50

|                   |        |        |        |        |        |           |             |
|-------------------|--------|--------|--------|--------|--------|-----------|-------------|
| Acc.$`\uparrow`$  |        |        |        |        |        |           |             |
| Method            | IN     | IN-V2  | IN-R   | IN-A   | IN-S   | ObjectNet | Avg. shifts |
| ZS                | 75.55  | 69.85  | 87.85  | 70.76  | 59.61  | 66.59     | 70.93       |
| FT                | 84.74  | 75.32  | 75.36  | 55.65  | 54.44  | 59.76     | 64.11       |
| LP-FT             | 85.26  | 76.76  | 80.21  | 55.95  | 56.84  | 60.12     | 65.98       |
| FLYP              | 86.19  | 78.21  | 83.81  | 68.85  | 60.20  | 66.15     | 71.44       |
| CaRot             | 86.95  | 79.28  | 87.96  | 72.68  | 62.66  | 68.05     | 74.13       |
| TRACER (Ours)     | 86.27  | 78.54  | 89.70  | 74.87  | 63.71  | 69.76     | 75.32       |
| ECE$`\downarrow`$ |        |        |        |        |        |           |             |
| ZS                | 0.0590 | 0.0686 | 0.0339 | 0.0640 | 0.1037 | 0.0852    | 0.0711      |
| FT                | 0.1056 | 0.1741 | 0.1613 | 0.3151 | 0.3234 | 0.2865    | 0.2521      |
| LP-FT             | 0.0993 | 0.1531 | 0.0872 | 0.2593 | 0.2613 | 0.2572    | 0.2036      |
| FLYP              | 0.0729 | 0.1219 | 0.0621 | 0.1443 | 0.2164 | 0.1903    | 0.1470      |
| CaRot             | 0.0349 | 0.0634 | 0.0353 | 0.0732 | 0.0914 | 0.1051    | 0.0737      |
| TRACER (Ours)     | 0.0507 | 0.0581 | 0.0442 | 0.0665 | 0.1052 | 0.0918    | 0.0732      |

Table 7: ImageNet results on CLIP ViT-L/14

![Refer to caption](2605.29380v1/teacher_student_kl_freq1_v2.png)

(a) Teacher–Student KL

![Refer to caption](2605.29380v1/teacher_entropy_freq1_v2.png)

(b) Teacher Entropy

![Refer to caption](2605.29380v1/teacher_confidence_freq1_v2.png)

(c) Teacher Confidence

Figure 6: Comparison of Teacher Dynamics (Update Frequency = 1). We
track the evolution of the teacher model for CaRot (EMA) and TRACER
(WMA) when updated at every step. The EMA teacher (blue) rapidly
collapses onto the student (KL $`\to`$ 0), losing its regularizing
capability. The WMA teacher (orange) maintains a persistent, stable gap,
providing continuous regularization without needing brittle update
schedules.

##### Ablation 1: Multi-perspective distillation.

The ablation study on multi-perspective distillation
(Table [8](#A2.T8 "Table 8 ‣ Ablation 1: Multi-perspective distillation. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
quantifies the contribution of each perspective to out-of-distribution
(OOD) accuracy and calibration. Results show that CRD and FD emerge as
the strongest individual components for OOD accuracy and ECE,
respectively, while combining all four perspectives yields the best
overall performance and remains among the top performers on OOD metrics.
These findings highlight the complementary nature of the terms: FD
stabilizes features, CRD preserves batch-level relational structure, ICL
enriches mutual information in the teacher’s space, and CrossKD blends
relational and interactive cues.

|  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
| Acc.$`\uparrow`$ |  |  |  |  |  |  |  |  |  |  |  |
| $`\mathcal{L}_{\text{FD}}`$ | $`\mathcal{L}_{\text{CrossKD}}`$ | $`\mathcal{L}_{\text{ICL}}`$ | $`\mathcal{L}_{\text{CRD}}`$ | IN | IN-V2 | IN-R | IN-A | IN-S | ObjectNet | Avg. shifts | Avg. All |
| $`-`$ | $`-`$ | $`-`$ | $`-`$ | 82.69 | 72.73 | 71.35 | 48.52 | 49.84 | 54.86 | 59.40 | 63.33 |
| $`-`$ | $`-`$ | $`-`$ | ✓ | 83.17 | 74.29 | 77.75 | 53.09 | 53.03 | 57.46 | 63.12 | 66.47 |
| $`-`$ | $`-`$ | ✓ | $`-`$ | 82.50 | 73.13 | 72.12 | 49.23 | 50.03 | 55.26 | 59.95 | 63.71 |
| $`-`$ | $`-`$ | ✓ | ✓ | 83.23 | 74.31 | 76.50 | 52.39 | 52.53 | 57.00 | 62.55 | 65.99 |
| $`-`$ | ✓ | $`-`$ | $`-`$ | 83.19 | 74.04 | 74.67 | 50.65 | 51.39 | 56.40 | 61.43 | 65.06 |
| $`-`$ | ✓ | $`-`$ | ✓ | 83.08 | 74.40 | 78.68 | 53.53 | 53.37 | 57.45 | 63.49 | 66.75 |
| $`-`$ | ✓ | ✓ | $`-`$ | 83.03 | 73.95 | 74.11 | 50.60 | 51.28 | 55.77 | 61.14 | 64.79 |
| $`-`$ | ✓ | ✓ | ✓ | 83.27 | 74.48 | 77.60 | 52.76 | 52.94 | 57.28 | 63.01 | 66.39 |
| ✓ | $`-`$ | $`-`$ | $`-`$ | 83.06 | 74.16 | 78.14 | 54.39 | 53.14 | 57.79 | 63.52 | 66.78 |
| ✓ | $`-`$ | $`-`$ | ✓ | 82.45 | 73.91 | 79.67 | 54.88 | 53.93 | 58.02 | 64.08 | 67.14 |
| ✓ | $`-`$ | ✓ | $`-`$ | 83.08 | 74.39 | 77.59 | 53.84 | 53.10 | 57.59 | 63.30 | 66.60 |
| ✓ | $`-`$ | ✓ | ✓ | 82.92 | 74.40 | 79.21 | 54.61 | 53.75 | 57.99 | 63.99 | 67.15 |
| ✓ | ✓ | $`-`$ | $`-`$ | 83.01 | 74.21 | 78.91 | 54.09 | 53.28 | 58.04 | 63.71 | 66.92 |
| ✓ | ✓ | $`-`$ | ✓ | 82.27 | 73.71 | 79.81 | 54.65 | 53.82 | 58.14 | 64.03 | 67.07 |
| ✓ | ✓ | ✓ | $`-`$ | 83.06 | 74.38 | 78.38 | 54.07 | 53.25 | 57.86 | 63.59 | 66.83 |
| ✓ | ✓ | ✓ | ✓ | 82.81 | 73.94 | 79.55 | 54.83 | 53.96 | 58.02 | 64.06 | 67.19 |

|  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
| ECE.$`\downarrow`$ |  |  |  |  |  |  |  |  |  |  |  |
| $`\mathcal{L}_{\text{FD}}`$ | $`\mathcal{L}_{\text{CrossKD}}`$ | $`\mathcal{L}_{\text{ICL}}`$ | $`\mathcal{L}_{\text{CRD}}`$ | IN | IN-V2 | IN-R | IN-A | IN-S | ObjectNet | Avg. shifts | Avg. All |
| $`-`$ | $`-`$ | $`-`$ | $`-`$ | 0.0635 | 0.1171 | 0.0967 | 0.2435 | 0.2200 | 0.2383 | 0.1836 | 0.1632 |
| $`-`$ | $`-`$ | $`-`$ | ✓ | 0.0415 | 0.0412 | 0.0413 | 0.1328 | 0.0860 | 0.1211 | 0.0845 | 0.0773 |
| $`-`$ | $`-`$ | ✓ | $`-`$ | 0.0585 | 0.1000 | 0.0817 | 0.2117 | 0.1974 | 0.2168 | 0.1615 | 0.1444 |
| $`-`$ | $`-`$ | ✓ | ✓ | 0.0393 | 0.0523 | 0.0429 | 0.1534 | 0.1111 | 0.1441 | 0.1008 | 0.0905 |
| $`-`$ | ✓ | $`-`$ | $`-`$ | 0.0483 | 0.0830 | 0.0662 | 0.2007 | 0.1660 | 0.1918 | 0.1415 | 0.1260 |
| $`-`$ | ✓ | $`-`$ | ✓ | 0.0453 | 0.0374 | 0.0434 | 0.1141 | 0.0753 | 0.1096 | 0.0760 | 0.0709 |
| $`-`$ | ✓ | ✓ | $`-`$ | 0.0507 | 0.0824 | 0.0691 | 0.2007 | 0.1684 | 0.1968 | 0.1435 | 0.1280 |
| $`-`$ | ✓ | ✓ | ✓ | 0.0401 | 0.0442 | 0.0392 | 0.1345 | 0.0897 | 0.1260 | 0.0867 | 0.0790 |
| ✓ | $`-`$ | $`-`$ | $`-`$ | 0.0430 | 0.0674 | 0.0479 | 0.1592 | 0.1383 | 0.1661 | 0.1158 | 0.1037 |
| ✓ | $`-`$ | $`-`$ | ✓ | 0.0474 | 0.0380 | 0.0455 | 0.1034 | 0.0720 | 0.0975 | 0.0713 | 0.0673 |
| ✓ | $`-`$ | ✓ | $`-`$ | 0.0436 | 0.0684 | 0.0482 | 0.1632 | 0.1384 | 0.1691 | 0.1175 | 0.1052 |
| ✓ | $`-`$ | ✓ | ✓ | 0.0419 | 0.0454 | 0.0397 | 0.1157 | 0.0853 | 0.1162 | 0.0805 | 0.0740 |
| ✓ | ✓ | $`-`$ | $`-`$ | 0.0399 | 0.0537 | 0.0398 | 0.1340 | 0.1082 | 0.1374 | 0.0946 | 0.0855 |
| ✓ | ✓ | $`-`$ | ✓ | 0.0531 | 0.0404 | 0.0500 | 0.0936 | 0.0703 | 0.0888 | 0.0686 | 0.0660 |
| ✓ | ✓ | ✓ | $`-`$ | 0.0402 | 0.0572 | 0.0418 | 0.1420 | 0.1176 | 0.1499 | 0.1017 | 0.0915 |
| ✓ | ✓ | ✓ | ✓ | 0.0446 | 0.0416 | 0.0430 | 0.1027 | 0.0757 | 0.1054 | 0.0737 | 0.0688 |

Table 8: Ablation of TRACER components across ImageNet (IN) and
distribution shifts. For each setting (row), accuracy (Acc.$`\uparrow`$)
is reported in %, and expected calibration error (ECE$`\downarrow`$) in
$`[0,1]`$. OOD Avg. is the mean over {IN-V2, IN-R, IN-A, IN-S,
ObjectNet}. Method names encode the presence of losses
$`(\mathcal{L}_{\text{FD}},\mathcal{L}_{\text{CrossKD}},\mathcal{L}_{\text{ICL}},\mathcal{L}_{\text{CRD}})`$
as ✓or $`-`$. Rows with only one loss term active are in gray.

##### Ablation 2: Distillation strength $`\lambda_{\text{SD}}`$.

The ablation on distillation strength $`\lambda_{\text{SD}}`$
(Table [9](#A2.T9 "Table 9 ‣ Ablation 2: Distillation strength 𝜆_"SD". ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
examines the balance between teacher influence and task adaptation. We
sweep
$`{\lambda_{\text{SD}}\in\{0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.2,1.5,2.0,3.0,4.0,5.0,10.0\}}`$.
Results indicate that moderate values of $`\lambda_{\text{SD}}`$
($`\approx 1.0`$–$`2.0`$) achieve the best OOD accuracy, while larger
values improve calibration by lowering ECE but slightly reduce
in-distribution (ID) accuracy. This aligns with our theory that stronger
distillation enhances calibration through teacher anchoring, whereas
moderate strength provides the optimal trade-off between adaptation and
preservation for OOD performance.

|                         |       |       |       |       |       |           |          |
|-------------------------|-------|-------|-------|-------|-------|-----------|----------|
| Acc.$`\uparrow`$ (%)    |       |       |       |       |       |           |          |
| $`\lambda_{\text{SD}}`$ | IN    | IN-V2 | IN-R  | IN-A  | IN-S  | ObjectNet | OOD Avg. |
| 10.0                    | 80.52 | 72.08 | 79.50 | 54.07 | 53.27 | 57.45     | 63.27    |
| 5.0                     | 81.08 | 72.54 | 79.79 | 54.37 | 53.64 | 57.65     | 63.60    |
| 4.0                     | 81.25 | 72.67 | 79.78 | 54.75 | 53.74 | 57.66     | 63.72    |
| 3.0                     | 81.58 | 73.12 | 79.90 | 54.83 | 53.84 | 57.75     | 63.89    |
| 2.0                     | 81.94 | 73.49 | 80.03 | 54.64 | 54.02 | 57.88     | 64.01    |
| 1.5                     | 82.27 | 73.52 | 79.72 | 55.28 | 53.99 | 58.08     | 64.12    |
| 1.2                     | 82.50 | 73.74 | 79.55 | 55.20 | 53.94 | 58.14     | 64.11    |
| 1.0                     | 82.70 | 74.07 | 79.64 | 54.87 | 53.85 | 58.08     | 64.10    |
| 0.7                     | 82.90 | 74.41 | 79.21 | 54.72 | 53.73 | 58.03     | 64.02    |
| 0.5                     | 83.16 | 74.31 | 78.76 | 54.13 | 53.52 | 57.76     | 63.70    |
| 0.4                     | 83.26 | 74.48 | 78.19 | 53.64 | 53.27 | 57.82     | 63.48    |
| 0.3                     | 83.28 | 74.52 | 77.53 | 53.55 | 52.91 | 57.44     | 63.19    |
| 0.2                     | 83.29 | 74.30 | 76.80 | 52.61 | 52.58 | 57.04     | 62.67    |
| 0.1                     | 83.25 | 74.08 | 75.34 | 51.52 | 51.89 | 56.49     | 61.86    |

|                         |        |        |        |        |        |           |          |
|-------------------------|--------|--------|--------|--------|--------|-----------|----------|
| ECE$`\downarrow`$       |        |        |        |        |        |           |          |
| $`\lambda_{\text{SD}}`$ | IN     | IN-V2  | IN-R   | IN-A   | IN-S   | ObjectNet | OOD Avg. |
| 10.0                    | 0.0637 | 0.0475 | 0.0621 | 0.0839 | 0.0725 | 0.0795    | 0.0691   |
| 5.0                     | 0.0631 | 0.0467 | 0.0600 | 0.0812 | 0.0700 | 0.0772    | 0.0670   |
| 4.0                     | 0.0606 | 0.0457 | 0.0583 | 0.0817 | 0.0705 | 0.0801    | 0.0673   |
| 3.0                     | 0.0590 | 0.0466 | 0.0566 | 0.0866 | 0.0701 | 0.0814    | 0.0683   |
| 2.0                     | 0.0547 | 0.0422 | 0.0528 | 0.0885 | 0.0682 | 0.0863    | 0.0676   |
| 1.5                     | 0.0511 | 0.0396 | 0.0490 | 0.0911 | 0.0707 | 0.0897    | 0.0680   |
| 1.2                     | 0.0484 | 0.0408 | 0.0455 | 0.0963 | 0.0713 | 0.0957    | 0.0699   |
| 1.0                     | 0.0465 | 0.0410 | 0.0445 | 0.1001 | 0.0742 | 0.1010    | 0.0722   |
| 0.7                     | 0.0419 | 0.0445 | 0.0416 | 0.1120 | 0.0841 | 0.1144    | 0.0793   |
| 0.5                     | 0.0409 | 0.0466 | 0.0390 | 0.1259 | 0.0953 | 0.1295    | 0.0873   |
| 0.4                     | 0.0394 | 0.0502 | 0.0415 | 0.1353 | 0.1030 | 0.1363    | 0.0933   |
| 0.3                     | 0.0397 | 0.0554 | 0.0424 | 0.1459 | 0.1161 | 0.1502    | 0.1020   |
| 0.2                     | 0.0421 | 0.0626 | 0.0463 | 0.1628 | 0.1341 | 0.1660    | 0.1144   |
| 0.1                     | 0.0470 | 0.0798 | 0.0601 | 0.1890 | 0.1594 | 0.1895    | 0.1356   |

Table 9: Ablation of distillation coefficient $`\lambda_{\text{SD}}`$
across ImageNet (IN) and distribution shifts. For each setting (row),
accuracy (Acc.$`\uparrow`$) is reported in %, and expected calibration
error (ECE$`\downarrow`$) in $`[0,1]`$. OOD Avg. is the mean over
{IN-V2, IN-R, IN-A, IN-S, ObjectNet}.

##### Ablation 3: Teacher update frequency.

The ablation on teacher update frequency
(Table [10](#A2.T10 "Table 10 ‣ Ablation 3: Teacher update frequency. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
investigates the trade-off between stability and plasticity in the
dynamic teacher. We vary the frequency from 1 to 2500 steps
($`\approx 1`$ epoch). The results show that updating every 50–100 steps
yields the highest OOD accuracy, whereas slower update schedules lead to
lower OOD ECE. These findings align with the dynamic-teacher analysis:
slower updates preserve early robustness and calibration, while faster
updates allow the teacher to better track the evolving task solution and
enhance accuracy.

|                      |       |       |       |       |       |           |          |
|----------------------|-------|-------|-------|-------|-------|-----------|----------|
| Acc.$`\uparrow`$ (%) |       |       |       |       |       |           |          |
| Update Freq.         | IN    | IN-V2 | IN-R  | IN-A  | IN-S  | ObjectNet | OOD Avg. |
| 2500                 | 81.38 | 72.97 | 79.83 | 55.05 | 53.88 | 58.17     | 63.98    |
| 1000                 | 81.90 | 73.27 | 79.71 | 54.93 | 54.21 | 58.26     | 64.08    |
| 500                  | 82.13 | 73.53 | 79.80 | 54.72 | 54.23 | 58.30     | 64.12    |
| 100                  | 82.54 | 73.92 | 79.76 | 55.09 | 54.00 | 58.31     | 64.22    |
| 50                   | 82.62 | 73.84 | 79.69 | 54.96 | 53.96 | 58.12     | 64.11    |
| 10                   | 82.57 | 74.01 | 79.58 | 54.96 | 53.88 | 58.00     | 64.09    |
| 5                    | 82.70 | 73.98 | 79.57 | 54.81 | 53.93 | 58.07     | 64.07    |
| 2                    | 82.73 | 74.12 | 79.57 | 54.51 | 53.99 | 58.09     | 64.06    |
| 1                    | 82.76 | 74.14 | 79.33 | 54.92 | 53.69 | 58.26     | 64.07    |

|                   |        |        |        |        |        |           |          |
|-------------------|--------|--------|--------|--------|--------|-----------|----------|
| ECE$`\downarrow`$ |        |        |        |        |        |           |          |
| Update Freq.      | IN     | IN-V2  | IN-R   | IN-A   | IN-S   | ObjectNet | OOD Avg. |
| 2500              | 0.0614 | 0.0426 | 0.0632 | 0.0839 | 0.0722 | 0.0764    | 0.0677   |
| 1000              | 0.0562 | 0.0434 | 0.0581 | 0.0908 | 0.0698 | 0.0825    | 0.0689   |
| 500               | 0.0524 | 0.0419 | 0.0548 | 0.0935 | 0.0677 | 0.0868    | 0.0689   |
| 100               | 0.0475 | 0.0410 | 0.0487 | 0.0985 | 0.0694 | 0.0920    | 0.0699   |
| 50                | 0.0481 | 0.0413 | 0.0471 | 0.0992 | 0.0713 | 0.0949    | 0.0708   |
| 10                | 0.0473 | 0.0408 | 0.0468 | 0.0983 | 0.0714 | 0.0978    | 0.0710   |
| 5                 | 0.0480 | 0.0400 | 0.0451 | 0.1001 | 0.0738 | 0.0975    | 0.0713   |
| 2                 | 0.0471 | 0.0409 | 0.0440 | 0.1034 | 0.0733 | 0.0990    | 0.0721   |
| 1                 | 0.0446 | 0.0394 | 0.0412 | 0.1041 | 0.0784 | 0.1030    | 0.0732   |

Table 10: Ablation of teacher update frequency in TRACER distillation
across ImageNet (IN) and distribution shifts. For each setting (row),
accuracy (Acc.$`\uparrow`$) is reported in %, and expected calibration
error (ECE$`\downarrow`$) in $`[0,1]`$. OOD Avg. is the mean over
{IN-V2, IN-R, IN-A, IN-S, ObjectNet}. The update frequency denotes the
number of training steps between each teacher model update from the
student; lower frequencies (e.g., 2-10 steps) result in a teacher that
closely follows the student’s trajectory providing fine-grained
regularization, while higher frequencies (e.g., 500-2500 steps) maintain
a more stable teacher that changes less frequently, providing stronger
regularization from earlier checkpoints and the initial pretrained
model.

##### Ablation 4: Beta kernel shape.

The ablation on the Beta kernel shape
(Table [11](#A2.T11 "Table 11 ‣ Ablation 4: Beta kernel shape. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
evaluates the role of endpoint-aware curricula. We vary
$`\beta\in\{0.2,0.5,0.7,0.9,1.0,1.5\}`$. Results show that smaller
$`\beta`$ values (0.2–0.5), which emphasize endpoints, enhance both OOD
accuracy and ECE by reinforcing strong early anchoring and late solution
emphasis. In contrast, larger $`\beta`$ values favor mid-trajectory
weighting, yielding marginal ID improvements at the cost of reduced OOD
gains. These findings suggest that arcsine-like weighting is
particularly effective for robust finetuning. Additional kernel families
are discussed in
§[C.5.1](#A3.SS5.SSS1 "C.5.1 WMA vs. EMA Teachers ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

|                      |       |       |       |       |       |           |          |
|----------------------|-------|-------|-------|-------|-------|-----------|----------|
| Acc.$`\uparrow`$ (%) |       |       |       |       |       |           |          |
| $`\beta`$            | IN    | IN-V2 | IN-R  | IN-A  | IN-S  | ObjectNet | OOD Avg. |
| 1.5                  | 83.19 | 74.38 | 77.15 | 52.43 | 52.95 | 57.06     | 62.79    |
| 1.0                  | 83.09 | 74.39 | 78.10 | 52.93 | 53.25 | 57.41     | 63.22    |
| 0.9                  | 83.14 | 74.26 | 78.28 | 53.59 | 53.35 | 57.41     | 63.38    |
| 0.7                  | 82.97 | 74.32 | 78.89 | 54.28 | 53.58 | 57.80     | 63.77    |
| 0.5                  | 82.76 | 74.14 | 79.33 | 54.92 | 53.69 | 58.26     | 64.07    |
| 0.2                  | 81.91 | 73.46 | 79.96 | 55.27 | 54.08 | 58.34     | 64.22    |

|                   |        |        |        |        |        |           |          |
|-------------------|--------|--------|--------|--------|--------|-----------|----------|
| ECE$`\downarrow`$ |        |        |        |        |        |           |          |
| $`\beta`$         | IN     | IN-V2  | IN-R   | IN-A   | IN-S   | ObjectNet | OOD Avg. |
| 1.5               | 0.0403 | 0.0485 | 0.0418 | 0.1433 | 0.1069 | 0.1418    | 0.0965   |
| 1.0               | 0.0407 | 0.0478 | 0.0393 | 0.1317 | 0.0957 | 0.1286    | 0.0886   |
| 0.9               | 0.0401 | 0.0477 | 0.0402 | 0.1280 | 0.0926 | 0.1262    | 0.0869   |
| 0.7               | 0.0416 | 0.0456 | 0.0388 | 0.1148 | 0.0856 | 0.1161    | 0.0802   |
| 0.5               | 0.0446 | 0.0394 | 0.0412 | 0.1041 | 0.0784 | 0.1030    | 0.0732   |
| 0.2               | 0.0548 | 0.0424 | 0.0542 | 0.0917 | 0.0670 | 0.0843    | 0.0679   |

Table 11: Ablation of $`\beta`$ value in Beta($`\beta`$, $`\beta`$)
distribution for teacher weighting in TRACER distillation across
ImageNet (IN) and distribution shifts. For each setting (row), accuracy
(Acc.$`\uparrow`$) is reported in %, and expected calibration error
(ECE$`\downarrow`$) in $`[0,1]`$. OOD Avg. is the mean over {IN-V2,
IN-R, IN-A, IN-S, ObjectNet}. The $`\beta`$ value controls the shape of
the distribution used for sampling teacher ensemble weights. Lower
$`\beta`$ values ($`<1`$) assign higher weights to the pretrained model
and early training steps, $`\beta=1`$ corresponds to uniform weighting,
while higher $`\beta`$ values ($`>1`$) emphasize intermediate training
steps and down-weight both the initial pretrained model and final
training steps.

## Appendix C Additional Theoretical Details

This section provides the full derivations, proofs, and detailed
geometric interpretations for the theoretical analysis presented in the
main paper.

### C.1 Derivation of $`\mathcal{L}_{\text{CL}}`$

We start with the linearized multimodal contrastive learning (MMCL) loss
function, which balances positive and negative pairs across a batch, as
commonly used in theoretical analyses ([Ji et al., 2023](#bib.bib11);
[Tian, 2022](#bib.bib13); [Nakada et al., 2023](#bib.bib14); [Xue et
al., 2024](#bib.bib15)). The original formulation is given by:

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{L}_{\text{MMCL}}(\mathbf{W}_{I},\mathbf{W}_{T})=\frac{1}{2n(n-1)}\sum_{i}\sum_{j\neq i}(s_{ij}-s_{ii})+\frac{1}{2n(n-1)}\sum_{i}\sum_{j\neq i}(s_{ji}-s_{ii})+\frac{\rho}{2}\|\mathbf{W}_{I}^{\top}\mathbf{W}_{T}\|_{F}^{2}
``` |  |

where
$`s_{ij}=(\mathbf{W}_{I}\mathbf{x}_{I}^{i})^{\top}(\mathbf{W}_{T}\mathbf{x}_{T}^{j})`$
represents the similarity score between image $`i`$ and text $`j`$.

Expanding the first term:

|  |  |  |
|----|----|----|
|  |
``` math
\frac{1}{2n(n-1)}\sum_{i}\sum_{j\neq i}(s_{ij}-s_{ii})=\frac{1}{2n(n-1)}\left[\sum_{i}\sum_{j\neq i}s_{ij}-\sum_{i}\sum_{j\neq i}s_{ii}\right]
``` |  |

Since for each $`i`$, there are $`(n-1)`$ values of $`j\neq i`$, the
second sub-sum simplifies:

|  |  |  |
|----|----|----|
|  |
``` math
=\frac{1}{2n(n-1)}\left[\sum_{i}\sum_{j\neq i}s_{ij}-(n-1)\sum_{i}s_{ii}\right]
``` |  |

Expanding the second term:

|  |  |  |
|----|----|----|
|  |
``` math
\frac{1}{2n(n-1)}\sum_{i}\sum_{j\neq i}(s_{ji}-s_{ii})=\frac{1}{2n(n-1)}\left[\sum_{i}\sum_{j\neq i}s_{ji}-\sum_{i}\sum_{j\neq i}s_{ii}\right]
``` |  |

Similarly, this becomes:

|  |  |  |
|----|----|----|
|  |
``` math
=\frac{1}{2n(n-1)}\left[\sum_{i}\sum_{j\neq i}s_{ji}-(n-1)\sum_{i}s_{ii}\right]
``` |  |

Combining both terms: Adding the first and second terms yields:

|  |  |  |
|----|----|----|
|  |
``` math
\frac{1}{2n(n-1)}\left[\sum_{i}\sum_{j\neq i}s_{ij}+\sum_{i}\sum_{j\neq i}s_{ji}-2(n-1)\sum_{i}s_{ii}\right]
``` |  |

Note that $`\sum_{i}\sum_{j\neq i}s_{ji}`$ is simply a re-indexing of
$`\sum_{j}\sum_{i\neq j}s_{ij}`$, which is equivalent to
$`\sum_{i}\sum_{j\neq i}s_{ij}`$. Therefore:

|  |  |  |
|----|----|----|
|  |
``` math
\sum_{i}\sum_{j\neq i}s_{ij}+\sum_{i}\sum_{j\neq i}s_{ji}=2\sum_{i}\sum_{j\neq i}s_{ij}
``` |  |

Substituting back into $`\mathcal{L}_{\text{MMCL}}`$:

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{L}_{\text{MMCL}}=\frac{1}{2n(n-1)}\left[2\sum_{i}\sum_{j\neq i}s_{ij}-2(n-1)\sum_{i}s_{ii}\right]+\frac{\rho}{2}\|\mathbf{W}_{I}^{\top}\mathbf{W}_{T}\|_{F}^{2}
``` |  |

|  |  |  |
|----|----|----|
|  |
``` math
=\frac{1}{n(n-1)}\left[\sum_{i}\sum_{j\neq i}s_{ij}-(n-1)\sum_{i}s_{ii}\right]+\frac{\rho}{2}\|\mathbf{W}_{I}^{\top}\mathbf{W}_{T}\|_{F}^{2}
``` |  |

We define the core contrastive alignment term as:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{CL}}=\sum_{i=1}^{n}\sum_{j\neq i}s_{ij}-(n-1)\sum_{i=1}^{n}s_{ii}.
``` |  | (4) |

And the regularization term as
$`R(\mathbf{W}_{I},\mathbf{W}_{T})=\frac{\rho}{2}\|\mathbf{W}_{I}^{\top}\mathbf{W}_{T}\|_{F}^{2}`$.
Thus, the total MMCL loss can be written as:

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{L}_{\text{MMCL}}=\frac{1}{n(n-1)}\mathcal{L}_{\text{CL}}+R(\mathbf{W}_{I},\mathbf{W}_{T})
``` |  |

### C.2 Reformulation of the Least-Squares Objective

We demonstrate how the contrastive alignment term
$`\mathcal{L}_{\text{CL}}`$ can be re-expressed as a matrix
least-squares problem, which is the foundation of our theoretical
analysis. Recall
$`\mathcal{L}_{\text{CL}}=\sum_{i=1}^{n}\sum_{j\neq i}s_{ij}-(n-1)\sum_{i=1}^{n}s_{ii}`$.
Let $`\mathbf{H}_{I}=\mathbf{W}_{I}\mathbf{X}_{I}`$ and
$`\mathbf{H}_{T}=\mathbf{W}_{T}^{0}\mathbf{X}_{T}`$. Then
$`s_{ij}=(\mathbf{H}_{I})_{i}^{\top}(\mathbf{H}_{T})_{j}`$. Let
$`\mathbf{S}=\mathbf{H}_{I}^{\top}\mathbf{H}_{T}`$. The sum of all
similarities is
$`\mathbf{1}^{\top}\mathbf{S}\mathbf{1}=\sum_{i,j}s_{ij}`$. The sum of
diagonal similarities is $`\Tr(\mathbf{S})=\sum_{i}s_{ii}`$. Then,
$`\sum_{i=1}^{n}\sum_{j\neq i}s_{ij}=\sum_{i,j}s_{ij}-\sum_{i}s_{ii}=\mathbf{1}^{\top}\mathbf{S}\mathbf{1}-\Tr(\mathbf{S})`$.
Substituting this into $`\mathcal{L}_{\text{CL}}`$:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathcal{L}_{\text{CL}}`$ | $`\displaystyle=(\mathbf{1}^{\top}\mathbf{S}\mathbf{1}-\Tr(\mathbf{S}))-(n-1)\Tr(\mathbf{S})`$ |  |
|  |  | $`\displaystyle=\mathbf{1}^{\top}\mathbf{S}\mathbf{1}-n\Tr(\mathbf{S})`$ |  |
|  |  | $`\displaystyle=\Tr(\mathbf{1}\mathbf{1}^{\top}\mathbf{S})-n\Tr(\mathbf{S})`$ |  |
|  |  | $`\displaystyle=\Tr((\mathbf{J}_{n}-n\mathbf{I}_{n})^{\top}\mathbf{S})`$ |  |
|  |  | $`\displaystyle=\Tr((\mathbf{J}_{n}-n\mathbf{I}_{n})^{\top}\mathbf{H}_{I}^{\top}\mathbf{H}_{T})`$ |  |
|  |  | $`\displaystyle=\Tr(\mathbf{H}_{T}(\mathbf{J}_{n}-n\mathbf{I}_{n})\mathbf{H}_{I}^{\top})`$ |  |
|  |  | $`\displaystyle=\Tr(\mathbf{W}_{T}^{0}\mathbf{X}_{T}(\mathbf{J}_{n}-n\mathbf{I}_{n})(\mathbf{W}_{I}\mathbf{X}_{I})^{\top})`$ |  |
|  |  | $`\displaystyle=-\Tr\!\left((\mathbf{W}_{T}^{0}\mathbf{X}_{T}(n\mathbf{I}_{n}-\mathbf{J}_{n}))^{\top}\mathbf{W}_{I}\mathbf{X}_{I}\right).`$ |  |

Let
$`\mathbf{Y}_{\text{FT}}=\mathbf{W}_{T}^{0}\mathbf{X}_{T}(n\mathbf{I}_{n}-\mathbf{J}_{n})`$,
as defined in
Definition [3.1](#S3.Thmtheorem1 "Definition 3.1 (Contrastive Target Matrix). ‣ 3.2 Loss Reformulation via the Contrastive Target Matrix ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
Then

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{L}_{\text{CL}}\;=\;-\Tr(\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I}).
``` |  |

Expanding the Frobenius norm of the least-squares residual gives:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}`$ | $`\displaystyle=\frac{1}{2}\Tr\!\big((\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}})^{\top}(\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}})\big)`$ |  |
|  |  | $`\displaystyle=\frac{1}{2}\Tr\!\big(\mathbf{X}_{I}^{\top}\mathbf{W}_{I}^{\top}\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{X}_{I}^{\top}\mathbf{W}_{I}^{\top}\mathbf{Y}_{\text{FT}}`$ |  |
|  |  | $`\displaystyle\hskip 9.24994pt-\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I}+\mathbf{Y}_{\text{FT}}^{\top}\mathbf{Y}_{\text{FT}}\big)`$ |  |
|  |  | $`\displaystyle=\underbrace{\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}}_{(\star)}\underbrace{-\Tr(\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I})}_{=\ \mathcal{L}_{\text{CL}}}+\underbrace{\frac{1}{2}\left\|\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}}_{\text{constant in }\mathbf{W}_{I}}.`$ |  |

##### A clarification on the trace–least-squares relationship.

As the expansion above shows,
$`\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}`$
is *not* equal to
$`-\Tr(\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I})`$ up
to a constant alone; the two differ by the data-dependent quadratic term
$`(\star)=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}`$.
Hence minimizing the pure trace functional
$`-\Tr(\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I})`$ in
isolation is unbounded below and is *not* equivalent to minimizing the
least-squares objective.

What we use throughout the rest of the analysis is the *least-squares*
reformulation

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\min_{\mathbf{W}_{I}}\ \tfrac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}\;=\;\min_{\mathbf{W}_{I}}\ \tfrac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}\;-\;\Tr(\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I})\;+\;\text{const},
``` |  | (5) |

which differs from the original linearized MMCL term by the additive
data-dependent quadratic $`(\star)`$. This extra term can be viewed as
the natural *data-dependent quadratic regularization* that arises
whenever a bilinear similarity is matched against a fixed target
$`\mathbf{Y}_{\text{FT}}`$ under a Frobenius surrogate, and it is
precisely what makes the resulting problem a well-posed matrix
least-squares program with closed-form solutions. All subsequent
closed-form solutions
(Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
and dynamic-teacher analyses
(§[C.5](#A3.SS5 "C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
are derived from this least-squares objective; the original
$`-\Tr(\cdot)`$ form is recovered by dropping $`(\star)`$, but our
results are unaffected because they follow from the least-squares
program.

### C.3 Unified Framework for Contrastive Finetuning: Proofs and Details

Our proofs rely on the following lemma for gradient descent on a matrix
quadratic program.

###### Lemma C.1 (Gradient Descent for Matrix Quadratic Programs).

Let $`\mathcal{Q}:\mathbb{R}^{p\times d}\to\mathbb{R}^{p\times d}`$ be a
positive semi-definite (PSD) linear operator and
$`\mathbf{P}\in\mathbb{R}^{p\times d}`$. Consider the quadratic
objective

|  |  |  |  |
|----|----|----|----|
|  |
``` math
f(\mathbf{W})=\frac{1}{2}\langle\mathbf{W},\mathcal{Q}(\mathbf{W})\rangle_{F}-\langle\mathbf{P},\mathbf{W}\rangle_{F},
``` |  | (6) |

where $`\langle\cdot,\cdot\rangle_{F}`$ denotes the Frobenius inner
product. Let $`\|\mathcal{Q}\|_{\mathrm{op}}`$ denote the operator norm
of $`\mathcal{Q}`$ induced by the Frobenius norm. If
$`\mathbf{P}\in\mathrm{Range}(\mathcal{Q})`$, then gradient descent
initialized at $`\mathbf{W}_{0}`$ with step size
$`\gamma\in(0,2/\|\mathcal{Q}\|_{\mathrm{op}})`$ converges to

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{\infty}=(\mathbf{I}-\Pi_{\mathcal{Q}})(\mathbf{W}_{0})+\mathcal{Q}^{+}(\mathbf{P}),
``` |  | (7) |

where $`\Pi_{\mathcal{Q}}`$ is the orthogonal projector onto
$`\mathrm{Range}(\mathcal{Q})`$ and $`\mathcal{Q}^{+}`$ is the
Moore-Penrose pseudoinverse of $`\mathcal{Q}`$.

###### Proof.

The gradient of $`f`$ is given by
$`\nabla f(\mathbf{W})=\mathcal{Q}(\mathbf{W})-\mathbf{P}`$, yielding
the gradient descent update

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{t+1}=\mathbf{W}_{t}-\gamma(\mathcal{Q}(\mathbf{W}_{t})-\mathbf{P}).
``` |  | (8) |

Since $`\mathcal{Q}`$ is PSD, we have the orthogonal decomposition

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbb{R}^{p\times d}=\mathrm{Range}(\mathcal{Q})\oplus\mathrm{Null}(\mathcal{Q}).
``` |  | (9) |

Let $`\Pi_{\mathcal{Q}}`$ and
$`\Pi_{\mathcal{Q}^{\perp}}=\mathbf{I}-\Pi_{\mathcal{Q}}`$ denote the
orthogonal projectors onto the range and null space of $`\mathcal{Q}`$,
respectively.

##### Analysis of the null space component.

Projecting the gradient descent update onto
$`\mathrm{Null}(\mathcal{Q})`$ yields

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{t+1})`$ | $`\displaystyle=\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{t})-\gamma\Pi_{\mathcal{Q}^{\perp}}(\mathcal{Q}(\mathbf{W}_{t}))+\gamma\Pi_{\mathcal{Q}^{\perp}}(\mathbf{P})`$ |  |
|  |  | $`\displaystyle=\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{t}),`$ |  |

where we used that
$`\mathcal{Q}(\mathbf{W}_{t})\in\mathrm{Range}(\mathcal{Q})`$ implies
$`\Pi_{\mathcal{Q}^{\perp}}(\mathcal{Q}(\mathbf{W}_{t}))=\mathbf{0}`$,
and our assumption $`\mathbf{P}\in\mathrm{Range}(\mathcal{Q})`$ implies
$`\Pi_{\mathcal{Q}^{\perp}}(\mathbf{P})=\mathbf{0}`$. Thus, the null
space component remains invariant throughout the optimization:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{t})=\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{0})\hskip 9.24994pt\forall t\geq 0.
``` |  | (10) |

##### Analysis of the range component.

Let $`\mathbf{W}_{t}^{\prime}=\Pi_{\mathcal{Q}}(\mathbf{W}_{t})`$ denote
the projection onto $`\mathrm{Range}(\mathcal{Q})`$. The dynamics of
this component follow

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{t+1}^{\prime}=(\mathbf{I}-\gamma\mathcal{Q})\mathbf{W}_{t}^{\prime}+\gamma\mathbf{P}.
``` |  | (11) |

The restriction of $`\mathcal{Q}`$ to its range, denoted
$`\mathcal{Q}_{R}:\mathrm{Range}(\mathcal{Q})\to\mathrm{Range}(\mathcal{Q})`$,
is positive definite (since for any non-zero
$`x\in\mathrm{Range}(\mathcal{Q})`$, we must have
$`\mathcal{Q}(x)\neq 0`$, otherwise $`x`$ would be in
$`\mathrm{Null}(\mathcal{Q})`$). For
$`\gamma\in(0,2/\|\mathcal{Q}\|_{\mathrm{op}})`$, the operator
$`\mathbf{I}-\gamma\mathcal{Q}_{R}`$ has spectral radius less than 1,
making it a contraction mapping. By the Banach fixed-point theorem, the
sequence $`\{\mathbf{W}_{t}^{\prime}\}`$ converges to the unique fixed
point $`\mathbf{W}_{\infty}^{\prime}\in\mathrm{Range}(\mathcal{Q})`$
satisfying

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{\infty}^{\prime}=(\mathbf{I}-\gamma\mathcal{Q})\mathbf{W}_{\infty}^{\prime}+\gamma\mathbf{P}.
``` |  | (12) |

Rearranging gives
$`\mathcal{Q}(\mathbf{W}_{\infty}^{\prime})=\mathbf{P}`$, which has the
unique solution
$`\mathbf{W}_{\infty}^{\prime}=\mathcal{Q}^{+}(\mathbf{P})`$ in
$`\mathrm{Range}(\mathcal{Q})`$.

##### Synthesis.

Combining the analyses of both components, we obtain

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbf{W}_{\infty}`$ | $`\displaystyle=\lim_{t\to\infty}\left(\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{t})+\mathbf{W}_{t}^{\prime}\right)`$ |  |
|  |  | $`\displaystyle=\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{0})+\mathcal{Q}^{+}(\mathbf{P})`$ |  |
|  |  | $`\displaystyle=(\mathbf{I}-\Pi_{\mathcal{Q}})(\mathbf{W}_{0})+\mathcal{Q}^{+}(\mathbf{P}),`$ |  |

completing the proof. ∎

###### Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)).

Let
$`\mathcal{P}_{I}\coloneqq\mathbf{X}_{I}(\mathbf{X}_{I}^{\top}\mathbf{X}_{I})^{+}\mathbf{X}_{I}^{\top}`$
denote the orthogonal projection onto the subspace spanned by the
finetuning data $`\mathbf{X}_{I}`$. Consider the general finetuning
objective:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}(\mathbf{W}_{I})=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}+\mathcal{R}(\mathbf{W}_{I})
``` |  | (13) |

where $`\mathcal{R}(\mathbf{W}_{I})`$ represents different
regularization strategies. Gradient descent initialized at
$`\mathbf{W}_{I}^{0}`$ with sufficiently small learning rate converges
to the following solutions:

|  |  |  |
|----|----|----|
| Strategy | $`\mathcal{R}(\mathbf{W}_{I})`$ | Solution |
|    Direct Finetuning | $`0`$ | $`\mathbf{W}_{\text{FT}}=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})+\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$ |
|    $`L_{2}`$ Regularization  (L2-SP ([Li et al., 2018](#bib.bib16))) | $`\frac{\lambda}{2}\left\|\mathbf{W}_{I}-\mathbf{W}_{I}^{0}\right\|_{\text{F}}^{2}`$ | $`\mathbf{W}_{L_{2}}=(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0})(\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{I})^{-1}`$ |
|    Self-Distillation  (SD ([Furlanello et al., 2018](#bib.bib18))) | $`\frac{\lambda}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{W}_{I}^{0}\mathbf{X}_{I}\right\|_{\text{F}}^{2}`$ | $`\mathbf{W}_{SD}=\mathbf{W}_{I}^{0}(\mathbf{I}-\frac{1}{1+\lambda}\mathcal{P}_{I})+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$ |

Here, ⁺ denotes the Moore-Penrose pseudoinverse and $`\lambda>0`$ is the
regularization parameter.

###### Proof.

Let $`\mathbf{C}_{I}=\mathbf{X}_{I}\mathbf{X}_{I}^{\top}`$.

##### Direct Finetuning.

The objective is
$`\mathcal{L}(\mathbf{W}_{I})=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}`$.
We rewrite this in the quadratic form of Lemma
[C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"):

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathcal{L}(\mathbf{W}_{I})`$ | $`\displaystyle=\frac{1}{2}\langle\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}},\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\rangle_{F}`$ |  |
|  |  | $`\displaystyle=\frac{1}{2}\langle\mathbf{W}_{I},\mathbf{W}_{I}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})\rangle_{F}-\langle\mathbf{W}_{I},\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}\rangle_{F}+\frac{1}{2}\left\|\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}`$ |  |
|  |  | $`\displaystyle=\frac{1}{2}\langle\mathbf{W}_{I},\mathbf{W}_{I}\mathbf{C}_{I}\rangle_{F}-\langle\mathbf{W}_{I},\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}\rangle_{F}+\text{const.}`$ |  |

This matches the form
$`f(\mathbf{W})=\frac{1}{2}\langle\mathbf{W},\mathcal{Q}(\mathbf{W})\rangle_{F}-\langle\mathbf{P},\mathbf{W}\rangle_{F}`$
with $`\mathcal{Q}(\mathbf{W})=\mathbf{W}\mathbf{C}_{I}`$ and
$`\mathbf{P}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}`$.

The operator $`\mathcal{Q}`$ is linear and positive semi-definite, as
$`\langle\mathbf{W}_{I},\mathcal{Q}(\mathbf{W}_{I})\rangle_{F}=\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}\geq 0`$.
The condition $`\mathbf{P}\in\text{Range}(\mathcal{Q})`$ holds because
the rows of $`\mathbf{P}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}`$
are linear combinations of the rows of $`\mathbf{X}_{I}^{\top}`$, which
form the row space of $`\mathbf{C}_{I}`$.

By Lemma
[C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
gradient descent converges to
$`\mathbf{W}_{\infty}=\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{I}^{0})+\mathcal{Q}^{+}(\mathbf{P})`$.

1.  1.
    Null Space Component: The null space of $`\mathcal{Q}`$ consists of
    matrices $`\mathbf{A}`$ such that
    $`\mathcal{Q}(\mathbf{A})=\mathbf{A}\mathbf{C}_{I}=\mathbf{0}`$.
    This holds if and only if the rows of $`\mathbf{A}`$ are in the null
    space of $`\mathbf{C}_{I}`$. The orthogonal projector onto this
    component of the initial matrix $`\mathbf{W}_{I}^{0}`$ is
    $`\Pi_{\mathcal{Q}^{\perp}}(\mathbf{W}_{I}^{0})=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})`$,
    where $`\mathcal{P}_{I}=\mathbf{C}_{I}\mathbf{C}_{I}^{+}`$ is the
    projector onto the row space of $`\mathbf{X}_{I}`$. This component
    is preserved.
2.  2.
    Range Component: The pseudoinverse $`\mathcal{Q}^{+}`$ finds the
    minimum Frobenius norm solution to
    $`\mathcal{Q}(\mathbf{W})=\mathbf{P}`$ that lies in
    $`\text{Range}(\mathcal{Q})`$. This is the solution to
    $`\mathbf{W}\mathbf{C}_{I}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}`$,
    which is
    $`\mathcal{Q}^{+}(\mathbf{P})=(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top})\mathbf{C}_{I}^{+}`$.

Combining the components gives the final solution:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\text{FT}}=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})+\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}.
``` |  |

##### $`L_{2}`$ Regularization.

The objective
$`\mathcal{L}(\mathbf{W}_{I})=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}+\frac{\lambda}{2}\left\|\mathbf{W}_{I}-\mathbf{W}_{I}^{0}\right\|_{\text{F}}^{2}`$.
This objective is strongly convex for $`\lambda>0`$. The unique
minimizer is found by setting the gradient to zero:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\nabla_{\mathbf{W}_{I}}\mathcal{L}=(\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}})\mathbf{X}_{I}^{\top}`$ | $`\displaystyle+\lambda(\mathbf{W}_{I}-\mathbf{W}_{I}^{0})=0`$ |  |
|  | $`\displaystyle\rightarrow\mathbf{W}_{I}\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}`$ | $`\displaystyle=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}`$ |  |
|  | $`\displaystyle\rightarrow\mathbf{W}_{I}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{I})`$ | $`\displaystyle=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}`$ |  |

Since $`\mathbf{X}_{I}\mathbf{X}_{I}^{\top}`$ is PSD, the matrix
$`(\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{I})`$ is positive
definite and thus invertible. The solution is:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{L_{2}}=(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0})(\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{I})^{-1}.
``` |  |

A more detailed analysis of the limit behavior of this solution as
$`\lambda\to 0`$ and $`\lambda\to\infty`$ is provided in
§[C.4](#A3.SS4.SSS0.Px3 "Detailed Analysis of the 𝐿_2 Regularization Solution. ‣ C.4 Geometric Interpretation of Solutions ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

##### Self-Distillation.

The objective is
$`\mathcal{L}(\mathbf{W}_{I})=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}+\frac{\lambda}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{W}_{I}^{0}\mathbf{X}_{I}\right\|_{\text{F}}^{2}`$.
Expanding and grouping terms reveals the quadratic structure:

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\mathcal{L}(\mathbf{W}_{I})=\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}-\Tr(\mathbf{Y}_{\text{FT}}^{\top}\mathbf{W}_{I}\mathbf{X}_{I})+\frac{1}{2}\left\|\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}`$ |  |
|  |  | $`\displaystyle\hskip 9.24994pt+\frac{\lambda}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}-\lambda\Tr((\mathbf{W}_{I}^{0}\mathbf{X}_{I})^{\top}\mathbf{W}_{I}\mathbf{X}_{I})+\frac{\lambda}{2}\left\|\mathbf{W}_{I}^{0}\mathbf{X}_{I}\right\|_{\text{F}}^{2}`$ |  |
|  |  | $`\displaystyle=\frac{1+\lambda}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}\right\|_{\text{F}}^{2}-\Tr((\mathbf{Y}_{\text{FT}}^{\top}+\lambda(\mathbf{W}_{I}^{0}\mathbf{X}_{I})^{\top})\mathbf{W}_{I}\mathbf{X}_{I})+\text{const.}`$ |  |
|  |  | $`\displaystyle=\frac{1+\lambda}{2}\langle\mathbf{W}_{I},\mathbf{W}_{I}\mathbf{C}_{I}\rangle_{F}-\langle\mathbf{W}_{I},\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\mathbf{C}_{I}\rangle_{F}+\text{const.}`$ |  |

This matches the form of Lemma
[C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
with
$`\mathcal{Q}_{SD}(\mathbf{W}_{I})=(1+\lambda)\mathbf{W}_{I}\mathbf{C}_{I}`$
and
$`\mathbf{P}_{SD}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\mathbf{C}_{I}`$.

The operator $`\mathcal{Q}_{SD}`$ is PSD. Its range and null space are
identical to those of $`\mathcal{Q}`$ from the Direct Finetuning case.
The terms $`\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}`$ and
$`\lambda\mathbf{W}_{I}^{0}\mathbf{C}_{I}`$ are both in
$`\text{Range}(\mathcal{Q}_{SD})`$ (as shown before for
$`\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}`$, and
$`\mathbf{W}_{I}^{0}\mathbf{C}_{I}`$ by definition). Thus, their sum
$`\mathbf{P}_{SD}`$ is also in the range.

We apply Lemma
[C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
to find the limit
$`\mathbf{W}_{\infty}=\Pi_{\mathcal{Q}_{SD}^{\perp}}(\mathbf{W}_{I}^{0})+\mathcal{Q}_{SD}^{+}(\mathbf{P}_{SD})`$.

1.  1.
    Null Space Component:
    $`\text{Null}(\mathcal{Q}_{SD})=\text{Null}(\mathcal{Q})`$, so the
    invariant component is again
    $`\Pi_{\mathcal{Q}_{SD}^{\perp}}(\mathbf{W}_{I}^{0})=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})`$.
2.  2.
    Range Component: The pseudoinverse is
    $`\mathcal{Q}_{SD}^{+}=\frac{1}{1+\lambda}\mathcal{Q}^{+}`$, where
    $`\mathcal{Q}^{+}`$ corresponds to the direct finetuning case.
    Applying it to $`\mathbf{P}_{SD}`$:

    |  |  |  |  |
    |----|----|----|----|
    |  | $`\displaystyle\mathcal{Q}_{SD}^{+}(\mathbf{P}_{SD})`$ | $`\displaystyle=\frac{1}{1+\lambda}\mathcal{Q}^{+}\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\mathbf{C}_{I}\right)`$ |  |
    |  |  | $`\displaystyle=\frac{1}{1+\lambda}\left((\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top})\mathbf{C}_{I}^{+}+\lambda\mathcal{Q}^{+}(\mathcal{Q}(\mathbf{W}_{I}^{0}))\right)`$ |  |
    |  |  | $`\displaystyle=\frac{1}{1+\lambda}\left((\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top})\mathbf{C}_{I}^{+}+\lambda\Pi_{\mathcal{Q}}(\mathbf{W}_{I}^{0})\right)`$ |  |
    |  |  | $`\displaystyle=\frac{1}{1+\lambda}\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}+\lambda\mathbf{W}_{I}^{0}\mathcal{P}_{I}\right).`$ |  |

Combining the components for the final solution $`\mathbf{W}_{SD}`$:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbf{W}_{SD}`$ | $`\displaystyle=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})+\frac{\lambda}{1+\lambda}\mathbf{W}_{I}^{0}\mathcal{P}_{I}+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$ |  |
|  |  | $`\displaystyle=\mathbf{W}_{I}^{0}\left(\mathbf{I}-\mathcal{P}_{I}+\frac{\lambda}{1+\lambda}\mathcal{P}_{I}\right)+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$ |  |
|  |  | $`\displaystyle=\mathbf{W}_{I}^{0}\left(\mathbf{I}-\frac{1}{1+\lambda}\mathcal{P}_{I}\right)+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}.`$ |  |

This completes the proof. ∎

### C.4 Geometric Interpretation of Solutions

The closed-form solutions presented in
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
provide a geometric understanding of how different finetuning strategies
modify pretrained representations. We decompose the solution for
$`\mathbf{W}_{I}`$ into components acting on the subspace spanned by the
finetuning data $`\mathbf{X}_{I}`$ (parallel component) and its
orthogonal complement (orthogonal component).

##### Direct Finetuning.

The solution $`\mathbf{W}_{\text{FT}}`$ is a sum of two orthogonal
parts: (1) $`\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})`$: This is
the projection of the pretrained weights onto the orthogonal complement
of the finetuning data subspace
($`\mathrm{Null}(\mathbf{X}_{I}^{\top})`$). This component preserves the
action of $`\mathbf{W}_{I}^{0}`$ on data vectors orthogonal to the
finetuning examples. (2)
$`\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$:
This is the minimum-norm solution that fits the new contrastive task
within the finetuning data subspace. This component lies entirely within
the range of $`\mathbf{X}_{I}^{\top}`$.

Interpretation: Direct finetuning completely replaces (forgets) any
pretrained knowledge related to features present in the finetuning data,
substituting it with the new task-specific solution. It only preserves
knowledge in directions entirely unrelated to the finetuning examples.

##### $`L_{2}`$ Regularization.

The solution $`\mathbf{W}_{L_{2}}`$ is the standard matrix ridge
regression solution. It creates a complex blend of the new task solution
and the initial weights. There is no clean separation of orthogonal and
parallel components as in direct finetuning or self-distillation. The
key insight is that $`L_{2}`$ regularization modifies the data
covariance matrix $`\mathbf{X}_{I}\mathbf{X}_{I}^{\top}`$ by adding
$`\lambda\mathbf{I}`$, which acts as a *ridge* that prevents overfitting
by shrinking the solution along all eigendirections of the data. Unlike
direct finetuning and self-distillation, which primarily modify weights
in the subspace spanned by $`\mathbf{X}_{I}`$, $`L_{2}`$ regularization
affects all directions in the weight space, blending the old and new
across the entire parameter space.

##### Detailed Analysis of the $`L_{2}`$ Regularization Solution.

The solution for $`L_{2}`$ regularization is given by:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{L_{2}}=\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\right)\left(\mathbf{X}_{I}\mathbf{X}_{I}^{\top}+\lambda\mathbf{I}\right)^{-1}.
``` |  |

To analyze its behavior, we consider the eigendecomposition of the data
covariance matrix
$`\mathbf{C}_{I}\coloneqq\mathbf{X}_{I}\mathbf{X}_{I}^{\top}`$. Since
$`\mathbf{C}_{I}`$ is a real, symmetric, positive semi-definite (PSD)
matrix, it has an eigendecomposition
$`\mathbf{C}_{I}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^{\top}`$, where
$`\mathbf{U}`$ is an orthogonal matrix of eigenvectors and
$`\mathbf{\Lambda}`$ is a diagonal matrix of non-negative eigenvalues.
Using this decomposition, the inverse term in the solution becomes:

|  |  |  |
|----|----|----|
|  |
``` math
(\mathbf{C}_{I}+\lambda\mathbf{I})^{-1}=(\mathbf{U}\mathbf{\Lambda}\mathbf{U}^{\top}+\lambda\mathbf{U}\mathbf{U}^{\top})^{-1}=(\mathbf{U}(\mathbf{\Lambda}+\lambda\mathbf{I})\mathbf{U}^{\top})^{-1}=\mathbf{U}(\mathbf{\Lambda}+\lambda\mathbf{I})^{-1}\mathbf{U}^{\top}.
``` |  |

The matrix $`(\mathbf{\Lambda}+\lambda\mathbf{I})`$ is diagonal with
entries $`\lambda_{k}+\lambda`$, so its inverse has entries
$`1/(\lambda_{k}+\lambda)`$.

##### Analysis of the Limit as $`\lambda\to 0`$.

Let $`r=\text{rank}(\mathbf{C}_{I})`$. We partition the eigenvectors
$`\mathbf{U}`$ and eigenvalues $`\mathbf{\Lambda}`$ into components
corresponding to non-zero and zero eigenvalues. Let
$`\mathbf{U}_{r}\in\mathbb{R}^{d_{I}\times r}`$ contain eigenvectors for
$`r`$ positive eigenvalues ($`\mathbf{\Lambda}_{r}`$), and
$`\mathbf{U}_{0}\in\mathbb{R}^{d_{I}\times(d_{I}-r)}`$ for zero
eigenvalues. The projectors onto the range and null space of
$`\mathbf{C}_{I}`$ are
$`\mathcal{P}_{\text{range}}=\mathbf{U}_{r}\mathbf{U}_{r}^{\top}`$ and
$`\mathcal{P}_{\text{null}}=\mathbf{U}_{0}\mathbf{U}_{0}^{\top}`$,
respectively. Note that $`\mathcal{P}_{\text{range}}=\mathcal{P}_{I}`$
and $`\mathcal{P}_{\text{null}}=\mathbf{I}-\mathcal{P}_{I}`$.

The inverse term can be split:

|  |  |  |
|----|----|----|
|  |
``` math
(\mathbf{C}_{I}+\lambda\mathbf{I})^{-1}=\mathbf{U}_{r}(\mathbf{\Lambda}_{r}+\lambda\mathbf{I}_{r})^{-1}\mathbf{U}_{r}^{\top}+\frac{1}{\lambda}\mathbf{U}_{0}\mathbf{U}_{0}^{\top}.
``` |  |

Substituting this back into $`\mathbf{W}_{L_{2}}`$:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbf{W}_{L_{2}}`$ | $`\displaystyle=\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\right)\left[\mathbf{U}_{r}(\mathbf{\Lambda}_{r}+\lambda\mathbf{I}_{r})^{-1}\mathbf{U}_{r}^{\top}+\frac{1}{\lambda}\mathbf{U}_{0}\mathbf{U}_{0}^{\top}\right]`$ |  |
|  |  | $`\displaystyle=\underbrace{\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\right)\mathbf{U}_{r}(\mathbf{\Lambda}_{r}+\lambda\mathbf{I}_{r})^{-1}\mathbf{U}_{r}^{\top}}_{\text{Term 1}}`$ |  |
|  |  | $`\displaystyle+\underbrace{\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\right)\frac{1}{\lambda}\mathbf{U}_{0}\mathbf{U}_{0}^{\top}}_{\text{Term 2}}.`$ |  |

For Term 2, since $`\mathbf{X}_{I}^{\top}\mathbf{U}_{0}=\mathbf{0}`$
(columns of $`\mathbf{U}_{0}`$ are in the null space of
$`\mathbf{C}_{I}`$), it simplifies to:

|  |  |  |
|----|----|----|
|  |
``` math
\text{Term 2}=\frac{1}{\lambda}\mathbf{Y}_{\text{FT}}\underbrace{\mathbf{X}_{I}^{\top}\mathbf{U}_{0}}_{\mathbf{0}}\mathbf{U}_{0}^{\top}+\mathbf{W}_{I}^{0}\mathbf{U}_{0}\mathbf{U}_{0}^{\top}=\mathbf{W}_{I}^{0}\mathcal{P}_{\text{null}}=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I}).
``` |  |

As $`\lambda\to 0`$, Term 1 converges to:

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{\lambda\to 0}\text{Term 1}=\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}\right)\mathbf{U}_{r}\mathbf{\Lambda}_{r}^{-1}\mathbf{U}_{r}^{\top}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}\mathbf{C}_{I}^{+},
``` |  |

where
$`\mathbf{C}_{I}^{+}=\mathbf{U}_{r}\mathbf{\Lambda}_{r}^{-1}\mathbf{U}_{r}^{\top}`$
is the Moore-Penrose pseudoinverse of $`\mathbf{C}_{I}`$. Combining the
limits of both terms, we get:

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{\lambda\to 0}\mathbf{W}_{L_{2}}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}+\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I}).
``` |  |

This is precisely the direct finetuning solution,
$`\mathbf{W}_{\text{FT}}`$.

##### Analysis of the Limit as $`\lambda\to\infty`$.

For the limit as $`\lambda\to\infty`$, we factor out $`\lambda`$:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbf{W}_{L_{2}}`$ | $`\displaystyle=\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\mathbf{W}_{I}^{0}\right)\frac{1}{\lambda}\left(\frac{1}{\lambda}\mathbf{C}_{I}+\mathbf{I}\right)^{-1}`$ |  |
|  |  | $`\displaystyle=\left(\frac{1}{\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\mathbf{W}_{I}^{0}\right)\left(\frac{1}{\lambda}\mathbf{C}_{I}+\mathbf{I}\right)^{-1}.`$ |  |

As $`\lambda\to\infty`$, the term $`\frac{1}{\lambda}\to 0`$. Therefore,
the expression converges to:

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{\lambda\to\infty}\mathbf{W}_{L_{2}}=\left(\mathbf{0}+\mathbf{W}_{I}^{0}\right)\left(\mathbf{0}+\mathbf{I}\right)^{-1}=\mathbf{W}_{I}^{0}.
``` |  |

Thus, the regularization parameter $`\lambda`$ smoothly interpolates the
solution between two meaningful extremes: pure task adaptation and pure
preservation of pretrained weights.

##### Self-Distillation.

The solution $`\mathbf{W}_{SD}`$ provides the most sophisticated and
effective compromise. We can rewrite it to reveal its structure:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbf{W}_{SD}`$ | $`\displaystyle=\mathbf{W}_{I}^{0}-\frac{1}{1+\lambda}\mathbf{W}_{I}^{0}\mathcal{P}_{I}+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$ |  |
|  |  | $`\displaystyle=\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})+\mathbf{W}_{I}^{0}\mathcal{P}_{I}-\frac{1}{1+\lambda}\mathbf{W}_{I}^{0}\mathcal{P}_{I}+\frac{1}{1+\lambda}\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}\right)`$ |  |
|  |  | $`\displaystyle=\underbrace{\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I})}_{\begin{subarray}{c}\text{Component orthogonal to finetuning data}\\
\textbf{(Preserved)}\end{subarray}}+\underbrace{\frac{\lambda}{1+\lambda}\left(\mathbf{W}_{I}^{0}\mathcal{P}_{I}\right)+\frac{1}{1+\lambda}\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}\right)}_{\begin{subarray}{c}\text{Component within finetuning data subspace}\\
\textbf{(Convex Combination)}\end{subarray}}`$ |  |

Interpretation: Self-Distillation operates with surgical precision:  1.
Outside the finetuning subspace, it acts as an identity function,
preserving the components of the pretrained model that are irrelevant to
the new task. 2. Inside the finetuning subspace, it does not discard the
pretrained knowledge. Instead, it computes a convex combination of the
projected pretrained weights and the optimal solution for the new
contrastive task. The hyperparameter $`\lambda`$ smoothly controls this
trade-off. This demonstrates that Self-Distillation achieves a “best of
both worlds” scenario: preserving general capabilities while adapting to
new information where necessary.

### C.5 Dynamic Self-Distillation: WMA Details and Convergence

We extend the analysis of static self-distillation to a dynamic teacher,
specifically a Weighted Moving Average (WMA) teacher, which adapts its
regularization throughout training. This section provides the detailed
definitions, dynamics, and convergence proofs.

###### Definition C.3 (SD–WMA Objective (Repeated from Main Text)).

At step $`t`$, the student weights $`\mathbf{W}_{I}^{t}`$ solve

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{SD-WMA}}(\mathbf{W}_{I})\;=\;\frac{1}{2}\,\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}\;+\;\frac{\lambda}{2}\,\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{W}_{\text{Teacher}}^{t-1}\mathbf{X}_{I}\right\|_{\text{F}}^{2},\hskip 18.49988pt\text{initialized from }\mathbf{W}_{I}^{t-1}.
``` |  | (14) |

###### Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)).

Let the normalized time grid be

|  |  |  |
|----|----|----|
|  |
``` math
\tau_{k}\;=\;\frac{k+c_{1}}{T+c_{2}}\in(0,1),\hskip 18.49988ptc_{1},c_{2}>0.
``` |  |

Choose any nonnegative *weighting kernel*
$`\kappa:[0,1]\to\mathbb{R}_{\geq 0}`$ and define unnormalized weights
$`\alpha_{k}\;=\;\kappa(\tau_{k})`$ The *online* normalization and
teacher recursion are

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\omega_{t}\;=\;\frac{\alpha_{t}}{\sum_{j=0}^{t}\alpha_{j}},\hskip 18.49988pt\mathbf{W}_{\text{Teacher}}^{t}\;=\;(1-\omega_{t})\,\mathbf{W}_{\text{Teacher}}^{t-1}\;+\;\omega_{t}\,\mathbf{W}_{I}^{t},\hskip 18.49988pt\mathbf{W}_{\text{Teacher}}^{0}\;=\;\mathbf{W}_{I}^{0}.
``` |  | (15) |

###### Remark C.5 (Teacher as a normalized history average).

Unrolling equation [15](#A3.E15 "Equation 15 ‣ Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)). ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
yields a normalized convex average of the student’s history:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\text{Teacher}}^{t}\;=\;\sum_{k=0}^{t}\underbrace{\frac{\alpha_{k}}{\sum_{j=0}^{t}\alpha_{j}}}_{\omega_{k\mid t}}\,\mathbf{W}_{I}^{k},\hskip 18.49988pt\omega_{k\mid t}\geq 0,\hskip 9.24994pt\sum_{k=0}^{t}\omega_{k\mid t}=1.
``` |  |

Thus the teacher is an *expectation* with respect to the discrete
distribution
$`\mathrm{Categorical}(\omega_{0\mid t},\ldots,\omega_{t\mid t})`$:
$`\ \mathbf{W}_{\text{Teacher}}^{t}=\mathbb{E}_{K\sim\omega_{\cdot\mid t}}[\mathbf{W}_{I}^{K}]`$.

#### C.5.1 WMA vs. EMA Teachers

This section contrasts the proposed *Weighted Moving Average* (WMA)
teacher with the standard *Exponential Moving Average* (EMA), which
underlies mean-teacher approaches.

##### EMA (mean-teacher).

EMA maintains an exponentially decaying average:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{\text{EMA}}^{t}\;=\;\rho\,\mathbf{W}_{\text{EMA}}^{t-1}+(1-\rho)\,\mathbf{W}_{I}^{t},\hskip 18.49988pt\rho\in(0,1),\ \ \mathbf{W}_{\text{EMA}}^{0}=\mathbf{W}_{I}^{0}.
``` |  | (16) |

Unrolling this recursion gives a geometric kernel over *lag*:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\text{EMA}}^{t}\;=\;\rho^{t}\mathbf{W}_{I}^{0}+(1-\rho)\sum_{k=1}^{t}\rho^{\,t-k}\,\mathbf{W}_{I}^{k}\;=\;\sum_{k=0}^{t}\underbrace{\omega^{\text{EMA}}_{k\mid t}}_{\text{depends on }t-k}\,\mathbf{W}_{I}^{k},
``` |  |

with $`\omega^{\text{EMA}}_{0\mid t}=\rho^{t}`$,
$`\omega^{\text{EMA}}_{k\mid t}=(1-\rho)\rho^{\,t-k}`$ for $`k\geq 1`$,
and $`\sum_{k=0}^{t}\omega^{\text{EMA}}_{k\mid t}=1`$. The kernel is
*stationary in lag*: weights depend only on recency $`t-k`$.

##### WMA (normalized-time kernel).

In contrast, WMA assigns weights via a *kernel over normalized time*
$`\tau_{k}=(k+c_{1})/(T+c_{2})`$:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\text{WMA}}^{t}\;=\;\sum_{k=0}^{t}\underbrace{\omega^{\text{WMA}}_{k\mid t}}_{\propto\ \kappa(\tau_{k})}\,\mathbf{W}_{I}^{k},\hskip 18.49988pt\omega^{\text{WMA}}_{k\mid t}=\frac{\alpha_{k}}{\sum_{j=0}^{t}\alpha_{j}},\hskip 9.24994pt\alpha_{k}=\kappa(\tau_{k}).
``` |  |

Here the kernel is *position-aware* in absolute (normalized) time, not
just lag. The symmetric Beta kernel ($`\beta_{1}=\beta_{2}`$) permits
simultaneous emphasis of *both* endpoints (early stability and late
convergence), a pattern that is *not* attainable with any
single-parameter EMA.

##### Key differences.

- •
  Shape control. EMA imposes a monotone geometric decay from the
  present; WMA can be early-peaked, late-peaked, flat (uniform), bimodal
  (e.g., arcsine), etc.
- •
  Invariance to schedule granularity. WMA weights are defined on
  normalized time: if the training is retimed or step granularity
  changes while preserving the path over $`[0,1]`$, the kernel
  $`\kappa`$ need not be retuned. EMA depends on the absolute decay
  $`\rho`$ and typically requires retuning when $`T`$ or logging cadence
  changes.
- •
  Endpoint behavior. With $`\beta_{1}=\beta_{2}=\tfrac{1}{2}`$
  (arcsine), WMA places substantial weight near $`k\approx 0`$ and
  $`k\approx t`$, preserving early information *and* emphasizing late
  iterates; EMA cannot simultaneously upweight both ends.
- •
  Recovering classical averages. Choosing $`\kappa`$ uniform
  (Beta$`(1,1)`$) yields the simple running average (Polyak/Ruppert;
  SWA ([Izmailov et al., 2018](#bib.bib21))). EMA cannot realize an
  exactly uniform window without time-varying $`\rho_{t}`$.
- •
  Online normalization. Both EMA and WMA are online and convex at each
  step; WMA’s $`\omega_{t}=\alpha_{t}/\sum_{j\leq t}\alpha_{j}`$ admits
  arbitrary nonnegative $`\alpha_{t}`$ induced by $`\kappa`$.

##### Mean-teacher within the WMA recursion.

In SD–WMA
(Definition [C.4](#A3.Thmapptheorem4 "Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)). ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")),
the step weight is $`\omega_{t}=\alpha_{t}/\sum_{j=0}^{t}\alpha_{j}`$,
which is generally time-varying. To *recover EMA exactly* with constant
$`\omega\equiv 1-\rho`$, choose any $`\alpha_{0}>0`$ and set, for
$`t\geq 1`$,

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\alpha_{t}\;=\;\frac{\omega}{(1-\omega)^{t}}\,\alpha_{0}\hskip 18.49988pt\Longleftrightarrow\hskip 18.49988pt\alpha_{t}\;=\;\frac{1-\rho}{\rho^{\,t}}\,\alpha_{0},
``` |  | (17) |

which yields $`\omega_{t}\equiv\omega`$ and makes the WMA recursion
identical
to equation [16](#A3.E16 "Equation 16 ‣ EMA (mean-teacher). ‣ C.5.1 WMA vs. EMA Teachers ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
If one insists on $`\alpha_{t}=\kappa(\tau_{t})`$ with
$`\tau_{t}=(t+c_{1})/(T+c_{2})`$, EMA corresponds to an exponential
kernel over normalized time,
$`\ \kappa(\tau)=C\,(1-\omega)^{-(T+c_{2})\tau+c_{1}^{\prime}}`$, for
suitable constants $`C,c_{1}^{\prime}`$ (fixed per run), which
reproduces $`\omega_{t}\equiv\omega`$
via equation [17](#A3.E17 "Equation 17 ‣ Mean-teacher within the WMA recursion. ‣ C.5.1 WMA vs. EMA Teachers ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").

#### C.5.2 The Persistent Regularizer of the WMA Teacher

A key advantage of the WMA teacher over the more common EMA teacher lies
in the dynamics of the regularization it provides. The self-distillation
loss, $`\mathcal{L}_{\text{SD}}`$, induces a regularizing gradient
field,
$`\mathbf{g}_{R}(\mathbf{W}_{I}^{t})\coloneqq\nabla_{\mathbf{W}_{I}}\mathcal{L}_{\text{SD}}(\mathbf{W}_{T}^{t},\mathbf{W}_{I}^{t})`$,
that pulls the student towards the teacher. The persistence of this
field is critical for preventing the student from over-specializing on
the finetuning task.

##### The Vanishing Regularizer of EMA.

An EMA teacher is a low-pass filter of the student’s trajectory:
$`\mathbf{W}_{\text{EMA}}^{t}=\rho\,\mathbf{W}_{\text{EMA}}^{t-1}+(1-\rho)\,\mathbf{W}_{I}^{t}`$.
As the student’s updates converge
($`\|\mathbf{W}_{I}^{t+1}-\mathbf{W}_{I}^{t}\|_{F}\to 0`$), the teacher
necessarily converges to the student’s final parameters
($`\lim_{t\to\infty}\|\mathbf{W}_{\text{EMA}}^{t}-\mathbf{W}_{I}^{t}\|_{F}=0`$).
Consequently, any regularizer based on the teacher–student gap vanishes:

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{t\to\infty}\|\mathbf{g}_{R}(\mathbf{W}_{I}^{t};\mathbf{W}_{\text{EMA}}^{t})\|_{F}=0,
``` |  |

allowing the optimization to be dominated entirely by the task loss near
the end of training.

##### The Finite-Horizon Persistence of WMA.

The WMA teacher is a weighted average of the *entire* student history:
$`\mathbf{W}_{\text{WMA}}^{t}=\sum_{k=0}^{t}\omega_{k\mid t}\mathbf{W}_{I}^{k}`$.
For any *fixed finite* run of length $`T`$, any kernel with
$`\alpha_{0}>0`$ yields $`\omega_{0\mid T}>0`$, so
$`\mathbf{W}_{I}^{0}`$ contributes nontrivially to
$`\mathbf{W}_{\text{WMA}}^{T}`$. Thus, when the student has moved away
from initialization ($`\mathbf{W}_{I}^{T}\neq\mathbf{W}_{I}^{0}`$), the
teacher can remain separated from the final iterate, yielding a
nontrivial regularizing gradient at step $`T`$. Note that under online
normalization, the relative weight on any fixed $`k`$ typically
satisfies $`\omega_{k\mid t}\to 0`$ as $`t\to\infty`$; therefore any
“non-vanishing” claim must be understood as a finite-horizon /
late-training statement. In particular, under infinite-horizon training
with online normalization and a convergent student
($`\mathbf{W}_{I}^{t}\to\mathbf{W}_{I}^{\infty}`$), one typically has
$`\mathbf{W}_{\text{WMA}}^{t}\to\mathbf{W}_{I}^{\infty}`$, so the
teacher–student gap can vanish asymptotically.

###### Theorem C.6 (Finite-Horizon Persistence of the WMA Regularizer).

Fix a training horizon $`T`$ and let the WMA teacher at the end of
training be

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\mathrm{WMA}}^{T}=\sum_{k=0}^{T}\omega_{k\mid T}\,\mathbf{W}_{I}^{k},\hskip 18.49988pt\omega_{k\mid T}\geq 0,\ \ \sum_{k=0}^{T}\omega_{k\mid T}=1,
``` |  |

with $`\omega_{0\mid T}>0`$ (true for any kernel with $`\alpha_{0}>0`$).
Assume the student trajectory is *monotone along a single direction* in
parameter space: there exist a unit matrix $`\mathbf{U}`$ with
$`\|\mathbf{U}\|_{F}=1`$ and scalars
$`0=a_{0}\leq a_{1}\leq\cdots\leq a_{T}`$ such that

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I}^{k}=\mathbf{W}_{I}^{0}+a_{k}\,\mathbf{U}\hskip 18.49988pt\forall k\in\{0,\ldots,T\}.
``` |  |

Then, if $`\mathbf{W}_{I}^{T}\neq\mathbf{W}_{I}^{0}`$, the teacher
remains *strictly behind* the final iterate and

|  |  |  |
|----|----|----|
|  |
``` math
\|\mathbf{W}_{I}^{T}-\mathbf{W}_{\mathrm{WMA}}^{T}\|_{F}\ \geq\ \omega_{0\mid T}\,\|\mathbf{W}_{I}^{T}-\mathbf{W}_{I}^{0}\|_{F}\ >\ 0.
``` |  |

Moreover, suppose the self-distillation loss is *locally approximately
quadratic* in the teacher–student parameter difference (e.g.,
second-order KL expansion) so that for $`W`$ near
$`\mathbf{W}_{I}^{T}`$,

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{g}_{R}(W;\mathbf{W}_{\mathrm{WMA}}^{T})=\nabla_{W}\mathcal{L}_{\mathrm{SD}}(\mathbf{W}_{\mathrm{WMA}}^{T},W)\approx\mathbf{F}_{T}\,(W-\mathbf{W}_{\mathrm{WMA}}^{T}),
``` |  |

and define the terminal linearization residual

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{r}_{T}\;\coloneqq\;\mathbf{g}_{R}(\mathbf{W}_{I}^{T};\mathbf{W}_{\mathrm{WMA}}^{T})-\mathbf{F}_{T}(\mathbf{W}_{I}^{T}-\mathbf{W}_{\mathrm{WMA}}^{T}).
``` |  |

If the curvature satisfies $`\mathbf{F}_{T}\succeq\mu\,\mathbf{I}`$ on
$`\mathrm{span}\{\mathbf{U}\}`$ for some $`\mu>0`$, then the
regularizing gradient at the end of training admits the
*signal-minus-error* lower bound

|  |  |  |
|----|----|----|
|  |
``` math
\|\mathbf{g}_{R}(\mathbf{W}_{I}^{T};\mathbf{W}_{\mathrm{WMA}}^{T})\|_{F}\ \geq\ \mu\,\omega_{0\mid T}\,\|\mathbf{W}_{I}^{T}-\mathbf{W}_{I}^{0}\|_{F}\;-\;\|\mathbf{r}_{T}\|_{F}.
``` |  |

In particular, if
$`\|\mathbf{r}_{T}\|_{F}<\mu\,\omega_{0\mid T}\,\|\mathbf{W}_{I}^{T}-\mathbf{W}_{I}^{0}\|_{F}`$,
then
$`\mathbf{g}_{R}(\mathbf{W}_{I}^{T};\mathbf{W}_{\mathrm{WMA}}^{T})\neq\mathbf{0}`$.

In contrast, for an EMA teacher with fixed $`\rho\in(0,1)`$, if
$`\mathbf{W}_{I}^{t}\to\mathbf{W}_{I}^{\infty}`$ then
$`\|\mathbf{W}_{\mathrm{EMA}}^{t}-\mathbf{W}_{I}^{t}\|_{F}\to 0`$, and
thus the corresponding regularizing gradient vanishes asymptotically.

###### Proof Sketch.

Step 1: Teacher–student gap under monotone 1D motion. Under the assumed
form $`\mathbf{W}_{I}^{k}=\mathbf{W}_{I}^{0}+a_{k}\mathbf{U}`$,

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\mathrm{WMA}}^{T}=\sum_{k=0}^{T}\omega_{k\mid T}(\mathbf{W}_{I}^{0}+a_{k}\mathbf{U})=\mathbf{W}_{I}^{0}+\Big(\sum_{k=0}^{T}\omega_{k\mid T}a_{k}\Big)\mathbf{U}.
``` |  |

Hence

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I}^{T}-\mathbf{W}_{\mathrm{WMA}}^{T}=\Big(a_{T}-\sum_{k=0}^{T}\omega_{k\mid T}a_{k}\Big)\mathbf{U}=\sum_{k=0}^{T}\omega_{k\mid T}(a_{T}-a_{k})\mathbf{U}.
``` |  |

Since $`a_{T}\geq a_{k}`$ and $`\omega_{k\mid T}\geq 0`$, all terms are
nonnegative multiples of the same direction, so there is no cancellation
and

|  |  |  |
|----|----|----|
|  |
``` math
\|\mathbf{W}_{I}^{T}-\mathbf{W}_{\mathrm{WMA}}^{T}\|_{F}=\sum_{k=0}^{T}\omega_{k\mid T}(a_{T}-a_{k})\ \geq\ \omega_{0\mid T}(a_{T}-a_{0})=\omega_{0\mid T}\|\mathbf{W}_{I}^{T}-\mathbf{W}_{I}^{0}\|_{F}.
``` |  |

Step 2: Gradient lower bound. Let
$`\Delta_{T}=\mathbf{W}_{I}^{T}-\mathbf{W}_{\mathrm{WMA}}^{T}\in\mathrm{span}\{\mathbf{U}\}`$.
By definition,
$`\mathbf{g}_{R}(\mathbf{W}_{I}^{T};\mathbf{W}_{\mathrm{WMA}}^{T})=\mathbf{F}_{T}\Delta_{T}+\mathbf{r}_{T}`$,
so by the triangle inequality,

|  |  |  |
|----|----|----|
|  |
``` math
\|\mathbf{g}_{R}(\mathbf{W}_{I}^{T};\mathbf{W}_{\mathrm{WMA}}^{T})\|_{F}\ \geq\ \|\mathbf{F}_{T}\Delta_{T}\|_{F}-\|\mathbf{r}_{T}\|_{F}.
``` |  |

Using $`\mathbf{F}_{T}\succeq\mu I`$ on $`\mathrm{span}\{\mathbf{U}\}`$
and Cauchy–Schwarz,
$`\|\mathbf{F}_{T}\Delta_{T}\|_{F}\geq\mu\|\Delta_{T}\|_{F}`$, hence

|  |  |  |
|----|----|----|
|  |
``` math
\|\mathbf{g}_{R}(\mathbf{W}_{I}^{T};\mathbf{W}_{\mathrm{WMA}}^{T})\|_{F}\ \geq\ \mu\|\Delta_{T}\|_{F}-\|\mathbf{r}_{T}\|_{F}\ \geq\ \mu\,\omega_{0\mid T}\,\|\mathbf{W}_{I}^{T}-\mathbf{W}_{I}^{0}\|_{F}-\|\mathbf{r}_{T}\|_{F}.
``` |  |

Step 3: EMA vanishing. If
$`\mathbf{W}_{I}^{t}\to\mathbf{W}_{I}^{\infty}`$, then the EMA recursion
is a stable linear filter of a convergent signal, implying
$`\mathbf{W}_{\mathrm{EMA}}^{t}\to\mathbf{W}_{I}^{\infty}`$ and thus
$`\|\mathbf{W}_{\mathrm{EMA}}^{t}-\mathbf{W}_{I}^{t}\|_{F}\to 0`$. ∎

Conclusion. This theorem characterizes a *finite-horizon* effect: even
when the student’s updates become small near the end of training, a WMA
teacher can remain separated from the terminal iterate at step $`T`$
because it retains positive mass on earlier states, yielding a
regularizing gradient whose magnitude is lower bounded by a *signal
minus approximation error* term. Over an *infinite* horizon with online
normalization (where $`\omega_{0\mid t}\to 0`$) and a convergent
student, the teacher–student gap can vanish, consistent with bias-free
convergence results proved later for SD–WMA in the task subspace.

#### C.5.3 Convergence Analysis

We first state the single-step solution and then derive global
convergence in the task subspace.

##### From static to dynamic SD as a sequence of quadratic problems.

Before stating the single-step solution, it is useful to make explicit
what the SD–WMA scheme is doing as an optimization process. At each step
$`t`$, the SD–WMA objective

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{L}_{\text{SD-WMA}}(\mathbf{W}_{I})\;=\;\tfrac{1}{2}\,\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}\;+\;\tfrac{\lambda}{2}\,\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{W}_{\text{Teacher}}^{t-1}\mathbf{X}_{I}\right\|_{\text{F}}^{2}
``` |  |

is *still a static quadratic problem in $`\mathbf{W}_{I}`$*: it has the
exact same algebraic form as the static self-distillation objective
analyzed in
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
with two simple substitutions relative to the static-SD case:

- •
  the anchor in the distillation term is replaced from the fixed
  pretrained weights $`\mathbf{W}_{I}^{0}`$ to the current WMA teacher
  $`\mathbf{W}_{\text{Teacher}}^{t-1}`$;
- •
  the initialization of gradient descent is replaced from
  $`\mathbf{W}_{I}^{0}`$ to the previous student iterate
  $`\mathbf{W}_{I}^{t-1}`$.

These two substitutions touch *different* parts of the closed-form
solution given by
Lemma [C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"):

- •
  changing the *anchor*
  $`\mathbf{W}_{I}^{0}\to\mathbf{W}_{\text{Teacher}}^{t-1}`$ modifies
  the range component of the solution (the part lying in
  $`\mathrm{range}(\mathbf{X}_{I})`$, i.e., the task subspace), since
  this anchor enters the data term
  $`\mathbf{P}_{SD}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}+\lambda\,\mathbf{W}_{\text{Teacher}}^{t-1}\mathbf{C}_{I}`$;
- •
  changing the *initialization*
  $`\mathbf{W}_{I}^{0}\to\mathbf{W}_{I}^{t-1}`$ modifies the null-space
  component $`\Pi_{\mathcal{Q}^{\perp}}(\cdot)`$ which
  Lemma [C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
  preserves from the initialization unchanged; this is what allows the
  orthogonal pretrained knowledge to be carried forward from one step to
  the next.

The dynamic-teacher scheme is therefore best understood as a sequence of
standard static-SD-style quadratic minimizations, with both the anchor
and the initialization updated between rounds in a way that affects
geometrically separate subspaces.
Proposition [C.7](#A3.Thmapptheorem7 "Proposition C.7 (Single-Step Solution). ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
below makes this decomposition concrete in closed form.

###### Proposition C.7 (Single-Step Solution).

Let
$`\mathbf{W}^{\star}_{\text{FT}}\!=\!\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$
be the minimum-norm solution for the direct finetuning task, and let
$`\mathcal{P}_{I}`$ be the orthogonal projector onto
$`\mathrm{range}(\mathbf{X}_{I})`$. The SD–WMA update at step $`t`$
yields

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{I}^{t}\;=\;\mathbf{W}_{I}^{t-1}(\mathbf{I}-\mathcal{P}_{I})\;+\;\frac{\lambda}{1+\lambda}\,\mathbf{W}_{\text{Teacher}}^{t-1}\mathcal{P}_{I}\;+\;\frac{1}{1+\lambda}\,\mathbf{W}^{\star}_{\text{FT}}.
``` |  | (18) |

###### Proof.

This proposition is immediate from applying
Lemma [C.1](#A3.Thmapptheorem1 "Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
to the objective in
Definition [C.3](#A3.Thmapptheorem3 "Definition C.3 (SD–WMA Objective (Repeated from Main Text)). ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning").
The objective at step $`t`$ has the same structure as static
self-distillation (analyzed in
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")),
but with the pretrained weights $`\mathbf{W}_{I}^{0}`$ in the
regularization term replaced by $`\mathbf{W}_{\text{Teacher}}^{t-1}`$,
and the initialization for gradient descent being
$`\mathbf{W}_{I}^{t-1}`$. Specifically, we find the minimizer of:
$`\min_{\mathbf{W}_{I}}\frac{1}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{Y}_{\text{FT}}\right\|_{\text{F}}^{2}+\frac{\lambda}{2}\left\|\mathbf{W}_{I}\mathbf{X}_{I}-\mathbf{W}_{\text{Teacher}}^{t-1}\mathbf{X}_{I}\right\|_{\text{F}}^{2}`$
This corresponds to the self-distillation case in
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
where $`\mathbf{W}_{I}^{0}`$ is effectively replaced by
$`\mathbf{W}_{\text{Teacher}}^{t-1}`$ for the purpose of defining the
fixed regularization target at this step. The solution form is then
directly obtained by substituting $`\mathbf{W}_{\text{Teacher}}^{t-1}`$
for $`\mathbf{W}_{I}^{0}`$ in the $`\mathbf{W}_{SD}`$ formula, which
yields:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I}^{t}=\mathbf{W}_{I}^{t-1}(\mathbf{I}-\mathcal{P}_{I})+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}+\frac{\lambda}{1+\lambda}\mathbf{W}_{\text{Teacher}}^{t-1}\mathcal{P}_{I}.
``` |  |

Recognizing
$`\mathbf{W}^{\star}_{\text{FT}}=\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}`$,
we get the desired result. ∎

The key advantage over static SD emerges from the teacher’s evolution.

###### Theorem C.8 (Bias-Free Convergence in the Task Subspace).

Let $`a=\tfrac{\lambda}{1+\lambda}`$ and define the teacher error
$`\mathbf{E}^{t}=(\mathbf{W}_{\text{Teacher}}^{t}-\mathbf{W}^{\star}_{\text{FT}})\mathcal{P}_{I}`$.
Then for any online weights $`\{\omega_{t}\}`$ as
in equation [15](#A3.E15 "Equation 15 ‣ Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)). ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"):

1.  (i)
    Teacher contraction.
    $`\mathbf{E}^{t}=\Big(1-\tfrac{\omega_{t}}{1+\lambda}\Big)\mathbf{E}^{t-1}`$.
2.  (ii)
    Student tracking.
    $`(\mathbf{W}_{I}^{t}-\mathbf{W}^{\star}_{\text{FT}})\mathcal{P}_{I}=a\,\mathbf{E}^{t-1}`$.
3.  (iii)
    Convergence. If $`\sum_{t\geq 1}\omega_{t}=\infty`$, then
    $`\mathbf{W}_{\text{Teacher}}^{t}\mathcal{P}_{I}\to\mathbf{W}^{\star}_{\text{FT}}`$
    and
    $`\mathbf{W}_{I}^{t}\mathcal{P}_{I}\to\mathbf{W}^{\star}_{\text{FT}}`$.

###### Proof.

Let $`\mathbf{W}_{I,\parallel}^{t}=\mathbf{W}_{I}^{t}\mathcal{P}_{I}`$
and
$`\mathbf{W}_{\text{Teacher},\parallel}^{t}=\mathbf{W}_{\text{Teacher}}^{t}\mathcal{P}_{I}`$.
From
Proposition [C.7](#A3.Thmapptheorem7 "Proposition C.7 (Single-Step Solution). ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
projecting onto the subspace $`\mathrm{range}(\mathbf{X}_{I})`$ gives:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I,\parallel}^{t}=\mathbf{W}_{I}^{t-1}(\mathbf{I}-\mathcal{P}_{I})\mathcal{P}_{I}+\frac{\lambda}{1+\lambda}\,\mathbf{W}_{\text{Teacher}}^{t-1}\mathcal{P}_{I}+\frac{1}{1+\lambda}\,\mathbf{W}^{\star}_{\text{FT}}\mathcal{P}_{I}.
``` |  |

Since $`(\mathbf{I}-\mathcal{P}_{I})\mathcal{P}_{I}=\mathbf{0}`$, and
$`\mathbf{W}^{\star}_{\text{FT}}`$ is already in the parallel subspace
(by definition), we have
$`\mathbf{W}^{\star}_{\text{FT}}\mathcal{P}_{I}=\mathbf{W}^{\star}_{\text{FT}}`$.
So,

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{I,\parallel}^{t}=a\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+(1-a)\,\mathbf{W}^{\star}_{\text{FT}},
``` |  | (19) |

where $`a=\tfrac{\lambda}{1+\lambda}`$. (ii) Subtracting
$`\mathbf{W}^{\star}_{\text{FT}}`$ from both sides of
equation [19](#A3.E19 "Equation 19 ‣ Proof. ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"):

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I,\parallel}^{t}-\mathbf{W}^{\star}_{\text{FT}}=a\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+(1-a)\,\mathbf{W}^{\star}_{\text{FT}}-\mathbf{W}^{\star}_{\text{FT}}=a\,(\mathbf{W}_{\text{Teacher},\parallel}^{t-1}-\mathbf{W}^{\star}_{\text{FT}})=a\,\mathbf{E}^{t-1}.
``` |  |

This proves part (ii).

(i) Now consider the teacher recursion
(Definition [C.4](#A3.Thmapptheorem4 "Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)). ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
projected onto $`\mathcal{P}_{I}`$:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\text{Teacher},\parallel}^{t}=(1-\omega_{t})\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+\omega_{t}\,\mathbf{W}_{I,\parallel}^{t}.
``` |  |

Substitute
equation [19](#A3.E19 "Equation 19 ‣ Proof. ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
into this:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{\text{Teacher},\parallel}^{t}=(1-\omega_{t})\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+\omega_{t}\,(a\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+(1-a)\,\mathbf{W}^{\star}_{\text{FT}}).
``` |  |

Rearranging terms to isolate
$`\mathbf{E}^{t}=\mathbf{W}_{\text{Teacher},\parallel}^{t}-\mathbf{W}^{\star}_{\text{FT}}`$:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbf{W}_{\text{Teacher},\parallel}^{t}-\mathbf{W}^{\star}_{\text{FT}}`$ | $`\displaystyle=(1-\omega_{t})\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+\omega_{t}\,a\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}+\omega_{t}\,(1-a)\,\mathbf{W}^{\star}_{\text{FT}}-\mathbf{W}^{\star}_{\text{FT}}`$ |  |
|  |  | $`\displaystyle=(1-\omega_{t}+\omega_{t}a)\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}-(1-\omega_{t}(1-a))\,\mathbf{W}^{\star}_{\text{FT}}`$ |  |
|  |  | $`\displaystyle=(1-\omega_{t}(1-a))\,(\mathbf{W}_{\text{Teacher},\parallel}^{t-1}-\mathbf{W}^{\star}_{\text{FT}}).`$ |  |

Since $`1-a=1-\frac{\lambda}{1+\lambda}=\frac{1}{1+\lambda}`$, we have:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{E}^{t}=\Big(1-\tfrac{\omega_{t}}{1+\lambda}\Big)\mathbf{E}^{t-1}.
``` |  |

This proves part (i).

(iii) Iterating the recurrence relation from part (i):

|  |  |  |
|----|----|----|
|  |
``` math
\|\mathbf{E}^{t}\|_{F}=\left(\prod_{k=1}^{t}\Big(1-\tfrac{\omega_{k}}{1+\lambda}\Big)\right)\|\mathbf{E}^{0}\|_{F}.
``` |  |

For $`\mathbf{E}^{t}`$ to converge to $`0`$, we need the product term to
converge to $`0`$. This occurs if and only if the sum
$`\sum_{k=1}^{\infty}\frac{\omega_{k}}{1+\lambda}`$ diverges to
$`\infty`$. Since $`\lambda>0`$, $`1+\lambda`$ is a finite constant.
Thus, the condition for convergence is
$`\sum_{k=1}^{\infty}\omega_{k}=\infty`$. From
Definition [C.4](#A3.Thmapptheorem4 "Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)). ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
$`\omega_{t}=\frac{\alpha_{t}}{\sum_{j=0}^{t}\alpha_{j}}`$. If
$`\kappa(\tau_{t})`$ is a continuous function on $`[0,1]`$ that is
non-zero on a set of positive measure, then $`\sum_{k}\alpha_{k}`$ will
diverge as $`T\to\infty`$ (assuming $`t`$ goes up to $`T`$), and thus
$`\sum_{k}\omega_{k}`$ will diverge. For common kernels like Beta
distributions (e.g., arcsine kernel), this condition holds. Since
$`\mathbf{E}^{t}\to\mathbf{0}`$, we have
$`\mathbf{W}_{\text{Teacher}}^{t}\mathcal{P}_{I}\to\mathbf{W}^{\star}_{\text{FT}}`$.
From part (ii), as $`\mathbf{E}^{t-1}\to\mathbf{0}`$, it follows that
$`(\mathbf{W}_{I}^{t}-\mathbf{W}^{\star}_{\text{FT}})\mathcal{P}_{I}\to\mathbf{0}`$,
meaning
$`\mathbf{W}_{I}^{t}\mathcal{P}_{I}\to\mathbf{W}^{\star}_{\text{FT}}`$.
∎

###### Corollary C.9 (Linear rate under a bounded step weight).

If $`\omega_{t}\geq\omega_{\min}>0`$ for all $`t\leq T`$, then

|  |  |  |
|----|----|----|
|  |
``` math
\|(\mathbf{W}_{\text{Teacher}}^{t}-\mathbf{W}^{\star}_{\text{FT}})\mathcal{P}_{I}\|_{F}\;\leq\;\Big(1-\tfrac{\omega_{\min}}{1+\lambda}\Big)^{t}\,\|(\mathbf{W}_{\text{Teacher}}^{0}-\mathbf{W}^{\star}_{\text{FT}})\mathcal{P}_{I}\|_{F}.
``` |  |

Hence the training loss in the task subspace decays at least
geometrically to the minimum, whereas static SD converges to a biased
point for any fixed $`\lambda>0`$.

###### Proof.

This follows directly from
Theorem [C.8](#A3.Thmapptheorem8 "Theorem C.8 (Bias-Free Convergence in the Task Subspace). ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
part (i). If $`\omega_{t}\geq\omega_{\min}`$, then
$`1-\tfrac{\omega_{t}}{1+\lambda}\leq 1-\tfrac{\omega_{\min}}{1+\lambda}`$.
Since $`0<\omega_{\min}\leq 1`$ and $`\lambda>0`$, we have
$`0<\tfrac{\omega_{\min}}{1+\lambda}<1`$, so
$`0<1-\tfrac{\omega_{\min}}{1+\lambda}<1`$. Thus, the error contracts
geometrically. Static SD, as derived in
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
converges to a solution that is a convex combination of
$`\mathbf{W}_{I}^{0}\mathcal{P}_{I}`$ and
$`\mathbf{W}^{\star}_{\text{FT}}`$. This is a biased point unless
$`\mathbf{W}_{I}^{0}\mathcal{P}_{I}=\mathbf{W}^{\star}_{\text{FT}}`$. ∎

##### Geometric Interpretation of Dynamic Self-Distillation.

We decompose the dynamics into orthogonal and parallel components with
respect to $`\mathrm{range}(\mathbf{X}_{I})`$.

##### Orthogonal Preservation.

Applying $`(\mathbf{I}-\mathcal{P}_{I})`$ to
Proposition [C.7](#A3.Thmapptheorem7 "Proposition C.7 (Single-Step Solution). ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
and using the idempotency of projectors,
$`\mathcal{P}_{I}(\mathbf{I}-\mathcal{P}_{I})=\mathbf{0}`$, we get:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I}^{t}(\mathbf{I}-\mathcal{P}_{I})\;=\;\mathbf{W}_{I}^{t-1}(\mathbf{I}-\mathcal{P}_{I})\;=\;\cdots\;=\;\mathbf{W}_{I}^{0}(\mathbf{I}-\mathcal{P}_{I}),
``` |  |

This demonstrates that SD–WMA preserves pretrained knowledge orthogonal
to the finetuning subspace, just like static self-distillation.

##### Adaptive Task-Space Evolution.

Within the task subspace, the student update is given by:

|  |  |  |
|----|----|----|
|  |
``` math
\mathbf{W}_{I,\parallel}^{t}\;=\;\frac{\lambda}{1+\lambda}\,\mathbf{W}_{\text{Teacher},\parallel}^{t-1}\;+\;\frac{1}{1+\lambda}\,\mathbf{W}^{\star}_{\text{FT}}.
``` |  |

Early training ($`t`$ small): The teacher
$`\mathbf{W}_{\text{Teacher}}^{t-1}`$ is still close to
$`\mathbf{W}_{I}^{0}`$ (as $`\omega_{k}`$ for small $`k`$ is often high
for U-shaped kernels, or simply because few updates have occurred). This
means the teacher acts as a strong anchor, mitigating catastrophic
forgetting during volatile updates.

Late training ($`t`$ large): As $`t\to\infty`$,
Theorem [C.8](#A3.Thmapptheorem8 "Theorem C.8 (Bias-Free Convergence in the Task Subspace). ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")
shows that $`\mathbf{W}_{\text{Teacher},\parallel}^{t-1}`$ converges to
$`\mathbf{W}^{\star}_{\text{FT}}`$. Substituting this into the student
update:

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{t\to\infty}\mathbf{W}_{I,\parallel}^{t}\;=\;\frac{\lambda}{1+\lambda}\,\mathbf{W}^{\star}_{\text{FT}}\;+\;\frac{1}{1+\lambda}\,\mathbf{W}^{\star}_{\text{FT}}\;=\;\mathbf{W}^{\star}_{\text{FT}}.
``` |  |

Thus, the dynamic teacher adapts, reducing anchor bias and enabling
exact convergence to $`\mathbf{W}^{\star}_{\text{FT}}`$ in
$`\mathrm{range}(\mathbf{X}_{I})`$.

###### Proposition C.10 (Dominance over Static SD).

If
$`\|\mathbf{W}_{\text{Teacher},\parallel}^{t-1}-\mathbf{W}_{\text{FT}}^{\star}\|_{F}\leq\|\mathbf{W}_{I,\parallel}^{0}-\mathbf{W}_{\text{FT}}^{\star}\|_{F}`$,
then for the same $`\lambda`$ the SD–WMA update attains lower squared
error than static SD in the task subspace.

###### Proof.

Let $`\mathbf{W}^{\star}_{\text{static SD}}`$ be the solution for static
SD (from
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
The squared error from $`\mathbf{W}^{\star}_{\text{FT}}`$ in the task
subspace for static SD is proportional to
$`\|\frac{\lambda}{1+\lambda}\mathbf{W}_{I}^{0}\mathcal{P}_{I}-\mathbf{W}^{\star}_{\text{FT}}\|_{F}^{2}`$.
For dynamic SD, the instantaneous target is proportional to
$`\|\frac{\lambda}{1+\lambda}\mathbf{W}_{\text{Teacher}}^{t-1}\mathcal{P}_{I}-\mathbf{W}^{\star}_{\text{FT}}\|_{F}^{2}`$.
If the teacher is closer to $`\mathbf{W}^{\star}_{\text{FT}}`$ in the
parallel subspace than the initial model $`\mathbf{W}_{I}^{0}`$, i.e.,
$`\|\mathbf{W}_{\text{Teacher},\parallel}^{t-1}-\mathbf{W}_{\text{FT}}^{\star}\|_{F}\leq\|\mathbf{W}_{I,\parallel}^{0}-\mathbf{W}_{\text{FT}}^{\star}\|_{F}`$,
then the dynamic SD solution will be closer to
$`\mathbf{W}^{\star}_{\text{FT}}`$ in that subspace, thus achieving
lower error. The convergence result
(Theorem [C.8](#A3.Thmapptheorem8 "Theorem C.8 (Bias-Free Convergence in the Task Subspace). ‣ From static to dynamic SD as a sequence of quadratic problems. ‣ C.5.3 Convergence Analysis ‣ C.5 Dynamic Self-Distillation: WMA Details and Convergence ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
guarantees that the teacher gets arbitrarily close to
$`\mathbf{W}^{\star}_{\text{FT}}`$, eventually satisfying this
condition. ∎

### C.6 Distillation Loss Definitions in TRACER

TRACER employs a composite self-distillation loss
$`\mathcal{L}_{\text{SD-WMA}}`$ from the WMA teacher, which consists of
several complementary terms to transfer different aspects of knowledge.
Let $`\mathbf{T}`$ denote the teacher model and $`\mathbf{S}`$ denote
the student model. $`\mathbf{h}_{I_{i}}^{\mathbf{T}}`$ and
$`\mathbf{h}_{T_{i}}^{\mathbf{T}}`$ are image and text embeddings from
the teacher for the $`i`$-th example, and similarly for the student.
$`\tau`$ denotes the temperature parameter.

##### Feature Distillation (FD).

This loss directly minimizes the Mean Squared Error between the
student’s and teacher’s embeddings for each corresponding image-text
pair in a mini-batch of size $`N`$. It helps align the feature spaces.

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{FD}}=\frac{1}{N}\sum_{i=1}^{N}\left(\left\|\mathbf{h}_{I_{i}}^{\mathbf{T}}-\mathbf{h}_{I_{i}}^{\mathbf{S}}\right\|_{2}^{2}+\left\|\mathbf{h}_{T_{i}}^{\mathbf{T}}-\mathbf{h}_{T_{i}}^{\mathbf{S}}\right\|_{2}^{2}\right)
``` |  | (20) |

##### Contrastive Relational Distillation (CRD).

CRD aligns the student’s contrastive similarity distribution with the
teacher’s. We first compute the image-to-text ($`p`$) and text-to-image
($`q`$) softmax distributions for both student and teacher across the
mini-batch:

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | $`\displaystyle p_{i}^{\mathbf{T}}[j]`$ | $`\displaystyle=\frac{\exp(\mathbf{h}_{I_{i}}^{\mathbf{T}\top}\mathbf{h}_{T_{j}}^{\mathbf{T}}/\tau)}{\sum_{b=1}^{N}\exp(\mathbf{h}_{I_{i}}^{\mathbf{T}\top}\mathbf{h}_{T_{b}}^{\mathbf{T}}/\tau)},`$ | $`\displaystyle p_{i}^{\mathbf{S}}[j]`$ | $`\displaystyle=\frac{\exp(\mathbf{h}_{I_{i}}^{\mathbf{S}\top}\mathbf{h}_{T_{j}}^{\mathbf{S}}/\tau)}{\sum_{b=1}^{N}\exp(\mathbf{h}_{I_{i}}^{\mathbf{S}\top}\mathbf{h}_{T_{b}}^{\mathbf{S}}/\tau)}`$ |  | (21) |
|  | $`\displaystyle q_{i}^{\mathbf{T}}[j]`$ | $`\displaystyle=\frac{\exp(\mathbf{h}_{T_{i}}^{\mathbf{T}\top}\mathbf{h}_{I_{j}}^{\mathbf{T}}/\tau)}{\sum_{b=1}^{N}\exp(\mathbf{h}_{T_{i}}^{\mathbf{T}\top}\mathbf{h}_{I_{b}}^{\mathbf{T}}/\tau)},`$ | $`\displaystyle q_{i}^{\mathbf{S}}[j]`$ | $`\displaystyle=\frac{\exp(\mathbf{h}_{T_{i}}^{\mathbf{S}\top}\mathbf{h}_{I_{j}}^{\mathbf{S}}/\tau)}{\sum_{b=1}^{N}\exp(\mathbf{h}_{T_{i}}^{\mathbf{S}\top}\mathbf{h}_{I_{b}}^{\mathbf{S}}/\tau)}`$ |  | (22) |

The distillation loss is the sum of the KL-divergences between these
distributions, averaged over the batch.

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{CRD}}=\frac{1}{N}\sum_{i=1}^{N}\left(D_{KL}(p_{i}^{\mathbf{T}}\|p_{i}^{\mathbf{S}})+D_{KL}(q_{i}^{\mathbf{T}}\|q_{i}^{\mathbf{S}})\right)
``` |  | (23) |

##### Interactive Contrastive Learning (ICL).

ICL forces the student to learn within the teacher’s embedding space by
performing contrastive learning between the student’s anchor embeddings
and the teacher’s key embeddings. The loss is a symmetric InfoNCE
objective computed on these mixed-model pairs.

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{ICL}}=-\frac{1}{2N}\sum_{i=1}^{N}\left(\log\frac{\exp(\mathbf{h}_{I_{i}}^{\mathbf{S}\top}\mathbf{h}_{T_{i}}^{\mathbf{T}}/\tau)}{\sum_{j=1}^{N}\exp(\mathbf{h}_{I_{i}}^{\mathbf{S}\top}\mathbf{h}_{T_{j}}^{\mathbf{T}}/\tau)}+\log\frac{\exp(\mathbf{h}_{T_{i}}^{\mathbf{S}\top}\mathbf{h}_{I_{i}}^{\mathbf{T}}/\tau)}{\sum_{j=1}^{N}\exp(\mathbf{h}_{T_{i}}^{\mathbf{S}\top}\mathbf{h}_{I_{j}}^{\mathbf{T}}/\tau)}\right)
``` |  | (24) |

##### Cross Knowledge Distillation (Cross-KD).

This method acts as a hybrid of CRD and ICL. It aligns the
student-to-teacher cross-modal similarity distribution with the
teacher’s self-modal distribution using KL-divergence. We define the
student-to-teacher cross-modal distributions
($`p^{\mathbf{S}\to\mathbf{T}}`$, $`q^{\mathbf{S}\to\mathbf{T}}`$) as:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle p_{i}^{\mathbf{S}\to\mathbf{T}}[j]`$ | $`\displaystyle=\frac{\exp(\mathbf{h}_{I_{i}}^{\mathbf{S}\top}\mathbf{h}_{T_{j}}^{\mathbf{T}}/\tau)}{\sum_{b=1}^{N}\exp(\mathbf{h}_{I_{i}}^{\mathbf{S}\top}\mathbf{h}_{T_{b}}^{\mathbf{T}}/\tau)}`$ |  | (25) |
|  | $`\displaystyle q_{i}^{\mathbf{S}\to\mathbf{T}}[j]`$ | $`\displaystyle=\frac{\exp(\mathbf{h}_{T_{i}}^{\mathbf{S}\top}\mathbf{h}_{I_{j}}^{\mathbf{T}}/\tau)}{\sum_{b=1}^{N}\exp(\mathbf{h}_{T_{i}}^{\mathbf{S}\top}\mathbf{h}_{I_{b}}^{\mathbf{T}}/\tau)}`$ |  | (26) |

The loss then minimizes the divergence from these distributions to the
teacher’s own relational distributions, $`p_{i}^{\mathbf{T}}`$ and
$`q_{i}^{\mathbf{T}}`$.

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{CrossKD}}=\frac{1}{2N}\sum_{i=1}^{N}\left(D_{KL}(p_{i}^{\mathbf{T}}\|p_{i}^{\mathbf{S}\to\mathbf{T}})+D_{KL}(q_{i}^{\mathbf{T}}\|q_{i}^{\mathbf{S}\to\mathbf{T}})\right)
``` |  | (27) |

##### Geometric bridge to composite distillation.

Our analysis decomposes learning into an orthogonal preservation term
and an in-subspace mixing term
(Equation [2](#S3.E2 "Equation 2 ‣ 3.2 Loss Reformulation via the Contrastive Target Matrix ‣ 3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
The composite distillation terms are chosen to preserve *structure*
consistent with this geometry: (i) FD anchors pointwise embeddings,
biasing updates toward the teacher component within
$`\mathrm{range}(\mathbf{X}_{I})`$ while damping drift in orthogonal
directions; (ii) CRD aligns the teacher’s batch-wise similarity
*distributions*, preserving inter-example geometry (a probabilistic
surrogate for preserving
$`\mathbf{S}{=}\mathbf{H}_{I}^{\top}\mathbf{H}_{T}`$); (iii) ICL
performs contrastive learning in the teacher’s semantic space,
encouraging the student to operate on the teacher’s subspace and thus to
mix along task-relevant directions; and (iv) CrossKD aligns cross-modal
logits to transmit cross-modal relational structure that vanilla InfoNCE
may underweight. Together with the WMA teacher, these terms
operationalize the geometric principle at feature-, relation-, and
cross-modal levels.

### C.7 Connection to Robustness via Inter-Class Feature Sharing

The self-distillation approach, particularly with a dynamic WMA teacher,
can be understood through the lens of recent theoretical work on
multimodal contrastive learning’s robustness mechanisms. [Xue et al.
(2024)](#bib.bib15) identify *inter-class feature sharing* as a key
mechanism behind MMCL’s strong robustness to distribution shift, where
models learn to leverage information about features appearing across
different classes to dissociate spurious correlations.

Building on the insight that self-distillation acts as instance-specific
label smoothing ([Zhang and Sabuncu, 2020](#bib.bib89)), we argue that
the self-distillation method provides a similar robustness benefit by
acting as an informed label smoothing mechanism that preserves
inter-class similarities learned during pretraining. To see this
connection, recall the self-distillation solution from
Theorem [C.2](#A3.Thmapptheorem2 "Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). ‣ Synthesis. ‣ C.3 Unified Framework for Contrastive Finetuning: Proofs and Details ‣ Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"):

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{W}_{SD}=\mathbf{W}_{I}^{0}\left(\mathbf{I}-\frac{1}{1+\lambda}\mathcal{P}_{I}\right)+\frac{1}{1+\lambda}\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}
``` |  | (28) |

This solution exhibits three key properties that enhance robustness:

##### Preservation of Cross-Class Knowledge.

The term
$`\mathbf{W}_{I}^{0}\left(\mathbf{I}-\frac{1}{1+\lambda}\mathcal{P}_{I}\right)`$
maintains the pretrained model’s understanding of feature relationships
across classes. Unlike direct finetuning which completely overwrites
representations in the finetuning subspace, self-distillation retains a
weighted contribution from the original cross-class feature covariances.
This is analogous to how [Xue et al. (2024)](#bib.bib15) show that MMCL
leverages features appearing in multiple contexts to learn their
independence from class labels.

##### Informed Smoothing via Pretrained Similarities.

By regularizing towards $`\mathbf{W}_{I}^{0}\mathbf{X}_{I}`$ rather than
arbitrary targets, self-distillation performs label smoothing that is
informed by the pretrained model’s learned inter-class similarities.
This extends the instance-specific label smoothing interpretation of
[Zhang and Sabuncu (2020)](#bib.bib89) to the finetuning setting, where
the smoothing is guided by pretrained knowledge. This regularization
preserves the cross-covariance structure that [Xue et al.
(2024)](#bib.bib15) identify as crucial for robustness, specifically the
covariance between features that appear independently across different
classes.

##### Robustness Through Feature Independence.

Within the finetuning subspace, self-distillation computes a convex
combination:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\frac{\lambda}{1+\lambda}\left(\mathbf{W}_{I}^{0}\mathcal{P}_{I}\right)+\frac{1}{1+\lambda}\left(\mathbf{Y}_{\text{FT}}\mathbf{X}_{I}^{\top}(\mathbf{X}_{I}\mathbf{X}_{I}^{\top})^{+}\right)
``` |  | (29) |

This combination maintains the pretrained understanding of feature
independence while adapting to the new task. As [Xue et al.
(2024)](#bib.bib15) demonstrate in their Data Model 2, when features can
occur independently across classes (e.g., “trees without green leaves”
appearing in non-tree classes), models that preserve these cross-class
relationships achieve stronger robustness. The self-distillation
mechanism explicitly preserves these relationships through the weighted
contribution of $`\mathbf{W}_{I}^{0}\mathcal{P}_{I}`$.

The hyperparameter $`\lambda`$ controls the strength of this inter-class
knowledge preservation: larger values of $`\lambda`$ maintain more of
the pretrained model’s understanding of how features vary independently
across different contexts, potentially enhancing robustness to
distribution shift. This suggests that self-distillation’s effectiveness
stems not merely from preventing catastrophic forgetting, but from
actively preserving the rich inter-class feature relationships that
contribute to robustness, a mechanism that parallels the theoretical
insights of [Xue et al. (2024)](#bib.bib15) on why MMCL achieves strong
out-of-distribution generalization.

## Appendix D TRACER Algorithm

1: Pretrained CLIP model
$`\theta_{\text{CLIP}}^{0}=\{\mathcal{E}_{\text{Image}}^{0},\mathcal{E}_{\text{Text}}^{0}\}`$

2: Finetuning dataset
$`\mathcal{D}_{\text{FT}}=\{(\mathbf{x}_{I},\mathbf{x}_{T})\}_{i=1}^{N}`$

3: Learning rate $`\eta`$, Weight decay $`\delta`$, Batch size $`B`$,
Number of epochs $`E`$

4: Distillation coefficient $`\lambda_{\text{SD}}`$

5: WMA kernel $`\kappa(\tau_{k})`$ (e.g., Beta($`\beta_{1},\beta_{2}`$))
and total steps $`T_{\text{total}}`$

6: Temperature $`\tau_{\text{NCE}}`$ for InfoNCE losses

7: Initialize Student Model:
$`\theta_{S}\leftarrow\theta_{\text{CLIP}}^{0}`$ (image encoder
$`\mathcal{E}_{\text{Image},S}`$, text encoder
$`\mathcal{E}_{\text{Text},S}`$)

8: Initialize Teacher Model:
$`\theta_{T}\leftarrow\text{copy}(\theta_{S})`$

9: Initialize Optimizer:
$`\text{Opt}\leftarrow\text{AdamW}(\theta_{S}.\text{parameters()},\eta,\delta)`$

10: Initialize WMA state: $`\text{cumulative\_alpha}\leftarrow 0`$

11: $`\text{global\_step}\leftarrow 0`$

12: for $`\text{epoch}=1\text{ to }E`$ do

13:   for
$`\text{batch}=\{(\mathbf{x}_{I},\mathbf{x}_{T})\}_{i=1}^{B}\text{ in }\mathcal{D}_{\text{FT}}`$
do

14:    $`\text{global\_step}\leftarrow\text{global\_step}+1`$ {— Student
Forward Pass —}

15:
   $`\mathbf{h}_{I,S}\leftarrow\mathcal{E}_{\text{Image},S}(\mathbf{x}_{I})`$

16:
   $`\mathbf{h}_{T,S}\leftarrow\mathcal{E}_{\text{Text},S}(\mathbf{x}_{T})`$

17:    Normalize student embeddings:
$`\mathbf{h}_{I,S}\leftarrow\text{normalize}(\mathbf{h}_{I,S})`$,
$`\mathbf{h}_{T,S}\leftarrow\text{normalize}(\mathbf{h}_{T,S})`$ {—
Compute Multi-Modal Contrastive Loss ($`\mathcal{L}_{\text{MMCL}}`$) —}

18:
   $`\text{logits}_{I\leftrightarrow T}\leftarrow\mathbf{h}_{I,S}\cdot\mathbf{h}_{T,S}^{\top}/\tau_{\text{NCE}}`$

19:
   $`\mathcal{L}_{\text{MMCL}}\leftarrow\text{InfoNCE}(\text{logits}_{I\leftrightarrow T})+\text{InfoNCE}(\text{logits}_{I\leftrightarrow T}^{\top})`$
{Symmetric InfoNCE} {— Teacher Forward Pass (with no gradient updates)
—}

20:    $`\textbf{with }\text{torch.no\_grad}():`$

21:
   $`\mathbf{h}_{I,T}\leftarrow\mathcal{E}_{\text{Image},T}(\mathbf{x}_{I})`$

22:
   $`\mathbf{h}_{T,T}\leftarrow\mathcal{E}_{\text{Text},T}(\mathbf{x}_{T})`$

23:    Normalize teacher embeddings:
$`\mathbf{h}_{I,T}\leftarrow\text{normalize}(\mathbf{h}_{I,T})`$,
$`\mathbf{h}_{T,T}\leftarrow\text{normalize}(\mathbf{h}_{T,T})`$ {—
Compute Dynamic Self-Distillation Loss ($`\mathcal{L}_{\text{SD-WMA}}`$)
—}

24:
   $`\mathcal{L}_{\text{FD}}\leftarrow\frac{1}{B}\sum_{i=1}^{B}(\left\|\mathbf{h}_{I,T}[i]-\mathbf{h}_{I,S}[i]\right\|_{2}^{2}+\left\|\mathbf{h}_{T,T}[i]-\mathbf{h}_{T,S}[i]\right\|_{2}^{2})`$

25:
   $`\mathcal{L}_{\text{CRD}}\leftarrow\text{KL}(\text{softmax}(\mathbf{h}_{I,T}\mathbf{h}_{T,T}^{\top}/\tau_{\text{NCE}})||\text{softmax}(\mathbf{h}_{I,S}\mathbf{h}_{T,S}^{\top}/\tau_{\text{NCE}}))`$
{+ text-to-image}

26:
   $`\mathcal{L}_{\text{ICL}}\leftarrow\text{InfoNCE}(\mathbf{h}_{I,S},\mathbf{h}_{T,T})+\text{InfoNCE}(\mathbf{h}_{T,S},\mathbf{h}_{I,T})`$

27:
   $`\mathcal{L}_{\text{CrossKD}}\leftarrow\text{KL}(\text{softmax}(\mathbf{h}_{I,T}\mathbf{h}_{T,T}^{\top}/\tau_{\text{NCE}})||\text{softmax}(\mathbf{h}_{I,S}\mathbf{h}_{T,T}^{\top}/\tau_{\text{NCE}}))`$
{+ text-to-image}

28:
   $`\mathcal{L}_{\text{SD-WMA}}\leftarrow\mathcal{L}_{\text{FD}}+\mathcal{L}_{\text{CRD}}+\mathcal{L}_{\text{ICL}}+\mathcal{L}_{\text{CrossKD}}`$
{— Total Loss and Optimization —}

29:
   $`\mathcal{L}_{\text{Total}}\leftarrow\mathcal{L}_{\text{MMCL}}+\lambda_{\text{SD}}\cdot\mathcal{L}_{\text{SD-WMA}}`$

30:    $`\text{Opt.zero\_grad}()`$

31:    $`\mathcal{L}_{\text{Total}}.\text{backward}()`$

32:    $`\text{Opt.step}()`$ {— Update WMA Teacher —}

33:
   $`\tau_{\text{current}}\leftarrow(\text{global\_step}+c_{1})/(T_{\text{total}}+c_{2})`$
{Normalized time}

34:
   $`\alpha_{\text{current}}\leftarrow\kappa(\tau_{\text{current}})`$

35:
   $`\text{cumulative\_alpha}\leftarrow\text{cumulative\_alpha}+\alpha_{\text{current}}`$

36:
   $`\omega_{\text{current}}\leftarrow\alpha_{\text{current}}/\text{cumulative\_alpha}`$

37:    for parameter $`p_{S}`$ in $`\theta_{S}`$ and $`p_{T}`$ in
$`\theta_{T}`$ do

38:
      $`p_{T}\leftarrow(1-\omega_{\text{current}})\cdot p_{T}+\omega_{\text{current}}\cdot p_{S}`$

39:    end for

40:   end for

41: end for

42: return $`\theta_{S}`$

Algorithm 1 TRACER (Trajectory-Robust Anchoring for Contrastive Encoder
Regularization)

## Appendix E Reproducibility Details

To ensure full reproducibility, we detail our experimental setup, key
hyperparameters, and implementation. The full codebase, configuration
files, and reproduction scripts are publicly available at
[https://github.com/HesamAsad/TRACER](https://github.com/HesamAsad/TRACER).

### E.1 Computational Environment

- •
  Operating System: Linux kernel 5.14.0-427.42.1.el9_4.x86_64.
- •
  GPU Hardware: NVIDIA H100 80GB HBM3.
- •
  NVIDIA Driver Version: 550.144.03.
- •
  CUDA Version: 12.4.
- •
  Python Version: 3.10.4.
- •
  PyTorch Version: 2.0.1+ (with CUDA support).

### E.2 Implementation and Training Details

Our implementation extends the OpenAI CLIP framework.

- •
  Model Architectures: We use pretrained CLIP models (ViT-B/16,
  ResNet50, ViT-L/14) from OpenAI’s official clip library.
- •
  Total Loss:
  $`\mathcal{L}_{\text{TRACER}}=\mathcal{L}_{\text{MMCL}}+\lambda_{\text{SD}}\,\mathcal{L}_{\text{SD-WMA}}`$.

  - –
    $`\mathcal{L}_{\text{MMCL}}`$: Symmetric InfoNCE loss, directly
    leveraging OpenAI CLIP’s core loss implementation. Optional
    cross-Frobenius regularizer coefficient was set to $`0.05`$.
  - –
    $`\mathcal{L}_{\text{SD-WMA}}`$: A composite self-distillation loss.
    For TRACER, this comprises Feature Distillation (FD), Contrastive
    Relational Distillation (CRD), Interactive Contrastive Learning
    (ICL), and Cross Knowledge Distillation (Cross-KD).
- •
  WMA Teacher: A custom Weighted Moving Average (WMA) teacher
  implementation, whose weighting kernel is a Beta distribution with
  $`\beta_{1}=\beta_{2}=0.5`$.

### E.3 Key Hyperparameters

The following hyperparameters were used for TRACER finetuning on
ImageNet-1K:

- •
  Epochs: 10.
- •
  Optimizer: AdamW.
- •
  Learning Rate: $`1\times 10^{-5}`$.
- •
  Weight Decay: 0.1.
- •
  Batch Size: 512 (ViT-B/16, RN50), 224 (ViT-L/14).
- •
  Warmup Length: 500 steps (cosine LR schedule).
- •
  Mixed Precision: Enabled using torch.amp.autocast with torch.bfloat16.
- •
  Distillation Coefficient $`\lambda_{\text{SD}}`$: 0.9.
- •
  WMA Beta Kernel Parameter: 0.5 (for Beta(0.5,0.5) kernel, i.e.,
  arcsine distribution).
- •
  Teacher Update Frequency: 0 or 1 (update every step).

### E.4 Data Processing

Standard OpenAI CLIP image preprocessing was applied. Input images are
sourced from ImageNet-1K, and finetuning captions from OpenAI class
templates.

### E.5 Code Availability

The full codebase is publicly available at
[https://github.com/HesamAsad/TRACER](https://github.com/HesamAsad/TRACER)
to facilitate direct reproduction. The repository includes the TRACER
training pipeline, evaluation scripts for all reported ImageNet and OOD
benchmarks, configuration files for the three CLIP backbones (RN50,
ViT-B/16, ViT-L/14), and the WMA teacher implementation.

## Appendix F Extended Discussion

This appendix expands on three discussion points that we touch on
briefly in the main text: how TRACER’s gain scales across backbones, how
TRACER relates to parameter-efficient and prompt-based adaptation, and
how the theory could be extended beyond the linearized setting.

### F.1 Backbone scaling: ViT-B/16 vs. ViT-L/14

A natural question is why TRACER’s OOD gain over CaRot is larger on
ViT-B/16 ($`+1.53`$ Avg. shifts in
Table [1](#S5.T1 "Table 1 ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
than on ViT-L/14 ($`+1.19`$ Avg. shifts in
Table [7](#A2.T7 "Table 7 ‣ Additional Experimental Results. ‣ Appendix B Additional Experiments and Ablations ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
Three factors plausibly contribute. First, ViT-L/14 is a substantially
stronger zero-shot starting point ($`70.93\%`$ vs. $`58.39\%`$ OOD), so
there is less absolute headroom for any regularizer to recover; both
methods sit closer to a ceiling, compressing differences. Second, even
with a smaller absolute gap, TRACER still attains the *best OOD
accuracy* among all compared methods on ViT-L/14 ($`75.32\%`$, vs.
$`74.13\%`$ for CaRot), with consistent gains on the hardest shifts
(IN-R $`+1.74`$, IN-A $`+2.19`$, ObjectNet $`+1.71`$), mirroring the
ViT-B/16 pattern. Third, TRACER scales *favorably in compute*: its
distillation cost is $`\mathcal{O}(B^{2})`$ in the batch and matches
standard EMA cost for the teacher update
(Table [4](#S5.T4 "Table 4 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")),
so its overhead does *not* grow with the cubic-in-$`d`$ spectral term
that dominates CaRot at large $`d`$. We therefore expect the favorable
cost–quality trade-off to persist or improve as backbones grow. We frame
this as an empirical observation: we do *not* claim a demonstrated
scaling law for TRACER across the full vision–language model size
spectrum, and we view systematic studies on larger VLMs (e.g., SigLIP
variants and CLIP-style backbones at $`\geq`$ViT-H/14) as natural
follow-up work.

### F.2 Relation to PEFT and prompt-based adaptation

TRACER is a regularization mechanism that operates on the standard
full-finetuning gradient flow: it modifies *which solution* the
optimizer converges to (anchoring it to a trajectory-weighted teacher),
but it does *not* restrict *where* parameters can move.
Parameter-efficient finetuning (PEFT) methods such as adapters ([Houlsby
et al., 2019](#bib.bib66)) and LoRA ([Hu et al., 2022](#bib.bib68)), as
well as prompt-based adaptation ([Zhou et al., 2022](#bib.bib107); [Jia
et al., 2022](#bib.bib108)), take the orthogonal route: they restrict
the hypothesis class so that only a small set of injected parameters or
input tokens can change, leaving the pretrained weights untouched by
construction. The two strategies address different failure modes of
finetuning. PEFT bounds the *capacity* for drift away from the
pretrained model; TRACER bounds the *direction* and *magnitude* of drift
within whatever capacity the optimizer is given. Because TRACER’s WMA
teacher and composite distillation losses act on the student’s
parameters (or, equivalently, its embedding statistics) regardless of
how many of those parameters are trainable, TRACER can in principle wrap
a LoRA-finetuned or prompt-tuned model directly: the WMA teacher would
then average a low-dimensional trajectory in adapter/prompt space rather
than full-encoder space. We therefore see TRACER as *complementary* to
PEFT and prompt-based adaptation rather than competing with them; a
systematic empirical study of TRACER on top of LoRA, adapters, and
prompt tuning is a natural follow-up.

### F.3 Future work: NTK and random-feature extensions of the theory

Our theoretical analysis
(§[3](#S3 "3 Theoretical Analysis ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"),
§[C](#A3 "Appendix C Additional Theoretical Details ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning"))
is developed in the linearized image/text-encoder setting that is
standard in the contrastive-learning theory literature ([Ji et al.,
2023](#bib.bib11); [Tian, 2022](#bib.bib13); [Nakada et al.,
2023](#bib.bib14); [Xue et al., 2024](#bib.bib15)). In that setting, the
closed-form solutions, the contrastive target matrix, and the bias-free
convergence of the SD–WMA teacher all admit clean derivations that we
believe capture the essential mechanism underlying TRACER’s empirical
robustness on nonlinear CLIP backbones
(Figures [3](#S4.F3 "Figure 3 ‣ Multi-Modal Contrastive Loss (ℒ_"MMCL"): ‣ 4 Methodology: TRACER ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")–[4](#S5.F4 "Figure 4 ‣ Static SD vs. TRACER. ‣ 5.2.2 Results and Analysis ‣ 5.2 Main ImageNet Results and Ablations ‣ 5 Experiments ‣ TRACER: Persistent Regularization for Robust Multimodal Finetuning")).
A natural next step is to lift these results to the *neural tangent
kernel* (NTK) regime ([Jacot et al., 2018](#bib.bib109)) and to
*random-feature* (RF) models, where the nonlinear encoders are
approximated by linear maps in a feature space induced by their
gradients or by random nonlinearities. In the NTK regime, the same
matrix least-squares reformulation should apply with $`\mathbf{X}_{I}`$
replaced by NTK features, allowing both the orthogonal/parallel
decomposition and the WMA bias-free convergence theorem to be re-derived
for finite-width networks under lazy training. The RF view, in turn,
would let us study how the spectrum of the random feature map interacts
with the WMA kernel $`\kappa(\tau)`$ and the regularization strength
$`\lambda`$. Establishing such extensions would tighten the connection
between our linearized analysis and large nonlinear vision–language
models, and we leave them as a concrete open direction.

## Appendix G AI Usage

Large Language Models were used to improve the manuscript’s grammar and
readability, and to assist with code formatting and refactoring during
implementation. All research design, theoretical analysis, experimental
protocols, and interpretation of results were conducted entirely by the
authors.
````
