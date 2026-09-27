---
identifier: arxiv:2601.08139v2
title: Subspace Alignment for Vision-Language Model Test-time Adaptation
authors:
  - Zhichen Zeng
  - Wenxuan Bao
  - Xiao Lin
  - Ruizhong Qiu
  - Tianxin Wei
  - Xuying Ning
  - Yuchen Yan
  - Chen Luo
  - Monica Xiao Cheng
  - Jingrui He
  - Hanghang Tong
published: "2026-01-13T02:02:41+00:00"
url: https://arxiv.org/abs/2601.08139v2
source: arxiv
doi: null
arxiv_id: 2601.08139v2
categories:
  - cs.AI
  - cs.CV
---

# Subspace Alignment for Vision-Language Model Test-time Adaptation

Zhichen Zeng Affiliation: University of Illinois Urbana-Champaign   
Wenxuan Bao Affiliation: University of Illinois Urbana-Champaign    Xiao
Lin Affiliation: University of Illinois Urbana-Champaign    Ruizhong Qiu
Affiliation: University of Illinois Urbana-Champaign    Tianxin Wei
Affiliation: University of Illinois Urbana-Champaign    Xuying Ning
Affiliation: University of Illinois Urbana-Champaign    Yuchen Yan
Affiliation: Amazon
Correspondence:[htong@illinois.edu](mailto:htong@illinois.edu)    Chen
Luo Affiliation: Amazon
Correspondence:[htong@illinois.edu](mailto:htong@illinois.edu)    Monica
Xiao Cheng Affiliation: Amazon
Correspondence:[htong@illinois.edu](mailto:htong@illinois.edu)   
Jingrui He Affiliation: University of Illinois Urbana-Champaign   
Hanghang Tong Affiliation: University of Illinois Urbana-Champaign

###### Abstract

Vision-language models (VLMs), despite their extraordinary zero-shot
capabilities, are vulnerable to distribution shifts. Test-time
adaptation (TTA) emerges as a predominant strategy to adapt VLMs to
unlabeled test data on the fly. However, existing TTA methods heavily
rely on zero-shot predictions as pseudo-labels for self-training, which
can be unreliable under distribution shifts and misguide adaptation due
to two fundamental limitations. First (Modality Gap), distribution
shifts induce gaps between visual and textual modalities, making
cross-modal relations inaccurate. Second (Visual Nuisance), visual
embeddings encode rich but task-irrelevant noise that often overwhelms
task-specific semantics under distribution shifts. To address these
limitations, we propose SubTTA, which aligns the semantic subspaces of
both modalities to enhance zero-shot predictions to better guide the TTA
process. To bridge the modality gap, SubTTA extracts the principal
subspaces of both modalities and aligns the visual manifold to the
textual semantic anchor by minimizing their chordal distance. To
eliminate visual nuisance, SubTTA projects the aligned visual features
onto the task-specific textual subspace, which filters out
task-irrelevant noise by constraining visual embeddings within the valid
semantic span, and standard TTA is further performed on the purified
space to refine the decision boundaries. Extensive experiments on
various benchmarks and VLM architectures demonstrate the effectiveness
of SubTTA, yielding an average improvement of 2.24% over
state-of-the-art TTA methods. Our code is available at
[https://github.com/zhichenz98/SubTTA_EMNLP26](https://github.com/zhichenz98/SubTTA_EMNLP26).

## 1 Introduction

Pretrained Vision-Language Models (VLMs), such as CLIP [Radford et al.
(2021)](#bib.bib19) and ALIGN [Jia et al. (2021)](#bib.bib20), have
demonstrated extraordinary zero-shot capabilities across diverse
downstream tasks, ranging from image classification [Radford et al.
(2021)](#bib.bib19); [Addepalli et al. (2024)](#bib.bib21) to image
captioning [Chen et al. (2022)](#bib.bib23); [Yu et al.
(2022)](#bib.bib22) and visual question-answering [Yu et al.
(2023)](#bib.bib24); [Huynh et al. (2025)](#bib.bib25); [Lin et al.
(2025b)](#bib.bib102). The success stems from the expressive joint
embedding space, where visual representations are globally aligned with
rich linguistic concepts, enabling task descriptions in natural language
to directly retrieve task-relevant visual semantics.

Despite these capabilities, VLMs often struggle when deployed in
open-world scenarios which are characterized by distribution shifts,
such as image corruptions [Hendrycks and Dietterich (2019)](#bib.bib27)
or stylistic changes [Patashnik et al. (2021)](#bib.bib26). These shifts
distort the vision-language embedding space and degrade the reliability
of zero-shot predictions. To mitigate this, test-time adaptation (TTA)
has emerged as a predominant paradigm to adapt pre-trained VLMs to
unlabeled test data on the fly [Osowiechi et al. (2024)](#bib.bib5);
[Maharana et al. (2025)](#bib.bib2); [Bao et al. (2025)](#bib.bib3).
Notably, most existing TTA approaches, either training-free methods
utilizing memory banks [Zhang et al. (2024)](#bib.bib7); [Li et al.
(2025)](#bib.bib8) or training-based methods optimizing pseudo
labels [Zhang et al. (2022)](#bib.bib9); [Shu et al.
(2022)](#bib.bib11), heavily rely on the VLMs’ raw zero-shot predictions
to guide the adaptation process, which can be precarious when the
aligned space is disrupted. We attribute the failure of standard TTA to
two fundamental limitations under distribution shifts, namely modality
gap and visual nuisance.

![Refer to caption](2601.08139v2/figures/teaser_gap.png)

(a) Modality Gap

![Refer to caption](2601.08139v2/figures/teaser_nuisance.png)

(b) Visual Nuisance

Figure 1: Failure modes of zero-shot prediction. (a) Modality gap:
Visual features drift away from the textual manifold. A dog image shifts
closer to the bird anchor. (b) Visual Nuisance: Task-irrelevant noise
overshadows core semantics. a dog is misclassified as bird due to
spurious correlation with the blue sky.

First (Modality Gap), distribution shifts induce a global drift of the
visual manifold relative to the textual manifold. Intuitively, as shown
in
Figure [1(a)](#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
the modality gap causes visual features (e.g., dog) to drift toward
incorrect textual anchors (e.g., bird), leading to incorrect zero-shot
predictions. To validate this empirically, we analyze the visual-textual
principal angles under different shift levels in
Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
We observe that correct predictions consistently exhibit smaller
principal angles than mispredictions, and that, as the shift level
increases to the right side, the entire distribution shifts toward
larger angles accompanied by a surge in errors. This geometric
divergence indicates that the pre-trained vision-language alignment is
structurally broken, creating a modality gap where the visual feature
space is globally rotated away from the textual anchor space.

Second (Visual Nuisance), unlike compact textual anchors, visual
embeddings encode rich but task-irrelevant information. As illustrated
in
Figure [1(b)](#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
irrelevant nuisances often overshadow core semantics: for example, a dog
on a blue background may be misclassified as a bird due to the spurious
correlation between the sky color and the bird class. We quantify this
phenomenon via semantic concentration, measured by the ratio of visual
energy projected onto the textual subspace relative to the raw
embedding, in
Figure [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
Correct samples exhibit markedly higher semantic concentration than
mispredictions, while increasing shift levels push the distributions
toward lower concentration, revealing that task-relevant components are
increasingly overshadowed by nuisance dimensions. Consequently,
pseudo-labels derived from raw visual embeddings are heavily
contaminated by irrelevant noise (e.g., background clutter or
domain-specific styles), causing TTA to reinforce prediction errors
rather than correct them.

To address these challenges, we propose SubTTA, a novel TTA framework
grounded in subspace alignment. First (Geometric Alignment), to bridge
the modality gap, we rectify the global drift of the visual manifold. We
construct compact principal subspaces for both modalities via
eigendecomposition. Treating the textual basis as an anchor, we
geometrically align the visual subspace to it by minimizing the chordal
distance. This step recalibrates the visual feature space to match the
pre-trained vision-language geometry. Second (Semantic Projection), to
eliminate visual nuisance, we project the aligned visual embeddings onto
the task-specific textual subspace. The projection acts as a semantic
filter which constrains visual features to lie within the semantic span
defined by the textual basis, effectively discarding irrelevant noise
and recovering the submerged semantic signal. Finally, standard
self-training objectives are applied within this purified space to
sharpen decision boundaries.

![Refer to caption](2601.08139v2/figures/principal_angle_cifar10_c.png)

Figure 2: Principal angles ($`\downarrow`$). Correct predictions exhibit
consistently smaller principal angles than mispredictions. Increased
shift level results in larger angles and more mispredictions.

![Refer to caption](2601.08139v2/figures/snr_cifar10_c.png)

Figure 3: Semantic concentration ($`\uparrow`$). Correct predictions
exhibit markedly higher semantic concentration than mispredictions.
Increased shift level results in lower concentration and more
mispredictions.

Our contributions are summarized as follows:

- •
  Analysis. We provide a novel perspective on VLM TTA failure,
  identifying two barriers, namely modality gap and visual nuisance.
- •
  Method. We propose SubTTA, a subspace-centric TTA framework to align
  visual-textual subspaces and filter visual nuisance via semantic
  projection, ensuring pseudo label quality.
- •
  Evaluation. Extensive experiments on diverse benchmarks and VLM
  architectures demonstrate that SubTTA significantly outperforms
  state-of-the-art TTA methods.

## 2 Related Works

##### Vision-Language Model Test-time Adaptation.

Existing VLM TTA methods are broadly categorized into training-based and
training-free approaches. _Training-based_ methods update parameters or
prompts via self-supervised objectives, such as entropy
minimization [Wang et al. (2020)](#bib.bib1); [Zhang et al.
(2022)](#bib.bib9), robust loss functions [Rusak et al.
(2022)](#bib.bib12); [Yuan et al. (2023)](#bib.bib13), or context prompt
tuning [Shu et al. (2022)](#bib.bib11). Recent variants further explore
cluster-based refinement [Maharana et al. (2025)](#bib.bib2); [Bao et
al. (2025)](#bib.bib3), model merging [Osowiechi et al.
(2024)](#bib.bib5), and negative augmentation [Deng et al.
(2025)](#bib.bib4). Conversely, _training-free_ methods avoid gradient
updates for better efficiency, calibrating distributions via
logit/prototype adjustments [Zhang et al. (2024)](#bib.bib7); [Li et al.
(2025)](#bib.bib8); [Karmanov et al. (2024)](#bib.bib17); [Huang et al.
(2026)](#bib.bib39) or closed-form feature alignment [Farina et al.
(2024)](#bib.bib10); [Döbler et al. (2024)](#bib.bib6). Despite their
effectiveness, both paradigms typically rely on raw zero-shot
predictions as pseudo-labels. These can be highly noisy under
distribution shifts, risking catastrophic failures. SubTTA addresses
this by intrinsically refining zero-shot predictions to enable reliable
adaptation.

##### Geometric Adaptation.

Exploiting feature space geometry is a growing trend for bridging
modality gaps. However, prior subspace methods typically focus on
few-shot [Zhu et al. (2024)](#bib.bib14); [Simon et al.
(2020)](#bib.bib36); [Wang et al. (2025)](#bib.bib37) or
uni-modal [Adachi et al. (2025)](#bib.bib16); [Fernando et al.
(2014)](#bib.bib35) settings, often utilizing rigid rank-to-rank
metrics [Fernando et al. (2014)](#bib.bib35); [Sun and Saenko
(2016)](#bib.bib38). For cross-modal VLM adaptation, STS [Dafnis and
Metaxas (2026)](#bib.bib15) aligns textual embeddings to visual
features, which inadvertently risks accommodating task-irrelevant visual
nuisances. In contrast, SubTTA differentiates itself in three key
aspects. First (Task), SubTTA targets the challenging zero-shot
cross-modal classification; Second (Metric), by optimizing the
basis-invariant chordal distance rather than rigid rank-to-rank metrics,
we avoid false correspondences caused by nuisance factors; Third
(Approach), we explicitly filter visual nuisances via semantic
projection, offering a plug-and-play module that integrates seamlessly
with existing TTA frameworks.

## 3 Methodology

In this section, we introduce our proposed SubTTA to improve TTA
performance via subspace alignment. We first introduce preliminaries in
Section [3.1](#S3.SS1 "3.1 Preliminaries ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
Afterwards, we introduce two key components, namely geometric alignment
and semantic projection, in
Sections [3.2](#S3.SS2 "3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
and
[3.3](#S3.SS3 "3.3 Semantic Projection ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
respectively.

![Refer to caption](2601.08139v2/figures/overview.png)

Figure 4: Overview of SubTTA. (a) VLM encodes images and textual prompts
into a shared raw space. (b) Geometric alignment alleviates modality gap
by minimizing the chordal distance. (c) Semantic projection retains
task-relevant information, e.g., category, while filtering out
irrelevant ones, e.g., background color. (d) Standard TTA is further
performed on the aligned subspace.

### 3.1 Preliminaries

We denote text space by $`\mathcal{T}`$ and image space by
$`\mathcal{V}`$. A VLM consisting of an image encoder
$`f_{v}:\mathcal{V}\to\mathbb{R}^{d}`$ and a text encoder
$`f_{t}:\mathcal{T}\to\mathbb{R}^{d}`$, which aligns image and text
embeddings in a shared space. For an image classification task with
$`C`$ classes, the text encoder $`f_{t}`$ embeds the textual
descriptions, e.g., "A photo of a \<class\>", into text embeddings
$`\mathbf{t}_{1},...,\mathbf{t}_{C}\in\mathbb{R}^{d}`$. Given a test
image, the image encoder $`f_{v}`$ embeds it into an image embedding
$`\mathbf{v}\in\mathbb{R}^{d}`$, and the prediction corresponds to the
class with the highest similarity score, i.e.,
$`\mathop{\arg\max}_{c}\mathbf{v}^{\top}\mathbf{t}_{c}`$.

### 3.2 Geometric Alignment

High-dimensional VLM embeddings are susceptible to the modality gap
under distribution shifts, where structural misalignment leads to
unreliable zero-shot predictions. To mitigate this, we propose to align
the visual and textual representations within a refined low-dimensional
subspace.

We begin by identifying the principal directions that capture textual
and visual semantics via eigendecomposition on feature covariance
matrices. For the textual modality, textual prompts encapsulate rich
task-specific semantic information (e.g., class categories), hence their
covariance matrix $`\mathbf{\Sigma}_{\mathcal{T}}`$ effectively defines
the target semantic space. Formally, given the normalized text
embeddings
$`\mathbf{T}=[\mathbf{t}_{1},\dots,\mathbf{t}_{C}]^{\top}\in\mathbb{R}^{C\times d}`$,
the text covariance matrix is computed as
$`\mathbf{\Sigma}_{\mathcal{T}}=\mathbf{T}^{\top}\mathbf{T}\in\mathbb{R}^{d\times d}`$.
For the visual modality, test images typically arrive in small batches,
which provide insufficient statistics to accurately estimate the global
visual distribution, leading to high estimation variance and noise [Bao
et al. (2025)](#bib.bib3). To mitigate such instability, we employ an
Exponential Moving Average (EMA) to maintain a robust estimate of the
visual covariance. Specifically, we initialize the visual covariance
using the textual covariance as a semantic prior, i.e.,
$`\mathbf{\Sigma}_{\mathcal{V}}\leftarrow\mathbf{\Sigma}_{\mathcal{T}}`$.
For each step $`k`$, with the normalized image batch
$`\mathbf{V}^{(k)}=[\mathbf{v}_{1},\dots,\mathbf{v}_{B}]^{\top}\in\mathbb{R}^{B\times d}`$,
the visual covariance matrix is gradually updated by

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
\mathbf{\Sigma}_{\mathcal{V}}\leftarrow(1-\alpha)\mathbf{\Sigma}_{\mathcal{V}}+\alpha(\mathbf{V}^{(k)^{\top}}\mathbf{V}^{(k)}),
``` |  | (1) |

where $`\alpha`$ is the momentum coefficient.

To extract the dominant components from the noisy covariance matrices,
we perform eigendecomposition to obtain the rank-$`r`$ approximations

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathbf{\Sigma}_{\mathcal{T}}\approx\mathbf{B}_{\mathcal{T}}^{\top}\mathbf{\Lambda}_{\mathcal{T}}\mathbf{B}_{\mathcal{T}},\mathbf{\Sigma}_{\mathcal{V}}\approx\mathbf{B}^{\top}_{\mathcal{V}}\mathbf{\Lambda}_{\mathcal{V}}\mathbf{B}_{\mathcal{V}},
``` |  | (2) |

where
$`\mathbf{B}_{\mathcal{T}},\mathbf{B}_{\mathcal{V}}\in\mathbb{R}^{r\times d}`$
are the top-$`r`$ orthonormal eigenvectors corresponding to the
principal basis, and
$`\mathbf{\Lambda}_{\mathcal{T}},\mathbf{\Lambda}_{\mathcal{V}}`$ are
diagonal matrices containing the corresponding top-$`r`$ eigenvalues.

After obtaining the principal bases, we follow prior works [Osowiechi et
al. (2024)](#bib.bib5); [Hakim et al. (2025)](#bib.bib18); [Bao et al.
(2025)](#bib.bib3) and adapt the normalization layers of the image
encoder to align the visual span $`\mathbf{B}_{\mathcal{V}}`$ to the
textual anchor $`\mathbf{B}_{\mathcal{T}}`$. Conventional metrics, such
as Frobenius norm
$`\|\mathbf{B}_{\mathcal{T}}-\mathbf{B}_{\mathcal{V}}\|_{F}`$ and cosine
similarity
$`\sum_{i=1}^{r}\mathbf{B}_{\mathcal{T}}(i)^{\top}\mathbf{B}_{\mathcal{V}}(i)`$,
enforce a *rigid rank-to-rank alignment*, i.e., forcing
$`\mathbf{B}_{\mathcal{T}}(i)`$ to align with
$`\mathbf{B}_{\mathcal{V}}(i)`$. However, this can be problematic
because the bases in $`\mathbf{B}_{\mathcal{T}}`$ and
$`\mathbf{B}_{\mathcal{V}}`$ are sorted by statistical variance rather
than semantic relevance. For instance, the top-ranked visual basis may
capture dominant but task-irrelevant signals (e.g., background intensity
or style) rather than target objects. Consequently, enforcing a rigid
rank-to-rank correspondence may erroneously align these nuisances with
the primary textual anchors. Therefore, the distance metric is expected
to be solely defined by the subspace geometry while remaining invariant
to the specific choice of basis vectors.

To satisfy this property, we employ the chordal distance on the
Grassmannian manifold as follows

|  |  |  |  |
|----|----|----|----|
|  |
``` math
d_{\text{ch}}(\mathbf{B}_{\mathcal{T}},\mathbf{B}_{\mathcal{V}})=\sqrt{\sum_{i=1}^{r}\sin^{2}\theta_{i}},
``` |  | (3) |

where $`\theta_{i}`$ represents the $`i`$-th principal angle between
subspaces $`\operatorname{span}(\mathbf{B}_{\mathcal{T}})`$ and
$`\operatorname{span}(\mathbf{B}_{\mathcal{V}})`$. Geometrically,
minimizing the chordal distance forces the principal angles toward zero,
thereby maximizing the overlap between the visual and textual subspaces.
Based on this, we formulate our alignment objective as the squared
chordal distance, which can be efficiently computed via the Frobenius
norm as follows [Conway et al. (1996)](#bib.bib30)

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{align}}=d_{\text{ch}}^{2}(\mathbf{B}_{\mathcal{T}},\mathbf{B}_{\mathcal{V}})=r-\|\mathbf{B}_{\mathcal{T}}\mathbf{B}_{\mathcal{V}}^{\top}\|_{F}^{2}.
``` |  | (4) |

Note that the chordal distance is defined solely by the principal angles
$`\{\theta_{i}\}_{i=1}^{r}`$ between the two subspaces, remaining
invariant to the basis choices. Formally, consider an orthogonal
rotation matrix $`\mathbf{Q}\in\mathbb{R}^{r\times r}`$ with
$`\mathbf{Q}^{\top}\mathbf{Q}=\mathbf{I}`$ that rotates the visual bases
by $`\mathbf{Q}\mathbf{B}_{\mathcal{V}}`$, we have

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\mathbf{B}_{\mathcal{T}}(\mathbf{Q}\mathbf{B}_{\mathcal{V}})^{\top}\|_{F}^{2}`$ | $`\displaystyle=\Tr\left(\mathbf{B}_{\mathcal{T}}\mathbf{B}_{\mathcal{V}}^{\top}\mathbf{Q}^{\top}\mathbf{Q}\mathbf{B}_{\mathcal{V}}\mathbf{B}_{\mathcal{T}}^{\top}\right)`$ |  |
|  |  | $`\displaystyle=\Tr\left(\mathbf{B}_{\mathcal{T}}\mathbf{B}_{\mathcal{V}}^{\top}\mathbf{B}_{\mathcal{V}}\mathbf{B}_{\mathcal{T}}^{\top}\right)`$ |  |
|  |  | $`\displaystyle=\|\mathbf{B}_{\mathcal{T}}\mathbf{B}_{\mathcal{V}}^{\top}\|_{F}^{2}.`$ |  |

This geometric flexibility avoids rigid rank-to-rank alignment, and
enables a global matching that allows relevant visual signals to align
with the correct textual anchors regardless of their rank order.

### 3.3 Semantic Projection

While geometric alignment effectively rectifies the global distribution
drift, it preserves the intrinsic structure of the visual features that
include the task-irrelevant nuisance. To eliminate such nuisance, we
leverage the textual subspace, which is compact and rich in
task-specific semantics, as a hard semantic filter. Specifically, for
image embedding $`\mathbf{v}`$, the semantic projection is defined as
follows

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\text{Proj}(\mathbf{v})=\mathbf{v}\mathbf{B}_{\mathcal{T}}^{\top}\mathbf{B}_{\mathcal{T}}.
``` |  | (5) |

This operation effectively discards noises orthogonal to the semantic
span, yielding a purified embedding space that facilitates high-quality
zero-shot predictions. Equipped with the purified embeddings, SubTTA can
be seamlessly integrated with various TTA objectives (e.g.,
entropy-based or cluster-based) to refine decision boundaries.

| Method |  | Noise |  |  | Blur |  |  |  | Weather |  |  |  | Digital |  |  |  | Mean |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  |  | Gauss. | Shot | Impul. | Defoc. | Glass | Motion | Zoom | Snow | Frost | Fog | Brit. | Contr. | Elast. | Pixel. | JPEG |  |
|   ImageNet-C | Source | 11.14 | 12.54 | 12.06 | 23.34 | 15.26 | 24.44 | 22.72 | 32.36 | 29.86 | 35.94 | 54.08 | 17.26 | 12.70 | 31.00 | 33.34 | 24.54 |
|  | TDA | 12.42 | 14.80 | 14.80 | 23.70 | 16.70 | 26.06 | 23.98 | 33.54 | 32.10 | 38.00 | 55.28 | 19.32 | 14.18 | 33.46 | 34.98 | 26.22 |
|  | DMN | 11.50 | 14.14 | 14.00 | 22.32 | 15.78 | 24.56 | 22.54 | 31.52 | 30.36 | 35.34 | 54.24 | 16.30 | 12.48 | 32.52 | 24.79 | 24.16 |
|  | VTE | 9.18 | 10.80 | 10.80 | 24.70 | 14.36 | 24.28 | 25.20 | 35.40 | 32.44 | 38.20 | 55.52 | 16.20 | 14.34 | 38.68 | 34.04 | 25.61 |
|  | ZERO | 10.50 | 11.50 | 11.12 | 24.78 | 14.88 | 24.36 | 23.28 | 33.04 | 30.40 | 37.06 | 54.52 | 16.90 | 13.40 | 34.90 | 32.90 | 24.90 |
|  | ECALP | 13.34 | 15.58 | 14.10 | 22.56 | 15.56 | 25.52 | 23.64 | 30.96 | 30.02 | 35.78 | 52.08 | 18.34 | 13.52 | 33.14 | 33.30 | 25.16 |
|  | TENT | 5.20 | 5.68 | 7.42 | 25.30 | 19.34 | 26.86 | 24.16 | 33.52 | 30.54 | 37.80 | 54.32 | 22.52 | 13.90 | 35.08 | 36.02 | 25.18 |
|  | RoTTA | 11.44 | 13.08 | 12.36 | 23.46 | 15.48 | 24.66 | 22.78 | 32.60 | 29.92 | 35.86 | 54.18 | 17.34 | 12.92 | 31.02 | 33.52 | 24.71 |
|  | TPT | 8.54 | 9.44 | 10.16 | 24.06 | 15.06 | 25.00 | 24.04 | 33.96 | 32.18 | 37.08 | 55.52 | 16.52 | 13.68 | 33.90 | 33.56 | 24.85 |
|  | MEMO | 11.08 | 12.38 | 12.02 | 23.54 | 15.44 | 24.60 | 22.90 | 32.54 | 30.06 | 35.94 | 54.02 | 17.46 | 12.80 | 31.24 | 33.40 | 24.63 |
|  | WATT | 11.18 | 12.78 | 13.02 | 25.12 | 18.08 | 26.90 | 25.22 | 33.42 | 30.22 | 37.40 | 53.96 | 21.84 | 15.16 | 33.46 | 34.94 | 26.18 |
|  | STS | 15.06 | 16.84 | 15.82 | 24.40 | 16.32 | 26.90 | 24.36 | 34.06 | 31.06 | 36.98 | 54.68 | 26.16 | 14.06 | 37.22 | 32.04 | 27.06 |
|  | MINT | 19.68 | 20.36 | 19.44 | 26.88 | 21.52 | 29.88 | 25.96 | 32.68 | 29.38 | 38.84 | 55.10 | 24.30 | 19.06 | 36.38 | 38.24 | 29.18 |
|  | BATCLIP | 19.48 | 21.02 | 19.30 | 25.92 | 21.26 | 29.96 | 28.52 | 35.22 | 31.38 | 40.24 | 55.20 | 25.68 | 23.72 | 36.82 | 37.22 | 30.06 |
|  | SubTTA | 21.16 | 22.88 | 21.44 | 25.34 | 24.82 | 30.70 | 29.64 | 37.24 | 35.90 | 41.26 | 56.96 | 29.74 | 29.40 | 37.58 | 38.18 | 32.15 |
|   CIFAR10-C | Source | 37.94 | 41.68 | 54.44 | 71.73 | 40.87 | 67.90 | 73.67 | 73.88 | 77.34 | 70.25 | 84.41 | 62.41 | 53.81 | 47.66 | 59.45 | 61.16 |
|  | TDA | 40.91 | 44.31 | 50.45 | 72.80 | 44.38 | 71.60 | 75.97 | 75.29 | 77.84 | 71.75 | 86.07 | 62.34 | 56.34 | 47.77 | 57.68 | 62.37 |
|  | DMN | 43.34 | 47.52 | 54.11 | 73.43 | 39.99 | 72.25 | 75.97 | 74.31 | 76.46 | 71.24 | 84.54 | 60.56 | 54.15 | 47.20 | 57.14 | 62.15 |
|  | VTE | 42.34 | 46.22 | 64.25 | 71.14 | 45.63 | 68.51 | 73.66 | 76.74 | 78.27 | 71.06 | 85.27 | 57.24 | 59.62 | 60.59 | 61.89 | 64.16 |
|  | ZERO | 39.22 | 43.77 | 57.19 | 71.91 | 40.71 | 69.22 | 74.11 | 74.36 | 77.24 | 72.44 | 83.85 | 60.74 | 55.64 | 48.97 | 62.29 | 62.11 |
|  | ECALP | 46.52 | 50.81 | 61.04 | 71.83 | 41.52 | 70.48 | 75.38 | 75.82 | 78.99 | 71.24 | 86.03 | 60.70 | 58.06 | 49.78 | 61.17 | 63.96 |
|  | TENT | 15.42 | 18.30 | 38.18 | 81.43 | 21.54 | 76.33 | 82.24 | 83.59 | 82.24 | 80.56 | 89.76 | 80.65 | 63.50 | 58.83 | 56.27 | 61.92 |
|  | RoTTA | 39.20 | 42.64 | 55.33 | 72.10 | 41.20 | 68.11 | 74.02 | 74.39 | 78.02 | 70.78 | 84.77 | 63.09 | 54.61 | 49.44 | 60.12 | 61.85 |
|  | TPT | 37.76 | 42.19 | 60.66 | 72.84 | 44.82 | 69.72 | 75.37 | 75.95 | 78.90 | 72.15 | 85.67 | 62.04 | 58.86 | 55.18 | 62.58 | 63.65 |
|  | MEMO | 37.61 | 41.26 | 55.64 | 72.08 | 41.58 | 68.60 | 73.97 | 74.96 | 77.34 | 71.50 | 84.79 | 62.07 | 55.50 | 49.23 | 61.02 | 61.81 |
|  | WATT | 45.98 | 53.25 | 60.31 | 74.99 | 38.58 | 71.49 | 75.92 | 77.69 | 80.23 | 76.33 | 87.56 | 75.59 | 55.42 | 62.04 | 63.21 | 66.57 |
|  | STS | 42.17 | 53.94 | 65.26 | 72.82 | 43.93 | 75.75 | 75.37 | 74.90 | 79.00 | 73.74 | 87.87 | 72.91 | 59.38 | 60.16 | 62.67 | 66.66 |
|  | MINT | 54.40 | 58.73 | 64.53 | 76.54 | 48.90 | 77.89 | 79.27 | 81.58 | 81.51 | 77.20 | 89.94 | 74.54 | 62.11 | 61.42 | 64.45 | 70.20 |
|  | BATCLIP | 61.89 | 65.44 | 67.07 | 80.06 | 55.47 | 80.02 | 81.60 | 82.47 | 83.69 | 80.43 | 88.47 | 81.19 | 69.12 | 62.67 | 67.29 | 73.79 |
|  | SubTTA | 60.18 | 66.62 | 69.73 | 81.47 | 59.94 | 81.24 | 82.39 | 83.72 | 83.75 | 83.48 | 91.42 | 86.26 | 70.94 | 74.42 | 69.41 | 76.33 |
|   CIFAR100-C | Source | 19.57 | 21.41 | 25.26 | 42.46 | 20.05 | 43.15 | 47.92 | 48.38 | 49.67 | 41.61 | 57.01 | 34.54 | 29.23 | 23.95 | 32.46 | 35.78 |
|  | TDA | 22.59 | 25.12 | 29.22 | 43.11 | 19.35 | 43.47 | 49.27 | 48.46 | 50.40 | 41.45 | 57.96 | 35.02 | 28.73 | 24.16 | 32.46 | 36.72 |
|  | DMN | 22.02 | 25.06 | 25.43 | 40.93 | 16.10 | 43.70 | 49.38 | 45.72 | 47.88 | 39.77 | 57.61 | 32.23 | 25.89 | 22.90 | 30.99 | 35.04 |
|  | VTE | 17.97 | 18.80 | 28.22 | 40.42 | 19.58 | 39.58 | 45.36 | 48.16 | 46.85 | 40.68 | 55.30 | 30.08 | 32.47 | 30.35 | 31.52 | 35.02 |
|  | ZERO | 18.91 | 21.07 | 28.57 | 43.97 | 19.49 | 43.31 | 48.69 | 49.11 | 50.07 | 44.02 | 57.62 | 34.51 | 31.08 | 24.80 | 34.04 | 36.62 |
|  | ECALP | 23.15 | 25.17 | 30.10 | 43.19 | 19.65 | 42.82 | 49.70 | 47.97 | 50.02 | 42.31 | 58.42 | 34.63 | 30.05 | 25.19 | 32.33 | 36.98 |
|  | TENT | 7.57 | 8.23 | 8.32 | 51.70 | 8.17 | 52.50 | 53.28 | 52.16 | 36.34 | 47.91 | 62.58 | 52.55 | 36.39 | 39.98 | 38.12 | 36.98 |
|  | RoTTA | 20.62 | 22.21 | 26.28 | 42.49 | 20.28 | 43.20 | 48.10 | 48.59 | 50.00 | 41.71 | 57.24 | 34.54 | 29.18 | 25.06 | 32.93 | 36.16 |
|  | TPT | 17.86 | 19.50 | 27.15 | 43.52 | 20.02 | 42.64 | 48.66 | 49.10 | 49.49 | 42.21 | 57.29 | 33.38 | 31.08 | 27.60 | 32.79 | 36.15 |
|  | MEMO | 20.20 | 21.89 | 27.16 | 44.09 | 19.58 | 43.92 | 49.14 | 50.58 | 50.44 | 43.90 | 58.55 | 34.57 | 30.45 | 25.22 | 34.20 | 36.93 |
|  | WATT | 25.55 | 27.34 | 31.46 | 48.54 | 23.27 | 48.44 | 53.08 | 52.85 | 52.28 | 48.08 | 62.52 | 45.11 | 35.53 | 36.74 | 37.84 | 41.91 |
|  | STS | 25.43 | 26.20 | 29.04 | 44.01 | 24.38 | 44.05 | 48.23 | 50.62 | 50.76 | 43.50 | 60.08 | 43.90 | 33.60 | 30.59 | 34.96 | 39.29 |
|  | MINT | 27.49 | 30.46 | 35.97 | 49.33 | 26.54 | 47.37 | 53.42 | 52.58 | 52.10 | 48.46 | 64.68 | 44.15 | 35.25 | 33.02 | 35.40 | 42.41 |
|  | BATCLIP | 25.52 | 28.35 | 34.60 | 49.66 | 26.46 | 48.81 | 54.77 | 51.90 | 51.60 | 48.30 | 62.77 | 45.68 | 34.82 | 33.01 | 37.25 | 42.23 |
|  | SubTTA | 31.85 | 34.25 | 36.31 | 49.48 | 29.66 | 48.86 | 53.86 | 53.62 | 54.14 | 48.19 | 63.20 | 46.47 | 36.52 | 40.20 | 40.78 | 44.49 |

Table 1: Benchmark results with ViT-B-16. We denote Top-1/2/3 by
Blue/Yellow/Red, respectively.

| Method | ImageNet-R |  |  | ImageNet-K |  |  | ImageNet-A |  |  |
|----|----|----|----|----|----|----|----|----|----|
|  | Vanilla | SubTTA | $`\Delta`$ | Vanilla | SubTTA | $`\Delta`$ | Vanilla | SubTTA | $`\Delta`$ |
| Source | 73.97 | 74.73 | +0.76 | 46.14 | 45.96 | -0.18 | 47.77 | 48.07 | +0.30 |
| TENT | 74.50 | 74.77 | +0.27 | 46.87 | 46.94 | +0.07 | 48.25 | 48.24 | -0.01 |
| MINT | 74.38 | 75.92 | +1.54 | 47.75 | 48.53 | +0.78 | 48.39 | 49.14 | +0.75 |
| BATCLIP | 74.42 | 75.65 | +1.23 | 47.07 | 47.53 | +0.46 | 48.35 | 49.08 | +0.73 |

Table 2: Results on ImageNet with ViT-B-16. Positive/ negative changes
are denoted by Blue/Red, respectively.

## 4 Experiments

### 4.1 Experiment Setup

##### Datasets and Metric.

We evaluate SubTTA on three corrupted image classification benchmarks:
CIFAR-10-C, CIFAR-100-C, and ImageNet-C [Hendrycks and Dietterich
(2019)](#bib.bib27), each comprising 15 types of visual corruptions. To
assess robustness against broader shifts, such as adversarial examples
and domain gaps, we additionally evaluate on ImageNet-A/R/K [Hendrycks
et al. (2021a)](#bib.bib33); [Hendrycks et al. (2021b)](#bib.bib32);
[Wang et al. (2019)](#bib.bib34). We adopt the classification accuracy
as the evaluation metric.

CLIP Models. We consider the following widely used CLIP [Radford et al.
(2021)](#bib.bib19) models, including ViT-B-16, ViT-B-32, and ViT-L-14.

##### Baselines.

We benchmark SubTTA against state-of-the-art TTA approaches.
Training-free methods include memory-based methods (TDA [Karmanov et al.
(2024)](#bib.bib17), DMN [Zhang et al. (2024)](#bib.bib7), ECALP [Li et
al. (2025)](#bib.bib8)) that leverage sample similarity to adjust
predictions, and augmentation-based methods (VTE [Döbler et al.
(2024)](#bib.bib6), ZERO [Farina et al. (2024)](#bib.bib10)) that
aggregate image embeddings from multiple augmentations. Training-based
methods include entropy-based methods (MEMO [Zhang et al.
(2022)](#bib.bib9), WATT [Osowiechi et al. (2024)](#bib.bib5),
RoTTA [Yuan et al. (2023)](#bib.bib13), TPT [Shu et al.
(2022)](#bib.bib11), STS [Dafnis and Metaxas (2026)](#bib.bib15)), and
cluster-based methods (MINT [Bao et al. (2025)](#bib.bib3),
BATCLIP [Maharana et al. (2025)](#bib.bib2)).

##### Pipeline

Following the established TTA protocol [Wang et al. (2020)](#bib.bib1),
we conduct experiments under the highest severity level (Level 5) to
simulate severe shifts. We adopt an online adaptation setting where the
model adapts to a continuous stream of unlabeled test data for each
corruption type independently, resetting the model state between
corruptions. For each batch, we perform one gradient update for the
alignment loss
(Eq. ([4](#S3.E4 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")))
and the TTA loss, respectively. All experiments are conducted on NVIDIA
A100 80GB GPUs.

### 4.2 Benchmark Results

We conduct experiments on ViT-B-16 with results in
Tables [1](#S3.T1 "Table 1 ‣ 3.3 Semantic Projection ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
and
[2](#S3.T2 "Table 2 ‣ 3.3 Semantic Projection ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
Additional results on ViT-B-32 and ViT-L-14 are reported in
Appendix [C](#A3 "Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

(1) SubTTA establishes a new state-of-the-art with consistent
robustness. SubTTA consistently delivers the strongest overall
performance across diverse benchmarks. In fine-grained settings, SubTTA
consistently ranks within the Top-3 and achieves the best performance in
the majority of scenarios On the large-scale ImageNet-C dataset, SubTTA
achieves a mean accuracy of 32.15%, surpassing the previous best method
BATCLIP (30.06%) by a significant margin of 2.09%. Such superiority is
equally pronounced on smaller-resolution benchmarks: SubTTA outperforms
the best competitor by 2.54% on CIFAR-10-C, and 2.08% on CIFAR-100-C.
Beyond synthetic corruptions, SubTTA consistently enhances base TTA
performance against natural distribution shifts and stylistic
variations, achieving an average improvement of 0.56%. This validates
that SubTTA constructs a geometrically rectified feature space that is
inherently more robust to varying forms of distribution shifts.

(2) SubTTA avoids negative adaptation and ensures stability. Existing
baseline methods often suffer from catastrophic failure under large
shifts, leading to negative adaptation where performance drops below the
source CLIP baseline. For instance, we observe that TENT fails to adapt
to noise corruptions due to unstable entropy minimization, while DMN
often struggles with digital corruptions. SubTTA exhibits remarkable
resilience, consistently outperforming the source CLIP model across
every corruption category. This stability confirms that our subspace
alignment effectively filters out the nuisance factors that typically
destabilize other adaptation algorithms.

### 4.3 Performance with Various TTA Objectives

Our proposed SubTTA is compatible with various TTA objectives. To
validate this, we integrate SubTTA with three categories of baseline
methods, including Source CLIP model, entropy minimization (TENT),
cluster-based methods (MINT, BATCLIP), and the results are shown in
Figure [5](#S4.F5 "Figure 5 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

![Refer to caption](2601.08139v2/figures/comp_all_models.png)

Figure 5: TTA performance w/ and w/o SubTTA.

![Refer to caption](2601.08139v2/figures/violin_ang.png)

(a) Principal angle.

![Refer to caption](2601.08139v2/figures/violin_snr.png)

(b) Semantic concentration.

![Refer to caption](2601.08139v2/figures/violin_sim.png)

(c) Visual-textual similarity.

Figure 6: Study on improving zero-shot predictions. Blue denotes source
CLIP w/o SubTTA and Orange denotes CLIP w/ SubTTA. (a) Principal angles
($`\downarrow`$): SubTTA mitigates modality gap; (b) Semantic
concentration ($`\uparrow`$): SubTTA alleviates visual nuisance; (c)
Visual-textual similarity ($`\uparrow`$): SubTTA improves pseudo-label
quality.

First, SubTTA acts as a universal performance booster, irrespective of
the adaptation objective. When integrated with diverse TTA strategies,
SubTTA consistently yields performance improvements. This indicates that
SubTTA effectively rectifies the underlying structure before standard
TTA is applied.

Second, we observe more pronounced performance gains on lower-capacity
models (e.g., ViT-B-16 and ViT-B-32) compared to stronger ones (e.g.,
ViT-L-14). We attribute this to the fact that weaker models are
inherently more susceptible to modality gap and distribution shifts.
SubTTA effectively compensates for these intrinsic representational
deficits, providing critical robustness where the pre-trained alignment
is most fragile.

![Refer to caption](2601.08139v2/figures/cifar10_c_pixelate_Source.png)

(a) Source

![Refer to caption](2601.08139v2/figures/cifar10_c_pixelate_TENT.png)

(b) TENT

![Refer to caption](2601.08139v2/figures/cifar10_c_pixelate_BATCLIP.png)

(c) BATCLIP

![Refer to caption](2601.08139v2/figures/cifar10_c_pixelate_MINT.png)

(d) MINT

![Refer to
caption](2601.08139v2/figures/cifar10_c_pixelate_SubTTA_S.png)

(e) SubTTA-S

![Refer to
caption](2601.08139v2/figures/cifar10_c_pixelate_SubTTA_T.png)

(f) SubTTA-T

![Refer to
caption](2601.08139v2/figures/cifar10_c_pixelate_SubTTA_B.png)

(g) SubTTA-B

![Refer to
caption](2601.08139v2/figures/cifar10_c_pixelate_SubTTA_M.png)

(h) SubTTA-M

Figure 7: CIFAR-10-C embedding visualization of different TTA methods
w/o (a-d) and w/ (e-h) SubTTA.

### 4.4 On Improving Zero-shot Prediction

In this section, we empirically validate the mechanisms through which
SubTTA enhances the zero-shot prediction capability of CLIP.
Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
visualizes the distributions of three key metrics, including principal
angle, semantic concentration, and visual-textual similarity. Comparing
the baseline CLIP model with our SubTTA-augmented version, we draw the
following observations.

(1) SubTTA rectifies modality gap. We first examine the modality gap by
analyzing the principal angles between the visual and textual subspaces.
As shown in
Figure [6(a)](#S4.F6.sf1 "In Figure 6 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
CLIP without SubTTA exhibits relatively large principal angles,
indicating that distribution shifts induce a significant misalignment
between the visual features and their corresponding textual anchors. In
contrast, applying SubTTA leads to a marked reduction in principal
angles across all datasets. Such reduction demonstrates that SubTTA
effectively rectifies the modality gap, pulling the drifted visual
manifold back into geometric alignment with the invariant textual
semantic space, thereby ensuring more reliable cross-modal matching.

(2) SubTTA alleviates visual nuisance. We then investigate visual
nuisance by measuring semantic concentration, the ratio of visual energy
preserved within the task-specific textual subspace. As shown in
Figure [6(b)](#S4.F6.sf2 "In Figure 6 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
CLIP without SubTTA exhibits a distribution skewed toward the lower end
of the semantic concentration spectrum. This indicates that in the raw
feature space, the task-relevant semantic signal is largely overwhelmed
by orthogonal components, which correspond to task-irrelevant nuisances
such as background clutter or domain-specific styles. Conversely, SubTTA
propels the entire distribution toward the high-concentration regime,
demonstrating that the adapted features are better aligned with the
textual semantic span.

(3) SubTTA improves pseudo-label quality. We also assess pseudo-label
quality by analyzing the cosine similarity between visual embeddings and
their corresponding ground-truth textual prompts. As shown in
Figure [6(c)](#S4.F6.sf3 "In Figure 6 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
CLIP without SubTTA suffers from low visual-textual similarity scores,
reflecting weak confidence in the correct semantic alignment. However,
SubTTA significantly shifts the similarities toward higher values. This
increase indicates that the rectified visual features possess much
higher semantic fidelity to their true labels. Consequently, the
zero-shot predictions derived from these enhanced similarities are not
only more accurate but also more confident, providing high-quality
pseudo-labels that are critical for preventing error accumulation during
test-time adaptation.

(4) SubTTA enhances embedding quality. In addition, we visualize the
learned representations of CIFAR-10-C with and without SubTTA using
t-SNE, and the results are shown in
Figure [7](#S4.F7 "Figure 7 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
As shown in the top row of
Figure [7](#S4.F7 "Figure 7 ‣ 4.3 Performance with Various TTA Objectives ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
the embeddings extracted by baselines are severely entangled, exhibiting
blurred decision boundaries and cluster overlap. This indicates a
failure to disentangle task-relevant semantics from shift-induced noise,
resulting in compromised discriminative power. In contrast, integrating
SubTTA (bottom row) induces a profound geometric rectification. First,
SubTTA significantly improves intra-class compactness, where dispersed
features are pulled tighter together. Second, SubTTA enhances
inter-class separability, effectively pushing apart overlapping
manifolds to form clear margins between categories. This visualization
validates our motivation: by projecting features onto the aligned
subspace, SubTTA effectively filters out task-irrelevant nuisances and
restores a discriminative semantic structure.

| Method     | ImageNet-C |       | CIFAR10-C |       | CIFAR100-C |       |
|------------|------------|-------|-----------|-------|------------|-------|
|            | Source     | TENT  | Source    | TENT  | Source     | TENT  |
| Vanilla    | 24.54      | 24.53 | 61.17     | 61.16 | 35.79      | 35.79 |
| SubTTA     | 24.68      | 24.74 | 61.49     | 61.95 | 36.24      | 36.45 |
| $`\Delta`$ | +0.14      | +0.21 | +0.32     | +0.79 | +0.45      | +0.66 |

Table 3: Performance under episodic TTA.

### 4.5 Studies

We conduct studies to evaluate key components, analyze hyperparameter
sensitivity, and assess the performance of SubTTA under episodic
setting.

#### 4.5.1 Generalization to Episodic TTA

Although primarily designed for online batched TTA [Maharana et al.
(2025)](#bib.bib2); [Bao et al. (2025)](#bib.bib3); [Karmanov et al.
(2024)](#bib.bib17); [Xiao et al. ()](#bib.bib31), SubTTA can be
generalized to the episodic TTA setting, where adaptation is performed
per single test sample without batch context. By maintaining a
lightweight memory bank [Zhang et al. (2024)](#bib.bib7); [Li et al.
(2025)](#bib.bib8) of recent test features, we estimate a local visual
subspace for geometric alignment without batch statistics, and results
are shown in
Table [3](#S4.T3 "Table 3 ‣ 4.4 On Improving Zero-shot Prediction ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

As observed, SubTTA safely enhances the model without suffering from the
catastrophic negative adaptation that frequently plagues standard
baseline methods (e.g., TENT) in the episodic setting. This demonstrates
the robustness and flexibility of our geometric alignment approach for
challenging single-sample adaptation scenarios.

#### 4.5.2 Visual-Textual Alignment Quality

We provide a quantitative analysis on how SubTTA improves the
visual-text alignment quality. Specifically, we adopt the principal
angle to measure the global geometric divergence between the entire
covariance structures, i.e., $`\text{span}(\mathbf{B}_{\mathcal{V}})`$
and $`\text{span}(\mathbf{B}_{\mathcal{T}})`$. A reduction in this
global metric proves that the entire visual manifold has been
fundamentally rotated back into alignment with the textual semantic
space. We exam the visual-textual principal angles with and without
SubTTA, and the result is shown in
Figure [8](#S4.F8 "Figure 8 ‣ 4.5.2 Visual-Textual Alignment Quality ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

![Refer to caption](2601.08139v2/figures/comp_principal_angle.png)

Figure 8: Visual-textual principle angles w/ (colored) and w/o (hatched)
SubTTA. SubTTA consistently reduces the principle angles with different
baselines, achieving better cross-modal alignment.

It is shown that SubTTA consistently reduces the visual-textual
principle angles across different baselines. Existing approaches, e.g.,
entropy-minimization [Wang et al. (2020)](#bib.bib1) and cluster-based
loss [Maharana et al. (2025)](#bib.bib2); [Bao et al.
(2025)](#bib.bib3), mainly optimize instance-level distributions by
pulling similar samples closer together, but they do not explicitly
align the global geometry of the two modalities. In contrast, SubTTA
addresses this limitation by minimizing the chordal distance, which
directly captures subspace-level geometric alignment between modalities,
and thus serves as an effective complement to existing approaches.

#### 4.5.3 Ablation Study

![Refer to caption](2601.08139v2/figures/ablation.png)

Figure 9: Ablation study on alignment and projection.

We investigate the contributions of geometric alignment and semantic
projection in
Figure [9](#S4.F9 "Figure 9 ‣ 4.5.3 Ablation Study ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
denoting their ablated versions as -align and -project, respectively. We
draw the following observations.

(1) Alignment is a prerequisite for projection. The significant drop in
-align indicates that projection is only valid when subspaces are
aligned; otherwise, features may be mapped to incorrect semantic bases.
Notably, TENT suffers from negative transfer w/o alignment, as naive
projection creates confidently-wrong predictions that mislead entropy
minimization.

(2) Projection is a semantic denoiser. The performance gap between
-project and SubTTA confirms that raw embeddings contain task-irrelevant
nuisances. Semantic projection effectively filters these dimensions,
constructing a purified space that prevents optimization on spurious
correlations.

(3) Synergy of components. SubTTA consistently outperforms both
variants, confirming neither is redundant. Alignment corrects global
modality drift while projection filters local visual nuisances; both are
indispensable for optimal adaptation.

#### 4.5.4 Hyperparameter Study

We investigate the sensitivity of SubTTA to three key hyperparameters:
the subspace rank $`r`$, the momentum coefficient $`\alpha`$ and batch
size. The results are shown in
Figure [10](#S4.F10 "Figure 10 ‣ 4.5.4 Hyperparameter Study ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

Impact of subspace rank $`r`$. When subspace rank $`r`$ increases, the
accuracy first increases then decreases. With small rank (e.g.,
$`r<64`$), the performance is suboptimal because the subspace is overly
compressed, causing the loss of critical semantic information.
Conversely, performance degradation is observed when $`r`$ approaches
the full feature dimension (i.e., $`d=512`$ for ViT-B-16). We attribute
this to the fact that a full-rank subspace preserves all trailing
principal components, hence task-irrelevant nuisances are not filtered
out, which negates the denoising benefits of our subspace approach.

Impact of momentum coefficient $`\alpha`$. Results show that SubTTA is
generally robust for $`\alpha\in(0.2,0.8)`$. However, performance drops
significantly at the extremes. When $`\alpha=0`$, the covariance
estimation relies entirely on the current mini-batch. This introduces
instability due to the high variance of statistics estimated from small
batches, failing to capture the global visual distribution. When
$`\alpha=1`$, the visual covariance is fixed to the initial textual
prior $`\mathbf{\Sigma}_{\mathcal{T}}`$ and never updates with visual
features. In this scenario, the subspace alignment mechanism is
effectively disabled, leading to a marked performance drop. This
confirms the necessity of our momentum-based update strategy.

Impact of batch size. As batch size increases, adaptation performance
initially improves then degrades at large scales (e.g., $`B\geq 256`$).
At small batch sizes (e.g., $`B=16`$), unreliable visual covariance
estimates cause subspace alignment to fluctuate erratically due to local
noise rather than true manifold geometry. While increasing the batch
size stabilizes this estimation, excessively large batches suffer from
an optimization deficit. Specifically, under a fixed-length test stream,
a larger batch size results in fewer total gradient updates, thereby
under-optimizing the adaptation objective. This phenomenon is
empirically validated in
Figure [10](#S4.F10 "Figure 10 ‣ 4.5.4 Hyperparameter Study ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
as batch-dependent objectives (e.g., MINT, BATCLIP) degrade noticeably
under fewer updates, whereas sample-wise objectives (e.g., TENT) remain
relatively stable.

![Refer to caption](2601.08139v2/figures/hyper_study.png)

Figure 10: Hyper-parameter study. Different curves represent SubTTA with
different TTA objectives.

## 5 Conclusion

In this work, we address the vulnerability of VLMs to distribution
shifts by investigating the prevalent yet precarious reliance on raw
zero-shot predictions. We identify that adaptation failures stem from
modality gap and visual nuisance within the shifted embedding space. To
overcome these barriers, we propose SubTTA, a novel framework that
shifts the TTA from noisy self-training to robust subspace geometry. By
explicitly aligning the visual subspace to the textual semantic anchor
via chordal distance and projecting features onto a purified
task-specific subspace, SubTTA effectively rectifies modality gap and
alleviates visual nuisance. Extensive experiments validate the
effectiveness of SubTTA in enhancing various TTA methods.

## Acknowledgement

This work is supported by NSF (2416070) and DARPA (HR0011263E079). The
content of the information in this document does not necessarily reflect
the position or the policy of the Government, and no official
endorsement should be inferred. The U.S. Government is authorized to
reproduce and distribute reprints for Government purposes
notwithstanding any copyright notation here on.

## Limitations

While SubTTA demonstrates robust performance, we acknowledge a few
limitations. First, our method relies on batch-level statistics to
estimate reliable subspaces. Consequently, it is not immediately
applicable to the strictly episodic setting, where model is adapted to a
single test sample, without a mechanism to accumulate historical samples
(e.g., a memory bank). Second, as an optimization-based approach, SubTTA
requires an open-source model with access to model gradients,
restricting its usage with closed-source proprietary APIs. Lastly, the
SVD operation introduces a slight computational overhead compared to
training-free baselines. Future work could explore efficient
approximations for eigendecomposition to further reduce latency, or
extend the subspace alignment principle to black-box adaptation
scenarios.

## References

- Achiam et al. (2023) J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I.
  Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S.
  Anadkat, et al. Gpt-4 technical report. arXiv preprint
  arXiv:2303.08774. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Adachi et al. (2025) K. Adachi, S. Yamaguchi, A. Kumagai, and T.
  Hamagami Test-time adaptation for regression by subspace alignment. In
  The Thirteenth International Conference on Learning Representations,
  Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Addepalli et al. (2024) S. Addepalli, A. R. Asokan, L. Sharma,
  and R. V. Babu Leveraging vision-language models for improving domain
  generalization in image classification. In Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition,
  pp. 23922–23932. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Ai et al. (2025) M. Ai, T. Wei, Y. Chen, Z. Zeng, R. Zhao, G.
  Varatkar, B. D. Rouhani, X. Tang, H. Tong, and J. He Resmoe:
  space-efficient compression of mixture of experts llms via residual
  restoration. arXiv preprint arXiv:2503.06881. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Alayrac et al. (2022) J. Alayrac, J. Donahue, P. Luc, A. Miech, I.
  Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al.
  Flamingo: a visual language model for few-shot learning. Advances in
  neural information processing systems 35, pp. 23716–23736. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Bao et al. (2025) W. Bao, R. Deng, and J. He Mint: a simple test-time
  adaptation of vision-language models against common corruptions. arXiv
  preprint arXiv:2510.22127. Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Textual Subspace as Semantic Anchors ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§3.2](#S3.SS2.p2.1 "3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§3.2](#S3.SS2.p4.1 "3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.1](#S4.SS5.SSS1.p1.1 "4.5.1 Generalization to Episodic TTA ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.2](#S4.SS5.SSS2.p2.1 "4.5.2 Visual-Textual Alignment Quality ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Bao et al. (2024) W. Bao, Z. Zeng, Z. Liu, H. Tong, and J. He Matcha:
  mitigating graph structure shifts with test-time adaptation. arXiv
  preprint arXiv:2410.06976. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Chen et al. (2026) L. Chen, Y. Bei, H. Xu, Y. Zhao, Y. Chen, and H.
  Tong TAG-dlm: diffusion language models for text-attributed graph
  learning. arXiv preprint arXiv:2606.31166. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Chen et al. (2020) L. Chen, Z. Gan, Y. Cheng, L. Li, L. Carin, and J.
  Liu Graph optimal transport for cross-domain alignment. In
  International Conference on Machine Learning, pp. 1542–1553. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Chen et al. (2022) X. Chen, X. Wang, S. Changpinyo, A. J.
  Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B.
  Mustafa, L. Beyer, et al. Pali: a jointly-scaled multilingual
  language-image model. arXiv preprint arXiv:2209.06794. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Conway et al. (1996) J. H. Conway, R. H. Hardin, and N. J. Sloane
  Packing lines, planes, etc.: packings in grassmannian spaces.
  Experimental mathematics 5 (2), pp. 139–159. Cited by:
  [§3.2](#S3.SS2.p5.2 "3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Cui et al. (2026) C. Cui, T. Wei, Z. Chen, R. Qiu, Z. Zeng, Z. Liu, X.
  Ning, D. Zhou, and J. He AdaFuse: adaptive ensemble decoding with
  test-time scaling for llms. External Links: 2601.06022,
  [Link](https://arxiv.org/abs/2601.06022) Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Dafnis and Metaxas (2026) K. Dafnis and D. Metaxas Test-time
  spectrum-aware latent steering for zero-shot generalization in
  vision-language models. Advances in Neural Information Processing
  Systems 38, pp. 151169–151194. Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Deng et al. (2009) J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L.
  Fei-Fei Imagenet: a large-scale hierarchical image database. In 2009
  IEEE conference on computer vision and pattern recognition,
  pp. 248–255. Cited by:
  [§G.1](#A7.SS1.p2.1 "G.1 Cite Creators Of Artifacts ‣ Appendix G Use Or Create Scientific Artifacts ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Deng et al. (2025) R. Deng, W. Bao, T. Wei, and J. He Panda: test-time
  adaptation with negative data augmentation. arXiv preprint
  arXiv:2511.10481. Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Textual Subspace as Semantic Anchors ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Döbler et al. (2024) M. Döbler, R. A. Marsden, T. Raichle, and B. Yang
  A lost opportunity for vision-language models: a comparative study of
  online test-time adaptation for vision-language models. In European
  Conference on Computer Vision, pp. 117–133. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Farina et al. (2024) M. Farina, G. Franchi, G. Iacca, M. Mancini,
  and E. Ricci Frustratingly easy test-time adaptation of
  vision-language models. Advances in Neural Information Processing
  Systems 37, pp. 129062–129093. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Fernando et al. (2013) B. Fernando, A. Habrard, M. Sebban, and T.
  Tuytelaars Unsupervised visual domain adaptation using subspace
  alignment. In Proceedings of the IEEE international conference on
  computer vision, pp. 2960–2967. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Fernando et al. (2014) B. Fernando, A. Habrard, M. Sebban, and T.
  Tuytelaars Subspace alignment for domain adaptation. arXiv preprint
  arXiv:1409.5241. Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Ganin and Lempitsky (2015) Y. Ganin and V. Lempitsky Unsupervised
  domain adaptation by backpropagation. In International conference on
  machine learning, pp. 1180–1189. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Gong et al. (2012) B. Gong, Y. Shi, F. Sha, and K. Grauman Geodesic
  flow kernel for unsupervised domain adaptation. In 2012 IEEE
  conference on computer vision and pattern recognition, pp. 2066–2073.
  Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Hakim et al. (2025) G. A. V. Hakim, D. Osowiechi, M. Noori, M.
  Cheraghalikhani, A. Bahri, M. Yazdanpanah, I. B. Ayed, and C.
  Desrosiers Clipartt: adaptation of clip to new domains at test time.
  In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision
  (WACV), pp. 7092–7101. Cited by:
  [§3.2](#S3.SS2.p4.1 "3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Hendrycks et al. (2021a) D. Hendrycks, S. Basart, N. Mu, S.
  Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo,
  et al. The many faces of robustness: a critical analysis of
  out-of-distribution generalization. In Proceedings of the IEEE/CVF
  international conference on computer vision, pp. 8340–8349. Cited by:
  [Appendix
  D](#A4.p3.1 "Appendix D Datasets ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Datasets and Metric. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Hendrycks and Dietterich (2019) D. Hendrycks and T. Dietterich
  Benchmarking neural network robustness to common corruptions and
  perturbations. arXiv preprint arXiv:1903.12261. Cited by: [Appendix
  D](#A4.p1.1 "Appendix D Datasets ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§G.1](#A7.SS1.p2.1 "G.1 Cite Creators Of Artifacts ‣ Appendix G Use Or Create Scientific Artifacts ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Datasets and Metric. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Hendrycks et al. (2021b) D. Hendrycks, K. Zhao, S. Basart, J.
  Steinhardt, and D. Song Natural adversarial examples. In Proceedings
  of the IEEE/CVF conference on computer vision and pattern recognition,
  pp. 15262–15271. Cited by: [Appendix
  D](#A4.p3.1 "Appendix D Datasets ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Datasets and Metric. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Huang et al. (2026) Z. Huang, Y. Zhang, W. Liu, F. Chao, and R. Ji
  Prototype-based test-time adaptation of vision-language models. arXiv
  preprint arXiv:2604.21360. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Huynh et al. (2025) N. D. Huynh, M. R. Bouadjenek, S. Aryal, I.
  Razzak, and H. Hacid Visual question answering: from early
  developments to recent advances–a survey. arXiv preprint
  arXiv:2501.03939. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Jia et al. (2021) C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H.
  Pham, Q. Le, Y. Sung, Z. Li, and T. Duerig Scaling up visual and
  vision-language representation learning with noisy text supervision.
  In International conference on machine learning, pp. 4904–4916. Cited
  by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Jing et al. (2026) B. Jing, S. Chen, L. Zheng, B. Liu, Z. Li, J.
  Zou, T. Wei, Z. Liu, Z. Zeng, R. Qiu, et al. Tsaqa: time series
  analysis question and answering benchmark. In Proceedings of the Fifth
  Workshop on Generation, Evaluation and Metrics (GEM), pp. 944–979.
  Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Kang et al. (2019) G. Kang, L. Jiang, Y. Yang, and A. G. Hauptmann
  Contrastive adaptation network for unsupervised domain adaptation. In
  Proceedings of the IEEE/CVF conference on computer vision and pattern
  recognition, pp. 4893–4902. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p2.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Karmanov et al. (2024) A. Karmanov, D. Guan, S. Lu, A. El Saddik,
  and E. Xing Efficient test-time adaptation of vision-language models.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, pp. 14162–14171. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.1](#S4.SS5.SSS1.p1.1 "4.5.1 Generalization to Episodic TTA ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Krizhevsky et al. (2009) A. Krizhevsky G. Hinton et al. Learning
  multiple layers of features from tiny images. Cited by:
  [§G.1](#A7.SS1.p2.1 "G.1 Cite Creators Of Artifacts ‣ Appendix G Use Or Create Scientific Artifacts ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Li et al. (2023) J. Li, D. Li, S. Savarese, and S. Hoi Blip-2:
  bootstrapping language-image pre-training with frozen image encoders
  and large language models. In International conference on machine
  learning, pp. 19730–19742. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Li et al. (2025) Y. Li, Y. Su, A. Goodge, K. Jia, and X. Xu Efficient
  and context-aware label propagation for zero-/few-shot training-free
  adaptation of vision-language model. In The Thirteenth International
  Conference on Learning Representations, Cited by:
  [§B.1](#A2.SS1.p2.1 "B.1 Generalization to Episodic TTA ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.1](#S4.SS5.SSS1.p1.1 "4.5.1 Generalization to Episodic TTA ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Li et al. (2026) Z. Li, X. Lin, Z. Liu, J. Zou, Z. Wu, L. Zheng, D.
  Fu, Y. Zhu, H. Hamann, H. Tong, et al. Language in the flow of time:
  time-series-paired texts weaved into a unified temporal narrative. In
  International Conference on Learning Representations, Vol. 2026,
  pp. 24437–24484. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Liang et al. (2025) M. Liang, X. Liu, R. Jin, B. Liu, Q. Suo, Q.
  Zhou, S. Zhou, L. Chen, H. Zheng, Z. Li, et al. External large
  foundation model: how to efficiently serve trillions of parameters for
  online ads recommendation. In Companion Proceedings of the ACM on Web
  Conference 2025, pp. 344–353. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2026a) H. Lin, X. Jia, S. Liu, S. Xia, W. Huang, H. Xu, J.
  Li, Y. Xiao, X. Xing, Z. Guo, et al. Efficient diffusion language
  models: a comprehensive survey. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2026b) H. Lin, X. Jia, H. Xu, B. Yao, X. Guo, Y. Wu, Z.
  Lu, Y. Wei, Q. Zhang, and Z. Sun DuQuant++: fine-grained rotation
  enhances microscaling fp4 quantization. arXiv preprint
  arXiv:2604.17789. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2024) H. Lin, H. Xu, Y. Wu, J. Cui, Y. Zhang, L. Mou, L.
  Song, Z. Sun, and Y. Wei Duquant: distributing outliers via dual
  transformation makes stronger quantized llms. Advances in Neural
  Information Processing Systems 37, pp. 87766–87800. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2025a) H. Lin, H. Xu, Y. Wu, Z. Guo, R. Zhang, Z. Lu, Y.
  Wei, Q. Zhang, and Z. Sun Quantization meets dllms: a systematic study
  of post-training quantization for diffusion llms. arXiv preprint
  arXiv:2508.14896. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2026c) X. Lin, P. Li, Z. Zeng, T. Li, T. Wei, X. Ning, G.
  Li, Y. Chen, and H. Tong ALERT: zero-shot llm jailbreak detection via
  internal discrepancy amplification. arXiv preprint arXiv:2601.03600.
  Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2025b) X. Lin, Z. Liu, Z. Yang, G. Li, R. Qiu, S. Wang, H.
  Liu, H. Li, S. Keswani, V. Pardeshi, et al. Moralise: a structured
  benchmark for moral alignment in visual language models. arXiv
  preprint arXiv:2505.14728. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2026d) X. Lin, Z. Tang, W. Cong, M. Hang, K. Wang, Y.
  Wang, Z. Zeng, T. Li, H. Yoo, Z. Liu, et al. Mixture of sequence:
  theme-aware mixture-of-experts for long-sequence recommendation. In
  Proceedings of the ACM Web Conference 2026, pp. 6469–6480. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Lin et al. (2025c) X. Lin, Z. Zeng, T. Wei, Z. Liu, H. Tong, et al.
  Cats: mitigating correlation shift for multivariate time series
  classification. arXiv preprint arXiv:2504.04283. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Liu et al. (2023) H. Liu, C. Li, Q. Wu, and Y. J. Lee Visual
  instruction tuning. Advances in neural information processing systems
  36, pp. 34892–34916. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Liu et al. (2024) X. Liu, Z. Zeng, X. Liu, S. Yuan, W. Song, M.
  Hang, Y. Liu, C. Yang, D. Kim, W. Chen, et al. A collaborative
  ensemble framework for ctr prediction. arXiv preprint
  arXiv:2411.13700. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Liu et al. (2021) Y. Liu, J. Deng, X. Gao, W. Li, and L. Duan
  Bapa-net: boundary adaptation and prototype alignment for cross-domain
  semantic segmentation. In Proceedings of the IEEE/CVF international
  conference on computer vision, pp. 8801–8811. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Long et al. (2015) M. Long, Y. Cao, J. Wang, and M. Jordan Learning
  transferable features with deep adaptation networks. In International
  conference on machine learning, pp. 97–105. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Maharana et al. (2025) S. Maharana, B. Zhang, L. Karlinsky, R. Feris,
  and Y. Guo Batclip: bimodal online test-time adaptation for clip. In
  Proceedings of the IEEE/CVF International Conference on Computer
  Vision, pp. 1569–1579. Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Textual Subspace as Semantic Anchors ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.1](#S4.SS5.SSS1.p1.1 "4.5.1 Generalization to Episodic TTA ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.2](#S4.SS5.SSS2.p2.1 "4.5.2 Visual-Textual Alignment Quality ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Motiian et al. (2017) S. Motiian, M. Piccirilli, D. A. Adjeroh, and G.
  Doretto Unified deep supervised domain adaptation and generalization.
  In Proceedings of the IEEE international conference on computer
  vision, pp. 5715–5725. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p2.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Osowiechi et al. (2024) D. Osowiechi, M. Noori, G. Vargas Hakim, M.
  Yazdanpanah, A. Bahri, M. Cheraghalikhani, S. Dastani, F. Beizaee, I.
  Ayed, and C. Desrosiers WATT: weight average test time adaptation of
  clip. Advances in neural information processing systems 37,
  pp. 48015–48044. Cited by: [1st
  item](#A2.I1.i1.p1.1 "In B.2 Textual Subspace as Semantic Anchors ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§B.2](#A2.SS2.p1.1 "B.2 Textual Subspace as Semantic Anchors ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§3.2](#S3.SS2.p4.1 "3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Patashnik et al. (2021) O. Patashnik, Z. Wu, E. Shechtman, D.
  Cohen-Or, and D. Lischinski Styleclip: text-driven manipulation of
  stylegan imagery. In Proceedings of the IEEE/CVF international
  conference on computer vision, pp. 2085–2094. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Qi et al. (2024) X. Qi, K. Huang, A. Panda, P. Henderson, M. Wang,
  and P. Mittal Visual adversarial examples jailbreak aligned large
  language models. In Proceedings of the AAAI conference on artificial
  intelligence, Vol. 38, pp. 21527–21536. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Radford et al. (2021) A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G.
  Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al.
  Learning transferable visual models from natural language supervision.
  In International conference on machine learning, pp. 8748–8763. Cited
  by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§G.1](#A7.SS1.p3.1 "G.1 Cite Creators Of Artifacts ‣ Appendix G Use Or Create Scientific Artifacts ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px1.p2.1 "Datasets and Metric. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Rusak et al. (2022) E. Rusak, S. Schneider, G. Pachitariu, L.
  Eck, P. V. Gehler, O. Bringmann, W. Brendel, and M. Bethge If your
  data distribution shifts, use self-learning. Transactions on Machine
  Learning Research. Note: Expert Certification External Links: ISSN
  2835-8856, [Link](https://openreview.net/forum?id=vqRzLv6POg) Cited
  by: [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Saito et al. (2019) K. Saito, D. Kim, S. Sclaroff, T. Darrell, and K.
  Saenko Semi-supervised domain adaptation via minimax entropy. In
  Proceedings of the IEEE/CVF international conference on computer
  vision, pp. 8050–8058. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p2.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Shen et al. (2018) J. Shen, Y. Qu, W. Zhang, and Y. Yu Wasserstein
  distance guided representation learning for domain adaptation. In
  Proceedings of the AAAI conference on artificial intelligence, Vol.
  32. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Shu et al. (2022) M. Shu, W. Nie, D. Huang, Z. Yu, T. Goldstein, A.
  Anandkumar, and C. Xiao Test-time prompt tuning for zero-shot
  generalization in vision-language models. Advances in Neural
  Information Processing Systems 35, pp. 14274–14289. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Simon et al. (2020) C. Simon, P. Koniusz, R. Nock, and M. Harandi
  Adaptive subspaces for few-shot learning. In Proceedings of the
  IEEE/CVF conference on computer vision and pattern recognition,
  pp. 4136–4145. Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Sun et al. (2017) B. Sun, J. Feng, and K. Saenko Correlation alignment
  for unsupervised domain adaptation. In Domain adaptation in computer
  vision applications, pp. 153–171. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Sun and Saenko (2016) B. Sun and K. Saenko Deep coral: correlation
  alignment for deep domain adaptation. In European conference on
  computer vision, pp. 443–450. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Touvron et al. (2023) H. Touvron, T. Lavril, G. Izacard, X.
  Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F.
  Azhar, et al. Llama: open and efficient foundation language models.
  arXiv preprint arXiv:2302.13971. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2023a) B. Wang, W. Chen, H. Pei, C. Xie, M. Kang, C.
  Zhang, C. Xu, Z. Xiong, R. Dutta, R. Schaeffer, et al. DecodingTrust:
  a comprehensive assessment of trustworthiness in gpt models.. In
  NeurIPS, Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2020) D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T.
  Darrell Tent: fully test-time adaptation by entropy minimization.
  arXiv preprint arXiv:2006.10726. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px3.p1.1 "Pipeline ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.2](#S4.SS5.SSS2.p2.1 "4.5.2 Visual-Textual Alignment Quality ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2023b) D. Wang, Y. Yan, R. Qiu, Y. Zhu, K. Guan, A.
  Margenot, and H. Tong Networked time series imputation via
  position-aware graph enhanced variational autoencoders. In Proceedings
  of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data
  Mining, pp. 2256–2268. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2019) H. Wang, S. Ge, Z. Lipton, and E. P. Xing Learning
  robust global representations by penalizing local predictive power. In
  Advances in Neural Information Processing Systems, pp. 10506–10518.
  Cited by: [Appendix
  D](#A4.p3.1 "Appendix D Datasets ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px1.p1.1 "Datasets and Metric. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2023c) R. Wang, B. Li, Y. Lu, D. Sun, J. Li, Y. Yan, S.
  Liu, H. Tong, and T. F. Abdelzaher Noisy positive-unlabeled learning
  with self-training for speculative knowledge graph reasoning. arXiv
  preprint arXiv:2306.07512. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2018) R. Wang, Y. Yan, J. Wang, Y. Jia, Y. Zhang, W.
  Zhang, and X. Wang Acekg: a large-scale knowledge graph for academic
  data mining. In Proceedings of the 27th ACM international conference
  on information and knowledge management, pp. 1487–1490. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Wang et al. (2025) Z. Wang, J. Dai, K. Li, X. Li, Y. Guo, and M. Xiang
  Complementary subspace low-rank adaptation of vision-language models
  for few-shot classification. arXiv preprint arXiv:2501.15040. Cited
  by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- \[70\] Z. Xiao, S. Yan, J. Hong, J. Cai, X. Jiang, Y. Hu, J. Shen, C.
  Wang, and C. G. Snoek DynaPrompt: dynamic test-time prompt tuning. In
  The Thirteenth International Conference on Learning Representations,
  Cited by:
  [§4.5.1](#S4.SS5.SSS1.p1.1 "4.5.1 Generalization to Episodic TTA ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Xu et al. (2026) H. Xu, S. Chen, R. Qiu, Y. Yan, C. Luo, M. X.
  Cheng, J. He, and H. Tong Prune as you generate: online rollout
  pruning for faster and better rlvr. In Proceedings of the 64th Annual
  Meeting of the Association for Computational Linguistics (Volume 1:
  Long Papers), pp. 13876–13893. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- \[72\] H. Xu, Y. Yan, D. Wang, Z. Xu, Z. Zeng, T. F. Abdelzaher, J.
  Han, and H. Tong Slog: an inductive spectral graph neural network
  beyond polynomial filter. In Forty-first International Conference on
  Machine Learning, Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Xu et al. (2020) M. Xu, H. Wang, B. Ni, Q. Tian, and W. Zhang
  Cross-domain detection via graph-induced prototype alignment. In
  Proceedings of the IEEE/CVF conference on computer vision and pattern
  recognition, pp. 12355–12364. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2017) H. Yan, Y. Ding, P. Li, Q. Wang, Y. Xu, and W. Zuo
  Mind the class weight bias: weighted maximum mean discrepancy for
  unsupervised domain adaptation. In Proceedings of the IEEE conference
  on computer vision and pattern recognition, pp. 2272–2281. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2024a) Y. Yan, Y. Chen, H. Chen, X. Li, Z. Xu, Z. Zeng, L.
  Liu, Z. Liu, and H. Tong Thegcn: temporal heterophilic graph
  convolutional network. arXiv preprint arXiv:2412.16435. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2023a) Y. Yan, Y. Chen, H. Chen, M. Xu, M. Das, H. Yang,
  and H. Tong From trainable negative depth to edge heterophily in
  graphs. Advances in Neural Information Processing Systems 36,
  pp. 70162–70178. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2024b) Y. Yan, Y. Hu, Q. Zhou, L. Liu, Z. Zeng, Y.
  Chen, M. Pan, H. Chen, M. Das, and H. Tong Pacer: network embedding
  from positional to structural. In Proceedings of the ACM Web
  Conference 2024, pp. 2485–2496. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2023b) Y. Yan, B. Jing, L. Liu, R. Wang, J. Li, T.
  Abdelzaher, and H. Tong Reconciling competing sampling strategies of
  network embedding. Advances in Neural Information Processing Systems
  36, pp. 6844–6861. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2025) Y. Yan, A. Kolekar, S. Genc, W. Xu, E. W. Huang, A.
  Srinivasan, M. Jain, Q. He, and H. Tong To answer or not to answer
  (taona): a robust textual graph understanding and question answering
  approach. In Findings of the Association for Computational
  Linguistics: EMNLP 2025, pp. 6360–6376. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2021a) Y. Yan, L. Liu, Y. Ban, B. Jing, and H. Tong
  Dynamic knowledge graph alignment. In Proceedings of the AAAI
  conference on artificial intelligence, Vol. 35, pp. 4564–4572. Cited
  by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yan et al. (2021b) Y. Yan, S. Zhang, and H. Tong Bright: a bridging
  algorithm for network alignment. In Proceedings of the web conference
  2021, pp. 3907–3917. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yang et al. (2024) X. Yang, H. Chen, Y. Yan, Y. Tang, Y. Zhao, E.
  Xu, Y. Cai, and H. Tong SimCE: simplifying cross-entropy loss for
  collaborative filtering. arXiv preprint arXiv:2406.16170. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yoo et al. (2024) H. Yoo, Z. Zeng, J. Kang, R. Qiu, D. Zhou, Z.
  Liu, F. Wang, C. Xu, E. Chan, and H. Tong Ensuring user-side fairness
  in dynamic recommender systems. In Proceedings of the ACM Web
  Conference 2024, pp. 3667–3678. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yu et al. (2022) J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M.
  Seyedhosseini, and Y. Wu Coca: contrastive captioners are image-text
  foundation models. arXiv preprint arXiv:2205.01917. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yu et al. (2026a) Q. Yu, R. Qiu, Z. Zeng, M. T. Thai, H. Liu, and H.
  Tong AvAtar: learning to align via active optimal transport. arXiv
  preprint arXiv:2605.24395. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yu et al. (2026b) Q. Yu, Z. Zeng, K. Tieu, X. Yang, R. Qiu, Y. Yan, L.
  Liu, Y. Zhao, L. Chen, J. He, et al. From inference to adaptation: a
  unified optimal transport view of vision language model. arXiv
  preprint arXiv:2608.18339. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yu et al. (2025a) Q. Yu, Z. Zeng, Y. Yan, Z. Liu, B. Jing, R. Qiu, A.
  Azad, and H. Tong PLANETALIGN: a comprehensive python library for
  benchmarking network alignment. arXiv preprint arXiv:2505.21366. Cited
  by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yu et al. (2025b) Q. Yu, Z. Zeng, Y. Yan, L. Ying, R. Srikant, and H.
  Tong Joint optimal transport and embedding for network alignment. In
  Proceedings of the ACM on Web Conference 2025, pp. 2064–2075. Cited
  by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yu et al. (2023) S. Yu, J. Cho, P. Yadav, and M. Bansal Self-chained
  image-language model for video localization and question answering.
  Advances in Neural Information Processing Systems 36, pp. 76749–76771.
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Yuan et al. (2023) L. Yuan, B. Xie, and S. Li Robust test-time
  adaptation in dynamic scenarios. In Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition,
  pp. 15922–15932. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zanella et al. (2025) M. Zanella, C. Fuchs, C. De Vleeschouwer, and I.
  Ben Ayed Realistic test-time adaptation of vision-language models. In
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, pp. 25103–25112. Cited by:
  [§C.5](#A3.SS5.p1.1 "C.5 Sensitivity to the Number of Classes ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2024a) Z. Zeng, B. Du, S. Zhang, Y. Xia, Z. Liu, and H.
  Tong Hierarchical multi-marginal optimal transport for network
  alignment. In Proceedings of the AAAI Conference on Artificial
  Intelligence, Vol. 38, pp. 16660–16668. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2025a) Z. Zeng, M. Hang, X. Liu, X. Liu, X. Lin, R.
  Qiu, T. Wei, Z. Liu, S. Yuan, C. Yang, et al. Hierarchical lora moe
  for efficient ctr model scaling. arXiv preprint arXiv:2510.10432.
  Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2025b) Z. Zeng, X. Liu, M. Hang, X. Liu, Q. Zhou, C.
  Yang, Y. Liu, Y. Ruan, L. Chen, Y. Chen, et al. InterFormer: effective
  heterogeneous interaction learning for click-through rate prediction.
  In Proceedings of the 34th ACM International Conference on Information
  and Knowledge Management, pp. 6225–6233. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2025c) Z. Zeng, R. Qiu, W. Bao, T. Wei, X. Lin, Y.
  Yan, T. F. Abdelzaher, J. Han, and H. Tong Pave your own path: graph
  gradual domain adaptation on fused gromov-wasserstein geodesics. arXiv
  preprint arXiv:2505.12709. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px2.p3.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2024b) Z. Zeng, R. Qiu, Z. Xu, Z. Liu, Y. Yan, T. Wei, L.
  Ying, J. He, and H. Tong Graph mixup on approximate gromov–wasserstein
  geodesics. In Forty-first International Conference on Machine
  Learning, Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2025d) Z. Zeng, Q. Yu, X. Lin, R. Qiu, X. Ning, T.
  Wei, Y. Yan, J. He, and H. Tong Harnessing consistency for robust
  test-time llm ensemble. arXiv preprint arXiv:2510.13855. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2023a) Z. Zeng, S. Zhang, Y. Xia, and H. Tong Parrot:
  position-aware regularized optimal transport for network alignment. In
  Proceedings of the ACM web conference 2023, pp. 372–382. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zeng et al. (2023b) Z. Zeng, R. Zhu, Y. Xia, H. Zeng, and H. Tong
  Generative graph dictionary learning. In International Conference on
  Machine Learning, pp. 40749–40769. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhang et al. (2022) M. Zhang, S. Levine, and C. Finn Memo: test time
  robustness via adaptation and augmentation. Advances in neural
  information processing systems 35, pp. 38629–38642. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px2.p4.1 "Domain Adaptation. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhang et al. (2024) Y. Zhang, W. Zhu, H. Tang, Z. Ma, K. Zhou, and L.
  Zhang Dual memory networks: a versatile adaptation approach for
  vision-language models. In Proceedings of the IEEE/CVF conference on
  computer vision and pattern recognition, pp. 28718–28728. Cited by:
  [§B.1](#A2.SS1.p2.1 "B.1 Generalization to Episodic TTA ‣ Appendix B Further Discussions ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§1](#S1.p2.1 "1 Introduction ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Vision-Language Model Test-time Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.1](#S4.SS1.SSS0.Px2.p1.1 "Baselines. ‣ 4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
  [§4.5.1](#S4.SS5.SSS1.p1.1 "4.5.1 Generalization to Episodic TTA ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhang et al. (2025) Y. Zhang, D. Yu, T. Ge, L. Song, Z. Zeng, H.
  Mi, N. Jiang, and D. Yu Improving llm general preference alignment via
  optimistic online mirror descent. arXiv preprint arXiv:2502.16852.
  Cited by: [Appendix
  E](#A5.SS0.SSS0.Px1.p1.1 "Pre-trained Foundation Models. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhao et al. (2023a) C. Zhao, H. Zhao, M. He, J. Zhang, and J. Fan
  Cross-domain recommendation via user interest alignment. In
  Proceedings of the ACM web conference 2023, pp. 887–896. Cited by:
  [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhao et al. (2023b) C. Zhao, H. Zhao, X. Li, M. He, J. Wang, and J.
  Fan Cross-domain recommendation via progressive structural alignment.
  IEEE Transactions on Knowledge and Data Engineering 36 (6),
  pp. 2401–2415. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhu et al. (2019) X. Zhu, J. Pang, C. Yang, J. Shi, and D. Lin
  Adapting object detectors via selective cross-domain alignment. In
  Proceedings of the IEEE/CVF conference on computer vision and pattern
  recognition, pp. 687–696. Cited by: [Appendix
  E](#A5.SS0.SSS0.Px3.p1.1 "Cross-Domain Alignment. ‣ Appendix E More Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
- Zhu et al. (2024) X. Zhu, B. Zhu, Y. Tan, S. Wang, Y. Hao, and H.
  Zhang Selective vision-language subspace projection for few-shot clip.
  In ACM Multimedia 2024, Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Geometric Adaptation. ‣ 2 Related Works ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

## Appendix

## Appendix A Algorithm

We provide the pseudo code of SubTTA in
Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix A Algorithm ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
which includes three steps: geometric alignment, semantic projection and
standard TTA.

0:  Image encoder $`f_{v}`$; text encoder $`f_{t}`$; image batches
$`\mathcal{D}_{\text{test}}=\{\mathbf{X}^{(1)},\mathbf{X}^{(2)},\dots\}`$;
textual prompts $`t_{1},\dots,t_{C}`$; subspace rank $`r`$; momentum
coefficient $`\alpha`$.

0:  Prediction $`\hat{Y}`$ for the test images.
Stage 1: Textual Subspace Initialization

1:  Extract normalized text features
$`\mathbf{T}=[\mathbf{t}_{1},\dots,\mathbf{t}_{C}]^{\top}\in\mathbb{R}^{C\times d}`$
with $`\mathbf{t}_{i}=f_{t}(t_{i})`$.

2:  Compute textual covariance
$`\mathbf{\Sigma}_{\mathcal{T}}=\mathbf{T}^{\top}\mathbf{T}`$.

3:  Compute textual basis
$`\mathbf{B}_{\mathcal{T}}\in\mathbb{R}^{r\times d}`$ via
Eq. ([2](#S3.E2 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")).

4:  Initialize visual covariance prior
$`\mathbf{\Sigma}_{\mathcal{V}}\leftarrow\mathbf{\Sigma}_{\mathcal{T}}`$.
Stage 2: Online Test-Time Adaptation

5:  for each test batch $`\mathbf{X}^{(k)}`$ in
$`\mathcal{D}_{\text{test}}`$ do

6:   Extract visual features
$`\mathbf{V}^{(k)}=f_{v}(\mathbf{X}^{(k)})\in\mathbb{R}^{B\times d}`$.

7:   Visual covariance EMA via
Eq. ([1](#S3.E1 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"))

8:   Compute visual basis
$`\mathbf{B}_{\mathcal{V}}\in\mathbb{R}^{r\times d}`$ via
Eq. ([2](#S3.E2 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")).
Step 1: Geometric Alignment

9:   Compute alignment loss $`\mathcal{L}_{\text{align}}`$ in
Eq. ([4](#S3.E4 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")).

10:   Update model parameters $`\theta`$ by Adam with
$`\nabla_{\theta}\mathcal{L}_{\text{align}}`$.
Step 2: Semantic Projection

11:   Extract image features $`\tilde{\mathbf{V}}^{(k)}`$ using updated
model.

12:   Project features onto textual subspace via
Eq. ([5](#S3.E5 "In 3.3 Semantic Projection ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")).
Step 3: Standard TTA

13:   Compute standard TTA loss $`\mathcal{L}_{\text{TTA}}`$ (e.g., MINT
or Entropy) on purified features $`\mathbf{V}_{\text{proj}}`$.

14:   Update model parameters $`\theta`$ by Adam with
$`\nabla_{\theta}\mathcal{L}_{\text{TTA}}`$.

15:   Compute prediction
$`\hat{Y}^{(k)}=\mathop{\arg\max}_{c}\mathbf{V}_{\text{proj}}\mathbf{T}^{\top}`$.

16:  end for

17:  return TTA predictions $`\hat{Y}^{(1)},\hat{Y}^{(2)},\dots`$.

Algorithm 1 SubTTA

## Appendix B Further Discussions

### B.1 Generalization to Episodic TTA

The default configuration of SubTTA operates under the standard online
TTA paradigm, where data arrives in continuous batches. However, in
strictly episodic TTA, where adaptation is performed on single isolated
test samples, or extreme small-batch scenarios, relying solely on
current statistics becomes infeasible. The lack of sufficient batch
context leads to severe high-variance or rank-deficient covariance
estimation, which subsequently collapses the visual subspace.

To elegantly extend SubTTA to these challenging regimes, we seamlessly
integrate a lightweight memory bank mechanism [Zhang et al.
(2024)](#bib.bib7); [Li et al. (2025)](#bib.bib8). Specifically, we
maintain a First-In-First-Out (FIFO) queue of size $`M`$ to dynamically
accumulate the most recent test features. Instead of computing the
visual subspace based purely on the current, inadequate observation, we
compute the empirical covariance over the buffered features in the
memory bank. This mechanism effectively smooths the statistical
fluctuations and constructs a reliable local representation of the data
distribution. Consequently, even in the complete absence of large batch
statistics, the memory bank ensures a smooth and stable statistical
subspace estimation, providing a robust foundation for SubTTA’s
cross-modal geometric alignment.

### B.2 Textual Subspace as Semantic Anchors

We follow the standard practice in existing VLM TTA approaches [Maharana
et al. (2025)](#bib.bib2); [Bao et al. (2025)](#bib.bib3); [Deng et al.
(2025)](#bib.bib4); [Osowiechi et al. (2024)](#bib.bib5) by updating the
normalization layers of the visual encoder while keeping the text
encoder and prompts frozen. The rationale for utilizing the textual
subspace as a semantic anchor stems from the inherent differences
between the two modalities: textual prompts (e.g., “a photo of a
\<class\>”) encode highly distilled, task-relevant semantics.
Conversely, image embeddings often entangle the core object with
task-irrelevant visual nuisances, such as background, lighting, and
stylistic variations.

However, relying exclusively on the textual subspace assumes it
perfectly encapsulates the task semantics. To mitigate the potential
risks of a biased or imperfect textual subspace, two potential
strategies can be integrated into our framework:

- •
  Prompt Ensembling: Similar to [Osowiechi et al. (2024)](#bib.bib5), by
  employing multiple diverse prompt templates, we can construct a more
  robust ensembled textual subspace. This approach broadens the semantic
  coverage and significantly reduces the model’s sensitivity to the
  biases of any single prompt.
- •
  Soft Projection via Residual Connections: Instead of enforcing a rigid
  projection in
  Eq. ([5](#S3.E5 "In 3.3 Semantic Projection ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")),
  we may introduce a residual connection as follows

  |  |  |  |
  |----|----|----|
  |  |
  ``` math
  \text{proj}(\mathbf{v})=\mathbf{v}\mathbf{B}_{T}^{\top}\mathbf{B}_{T}+\lambda\mathbf{v},
  ``` |  |

  where $`\lambda`$ is a hyperparameter governing the weight of the
  residual signal. This soft projection formulation deliberately
  preserves a fraction of the features outside the textual subspace,
  thereby preventing the excessive discarding of potentially useful
  visual cues when the textual anchor is imperfect.

## Appendix C Experiments

### C.1 Results with Different Backbones

To verify the scalability and generalization of our approach, we provide
detailed benchmark results using ViT-B-32 and ViT-L-14 backbones in
Table [4](#A3.T4 "Table 4 ‣ C.1 Results with Different Backbones ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
and
Table [5](#A3.T5 "Table 5 ‣ C.1 Results with Different Backbones ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
respectively. The results align with the observations on ViT-B-16
reported in the main text, confirming three key trends:

| Method |  | Noise |  |  | Blur |  |  |  | Weather |  |  |  | Digital |  |  |  | Mean |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  |  | Gauss. | Shot | Impul. | Defoc. | Glass | Motion | Zoom | Snow | Frost | Fog | Brit. | Contr. | Elast. | Pixel. | JPEG |  |
|   ImageNet-C | Source | 12.90 | 13.06 | 12.80 | 24.38 | 11.82 | 22.68 | 20.24 | 25.60 | 25.84 | 30.24 | 50.48 | 17.30 | 18.98 | 32.32 | 29.10 | 23.18 |
|  | TDA | 12.80 | 14.72 | 14.84 | 24.14 | 12.98 | 23.46 | 20.98 | 26.74 | 28.00 | 32.28 | 51.64 | 18.16 | 20.70 | 33.00 | 30.20 | 24.31 |
|  | DMN | 13.00 | 14.32 | 14.50 | 22.92 | 11.78 | 22.18 | 19.68 | 23.70 | 25.72 | 29.62 | 50.50 | 14.78 | 19.92 | 32.46 | 29.36 | 22.96 |
|  | VTE | 11.98 | 12.32 | 13.44 | 25.06 | 11.60 | 22.60 | 22.34 | 27.40 | 27.04 | 32.28 | 51.62 | 16.82 | 20.02 | 34.80 | 32.78 | 24.14 |
|  | ZERO | 12.36 | 12.88 | 12.58 | 24.90 | 12.06 | 22.84 | 21.46 | 26.72 | 26.64 | 30.58 | 50.98 | 16.80 | 19.66 | 33.82 | 30.52 | 23.65 |
|  | ECALP | 14.40 | 15.10 | 14.68 | 23.28 | 12.18 | 23.18 | 21.08 | 24.40 | 25.38 | 29.50 | 47.92 | 18.26 | 19.12 | 31.42 | 29.00 | 23.26 |
|  | RoTTA | 13.22 | 13.30 | 13.10 | 24.52 | 12.02 | 22.93 | 20.30 | 25.88 | 26.00 | 30.34 | 50.46 | 17.40 | 19.10 | 32.38 | 29.22 | 23.34 |
|  | TPT | 12.16 | 12.60 | 12.48 | 25.36 | 12.22 | 22.42 | 20.94 | 26.70 | 26.78 | 30.50 | 50.82 | 16.88 | 19.88 | 33.40 | 30.50 | 23.58 |
|  | MEMO | 12.84 | 13.00 | 12.78 | 24.66 | 11.88 | 22.82 | 20.40 | 25.70 | 26.00 | 30.28 | 50.50 | 17.32 | 19.02 | 32.48 | 29.14 | 23.25 |
|  | WATT | 14.28 | 14.64 | 14.42 | 25.62 | 15.26 | 25.52 | 21.92 | 26.76 | 25.86 | 31.44 | 50.70 | 22.22 | 19.68 | 33.56 | 30.64 | 24.83 |
|  | MINT | 18.84 | 19.98 | 19.42 | 24.86 | 19.08 | 27.20 | 22.44 | 27.94 | 27.28 | 32.72 | 50.64 | 20.18 | 24.04 | 34.84 | 33.14 | 26.84 |
|  | BATCLIP | 17.64 | 18.60 | 16.98 | 24.56 | 20.40 | 28.46 | 24.66 | 29.16 | 27.02 | 35.76 | 49.56 | 20.56 | 27.62 | 35.24 | 34.26 | 27.37 |
|  | SubTTA | 22.56 | 23.52 | 23.02 | 25.66 | 23.36 | 28.46 | 27.42 | 31.18 | 32.32 | 39.08 | 50.50 | 26.00 | 33.00 | 37.12 | 34.28 | 30.50 |
|   CIFAR10-C | Source | 35.51 | 40.01 | 43.17 | 69.91 | 41.46 | 64.52 | 70.10 | 70.84 | 72.32 | 66.65 | 81.35 | 64.48 | 59.69 | 48.18 | 56.62 | 58.99 |
|  | TDA | 41.49 | 43.44 | 41.44 | 71.14 | 44.86 | 66.96 | 72.37 | 72.20 | 74.53 | 68.46 | 83.57 | 65.64 | 62.61 | 51.14 | 55.67 | 61.03 |
|  | DMN | 38.83 | 39.45 | 42.12 | 70.74 | 39.48 | 65.01 | 72.78 | 72.52 | 73.41 | 66.45 | 82.39 | 58.46 | 60.51 | 49.05 | 55.92 | 59.14 |
|  | VTE | 47.55 | 50.18 | 53.11 | 71.35 | 53.87 | 67.89 | 72.95 | 76.37 | 76.24 | 70.76 | 83.33 | 61.05 | 68.99 | 58.57 | 61.07 | 64.89 |
|  | ZERO | 37.11 | 41.66 | 46.79 | 71.21 | 42.51 | 65.21 | 71.58 | 72.27 | 73.79 | 68.92 | 82.78 | 65.53 | 62.35 | 47.66 | 58.80 | 60.54 |
|  | ECALP | 44.36 | 47.05 | 43.93 | 71.17 | 43.56 | 68.32 | 73.92 | 72.49 | 74.92 | 68.06 | 82.61 | 62.95 | 61.91 | 47.18 | 55.67 | 61.21 |
|  | RoTTA | 36.36 | 41.12 | 43.76 | 69.94 | 42.44 | 64.68 | 70.06 | 71.34 | 72.83 | 67.33 | 81.74 | 65.12 | 60.41 | 49.37 | 57.26 | 59.58 |
|  | TPT | 43.13 | 46.67 | 48.30 | 71.33 | 47.77 | 66.95 | 72.05 | 73.95 | 76.09 | 68.73 | 84.16 | 66.31 | 63.91 | 51.82 | 58.02 | 62.61 |
|  | MEMO | 36.50 | 40.68 | 44.28 | 70.81 | 42.10 | 64.67 | 70.63 | 72.17 | 73.01 | 67.63 | 82.10 | 64.25 | 61.03 | 47.84 | 57.61 | 59.69 |
|  | WATT | 43.63 | 48.69 | 49.13 | 72.23 | 46.15 | 66.74 | 71.04 | 73.64 | 74.46 | 71.25 | 83.99 | 73.12 | 61.98 | 59.25 | 62.98 | 63.89 |
|  | MINT | 54.18 | 57.78 | 47.30 | 73.49 | 56.01 | 73.71 | 76.39 | 74.94 | 74.46 | 70.16 | 85.66 | 70.60 | 64.66 | 58.85 | 59.53 | 66.51 |
|  | BATCLIP | 52.19 | 55.70 | 53.96 | 76.09 | 55.09 | 74.75 | 75.18 | 77.23 | 78.09 | 74.97 | 86.03 | 77.37 | 67.53 | 58.01 | 61.90 | 68.27 |
|  | SubTTA | 60.43 | 63.78 | 56.40 | 78.48 | 61.66 | 76.24 | 77.26 | 80.03 | 79.99 | 77.95 | 88.72 | 83.62 | 70.51 | 73.77 | 63.46 | 72.82 |
|   CIFAR100-C | Source | 16.18 | 17.76 | 17.55 | 39.08 | 17.63 | 38.55 | 43.81 | 42.34 | 43.41 | 39.59 | 50.38 | 29.41 | 28.80 | 22.83 | 29.38 | 31.78 |
|  | TDA | 18.21 | 21.36 | 19.35 | 40.51 | 17.73 | 40.16 | 45.11 | 44.78 | 45.55 | 40.37 | 52.16 | 30.42 | 29.32 | 22.58 | 31.42 | 33.27 |
|  | DMN | 17.26 | 20.32 | 13.44 | 36.86 | 15.23 | 40.23 | 45.85 | 42.77 | 44.00 | 39.05 | 51.71 | 27.21 | 27.93 | 19.54 | 29.44 | 31.39 |
|  | VTE | 16.83 | 18.33 | 19.01 | 39.62 | 22.89 | 39.10 | 43.83 | 44.58 | 44.86 | 39.18 | 49.39 | 28.34 | 34.14 | 26.92 | 30.11 | 33.14 |
|  | ZERO | 15.91 | 17.91 | 19.53 | 41.10 | 17.18 | 39.79 | 44.78 | 43.33 | 43.47 | 40.99 | 51.27 | 29.41 | 30.62 | 21.81 | 31.14 | 32.55 |
|  | ECALP | 18.36 | 19.56 | 17.87 | 38.74 | 17.72 | 39.64 | 44.71 | 43.34 | 45.14 | 40.60 | 52.56 | 31.01 | 30.33 | 23.01 | 30.17 | 32.85 |
|  | RoTTA | 16.57 | 18.30 | 17.73 | 38.58 | 17.67 | 38.58 | 43.71 | 42.44 | 43.38 | 39.25 | 50.72 | 28.96 | 28.83 | 23.43 | 29.79 | 31.86 |
|  | TPT | 15.99 | 17.58 | 17.44 | 39.17 | 19.57 | 38.90 | 43.89 | 43.56 | 44.49 | 40.03 | 50.92 | 27.73 | 30.95 | 23.33 | 29.61 | 32.21 |
|  | MEMO | 16.51 | 18.17 | 18.19 | 41.49 | 17.48 | 40.34 | 45.94 | 44.18 | 43.94 | 40.79 | 52.64 | 29.60 | 30.52 | 22.78 | 31.05 | 32.91 |
|  | WATT | 21.40 | 22.34 | 23.29 | 47.58 | 18.63 | 44.03 | 49.82 | 47.62 | 47.86 | 45.32 | 57.62 | 43.61 | 33.05 | 29.71 | 35.81 | 37.85 |
|  | MINT | 23.81 | 26.29 | 21.36 | 45.67 | 24.00 | 42.27 | 49.00 | 46.20 | 44.95 | 43.11 | 56.20 | 38.88 | 34.05 | 28.08 | 32.47 | 37.09 |
|  | BATCLIP | 21.45 | 24.54 | 22.85 | 46.17 | 23.20 | 44.86 | 49.82 | 47.01 | 46.62 | 44.98 | 58.58 | 38.72 | 34.69 | 28.45 | 33.23 | 37.68 |
|  | SubTTA | 25.58 | 27.39 | 24.28 | 46.73 | 26.62 | 44.17 | 50.04 | 47.87 | 47.90 | 45.58 | 57.72 | 41.09 | 37.02 | 32.97 | 35.75 | 39.38 |

Table 4: Benchmark results with ViT-B-32. We denote Top-1/2/3 by
Blue/Yellow/Red, respectively.

| Method |  | Noise |  |  | Blur |  |  |  | Weather |  |  |  | Digital |  |  |  | Mean |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  |  | Gauss. | Shot | Impul. | Defoc. | Glass | Motion | Zoom | Snow | Frost | Fog | Brit. | Contr. | Elast. | Pixel. | JPEG |  |
|   ImageNet-C | Source | 27.40 | 29.42 | 28.72 | 34.58 | 25.28 | 40.96 | 36.70 | 49.80 | 44.10 | 49.78 | 65.36 | 35.18 | 30.42 | 53.48 | 42.24 | 39.56 |
|  | TDA | 27.52 | 30.72 | 32.30 | 35.20 | 27.04 | 42.60 | 38.20 | 51.40 | 46.36 | 52.08 | 66.76 | 37.70 | 31.76 | 55.22 | 44.00 | 41.26 |
|  | DMN | 27.44 | 30.16 | 30.84 | 32.72 | 25.98 | 41.26 | 37.00 | 49.54 | 44.56 | 50.02 | 66.00 | 29.50 | 31.10 | 54.90 | 42.18 | 39.55 |
|  | VTE | 26.36 | 29.20 | 28.98 | 36.38 | 24.88 | 40.92 | 39.36 | 50.52 | 44.92 | 50.86 | 67.42 | 33.80 | 30.56 | 55.58 | 47.76 | 40.50 |
|  | ZERO | 27.12 | 28.70 | 28.72 | 35.54 | 25.66 | 40.50 | 38.16 | 50.56 | 43.84 | 50.32 | 66.04 | 35.24 | 30.74 | 55.08 | 44.98 | 40.08 |
|  | ECALP | 29.48 | 30.34 | 30.86 | 35.60 | 27.50 | 42.82 | 38.38 | 50.48 | 45.42 | 50.52 | 65.40 | 37.44 | 30.12 | 55.16 | 42.80 | 40.82 |
|  | RoTTA | 27.62 | 29.60 | 29.02 | 34.68 | 25.50 | 41.18 | 36.82 | 50.04 | 44.20 | 49.76 | 65.40 | 35.08 | 30.82 | 53.64 | 42.36 | 39.71 |
|  | TPT | 27.08 | 30.00 | 29.54 | 35.84 | 25.78 | 41.50 | 37.98 | 51.64 | 45.72 | 51.78 | 67.78 | 36.34 | 31.64 | 55.56 | 46.16 | 40.96 |
|  | MEMO | 27.42 | 29.46 | 28.74 | 34.66 | 25.34 | 41.00 | 36.74 | 49.78 | 44.16 | 49.80 | 65.42 | 35.16 | 30.40 | 53.56 | 42.24 | 39.59 |
|  | WATT | 29.82 | 31.42 | 30.70 | 36.68 | 28.92 | 42.30 | 38.48 | 51.10 | 44.46 | 50.70 | 65.60 | 39.18 | 33.54 | 54.16 | 43.30 | 41.36 |
|  | MINT | 32.00 | 33.20 | 33.84 | 37.22 | 33.38 | 43.68 | 39.28 | 53.62 | 45.62 | 53.42 | 66.34 | 39.16 | 37.32 | 55.46 | 49.22 | 43.52 |
|  | BATCLIP | 31.44 | 31.40 | 30.94 | 35.84 | 31.68 | 43.72 | 39.26 | 50.16 | 43.02 | 51.12 | 65.50 | 40.14 | 35.38 | 53.68 | 48.92 | 42.15 |
|  | SubTTA | 32.96 | 35.28 | 36.48 | 35.70 | 34.86 | 44.02 | 41.34 | 53.22 | 47.12 | 55.70 | 65.74 | 46.30 | 35.88 | 55.82 | 47.84 | 44.55 |
|   CIFAR10-C | Source | 64.33 | 67.10 | 75.90 | 80.47 | 51.55 | 80.34 | 83.04 | 83.21 | 84.92 | 79.14 | 91.07 | 84.09 | 66.44 | 73.60 | 72.26 | 75.83 |
|  | TDA | 68.32 | 70.62 | 75.97 | 80.67 | 50.18 | 81.03 | 83.07 | 84.32 | 85.82 | 79.02 | 91.78 | 84.65 | 66.94 | 73.93 | 73.70 | 76.67 |
|  | DMN | 68.64 | 69.94 | 74.76 | 76.33 | 51.12 | 79.33 | 80.86 | 84.94 | 85.87 | 79.15 | 91.94 | 84.57 | 66.20 | 71.97 | 73.34 | 75.93 |
|  | VTE | 63.90 | 66.97 | 75.24 | 78.72 | 51.65 | 79.43 | 81.32 | 83.60 | 85.27 | 76.43 | 90.13 | 81.33 | 70.11 | 74.85 | 70.79 | 75.32 |
|  | ZERO | 66.12 | 69.49 | 76.85 | 81.97 | 53.45 | 80.71 | 84.40 | 85.19 | 86.59 | 79.83 | 92.19 | 84.51 | 69.94 | 74.40 | 73.71 | 77.29 |
|  | ECALP | 70.41 | 72.13 | 79.90 | 80.93 | 54.42 | 81.07 | 83.70 | 84.92 | 86.51 | 79.44 | 92.25 | 84.69 | 68.12 | 74.67 | 73.61 | 77.78 |
|  | RoTTA | 65.73 | 68.49 | 76.57 | 80.54 | 52.39 | 80.51 | 83.29 | 83.56 | 85.35 | 79.51 | 91.28 | 84.30 | 67.01 | 74.62 | 73.07 | 76.41 |
|  | TPT | 67.15 | 69.87 | 79.01 | 80.83 | 55.33 | 81.17 | 83.43 | 85.14 | 86.97 | 79.81 | 92.16 | 83.87 | 70.87 | 76.52 | 74.07 | 77.75 |
|  | MEMO | 64.88 | 67.74 | 76.40 | 80.83 | 52.16 | 80.58 | 83.46 | 83.84 | 85.51 | 79.40 | 91.38 | 84.11 | 67.45 | 73.90 | 73.20 | 76.32 |
|  | WATT | 70.84 | 72.44 | 77.03 | 82.83 | 61.91 | 82.35 | 85.15 | 85.70 | 86.98 | 81.93 | 91.84 | 88.25 | 72.04 | 77.76 | 77.15 | 79.61 |
|  | MINT | 68.12 | 73.40 | 77.52 | 85.58 | 63.27 | 83.93 | 87.00 | 86.77 | 88.39 | 83.15 | 94.12 | 88.96 | 73.42 | 79.10 | 76.07 | 80.59 |
|  | BATCLIP | 74.81 | 78.11 | 83.70 | 87.53 | 71.62 | 86.74 | 89.29 | 89.99 | 89.73 | 87.53 | 94.65 | 92.29 | 78.92 | 83.37 | 80.09 | 84.56 |
|  | SubTTA | 77.28 | 79.97 | 84.14 | 87.67 | 70.00 | 86.87 | 88.82 | 88.95 | 89.87 | 88.28 | 93.81 | 93.37 | 78.42 | 83.86 | 78.90 | 84.68 |
|   CIFAR100-C | Source | 35.52 | 37.69 | 50.30 | 50.20 | 25.98 | 52.58 | 57.15 | 54.87 | 58.85 | 49.63 | 67.13 | 53.50 | 36.43 | 44.82 | 42.79 | 47.83 |
|  | TDA | 40.09 | 42.74 | 52.27 | 50.96 | 26.13 | 53.67 | 58.55 | 56.48 | 60.12 | 50.90 | 68.78 | 55.08 | 37.23 | 46.46 | 46.37 | 49.72 |
|  | DMN | 38.07 | 41.97 | 51.01 | 46.36 | 22.71 | 53.35 | 58.83 | 50.67 | 55.05 | 50.17 | 65.81 | 52.00 | 35.73 | 45.65 | 42.54 | 47.33 |
|  | VTE | 36.31 | 39.04 | 48.97 | 50.43 | 26.06 | 51.30 | 55.67 | 58.59 | 60.31 | 47.37 | 66.88 | 51.76 | 41.59 | 49.06 | 44.67 | 48.53 |
|  | ZERO | 35.70 | 37.36 | 52.04 | 53.51 | 26.63 | 54.08 | 58.97 | 57.54 | 60.92 | 51.68 | 69.08 | 54.92 | 39.35 | 46.04 | 45.72 | 49.57 |
|  | ECALP | 40.27 | 42.16 | 53.61 | 51.81 | 27.91 | 55.20 | 59.93 | 58.24 | 61.27 | 52.04 | 70.25 | 56.14 | 38.46 | 46.98 | 46.69 | 50.73 |
|  | RoTTA | 36.03 | 38.46 | 51.49 | 50.42 | 26.51 | 52.62 | 57.25 | 54.64 | 58.97 | 49.58 | 66.94 | 53.53 | 36.84 | 45.90 | 43.22 | 48.16 |
|  | TPT | 38.53 | 40.29 | 51.86 | 51.54 | 26.98 | 53.05 | 57.86 | 58.52 | 61.20 | 49.99 | 68.27 | 52.75 | 40.30 | 47.36 | 45.73 | 49.62 |
|  | MEMO | 36.28 | 38.44 | 51.69 | 51.60 | 26.67 | 53.43 | 58.14 | 56.22 | 60.04 | 50.38 | 68.19 | 53.89 | 37.61 | 45.60 | 44.23 | 48.83 |
|  | WATT | 42.07 | 43.50 | 53.95 | 55.36 | 30.25 | 56.28 | 61.47 | 59.15 | 61.15 | 54.11 | 70.24 | 60.34 | 40.63 | 48.51 | 47.69 | 52.31 |
|  | MINT | 43.26 | 46.02 | 54.94 | 57.24 | 32.89 | 55.69 | 62.82 | 61.85 | 61.02 | 54.81 | 73.72 | 62.42 | 40.73 | 51.84 | 47.69 | 53.80 |
|  | BATCLIP | 40.11 | 43.10 | 53.47 | 54.79 | 30.58 | 53.30 | 59.77 | 57.27 | 58.60 | 50.18 | 70.44 | 60.87 | 35.94 | 47.77 | 47.17 | 50.89 |
|  | SubTTA | 41.36 | 44.54 | 57.17 | 57.67 | 34.39 | 55.84 | 62.35 | 60.85 | 61.72 | 54.93 | 72.27 | 64.54 | 41.19 | 50.81 | 48.01 | 53.84 |

Table 5: Benchmark results with ViT-L-14. We denote Top-1/2/3 by
Blue/Yellow/Red, respectively.

(1) Consistent SOTA performance across model capacities. SubTTA
maintains its superiority regardless of the backbone architecture. On
the lower-capacity ViT-B-32, SubTTA achieves a mean accuracy of 30.50%
on ImageNet-C, outperforming the runner-up BATCLIP (27.37%) by a
substantial margin of 3.13%. On the stronger ViT-L-14, SubTTA continues
to set the state-of-the-art with 44.55% mean accuracy. This confirms
that our geometric rectification is effective for both rescuing weaker
models and refining stronger ones.

(2) Elimination of negative adaptation. Similar to the ViT-B-16
settings, baseline methods exhibit instability under heavy noise. For
instance, on ViT-B-32, TENT suffers from negative adaptation on Gaussian
Noise (12.16% vs. Source 12.90%). In contrast, SubTTA consistently
improves over the source baseline across all corruption categories for
both backbones, demonstrating robust stability against severe
distribution shifts.

(3) Validation of the paradigm hierarchy. The performance hierarchy
observed in the main text holds true across different backbones:
training-based methods generally outperform training-free ones, and
cluster-based objectives (e.g., MINT, BATCLIP) consistently surpass
entropy-based approaches (e.g., TENT). SubTTA builds upon the robust
cluster-based paradigm and further elevates it via subspace alignment,
yielding the most reliable adaptation performance.

### C.2 Robustness against Shift Levels

We evaluate how model performs under different shift levels, and the
results are shown in
Figure [11](#A3.F11 "Figure 11 ‣ C.2 Robustness against Shift Levels ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
First, we observe a consistent performance degradation in all baseline
methods (colored bars) as severity increases, confirming the
vulnerability of pre-trained CLIP models to severe distribution shifts.
Second, the efficacy of SubTTA correlates positively with shift
severity. While the improvements are consistent at lower levels, the
performance boost (hatched bars) becomes more significant at higher
levels. This indicates that SubTTA is particularly critical in extreme
scenarios, effectively acting as a safeguard where standard baselines
struggle most.

![Refer to caption](2601.08139v2/figures/severity.png)

Figure 11: TTA performance under different shift levels.

### C.3 Effect of Exponential Moving Average

We carry out experiments to understand the role of exponential moving
average (EMA) in statistical estimation of SubTTA. While an EMA may
theoretically reduce the immediate responsiveness of test-time
adaptation, it is a crucial component in SubTTA for ensuring smooth and
reliable subspace estimation. Under challenging conditions, such as
extremely small batch sizes or datasets with a massive number of
classes, a single batch of test samples may fail to cover the global
data distribution comprehensively. Relying exclusively on such isolated
batches can result in ill-conditioned or heavily biased subspace
representations, which significantly degrades overall performance.

To empirically validate this, we conduct an ablation study by removing
the EMA module. Specifically, we set the update rate to $`\alpha=1`$ in
Eq. ([1](#S3.E1 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")),
forcing the model to rely solely on the current batch statistics for
subspace computation. As demonstrated in
Table [6](#A3.T6 "Table 6 ‣ C.3 Effect of Exponential Moving Average ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
discarding EMA leads to a consistent and noticeable performance drop
across benchmarks, confirming that historical statistical aggregation is
indispensable.

Fundamentally, the update hyperparameter $`\alpha`$ governs the
trade-off between responsiveness (favoring the current batch) and
robustness (leveraging historical context). Based on empirical
observations, an appropriate $`\alpha`$ can be selected based on the
following principles:

- •
  Favor a smaller $`\alpha`$ (More Robustness): In scenarios
  characterized by small batch sizes, a large number of classes, or
  relatively stationary test distributions, a smaller $`\alpha`$ is
  preferred. This retains more historical information, ensuring stable
  statistical estimation despite high per-batch variance.
- •
  Favor a larger $`\alpha`$ (More Responsiveness): Conversely, in
  settings with large batch sizes, a compact label space, or rapidly
  changing, non-stationary test streams, a larger $`\alpha`$ allows the
  model to swiftly adapt to new distributions by relying more on recent
  batch statistics.

| Dataset     |            | Source | TENT  | BATCLIP | MINT  |
|-------------|------------|--------|-------|---------|-------|
|  ImageNet   | w/o EMA    | 24.51  | 21.03 | 31.23   | 31.43 |
|             | w/ EMA     | 25.87  | 24.98 | 31.90   | 32.15 |
|             | $`\Delta`$ | +1.36  | +3.95 | +0.67   | +0.72 |
|  CIFAR10    | w/o EMA    | 61.15  | 61.30 | 75.48   | 72.95 |
|             | w/ EMA     | 68.87  | 68.18 | 77.79   | 76.33 |
|             | $`\Delta`$ | +7.72  | +6.88 | +2.31   | +3.38 |
|  CIFAR100   | w/o EMA    | 35.78  | 35.41 | 41.67   | 42.88 |
|             | w/ EMA     | 40.12  | 41.13 | 43.87   | 44.49 |
|             | $`\Delta`$ | +4.34  | +5.72 | +2.20   | +1.61 |

Table 6: Study on exponential moving average (EMA).

### C.4 Sensitivity to Prompt Design

To understand SubTTA’s reliance on the textual anchor, e.g., when
textual prompts are noisy or incomplete, we conduct experiments to
analyze SubTTA’s sensitivity to different prompt designs. Specifically,
we consider the following 8 prompts, and evaluate VLM performance with
and without SubTTA.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTMuU1M0LnAyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIzMDQuMjciIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA2NTMuMTUgMzA0LjI3IiB3aWR0aD0iNjUzLjE1Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDMwNC4yNykgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiM0MDQwNDA7IiBmaWxsPSIjNDA0MDQwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNS45MSBMIDAgMjk4LjM3IEMgMCAzMDEuNjMgMi42NCAzMDQuMjcgNS45MSAzMDQuMjcgTCA2NDcuMjQgMzA0LjI3IEMgNjUwLjUxIDMwNC4yNyA2NTMuMTUgMzAxLjYzIDY1My4xNSAyOTguMzcgTCA2NTMuMTUgNS45MSBDIDY1My4xNSAyLjY0IDY1MC41MSAwIDY0Ny4yNCAwIEwgNS45MSAwIEMgMi42NCAwIDAgMi42NCAwIDUuOTEgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0Y5RjlGOTsiIGZpbGw9IiNGOUY5RjkiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMS45NyA1LjkxIEwgMS45NyAyODAuMTYgTCA2NTEuMTggMjgwLjE2IEwgNjUxLjE4IDUuOTEgQyA2NTEuMTggMy43MyA2NDkuNDIgMS45NyA2NDcuMjQgMS45NyBMIDUuOTEgMS45NyBDIDMuNzMgMS45NyAxLjk3IDMuNzMgMS45NyA1LjkxIFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMTAuMDYgMjg4Ljc2KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjQ1Ljc1ZW07LS1sdHgtZm8taGVpZ2h0OjAuNjllbTstLWx0eC1mby1kZXB0aDowLjE5ZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjEyLjMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDkuNjEpIiB3aWR0aD0iNjMzLjA1Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMy5TUzQucDIucGljMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjQ1Ljc1ZW07Ij4KPHNwYW4gaWQ9IkEzLlNTNC5wMi5waWMxLjEuMSIgY2xhc3M9Imx0eF9wIj48c3BhbiBpZD0iQTMuU1M0LnAyLnBpYzEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkZGRkY7Ij5Qcm9tcHQgVGVtcGxhdGVzPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDEwLjA2IDEzLjUyKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjQ1Ljc1ZW07LS1sdHgtZm8taGVpZ2h0OjE4LjY5ZW07LS1sdHgtZm8tZGVwdGg6MC4yNWVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIyNjIuMDIiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDI1OC41NikiIHdpZHRoPSI2MzMuMDUiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkEzLlNTNC5wMi5waWMxLjIiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6NDUuNzVlbTsiPgo8c3BhbiBpZD0iQTMuU1M0LnAyLnBpYzEuMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMy5TUzQucDIucGljMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+VDA8c3BhbiBpZD0iQTMuU1M0LnAyLnBpYzEuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSI+OiBBIHBob3RvIG9mIGEge2NsYXNzfQo8YnIgY2xhc3M9Imx0eF9icmVhayI+PC9zcGFuPlQxPHNwYW4gaWQ9IkEzLlNTNC5wMi5waWMxLjIuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPjogaXRhcCBvZiBhIHtjbGFzc30KPGJyIGNsYXNzPSJsdHhfYnJlYWsiPjwvc3Bhbj5UMjxzcGFuIGlkPSJBMy5TUzQucDIucGljMS4yLjEuMS4zIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfbWVkaXVtIj46IGEgYmFkIHBob3RvIG9mIHRoZSB7Y2xhc3N9CjxiciBjbGFzcz0ibHR4X2JyZWFrIj48L3NwYW4+VDM8c3BhbiBpZD0iQTMuU1M0LnAyLnBpYzEuMi4xLjEuNCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSI+OiBhIG9yaWdhbWkge2NsYXNzfQo8YnIgY2xhc3M9Imx0eF9icmVhayI+PC9zcGFuPlQ0PHNwYW4gaWQ9IkEzLlNTNC5wMi5waWMxLjIuMS4xLjUiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPjogYSBwaG90byBvZiB0aGUgbGFyZ2Uge2NsYXNzfQo8YnIgY2xhc3M9Imx0eF9icmVhayI+PC9zcGFuPlQ1PHNwYW4gaWQ9IkEzLlNTNC5wMi5waWMxLjIuMS4xLjYiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPjogYSB7Y2xhc3N9IGluIGEgdmlkZW8gZ2FtZQo8YnIgY2xhc3M9Imx0eF9icmVhayI+PC9zcGFuPlQ2PHNwYW4gaWQ9IkEzLlNTNC5wMi5waWMxLjIuMS4xLjciIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9tZWRpdW0iPjogYXJ0IG9mIHRoZSB7Y2xhc3N9CjxiciBjbGFzcz0ibHR4X2JyZWFrIj48L3NwYW4+VDc8c3BhbiBpZD0iQTMuU1M0LnAyLnBpYzEuMi4xLjEuOCIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X21lZGl1bSI+OiBhIHBob3RvIG9mIHRoZSBzbWFsbCB7Y2xhc3N9PC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==)

| Type | ImageNet-C |  |  | CIFAR10-C |  |  | CIFAR100-C |  |  |
|----|----|----|----|----|----|----|----|----|----|
|  | Source | SubTTA | $`\Delta`$ | Source | SubTTA | $`\Delta`$ | Source | SubTTA | $`\Delta`$ |
| T0 | 24.54 | 25.87 | +1.33 | 61.16 | 68.87 | +7.71 | 35.78 | 39.98 | +4.20 |
| T1 | 23.61 | 25.57 | +1.96 | 62.05 | 63.86 | +1.81 | 36.26 | 40.64 | +4.38 |
| T2 | 25.28 | 26.53 | +1.25 | 61.19 | 68.80 | +7.61 | 34.74 | 39.05 | +4.31 |
| T3 | 21.38 | 22.26 | +0.88 | 60.14 | 70.09 | +9.95 | 32.25 | 35.80 | +3.55 |
| T4 | 25.25 | 25.98 | +0.73 | 60.43 | 68.05 | +7.62 | 34.49 | 38.19 | +3.70 |
| T5 | 24.10 | 25.29 | +1.19 | 64.21 | 65.41 | +1.20 | 34.42 | 38.70 | +4.28 |
| T6 | 23.25 | 23.95 | +0.70 | 61.33 | 68.94 | +7.61 | 33.27 | 37.28 | +4.01 |
| T7 | 24.78 | 26.01 | +1.23 | 61.45 | 69.03 | +7.58 | 34.62 | 38.66 | +4.04 |

Table 7: Sensitivity analysis with different prompt designs. SubTTA is
robust to different prompt templates, consistently enhancing the
baseline performance.

As the results in
Table [7](#A3.T7 "Table 7 ‣ C.4 Sensitivity to Prompt Design ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
shows, SubTTA consistently enhances baseline performance across all
datasets and prompt templates, with average improvement of 1.16% on
ImageNet-C, 6.39% on CIFAR10-C and 4.06% on CIFAR100-C. Depite strong
stylistic or domain biases of different prompts, SubTTA maintains steady
improvements even under these conditions. This indicates that as long as
the prompt spans a valid semantic category, SubTTA can successfully
filter out orthogonal visual corruptions regardless of the stylistic
anchor.

### C.5 Sensitivity to the Number of Classes

The number of classes in the target domain plays a critical role in the
quality of subspace estimation [Zanella et al. (2025)](#bib.bib40), as a
larger semantic space inherently introduces greater complexity and
potential for misaligned feature projections. To investigate the
robustness of our approach, we evaluate SubTTA on ImageNet-C by sampling
subsets of classes, with the total number of classes varying from 100 to
1,000.

As illustrated in
Table [8](#A3.T8 "Table 8 ‣ C.5 Sensitivity to the Number of Classes ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"),
we observe two key findings. First, SubTTA exhibits remarkable
robustness to the expansion of the label space. While the absolute
accuracy of all methods naturally decays as the classification task
becomes more challenging (with \#classes from 100 to 1,000), the
performance of SubTTA scales gracefully without catastrophic
degradation. Second, and more importantly, SubTTA enhances the
performance of the baseline TTA method across all class regimes in most
cases (38 out of 40). Notably, the performance gap between MINT equipped
with SubTTA and the vanilla MINT baseline remains stable, and even
becomes more pronounced, at larger class counts (e.g., 500 to 1,000
classes). This demonstrates that our geometric filtering mechanism
effectively purifies representations even when the semantic space is
highly dense and complex.

| \# Class |            | 100   | 200   | 300   | 400   | 500   | 600   | 700   | 800   | 900   | 1000  |
|----------|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Source   | Vanilla    | 25.35 | 25.52 | 25.66 | 25.20 | 25.29 | 25.13 | 24.88 | 24.55 | 24.38 | 24.54 |
|          | SubTTA     | 26.23 | 26.62 | 27.05 | 26.41 | 26.74 | 26.73 | 26.57 | 26.21 | 25.59 | 25.87 |
|          | $`\Delta`$ | +0.88 | +1.10 | +1.39 | +1.21 | +1.45 | +1.60 | +1.69 | +1.66 | +1.21 | +1.33 |
| TENT     | Vanilla    | 25.47 | 25.91 | 26.28 | 25.82 | 26.05 | 25.95 | 25.59 | 25.23 | 25.03 | 25.18 |
|          | SubTTA     | 26.30 | 26.77 | 27.13 | 26.60 | 26.99 | 26.64 | 26.04 | 25.53 | 25.06 | 24.98 |
|          | $`\Delta`$ | +0.83 | +0.86 | +0.85 | +0.78 | +0.94 | +0.69 | +0.45 | +0.30 | +0.03 | -0.20 |
| MINT     | Vanilla    | 29.70 | 30.27 | 30.49 | 29.94 | 30.11 | 29.81 | 29.47 | 29.14 | 28.99 | 30.06 |
|          | SubTTA     | 28.64 | 31.07 | 32.33 | 32.47 | 32.81 | 32.58 | 32.51 | 32.27 | 31.99 | 32.15 |
|          | $`\Delta`$ | -1.06 | +0.80 | +1.84 | +2.53 | +2.70 | +2.77 | +3.04 | +3.13 | +3.00 | +2.09 |
| BATCLIP  | Vanilla    | 27.16 | 29.01 | 30.15 | 30.25 | 30.77 | 30.62 | 30.37 | 30.04 | 29.90 | 29.20 |
|          | SubTTA     | 28.64 | 30.97 | 32.26 | 32.38 | 32.76 | 32.48 | 32.41 | 32.13 | 31.78 | 31.90 |
|          | $`\Delta`$ | +1.48 | +1.96 | +2.11 | +2.13 | +1.99 | +1.86 | +2.04 | +2.09 | +1.88 | +2.70 |

Table 8: Sensitivity analysis to the number of classes. We sample
subsets from the ImageNet-C dataset with the number of classes varying
from 100 to 1,000.

### C.6 Latency Analysis

We provide a latency analysis of the proposed SubTTA. Theoretically, let
$`B`$ denote the batch size, $`d`$ the feature dimension, $`r`$ the
subspace rank, and $`C`$ the number of classes. The dominant
computational overhead of SubTTA arises from the subspace computation
(Eq. ([2](#S3.E2 "In 3.2 Geometric Alignment ‣ 3 Methodology ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation"))),
which is bounded by $`O(d^{2}r)`$. In contrast, baseline TTA methods,
e.g., MINT and BATCLIP, the cluster-based TTA loss incurs a
computational complexity of $`O(BCd)`$.

Under practical deployment scenarios, these two complexities are of
comparable magnitude. For typical VLM configurations, the hidden
dimension is $`d\in[512,1024]`$, the subspace rank is $`r\ll d`$, the
batch size is $`B\in[64,256]`$, and the number of classes $`C`$ ranges
from hundreds to over a thousand. Crucially, the complexity of SubTTA is
independent of the number of classes $`C`$, whereas cluster-based TTA
methods scale linearly with $`C`$. This property renders SubTTA highly
scalable and particularly well-suited for settings with a large
vocabulary space.

Empirically, we evaluate the runtime efficiency across the three
benchmark datasets under default configurations.
Table [9](#A3.T9 "Table 9 ‣ C.6 Latency Analysis ‣ Appendix C Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
reports the per-batch latency (in seconds) attributed to the subspace
operations ($`t_{\text{Sub}}`$) and the subsequent adaptation process
($`t_{\text{TTA}}`$). As demonstrated,$`t_{\text{Sub}}`$ and
$`t_{\text{TTA}}`$ are comparable, indicating that the subspace
projection does not introduce a substantial per-batch latency increase.
These findings confirm that the geometric filtering mechanism in SubTTA
imposes a marginal computational burden, maintaining overall efficiency
relative to existing TTA routines.

| Time | ImageNet-A |  |  | ImageNet-R |  |  | ImageNet-K |  |  |
|----|----|----|----|----|----|----|----|----|----|
|  | TENT | BATCLIP | MINT | TENT | BATCLIP | MINT | TENT | BATCLIP | MINT |
| $`t_{\text{Sub}}`$ | 0.16 | 0.17 | 0.15 | 0.16 | 0.16 | 0.15 | 0.15 | 0.16 | 0.15 |
| $`t_{\text{TTA}}`$ | 0.09 | 0.14 | 0.15 | 0.09 | 0.15 | 0.16 | 0.10 | 0.15 | 0.15 |

Table 9: Running time (seconds) of two stages: SubTTA
($`t_{\text{Sub}}`$) and standard TTA ($`t_{\text{TTA}}`$).

## Appendix D Datasets

We first introduce three corrupted datasets, including ImageNet-C,
CIFAR10-C, and CIFAR100-C. ImageNet-C [Hendrycks and Dietterich
(2019)](#bib.bib27) is a robustness benchmark derived from ImageNet. It
spans 1,000 object classes and contains 50,000 test images for each
corruption type and severity level. CIFAR-10-C [Hendrycks and Dietterich
(2019)](#bib.bib27) is constructed from the CIFAR-10. It comprises
10,000 images distributed across 10 distinct classes.
CIFAR-100-C [Hendrycks and Dietterich (2019)](#bib.bib27) is an
extension of the CIFAR-100, containing 10,000 images covering 100
fine-grained classes.

All three datasets employ a shared set of 15 algorithmically generated
corruptions to assess model robustness under distribution shifts. These
corruptions are categorized into four primary groups: noise, blur,
weather, and digital distortions, and each is applied at five severity
levels. The noise category includes Gaussian, shot, and impulse noise,
which introduce random pixel-level variations. The blur category
encompasses defocus, glass, motion, and zoom blur, simulating various
optical distortions. Weather-related corruptions, such as snow, frost,
and fog, replicate environmental conditions that obscure image details.
Lastly, digital distortions include brightness, contrast, elastic
transform, pixelate, and JPEG compression, reflecting common
post-processing or compression artifacts.

Besides, we also consider three datasets with realistic perturbations,
including ImageNet-A, ImageNet-R and ImageNet-K. ImageNet-A [Hendrycks
et al. (2021b)](#bib.bib32) is a challenging out-of-distribution
benchmark containing natural, unperturbed images that commonly cause
classification errors in standard ResNet models. It spans 200 object
classes sampled from the original ImageNet and consists of 7,500 test
images. ImageNet-R [Hendrycks et al. (2021a)](#bib.bib33) is constructed
to evaluate model robustness to abstract visual renditions. It comprises
around 30,000 images distributed across 200 distinct ImageNet classes.
ImageNet-K [Wang et al. (2019)](#bib.bib34) is a domain shift benchmark
constructed from sketch-like images. It contains 50,000 images covering
the full 1,000 fine-grained classes of the original ImageNet.

## Appendix E More Related Works

We provide more related works on pre-trained foundation models, domain
adaptation and cross-domain alignment.

##### Pre-trained Foundation Models.

Foundation models have reshaped the field by pretraining models on vast
amounts of web-scale data [Achiam et al. (2023)](#bib.bib95); [Touvron
et al. (2023)](#bib.bib96); [Lin et al. (2025a)](#bib.bib73); [Ai et al.
(2025)](#bib.bib77); [Cui et al. (2026)](#bib.bib78); [Xu et al.
(2026)](#bib.bib105). Large Language Models (LLMs) have demonstrated
remarkable generalization capabilities with various applications in
natural language understanding [Touvron et al. (2023)](#bib.bib96);
[Zhang et al. (2025)](#bib.bib74); [Lin et al. (2026a)](#bib.bib103),
numerical analysis [Li et al. (2026)](#bib.bib98); [Jing et al.
(2026)](#bib.bib97); [Lin et al. (2026d)](#bib.bib101), and ethics
evaluation [Lin et al. (2025b)](#bib.bib102); [Lin et al.
(2026c)](#bib.bib44). Parallel to this success, encoder-decoder VLMs
such as CLIP [Radford et al. (2021)](#bib.bib19) and ALIGN [Jia et al.
(2021)](#bib.bib20) extend this paradigm to the visual domain by
aligning images and text in a shared embedding space via contrastive
learning. This joint training enables powerful zero-shot transfer to
downstream tasks without task-specific fine-tuning. Decoder-only VLMs
like Flamingo [Alayrac et al. (2022)](#bib.bib84), BLIP-2 [Li et al.
(2023)](#bib.bib83), and LLaVA [Liu et al. (2023)](#bib.bib82) leverage
visual instruction tuning to project visual features into the input
space of frozen LLMs, unlocking capabilities for complex multimodal
understanding and dialogue. Despite their impressive capabilities, these
models remain vulnerable to distribution shifts when deployed in
open-world environments, and significant efforts have been made on
improving the robustness of LLMs and VLMs [Qi et al.
(2024)](#bib.bib90); [Wang et al. (2023a)](#bib.bib86); [Yan et al.
(2025)](#bib.bib60); [Zeng et al. (2025d)](#bib.bib45); [Lin et al.
(2024)](#bib.bib72); [Lin et al. (2026b)](#bib.bib104), necessitating
effective adaptation strategies to bridge the gap between pre-training
and testing stages.

##### Domain Adaptation.

The challenge of generalizing models to out-of-distribution data has
evolved through increasingly constrained settings, spanning from
label-scarce scenarios to fully unsupervised and source-free
environments.

*(Semi-)supervised domain adaptation (SSDA)* represents the most
accessible settings, assuming the availability of at least a few labeled
samples in the target domain [Motiian et al. (2017)](#bib.bib91); [Saito
et al. (2019)](#bib.bib92). Approaches typically leverage the limited
target labels to perform fine-tuning or align class-conditional
distributions via metric learning [Kang et al. (2019)](#bib.bib85) and
minimax entropy training [Saito et al. (2019)](#bib.bib92). However, the
reliance on target annotation, even if minimal, limits their scalability
in open-world deployments.

*Unsupervised Domain Adaptation (UDA)* removes the reliance on target
labels, assuming access to both labeled source data and fully unlabeled
target data. Methods in this realm can be broadly categorized into two
streams. One prominent line of works employ adversarial learning [Ganin
and Lempitsky (2015)](#bib.bib80), optimizing a domain discriminator to
force the feature extractor to learn domain-invariant representations.
Another line of works minimize the discrepancy between source and target
distributions such as maximum mean discrepancy [Long et al.
(2015)](#bib.bib81); [Yan et al. (2017)](#bib.bib89), Wasserstein
distance [Shen et al. (2018)](#bib.bib87); [Zeng et al.
(2025c)](#bib.bib41) and correlation alignment [Sun and Saenko
(2016)](#bib.bib38); [Sun et al. (2017)](#bib.bib88); [Lin et al.
(2025c)](#bib.bib43).

*Test-time adaptation (TTA)* faces the most challenging setting,
imposing strict latency and online constraints where data arrives in
streams [Wang et al. (2020)](#bib.bib1); [Yu et al.
(2026b)](#bib.bib99). Existing TTA strategies generally diverge into
three optimization categories: First (entropy minimization [Wang et al.
(2020)](#bib.bib1); [Zhang et al. (2022)](#bib.bib9)), which sharpens
prediction distributions to reduce uncertainty; Second (self-training
with pseudo-labels [Rusak et al. (2022)](#bib.bib12); [Bao et al.
(2024)](#bib.bib42); [Yang et al. (2024)](#bib.bib57)), which utilizes
robust loss functions or cluster structures to refine decision
boundaries. Third (parameter-efficient tuning [Shu et al.
(2022)](#bib.bib11)), which updates only specific modules (e.g., prompts
or normalization layers) to prevent catastrophic forgetting.

##### Cross-Domain Alignment.

To mitigate the discrepancy between domains, various alignment
strategies [Yan et al. (2021a)](#bib.bib47); [Yan et al.
(2021b)](#bib.bib48); [Zeng et al. (2023a)](#bib.bib61); [Zeng et al.
(2024a)](#bib.bib62); [Yu et al. (2025b)](#bib.bib58); [Yu et al.
(2025a)](#bib.bib59) have been proposed to learn domain-invariant
representations [Wang et al. (2018)](#bib.bib46); [Wang et al.
(2023b)](#bib.bib49); [Wang et al. (2023c)](#bib.bib55), with
applications in various data modalities such as image [Zhu et al.
(2019)](#bib.bib66); [Xu et al. (2020)](#bib.bib70); [Yu et al.
(2026a)](#bib.bib100), text [Liu et al. (2021)](#bib.bib67); [Chen et
al. (2020)](#bib.bib69), graphs [Yan et al. (2023a)](#bib.bib50); [Yan
et al. (2023b)](#bib.bib51); [Yan et al. (2024b)](#bib.bib53); [Yan et
al. (2024a)](#bib.bib56); [Xu et al. ()](#bib.bib54); [Zeng et al.
(2023b)](#bib.bib63); [Zeng et al. (2024b)](#bib.bib52); [Chen et al.
(2026)](#bib.bib106), and recommendation [Zhao et al.
(2023a)](#bib.bib71); [Zhao et al. (2023b)](#bib.bib68); [Zeng et al.
(2025b)](#bib.bib65); [Zeng et al. (2025a)](#bib.bib64); [Yoo et al.
(2024)](#bib.bib75); [Liu et al. (2024)](#bib.bib79); [Liang et al.
(2025)](#bib.bib76). One prevalent approach is statistical moment
matching, which explicitly minimizes the distance between feature
distributions [Long et al. (2015)](#bib.bib81); [Sun and Saenko
(2016)](#bib.bib38). Another significant direction is manifold
alignment, which posits that domain shifts can be modeled as geometric
transformations [Fernando et al. (2013)](#bib.bib93); [Gong et al.
(2012)](#bib.bib94). While these classical methods laid the theoretical
groundwork, they typically require offline processing or access to
source data. SubTTA revitalizes these geometric principles, adapting the
manifold alignment concept to the challenging online, source-free TTA
setting by utilizing the textual subspace as a stable semantic anchor.

## Appendix F Potential Risks

The proposed SubTTA, while enhancing the robustness of VLMs under
distribution shifts, relies on statistical properties of test data
streams. SubTTA estimates covariance matrices from incoming test batches
to perform subspace alignment. As analyzed in our hyperparameter studies
(Section [4.5.4](#S4.SS5.SSS4 "4.5.4 Hyperparameter Study ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")),
extremely small batch sizes (e.g., $`B=16`$) may result in statistically
unreliable covariance estimation, leading to performance fluctuations.
Additionally, like most TTA methods, our approach requires access to the
model’s visual and textual features during inference, which may limit
applicability in strictly black-box API scenarios where feature
embeddings are inaccessible.

## Appendix G Use Or Create Scientific Artifacts

Our work is built on established public benchmarks and pre-trained
models. For evaluation, we use widely adopted corrupted image
classification datasets including CIFAR-10-C, CIFAR-100-C, and
ImageNet-C. We do not modify the original content of these datasets. We
utilize pre-trained CLIP models (ViT-B-16, ViT-B-32, ViT-L-14) as the
backbone. We will release the codebase of SubTTA upon publication to
facilitate reproducibility and future research.

### G.1 Cite Creators Of Artifacts

All external artifacts are properly credited to their original
publications and repositories.

The benchmarks used in this work, including CIFAR-10-C and
CIFAR-100-C [Hendrycks and Dietterich (2019)](#bib.bib27) (based on
CIFAR [Krizhevsky et al. (2009)](#bib.bib28)), and ImageNet-C [Hendrycks
and Dietterich (2019)](#bib.bib27) (based on ImageNet [Deng et al.
(2009)](#bib.bib29)), are credited to their respective authors.

The CLIP [Radford et al. (2021)](#bib.bib19) backbones used in this
work, including ViT-B-16. ViT-B-32, ViT-L-14, are referenced to its
official publication.

### G.2 Discuss The License For Artifacts

We comply with the licenses of all artifacts used in this work. The
datasets (CIFAR-10-C, CIFAR-100-C, ImageNet-C) are available for
research purposes under their respective licenses (typically Creative
Commons or similar non-commercial research licenses). The CLIP models
are released under the MIT License. Our own code will be released under
a compatible open-source license (e.g., MIT).

### G.3 Artifact Use Consistent With Intended Use

We confirm that our use of datasets and pre-trained models is consistent
with their intended purpose. The corrupted datasets are specifically
designed for benchmarking model robustness, which aligns directly with
our goal of evaluating Test-Time Adaptation. The pre-trained CLIP models
are used for zero-shot classification and feature extraction as intended
by their creators.

### G.4 Data Contains Personally Identifying Info Or Offensive Content

The datasets used (CIFAR-10-C, CIFAR-100-C, ImageNet-C) are standard
computer vision benchmarks derived from public object classification
datasets. To our knowledge, they do not contain sensitive personally
identifying information or offensive content beyond what exists in the
standard ImageNet/CIFAR distributions, which are widely accepted in the
community.

### G.5 Documentation Of Artifacts

We provide detailed documentation of all evaluation datasets and model
combinations used in our experiments. Specifically,
Appendix [D](#A4 "Appendix D Datasets ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
describes the datasets statistics, while
Appendix [I](#A9 "Appendix I Computational Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
discusses the VLMs included along with their sizes and sources.
Together, these materials ensure transparency regarding the domains,
data characteristics, and model diversity involved in our study.

## Appendix H Statistics For Data

Dataset statistics are summarized as follows based on the details in
Appendix [D](#A4 "Appendix D Datasets ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

### H.1 ImageNet-C

- •
  Number of classes: 1,000 object classes.
- •
  Size: 50,000 test images for each corruption type and severity level.
- •
  Structure: Derived from ImageNet validation set with 15
  algorithmically generated corruptions.

### H.2 CIFAR-10-C

- •
  Number of classes: 10 distinct classes.
- •
  Size: 10,000 test images.
- •
  Structure: Derived from CIFAR-10 test set with 15 algorithmically
  generated corruptions.

### H.3 CIFAR-100-C

- •
  Number of classes: 100 fine-grained classes.
- •
  Size: 10,000 test images.
- •
  Structure: Derived from CIFAR-100 test set with 15 algorithmically
  generated corruptions.

### H.4 ImageNet-A

- •
  Number of classes: 200 distinct classes.
- •
  Size: 7,500 test images.
- •
  Structure: Composed of naturally occurring adversarial examples
  (real-world, unmodified images) that standard ResNet models
  consistently misclassify, designed to evaluate model robustness.

### H.5 ImageNet-R

- •
  Number of classes: 200 distinct classes.
- •
  Size: 30,000 test images.
- •
  Structure: Contains various artistic renditions of ImageNet objects,
  including art, cartoons, paintings, origami, toys, and video game
  graphics, to test out-of-distribution generalization to different
  textures and styles.

### H.6 ImageNet-K

- •
  Number of classes: 1,000 object classes.
- •
  Size: 50,000 test images.
- •
  Structure: Consists of black-and-white sketch images corresponding to
  the original ImageNet classes, collected via web search to evaluate
  model robustness to severe domain shifts.

## Appendix I Computational Experiments

All computational experiments in this work are fully reproducible, with
details provided in
Section [4.1](#S4.SS1 "4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").

### I.1 Model Size And Budget

We evaluate our method on CLIP models with varying capacities,
including:

- •
  ViT-B-16: 112M total parameters, and 41k trainable parameters.
- •
  ViT-B-32: 113M total parameters, and 41k trainable parameters.
- •
  ViT-L-14: 343M total parameters, and 104k trainable parameters.

All experiments are executed on NVIDIA A100 80GB GPUs.

### I.2 Experimental Setup And Hyper-params

We describe experimental settings in
Section [4.1](#S4.SS1 "4.1 Experiment Setup ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation").
We adopt an online adaptation setting where the model adapts to a
continuous stream of unlabeled test data for each corruption type
independently (resetting between corruptions). Key hyperparameters
studied in
Section [4.5.4](#S4.SS5.SSS4 "4.5.4 Hyperparameter Study ‣ 4.5 Studies ‣ 4 Experiments ‣ Subspace Alignment for Vision-Language Model Test-time Adaptation")
include:

- •
  Subspace rank $`r`$: We vary $`r`$ (e.g., 64, 128, 256) and find that
  avoiding extremes (too compressed or full-rank) is optimal.
- •
  Momentum coefficient $`\alpha`$: We use an EMA strategy for covariance
  updates, with stability observed for $`\alpha\in(0.2,0.8)`$.
- •
  Batch Size: We investigate the impact of batch size, and results
  indicate that larger batch sizes (e.g., 64 and above) provide stable
  covariance estimation and consistent performance, whereas smaller
  batches (e.g., 16) may lead to instability.

### I.3 Descriptive Statistics

We report the classification accuracy as the evaluation metric. We
report the mean performance across 15 corruption types at severity level
5.

### I.4 Parameters For Packages

The existing packages used are specified as follows. We use PyTorch
(v2.2.1) as the core deep learning framework, together with
open-clip-torch (v2.24.0) and huggingface-hub (v0.21.4) for model
implementation and text pre-processing.

## Appendix J AI Assistants In Research Or Writing

In this paper, AI assistant tool is used to edit and improve the quality
of the text, including checking the spelling, grammar, punctuation and
clarity.
````
