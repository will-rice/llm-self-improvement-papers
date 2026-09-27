---
identifier: arxiv:2511.19046
title: "MedSAM3: Delving into Segment Anything with Medical Concepts"
authors:
  - Anglin Liu
  - Rundong Xue
  - Xu R. Cao
  - Yifan Shen
  - Yi Lu
  - Xiang Li
  - Qianqian Chen
  - Jintai Chen
published: "2025-11-24T12:34:38+00:00"
url: https://huggingface.co/papers/2511.19046
source: huggingface
doi: null
arxiv_id: "2511.19046"
categories: []
---

# MedSAM3: Delving into Segment Anything with Medical Concepts

Anglin Liu Affiliation:  The Hong Kong University of Science and
Technology (Guangzhou)    Xu R. Cao Affiliation:  University of Illinois
Urbana-Champaign    Yifan Shen Affiliation:  University of Illinois
Urbana-Champaign    Yi Lu Affiliation:  The Hong Kong University of
Science and Technology (Guangzhou)    Xiang Li Affiliation:  University
of Illinois Urbana-Champaign    Qianqian Chen Affiliation:  Southeast
University    Jintai Chen ^(†)^(†)thanks: Corresponding to:
[jintaiCHEN@hkust-gz.edu.cn](mailto:jintaiCHEN@hkust-gz.edu.cn) (J.
Chen), [xucao2@illinois.edu](mailto:xucao2@illinois.edu) (X. Cao).
Affiliation:  The Hong Kong University of Science and Technology
(Guangzhou) Affiliation:  The Hong Kong University of Science and
Technology

###### Abstract

Medical image segmentation is fundamental for biomedical discovery.
Existing methods lack generalizability and demand extensive,
time-consuming manual annotation for new clinical application. Here, we
propose MedSAM-3, a text promptable medical segmentation model for
medical image and video segmentation. By fine-tuning the Segment
Anything Model (SAM) 3 architecture on medical images paired with
semantic conceptual labels, our MedSAM-3 enables medical Promptable
Concept Segmentation (PCS), allowing precise targeting of anatomical
structures via open-vocabulary text descriptions rather than solely
geometric prompts. We further introduce the MedSAM-3 Agent, a framework
that integrates Multimodal Large Language Models (MLLMs) to perform
complex reasoning and iterative refinement in an agent-in-the-loop
workflow. Comprehensive experiments across diverse medical imaging
modalities, including X-ray, MRI, Ultrasound, CT, and video, demonstrate
that our approach significantly outperforms existing specialist and
foundation models. We will release our code and model at
[https://github.com/Joey-S-Liu/MedSAM3](https://github.com/Joey-S-Liu/MedSAM3).

## 1 Introduction

![Refer to caption](2511.19046v2/figs/overview.png)

Figure 1: Overview of concept-driven medical image and video
segmentation across multiple modalities using MedSAM-3, highlighting
that concise clinical concepts directly guide MedSAM-3 to produce
reliable segmentations and thereby simplify physicians’ workflow.

Medical segmentation is the cornerstone of the modern healthcare system,
providing the quantitative analysis necessary for accurate diagnosis,
precise treatment planning, and effective monitoring of disease
progression \[[5](#bib.bib5)\]. While deep learning has driven
considerable progress, the development of specialist models for every
unique task, modality, and pathology is inefficient and scales poorly.
Such models lack generalizability and demand extensive, time-consuming
manual annotation for each new clinical application.

The emergence of large-scale foundation models, such as the Segment
Anything Model (SAM) \[[28](#bib.bib1), [47](#bib.bib2)\], has marked a
paradigm shift towards building generalist systems that can handle
diverse tasks. In the medical field, this approach was successfully
validated by models like MedSAM \[[41](#bib.bib18)\],
MedSAM-2 \[[64](#bib.bib3)\] and MedSAM2 \[[43](#bib.bib4)\], which
adapted the original SAM for medical-specific challenges. MedSAM2, in
particular, demonstrated the power of a promptable foundation model for
segmenting 3D medical images and videos, proving that such systems can
drastically reduce manual annotation costs \[[43](#bib.bib4)\]. However,
these models primarily rely on geometric prompts, which can still be
laborious for complex structures and do not fully capture the rich
semantic intent of clinicians. In addition, these models can only serve
as one tool, lacking potential to connect with the agentic ecosystem
supported by multimodal large language models (LLMs) \[[33](#bib.bib35),
[58](#bib.bib33), [1](#bib.bib34)\].

The recent introduction of SAM 3 marks a significant leap in interactive
segmentation with its “Promptable Concept Segmentation” (PCS)
capability \[[4](#bib.bib6)\]. Unlike methods reliant on geometric cues,
SAM 3 can detect and segment objects based on open-vocabulary conceptual
prompts, such as natural language descriptions (e.g., “a yellow school
bus”) or visual exemplars. This ability to operate on semantic concepts
presents a transformative opportunity for medical imaging, where
clinical language is inherently conceptual (e.g., “segment the tumor and
surrounding edema” or “identify all enlarged lymph nodes”). This
directly addresses a fundamental limitation of prior text-guidance
segmentation models, such as BiomedParse \[[61](#bib.bib64)\], which
were constrained to a fixed, pre-defined vocabulary and thus could not
generalize to the vast and nuanced range of concepts encountered in
clinical practice \[[62](#bib.bib36)\].

To address these limitations, we present MedSAM-3, a concept-driven
framework designed to segment medical imagery through semantic guidance
(Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")).
We began by benchmarking the original SAM 3 on multiple medical
segmentation datasets to validate its baseline capabilities on both text
prompting and visual prompting. However, raw SAM 3 struggled with the
healthcare domain. Consequently, we fine-tuned the architecture on a
curated dataset of diverse medical images paired with rich conceptual
labels. The resulting model allows users to segment complex anatomical
structures and pathologies using simple text descriptions or visual
references from inter- or intra-scan examples. This paradigm shifts the
interaction from simple geometric prompting to an intuitive, clinically
aligned semantic workflow. Through comprehensive experiments, we
demonstrate that MedSAM-3 not only establishes a new state-of-the-art
for generalist medical segmentation but also significantly streamlines
clinical annotation, paving the way for more intelligent, collaborative
medical AI systems.

Our main contributions are:

- •
  We propose MedSAM-3, adapting the SAM 3 architecture to the medical
  domain to enable precise Promptable Concept Segmentation (PCS) using
  medical text and visual prompts.
- •
  We introduce the MedSAM3 Agent, an agentic framework that extends
  MedSAM-3 to process complex, long-form clinical instructions and
  improve accuracy through an iterative agent-in-the-loop paradigm.
- •
  We conduct extensive experiments across diverse medical imaging
  modalities, demonstrating the effectiveness of our design and
  providing valuable insights into the deployment of concept-based
  segmentation models in healthcare.

## 2 Related Works

Segmentation in Medical Images. The field of medical image segmentation
has witnessed a remarkable evolution, primarily driven by the transition
from convolutional neural networks (CNNs) to Transformer-based
architectures. Early advancements were cemented by the U-Net
series \[[49](#bib.bib10), [44](#bib.bib11), [63](#bib.bib12),
[46](#bib.bib13), [26](#bib.bib14), [25](#bib.bib51)\]. With the advent
of Vision Transformers, researchers sought to overcome the limited
receptive field of CNNs, proposing CNN-Transformer hybrid architecture
in medical image segmentation \[[10](#bib.bib15), [9](#bib.bib16),
[24](#bib.bib17)\]. Despite their success, these specialist models
typically require training from scratch for specific organs or
modalities, limiting their scalability and generalization across the
diverse landscape of clinical tasks \[[5](#bib.bib5),
[29](#bib.bib55)\]. The focus has shifted towards developing large-scale
foundation models capable of universal segmentation. The
SAM \[[28](#bib.bib1)\] demonstrated unprecedented zero-shot
generalization in natural images, sparking a wave of adaptations for the
medical domain. Initial efforts \[[41](#bib.bib18), [11](#bib.bib19),
[55](#bib.bib20)\] adapted SAM via fine-tuning or adapter layers to
handle medical modalities. This was further extended to 3D volumetric
data by models \[[16](#bib.bib21), [53](#bib.bib22), [43](#bib.bib4),
[64](#bib.bib3)\], which leverage temporal or spatial consistency. While
these models excel at geometric prompting (points/boxes), they often
lack semantic understanding. Recent works like
BiomedParse \[[61](#bib.bib64)\] and UniverSeg \[[7](#bib.bib23)\] have
attempted to integrate text guidance. However, as noted in recent
studies, these systems are often constrained to fixed vocabularies or
lack the reasoning capabilities to interpret complex, open-ended
clinical concepts, necessitating a shift towards more agentic
architectures \[[43](#bib.bib4), [47](#bib.bib2)\]. Meanwhile,
specialized vision language model (VLM) designed for medical image
segmentation have emerged \[[36](#bib.bib57), [56](#bib.bib58)\]. In the
3D domain, models such as M3D-LaMed \[[6](#bib.bib59)\] and other
promptable frameworks \[[39](#bib.bib60), [32](#bib.bib61),
[35](#bib.bib62)\] have further extended these capabilities. However,
due to the suboptimal performance and lack of interactivity in these
static segmentation VLMs, developing agent-based systems has become a
promising future trend for handling complex clinical scenarios.

Segmentation Agent. The integration of Large Language Models (LLMs) with
vision systems has given rise to "Segmentation Agents" capable of
complex reasoning and interactive understanding. This paradigm moves
beyond simple instruction following to "Reasoning Segmentation," where
the model must interpret implicit queries (e.g., "segment the reason for
the patient’s pain"). Pioneering works in the general domain include
LISA \[[31](#bib.bib24)\] and PixelLM \[[48](#bib.bib25)\]. This
direction was further advanced by multimodal agents \[[59](#bib.bib26),
[38](#bib.bib27)\] and SAM 3 \[[4](#bib.bib6)\]. These agents differ
from static models by maintaining a working memory and iteratively
refining predictions based on user feedback, a critical feature for
high-stakes decision-making processes in healthcare \[[57](#bib.bib28),
[45](#bib.bib29)\]. In the specialized domain of professional workflows,
agentic systems are rapidly transforming how experts interact with data
across various verticals. In radiology, agents and models such as
MedRAX \[[17](#bib.bib7)\], LLaVA-Med \[[34](#bib.bib30)\], and
RadFM \[[54](#bib.bib63)\] have been proposed. Beyond healthcare,
similar trends are observed in other fields with systems like
mDocAgent \[[21](#bib.bib8)\], NovelSeek \[[52](#bib.bib9)\],
ViperGPT \[[50](#bib.bib31)\], and ChemLLM \[[60](#bib.bib32)\]. Our
work unifies these directions by proposing MedSAM3 Agent, an agent
tailored specifically for medical segmentation that combines the
reasoning of LLMs with the precise, concept-driven segmentation
capabilities of the MedSAM-3 architecture.

## 3 Methodology

### 3.1 Enabling Medical Concepts in SAM 3

Our MedSAM-3 is developed as a generalization of MedSAM-2 and MedSAM2,
adopting the unified architecture of SAM 3 to support both the novel
Promptable Concept Segmentation (PCS) task and the traditional
Promptable Visual Segmentation (PVS) tasks. In the PVS setting, the
model accepts diverse visual prompts, such as points, boxes, or masks,
to spatially and temporally define individual objects for segmentation.
For medical PVS, MedSAM-3 retains the box-based prompting strategy
supported by MedSAM2; compared to points and masks, bounding boxes offer
a less ambiguous method for specifying clinically useful targets, making
them particularly effective for delineating organs and lesions. In
addition to PVS, MedSAM-3 introduces support for PCS, allowing the model
to target objects using short medical noun phrases, including those with
positional adjectives. While the original SAM 3 model is optimized for
these concise atomic prompts, MedSAM-3 can handle more complex language
queries and reasoning by composing the model with a MLLM within an
agentic pipeline.

Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Enabling Medical Concepts in SAM 3 ‣ 3 Methodology ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")
illustrates the MedSAM-3 architecture, which features a dual
encoder-decoder transformer design. This consists of a detector for
image-level capabilities and a tracker paired with a memory module for
video tasks. The memory blocks, inherited from SAM 2, employ transformer
layers with self-attention and cross-attention mechanisms to condition
current frame features on predictions from previous frames via a
streaming memory bank. Both the detector and tracker ingest aligned
vision-language inputs from a shared Perception Encoder (PE) backbone.

![Refer to caption](2511.19046v2/medsam3architecture.png)

Figure 2: Overview of MedSAM-3.

### 3.2 Supervised Fine-Tuning with Medical Concepts

Based on the SAM 3 architecture, the MedSAM-3 model freezes the image
and text encoder and updates the remaining detector components during
fine-tuning. This design preserves the strong visual and concept prior
established by SAM 3 while allowing the task-specific modules to adapt
to medical concepts efficiently. The model is optimized using paired
medical images and concise concept phrases, each limited to no more than
three words and selected strictly according to the dataset’s official
documentation or repository descriptions. Such careful curation is
motivated by several findings from the SAM 3 evaluation
(Section [5](#S5 "5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"))
and is intended to ensure semantic precision, minimize ambiguity in
textual guidance, and reduce noise arising from overly broad or
inconsistent language. Through this approach, MedSAM-3 strengthens its
ability to map semantic medical concepts to anatomically meaningful
structures, ultimately enhancing segmentation robustness across diverse
clinical scenarios.

### 3.3 Scalable Medical Segmentation Agent

We introduce the MedSAM-3 Agent
(Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")),
an agentic framework that dynamically reasons, plans, and executes
multistep medical segmentation workflows. Unlike previous approaches,
the MedSA-3 Agent integrates multimodal reasoning with concept-guided
segmentation capabilities. Given any medical image and a user request, a
general MLLM or medical vision-language model (VLM) acts as the core
planner: it analyzes the image, devises a step-by-step plan, and invokes
MedSAM-3 to generate segmentation masks. After each step, the agent
inspects the results, using visual and textual feedback to update its
understanding of the environment. This perception-action
agent-in-the-loop enables the agent to continuously revise its plan and
decide the next action. The process continues until the agent is
confident it has satisfied the user’s goal (or determines that no valid
mask exists), at which point it returns a final set of masks. The
resulting pipeline can handle queries far more complex than simple noun
phrases, allowing it to understand relationships between anatomical
structures and apply visual common sense. We further conduct an
experiment to validate a Gemini 3 Pro supported MedSAM-3 Agent can
surpass MedSAM-3.

## 4 Experiments and Results

### 4.1 Datasets

To evaluate the segmentation performance of SAM 3 across various medical
scenarios and build MedSAM-3, we collected several datasets encompassing
multiple imaging modalities, including X-ray, MRI, ultrasound (US), OCT,
fundus, dermoscopy, histopathology, nuclear imaging, infrared,
endoscopy, and CT, as well as different dimensions, including 2D, 3D,
and video. We converted all 3D datasets into frame-sequence formats.
When a dataset does not provide an official train–test split, we divide
the data using an 80%–20% (4:1) ratio. The detailed information of these
datasets is presented below.

- •
  COVID-QU-Ex \[[51](#bib.bib37)\]. The COVID-QU-Ex dataset contains
  33,920 chest X-ray images and their corresponding infection GTs,
  including COVID-19 infection, non-COVID-19 infection, and normal
  cases. A total of 11,956 COVID-19 infection cases were used in this
  study.
- •
  BUSI \[[2](#bib.bib38)\]. The Breast Ultrasound Images Dataset (BUSI)
  includes 780 breast ultrasound images with corresponding breast cancer
  GTs, covering standard, benign, and malignant cases.
- •
  iChallenge-GOALS \[[18](#bib.bib39)\]. The Glaucoma OCT Analysis and
  Layer Segmentation (GOALS) dataset consists of 300 OCT images and
  their corresponding GTs for the retinal nerve fiber layer, ganglion
  cell layer, and choroid layer.
- •
  RIM-ONE \[[19](#bib.bib40)\]. The RIM-ONE retinal dataset includes 485
  retinal fundus images with corresponding optic disc and cup GTs,
  comprising 313 images from healthy subjects and 172 from glaucoma
  patients.
- •
  ISIC 2018 \[[13](#bib.bib41)\]. The International Skin Imaging
  Collaboration (ISIC) 2018 dataset contains 2,594 training images and
  1,000 test images of skin images, each with associated lesion GTs.
- •
  MoNuSeg \[[30](#bib.bib42)\]. The Multi-organ Nucleus Segmentation
  Challenge (MoNuSeg) dataset includes 51 histology tissue images from
  patients with tumors of different organs, each providing a list of
  annotated nuclei instances.
- •
  DSB 2018 \[[8](#bib.bib43)\]. The 2018 Data Science Bowl (DSB 2018)
  dataset consists of 670 segmented nuclei images acquired under various
  conditions, differing in cell type, magnification, and imaging
  modality.
- •
  RAVIR \[[22](#bib.bib44)\]. The Retinal Arteries and Veins in Infrared
  Reflectance Imaging (RAVIR) dataset contains 42 infrared reflectance
  (IR) images and their corresponding retinal artery and vein GTs.
- •
  Kvasir-SEG \[[27](#bib.bib45)\]. The Kvasir-SEG dataset includes 1,000
  gastrointestinal endoscopy images and their corresponding polyp GTs.
- •
  Parse2022 \[[40](#bib.bib46)\]. The Pulmonary Artery Segmentation
  Challenge 2022(Parse2022) dataset comprises 100 3D lung CT scans with
  corresponding pulmonary artery GTs.
- •
  LiTS \[[12](#bib.bib47)\]. The Liver Tumor Segmentation Challenge
  (LiTS) dataset contains 130 3D abdominal CT scans and their
  corresponding liver and liver tumor GTs.
- •
  PROMISE12 \[[37](#bib.bib48)\]. The Prostate MR Image Segmentation
  2012 (PROMISE12) dataset includes 80 3D transversal T2-weighted MRI
  scans and their corresponding prostate GTs.
- •
  ISLES 2024 \[[14](#bib.bib49)\]. The Ischemic Stroke Lesion
  Segmentation Challenge 2024 (ISLES 2024) dataset consists of 250 3D
  brain MRI scans with corresponding ischemic stroke lesion GTs.
- •
  PolypGen \[[3](#bib.bib50)\]. The PolypGen dataset is an open-access
  dataset that comprises 1,537 polyp images, 2,225 positive video
  sequences with polyp GTs, and 4,275 negative frames.

![Refer to caption](2511.19046v2/MedSAM3Agent.png)

Figure 3: Overview of MedSAM-3 Agent refinement loop. The MedSAM-3 Agent
plans and executes multi-step medical image segmentation using a MLLM,
generating masks and refining them iteratively with visual and textual
feedback.

### 4.2 Experimental settings

For the performance evaluation on the 2D datasets, we employed three
classical 2D segmentation networks: U-Net \[[49](#bib.bib10)\],
Unet3+ \[[25](#bib.bib51)\], and Polyp-PVT \[[15](#bib.bib52)\]. For the
performance evaluation on the 3D datasets, we also adopted three
representative 3D segmentation networks: nn-Unet \[[26](#bib.bib14)\],
Swin UNETR \[[23](#bib.bib53)\], and U-Mamba \[[42](#bib.bib54)\]. To
ensure a fair comparison across different methods, all competing
approaches except SAM 3 were trained on the training split of each
dataset and evaluated on the corresponding test split. SAM 3 was
directly tested on the test set without additional training.

For the experiments on the 2D datasets, SAM 3 was evaluated under two
settings. In the first setting, the concept input consisted solely of a
short phrase describing the target, limited to no more than three
words(hereafter referred to as SAM 3 T). In the second setting, the
concept input included both the textual phrase and a bounding box
enclosing the largest connected component of the target as an
image-based reference(hereafter referred to as SAM 3 T+I). For the
experiments on the 3D datasets, only the textual phrase was used as the
concept input. The phrase inputs used for different datasets are shown
in
Table [1](#S4.T1 "Table 1 ‣ 4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
The comprehensive evaluation of SAM 3 is presented in
Section [5](#S5 "5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").

|             |                                  |
| ----------- | -------------------------------- |
| Dataset     | Phrase input                     |
| COVID-QU-Ex | lung infection                   |
| DSB 2018    | nuclei                           |
| BUSI        | breast tumor                     |
| GOALS       | RNFL & GCIPL & choroid           |
| RIM-ONE     | optic cup & optic disc           |
| ISIC 2018   | skin lesion                      |
| RAVIR       | retinal arteries & retinal veins |
| Kvasir-SEG  | polyp                            |
| MoNuSeg     | nuclei                           |
| PolypGen    | polyp                            |
| LiTS        | liver & liver tumor              |
| PROMISE12   | prostate                         |
| ISLES 2024  | ischemic stroke lesion           |
| Parse2022   | pulmonary artery                 |

Table 1: The phrase inputs used for different datasets.

We conducted supervised fine-tuning to build MedSAM-3 on four
representative 2D medical datasets from diverse imaging modalities:
BUSI, RIM-ONE(Cup), ISIC 2018, and Kvasir-SEG. In addition, the previous
prompt-based SOTA medical segmentation model, MedSAM, was included in
the evaluation. Critically, our fine-tuning exclusively targeted the
detector module of the underlying SAM 3 architecture, a specific
adaptation designed to optimize domain-specific feature detection for
medical tasks. To comprehensively assess the model’s responsiveness to
different types of prompts during the adaptation process, we adopted two
distinct fine-tuning paradigms:

- •
  1\) Pure Text Prompt Fine-tuning (MedSAM-3 T): In this setting, the
  model is trained using only the input image and the text description
  as the prompt. This paradigm aims to enhance the model’s ability to
  ground medical concepts in visual features without explicit spatial
  guidance.
- •
  2\) Text Prompt + Bounding Box Fine-tuning (MedSAM-3 T+I): In this
  setting, the model is provided with both the semantic text description
  and a bounding box derived from the ground truth mask. This paradigm
  evaluates the synergistic effect of combining semantic intent with
  geometric cues to improve segmentation precision.

To evaluate the MedSAM-3 Agent, the test set of the BUSI dataset was
used for inference, where the agent model was implemented using Gemini 3
Pro.

All training and inference experiments were conducted on one or two A100
GPUs, each equipped with 80 GB of memory.

### 4.3 MedSAM-3 Performance

The performance of MedSAM-3 is summarized in
Table [2](#S4.T2 "Table 2 ‣ 4.3 MedSAM-3 Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")
and
Figure [4](#S4.F4 "Figure 4 ‣ 4.3 MedSAM-3 Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
The text-only inference (MedSAM-3 T) shows clear limitations across all
datasets, indicating that text signals alone are insufficient for
reliable medical image segmentation. In contrast, the text-and-image
inference (MedSAM-3 T+I) yields consistent performance gains and
achieves the best results on all four benchmarks, demonstrating visual
features with medical-domain text priors enhances robustness across
medical imaging conditions. In summary, MedSAM-3 demonstrates strong
potential for extension toward universal medical image segmentation.

![Refer to caption](2511.19046v2/figs/MedSAM-3_performance.png)

Figure 4: Performance comparison between MedSAM-3 and competing methods
on four medical datasets.

We also present qualitative visualizations of the segmentation results
on the BUSI, RIM-ONE (Cup), ISIC 2018, and Kvasir-SEG datasets. As shown
in
Figure [5](#S4.F5 "Figure 5 ‣ 4.3 MedSAM-3 Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
MedSAM-3 achieves consistently accurate and visually coherent
segmentation across diverse modalities, demonstrating strong performance
even in challenging low-contrast or irregular-boundary regions. In
contrast, SAM 3 shows noticeable performance degradation. Remarkably,
MedSAM-3 attains these improvements with only a small amount of
domain-specific fine-tuning data, highlighting the substantial potential
of lightweight adaptation for medical image segmentation. More
discussion would be posted in
Section [5](#S5 "5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").

![Refer to caption](2511.19046v2/figs/medsam_visual.png)

Figure 5: Visualization of the segmentation performance of MedSAM-3, SAM
3(both T+I versions), and other comparison methods.

|              |                            |                            |                              |                            |
| ------------ | -------------------------- | -------------------------- | ---------------------------- | -------------------------- |
| Methods      | BUSI                       | RIM-ONE(Cup)               | ISIC 2018                    | Kvasir-SEG                 |
| U-Net        | 0.7618                     | 0.8480                     | 0.8760                       | 0.8244                     |
| MedSAM       | 0.7514                     | 0.8479                     | 0.9177                       | 0.7657                     |
| SAM 3 T      | 0                          | 0                          | 0.2189                       | 0                          |
| SAM 3 T+I    | 0.7110                     | 0.8303                     | 0.8178                       | 0.7671                     |
| MedSAM-3 T   | 0.2674                     | 0.0826                     | 0.5687                       | 0.1441                     |
| MedSAM-3 T+I | 0.7772 $`\uparrow`$ 0.0080 | 0.8977 $`\uparrow`$ 0.0497 | 0.9058 $`\downarrow`$ 0.0119 | 0.8831 $`\uparrow`$ 0.0587 |

Table 2: Performance comparison between MedSAM-3 and other methods on
four datasets. The best result on each dataset is highlighted in bold.
Colored arrows indicate SAM performance changes relative to the best
method per dataset.

### 4.4 MedSAM-3 Agent Performance

|                             |                                   |        |
| --------------------------- | --------------------------------- | ------ |
| Methods                     | MLLM                              | BUSI   |
| U-Net \[[49](#bib.bib10)\]  | \-                                | 0.7618 |
| MedSAM \[[41](#bib.bib18)\] | \-                                | 0.7514 |
| MedSAM-3 T+I                | \-                                | 0.7772 |
| MedSAM-3 Agent              | Gemini 3 Pro \[[20](#bib.bib56)\] | 0.8064 |

Table 3: Using Gemini 3 Pro as a controlled and evaluation agent can
improve the result.

We compared the performance of MedSAM-3 against its agentic variant,
MedSAM3 Agent, using the BUSI test set
(Table [3](#S4.T3 "Table 3 ‣ 4.4 MedSAM-3 Agent Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")).
By integrating Gemini 3 Pro \[[20](#bib.bib56)\] as the core multimodal
LLM—orchestrating query interpretation and three rounds of iterative
feedback—we achieved a significant performance boost. Specifically, the
Dice score improved from 0.7772 to 0.8064. It reveals that agentic
workflow could significantly enhance the performance of segmentation
models.

## 5 Discussion

Prior to developing MedSAM-3, we evaluated the off-the-shelf
capabilities of SAM 3 within the medical segmentation domain. Our
experiments revealed that the standard SAM 3 model lacks the necessary
generalization to handle diverse clinical modalities effectively. These
limitations motivated the design of MedSAM-3 and the agentic MedSAM-3
Agent framework.

### 5.1 SAM 3 Performance

#### 5.1.1 SAM 3 Performance on 2D/Video Datasets

![Refer to caption](2511.19046v2/figs/radar.png)

Figure 6: Radar charts of different models’ performance on different
datasets. Left: 2D/video scene; Right: 3D scene.

Table
[4](#S5.T4 "Table 4 ‣ 5.1.1 SAM 3 Performance on 2D/Video Datasets ‣ 5.1 SAM 3 Performance ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")
summarizes the segmentation results of SAM 3 on several representative
2D/video medical imaging datasets. Overall, SAM 3 exhibits highly uneven
performance, showing a strong “subject bias.” In some datasets, such as
RIM-ONE, the model achieves relatively high accuracy. However, in
datasets like DSB 2018 and RAVIR, it almost completely fails. This
inconsistency may arise from the model’s limited sensitivity to medical
concepts and its instability in distinguishing semantically similar
medical terms. Furthermore, even though part of the evaluation data may
have been included in SAM 3’s pretraining corpus, its transferability to
medical scenarios remains weak, revealing the difficulty of cross-domain
generalization.

Notably, incorporating bounding box guidance (Text + BBX) leads to a
substantial improvement in segmentation quality, where the Dice scores
approach or even surpass those of conventional supervised methods. This
demonstrates the critical role of geometric cues in assisting conceptual
understanding, consistent with the findings in the natural domain. In
summary, SAM 3 performs inconsistently across 2D medical segmentation
tasks, with strong reliance on spatial hints to compensate for its
limited grasp of fine-grained medical semantics. This observation
suggests that combining conceptual and geometric information remains
essential for achieving reliable segmentation in medical imaging.

|             |                              |                              |                              |                              |                              |                              |                              |                              |                              |                              |                              |
| ----------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| Methods     | COVID-QU-Ex                  | DSB 2018                     | BUSI                         | GOALS                        | RIM-ONE(Cup)                 | RIM-ONE(Disc)                | ISIC 2018                    | RAVIR                        | Kvasir-SEG                   | MoNuSeg                      | PolypGen                     |
| U-Net       | 0.7880                       | 0.8936                       | 0.7618                       | 0.7902                       | 0.8480                       | 0.9514                       | 0.8760                       | 0.7539                       | 0.8244                       | 0.6696                       | 0.6897                       |
| Unet3+      | 0.7928                       | 0.9545                       | 0.7782                       | 0.8513                       | 0.8206                       | 0.9545                       | 0.8797                       | 0.7681                       | 0.8321                       | 0.6595                       | 0.7634                       |
| Polyp-PVT   | 0.7800                       | 0.9420                       | 0.7457                       | 0.8487                       | 0.8406                       | 0.9420                       | 0.8917                       | 0.5284                       | 0.8536                       | 0.3472                       | 0.6205                       |
| SAM 3 T     | 0.0305 $`\downarrow`$ 0.7623 | 0.0803 $`\downarrow`$ 0.8742 | 0 $`\downarrow`$ 0.7782      | 0 $`\downarrow`$ 0.8513      | 0 $`\downarrow`$ 0.8480      | 0.3858 $`\downarrow`$ 0.5687 | 0.2189 $`\downarrow`$ 0.6728 | 0 $`\downarrow`$ 0.7681      | 0 $`\downarrow`$ 0.8536      | 0 $`\downarrow`$ 0.6696      | 0 $`\downarrow`$ 0.7634      |
| SAM 3 T + I | 0.7405 $`\downarrow`$ 0.0523 | 0.6953 $`\downarrow`$ 0.2592 | 0.7110 $`\downarrow`$ 0.0672 | 0.8108 $`\downarrow`$ 0.0405 | 0.8303 $`\downarrow`$ 0.0177 | 0.9270 $`\downarrow`$ 0.0275 | 0.8178 $`\downarrow`$ 0.0739 | 0.2163 $`\downarrow`$ 0.5518 | 0.7671 $`\downarrow`$ 0.0865 | 0.4135 $`\downarrow`$ 0.2561 | 0.6903 $`\downarrow`$ 0.0731 |

Table 4: Performance comparison between SAM 3 and traditional
segmentation models. The best result on each dataset is highlighted in
bold. Colored arrows indicate SAM performance changes relative to the
best method per dataset.

#### 5.1.2 SAM 3 Performance on 3D Datasets

Table
[5](#S5.T5 "Table 5 ‣ 5.1.2 SAM 3 Performance on 3D Datasets ‣ 5.1 SAM 3 Performance ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")
summarizes the performance of different methods on several 3D medical
image datasets. Overall, nn-UNet, Swin UNETR, and U-Mamba achieve
relatively stable and high Dice scores across tasks, whereas SAM 3 shows
consistently lower performance on all four datasets, with particularly
large gaps on more challenging data such as LiTS and ISLES 2024. These
observations indicate that SAM 3 remains limited when applied to
volumetric segmentation.

|            |                              |                              |                              |                              |
| ---------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| Methods    | Parse2022                    | LiTS                         | PROMISE12                    | ISLES2024                    |
| nn-Unet    | 0.8311                       | 0.7714                       | 0.9011                       | 0.7718                       |
| Swin UNETR | 0.8134                       | 0.7425                       | 0.8934                       | 0.7523                       |
| U-Mamba    | 0.7692                       | 0.7910                       | 0.9002                       | 0.7566                       |
| SAM 3 T    | 0.5295 $`\downarrow`$ 0.3016 | 0.1374 $`\downarrow`$ 0.6536 | 0.6110 $`\downarrow`$ 0.2901 | 0.3033 $`\downarrow`$ 0.4685 |

Table 5: Performance comparison between SAM 3 and other methods on 3D
medical image datasets. The best result on each dataset is highlighted
in bold. Colored arrows indicate SAM performance changes relative to the
best method per dataset.

Figure [6](#S5.F6 "Figure 6 ‣ 5.1.1 SAM 3 Performance on 2D/Video Datasets ‣ 5.1 SAM 3 Performance ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts")
also visually illustrates the substantial performance disparities of SAM
3 across different datasets.

### 5.2 Findings

We summarizes the key observations from applying SAM 3 to medical image
segmentation tasks, further fine-tuning the model on domain-specific
datasets to develop MedSAM-3, and employing an advanced MLLM to
construct the MedSAM-3 Agent. Through a systematic evaluation across
different modalities and targets, several recurring patterns were
identified. These observations reflect the limitations of SAM 3 when
applied directly to medical scenarios and the behaviors that persist or
newly emerge after task-specific adaptation. The analysis further
indicates that fine-tuning can substantially improve performance, yet
its effectiveness is constrained by the limited scale of medical
datasets and the scarcity of high-quality data containing rich clinical
terminology and domain-specific textual descriptions. Moreover,
integrating an MLLM-based agent reveals additional potential of
MedSAM-3, enabling more flexible interaction and better utilization of
medical knowledge. Together, these findings highlight both the
challenges and opportunities in adapting general-purpose vision-language
models to meet the precision and structural requirements of medical
image analysis. The main findings are outlined below.

Finding 1: Substantial Performance Discrepancy Between SAM 3 and
Established Medical Segmentation Baselines. Across all evaluated
datasets, SAM 3 shows a large and unusual performance gap compared with
standard medical segmentation models. This pattern is consistent across
2D, video, and 3D tasks. A representative example is the PROMISE12
dataset. Although PROMISE12 has clear anatomy and minimal semantic
ambiguity, some are segmented reasonably well while many others fail
severely among the 30 test cases shown in
Figure [7](#S5.F7 "Figure 7 ‣ 5.2 Findings ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").

![Refer to caption](2511.19046v2/figs/promise12_gap.png)

Figure 7: Comparison of per-case performance between SAM 3 and nn-Unet
on the PROMISE12 test set. The bar chart shows the absolute Dice values
of different methods across cases, while the line plot illustrates the
performance differences between the two methods for each case.

Finding 2: Systematic Misalignment Between Concept Prompts and
Anatomical Target Regions in SAM 3. SAM 3 exhibits a consistent pattern
of misalignment between concepts and the regions it predicts, as shown
in
Figure [8](#S5.F8 "Figure 8 ‣ 5.2 Findings ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
In the LiTS liver segmentation task, concept specifying “liver” often
leads the model to segment the left and right lungs while the true liver
region is almost entirely ignored. A similar phenomenon occurs in the
ISIC 2018 dataset. When the concept is “lesion”, the model frequently
produces a prediction that covers broad non-lesion areas and excludes
the actual lesion. The recurrence of this behavior across tasks suggests
that SAM 3 does not reliably associate textual prompts with their
corresponding visual targets.

![Refer to caption](2511.19046v2/figs/liver_to_lung.png)

Figure 8: Examples of SAM 3 inference. On the left, using the LiTS
dataset, the model segments the lung regions instead of the liver with
the concept “liver”. On the right, using the ISIC2018 dataset, the model
segments surrounding non-lesion regions instead of the lesion with the
concept “lesion”.

Finding 3: Limited Semantic Discrimination of Fine-Grained Medical
Terminology by SAM 3. The SAM 3 model struggles to distinguish between
closely related biological or anatomical concepts. We take the MoNuSeg
dataset and DSB 2018 dataset for example, shown in
Figure [9](#S5.F9 "Figure 9 ‣ 5.2 Findings ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
The concept “nucleus” or “nuclei” was totally unrecognized, even the
full name of the MoNuSeg dataset is actually the Multi-organ Nucleus
Segmentation Challenge, whereas the more generic prompt “cell” produced
reasonable segmentation results. The details are shown in
Table [6](#S5.T6 "Table 6 ‣ 5.2 Findings ‣ 5 Discussion ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").

![Refer to caption](2511.19046v2/figs/MoNuSeg_example.png)

Figure 9: An example illustrating the large performance gap of SAM 3 on
the MoNuSeg and DSB2018 dataset when provided with semantically similar
concept inputs.

|                       |             |        |             |        |
| --------------------- | ----------- | ------ | ----------- | ------ |
| \Block2-1Methods      | MoNuSeg     |        | DSB 2018    |        |
|                       | nuclei      | cell   | nuclei      | cell   |
| SAM 3 T               | 0           | 0.6786 | 0.0803      | 0.5753 |
| SAM 3 T + I           | 0.4135      | 0.7907 | 0.6953      | 0.7384 |

Table 6: Performance of SAM 3 on the MoNuSeg and DSB 2018 under
different concepts.

Finding 4: MedSAM-3 Demonstrates Promising Performance Through Efficient
Domain Adaptation. Through domain-specific fine-tuning with curated
medical concept annotations, MedSAM-3 shows encouraging improvements in
concept alignment and segmentation reliability across diverse clinical
imaging modalities. While further scaling would benefit from broader
concept-annotated datasets, the current approach demonstrates meaningful
progress toward more generalizable medical segmentation systems.

Finding 5: The Agentic Framework Effectively Raises the Performance
Ceiling of MedSAM-3. Integrating an MLLM-based agent brings measurable
improvements to MedSAM-3, particularly in handling complex clinical
instructions and performing iterative refinement. By orchestrating query
interpretation through Gemini 3 Pro with iterative feedback loops, the
agentic framework elevates segmentation accuracy through multi-step
reasoning, intelligent prompt refinement, and error correction. These
results highlight the promising potential of agentic workflows to
enhance foundation models for increasingly complex and nuanced medical
segmentation scenarios in real-world clinical practice.

### 5.3 Why MedSAM-3?

Directly applying SAM 3 in medical scenarios leads to substantial
performance degradation because the model lacks the domain-specific
semantic grounding required for precise concept–region alignment.
Medical structures often exhibit low contrast, subtle boundaries, and
significant inter-patient variability, and these properties are not
represented in the natural-image corpus used for SAM 3 pre-training. As
a result, the model struggles to interpret fine-grained anatomical or
pathological terms and frequently produces unstable masks under
text-only or weak prompt settings.

MedSAM-3 addresses these limitations by introducing medical concepts
with explicit, high-quality supervision. The fine-tuning stage exposes
the model to medically meaningful terminology, consistent labeling
rules, and clinically relevant spatial patterns, enabling it to learn
the specialized semantic relationships absent from the original
pre-training. This adaptation substantially improves its reliability
across modalities and tasks, allowing the model to generate semantically
grounded predictions even when visual cues are subtle or ambiguous.

Our empirical results show that the improvements brought by the agentic
framework depend critically on the quality of the underlying model. The
agent can refine prompts and perform iterative correction, but its
effectiveness diminishes when the base segmentation is semantically
misaligned. In contrast, MedSAM-3 provides a stable and medically
aligned starting point, ensuring that subsequent agentic reasoning
operates on a meaningful foundation.

Overall, MedSAM-3 serves as the essential bridge between general-purpose
foundation models and the precision required in clinical image analysis.
It transforms SAM 3 from a broadly capable vision-language model into
one that can consistently operate within the constraints and
expectations of the medical domain, thereby establishing a reliable
performance baseline for both direct inference and agent-assisted
segmentation.

## 6 Conclusion

We propose MedSAM-3, extending the SAM 3 architecture to address the
unique challenges of medical concept grounding. Through domain-specific
fine-tuning on PCS task, MedSAM-3 significantly outperforms the original
SAM 3, particularly in handling complex medical semantics and temporal
consistency. We further enhanced this backbone with the MedSAM-3 Agent,
an agent-in-the-loop framework that improves usability via iterative
feedback. Our analysis reveals that MedSAM-3 determines the fundamental
segmentation quality, while the agent leverages reasoning to correct
errors and optimize prompts, effectively pushing the performance
ceiling. This work demonstrates the potential of coupling domain
adaptation with agentic workflows. Future work will address current
limitations in concept granularity and text–image alignment, with the
aim of scaling MedSAM-3 to a wider range of clinical applications. We
will release our code and models to support the community.

## References

- \[1\] A. M. Al Radi, X. Cao, F. Yu, Y. Liu, F. Liu, C. Wang, Y.
  Chen, J. Chen, H. Wang, Y. Meng, et al. (2025) Agentic
  large-language-model systems in medicine: a systematic review and
  taxonomy. Authorea Preprints. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[2\] W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy (2020) Dataset
  of breast ultrasound images. Data in brief 28, pp. 104863. Cited by:
  [2nd
  item](#S4.I1.i2.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[3\] S. Ali, D. Jha, N. Ghatwary, S. Realdon, R. Cannizzaro, O. E.
  Salem, D. Lamarque, C. Daul, M. A. Riegler, K. V. Anonsen, et
  al. (2023) A multi-centre polyp detection and segmentation dataset for
  generalisability assessment. Scientific Data 10 (1), pp. 75. Cited by:
  [14th
  item](#S4.I1.i14.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[4\] Anonymous (2025) SAM 3: segment anything with concepts. In
  Submitted to The Fourteenth International Conference on Learning
  Representations, Note: under review External Links:
  [Link](https://openreview.net/forum?id=r35clVtGzw) Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[5\] M. Antonelli, A. Reinke, S. Bakas, K. Farahani, A.
  Kopp-Schneider, B. A. Landman, G. Litjens, B. Menze, O.
  Ronneberger, R. M. Summers, et al. (2022) The medical segmentation
  decathlon. Nature communications 13 (1), pp. 4128. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[6\] F. Bai, Y. Du, T. Huang, M. Q. Meng, and B. Zhao (2024) M3d:
  advancing 3d medical image analysis with multi-modal large language
  models. arXiv preprint arXiv:2404.00578. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[7\] V. I. Butoi, J. J. G. Ortiz, T. Ma, M. R. Sabuncu, J. Guttag,
  and A. V. Dalca (2023) Universeg: universal medical image
  segmentation. In Proceedings of the IEEE/CVF International Conference
  on Computer Vision, pp. 21438–21451. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[8\] J. C. Caicedo, A. Goodman, K. W. Karhohs, B. A. Cimini, J.
  Ackerman, M. Haghighi, C. Heng, T. Becker, M. Doan, C. McQuin, et
  al. (2019) Nucleus segmentation across imaging experiments: the 2018
  data science bowl. Nature methods 16 (12), pp. 1247–1253. Cited by:
  [7th
  item](#S4.I1.i7.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[9\] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M.
  Wang (2022) Swin-unet: unet-like pure transformer for medical image
  segmentation. In European conference on computer vision, pp. 205–218.
  Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[10\] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L.
  Yuille, and Y. Zhou (2021) Transunet: transformers make strong
  encoders for medical image segmentation. arXiv preprint
  arXiv:2102.04306. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[11\] J. Cheng, J. Ye, Z. Deng, J. Chen, T. Li, H. Wang, Y. Su, Z.
  Huang, J. Chen, L. Jiang, H. Sun, J. He, S. Zhang, M. Zhu, and Y.
  Qiao (2023) SAM-med2d. External Links: 2308.16184,
  [Link](https://arxiv.org/abs/2308.16184) Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[12\] Y. Chu, G. Luo, L. Zhou, S. Cao, G. Ma, X. Meng, J. Zhou, C.
  Yang, D. Xie, D. Mu, et al. (2025) Deep learning-driven pulmonary
  artery and vein segmentation reveals demography-associated vasculature
  anatomical differences. Nature Communications 16 (1), pp. 2262. Cited
  by: [11st
  item](#S4.I1.i11.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[13\] N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S.
  Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, et
  al. (2019) Skin lesion analysis toward melanoma detection 2018: a
  challenge hosted by the international skin imaging collaboration
  (isic). arXiv preprint arXiv:1902.03368. Cited by: [5th
  item](#S4.I1.i5.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[14\] E. de la Rosa, R. Su, M. Reyes, R. Wiest, E. O. Riedel, F.
  Kofler, K. Yang, H. Baazaoui, D. Robben, S. Wegener, et al. (2024)
  ISLES’24: improving final infarct prediction in ischemic stroke using
  multimodal imaging and clinical data. arXiv preprint arXiv:2408.10966.
  Cited by: [13rd
  item](#S4.I1.i13.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[15\] B. Dong, W. Wang, D. Fan, J. Li, H. Fu, and L. Shao (2021)
  Polyp-pvt: polyp segmentation with pyramid vision transformers. arXiv
  preprint arXiv:2108.06932. Cited by:
  [§4.2](#S4.SS2.p1.1 "4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[16\] Y. Du, F. Bai, T. Huang, and B. Zhao (2024) Segvol: universal
  and interactive volumetric medical image segmentation. Advances in
  Neural Information Processing Systems 37, pp. 110746–110783. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[17\] A. Fallahpour, J. Ma, A. Munim, H. Lyu, and B. Wang (2025)
  Medrax: medical reasoning agent for chest x-ray. arXiv preprint
  arXiv:2502.02673. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[18\] H. Fang, F. Li, H. Fu, J. Wu, X. Zhang, and Y. Xu (2022)
  Dataset and evaluation algorithm design for goals challenge. In
  International Workshop on Ophthalmic Medical Image Analysis,
  pp. 135–142. Cited by: [3rd
  item](#S4.I1.i3.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[19\] F. Fumero, S. Alayón, J. L. Sanchez, J. Sigut, and M.
  Gonzalez-Hernandez (2011) RIM-one: an open retinal image database for
  optic nerve evaluation. In 2011 24th international symposium on
  computer-based medical systems (CBMS), pp. 1–6. Cited by: [4th
  item](#S4.I1.i4.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[20\] Gemini Team, Google (2025) A new era of intelligence with
  gemini 3. Technical report Google DeepMind. Note: Accessed: 2025-11-23
  External Links: [Link](https://blog.google/products/gemini/gemini-3)
  Cited by:
  [§4.4](#S4.SS4.p1.1 "4.4 MedSAM-3 Agent Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [Table
  3](#S4.T3.3.1.5.2 "In 4.4 MedSAM-3 Agent Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[21\] S. Han, P. Xia, R. Zhang, T. Sun, Y. Li, H. Zhu, and H.
  Yao (2025) Mdocagent: a multi-modal multi-agent framework for document
  understanding. arXiv preprint arXiv:2503.13964. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[22\] A. Hatamizadeh, H. Hosseini, N. Patel, J. Choi, C. C.
  Pole, C. M. Hoeferlin, S. D. Schwartz, and D. Terzopoulos (2022)
  RAVIR: a dataset and methodology for the semantic segmentation and
  quantitative analysis of retinal arteries and veins in infrared
  reflectance imaging. IEEE Journal of Biomedical and Health Informatics
  26 (7), pp. 3272–3283. Cited by: [8th
  item](#S4.I1.i8.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[23\] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D.
  Xu (2021) Swin unetr: swin transformers for semantic segmentation of
  brain tumors in mri images. In International MICCAI brainlesion
  workshop, pp. 272–284. Cited by:
  [§4.2](#S4.SS2.p1.1 "4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[24\] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B.
  Landman, H. R. Roth, and D. Xu (2022) Unetr: transformers for 3d
  medical image segmentation. In Proceedings of the IEEE/CVF winter
  conference on applications of computer vision, pp. 574–584. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[25\] H. Huang, L. Lin, R. Tong, H. Hu, Q. Zhang, Y. Iwamoto, X.
  Han, Y. Chen, and J. Wu (2020) Unet 3+: a full-scale connected unet
  for medical image segmentation. In ICASSP 2020-2020 IEEE international
  conference on acoustics, speech and signal processing (ICASSP),
  pp. 1055–1059. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§4.2](#S4.SS2.p1.1 "4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[26\] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H.
  Maier-Hein (2021) NnU-net: a self-configuring method for deep
  learning-based biomedical image segmentation. Nature methods 18 (2),
  pp. 203–211. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§4.2](#S4.SS2.p1.1 "4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[27\] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. De
  Lange, D. Johansen, and H. D. Johansen (2019) Kvasir-seg: a segmented
  polyp dataset. In International conference on multimedia modeling,
  pp. 451–462. Cited by: [9th
  item](#S4.I1.i9.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[28\] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L.
  Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W. Lo, et al. (2023)
  Segment anything. In Proceedings of the IEEE/CVF international
  conference on computer vision, pp. 4015–4026. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[29\] T. Koleilat, H. Asgariandehkordi, H. Rivaz, and Y. Xiao (2024)
  Medclip-sam: bridging text and image towards universal medical image
  segmentation. In International conference on medical image computing
  and computer-assisted intervention, pp. 643–653. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[30\] N. Kumar, R. Verma, D. Anand, Y. Zhou, O. F. Onder, E.
  Tsougenis, H. Chen, P. Heng, J. Li, Z. Hu, et al. (2019) A multi-organ
  nucleus segmentation challenge. IEEE transactions on medical imaging
  39 (5), pp. 1380–1391. Cited by: [6th
  item](#S4.I1.i6.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[31\] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J.
  Jia (2024) Lisa: reasoning segmentation via large language model. In
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, pp. 9579–9589. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[32\] W. Lei, W. Xu, K. Li, X. Zhang, and S. Zhang (2025) MedLSAM:
  localize and segment anything model for 3d ct images. Medical Image
  Analysis 99, pp. 103370. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[33\] B. Li, T. Yan, Y. Pan, J. Luo, R. Ji, J. Ding, Z. Xu, S.
  Liu, H. Dong, Z. Lin, et al. (2024) Mmedagent: learning to use medical
  tools with multi-modal agent. arXiv preprint arXiv:2407.02483. Cited
  by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[34\] C. Li, C. Wong, S. Zhang, N. Usuyama, H. Liu, J. Yang, T.
  Naumann, H. Poon, and J. Gao (2023) Llava-med: training a large
  language-and-vision assistant for biomedicine in one day. Advances in
  Neural Information Processing Systems 36, pp. 28541–28564. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[35\] M. Li, M. Meng, S. Ye, M. Fulham, L. Bi, and J. Kim (2024)
  Language-guided medical image segmentation with target-informed
  multi-level contrastive alignments. arXiv preprint arXiv:2412.13533.
  Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[36\] Z. Li, Y. Li, Q. Li, P. Wang, D. Guo, L. Lu, D. Jin, Y. Zhang,
  and Q. Hong (2023) Lvit: language meets vision transformer in medical
  image segmentation. IEEE transactions on medical imaging 43 (1),
  pp. 96–107. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[37\] G. Litjens, R. Toth, W. Van De Ven, C. Hoeks, S. Kerkstra, B.
  Van Ginneken, G. Vincent, G. Guillard, N. Birbeck, J. Zhang, et
  al. (2014) Evaluation of prostate segmentation algorithms for mri: the
  promise12 challenge. Medical image analysis 18 (2), pp. 359–373. Cited
  by: [12nd
  item](#S4.I1.i12.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[38\] C. Liu, H. Ding, and X. Jiang (2023) Gres: generalized
  referring expression segmentation. In Proceedings of the IEEE/CVF
  conference on computer vision and pattern recognition,
  pp. 23592–23601. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[39\] J. Liu, Y. Zhang, J. Chen, J. Xiao, Y. Lu, B. A Landman, Y.
  Yuan, A. Yuille, Y. Tang, and Z. Zhou (2023) Clip-driven universal
  model for organ segmentation and tumor detection. In Proceedings of
  the IEEE/CVF international conference on computer vision,
  pp. 21152–21164. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[40\] G. Luo, K. Wang, J. Liu, S. Li, X. Liang, X. Li, S. Gan, W.
  Wang, S. Dong, W. Wang, et al. (2023) Efficient automatic segmentation
  for multi-level pulmonary arteries: the parse challenge. arXiv
  preprint arXiv:2304.03708. Cited by: [10th
  item](#S4.I1.i10.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[41\] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang (2024) Segment
  anything in medical images. Nature Communications 15 (1), pp. 654.
  Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [Table
  3](#S4.T3.3.1.3.1 "In 4.4 MedSAM-3 Agent Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[42\] J. Ma, F. Li, and B. Wang (2024) U-mamba: enhancing long-range
  dependency for biomedical image segmentation. arXiv preprint
  arXiv:2401.04722. Cited by:
  [§4.2](#S4.SS2.p1.1 "4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[43\] J. Ma, Z. Yang, S. Kim, B. Chen, M. Baharoon, A. Fallahpour, R.
  Asakereh, H. Lyu, and B. Wang (2025) Medsam2: segment anything in 3d
  medical images and videos. arXiv preprint arXiv:2504.03600. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[44\] F. Milletari, N. Navab, and S. Ahmadi (2016) V-net: fully
  convolutional neural networks for volumetric medical image
  segmentation. In 2016 fourth international conference on 3D vision
  (3DV), pp. 565–571. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[45\] M. Moor, O. Banerjee, Z. S. H. Abad, H. M. Krumholz, J.
  Leskovec, E. J. Topol, and P. Rajpurkar (2023) Foundation models for
  generalist medical artificial intelligence. Nature 616 (7956),
  pp. 259–265. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[46\] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K.
  Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, et al. (2018)
  Attention u-net: learning where to look for the pancreas. arXiv
  preprint arXiv:1804.03999. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[47\] N. Ravi, V. Gabeur, Y. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R.
  Rädle, C. Rolland, L. Gustafson, et al. (2024) Sam 2: segment anything
  in images and videos. arXiv preprint arXiv:2408.00714. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[48\] Z. Ren, Z. Huang, Y. Wei, Y. Zhao, D. Fu, J. Feng, and X.
  Jin (2024) Pixellm: pixel reasoning with large multimodal model. In
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition, pp. 26374–26383. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[49\] O. Ronneberger, P. Fischer, and T. Brox (2015) U-net:
  convolutional networks for biomedical image segmentation. In
  International Conference on Medical image computing and
  computer-assisted intervention, pp. 234–241. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§4.2](#S4.SS2.p1.1 "4.2 Experimental settings ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [Table
  3](#S4.T3.3.1.2.1 "In 4.4 MedSAM-3 Agent Performance ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[50\] D. Surís, S. Menon, and C. Vondrick (2023) Vipergpt: visual
  inference via python execution for reasoning. In Proceedings of the
  IEEE/CVF international conference on computer vision, pp. 11888–11898.
  Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[51\] A. M. Tahir, M. E. Chowdhury, A. Khandakar, T. Rahman, Y.
  Qiblawey, U. Khurshid, S. Kiranyaz, N. Ibtehaz, M. S. Rahman, S.
  Al-Maadeed, et al. (2021) COVID-19 infection localization and severity
  grading from chest x-ray images. Computers in biology and medicine
  139, pp. 105002. Cited by: [1st
  item](#S4.I1.i1.p1.1.1 "In 4.1 Datasets ‣ 4 Experiments and Results ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[52\] N. Team, B. Zhang, S. Feng, X. Yan, J. Yuan, Z. Yu, X. He, S.
  Huang, S. Hou, Z. Nie, et al. (2025) NovelSeek: when agent becomes the
  scientist–building closed-loop system from hypothesis to verification.
  arXiv preprint arXiv:2505.16938. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[53\] H. Wang, S. Guo, J. Ye, Z. Deng, J. Cheng, T. Li, J. Chen, Y.
  Su, Z. Huang, Y. Shen, et al. (2025) SAM-med3d: a vision foundation
  model for general-purpose segmentation on volumetric medical images.
  IEEE Transactions on Neural Networks and Learning Systems. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[54\] C. Wu, X. Zhang, Y. Zhang, H. Hui, Y. Wang, and W. Xie (2025)
  Towards generalist foundation model for radiology by leveraging
  web-scale 2d&3d medical data. Nature Communications 16 (1), pp. 7866.
  Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[55\] J. Wu, Z. Wang, M. Hong, W. Ji, H. Fu, Y. Xu, M. Xu, and Y.
  Jin (2025) Medical sam adapter: adapting segment anything model for
  medical image segmentation. Medical image analysis 102, pp. 103547.
  Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[56\] Z. Yang, J. Wang, Y. Tang, K. Chen, H. Zhao, and P. H.
  Torr (2022) Lavt: language-aware vision transformer for referring
  image segmentation. In Proceedings of the IEEE/CVF conference on
  computer vision and pattern recognition, pp. 18155–18165. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[57\] Z. Yang, L. Li, K. Lin, J. Wang, C. Lin, Z. Liu, and L.
  Wang (2023) The dawn of lmms: preliminary explorations with gpt-4v
  (ision). arXiv preprint arXiv:2309.17421. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[58\] H. Yao, R. Zhang, J. Huang, J. Zhang, Y. Wang, B. Fang, R.
  Zhu, Y. Jing, S. Liu, G. Li, et al. (2025) A survey on agentic
  multimodal large language models. arXiv preprint arXiv:2510.10991.
  Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[59\] Y. Yuan, W. Li, J. Liu, D. Tang, X. Luo, C. Qin, L. Zhang,
  and J. Zhu (2024) Osprey: pixel understanding with visual instruction
  tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition, pp. 28202–28211. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[60\] D. Zhang, W. Liu, Q. Tan, J. Chen, H. Yan, Y. Yan, J. Li, W.
  Huang, X. Yue, W. Ouyang, et al. (2024) Chemllm: a chemical large
  language model. arXiv preprint arXiv:2402.06852. Cited by:
  [§2](#S2.p2.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[61\] T. Zhao, Y. Gu, J. Yang, N. Usuyama, H. H. Lee, S. Kiblawi, T.
  Naumann, J. Gao, A. Crabtree, J. Abel, et al. (2025) A foundation
  model for joint segmentation, detection and recognition of biomedical
  objects across nine modalities. Nature methods 22 (1), pp. 166–176.
  Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[62\] T. Zhao, Y. Gu, J. Yang, N. Usuyama, H. H. Lee, T. Naumann, J.
  Gao, A. Crabtree, J. Abel, C. Moung-Wen, et al. (2024) Biomedparse: a
  biomedical foundation model for image parsing of everything everywhere
  all at once. arXiv preprint arXiv:2405.12971. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[63\] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J.
  Liang (2018) Unet++: a nested u-net architecture for medical image
  segmentation. In International workshop on deep learning in medical
  image analysis, pp. 3–11. Cited by:
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
- \[64\] J. Zhu, A. Hamdi, Y. Qi, Y. Jin, and J. Wu (2024) Medical sam
  2: segment medical images as video via segment anything model 2. arXiv
  preprint arXiv:2408.00874. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ MedSAM3: Delving into Segment Anything with Medical Concepts"),
  [§2](#S2.p1.1 "2 Related Works ‣ MedSAM3: Delving into Segment Anything with Medical Concepts").
