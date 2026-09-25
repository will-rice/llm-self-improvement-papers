---
identifier: arxiv:2510.06638v3
title: "StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"
authors:
  - Zhihao Wen
  - Wenkang Wei
  - Yuan Fang
  - Xingtong Yu
  - Hui Zhang
  - Weicheng Zhu
  - Xin Zhang
published: "2025-10-08T04:37:53+00:00"
url: https://arxiv.org/abs/2510.06638v3
source: arxiv
doi: null
arxiv_id: 2510.06638v3
categories:
  - cs.AI
  - cs.CV
---

# StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering

Zhihao Wen ^(†)^(†)thanks: These authors contributed equally.
Affiliation: Ant International, Ant Group    Wenkang Wei¹¹footnotemark:
1 Affiliation: School of Computer Science and Technology, University of
Science and Technology of China    Yuan Fang Affiliation: School of
Computing and Information Systems, Singapore Management University   
Xingtong Yu Affiliation: School of Computing and Information Systems,
Singapore Management University    Hui Zhang ^(†)^(†)thanks:
Corresponding authors. Affiliation: School of Computer Science and
Technology, University of Science and Technology of China
Affiliation: Anhui Provincial Key Laboratory of High Performance
Computing    Weicheng Zhu Affiliation: Ant International, Ant Group   
Xin Zhang²²footnotemark: 2 Email: [{z.wen, weicheng.zhu,
evan.zx}@antgroup.comyizhilouyi@mail.ustc.edu.cn, {yfang,
xingtongyu}@smu.edu.sg, fzhh@ustc.edu.cn](mailto:) Affiliation: Ant
International, Ant Group

###### Abstract

Knowledge-based Visual Question Answering (KVQA) requires models to
ground entities in images and reason over factual knowledge. Recent work
has introduced its implicit-knowledge variant, _IK-KVQA_, where a
multimodal large language model (MLLM) is the sole knowledge source and
answers are produced without external retrieval. Existing IK-KVQA
approaches, however, are typically trained with answer-only supervision:
reasoning remains implicit, justifications are often weak or
inconsistent, and generalization after standard supervised fine-tuning
(SFT) can be brittle. We propose StaR-KVQA, a framework that equips
IK-KVQA with _dual-path structured reasoning traces_—symbolic relation
paths over text and vision together with path-grounded natural-language
explanations—to provide a stronger inductive bias than generic
answer-only supervision. These traces act as modality-aware scaffolds
that guide the model toward relevant entities and attributes, offering
more structure than generic chain-of-thought supervision while not
constraining reasoning to any single fixed path. With a single
open-source MLLM, StaR-KVQA constructs and selects traces to build an
offline trace-enriched dataset and then performs structure-aware
self-distillation; no external retrievers, verifiers, or curated
knowledge bases are used, and inference is a single autoregressive pass.
Across benchmarks, StaR-KVQA consistently improves both answer accuracy
and the transparency of intermediate reasoning, achieving up to +11.3%
higher answer accuracy on OK-VQA over the strongest baseline.

## 1 Introduction

Knowledge-based Visual Question Answering (KVQA) targets real-world
scenarios where users ask questions about images that require factual
knowledge beyond what is explicitly visible, and thus sits at the
intersection of computer vision, natural language processing, and
knowledge reasoning \[[46](#bib.bib1), [34](#bib.bib2),
[39](#bib.bib3)\]. Unlike conventional VQA that often learns a direct
mapping from image features to textual answers, KVQA additionally
requires _grounding entities in the image_ and _linking them to relevant
knowledge_. For example, answering _“Which breed of dog is this?”_
involves recognizing visual cues (e.g., color, size) and associating
them with prior knowledge about dog breeds. In practical deployments,
however, KVQA systems are often constrained by privacy/compliance,
latency/cost, and reliability requirements, which can limit heavy
external-retrieval pipelines and motivate more self-contained solutions.
The challenge is therefore not only perceiving pixels and text, but also
_organizing and using knowledge_ in a way that improves answer quality
under these constraints.

![Refer to caption](2510.06638v3/setting-cropped.png)

Figure 1: Traditional KVQA vs. implicit-knowledge KVQA (IK-KVQA).
Traditional KVQA often relies on external knowledge sources (e.g.,
retrieval or KGs) on top of a perception backbone. In contrast, IK-KVQA
retains the “K” to emphasize its knowledge-based nature while removing
external sources: answers are predicted solely from $`(I,Q)`$ and
parametric knowledge $`f_{\theta}(I,Q)`$.

Early KVQA systems often rely on explicit knowledge graphs (KGs) or
retrieval modules \[[10](#bib.bib4)\]. While effective, such pipelines
are not always a good fit for _high-throughput_ deployment. First,
external retrieval can introduce privacy/compliance risks when user
images, queries, or extracted entities must be sent to third-party
services or stored in external indices. Second, retrieval and evidence
fusion incur non-trivial latency/cost at scale, and performance can
fluctuate with index freshness, domain shift, or infrastructure
constraints. Third, multi-stage designs reduce reliability and
debuggability: errors in recognition or retrieval propagate, and
evidence fusion can be brittle, making failures harder to attribute and
audit. These constraints motivate _implicit-knowledge KVQA (IK-KVQA)_
\[[58](#bib.bib16)\], where the task remains knowledge-based but
external sources are disallowed: multimodal large language models
(MLLMs)¹¹ 1 In the literature, models such as Qwen2.5-VL are also
referred to as VLMs. We use the term _MLLMs_ in this paper for
consistency. must answer directly from $`(I,Q)`$ by leveraging
_parametric_ knowledge, as illustrated in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
Importantly, IK-KVQA is not meant to replace KB/RAG-based KVQA; rather,
it captures a practically common regime where the system must be
_self-contained_, _cost-effective_, and _auditable_. In this stricter
setting, the bottleneck shifts from retrieving knowledge to _eliciting,
structuring, and validating_ the model’s internal knowledge so that it
supports accurate predictions without relying on opaque shortcuts.

Challenges and Our Approach. The IK-KVQA setting simplifies system
design and removes external dependencies, but also imposes stricter
demands: the model must rely solely on its parameters to ground
evidence, recall factual knowledge, and reason. In practice, MLLMs often
behave as _black boxes_—sometimes producing correct answers while
intermediate descriptions are _underspecified, weakly grounded, or
inconsistent_. The absence of explicit, structured supervision on
intermediate steps complicates analysis and can affect reliability.
Concretely, IK-KVQA faces three core challenges: (1) _Lack of explicit
supervision_, since models are typically trained only on final answers
while reasoning traces remain hidden; (2) _Underspecified intermediate
signals_, where predictions may lack consistently aligned stepwise
descriptions; and (3) _Potential overfitting_, as conventional
fine-tuning can bias toward in-domain patterns with reduced robustness
beyond the training distribution.

To address these issues, we propose StaR-KVQA, which equips MLLMs with
_dual-path structured reasoning traces_. Instead of leaving reasoning
implicit, StaR-KVQA supervises both symbolic relation paths and
natural-language explanations, so training better reflects how models
should connect visual cues with internal knowledge. We use _relation
paths_ as planning scaffolds: relations are more stable than surface
entities, share a compact ontology across text and vision, and align
naturally with object- and scene-level attributes. These traces serve as
_soft plans_ that highlight salient entities/attributes while keeping
generation flexible.

StaR-KVQA reuses a single open-source MLLM (e.g., Qwen2.5-VL-7B) to
generate dual relation paths, compose explanations, and select the most
consistent triplet, yielding an augmented dataset with explicit traces.
Fine-tuning on it performs _structure-aware self-distillation_, learning
from both final answers and intermediate signals (paths + explanations).
This supervision provides a stronger inductive bias, reducing shortcut
reliance and improving accuracy. At inference, the fine-tuned model
generates traces and answers in a single autoregressive pass without
external knowledge. Overall, StaR-KVQA extends self-distillation to
multimodal reasoning by distilling _structured intermediate reasoning_
rather than answer-only outputs. In summary, our contributions are
threefold:

- •
  Structured supervision for IK-KVQA. We introduce StaR-KVQA, replacing
  answer-only supervision with _structured reasoning traces_: dual
  symbolic relation paths and path-grounded explanations as
  modality-aware scaffolds, without constraining reasoning to a single
  path.
- •
  Single-model, dependency-free pipeline. We develop an
  implementation-friendly pipeline—_dual-path planner_, _reasoning
  composer_, and an _internal selector_ instantiated by the same
  model—to construct trace-enriched data for _structure-aware
  self-distillation_. The system uses a single open-source MLLM, adds no
  retrievers/verifiers or extra trainable modules, and keeps inference
  to one pass.
- •
  Empirical gains in accuracy and transparency. Fine-tuning on
  trace-enriched data consistently improves accuracy and
  intermediate-trace transparency across benchmarks (e.g., up to +11.3%
  on OK-VQA over the strongest baseline), while remaining fully in the
  no-retrieval setting.

## 2 Related Work

We review KVQA with retrieval, KVQA with LLMs / MLLMs, and
self-distillation, and position our contributions.

KVQA with knowledge graphs or retrieval. Early datasets (FVQA
\[[46](#bib.bib1)\], OK-VQA \[[34](#bib.bib2), [40](#bib.bib8)\], KVQA
\[[41](#bib.bib9)\]) spurred pipelines that integrate explicit KGs or
retrievers, e.g., ConceptBERT \[[18](#bib.bib10)\], MAVEx
\[[51](#bib.bib11)\], and KRISP \[[33](#bib.bib12)\]. More recent
retrieval-augmented systems such as Wiki-LLaVA \[[8](#bib.bib13)\],
RoRA-VLM \[[37](#bib.bib14)\], and EchoSight \[[55](#bib.bib15)\]
further demonstrate the value of external knowledge, but introduce
pipeline complexity, error propagation, and maintenance costs with
limited transparency. Most still rely on answer supervision, fusing
retrieved facts into hidden representations rather than supervising
explicit, auditable reasoning traces.

KVQA with LLMs / MLLMs. To reduce reliance on explicit KGs, LLMs have
been used as implicit knowledge engines: PICa \[[58](#bib.bib16)\] shows
GPT-3 \[[7](#bib.bib17)\] can answer knowledge-intensive questions from
captions; KAT \[[19](#bib.bib6)\] and REVIVE \[[28](#bib.bib7)\] add
supporting evidence, while MAIL \[[14](#bib.bib48)\] and ReflectiVA
\[[11](#bib.bib18)\] explore reflective or adaptive fusion. However,
reasoning traces in these systems are often absent or weakly grounded,
limiting interpretability and fine-grained control of knowledge use.
Recent MLLMs perform end-to-end image–text reasoning via lightweight
projections \[[30](#bib.bib19), [29](#bib.bib20)\], Q-Former-style
modules \[[27](#bib.bib21), [13](#bib.bib22)\], Perceiver-style encoders
\[[26](#bib.bib24)\], or cross-attention as in Flamingo
\[[3](#bib.bib23), [4](#bib.bib25)\]. Training typically combines
large-scale caption alignment \[[9](#bib.bib27), [17](#bib.bib28),
[26](#bib.bib24)\] with visual instruction tuning \[[25](#bib.bib26)\].
In the IK-KVQA regime, their reasoning remains largely implicit and
weakly supervised.

Self-distillation and reasoning supervision. Self-distillation
\[[61](#bib.bib46), [60](#bib.bib47)\] uses model outputs as auxiliary
supervision; SDFT \[[56](#bib.bib45)\] rewrites responses to mitigate
forgetting, and [Wang et al. \[47\]](#bib.bib50) add structural signals
for multi-hop QA. Yet most works distill only answers, leaving reasoning
implicit. In contrast, we distill _structured reasoning traces_—dual
symbolic paths and path-grounded explanations—as explicit supervision
for IK-KVQA in a single-model, parametric-only setting, enabling
self-distillation with more transparent intermediate reasoning.

![Refer to caption](2510.06638v3/frame_1113-cropped.png)

Figure 2: Overview of StaR-KVQA. Given a training image–text pair, a
single $`\mathrm{MLLM}_{\phi}`$ generates multiple dual relation paths
(a) and corresponding explanations (b). A selector (c) identifies the
most consistent triplet, which, combined with the ground-truth answer,
forms reasoning-augmented supervision (d). The fine-tuned
$`f_{\theta}^{\prime}`$ then performs single-pass inference (e),
producing reasoning traces and answers without external knowledge. An
example dual-path scaffold is shown, highlighting relevant visual
attributes and semantic priors; paths need not be minimal or sufficient
but guide the model toward useful evidence before composing a full
explanation.

## 3 Preliminaries

We formalize the IK-KVQA setting, and task notation.

Problem definition. Given an image $`I`$ and a question $`Q`$,
knowledge-based visual question answering (KVQA) aims to predict an
answer $`\hat{a}\in\mathcal{A}`$:

|     |                                    |     |     |
| --- | ---------------------------------- | --- | --- |
|     | $`\displaystyle\hat{a}=f(I,Q,K),`$ |     | (1) |

where $`f`$ is the answering model and $`K`$ denotes external knowledge
retrieved from a knowledge graph or textual corpus. Traditional KVQA
pipelines typically ground entities in the image and query $`K`$ to
supplement factual reasoning.

Implicit-knowledge KVQA. In contrast, we consider the implicit-knowledge
setting (IK-KVQA), where $`K`$ is unavailable. The only information
sources are (i) visual evidence from $`I`$, (ii) linguistic cues from
$`Q`$, and (iii) _parametric knowledge_ encoded in model parameters.
Under this setting, the answer is predicted as

|     |                                           |     |     |
| --- | ----------------------------------------- | --- | --- |
|     | $`\displaystyle\hat{a}=f_{\theta}(I,Q),`$ |     | (2) |

where $`f_{\theta}`$ is trained solely without external retrieval. This
formulation removes external dependencies but leaves intermediate
reasoning largely implicit. Our framework addresses this gap by
augmenting supervision with explicit reasoning traces (dual paths and
natural-language explanations), keeping all computation within the
single model.

## 4 Structured Reasoning Traces for IK-KVQA

We introduce StaR-KVQA, which replaces answer-only supervision with
_structured reasoning traces_: dual relation paths $`(P_{t},P_{v})`$ and
a path-grounded explanation $`C`$. This converts reasoning from an
implicit by-product into explicit, structured training signals. The
entire pipeline runs within a single $`\mathrm{MLLM}_{\phi}`$, which
generates paths, composes explanations, and selects the best triplet,
all without external retrievers, verifiers, or curated knowledge bases
(Figure [2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")).

Design principles. (i) _Inductive structure:_ relation paths act as
low-dimensional planning scaffolds for cross-modal reasoning; (ii)
_Trace-level consistency:_ the
paths$`\rightarrow`$explanation$`\rightarrow`$answer pipeline encourages
alignment between intermediate traces and final predictions; (iii)
_Single-family learning:_ generation and supervision remain
style-aligned via self-distillation within the same MLLM family.

### 4.1 Dual-Path Planner

To explicitly structure reasoning across linguistic and visual
modalities, we design a dual-path planner that generates symbolic
_relation paths_. These paths capture semantic relations between
entities and attributes, and are commonly used in knowledge-graph
reasoning due to their stability and interpretability
\[[43](#bib.bib60), [54](#bib.bib61), [44](#bib.bib62),
[49](#bib.bib63), [50](#bib.bib64)\]. Unlike dynamic entities, relations
are more stable and reusable, making them low-dimensional, discrete
surrogates for reasoning scaffolds rather than exact proofs.

Formally, given an image–question pair $`(I,Q)`$, the frozen backbone
$`\mathrm{MLLM}_{\phi}`$ generates $`K`$ candidate path pairs:

|     |                                                                                        |     |     |
| --- | -------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\{(P_{t}^{(k)},P_{v}^{(k)})\}_{k=1}^{K}=\mathrm{Planner}_{\phi}(I,Q),`$ |     | (3) |

where each $`(P_{t}^{(k)},P_{v}^{(k)})`$ consists of: (1) a _text path_
$`P_{t}^{(k)}`$ capturing semantic associations from $`Q`$ and
linguistic priors, and (2) a _vision path_ $`P_{v}^{(k)}`$ encoding
attributes and relations grounded in $`I`$.

These dual paths serve as _soft planning hints_: they guide which
entities and attributes to consider, without rigidly constraining the
reasoning chain. And the downstream reasoning composer
(Section [4.2](#S4.SS2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"))
can incorporate additional cues or skip redundant links for final
answers.

We operationalize _plan-then-solve_ ideas for IK-KVQA by planning
_internally_ over relation paths within a single-model setup, avoiding
external KGs or retrieval. Unlike prior plan-first prompting
\[[45](#bib.bib51)\] or KG path reasoning \[[32](#bib.bib52)\], our
planner unifies textual priors and visual attributes into dual relation
paths inside a single-model pipeline. This approach provides multiple
candidate routes and acts as an _inductive bias_, narrowing the search
space while maintaining auditable intermediate steps.

For example, consider the question “Which breed of dog is this?” with
image $`I`$. One candidate might be: “$`P_{v}^{(k)}`$: dog.color →
dog.coat*length → dog.size, $`P*{t}^{(k)}`$: dog.size → dog.breed_group
→ dog.breed,” as shown in
Figure [2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")
(a). These complementary relation paths connect visual cues with
semantic priors, steering the model away from label-memorization
shortcuts. In practice, paths can include redundant or mildly spurious
hops; we treat them as noisy but useful scaffolds. The best-triplet
selector and self-distillation
(Sections [4.3](#S4.SS3 "4.3 Best-Triplet Selector ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")–[4.4](#S4.SS4 "4.4 Training with Augmented Data ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"))
further refine these paths, preferring those that consistently support
correct answers.

### 4.2 Reasoning Composer

Given a dual-path pair $`(P_{t}^{k},P_{v}^{k})`$, the reasoning composer
turns abstract plans into natural-language _reasoning content_ $`C^{k}`$
using the same backbone:

|     |                                                                           |     |     |
| --- | ------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle C^{k}=\mathrm{Compose}_{\phi}(I,Q,P_{t}^{k},P_{v}^{k}).`$ |     | (4) |

We build on evidence that explanations can act as supervision:
VQA-NLE-style rationales improve answer quality and interpretability
\[[42](#bib.bib54), [21](#bib.bib55), [52](#bib.bib56)\];
chain-of-thought prompts in ScienceQA induce more structured reasoning
\[[31](#bib.bib57), [63](#bib.bib58)\]; and explicit clues or
explanation–answer agreement (DCLUB, MCLE) reduce shortcutting and
inconsistency \[[16](#bib.bib59), [24](#bib.bib53)\].

Our composer instantiates these insights _within the IK setting_ by
explicitly _binding_ the rationale to the proposed paths. During trace
construction, we instruct $`\mathrm{Compose}_{\phi}`$ to (i) mention at
least one attribute or relation token from $`P_{v}^{k}`$ and (ii)
include at least one semantic hop from $`P_{t}^{k}`$ in $`C^{k}`$. We
then compute a simple coverage score between the tokenized explanation
and the path tokens, and discard candidates with very low coverage
(e.g., no overlap on either path). The best-triplet selector
(Section [4.3](#S4.SS3 "4.3 Best-Triplet Selector ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"))
is applied after this filtering and further down-weights explanations
that only weakly cite elements from $`P_{t}^{k}`$/$`P_{v}^{k}`$. This
conditioning discourages free-form but ungrounded narratives and keeps
explanations focused on the entities and attributes used in the symbolic
plan. This binding makes explanations easier to audit against the paths
while still allowing additional cues. In practice, it turns explanation
quality into a path-aware supervision signal that the single-model
system can learn from, rather than treating interpretability as a purely
post-hoc by-product.

### 4.3 Best-Triplet Selector

Not all triplets $`(P_{t}^{k},P_{v}^{k},C^{k})`$ are reliable, and
directly using them may introduce noisy or inconsistent supervision. We
therefore introduce a best-triplet selector that filters candidates
during the _data augmentation stage_, where dual paths and reasoning
contents are turned into training signals.

The selector is instantiated as an _LLM-as-a-judge_ within the same
single-model setup, reusing $`\mathrm{MLLM}_{\phi}`$. Given $`(I,Q)`$
and a set of candidates $`\{(P_{t}^{k},P_{v}^{k},C^{k})\}_{k=1}^{K}`$,
we prompt $`\mathrm{MLLM}_{\phi}`$ to rank triplets according to three
criteria: (i) _answer-oriented path consistency_ (the answer naturally
follows from the explanation and paths), (ii) _internal coherence and
conciseness_, and (iii) _path citation_ (explicitly mentioning elements
from $`P_{t}^{k}`$/$`P_{v}^{k}`$). The primary objective is answer
quality: the selector prefers triplets for which the predicted answer is
well supported by the textual explanation and dual paths, while
faithfulness is encouraged but not enforced as a hard constraint.
Formally, with score $`s_{\phi}`$:

|     |                             |                                                                                    |     |     |
| --- | --------------------------- | ---------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle b^{*}`$     | $`\displaystyle=\underset{b}{\arg\max}\;s_{\phi}(I,Q,P_{t}^{b},P_{v}^{b},C^{b}),`$ |     |     |
|     | $`\displaystyle T_{b^{*}}`$ | $`\displaystyle=(P_{t}^{b^{*}},P_{v}^{b^{*}},C^{b^{*}}).`$                         |     | (5) |

This step adds _no additional trainable parameters_. Reusing the same
backbone keeps $`\mathcal{D}_{\text{aug}}`$ style-aligned with the
model’s own traces. The selected triplet reflects what the MLLM
_internally_ finds most helpful for answering the question—possibly not
the most intuitive chain for humans, but empirically providing stronger
supervision in a lightweight, fully parametric pipeline.

Why single-model trace construction (vs. extra modules)? We avoid extra
verifiers or retrievers because: (i) _Homogeneous generation–learning:_
planning, composing, and selecting all use the same model family, so the
student $`f_{\theta}`$ learns from in-family traces, mitigating
supervision–generation mismatch and catastrophic forgetting
\[[56](#bib.bib45)\]; (ii) _Test-time simplicity:_ selection is used
only offline during augmentation, leaving inference as a single
autoregressive pass; (iii) _IK compliance:_ the design stays fully
parametric, with no external knowledge or modules.

### 4.4 Training with Augmented Data

Let the training split be
$`\{(I_{tr}^{i},Q_{tr}^{i},a_{tr}^{i})\}_{i=1}^{N}`$, where
$`I_{tr}^{i}`$ is the image, $`Q_{tr}^{i}`$ the question, and
$`a_{tr}^{i}`$ the ground-truth answer. For each pair
$`(I_{tr}^{i},Q_{tr}^{i})`$, the planner and composer generate multiple
candidate triplets, and the selector chooses the best one
$`T^{i}_{b^{*}}=(P_{t}^{b^{*}},P_{v}^{b^{*}},C^{b^{*}})`$. We then
construct the augmented training set:

|     |                                                                                                           |     |     |
| --- | --------------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\mathcal{D}_{\text{aug}}=\{(I_{tr}^{i},Q_{tr}^{i},T^{i}_{b^{*}},a_{tr}^{i})\}_{i=1}^{N}.`$ |     | (6) |

The base model $`f_{\theta}`$ is fine-tuned on
$`\mathcal{D}_{\text{aug}}`$ with a token-level cross-entropy loss:

|     |                                                                                                                                                      |     |     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\mathcal{L}_{\text{SFT}}(\theta;\mathcal{D}_{\text{aug}})=-\sum_{(I,Q,T,a)\in\mathcal{D}_{\text{aug}}}\log p_{\theta}(T,a\mid I,Q),`$ |     | (7) |

where the target sequence concatenates the reasoning paths
$`P_{t},P_{v}`$, reasoning content $`C`$, and the final answer $`a`$.
This objective encourages the fine-tuned model $`f_{\theta}^{\prime}`$
to jointly generate structured reasoning traces and correct answers in a
single coherent output.

### 4.5 Single-pass Inference

At test time, given $`(I_{te},Q_{te})`$, the fine-tuned model
$`f_{\theta}^{\prime}`$ performs a _single_ autoregressive decode that
jointly emits dual paths, a path-grounded explanation, and the final
answer:

|     |                                                                                                 |     |     |
| --- | ----------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle f_{\theta}^{\prime}(I_{te},Q_{te})=(\hat{P}_{t},\hat{P}_{v},\hat{C},\hat{a}).`$ |     | (8) |

No selector or auxiliary module is invoked at inference, and the
paths$`\rightarrow`$explanation$`\rightarrow`$answer structure directly
exposes a complete trace for auditing, without any external retrieval.

|                                                |                                        |                                              |                |
| ---------------------------------------------- | -------------------------------------- | -------------------------------------------- | -------------- |
|    Method                                      |    Model Inputs                        |    External Knowledge                        |    Acc. (%)    |
|    Q Only                                      |    Question + Image                    |    -                                         |    14.93       |
|    KVQA with Knowledge Graphs and Retrieval    |                                        |                                              |                |
|    BAN                                         |    Question + Image                    |    -                                         |    25.17       |
|    BAN +AN                                     |    Question + Image                    |    Wikipedia                                 |    25.61       |
|    MUTAN                                       |    Question + Image                    |    -                                         |    26.41       |
|    MUTAN +AN                                   |    Question + Image                    |    Wikipedia                                 |    27.84       |
|    ConceptBERT                                 |    Question + Image                    |    ConceptNet                                |    33.66       |
|    HCNMN                                       |    Question + Image                    |    WordNet                                   |    36.74       |
|    Krisp                                       |    Question + Image                    |    Wikipedia + ConceptNet                    |    38.90       |
|    MAVEx                                       |    Question + Image                    |    Wikipedia + ConceptNet + Google Images    |    41.37       |
|    VLC-BERT                                    |    Question + Image                    |    COMET + ConceptNet                        |    43.14       |
|    MCAN                                        |    Question + Image                    |    -                                         |    44.65       |
|    KVQA with LLMs / MLLMs                      |                                        |                                              |                |
|    PICA-Base                                   |    Question + Caption + Object Tags    |    Frozen GPT-3 (175B)                       |    43.30       |
|    Pica-Full                                   |    Question + Caption + Object Tags    |    Frozen GPT-3 (175B)                       |    48.00       |
|    KAT (Single)                                |    Question + Caption + Object Tags    |    Frozen GPT-3 (175B) + Wikidata            |    53.09       |
|    KAT (Ensemble)                              |    Question + Caption + Object Tags    |    Frozen GPT-3 (175B) + Wikidata            |    54.41       |
|    REVIVE                                      |    Question + Caption + Region Tags    |    Frozen GPT-3 (175B) + Wikidata            |    53.83       |
|    MAIL                                        |    Question + Image                    |    Frozen MiniGPT-4 (7B) + ConceptNet        |    56.69       |
|    IK-KVQA with MLLMs                          |                                        |                                              |                |
|    Qwen2.5-VL-7B                               |    Question + Image                    |    Qwen2.5-VL-7B                             |    75.74       |
|    Llama-3.2-11B-Vision                        |    Question + Image                    |    Llama-3.2-11B-Vision                      |    67.84       |
|    Gemma-3-12B                                 |    Question + Image                    |    Gemma-3-12B                               |    71.40       |
|    Gemma-3-27B                                 |    Question + Image                    |    Gemma-3-27B                               |    79.34       |
|    Qwen2.5-VL-72B                              |    Question + Image                    |    Qwen2.5-VL-72B                            |    80.75       |
|    InternVL3-78B                               |    Question + Image                    |    InternVL3-78B                             |    67.61       |
|    GPT-4o                                      |    Question + Image                    |    GPT-4o                                    |    77.86       |
|    Gemini 2.5 Flash                            |    Question + Image                    |    Gemini 2.5 Flash                          |    79.97       |
|    Gemini 2.5 Pro                              |    Question + Image                    |    Gemini 2.5 Pro                            |    80.53       |
|    SFT                                         |    Question + Image                    |    Fine-tuned Qwen2.5-VL-7B                  |    76.36       |
|    CoT                                         |    Question + Image                    |    Qwen2.5-VL-7B                             |    76.88       |
|    CoT + SFT                                   |    Question + Image                    |    Fine-tuned Qwen2.5-VL-7B                  |    79.58       |
|    LLaVA-CoT                                   |    Question + Image                    |    Fine-tuned Llama-3.2-11B-Vision           |    76.57       |
|    M2-Reasoning                                |    Question + Image                    |    M2-Reasoning-7B                           |    78.63       |
|    SDFT                                        |    Question + Image                    |    Fine-tuned Qwen2.5-VL-7B                  |    82.56       |
|    StaR-KVQA\_(Qwen)                           |    Question + Image                    |    Fine-tuned Qwen2.5-VL-7B                  |    91.51       |
|    StaR-KVQA\_(Llama)                          |    Question + Image                    |    Fine-tuned Llama-3.2-11B-Vision           |    90.01       |
|    StaR-KVQA\_(Gemma)                          |    Question + Image                    |    Fine-tuned Gemma-3-12B                    |    91.90       |

Table 1: Performance comparison on OK-VQA.

## 5 Experiments

We conduct extensive experiments to evaluate StaR-KVQA, with comparison
to state-of-the-art baselines and in-depth model analysis.

### 5.1 Experimental Setup

Datasets. In line with recent advances in the field \[[34](#bib.bib2),
[57](#bib.bib5), [19](#bib.bib6), [51](#bib.bib11), [28](#bib.bib7)\],
we performed our primary validation on the OK-VQA dataset. Comprising
14,055 image-question pairs, this benchmark is currently the most
demanding in the domain. Furthermore, to establish the broader
applicability of our model, we performed supplementary experiments on
FVQA \[[46](#bib.bib1)\], which initiated the exploration of KVQA.

Baselines. Three categories. (i) KVQA+KG/Retrieval: Q
Only \[[34](#bib.bib2)\], BAN \[[23](#bib.bib29)\],
MUTAN \[[6](#bib.bib30)\], ConceptBERT \[[18](#bib.bib10)\],
KRISP \[[33](#bib.bib12)\], MAVEx \[[51](#bib.bib11)\],
VLCBERT \[[38](#bib.bib31)\], HCNMN \[[62](#bib.bib32)\],
MCAN \[[59](#bib.bib33)\]; BAN/MUTAN use ArticleNet \[[34](#bib.bib2)\].
(ii) KVQA+LLMs: PICa \[[57](#bib.bib5)\], KAT \[[19](#bib.bib6)\],
REVIVE \[[28](#bib.bib7)\]. (iii) IK-KVQA+MLLMs: open-source
Qwen2.5-VL-7B/72B \[[5](#bib.bib34)\],
Llama-3.2-11B-Vision \[[15](#bib.bib35)\],
Gemma-3-12B/27B \[[22](#bib.bib36)\],
InternVL3-78B \[[64](#bib.bib37)\]; proprietary Gemini 2.5
Flash/Pro \[[12](#bib.bib38)\], GPT-4o \[[20](#bib.bib39)\];
SFT \[[36](#bib.bib42)\], CoT \[[48](#bib.bib43)\],
LLaVA-CoT \[[53](#bib.bib44)\], M2-Reasoning (7B) \[[2](#bib.bib41)\],
SDFT \[[56](#bib.bib45)\]. SFT, CoT, CoT + SFT, and SDFT are augmented
on Qwen2.5-VL-7B. “CoT + SFT” is a well-optimized CoT-prompted SFT
baseline. All MLLMs are instruction-tuned. Results for (i)(ii) follow
[Dong et al. \[14\]](#bib.bib48).

Protocol: Seed 42; default decoding; no CoT is used unless specified ;
inputs=(image, question) only for IK-KVQA approaches; single-run
reporting per [Dong et al. \[14\]](#bib.bib48).

Implementation: StaR-KVQA is trained in PyTorch 2.7.0, Python 3.10 on
NVIDIA L20; batch (accum.) $`16`$, LR $`1\mathrm{e}{-4}`$, LoRA rank
$`32`$, alpha $`64`$, $`3`$ epochs; $`K{=}3`$ (OK-VQA), $`K{=}4`$
(FVQA).The backbone MLLMs include Qwen2.5-VL-7B, Llama-3.2-11B-Vision,
and Gemma-3-12B.

Metric: Direct-answer VQA accuracy \[[1](#bib.bib40)\]:
$`\text{Acc}=\min\left(\frac{\#\text{humans with that answer}}{3},1\right).`$
Normalize (lowercase, digits, remove punctuation/articles).

|     Method                   |     Acc. (%)     |
| ---------------------------- | ---------------- |
|     Qwen2.5-VL-7B            |     71.61        |
|     Llama-3.2-11B-Vision     |     66.09        |
|     Gemma-3-12B              |     70.64        |
|     Gemma-3-27B              |     76.82        |
|     Qwen2.5-VL-72B           |     75.95        |
|     InternVL3-78B            |     70.99        |
|     GPT-4o                   |     72.36        |
|     Gemini 2.5 Flash         |     74.51        |
|     Gemini 2.5 Pro           |     73.39        |
|     SFT                      |     73.91        |
|     CoT                      |     74.66        |
|     CoT + SFT                |     75.13        |
|     LLaVA-CoT                |     78.45        |
|     M2-Reasoning             |     72.53        |
|     SDFT                     |     75.54        |
|     StaR-KVQA\_(Qwen)        |     82.82        |
|     StaR-KVQA\_(Llama)       |     80.19        |
|     StaR-KVQA\_(Gemma)       |     81.20        |

Table 2: Performance comparison of IK-KVQA with MLLMs approaches on
FVQA.

| Variants       | Vision         | Text           | Reasoning      | Best-Triplet   | OK-VQA |       |       | FVQA  |       |       | Average |
| -------------- | -------------- | -------------- | -------------- | -------------- | ------ | ----- | ----- | ----- | ----- | ----- | ------- |
|                | Path           | Path           | Composer       | Selector       | Qwen   | Llama | Gemma | Qwen  | Llama | Gemma |         |
| No paths       | $`\times`$     | $`\times`$     | $`\checkmark`$ | $`\checkmark`$ | 87.47  | 72.57 | 89.09 | 76.31 | 76.31 | 79.14 | 80.15   |
| No content     | $`\checkmark`$ | $`\checkmark`$ | $`\times`$     | $`\checkmark`$ | 87.53  | 86.00 | 88.26 | 76.22 | 76.05 | 73.34 | 81.23   |
| No text path   | $`\checkmark`$ | $`\times`$     | $`\checkmark`$ | $`\checkmark`$ | 83.66  | 72.77 | 87.84 | 76.91 | 56.91 | 79.66 | 76.29   |
| No vision path | $`\times`$     | $`\checkmark`$ | $`\checkmark`$ | $`\checkmark`$ | 92.65  | 70.01 | 86.92 | 74.42 | 64.81 | 78.20 | 77.84   |
| No selector    | $`\checkmark`$ | $`\checkmark`$ | $`\checkmark`$ | $`\times`$     | 91.76  | 72.17 | 91.94 | 84.55 | 49.18 | 83.18 | 78.80   |
| StaR-KVQA      | $`\checkmark`$ | $`\checkmark`$ | $`\checkmark`$ | $`\checkmark`$ | 91.51  | 90.01 | 91.90 | 82.82 | 80.19 | 81.20 | 86.27   |

Table 3: Ablation studies.

### 5.2 Main Results

We report comparisons with representative baselines in
Table [1](#S4.T1 "Table 1 ‣ 4.5 Single-pass Inference ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")
and
Table [2](#S5.T2 "Table 2 ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
Several key observations emerge: (i) MLLMs as strong backbones. Methods
based on state-of-the-art multimodal large language models (MLLMs)
achieve the strongest overall performance, even without explicit
external knowledge. This confirms that parametric knowledge acquired in
large-scale pretraining is already highly effective for KVQA, while
being simpler to use than retrieval- or KG-based approaches. (ii)
StaR-KVQA achieves the best results. Among the MLLM-based methods, our
reasoning-augmented framework consistently delivers the best
performance. On OK-VQA, it surpasses the strongest baseline by up to
+11.3%, highlighting the effectiveness of augmenting training with
_structured reasoning traces_. (iii) Closed-source models are strong but
surpassed. Closed-source commercial systems achieve competitive results
but still fall short of our approach. Notably, StaR-KVQA outperforms
Gemini 2.5 Pro, one of the most advanced multimodal reasoning models in
our comparison. (iv) Self-distillation is strong but limited. We also
evaluate Self-Distillation Fine-Tuning (SDFT) \[[56](#bib.bib45)\],
which rewrites task responses into the model’s own style for
fine-tuning. With Qwen2.5-VL-7B as the backbone, SDFT already exceeds
Gemini 2.5 Pro by over 2% on OK-VQA and ranks just below our method,
underscoring the strength of self-distillation in the IK-KVQA regime.
StaR-KVQA goes further: by supervising both symbolic paths and
natural-language explanations as _structured reasoning traces_, it
retains SDFT’s accuracy gains while providing more transparent
intermediate reasoning.

In summary, StaR-KVQA not only surpasses strong open-source and
closed-source baselines but also sets a new state of the art in IK-KVQA,
combining superior accuracy with more interpretable reasoning behavior.

### 5.3 Ablation Studies & Hyperparameters

Ablation studies. We conduct ablations to examine the role of each
component in our reasoning-augmented framework
(Table [3](#S5.T3 "Table 3 ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")).
For each variant, we regenerate the augmented training data and retrain
the model so that the reported performance reflects the absence of the
removed component. Removing either the dual paths (no paths) or the
explanations (no content) leads to clear accuracy drops, confirming that
symbolic paths and natural-language reasoning provide complementary
supervision. Restricting the framework to a single modality (vision-only
or text-only) further degrades performance, underscoring the need to
align textual priors with visual grounding. Replacing the best-triplet
selector with random selection (no selector) yields mixed outcomes: it
can slightly improve Qwen and Gemma on some datasets but severely harms
LLaMA, indicating that the selector is important for robustness across
backbones even if random choice occasionally preserves strong
candidates. Overall, these ablations show that dual paths, reasoning
content, and the selector all contribute and, in combination, explain
why the full StaR-KVQA model delivers strong and balanced performance
across benchmarks.

Hyperparameters. We next study the sensitivity to the number of
candidate paths $`K`$, using Qwen2.5-VL-7B as the backbone. For each
$`K`$, we report both answer accuracy and the average time cost of
running the full StaR-KVQA augmentation pipeline per training example.
As shown in
Figure [3](#S5.F3 "Figure 3 ‣ 5.3 Ablation Studies & Hyperparameters ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
increasing $`K`$ initially improves performance by providing richer
reasoning options, but when $`K`$ becomes too large (e.g., $`K=5`$),
accuracy drops due to overly long contexts that hinder the selector.
Overall, the framework is not highly sensitive to $`K`$, and since
augmentation time grows roughly linearly while gains quickly saturate, a
moderate choice such as $`K=3`$ offers the best trade-off between
efficiency and effectiveness.

Efficiency.
Figure [3](#S5.F3 "Figure 3 ‣ 5.3 Ablation Studies & Hyperparameters ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")
also shows that the end-to-end construction of the augmented training
set is computationally affordable: on a single-node server equipped with
L20 GPUs and using vLLM for inference, the pipeline requires roughly 1–2
seconds per example on average. This overhead is modest, making the
proposed trace-construction procedure feasible for deployment in
real-world production settings.

(a) OK-VQA

(b) FVQA

Figure 3: $`K`$, the number of candidate paths.

|                    |                          |                |                             |                |
| ------------------ | ------------------------ | -------------- | --------------------------- | -------------- |
|                    | In-domain generalization |                | Cross-domain generalization |                |
| Source (Tuning)    | OK-VQA                   | FVQA           | OK-VQA                      | FVQA           |
| Target (Testing)   | OK-VQA                   | FVQA           | FVQA                        | OK-VQA         |
| Frozen\_(Qwen)     | 75.74                    | 71.61          | 71.61                       | 75.74          |
| SFT\_(Qwen)        | 76.36 (+0.62)            | 73.91 (+2.30)  | 64.77 (-6.84)               | 67.50 (-8.24)  |
| StaR-KVQA\_(Qwen)  | 91.51 (+15.77)           | 82.82 (+11.21) | 82.09 (+10.48)              | 85.45 (+9.71)  |
| Frozen\_(Llama)    | 67.84                    | 66.09          | 66.09                       | 67.84          |
| SFT\_(Llama)       | 75.30 (+7.46)            | 74.68 (+8.59)  | 63.45 (-2.64)               | 64.19 (-3.65)  |
| StaR-KVQA\_(Llama) | 90.01 (+22.17)           | 80.19 (+14.10) | 80.09 (+14.00)              | 79.59 (+11.75) |
| Frozen\_(Gemma)    | 71.40                    | 70.64          | 70.64                       | 71.40          |
| SFT\_(Gemma)       | 74.45 (+3.05)            | 73.73 (+3.09)  | 66.83 (-3.81)               | 63.91 (-7.49)  |
| StaR-KVQA\_(Gemma) | 91.90 (+20.50)           | 81.20 (+10.56) | 81.20 (+10.56)              | 83.43 (+12.03) |

Table 4: Cross-domain generalization.

### 5.4 Cross-domain Generalization

Robust generalization to out-of-distribution (OOD) data is crucial in
real-world applications. We therefore evaluate both _in-domain_ and
_cross-domain_ generalization across OK-VQA and FVQA, using three model
variants: Frozen (backbone without fine-tuning), SFT (standard
supervised fine-tuning), and our proposed StaR-KVQA, each instantiated
with three MLLM backbones. In-domain generalization. When training and
testing on the same dataset (OK-VQA or FVQA), both SFT and StaR-KVQA
yield substantial gains over the Frozen model (left half of
Table [4](#S5.T4 "Table 4 ‣ 5.3 Ablation Studies & Hyperparameters ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")),
confirming that fine-tuning effectively adapts to the target domain. Our
framework further improves performance by explicitly supervising
intermediate reasoning. Cross-domain generalization. We then examine
transfer across datasets in both directions, OK-VQA $`\rightarrow`$ FVQA
and FVQA $`\rightarrow`$ OK-VQA, which induces a substantial
distribution shift. As shown in the right half of
Table [4](#S5.T4 "Table 4 ‣ 5.3 Ablation Studies & Hyperparameters ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
SFT often degrades sharply—and in some cases even underperforms the
Frozen baseline—highlighting vulnerability to catastrophic forgetting
and limited cross-domain robustness. In contrast, StaR-KVQA consistently
avoids such degradation and even improves performance on the unseen
domain, indicating stronger resistance to forgetting and superior
cross-domain generalization.

### 5.5 Qualitative Case Study

As shown in
[Tables 5](#S5.T5 "In 5.5 Qualitative Case Study ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")
and [6](#S5.T6 "Table 6 ‣ 5.5 Qualitative Case Study ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
StaR-KVQA differs from strong MLLMs by emitting _trace-first_ dual paths
that expose key entity$`\rightarrow`$relation hops before answering. In
the OK-VQA example, Gemma-3-12B and Gemini 2.5 Pro both produce
confident free-form narratives and predict the wrong “Persian Gulf”,
whereas StaR-KVQA follows a compact scaffold (ship.hull_number →
ship.name → location.island_group → ocean.name) and outputs the
annotator-favored atlantic. In the FVQA example, baselines over-focus on
the patterned “rug”, while StaR-KVQA reasons via human–object
_affordances_ and correctly selects sofa. Overall, these suggest that
StaR-KVQA reduces overconfident, opaque failure modes and texture-based
shortcuts by grounding predictions in short, relation-based scaffolds
linking visual cues and parametric knowledge to the final answer.

| ![[Uncaptioned image]](2510.06638v3/figures/img_2626515.JPEG)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Question: Name the sea shown in this picture where the ship is seen? Answers: \[atlantic, atlantic, atlantic, atlantic, dock of red sea, dock of red sea, kamchatka, kamchatka, philippine, philippine\] |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Answer from Gemma-3-12B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                                                                                          |
| Here’s a breakdown of the image and the answer to your question: 1. Object Identification and Attributes: \* Ships: Two large, grey ships are prominently featured. They appear to be military or transport vessels, with the designation “L3005” visible on their hulls. \* Containers: Numerous shipping containers in various colors (red, green, grey) are stacked near the ships and along the docks. \* Dock/Port Area: A large port or dock area is visible, with cranes, storage areas, and other infrastructure. \* Crane: A large gantry crane is present, likely used for loading and unloading cargo. \* Water: A body of water (sea or harbor) surrounds the ships and docks. \* Landscape: A flat, arid landscape is visible in the background. \* Weather: The sky is hazy and the lighting suggests a bright, sunny day. 2. Relationship Path & Reasoning: The question asks “Name the sea shown in this picture where the ship is seen?”. To answer this, we need to consider the context of the image. The presence of a large port, military-style ships, and the arid landscape provide clues. Based on the visual cues, this is likely the Persian Gulf. The arid landscape, the presence of a large port facility, and the type of ships are all consistent with the Persian Gulf region. 3. Answer: The sea shown in this picture is the Persian Gulf. |                                                                                                                                                                                                          |
| Answer from Gemini 2.5 Pro                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                                                                                                                                                          |
| Based on the visual evidence, the ship in the image can be identified, which helps pinpoint the location. The ship with the pennant number L3005 is the British Royal Fleet Auxiliary landing ship RFA Sir Galahad. This photograph was taken at the port of Umm Qasr, Iraq, likely around 2003 during humanitarian relief operations following the coalition invasion. The port of Umm Qasr is situated on the Khawr Abd Allah waterway, which flows into the Persian Gulf. Therefore, the sea connected to the port shown in the picture is the Persian Gulf.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                          |
| Answer from StaR-KVQA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                          |
| vision path: ship.hull_number $`\rightarrow`$ ship.name $`\rightarrow`$ location.island_group text path: location.island_group $`\rightarrow`$ location.ocean $`\rightarrow`$ ocean.name Here is one way to reason about this: 1. Identify the ship: The ship in the image has the hull number L3005, which identifies it as the Royal Fleet Auxiliary ship _Sir Galahad_. 2. Link the ship to a location: RFA Sir Galahad is known for its role in the Falklands War and was sunk near the Falkland Islands. Thus we can link the scene to the island group “Falkland Islands”. 3. Map the island group to a sea: The Falkland Islands are located in the South Atlantic Ocean. Therefore, the body of water in which this ship is seen is the South Atlantic (part of the Atlantic Ocean). Therefore, the possible answers include: atlantic, pacific                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                                                                                                                                                          |

Table 5: Qualitative case study example, in OKVQA

| ![[Uncaptioned image]](2510.06638v3/figures/img_1254.JPEG)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Question: Which object in this image is used for play? Answer: sofa |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Answer from Gemma-3-12B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |                                                                     |
| Based on the image, the object most likely used for play is the rug. Here’s why: \* Location: It’s placed on the floor in a living room setting, a common area for play. \* Pattern: The rug has a playful, abstract pattern that could be engaging for children. \* Function: Rugs are often used as play surfaces for kids. Let me know if you’d like me to analyze the image for other details!                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                                                     |
| Answer from Gemini 2.5 Pro                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                                                     |
| Based on the image provided, there are no explicit toys or games visible. However, the object most likely to be used for play, especially imaginative play for a child, is the rug. The rug features a pattern of stylized, repeating figures (which look like birds or other animals). Children often use such patterns on floors or rugs as a landscape for their games, moving toys around them or creating stories about the figures.                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                     |
| Answer from StaR-KVQA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                                                                     |
| vision path: scene.room_type $`\rightarrow`$ object.sofa $`\rightarrow`$ object.affordance.play text path: object.category $`\rightarrow`$ object.typical_use $`\rightarrow`$ play.activity Based on the image, the most plausible object used for play is the sofa. Visually, the scene appears to be a living room with a large, soft sofa that offers comfortable space for sitting, lying down, or jumping around, which are typical play activities, especially for children. Other visible objects are less likely to be directly associated with play: the rug mainly serves as a floor covering, the floor lamp provides lighting, the artwork is decorative, and the glass railing is a structural element for safety rather than play. The sofa, by contrast, affords relaxed and playful behavior in this setting. Therefore, the possible answers include: sofa. |                                                                     |

Table 6: Qualitative case study example, in FVQA

## 6 Conclusion

We presented StaR-KVQA, which supervises dual symbolic relation paths
and path-grounded explanations as _structured reasoning traces_, turning
reasoning from an implicit by-product into explicit, structured, and
more transparent intermediate steps for IK-KVQA. Implemented as an
implementation-friendly _single-model_ pipeline (_dual-path planner_ →
_reasoning composer_ → _best-triplet selector_, all with the same
open-source MLLM), it requires no external retrievers/verifiers or extra
trainable modules; traces are built offline and inference is a single
autoregressive pass. With structure-aware self-distillation on
trace-enriched data, StaR-KVQA attains strong answer accuracy and
improved transparency of intermediate reasoning across benchmarks,
achieving up to +11.3% on OK-VQA over the strongest baseline and
surpassing advanced closed-source systems (e.g., Gemini 2.5 Pro) under
the IK-KVQA setting. Remaining limitations include residual
hallucination inherited from the backbone and the fact that, since the
best-triplet selector is primarily optimized for answer-oriented
consistency, the selected traces are not guaranteed to be the most
intuitive or fully faithful explanations for humans. Faithfulness
between explanations and paths is enforced only through coverage-based
filtering and an LLM-as-a-judge selector, rather than formal guarantees,
so occasional mismatches remain. Future work includes plug-compatible
verification (e.g., retrieval-based checks or lightweight consistency
modules), explicit objectives for explanation faithfulness and human
preference, and broader cross-domain evaluations. Overall, StaR-KVQA
advances more transparent multimodal reasoning for IK-KVQA while
maintaining a simple, deployable pipeline.

## Acknowledgement

The authors wish to thank Dr. Junnan Dong from Tencent Youtu Lab and Dr.
Qinggang Zhang from Hong Kong Polytechnic University for their valuable
support of this work.

## References

- \[1\] A. Agrawal, J. Lu, S. Antol, M. Mitchell, C. L. Zitnick, D.
  Parikh, and D. Batra (2015) VQA: visual question answering.
  International Journal of Computer Vision 123, pp. 4 – 31. External
  Links: [Link](https://api.semanticscholar.org/CorpusID:3180429) Cited
  by:
  [§5.1](#S5.SS1.p5.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[2\] I. AI, F. Wang, J. Liu, J. Chen, J. Zhou, K. Ji, L. Ru, Q.
  Guo, R. Zheng, T. Li, et al. (2025) M2-reasoning: empowering mllms
  with unified general and spatial reasoning. arXiv preprint
  arXiv:2507.08306. Cited by: [5th
  item](#A1.I1.i3.I1.i5.p1.1 "In 3rd item ‣ Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[3\] J. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K.
  Lenc, A. Mensch, K. Millican, M. Reynolds, et al. (2022) Flamingo: a
  visual language model for few-shot learning. Advances in neural
  information processing systems 35, pp. 23716–23736. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[4\] A. Awadalla, I. Gao, J. Gardner, J. Hessel, Y. Hanafy, W.
  Zhu, K. Marathe, Y. Bitton, S. Gadre, S. Sagawa, et al. (2023)
  Openflamingo: an open-source framework for training large
  autoregressive vision-language models. arXiv preprint
  arXiv:2308.01390. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[5\] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P.
  Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P.
  Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H.
  Zhang, Z. Yang, H. Xu, and J. Lin (2025) Qwen2.5-vl technical report.
  ArXiv abs/2502.13923. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:276449796) Cited by:
  [1st
  item](#A1.I1.i1.p1.1 "In Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[6\] H. Ben-younes, R. Cadène, M. Cord, and N. Thome (2017) MUTAN:
  multimodal tucker fusion for visual question answering. 2017 IEEE
  International Conference on Computer Vision (ICCV), pp. 2631–2639.
  External Links:
  [Link](https://api.semanticscholar.org/CorpusID:12913776) Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[7\] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P.
  Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et
  al. (2020) Language models are few-shot learners. Advances in neural
  information processing systems 33, pp. 1877–1901. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[8\] D. Caffagni, F. Cocchi, N. Moratelli, S. Sarto, M. Cornia, L.
  Baraldi, and R. Cucchiara (2024) Wiki-llava: hierarchical
  retrieval-augmented generation for multimodal llms. In Proceedings of
  the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
  pp. 1818–1826. Cited by:
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[9\] S. Changpinyo, P. Sharma, N. Ding, and R. Soricut (2021)
  Conceptual 12m: pushing web-scale image-text pre-training to recognize
  long-tail visual concepts. In Proceedings of the IEEE/CVF conference
  on computer vision and pattern recognition, pp. 3558–3568. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[10\] Z. Chen, Y. Zhang, Y. Fang, Y. Geng, L. Guo, X. Chen, Q. Li, W.
  Zhang, J. Chen, Y. Zhu, J. Li, X. Liu, J. Z. Pan, N. Zhang, and H.
  Chen (2024) Knowledge graphs meet multi-modal learning: a
  comprehensive survey. ArXiv abs/2402.05391. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:267547866) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[11\] F. Cocchi, N. Moratelli, M. Cornia, L. Baraldi, and R.
  Cucchiara (2025) Augmenting multimodal llms with self-reflective
  tokens for knowledge-based visual question answering. In Proceedings
  of the Computer Vision and Pattern Recognition Conference,
  pp. 9199–9209. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[12\] G. Comanici, E. Bieber, and M. S. et al. (2025) Gemini 2.5:
  pushing the frontier with advanced reasoning, multimodality, long
  context, and next generation agentic capabilities. ArXiv
  abs/2507.06261. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:280151524) Cited by:
  [2nd
  item](#A1.I1.i2.p1.1 "In Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[13\] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N.
  Fung, and S. Hoi (2023) Instructblip: towards general-purpose
  vision-language models with instruction tuning. Advances in neural
  information processing systems 36, pp. 49250–49267. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[14\] J. Dong, Q. Zhang, H. Zhou, D. Zha, P. Zheng, and X.
  Huang (2024) Modality-aware integration with large language models for
  knowledge-based visual question answering. In Proceedings of the 62nd
  Annual Meeting of the Association for Computational Linguistics
  (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.),
  Bangkok, Thailand, pp. 2417–2429. External Links:
  [Link](https://aclanthology.org/2024.acl-long.132/),
  [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.132) Cited by:
  [§A.1](#A1.SS1.p2.1 "A.1 Implementation Details. ‣ Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [Appendix
  A](#A1.p2.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p3.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[15\] A. Dubey, A. Jauhri, and A. P. et al. (2024) The llama 3 herd
  of models. ArXiv abs/2407.21783. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:271571434) Cited by:
  [1st
  item](#A1.I1.i1.p1.1 "In Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[16\] X. Fu, B. Zhou, S. Chen, M. Yatskar, and D. Roth (2023) Dynamic
  clue bottlenecks: towards interpretable-by-design visual question
  answering. arXiv preprint arXiv:2305.14882. Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[17\] S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T.
  Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang, et al. (2023)
  Datacomp: in search of the next generation of multimodal datasets.
  Advances in Neural Information Processing Systems 36, pp. 27092–27112.
  Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[18\] F. Gardères, M. Ziaeefard, B. Abeloos, and F. Lecue (2020)
  Conceptbert: concept-aware representation for visual question
  answering. In Findings of the Association for Computational
  Linguistics: EMNLP 2020, pp. 489–498. Cited by: [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[19\] L. Gui, B. Wang, Q. Huang, A. G. Hauptmann, Y. Bisk, and J.
  Gao (2021) KAT: a knowledge augmented transformer for
  vision-and-language. ArXiv abs/2112.08614. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:245219268) Cited by:
  [Appendix
  A](#A1.p2.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p1.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[20\] O. A. Hurst, A. Lerer, and A. P. G. et al. (2024) GPT-4o system
  card. ArXiv abs/2410.21276. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:273662196) Cited by:
  [2nd
  item](#A1.I1.i2.p1.1 "In Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[21\] P. A. Irawan, G. I. Winata, S. Cahyawijaya, and A.
  Purwarianti (2024) Towards efficient and robust vqa-nle data
  generation with large vision-language models. In International
  Conference on Computational Linguistics, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:272826678) Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[22\] G. T. A. Kamath, J. Ferret, and S. P. et al. (2025) Gemma 3
  technical report. ArXiv abs/2503.19786. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:277313563) Cited by:
  [1st
  item](#A1.I1.i1.p1.1 "In Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[23\] J. Kim, J. Jun, and B. Zhang (2018) Bilinear attention
  networks. In Neural Information Processing Systems, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:29150617) Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[24\] C. Lai, S. Song, S. Meng, J. Li, S. Yan, and G. Hu (2023)
  Towards more faithful natural language explanation using multi-level
  contrastive learning in vqa. In AAAI Conference on Artificial
  Intelligence, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:266435704) Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[25\] H. Laurençon, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A.
  Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela, et al. (2023)
  Obelics: an open web-scale filtered dataset of interleaved image-text
  documents. Advances in Neural Information Processing Systems 36,
  pp. 71683–71702. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[26\] H. Laurençon, L. Tronchon, M. Cord, and V. Sanh (2024) What
  matters when building vision-language models?. Advances in Neural
  Information Processing Systems 37, pp. 87874–87907. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[27\] J. Li, D. Li, S. Savarese, and S. Hoi (2023) Blip-2:
  bootstrapping language-image pre-training with frozen image encoders
  and large language models. In International conference on machine
  learning, pp. 19730–19742. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[28\] Y. Lin, Y. Xie, D. Chen, Y. Xu, C. Zhu, and L. Yuan (2022)
  REVIVE: regional visual representation matters in knowledge-based
  visual question answering. ArXiv abs/2206.01201. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:249282486) Cited by:
  [Appendix
  A](#A1.p2.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p1.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[29\] H. Liu, C. Li, Y. Li, and Y. J. Lee (2024) Improved baselines
  with visual instruction tuning. In Proceedings of the IEEE/CVF
  conference on computer vision and pattern recognition,
  pp. 26296–26306. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[30\] H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023) Visual instruction
  tuning. Advances in neural information processing systems 36,
  pp. 34892–34916. Cited by:
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[31\] P. Lu, S. Mishra, T. Xia, L. Qiu, K. Chang, S. Zhu, O.
  Tafjord, P. Clark, and A. Kalyan (2022) Learn to explain: multimodal
  reasoning via thought chains for science question answering. ArXiv
  abs/2209.09513. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:252383606) Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[32\] L. Luo, Y. Li, G. Haffari, and S. Pan (2023) Reasoning on
  graphs: faithful and interpretable large language model reasoning.
  ArXiv abs/2310.01061. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:263605944) Cited by:
  [§4.1](#S4.SS1.p4.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[33\] K. Marino, X. Chen, D. Parikh, A. Gupta, and M. Rohrbach (2021)
  Krisp: integrating implicit and symbolic knowledge for open-domain
  knowledge-based vqa. In Proceedings of the IEEE/CVF conference on
  computer vision and pattern recognition, pp. 14111–14121. Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[34\] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi (2019)
  OK-vqa: a visual question answering benchmark requiring external
  knowledge. 2019 IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR), pp. 3190–3199. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:173991173) Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [Appendix
  E](#A5.p1.1 "Appendix E Data Ethics Statement ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§1](#S1.p1.1 "1 Introduction ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p1.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[35\] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J.
  Dean (2013) Distributed representations of words and phrases and their
  compositionality. In Neural Information Processing Systems, External
  Links: [Link](https://api.semanticscholar.org/CorpusID:16447573) Cited
  by: [Appendix
  B](#A2.p2.1 "Appendix B Metric ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[36\] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P.
  Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022)
  Training language models to follow instructions with human feedback.
  Advances in neural information processing systems 35, pp. 27730–27744.
  Cited by: [1st
  item](#A1.I1.i3.I1.i1.p1.1 "In 3rd item ‣ Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[37\] J. Qi, Z. Xu, R. Shao, Y. Chen, J. Di, Y. Cheng, Q. Wang,
  and L. Huang (2024) Rora-vlm: robust retrieval-augmented vision
  language models. arXiv preprint arXiv:2410.08876. Cited by:
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[38\] S. Ravi, A. Chinchure, L. Sigal, R. Liao, and V. Shwartz (2022)
  VLC-bert: visual question answering with contextualized commonsense
  knowledge. 2023 IEEE/CVF Winter Conference on Applications of Computer
  Vision (WACV), pp. 1155–1165. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:253107241) Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[39\] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R.
  Mottaghi (2022) A-okvqa: a benchmark for visual question answering
  using world knowledge. In European Conference on Computer Vision,
  External Links:
  [Link](https://api.semanticscholar.org/CorpusID:249375629) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[40\] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R.
  Mottaghi (2022) A-okvqa: a benchmark for visual question answering
  using world knowledge. In European conference on computer vision,
  pp. 146–162. Cited by:
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[41\] S. Shah, A. Mishra, N. Yadati, and P. P. Talukdar (2019) Kvqa:
  knowledge-aware visual question answering. In Proceedings of the AAAI
  conference on artificial intelligence, Vol. 33, pp. 8876–8884. Cited
  by:
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[42\] W. Suo, M. Sun, W. Liu, Y. Gao, P. Wang, Y. Zhang, and Q.
  Wu (2023) S3C: semi-supervised vqa natural language explanation via
  self-critical learning. 2023 IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR), pp. 2646–2656. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:260084857) Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[43\] H. Wang, H. Ren, and J. Leskovec (2021) Relational message
  passing for knowledge graph completion. In Proceedings of the 27th ACM
  SIGKDD conference on knowledge discovery & data mining, pp. 1697–1707.
  Cited by:
  [§4.1](#S4.SS1.p1.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[44\] J. Wang, B. Wang, M. Qiu, S. Pan, B. Xiong, H. Liu, L. Luo, T.
  Liu, Y. Hu, B. Yin, et al. (2023) A survey on temporal knowledge graph
  completion: taxonomy, progress, and prospects. arXiv preprint
  arXiv:2308.02457. Cited by:
  [§4.1](#S4.SS1.p1.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[45\] L. Wang, W. Xu, Y. Lan, Z. Hu, Y. Lan, R. K. Lee, and E.
  Lim (2023) Plan-and-solve prompting: improving zero-shot
  chain-of-thought reasoning by large language models. In Annual Meeting
  of the Association for Computational Linguistics, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:258558102) Cited by:
  [§4.1](#S4.SS1.p4.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[46\] P. Wang, Q. Wu, C. Shen, A. R. Dick, and A. van den
  Hengel (2016) FVQA: fact-based visual question answering. IEEE
  Transactions on Pattern Analysis and Machine Intelligence 40,
  pp. 2413–2427. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:7483388) Cited by:
  [Appendix
  E](#A5.p1.1 "Appendix E Data Ethics Statement ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§1](#S1.p1.1 "1 Introduction ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p1.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[47\] S. Wang, Z. Wei, J. Xu, T. Li, and Z. Fan (2023) Unifying
  structure reasoning and language pre-training for complex reasoning
  tasks. IEEE/ACM Transactions on Audio, Speech, and Language Processing
  32, pp. 1586–1595. Cited by:
  [§2](#S2.p4.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[48\] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V.
  Le, D. Zhou, et al. (2022) Chain-of-thought prompting elicits
  reasoning in large language models. Advances in neural information
  processing systems 35, pp. 24824–24837. Cited by: [2nd
  item](#A1.I1.i3.I1.i2.p1.1 "In 3rd item ‣ Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[49\] Z. Wen and Y. Fang (2023) Augmenting low-resource text
  classification with graph-grounded pre-training and prompting. In
  Proceedings of the 46th International ACM SIGIR Conference on Research
  and Development in Information Retrieval, pp. 506–516. Cited by:
  [§4.1](#S4.SS1.p1.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[50\] Z. Wen and Y. Fang (2024) Prompt tuning on graph-augmented
  low-resource text classification. IEEE Transactions on Knowledge and
  Data Engineering 36 (12), pp. 9080–9095. Cited by:
  [§4.1](#S4.SS1.p1.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[51\] J. Wu, J. Lu, A. Sabharwal, and R. Mottaghi (2022) Multi-modal
  answer validation for knowledge-based vqa. In Proceedings of the AAAI
  conference on artificial intelligence, Vol. 36, pp. 2712–2721. Cited
  by: [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p1.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[52\] J. Xie, Y. Cai, J. Chen, R. Xu, J. Wang, and Q. Li (2024)
  Knowledge-augmented visual question answering with natural language
  explanation. IEEE Transactions on Image Processing 33, pp. 2652–2664.
  External Links:
  [Link](https://api.semanticscholar.org/CorpusID:268739949) Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[53\] G. Xu, P. Jin, Z. Wu, H. Li, Y. Song, L. Sun, and L.
  Yuan (2024) Llava-cot: let vision language models reason step-by-step.
  arXiv preprint arXiv:2411.10440. Cited by: [4th
  item](#A1.I1.i3.I1.i4.p1.1 "In 3rd item ‣ Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[54\] X. Xu, P. Zhang, Y. He, C. Chao, and C. Yan (2022) Subgraph
  neighboring relations infomax for inductive link prediction on
  knowledge graphs. arXiv preprint arXiv:2208.00850. Cited by:
  [§4.1](#S4.SS1.p1.1 "4.1 Dual-Path Planner ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[55\] Y. Yan and W. Xie (2024) EchoSight: advancing visual-language
  models with wiki knowledge. arXiv preprint arXiv:2407.12735. Cited by:
  [§2](#S2.p2.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[56\] Z. Yang, Q. Liu, T. Pang, H. Wang, H. Feng, M. Zhu, and W.
  Chen (2024) Self-distillation bridges distribution gap in language
  model fine-tuning. In Annual Meeting of the Association for
  Computational Linguistics, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:267769989) Cited by:
  [6th
  item](#A1.I1.i3.I1.i6.p1.1 "In 3rd item ‣ Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p4.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§4.3](#S4.SS3.p3.1 "4.3 Best-Triplet Selector ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.2](#S5.SS2.p1.1 "5.2 Main Results ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[57\] Z. Yang, Z. Gan, J. Wang, X. Hu, Y. Lu, Z. Liu, and L.
  Wang (2021) An empirical study of gpt-3 for few-shot knowledge-based
  vqa. ArXiv abs/2109.05014. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:237485500) Cited by:
  [Appendix
  A](#A1.p2.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p1.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[58\] Z. Yang, Z. Gan, J. Wang, X. Hu, Y. Lu, Z. Liu, and L.
  Wang (2022) An empirical study of gpt-3 for few-shot knowledge-based
  vqa. In Proceedings of the AAAI conference on artificial intelligence,
  Vol. 36, pp. 3081–3089. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§2](#S2.p3.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[59\] Z. Yu, J. Yu, Y. Cui, D. Tao, and Q. Tian (2019) Deep modular
  co-attention networks for visual question answering. 2019 IEEE/CVF
  Conference on Computer Vision and Pattern Recognition (CVPR),
  pp. 6274–6283. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:195657908) Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[60\] L. Zhang, C. Bao, and K. Ma (2021) Self-distillation: towards
  efficient and compact neural networks. IEEE Transactions on Pattern
  Analysis and Machine Intelligence 44, pp. 4388–4403. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:232302458) Cited by:
  [§2](#S2.p4.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[61\] L. Zhang, J. Song, A. Gao, J. Chen, C. Bao, and K. Ma (2019) Be
  your own teacher: improve the performance of convolutional neural
  networks via self distillation. 2019 IEEE/CVF International Conference
  on Computer Vision (ICCV), pp. 3712–3721. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:159041406) Cited by:
  [§2](#S2.p4.1 "2 Related Work ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[62\] Y. Zhang, S. Chen, and Q. Zhao (2023) Toward multi-granularity
  decision-making: explicit visual reasoning with hierarchical
  knowledge. 2023 IEEE/CVF International Conference on Computer Vision
  (ICCV), pp. 2573–2583. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:267023103) Cited by:
  [Appendix
  A](#A1.p1.1 "Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[63\] Z. Zhang, A. Zhang, M. Li, H. Zhao, G. Karypis, and A. J.
  Smola (2023) Multimodal chain-of-thought reasoning in language models.
  Trans. Mach. Learn. Res. 2024. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:256504063) Cited by:
  [§4.2](#S4.SS2.p1.2 "4.2 Reasoning Composer ‣ 4 Structured Reasoning Traces for IK-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
- \[64\] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, Y. Duan, H.
  Tian, W. Su, J. Shao, Z. Gao, E. Cui, Y. Cao, Y. Liu, H. Wang, W.
  Xu, H. Li, J. Wang, H. Lv, D. Chen, S. Li, Y. He, T. Jiang, J. Luo, Y.
  Wang, C. He, B. Shi, X. Zhang, W. Shao, J. He, Y. Xiong, W. Qu, P.
  Sun, P. Jiao, L. Wu, K. Zhang, H. Deng, J. Ge, K. Chen, L. Wang, M.
  Dou, L. Lu, X. Zhu, T. Lu, D. Lin, Y. Qiao, J. Dai, and W. Wang (2025)
  InternVL3: exploring advanced training and test-time recipes for
  open-source multimodal models. ArXiv abs/2504.10479. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:277780955) Cited by:
  [1st
  item](#A1.I1.i1.p1.1 "In Appendix A Baselines ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering"),
  [§5.1](#S5.SS1.p2.1 "5.1 Experimental Setup ‣ 5 Experiments ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").

## Appendices

## Appendix A Baselines

KVQA with Knowledge Graphs and Retrieval. We select representative
state-of-the-art approaches, including direct question-only answering (Q
Only) \[[34](#bib.bib2)\], BAN \[[23](#bib.bib29)\], MUTAN
\[[6](#bib.bib30)\], ConceptBERT \[[18](#bib.bib10)\], KRISP
\[[33](#bib.bib12)\], MAVEx \[[51](#bib.bib11)\], VLCBERT
\[[38](#bib.bib31)\], HCNMN \[[62](#bib.bib32)\], and MCAN
\[[59](#bib.bib33)\]. Since BAN and MUTAN are limited to learning
unimodal visual features, we enhance them with ArticleNet (AN)
\[[34](#bib.bib2)\], which retrieves relevant information from Wikipedia
based on the given question–image pair to support external knowledge
reasoning. These enhanced versions are referred to as “BAN + AN” and
“MUTAN + AN” \[[34](#bib.bib2)\].

KVQA with LLMs / MLLMs. We employ PICa \[[57](#bib.bib5)\], KAT
\[[19](#bib.bib6)\], and REVIVE \[[28](#bib.bib7)\]. The results of KVQA
with Knowledge Graphs and Retrieval as well as KVQA with Large Language
Models are from prior work \[[14](#bib.bib48)\], where the exact same
experimental setup and evaluation protocols are adopted.

IK-KVQA with Multimodal Large Language Models. We employed three types
of Multimodal Large Language Models (MLLMs):

- •
  Advanced open-source MLLMs: Including three regular-sized models:
  Qwen2.5-VL-7B \[[5](#bib.bib34)\], Llama-3.2-11B-Vision
  \[[15](#bib.bib35)\], and Gemma-3-12B \[[22](#bib.bib36)\]; as well as
  three larger and more advanced models: Gemma-3-27B
  \[[22](#bib.bib36)\], Qwen2.5-VL-72B \[[5](#bib.bib34)\], and
  InternVL3-78B \[[64](#bib.bib37)\]. All of them are instruction-tuned
  versions.
- •
  Proprietary state-of-the-art MLLMs: Including two of Google’s most
  advanced models, Gemini 2.5 Flash and Gemini 2.5 Pro
  \[[12](#bib.bib38)\], as well as OpenAI’s flagship multimodal model,
  GPT-4o \[[20](#bib.bib39)\]. Both Gemini 2.5 Flash and Gemini 2.5 Pro
  perform inference in the Dynamic Thinking mode.
- •
  Augmented MLLMs:
  - –
    Supervised fine-tuning (SFT) \[[36](#bib.bib42)\] is a crucial
    process that trains a pre-trained MLLM on a high-quality dataset of
    instructions and responses, making it more effective at following
    specific commands and performing user-facing tasks. The MLLM
    backbone is Qwen2.5-VL-7B.
  - –
    Chain of Thought (CoT) \[[48](#bib.bib43)\] is a prompting technique
    that improves the reasoning abilities of large language models by
    guiding them to break down a complex problem into a series of
    intermediate steps before providing a final answer. The MLLM
    backbone is Qwen2.5-VL-7B.
  - –
    CoT + SFT is a well-optimized CoT-prompted SFT baseline.
  - –
    LLaVA-CoT \[[53](#bib.bib44)\], a new multimodal model that uses a
    chain-of-thought method to improve vision-language models’ ability
    to reason step-by-step.
  - –
    M2-Reasoning (7B) \[[2](#bib.bib41)\] is a multimodal large language
    model (MLLM) that achieves state-of-the-art (SOTA) performance in
    both general and spatial reasoning by using a high-quality data
    pipeline and a dynamic multi-task training strategy.
  - –
    Self-Distillation Fine-Tuning (SDFT) \[[56](#bib.bib45)\] rewrites
    task responses into its own style and fine-tunes on them to reduce
    distribution shift and forgetting. The MLLM backbone is
    Qwen2.5-VL-7B.

### A.1 Implementation Details.

Our approach StaR-KVQA has been implemented using PyTorch 2.7.0 as well
as Python 3.10, and all experiments have been conducted on the NVIDIA
L20 GPU. During training, the batch size (with accumulation) is set to
$`16`$, the learning rate is $`1\mathrm{e}{-4}`$, the LoRA rank is 32,
the LoRA alpha is 64, the traininig epoch is 3. In the OK-VQA dataset,
$`K`$ is set as 3, and in the FVQA dataset, $`K`$ is set as 4.

We adhere to the established evaluation setting and fix the random seed
to 42 throughout data loading, parameter initialization, and decoding.
Consistent with prior work \[[14](#bib.bib48)\], we report single-run
results in the main tables to maintain strict comparability with
published baselines. We did not sweep over seeds or report standard
deviations; we view multi-seed evaluation as complementary and leave it
to future extensions or large-scale replication studies.

To ensure a level playing field across closed- and open-source models,
we (i) supply only the image and the question as inputs, without
chain-of-thought or auxiliary prompts; and (ii) adopt each model’s
_default_ inference hyperparameters (decoding temperature and maximum
generation length), avoiding any model-specific tuning. This protocol
matches the default settings recommended by the model providers and
prevents gains from hyperparameter overfitting.

## Appendix B Metric

For the open-ended task, _i.e_., direct answer (DA) setting, we evaluate
generated answers using the following accuracy definition:

|     |                                                                                                           |     |     |
| --- | --------------------------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\text{Accuracy}=\min\left(\frac{\#\text{humans that provided that answer}}{3},\;1\right)`$ |     | (9) |

i.e., an answer is considered fully correct (100% accuracy) if it
matches the responses of at least three annotators. Before comparison,
all responses are normalized by lowercase, converting numbers to digits,
and removing punctuation and articles. We deliberately avoid soft
similarity measures such as Word2Vec \[[35](#bib.bib49)\], which may
incorrectly cluster semantically distinct words (e.g., “left” vs.
“right”). Likewise, we exclude machine translation metrics such as BLEU
and ROUGE, as they are mainly suited for multi-word sentence evaluation
rather than short answers typically found in VQA.

## Appendix C Theoretical Notes for StaR-KVQA

This appendix offers compact analyses that formalize how (i) typed,
path-grounded traces (planner + _reasoning composer_), (ii) the
single-model selector, and (iii) single-model self-distillation
contribute to StaR-KVQA. The statements are backbone-agnostic and match
the components introduced in Sec 4.

### C.1 Notation and Standing Assumptions

Let $`(I,Q,a^{\star})\sim\mathcal{D}`$ denote image, question, and
ground-truth answer. A _trace_ is $`T=(P_{t},P_{v},C)`$. Our model with
parameters $`\theta`$ induces

|     |                                   |                                      |                                           |                         |
| --- | --------------------------------- | ------------------------------------ | ----------------------------------------- | ----------------------- | -------------------- | --- | --- |
|     | $`\displaystyle p\_{\theta}(T,a\, | \,I,Q)`$                             | $`\displaystyle=p*{\theta}(P*{t},P\_{v}\, | \,I,Q)\;p\_{\theta}(C\, | \,I,Q,P*{t},P*{v})`$ |     |     |
|     |                                   | $`\displaystyle\quad p\_{\theta}(a\, | \,I,Q,T).`$                               |                         | (10)                 |

We reuse two structural predicates from Sec. 4.2:

|     |                                                                                           |     |      |
| --- | ----------------------------------------------------------------------------------------- | --- | ---- |
|     | $`\displaystyle\mathrm{Cover}(C;P_{t},P_{v})\geq\kappa,\qquad\mathrm{Vis}(C;I)\geq\rho,`$ |     | (11) |

encoding path–sentence coverage and visual attestability. Define the
feasible set
$`\mathcal{T}_{\kappa,\rho}=\{T:\mathrm{Cover}\geq\kappa,\ \mathrm{Vis}\geq\rho\}`$.

### C.2 Generalization Benefit from Typed and Verifiable Traces

We compare an _answer-only_ class with a _trace-constrained_ class that
must produce $`T\in\mathcal{T}_{\kappa,\rho}`$ alongside $`a`$.

#### Hypothesis classes.

Let $`\mathcal{H}_{\mathrm{ans}}=\{h:(I,Q)\mapsto a\}`$ and

|     |                                                                                                                    |     |      |
| --- | ------------------------------------------------------------------------------------------------------------------ | --- | ---- |
|     | $`\displaystyle\mathcal{H}_{\mathrm{trace}}=\{h:(I,Q)\mapsto(T,a)\ \text{s.t.}\ T\in\mathcal{T}_{\kappa,\rho}\}.`$ |     | (12) |

Both are realized by the _same_ architecture but trained with different
supervision.

###### Theorem 1 (Rademacher shrinkage via verifiable structure).

Assume bounded losses $`\ell(a,a^{\star})\in[0,1]`$ and
$`\ell_{\mathrm{trace}}(T,a;a^{\star})\in[0,1]`$ with
$`\ell_{\mathrm{trace}}(T,a;a^{\star})\geq\ell(a,a^{\star})`$ and
equality whenever $`T\in\mathcal{T}_{\kappa,\rho}`$. Then for any sample
size $`N`$ and $`\delta\in(0,1)`$, with probability at least
$`1-\delta`$,

|     |                                                                                                                                                                                                     |     |      |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ---- |
|     | $`\displaystyle\mathcal{R}_{\mathcal{D}}(h_{\mathrm{trace}})\leq\widehat{\mathcal{R}}_{N}(h_{\mathrm{trace}})+2\,\mathfrak{R}_{N}(\mathcal{H}_{\mathrm{trace}})+\sqrt{\tfrac{\ln(1/\delta)}{2N}},`$ |     | (13) |

and moreover
$`\mathfrak{R}_{N}(\mathcal{H}_{\mathrm{trace}})\leq\mathfrak{R}_{N}(\mathcal{H}_{\mathrm{ans}})\cdot\sqrt{\Pi(\mathcal{T}_{\kappa,\rho})/\Pi(\mathcal{T})}`$,
where $`\mathfrak{R}_{N}(\cdot)`$ is the empirical Rademacher complexity
and $`\Pi(\cdot)`$ the growth function.

Intuition. Enforcing typed, verifiable traces prunes implausible
labelings (fewer admissible traces per example), which lowers the
effective complexity term and tightens the bound. Practical takeaway.
Structure acts as an inductive bias without changing the backbone.

### C.3 Selector as Maximum Likelihood under a Consistency-Noise Model

Our best-triplet selector uses the _single-model_ setup to score
candidates. The score can be interpreted as a log-likelihood under a
simple noise model.

#### Model.

For candidate $`b`$, define binary indicators
$`Y_{b}^{(\text{ans})},Y_{b}^{(\text{ent})},Y_{b}^{(\text{align})},Y_{b}^{(\text{coh})}\in\{0,1\}`$
for answer correctness, explanation$`\Rightarrow`$answer entailment,
path$`\rightarrow`$explanation alignment, and explanation coherence.
Assume conditional independence given a latent quality $`q_{b}`$:

|     |                                                                                                              |     |      |
| --- | ------------------------------------------------------------------------------------------------------------ | --- | ---- |
|     | $`\displaystyle\Pr(Y_{b}^{(j)}=1\mid q_{b})=\sigma(w_{j}q_{b}),\qquad j\in\{\text{ans, ent, align, coh}\},`$ |     | (14) |

with logistic $`\sigma`$ and weights $`w_{j}>0`$. Let
$`\hat{y}_{b}^{(j)}\in[0,1]`$ be soft proxies estimated by the model;
the log-likelihood is
$`\log L_{b}(q_{b})=\sum_{j}\hat{y}_{b}^{(j)}\log\sigma(w_{j}q_{b})+(1-\hat{y}_{b}^{(j)})\log(1-\sigma(w_{j}q_{b}))`$.

###### Proposition 2 (Selector equals MLE/MAP ranking).

The maximizer $`\hat{q}_{b}=\arg\max_{q}\log L_{b}(q)`$ is monotone in
$`s_{\phi}(b):=\sum_{j}w_{j}(2\hat{y}_{b}^{(j)}-1)`$. Therefore
selecting $`b^{\star}=\arg\max_{b}s_{\phi}(b)`$ agrees with MLE (and
with MAP under any log-concave prior).

Intuition. The weighted consistency cues act like independent “votes.” A
larger weighted sum implies a larger MLE quality and thus a higher rank.
Practical takeaway. Our LLM-as-a-judge ranking matches likelihood-based
selection under a reasonable noise model.

### C.4 Single-Model Self-Distillation Reduces Supervision–Generation Shift

Let $`P`$ be the generator distribution over traces (from
$`\mathrm{MLLM}_{\phi}`$) and $`Q_{\theta}`$ the student’s distribution
after fine-tuning. Let $`\mathcal{L}\in[0,1]`$ be a bounded loss on
completions.

###### Lemma 1 (Risk gap upper bounded by divergence).

For any $`(I,Q)`$,

|     |                     |                                                                                         |                                                                   |
| --- | ------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --- | ---- |
|     | $`\displaystyle\big | \,\mathbb{E}_{T\sim P}\mathcal{L}(T)-\mathbb{E}_{T\sim Q\_{\theta}}\mathcal{L}(T)\,\big | \;\leq\;\sqrt{2\,\mathrm{KL}\!\left(P\,\|\,Q\_{\theta}\right)}.`$ |     | (15) |

###### Proof.

By total variation (TV) and Pinsker’s inequality:
$`|\,\mathbb{E}_{P}f-\mathbb{E}_{Q}f\,|\leq 2\,\mathrm{TV}(P,Q)`$ for
$`f\in[0,1]`$, and
$`\mathrm{TV}(P,Q)\leq\sqrt{\tfrac{1}{2}\mathrm{KL}(P\|Q)}`$. Combining
gives the stated bound. ∎

###### Theorem 3 (Self-distillation alignment).

If fine-tuning reduces $`\mathrm{KL}(P\|Q_{\theta})`$ on augmented
traces (i.e., the student learns from traces in the generator’s style),
the supervision–generation risk gap is
$`O(\sqrt{\mathrm{KL}(P\|Q_{\theta})})`$ by
Lemma [1](#Thmlemma1 "Lemma 1 (Risk gap upper bounded by divergence). ‣ C.4 Single-Model Self-Distillation Reduces Supervision–Generation Shift ‣ Appendix C Theoretical Notes for StaR-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering").
Using a _single-model_ setup (shared format/tokenization) typically
attains a smaller KL than heterogeneous teachers.

Intuition. Learning from “in-style” traces narrows the distribution gap,
which directly controls the risk gap. Practical takeaway. Single-model
self-distillation stabilizes training and mitigates forgetting.

### C.5 Training Objective as a Joint-Likelihood Lower Bound

Our loss in Sec. 4 supervises $`(P_{t},P_{v})`$, $`C`$, and $`a`$. It
can be seen as maximizing a lower bound on
$`\log p_{\theta}(a^{\star}\,|\,I,Q)`$ marginalized over feasible
traces.

###### Proposition 4 (ELBO-style lower bound with feasible traces).

Let $`\mathcal{T}_{\kappa,\rho}`$ be the feasible set. For any auxiliary
distribution $`q(T\,|\,I,Q)`$ supported on
$`\mathcal{T}_{\kappa,\rho}`$,

|     |                                             |                                                                               |                                                                                |                                      |
| --- | ------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------ | --- | ---- |
|     | $`\displaystyle\log p\_{\theta}(a^{\star}\, | \,I,Q)\;\geq`$                                                                | $`\displaystyle\underbrace{\mathbb{E}_{q}\!\left[\log p_{\theta}(P*{t},P*{v}\, | \,I,Q)\right]}\_{\text{path term}}`$ |     | (16) |
|     |                                             | $`\displaystyle+\underbrace{\mathbb{E}_{q}\!\left[\log p_{\theta}(C\,         | \,I,Q,P*{t},P*{v})\right]}\_{\text{explanation term}}`$                        |                                      |     |
|     |                                             | $`\displaystyle+\underbrace{\mathbb{E}_{q}\!\left[\log p_{\theta}(a^{\star}\, | \,I,Q,T)\right]}\_{\text{answer term}}`$                                       |                                      |     |
|     |                                             | $`\displaystyle-\mathrm{KL}\!\left(q(T\,                                      | \,I,Q)\,\|\,p\_{\theta}(T\,                                                    | \,I,Q,a^{\star})\right).`$           |     |      |

###### Proof.

Write
$`\log p_{\theta}(a^{\star}\,|\,I,Q)=\log\sum_{T\in\mathcal{T}_{\kappa,\rho}}p_{\theta}(T,a^{\star}\,|\,I,Q)`$,
insert $`q(T\,|\,I,Q)`$, and apply Jensen:

|     |     |     |
| --- | --- | --- |
|     |

````math
\log\sum_{T}q(T)\frac{p_{\theta}(T,a^{\star})}{q(T)}\geq\mathbb{E}_{q}\big[\log p_{\theta}(T,a^{\star})-\log q(T)\big].
``` |  |

Factorize $`p_{\theta}(T,a^{\star})`$ using the model and rearrange. ∎

Intuition. Supervising paths, explanations, and answers maximizes a
tractable surrogate of the marginal likelihood; better selection of
$`q`$ (stronger traces) tightens the bound. Practical takeaway.
Improving the selector/feasibility checks translates into better
training signals.

### C.6 Putting Pieces Together

Theorems [1](#Thmtheorem1 "Theorem 1 (Rademacher shrinkage via verifiable structure). ‣ Hypothesis classes. ‣ C.2 Generalization Benefit from Typed and Verifiable Traces ‣ Appendix C Theoretical Notes for StaR-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")–[3](#Thmtheorem3 "Theorem 3 (Self-distillation alignment). ‣ C.4 Single-Model Self-Distillation Reduces Supervision–Generation Shift ‣ Appendix C Theoretical Notes for StaR-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")
and
Prop. [4](#Thmtheorem4 "Proposition 4 (ELBO-style lower bound with feasible traces). ‣ C.5 Training Objective as a Joint-Likelihood Lower Bound ‣ Appendix C Theoretical Notes for StaR-KVQA ‣ StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering")
jointly suggest: (i) typed, verifiable traces reduce effective
hypothesis space; (ii) the single-model selector is equivalent to
MLE/MAP under a simple consistency–noise view; (iii) single-model
self-distillation reduces supervision–generation shift; and (iv) the
training objective maximizes a joint-likelihood lower bound whose
tightness benefits from stronger traces and selection.

## Appendix D Use of Large Language Models

In preparing this article, Large Language Models (LLMs) were employed
only for stylistic refinement. Their role was limited to editing the
wording of certain sections in order to improve readability and fluency
of the manuscript. The intellectual contributions—including the
development of ideas, design of experiments, analysis of results, and
formulation of conclusions—were carried out entirely by the authors. No
part of the research process, data interpretation, or scientific claims
relied on the use of LLMs. The authors assume full responsibility for
the content presented and ensure its originality and accuracy.

## Appendix E Data Ethics Statement

To evaluate the efficacy of StaR-KVQA, we conducted experiments which
only use publicly available datasets, namely, OK-VQA \[[34](#bib.bib2)\]
and FVQA \[[46](#bib.bib1)\]. We also confirm that no personally
identifiable information was utilized, and this research did not involve
any human or animal subjects.
````
