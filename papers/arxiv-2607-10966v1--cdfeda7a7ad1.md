---
identifier: arxiv:2607.10966v1
title: "SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"
authors:
  - Mingyuan Wu
  - Jingcheng Yang
  - Shengyi Qian
  - Xudong Wang
  - Jize Jiang
  - Qifan Wang
  - Aashu Singh
  - Khoi Pham
  - Fei Liu
  - Zhaolun Su
  - Zhuokai Zhao
  - Klara Nahrstedt
  - Jianyu Wang
  - Hanchao Yu
published: "2026-07-13T00:21:30+00:00"
url: https://arxiv.org/abs/2607.10966v1
source: arxiv
doi: null
arxiv_id: 2607.10966v1
categories:
  - cs.AI
---

# SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning

Mingyuan Wu Affiliation: University of Illinois Urbana-Champaign
Email: [mw34@illinois.edu](mailto:)    Jingcheng Yang
Affiliation: University of Illinois Urbana-Champaign    Shengyi Qian
Affiliation: Meta    Xudong Wang Affiliation: Meta    Jize Jiang
Affiliation: University of Illinois Urbana-Champaign    Qifan Wang
Affiliation: Meta    Aashu Singh Affiliation: Meta    Khoi Pham
Affiliation: Meta    Fei Liu Affiliation: Meta    Zhaolun Su
Affiliation: Meta    Zhuokai Zhao Affiliation: Meta    Klara Nahrstedt
Affiliation: University of Illinois Urbana-Champaign    Jianyu Wang
Affiliation: Meta    Hanchao Yu Affiliation: Meta

###### Abstract

We introduce Self-Verified Reasoner (SVR-R1), a multi-turn RL framework
that turns a model’s own verification into a learning signal for
multimodal reasoning. For each query, the model proposes an answer using
the same weights, and issues a binary self-verdict (Yes/No). A “No”
triggers a second-chance rethink; a “Yes,” or a turn cap, finalizes the
output for computing the outcome-based reward. SVR-R1 is implemented
with GRPO and an asynchronous multi-turn rollout framework and needs no
external supervision or auxiliary critics. We evaluate SVR-R1 on
vision-language reasoning benchmarks and show that it improves accuracy
by a large margin over strong standard GRPO baselines. Training dynamics
show decreasing reliance on verification-fewer verification turns, yet
higher test accuracy-indicating that the gap between verification and
generation narrows as the policy internalizes self-correction and
chooses the most confident answer via our framework. SVR-R1 bridges the
less explored intersection of inference-time self-refinement and RL
training for VLMs, offering a simple yet effective recipe for
bootstrapping multimodal reasoning. We will open-source SVR-R1 to
facilitate future research in VLMs.

|     |
| --- |
|     |

## 1 Introduction

![Refer to caption](2607.10966v1/figures/figure1_svr.png)

Figure 1: Merging VLM Self-Verification Loop into RL Rollout

When given a second chance to think, humans often reason their way
toward better solutions on complex problems. Recently, Large Language
Models (LLMs) have demonstrated a similar pattern of iterative
rethinking in the reasoning process ([OpenAI, 2024b](#bib.bib12)),
particularly when fine-tuned with reinforcement learning (RL) on
task-specific rewards ([DeepSeek-AI, 2025](#bib.bib3); [Zeng et al.,
2025](#bib.bib39); [Zhou et al., 2025b](#bib.bib18); [Wang et al.,
2025b](#bib.bib42)). After such training, these models can intrinsically
exhibit reasoning behaviors that allow them to iteratively improve their
reasoning paths through behaviors like self-correction and
backtracking ([Gandhi et al., 2025](#bib.bib31)). This approach was
initially applied to verifiable tasks in the language domain, such as
mathematics and coding, and has since been extended to non-verifiable
domains and to vision-language tasks ([Zhou et al., 2025a](#bib.bib4);
[Chen et al., 2025a](#bib.bib5); [Zhang et al., 2025](#bib.bib6); [Huang
et al., 2025a](#bib.bib7); [Deng et al., 2025](#bib.bib9); [Wang et al.,
2025a](#bib.bib10)).

Even before the widespread adoption of RL fine-tuning for enabling
self-rethinking in LLMs ([Kumar et al., 2025](#bib.bib14)), it was
observed that simply prompting an LLM to second-guess to refine or
correct its previous answer could lead to improved reasoning performance
in inference ([Madaan et al., 2023](#bib.bib30); [Shinn et al.,
2023](#bib.bib37)). This improvement may be attributed to the gap
between verification and generation ([Song et al., 2025](#bib.bib29)):
“if verification is easier than generation, one can hypothesize that the
model may act as a better-than-random verifier of its own outputs,
enabling self-improvement”. Beyond pure language, pioneering work ([Liao
et al., 2025](#bib.bib36)) has shown that strong commercial Vision
Language Models (VLMs) such as GPT-4o can, to some extent, self-verify
their outputs in certain tasks (e.g. segmentation), though effective
self-rethinking generally remains challenging and underexplored in
multi-modal reasoning scenarios [Wu et al. (2025)](#bib.bib35); [Huang
et al. (2025b)](#bib.bib23).

In this work, we explore a previously unexplored middle ground between
explicit prompting and RL fine-tuning for self-rethinking by leveraging
VLMs’ pre-existing self-verification capabilities in
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
Specifically, we aim to bootstrap a VLM’s reasoning abilities, prompting
it to rethink if the model itself deems its (tentative) answer wrong,
and finalizes its output if it deems its answer correct. To accomplish
this, we integrate model self-verification rounds, which share the same
copy of model weights, into the RL training process prior to reward
assignment. Specifically, we interleave multi-turn self-generation and
verification during RL rollouts: the self-verifier is presented with the
question and the initial self-generated solution, and is prompted to
provide a binary verification of correctness (Yes or No). If the answer
is No, the model is given a second chance to think and regenerate its
answer accordingly. If the answer is Yes, or if a manually set maximum
number of verification rounds is reached, the model’s final response is
passed to reward calculation. The overall training is conducted using
the widely adopted Group Relative Policy Optimization algorithm
(GRPO) ([Shao et al., 2024](#bib.bib16)) and outcome-based reward
matching, and supported by asynchronous multi-modal multi-turn RL
framework ([Sheng et al., 2024](#bib.bib28)) to coordinate the
self-verifier and generator in the rollout. We refer to our approach as
the Self-Verified Reasoner (SVR-R1).

![Refer to caption](2607.10966v1/figures/figure1_graph_2.png)

Figure 2: Validation Reasoning Accuracy (%) vs. Training Steps. SVR-R1
compared with standard GRPO ([Shao et al., 2024](#bib.bib16)) on the
Qwen2.5-VL ([Bai et al., 2025](#bib.bib13)) 3B model, with Mean number
of Turns decreasing.

We demonstrate the effectiveness of SVR-R1 across multiple challenging
multi-modal table and chart reasoning benchmarks ([Fu et al.,
2025](#bib.bib15)), as well as general reasoning tasks ([Wang et al.,
2025b](#bib.bib42)), showing significant improvements in vision-language
reasoning performance. Interestingly, we observe that, over the course
of training, the policy model gradually performs fewer verification
rounds, eventually leading to the self-verifier almost always
immediately affirming the answer in a single round, while still
achieving improved accuracy on the test set, as it is demonstrated in
Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
This suggests that SVR-R1 helps the model learn to close the gap between
generation and verification by exploiting self-verification during RL
training.

SVR-R1 also contributes to the broader direction of self-improvement
without external supervision. While most early self-improvement
approaches ([Huang et al., 2023](#bib.bib33); [Zelikman et al.,
2022](#bib.bib34); [Zhao et al., 2025](#bib.bib22)) have focused on
using self-generated, high-confidence reasoning traces for supervised
fine-tuning, they have not fully integrated these traces into the
iterative RL pipeline. In this regard, SVR-R1 advances the understanding
of how models can continuously bootstrap themselves through interleaved
generation and verification in RL training, without relying on extra
external data or model supervision.

## 2 Related Work

### 2.1 RL for LLM and VLM Reasoning

Reinforcement learning (RL) became widely adopted for LLM development
through RL from Human Feedback (RLHF) ([Ouyang et al.,
2022](#bib.bib25)), which uses a reward model trained on human
preferences to optimize the LLM policy via Proximal Policy Optimization
(PPO) ([Schulman et al., 2017](#bib.bib24)). More recent work has
introduced computationally efficient variants of PPO ([Rafailov et al.,
2023](#bib.bib17); [Wang et al., 2024](#bib.bib20); [Shao et al.,
2024](#bib.bib16); [Zhou et al., 2025c](#bib.bib21)), making RL-based
training more accessible.

Beyond alignment on human preference, RL has demonstrated significant
improvements in LLM reasoning and self-correction capabilities ([Kumar
et al., 2025](#bib.bib14); [DeepSeek-AI, 2025](#bib.bib3); [Zeng et al.,
2025](#bib.bib39)). Recent studies have investigated the intrinsic
properties that enable effective self-improvement, revealing emergent
behaviors such as ”aha moments” that arise through RL-based
fine-tuning ([Gandhi et al., 2025](#bib.bib31); [Zeng et al.,
2025](#bib.bib39)). Building on these advances, RL fine-tuning has also
been successfully adopted for multimodal reasoning tasks ([Zhou et al.,
2025a](#bib.bib4); [Chen et al., 2025a](#bib.bib5); [Zhang et al.,
2025](#bib.bib6); [Huang et al., 2025a](#bib.bib7); [Liu et al.,
2025](#bib.bib8); [Deng et al., 2025](#bib.bib9); [Wang et al.,
2025a](#bib.bib10); [Wang et al., 2025b](#bib.bib42)). Notably,
VL-Rethinker ([Wang et al., 2025a](#bib.bib10)) elicits self-reflection
in vision–language models through prompting. In contrast, our work is
the first to integrate self-reflection directly into the RL
post-training process.

### 2.2 Self-Improvement of LLMs/VLMs

Early work explored inference- or prompt-level self-improvement, such as
asking the model to generate feedback and revise its responses, or
manually incorporating re-verification into the prompt ([Madaan et al.,
2023](#bib.bib30); [Weng et al., 2023](#bib.bib32)). Other methods
([Huang et al., 2023](#bib.bib33); [Zelikman et al., 2022](#bib.bib34);
[Zhao et al., 2025](#bib.bib22); [Zhou et al., 2025b](#bib.bib18))
encourage models to generate reasoning traces, filter high-quality
responses, and then fine-tune on these traces to achieve model
self-improvement via next-token prediction. Subsequent studies attribute
such self-improvements to mechanisms such as sharpening ([Huang et al.,
2024](#bib.bib2)) or to the verification–generation gap in reasoning
tasks ([Song et al., 2025](#bib.bib29)), and try merging with the RL
workstreams ([Jiang et al., 2025](#bib.bib40); [Chen et al.,
2025b](#bib.bib19)).

Recently, efforts to enable self-improvement have expanded into the
vision-language domain ([Liao et al., 2025](#bib.bib36); [Wu et al.,
2025](#bib.bib35); [Ding and Zhang, 2025](#bib.bib38)), though these
studies are typically limited to constrained settings. Progress in this
area remains challenging, likely due to difficulties in multi-modal
pretraining and lack of high quality reasoning data compared to pure
language setting.

## 3 Method

This section describes SVR-R1’s bootstraping of VLMs’ reasoning
abilities: SVR-R1 aims to enable VLMs to reach final reasoning answers
that the model itself deems correct via guiding the model to re-iterate
on answers that the model deems wrong. We describe SVR-R1’s overall
pipeline
in [Section 3.1](#S3.SS1 "3.1 SVR-R1 Overview ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
self-generator and verifier protocol
in [Section 3.2](#S3.SS2 "3.2 Self-Verification and Generation ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
and accordingly, RL-based multi-turn self-generation and verification
in [Section 3.3](#S3.SS3 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
with learning objectives and outcome-based reward design.

![Refer to caption](2607.10966v1/figures/figure2_svr.png)

Figure 3: Multi-rollout with self-verification. Each rollout, along with
additional prompts, becomes part of the context for the next.

### 3.1 SVR-R1 Overview

This section describes how SVR-R1, as a VLM policy, is designed to
process and respond to multi-modal prompts.

Preliminaries. We denote a VLM policy as $`\pi_{\theta}`$ parametrized
with model weights $`\theta`$ in this paper. Given a text prompt
sequence $`x`$ and an image $`I`$, the model can generate a text
response sequence $`y`$, sampled from the $`\pi_{\theta}(I,x)`$. The
weight $`\theta`$ is shared by SVR-R1’s self-generator and verifier
during RL rollouts
([Section 3.3](#S3.SS3 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"))
despite operating with different input text instruction prompts
([Section 3.2](#S3.SS2 "3.2 Self-Verification and Generation ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")).

SVR-R1 Pipeline. We construct the initial prompt headers to be passed to
SVR-R1 to include carefully designed requirements and few-shot examples
that can enhance multimodal reasoning (prompt in Appenedix
[Figure 10](#A1.F10 "In A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")),
inspired by the approach of previous work ([Fu et al.,
2025](#bib.bib15)). Then, SVR-R1’s multi-turn conversation for
self-generation and verification proceeds as an interleaving sequence of
user turns (for model prompt input) and assistant turns (for model
output), following the standard conversational templates commonly used
in VLMs such as Qwen series ([Bai et al., 2025](#bib.bib13)): In the
assistant turn, the model generates a reasoning response that includes
both its rationale and answer, which is then passed to the following
self-verification step. All user and assistant turns are preserved in
the conversation history as the model progresses through successive
rounds of generation and verification towards the outcome-based judge.
An overview of the entire SVR-R1 pipeline is depicted in
[Figure 3](#S3.F3 "In 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
where we visualize the different model inputs, outputs, and
conversational turns in distinct colors, with all elements embedded
within the history.

### 3.2 Self-Verification and Generation

This section describes SVR-R1’s self-verifier and generators, which
share the same set of model weights $`\theta`$ during RL rollouts but
notably operate with different input text prompts. SVR-R1 enforces
interleaving of self-verification and generation steps, repeating this
process until the maximum number of rounds is reached.

![Refer to caption](2607.10966v1/final_figure3.png)

Figure 4: Multi-Modal GRPO w. Self Verification Training Pipeline.

Self-verification Protocol. SVR-R1 adopts binary feedback for the
verifier, restricting its output to only Yes or No via instructing it
with the proper text prompts. This is because VLMs are generally known
to be less capable than LLMs at providing detailed feedback with
rationales, and prior work ([Liao et al., 2025](#bib.bib36)) has shown
simple binary feedback to be the most effective in the VLM domain at
minimizing the risk of hallucinations.
[Figure 8](#A1.F8 "In A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
in Appendix depicts the prompt instruction $`x_{v}`$ SVR-R1 uses for the
self-verifier, constraining it to only indicate whether the previous
prediction is correct or not. Empirically, we find that existing VLMs
reliably follow this instruction, and the sampled output from
self-verification is consistently either “YES” or “NO”. Formally, in the
self verification response
$`y^{\prime}\sim\pi_{\theta}(\cdot\mid I,x^{\prime})`$, (1) $`\theta`$
is the same copy of parameters as generator (described shortly), (2)
$`x^{\prime}`$ includes all of the multi-modal question, the response in
previous self-generation turn, and the aforementioned instruction prompt
for verification, and (3) $`y^{\prime}\in\{\mathrm{YES},\mathrm{NO}\}`$.
For example, in the verification round following the initial generation,
input prompt can be denoted as $`x^{\prime}=x\oplus y_{0}\oplus x_{v}`$,
where $`\oplus`$ is simple concatenation.

Self-(Re)generation Protocol. In the self-(re)generation step, if the
obtained binary self-verification result $`y^{\prime}=\mathrm{No}`$ and
a user-specified maximum number of turns has not been reached, the model
will be prompted (via appending a textual rethink trigger $`x_{r}`$ to
the self-verifier’s response,
[Figure 9](#A1.F9 "In A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
in Appendix) to generating a revised response in a new assistant turn,
while keeping previous turns in mind.

Formally, if the verifier disagrees at the $`i_{th}`$ step, the new
re-generated answer is sampled from the same VLM policy
$`\pi_{\theta}`$, with the input recursively constructed by appending
the previous response and a rethink trigger to the input from the
previous round: $`y_{i+1}\sim\pi_{\theta}(\cdot\mid I,x_{i})`$, where
$`x_{i}=x_{i-1}\oplus y_{i}\oplus x_{v}`$. This recursive multi-turn
process encourages the model to allocate additional reasoning tokens and
regenerate its answer whenever it fails self-verification, effectively
simulating a verifier-driven ”re-thinking” process without relying on
external knowledge. Finally, when either $`y^{\prime}=\mathrm{yes}`$ or
the turn limit has been reached, the finalized answer is used for
accuracy measurement (during inference) or reward assignment (in RL
rollouts).
[Figure 3](#S3.F3 "In 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
depicts an overview of this process: the model performs repeated
generation-verification cycles from the 1st to the $`(i-1)`$th step,
with each rollout becoming part of the context for the next. Only the
final answer, once verification succeeds or the turn limit is reached,
interacts with the outcome reward judge. No process rewards are assigned
to verification.

### 3.3 Multi-turn RL with Self-verification

This section describes SVR-R1’s multi-modal and multi-turn RL procedure
illustrated in
[Figure 4](#S3.F4 "In 3.2 Self-Verification and Generation ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
which aims to optimize for the final reasoning response $`y`$ with VLM
rollouts including self-verifications.

SVR-R1’s training objective is to encourage the VLM policy
$`\pi_{\theta}`$ to improve its reasoning ability with iterative
self-verification, while penalizing large update step from the reference
model $`\pi_{\text{ref}}`$. Formally, we maximize the following training
objective:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
\max_{\pi_{\theta}}\mathbb{E}_{[I,T]\sim\mathcal{D},\;y\sim\pi_{\theta}(\cdot\mid I,T;\pi_{\theta})}\bigl[r_{\phi}(I,T,y)\bigr]-\beta\,\mathbb{D}_{\mathrm{KL}}\!\bigl[\pi_{\theta}(\cdot\mid I,T;\pi_{\theta})\,\|\,\pi_{\mathrm{ref}}(\cdot\mid I,T;\pi_{\theta})\bigr]
``` |  | (1) |

where $`\pi_{\theta}`$ is the trainable VLM policy, $`\pi_{\text{ref}}`$
is the frozen reference model, $`r_{\phi}`$ is the reward function, and
$`\beta>0`$ is the KL penalty coefficient. The input $`[I,T]`$
represents multimodal samples (image $`I`$ and text question query
$`T`$) drawn from dataset $`\mathcal{D}`$. We explicitly include
$`\pi_{\theta}`$ in the sampling formulation to indicate that, during
VLM policy generation, the policy itself also serves as the verifier.

End-to-end Optimization Objective for Multi-turn Rollout. Unlike prior
RL fine-tuning approaches that optimize single-pass generations ([Ouyang
et al., 2022](#bib.bib25); [Shao et al., 2024](#bib.bib16)), SVR-R1
explicitly incorporates multi-turn self-verification into rollouts. Each
rollout includes intermediate verifier binary feedback steps, where the
model itself decides whether to rethink. Thus,
$`y\sim\pi_{\theta}(\cdot\mid I,T)`$ can denote either a direct answer
or an answer refined after multiple verification turns. Notably, SVR-R1
does not directly optimize the intermediate verification responses
$`y^{\prime}`$; it instead only optimizes the final response $`y`$ via
outcome-based reward—either the first response that receives a ”YES”
from the verifier or the last response generated after reaching the
predefined maximum number of turns
([Section 3.2](#S3.SS2 "3.2 Self-Verification and Generation ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")),
while allowing the model to autonomously decide whether invoking
additional verification steps improves reasoning, incentivizing
accurate, confident final answers and the generation of self-verifiable
reasoning traces.

Loss Masking for Verification Tokens. In RL finetuning ([Shao et al.,
2024](#bib.bib16); [Schulman et al., 2017](#bib.bib24)), losses are
computed over the entire rollout sequence. In SVR-R1, however, the
rollout includes multiple turns of both self-generated tokens and
self-verification tokens (Yes or No). Since our objective is to optimize
the final generation response—and the trigger ”verifier disagrees” is
already manually incorporated into the generation—directly including
verification tokens in the loss calculation can introduce unintended
learning dynamics. (For example, optimizing both self-verification and
generation simultaneously may lead to conflicting training objectives.)
Instead, our focus is on improving end-to-end reasoning performance with
self-verification in the loop. Inspired by the multi-turn RL framework
Search-R1 ([Jin et al., 2025](#bib.bib11)), we mask out the
self-verification tokens during loss calculation to stabilize policy
updates during training.

GRPO with Self-verification. Specifically, SVR-R1’s optimization for
parameters $`\theta`$ builds on Group Relative Policy Optimization
(GRPO) ([Shao et al., 2024](#bib.bib16)), a stable and
resource-efficient online policy-gradient algorithm
([Figure 4](#S3.F4 "In 3.2 Self-Verification and Generation ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")),
without a learned value function approximation as in PPO ([Schulman et
al., 2017](#bib.bib24)). GRPO estimates baselines from a group of
Monte-Carlo sampled rollouts, eliminating the need for a critic,
reducing training overhead and costs and demonstrating strong empirical
performance [Shao et al. (2024)](#bib.bib16). This efficiency makes GRPO
particularly well-suited SVR-R1 complex, multi-turn setting where
training costs may otherwise be prohibitive. Concretely, for each input
$`[I,T]`$, SVR-R1 samples a group of responses including
self-verification steps,
$`\{y_{i}\}_{i=1}^{G}\sim\pi_{\text{old}}(\cdot\mid I,T;\pi_{\text{old}}),`$
from the old policy. The current policy $`\pi_{\theta}`$ is then updated
by maximizing the group-relative objective:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{GRPO}}(\theta)=\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\min\!\Big(r_{i}(\theta)\widehat{A}(y_{i}),\,\mathrm{clip}\!\left(r_{i}(\theta),1-\epsilon,1+\epsilon\right)\widehat{A}(y_{i})\Big)\right]-\beta\,\mathrm{D}_{\mathrm{KL}}\!\left(\pi_{\theta}\,\|\,\pi_{\mathrm{old}}\right),
``` |  | (2) |

where

|  |  |  |
|----|----|----|
|  |
``` math
r_{i}(\theta)=\frac{\pi_{\theta}(y_{i}\mid I,T;\pi_{\theta})}{\pi_{\mathrm{old}}(y_{i}\mid I,T;\pi_{\mathrm{old}})}.
``` |  |

where $`\widehat{A}(y_{i})`$ is the group-relative advantage for
response $`y_{i}`$, computed by normalizing outcome rewards within the
group, and positive $`\epsilon`$ is the clip threshold. This formulation
encourages exploration of diverse reasoning strategies, while
maintaining stability and ensuring that the policy improves relative to
its peers within the sampled group.

Reward Design. SVR-R1 utilizes an outcome-based binary reward without
format rewards. For complex, semi-open-form visual question answering
tasks, such as the visual table and chart reasoning dataset in ReFocus
([Section 4.1](#S4.SS1 "4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")),
we follow prior work ([Fu et al., 2025](#bib.bib15)) and adopt a large
language model judge to compare the ground truth with the final
prediction, with detailed prompts provided in the supplementary
materials. For verifiable reasoning tasks, such as geometry math or
multiple-choice questions, we use a rule-based judge.

## 4 Experiment

In this section, we empirically evaluate the effectiveness of SVR in
multi-modal reasoning, present the findings and analyze the key factors
contributing to SVR-R1’s success.

### 4.1 Experiment Setup

Dataset. We use three dataset splits containing challenging table and
chart reasoning tasks for proper assessment and benchmarking of
performance. First two splits are prepared following the data
pre-processing pipeline in ReFocus ([Fu et al., 2025](#bib.bib15)):

1.  1.
    ChartQA Split ([Masry et al., 2022](#bib.bib27)) contains 826 test
    QA pairs split into 444 horizontal and 382 vertical bar chart
    questions, each requiring logical comparisons and visual reasoning
    over chart structures. We use the official ReFocus training set
    comprising 14,344 examples selected from the ChartQA training split
    (out of 15,059 available) for training.
2.  2.
    TableVQA Split ([Kim et al., 2024](#bib.bib26)) contains 1,250
    table-based questions incorporating the VWTQ, VWTQ-syn, and VTabFact
    splits. We allocate 70% of questions for training and 30% for
    testing.
3.  3.
    ThinkLite-VL-70K from recent multimodal reasoning work ([Wang et
    al., 2025b](#bib.bib42)), which includes general multimodal
    reasoning tasks for training.

Full dataset details about ThinkLite-VL-70K are in the supplementary
materials.

Implementation. We build SVR-R1 upon a state-of-the-art open-source VLM
Qwen-VL 2.5 ([Bai et al., 2025](#bib.bib13)) at the 3B and 7B scales. We
implement SVR-R1 with GRPO
([Section 3.3](#S3.SS3 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"))
in the open-source VeRL framework ([Sheng et al., 2024](#bib.bib28)) for
multi-modal and multi-turn RL fine-tuning. Additional implementation
details are provided in
[Section B.4](#A2.SS4 "B.4 Implementation Details ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
of the Appendix.

Methods. We compare SVR-R1 against the following notable methods:

1.  1.
    Qwen2.5-VL ([Bai et al., 2025](#bib.bib13)): The baseline model
    which SVR-R1 builds upon; we use it without SVR-R1’s training.
2.  2.
    GPT-4o ([OpenAI, 2024b](#bib.bib12)): a commercial VLM which we
    access via API. We use the 2024-08-06 checkpoint ([OpenAI,
    2024a](#bib.bib43)) for maximal reproducibility.
3.  3.
    R1-VL ([Zhang et al., 2025](#bib.bib6)): a recently open-sourced
    reasoning VLM RL-trained for general reasoning tasks in academic.
4.  4.
    Weaker Baselines:As included in the original ReFocus paper ([Fu et
    al., 2025](#bib.bib15)), these comprise classic models such as
    Llava-next ([Liu et al., 2024](#bib.bib44)), Phi-3 ([Team,
    2024](#bib.bib45)), Gemini 1.5 ([Team, 2025a](#bib.bib46)), and
    VisProg ([Gupta and Kembhavi, 2023](#bib.bib47)).

Additionally, we use an ablated version of Qwen-VL 2.5, Qwen-RL, which
we train with standard GRPO with exact the same data and hyperparameter
setting, but do not perform any self-verification rounds in the rollout,
to study the impact of self-verification on reasoning performance.

Inference Setup. To demonstrate how SVR-R1 internalizes
self-verification advantages, we evaluate models under two distinct
settings:

1.  1.
    Inference Pure Run (PR): Direct inference run without any
    self-verification step.
2.  2.
    Inference with Final Verification (FV): Inference run that includes
    self-verification and rethinking steps, following the same protocol
    used during training rollouts. This process continues until either
    an early Yes is obtained or a maximum of $`MAX=3`$ iterations is
    reached.

Reward Judge. We employ the state-of-the-art open-source large language
model, gpt-oss-120b ([Team, 2025b](#bib.bib48)), to evaluate model
predictions against ground-truth answers. The reward is binary: 1 for
correct predictions and 0 for incorrect ones. For the ThinkLite
experiments, we follow the original work ([Wang et al.,
2025b](#bib.bib42)) to use a rule-based binary reward to match the
answer.

| Qwen2.5-VL | 3B |  |  |  |  |  | 7B |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | PR | FV | RL-PR | RL-FV | SVR-PR | SVR-FV | PR | FV | RL-PR | RL-FV | SVR-PR | SVR-FV |
| ChartQA Split | 63.3 | 64.9 | 80.5 | 80.9 | 83.3 | 83.3 | 76.0 | 76.9 | 80.4 | 81.1 | 82.9 | 82.9 |
| TableVQA Split | 56.4 | 59.0 | 68.3 | 68.7 | 72.4 | 72.9 | 67.8 | 68.6 | 78.7 | 78.5 | 80.3 | 80.6 |

Table 1: Main results (accuracy in %) of raw Qwen2.5-VL, SVR-R1, and
Qwen-RL. PR: Pure Run; FV: with Final Verification.

### 4.2 Empirical Findings

We conduct controlled experiments using identical datasets and
hyperparameters for both standard GRPO and SVR-R1, and find that SVR-R1
significantly boosts multimodal reasoning with same amount of data.
Table [1](#S4.T1 "Table 1 ‣ 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
and
Table [2](#S4.T2 "Table 2 ‣ 4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
present our main results, highlighting the effectiveness of SVR-R1 for
fine-tuning VLM on challenging visual table and chart reasoning tasks.
At both the 3B and 7B model scales, SVR-R1 consistently outperforms the
baseline GRPO methods by a substantial margin, even though GRPO already
demonstrates strong capabilities in fine-tuning models for these tasks.
Similar patterns exist in the general reasoning dataset with larger
scale, as detailed in
Table [3](#S4.T3 "Table 3 ‣ 4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
models trained on ThinkLite data exhibit improved performance on more
general reasoning tasks.

|  |     R1VL-2B-PR |     R1VL-2B-FV |     R1VL-7B-PR |     R1VL-7B-FV |     GPT4o-PR |     GPT4o-FV |     LLaVA-34B |     Phi-3V |     Gem-Pro1.5 |     VisProg |     SVR-3B |     SVR-7B |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Chart |  53.6 |  52.7 |  73.2 |  73.1 |  77.8 |  79.3 |  43.7 |  52.3 |  46.9 |  59.6 |  83.3 |  82.9 |
| Table |  37.8 |  34.8 |  58.2 |  57.4 |  76.9 |  76.6 |  18.4 |  63.4 |  61.3 |  69.2 |  72.9 |  80.6 |
| Extra SFT |  ✓ |  ✓ |  ✓ |  ✓ |  ✗/✓ |  ✗/✓ |  ✓ |  ✗/✓ |  ✗/✓ |  ✗ |  ✗ |  ✗ |

Table 2: Ours vs. other models. Colored groups: SVR-R1, R1-VL, GPT-4o,
and weaker baseline models in Chart and Table Reasoning Tasks.

| 7B ThinkLite | MathVista | MathVision | MMStar | AI2D |
|--------------|-----------|------------|--------|------|
| RL-BEST      | 70.8      | 17.4       | 48.7   | 81.5 |
| SVR-R1       | 71.6      | 19.1       | 49.3   | 81.4 |

Table 3: SVR-R1 outperforms standard GRPO on general reasoning when
trained in ThinkLite under controlled implementations.

To further illustrate the training dynamics, we provide train-test plots
for both our method and standard GRPO in the thumbnail
Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
for chart reasoning, and detailed results for table tasks in
Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
These figures clearly show that incorporating self-verification into RL
training both accelerates learning under the same data budget and
improves the final saturated performance. Given the difficulty of
curating high-quality relevant multimodal RL datasets, effective
self-verification during RL training is especially valuable, enabling
stronger post-training results on the target tasks.

Robustness of Gain. Comparing the table and chart tasks, we observe that
chart reasoning benefits from a larger dataset and a more stable
training curve, while table reasoning is limited by a smaller training
set (only several hundred samples), resulting in a comparatively noisier
curve in
Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
Nevertheless, our method demonstrates robust performance gains even
under less ideal data conditions for table tasks.

![Refer to caption](2607.10966v1/figures/table_plot.png)

Figure 5: SVR-R1 surpasses standard GRPO on table tasks.

Decreased Verification Turns and Increased Confidence. Throughout
training, we observe that both validation and training verification
turns gradually decrease, as the models become increasingly confident in
their initial answers and tend to affirm their responses: In
Figures [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")
and
[15](#A2.F15 "Figure 15 ‣ B.5 Computation ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
these curves converge to approximately 2 - one generation step followed
by a Yes from the self-verifier round. For the chart task, SVR-R1
achieves nearly identical accuracy in the pure-run setting (without
final verification) and the final-verification setting during the later
stages of training, as reported in
Table [1](#S4.T1 "Table 1 ‣ 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
This indicates that VLMs are able to internalize the verification and
generation gap during SVR-R1 training, providing self-assured answers
when correct—even without directly optimizing for self-verification
capabilities in our RL training process. Intuitively, this is a
desirable outcome, as it indicates that the VLM can produce correct
answers while recognizing their correctness in future tasks. And it
brings benefits in the inference efficiency if we do not need to go over
long chain of self correction.

Lower Entropy. We observe that incorporating self-verification into RL
training consistently results in a model with relatively lower entropy
for a given task compared to standard GRPO. While this reduction
reflects increased confidence, however, excessively trading exploration
for accuracy is not always desirable in reasoning tasks that require
exloration such as math ([Yu et al., 2025](#bib.bib49)). To ensure a
fair comparison, we employ an established entropy-controlled technique
from DAPO ([Yu et al., 2025](#bib.bib49)), applying a higher clip-high
threshold $`\epsilon_{h}`$ for RL training, to assess whether the lower
entropy observed in our approach negatively impacts performance on chart
tasks in 3B model training. Specifically, we investigate whether
increasing entropy over our solutions or standard GRPO can yield
performance gains in our task settings. The results, presented in
Figure [6](#S4.F6 "Figure 6 ‣ 4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
reveal two key findings: (1) higher entropy does not lead to improved
results in our tasks, even in the standard GRPO setting, and (2)
self-verification does not benefit from high clip techniques which
preserve higher entropy. These experiments demonstrate that SVR-R1 does
not ”over-sacrifice” entropy for accuracy, and maintains robust
performance without compromising necessary exploration.

Eliciting Self-reflection via Rethinking Trigger. We present a real
validation example in the Appendix
Figure [7](#A1.F7 "Figure 7 ‣ A.1 Qualitative Example ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
demonstrating that SVR-R1 explicitly prompts the model to engage in
self-reflection, moving beyond simple rejection and regeneration. When
the self-verifier returns no, we prompt the model to reflect directly
(see
Figure [9](#A1.F9 "Figure 9 ‣ A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")),
successfully correcting its wrong answer in this cat breed
identification example. Notably, even though the verifier does not
successfully affirm the correct answer in the second round, we still
observe benefits from rethinking in the third round: the model still
arrives at the correct answer ”calico” and adopts more conservative
phrasing, such as ”also acknowledge the possibility”. Another
interesting finding is that after the rethinking phrase ”given the
verifier’s disagreement,” the model explores diverse reasoning paths,
such as physical features or visible characteristics, with increasing
confidence across rounds.

![Refer to caption](2607.10966v1/figures/chart_accuracy.png)

![Refer to caption](2607.10966v1/figures/chartentropy.png)

Figure 6: SVR-R1 outperforms high-entropy baselines. High-entropy
variants are trained with $`\epsilon_{h}`$ = 2.8, compared to the
standard setting of $`\epsilon_{h}`$ = 2.0. Accuracy, Entropy vs. Steps.

VLMs in the context of Inference-Time Scaling. We attribute part of
SVR-R1’s performance gains to its effective use of existing VLM
capabilities during scaled inference and rollout, even without requiring
highly sophisticated self-verification mechanism. As self-verification
capabilities of VLMs continue to improve, important research questions
would emerge: How can we optimize VLM policy with inference-time scaling
directly through RL? SVR-R1 offers an early and promising step forward
by demonstrating that integrating self-verification into both training
and inference-time reasoning yields substantial benefits.

Squeeze What Models Can Answer with Effort. SVR-R1’s performance gain
can be attributed to it finalizing high-quality answers to medium
difficulty questions in the datasets via multiple rollout rounds. In
contrast, repeatedly rethinking on questions that are too difficult
contributes little to training, as the correct answer may remain
unreachable even with unlimited self-verification turns. We further
validate this by training with self-verification only on a dataset of
difficult questions ([Wang et al., 2025b](#bib.bib42)), where we observe
no improvement in reasoning performance while the number of verification
turns grows dramatically during RL training. This performance gain
brought by focusing efforts in solving medium-difficulty questions has
been observed in prior works such as ([Gao et al., 2025](#bib.bib51)) in
the language domain, have demonstrated benefits by creating training
curriculum with batches of intermediate difficulty tailored to current
model weights in current training step.

## 5 Conclusion

We introduced Self-Verified Reasoner (SVR-R1), which integrates
self-verification rounds into GRPO training to bootstrap VLM’s reasoning
capabilities. SVR-R1 significantly outperforms standard GRPO on
multi-modal reasoning benchmarks with exactly the same data and
hyperparameter setup. Notably, models gradually perform fewer
verification rounds during training while maintaining improved accuracy,
suggesting they learn to close the generation-verification gap by
producing initial answers that pass self-verification. This work
demonstrates that models can effectively leverage their inherent
verification capabilities for self-improvement within the RL training
loop, advancing multimodal reasoning.

## References

- Bai et al. (2025) S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song,
  and et al. Qwen2.5-vl technical report. External Links: 2502.13923,
  [Link](https://arxiv.org/abs/2502.13923) Cited by: [Figure
  2](#S1.F2 "In 1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.1](#S3.SS1.p3.1 "3.1 SVR-R1 Overview ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [item 1](#S4.I2.i1.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Chen et al. (2021) J. Chen, J. Tang, J. Qin, X. Liang, L. Liu, E.
  Xing, and L. Lin GeoQA: a geometric question answering benchmark
  towards multimodal numerical reasoning. In Findings of the Association
  for Computational Linguistics: ACL-IJCNLP 2021, C. Zong, F. Xia, W.
  Li, and R. Navigli (Eds.), Online, pp. 513–523. External Links:
  [Link](https://aclanthology.org/2021.findings-acl.46/),
  [Document](https://dx.doi.org/10.18653/v1/2021.findings-acl.46) Cited
  by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Chen et al. (2022) J. Chen, J. Tang, J. Qin, X. Liang, L. Liu, E. P.
  Xing, and L. Lin GeoQA: a geometric question answering benchmark
  towards multimodal numerical reasoning. External Links: 2105.14517,
  [Link](https://arxiv.org/abs/2105.14517) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Chen et al. (2025a) L. Chen, L. Li, H. Zhao, Y. Song, and Vinci R1-v:
  reinforcing super generalization ability in vision-language models
  with less than \$3. Note:
  [https://github.com/Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V)Accessed:
  2025-02-02 Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Chen et al. (2025b) Z. Chen, Z. Zhao, K. Zhang, B. Liu, Q. Qi, Y.
  Wu, T. Kalluri, S. Cao, Y. Xiong, H. Tong, et al. Scaling agent
  learning via experience synthesis. arXiv preprint arXiv:2511.03773.
  Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- DeepSeek-AI (2025) DeepSeek-AI DeepSeek-r1: incentivizing reasoning
  capability in llms via reinforcement learning. Vol. abs/2501.12948.
  External Links: [Link](https://arxiv.org/abs/2501.12948) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Deng et al. (2025) Y. Deng, H. Bansal, F. Yin, N. Peng, W. Wang,
  and K. Chang OpenVLThinker: an early exploration to complex
  vision-language reasoning via iterative self-improvement. External
  Links: 2503.17352, [Link](https://arxiv.org/abs/2503.17352) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Ding and Zhang (2025) Y. Ding and R. Zhang Sherlock: self-correcting
  reasoning in vision-language models. External Links: 2505.22651,
  [Link](https://arxiv.org/abs/2505.22651) Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Fu et al. (2025) X. Fu, M. Liu, Z. Yang, J. Corring, Y. Lu, J.
  Yang, D. Roth, D. Florencio, and C. Zhang ReFocus: visual editing as a
  chain of thought for structured image understanding. In Proceedings of
  the 42st International Conference on Machine Learning, ICML’ 25. Cited
  by:
  [§A.2](#A1.SS2.p1.1 "A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p4.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.1](#S3.SS1.p3.1 "3.1 SVR-R1 Overview ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p8.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [item 4](#S4.I2.i4.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§4.1](#S4.SS1.p1.1 "4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Gandhi et al. (2025) K. Gandhi, A. Chakravarthy, A. Singh, N. Lile,
  and N. D. Goodman Cognitive behaviors that enable self-improving
  reasoners, or, four habits of highly effective stars. External Links:
  2503.01307, [Link](https://arxiv.org/abs/2503.01307) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Gao et al. (2025) Z. Gao, J. Kim, W. Sun, T. Joachims, S. Wang, R. Y.
  Pang, and L. Tan Prompt curriculum learning for efficient llm
  post-training. External Links: 2510.01135,
  [Link](https://arxiv.org/abs/2510.01135) Cited by:
  [§4.2](#S4.SS2.p8.1 "4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Gupta and Kembhavi (2023) T. Gupta and A. Kembhavi Visual programming:
  compositional visual reasoning without training. In 2023 IEEE/CVF
  Conference on Computer Vision and Pattern Recognition (CVPR), Vol. ,
  pp. 14953–14962. External Links:
  [Document](https://dx.doi.org/10.1109/CVPR52729.2023.01436) Cited by:
  [item 4](#S4.I2.i4.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Huang et al. (2024) A. Huang, A. Block, D. J. Foster, D. Rohatgi, C.
  Zhang, M. Simchowitz, J. T. Ash, and A. Krishnamurthy Self-improvement
  in language models: the sharpening mechanism. External Links:
  2412.01951, [Link](https://arxiv.org/abs/2412.01951) Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Huang et al. (2023) J. Huang, S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu,
  and J. Han Large language models can self-improve. In Proceedings of
  the 2023 Conference on Empirical Methods in Natural Language
  Processing, H. Bouamor, J. Pino, and K. Bali (Eds.), Singapore,
  pp. 1051–1068. External Links:
  [Link](https://aclanthology.org/2023.emnlp-main.67/),
  [Document](https://dx.doi.org/10.18653/v1/2023.emnlp-main.67) Cited
  by:
  [§1](#S1.p5.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Huang et al. (2025a) W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F.
  Zhao, Z. Xu, Y. Hu, and S. Lin Vision-r1: incentivizing reasoning
  capability in multimodal large language models. External Links:
  2503.06749, [Link](https://arxiv.org/abs/2503.06749) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Huang et al. (2025b) Y. Huang, Q. He, Z. Chen, H. Zhang, H. Yu, and Z.
  Zhao Autonomous multimodal reasoning via implicit chain-of-vision. In
  Proceedings of the Computer Vision and Pattern Recognition Conference,
  pp. 2963–2972. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Jiang et al. (2025) M. Jiang, A. Lupu, and Y. Bachrach Bootstrapping
  task spaces for self-improvement. External Links: 2509.04575,
  [Link](https://arxiv.org/abs/2509.04575) Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Jin et al. (2025) B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D.
  Wang, H. Zamani, and J. Han Search-r1: training llms to reason and
  leverage search engines with reinforcement learning. External Links:
  2503.09516, [Link](https://arxiv.org/abs/2503.09516) Cited by:
  [§3.3](#S3.SS3.p5.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Kahou et al. (2018) S. E. Kahou, V. Michalski, A. Atkinson, A.
  Kadar, A. Trischler, and Y. Bengio FigureQA: an annotated figure
  dataset for visual reasoning. External Links: 1710.07300,
  [Link](https://arxiv.org/abs/1710.07300) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Kim et al. (2024) Y. Kim, M. Yim, and K. Y. Song TableVQA-bench: a
  visual question answering benchmark on multiple table domains. arXiv
  preprint arXiv:2404.19205. Cited by:
  [item 2](#S4.I1.i2.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Kumar et al. (2025) A. Kumar, V. Zhuang, R. Agarwal, Y. Su, J. D.
  Co-Reyes, A. Singh, K. Baumli, S. Iqbal, C. Bishop, R. Roelofs, L. M.
  Zhang, K. McKinney, D. Shrivastava, C. Paduraru, G. Tucker, D.
  Precup, F. Behbahani, and A. Faust Training language models to
  self-correct via reinforcement learning. In Proceedings of the 12th
  International Conference on Learning Representations (ICLR), External
  Links: [Link](https://arxiv.org/abs/2409.12917) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Liao et al. (2025) Y. Liao, R. Mahmood, S. Fidler, and D. Acuna Can
  large vision-language models correct semantic grounding errors by
  themselves?. In 2025 IEEE/CVF Conference on Computer Vision and
  Pattern Recognition (CVPR), Vol. , pp. 14667–14678. External Links:
  [Document](https://dx.doi.org/10.1109/CVPR52734.2025.01367) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p2.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.2](#S3.SS2.p2.1 "3.2 Self-Verification and Generation ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Liu et al. (2024) H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen,
  and Y. J. Lee LLaVA-next: improved reasoning, ocr, and world
  knowledge. External Links:
  [Link](https://llava-vl.github.io/blog/2024-01-30-llava-next/) Cited
  by:
  [item 4](#S4.I2.i4.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Liu et al. (2025) Y. Liu, B. Peng, Z. Zhong, Z. Yue, F. Lu, B. Yu,
  and J. Jia Seg-zero: reasoning-chain guided segmentation via cognitive
  reinforcement. External Links: 2503.06520,
  [Link](https://arxiv.org/abs/2503.06520) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Loshchilov and Hutter (2019) I. Loshchilov and F. Hutter Decoupled
  weight decay regularization. External Links: 1711.05101,
  [Link](https://arxiv.org/abs/1711.05101) Cited by:
  [§B.2](#A2.SS2.p3.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§B.4](#A2.SS4.p1.1 "B.4 Implementation Details ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Lu et al. (2021) P. Lu, R. Gong, S. Jiang, L. Qiu, S. Huang, X. Liang,
  and S. Zhu Inter-gps: interpretable geometry problem solving with
  formal language and symbolic reasoning. External Links: 2105.04165,
  [Link](https://arxiv.org/abs/2105.04165) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Lu et al. (2022a) P. Lu, S. Mishra, T. Xia, L. Qiu, K. Chang, S.
  Zhu, O. Tafjord, P. Clark, and A. Kalyan Learn to explain: multimodal
  reasoning via thought chains for science question answering. In The
  36th Conference on Neural Information Processing Systems (NeurIPS),
  Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Lu et al. (2023) P. Lu, L. Qiu, K. Chang, Y. N. Wu, S. Zhu, T.
  Rajpurohit, P. Clark, and A. Kalyan Dynamic prompt learning via policy
  gradient for semi-structured mathematical reasoning. External Links:
  2209.14610, [Link](https://arxiv.org/abs/2209.14610) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Lu et al. (2022b) P. Lu, L. Qiu, J. Chen, T. Xia, Y. Zhao, W.
  Zhang, Z. Yu, X. Liang, and S. Zhu IconQA: a new benchmark for
  abstract diagram understanding and visual language reasoning. External
  Links: 2110.13214, [Link](https://arxiv.org/abs/2110.13214) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Madaan et al. (2023) A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L.
  Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S.
  Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P.
  Clark Self-refine: iterative refinement with self-feedback. External
  Links: 2303.17651, [Link](https://arxiv.org/abs/2303.17651) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Marino et al. (2019) K. Marino, M. Rastegari, A. Farhadi, and R.
  Mottaghi OK-vqa: a visual question answering benchmark requiring
  external knowledge. External Links: 1906.00067,
  [Link](https://arxiv.org/abs/1906.00067) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Masry et al. (2022) A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E.
  Hoque Chartqa: a benchmark for question answering about charts with
  visual and logical reasoning. arXiv preprint arXiv:2203.10244. Cited
  by:
  [item 1](#S4.I1.i1.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- OpenAI (2024a) OpenAI GPT-4o api documentation. Note:
  [https://platform.openai.com/docs/models/gpt-4o](https://platform.openai.com/docs/models/gpt-4o)Accessed:
  2025-11-05 Cited by:
  [item 2](#S4.I2.i2.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- OpenAI (2024b) OpenAI OpenAI o1 system card. External Links:
  2412.16720, [Link](https://arxiv.org/abs/2412.16720) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [item 2](#S4.I2.i2.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Ouyang et al. (2022) L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L.
  Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J.
  Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P.
  Welinder, P. Christiano, J. Leike, and R. Lowe Training language
  models to follow instructions with human feedback. External Links:
  2203.02155, [Link](https://arxiv.org/abs/2203.02155) Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p4.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Rafailov et al. (2023) R. Rafailov, A. Sharma, E. Mitchell, C. D.
  Manning, S. Ermon, and C. Finn Direct preference optimization: your
  language model is secretly a reward model. Advances in Neural
  Information Processing Systems 36, pp. 53728–53741. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Schulman et al. (2017) J. Schulman, F. Wolski, P. Dhariwal, A.
  Radford, and O. Klimov Proximal policy optimization algorithms. arXiv
  preprint arXiv:1707.06347. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p5.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p6.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Seo et al. (2015) M. Seo, H. Hajishirzi, A. Farhadi, O. Etzioni,
  and C. Malcolm Solving geometry problems: combining text and diagram
  interpretation. In Proceedings of the 2015 Conference on Empirical
  Methods in Natural Language Processing, L. Màrquez, C. Callison-Burch,
  and J. Su (Eds.), Lisbon, Portugal, pp. 1466–1476. External Links:
  [Link](https://aclanthology.org/D15-1171/),
  [Document](https://dx.doi.org/10.18653/v1/D15-1171) Cited by:
  [§B.2](#A2.SS2.p1.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Shao et al. (2024) Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H.
  Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo DeepSeekMath: pushing the
  limits of mathematical reasoning in open language models. External
  Links: 2402.03300, [Link](https://arxiv.org/abs/2402.03300) Cited by:
  [Figure
  2](#S1.F2 "In 1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p3.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p4.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p5.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§3.3](#S3.SS3.p6.1 "3.3 Multi-turn RL with Self-verification ‣ 3 Method ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Sheng et al. (2024) G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R.
  Zhang, Y. Peng, H. Lin, and C. Wu HybridFlow: a flexible and efficient
  rlhf framework. arXiv preprint arXiv: 2409.19256. Cited by:
  [§1](#S1.p3.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§4.1](#S4.SS1.p2.1 "4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Shinn et al. (2023) N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan,
  and S. Yao Reflexion: language agents with verbal reinforcement
  learning. In Proceedings of the 37th International Conference on
  Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA.
  Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Song et al. (2025) Y. Song, H. Zhang, C. Eisenach, S. Kakade, D.
  Foster, and U. Ghai Mind the gap: examining the self-improvement
  capabilities of large language models. External Links: 2412.02674,
  [Link](https://arxiv.org/abs/2412.02674) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Team (2025a) G. Team Gemini: a family of highly capable multimodal
  models. External Links: 2312.11805,
  [Link](https://arxiv.org/abs/2312.11805) Cited by:
  [item 4](#S4.I2.i4.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Team (2025b) O. Team Gpt-oss-120b and gpt-oss-20b model card. External
  Links: 2508.10925, [Link](https://arxiv.org/abs/2508.10925) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§4.1](#S4.SS1.p4.2 "4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Team (2024) P. Team Phi-3 technical report: a highly capable language
  model locally on your phone. External Links: 2404.14219,
  [Link](https://arxiv.org/abs/2404.14219) Cited by:
  [item 4](#S4.I2.i4.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Wang et al. (2024) C. Wang, Z. Zhao, C. Zhu, K. A. Sankararaman, M.
  Valko, X. Cao, Z. Chen, M. Khabsa, Y. Chen, H. Ma, et al. Preference
  optimization with multi-sample comparisons. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Wang et al. (2025a) H. Wang, C. Qu, Z. Huang, W. Chu, F. Lin, and W.
  Chen VL-rethinker: incentivizing self-reflection of vision-language
  models with reinforcement learning. External Links: 2504.08837,
  [Link](https://arxiv.org/abs/2504.08837) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Wang et al. (2025b) X. Wang, Z. Yang, C. Feng, H. Lu, L. Li, C.
  Lin, K. Lin, F. Huang, and L. Wang SoTA with less: mcts-guided sample
  selection for data-efficient visual reasoning self-improvement.
  External Links: 2504.07934, [Link](https://arxiv.org/abs/2504.07934)
  Cited by: [Figure
  7](#A1.F7 "In A.1 Qualitative Example ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§B.1](#A2.SS1.p1.1 "B.1 Full Prompt for VLM Rollout and Inference ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§B.2](#A2.SS2.p2.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§B.2](#A2.SS2.p3.1 "B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§B.4](#A2.SS4.p1.1 "B.4 Implementation Details ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [Appendix
  B](#A2.p1.1 "Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p4.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [item 3](#S4.I1.i3.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§4.1](#S4.SS1.p4.2 "4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§4.2](#S4.SS2.p8.1 "4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Wei et al. (2022) J. Wei, X. Wang, D. Schuurmans, M. Bosma, B.
  Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou Chain-of-thought
  prompting elicits reasoning in large language models. In Proceedings
  of the 36th International Conference on Neural Information Processing
  Systems, NIPS ’22, Red Hook, NY, USA. External Links: ISBN
  9781713871088 Cited by:
  [§A.2](#A1.SS2.p1.1 "A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Weng et al. (2023) Y. Weng, M. Zhu, F. Xia, B. Li, S. He, K. Liu,
  and J. Zhao Large language models are better reasoners with
  self-verification. External Links: 2212.09561 Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Wu et al. (2025) M. Wu, M. Li, J. Yang, J. Jiang, K. Yan, Z. Li, M.
  Zhang, and K. Nahrstedt Aha moment revisited: are vlms truly capable
  of self verification in inference-time scaling?. External Links:
  2506.17417, [Link](https://arxiv.org/abs/2506.17417) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p2.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Yu et al. (2025) Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W.
  Dai, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G.
  Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J.
  Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W.
  Ma, Y. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang DAPO: an open-source
  llm reinforcement learning system at scale. External Links:
  2503.14476, [Link](https://arxiv.org/abs/2503.14476) Cited by:
  [§4.2](#S4.SS2.p5.1 "4.2 Empirical Findings ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zelikman et al. (2022) E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman
  STaR: self-taught reasoner bootstrapping reasoning with reasoning. In
  Proceedings of the 36th International Conference on Neural Information
  Processing Systems, NIPS ’22, Red Hook, NY, USA. External Links: ISBN
  9781713871088 Cited by:
  [§1](#S1.p5.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zeng et al. (2025) W. Zeng, Y. Huang, Q. Liu, W. Liu, K. He, Z. Ma,
  and J. He SimpleRL-zoo: investigating and taming zero reinforcement
  learning for open base models in the wild. In Second Conference on
  Language Modeling, Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zhang et al. (2025) J. Zhang, J. Huang, H. Yao, S. Liu, X. Zhang, S.
  Lu, and D. Tao R1-vl: learning to reason with multimodal large
  language models via step-wise group relative policy optimization.
  External Links: 2503.12937, [Link](https://arxiv.org/abs/2503.12937)
  Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [item 3](#S4.I2.i3.p1.1 "In 4.1 Experiment Setup ‣ 4 Experiment ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zhao et al. (2025) X. Zhao, T. Xu, X. Wang, Z. Chen, D. Jin, L.
  Tan, Z. Yu, Z. Zhao, Y. He, S. Wang, et al. Boosting llm reasoning via
  spontaneous self-correction. arXiv preprint arXiv:2506.06923. Cited
  by:
  [§1](#S1.p5.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zhou et al. (2025a) H. Zhou, X. Li, R. Wang, M. Cheng, T. Zhou, and C.
  Hsieh R1-zero’s ”aha moment” in visual reasoning on a 2b non-sft
  model. External Links: 2503.05132,
  [Link](https://arxiv.org/abs/2503.05132) Cited by:
  [§A.3](#A1.SS3.p1.1 "A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.1](#S2.SS1.p2.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zhou et al. (2025b) Y. Zhou, M. Zhang, K. Li, M. Wang, Q. Liu, Q.
  Wang, J. Liu, F. Liu, S. Li, W. Li, et al. Mixture-of-minds:
  multi-agent reinforcement learning for table understanding. arXiv
  preprint arXiv:2510.20176. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"),
  [§2.2](#S2.SS2.p1.1 "2.2 Self-Improvement of LLMs/VLMs ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").
- Zhou et al. (2025c) Y. Zhou, J. Zhu, S. Qian, Z. Zhao, X. Wang, X.
  Liu, M. Li, P. Xu, W. Ai, and F. Huang DISCO balances the scales:
  adaptive domain-and difficulty-aware reinforcement learning on
  imbalanced data. arXiv preprint arXiv:2505.15074. Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 RL for LLM and VLM Reasoning ‣ 2 Related Work ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").

## Appendix A Appendix

### A.1 Qualitative Example

We list one full qualitative example in
[Figure 7](#A1.F7 "In A.1 Qualitative Example ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning").

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEuRjcucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmciIGhlaWdodD0iOTk3LjUzIiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNTUwIDk5Ny41MyIgd2lkdGg9IjU1MCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCw5OTcuNTMpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48Y2xpcHBhdGggaWQ9InBnZmNwMTMiPjxwYXRoIGQ9Ik0gLTIyNjcwLjU0IC0yMjY3MC41NCBMIDIyNjcwLjU0IC0yMjY3MC41NCBMIDIyNjcwLjU0IDIyNjcwLjU0IEwgLTIyNjcwLjU0IDIyNjcwLjU0IFogTSAwIDQuNjMgTCAwIDk5Mi45IEMgMCA5OTUuNDYgMi4wNyA5OTcuNTMgNC42MyA5OTcuNTMgTCA1NDUuMzcgOTk3LjUzIEMgNTQ3LjkzIDk5Ny41MyA1NTAgOTk1LjQ2IDU1MCA5OTIuOSBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIiAvPjwvY2xpcHBhdGg+PGcgZmlsbC1ydWxlPSJldmVub2RkIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA2MDYwOyIgZmlsbD0iIzAwNjA2MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDQuNjMgTCAwIDk5Mi45IEMgMCA5OTUuNDYgMi4wNyA5OTcuNTMgNC42MyA5OTcuNTMgTCA1NDUuMzcgOTk3LjUzIEMgNTQ3LjkzIDk5Ny41MyA1NTAgOTk1LjQ2IDU1MCA5OTIuOSBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIE0gMC42OSA0LjYzIEwgMC42OSA5NzguNjYgTCA1NDkuMzEgOTc4LjY2IEwgNTQ5LjMxIDQuNjMgQyA1NDkuMzEgMi40NSA1NDcuNTUgMC42OSA1NDUuMzcgMC42OSBMIDQuNjMgMC42OSBDIDIuNDUgMC42OSAwLjY5IDIuNDUgMC42OSA0LjYzIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGMkY5Rjk7IiBmaWxsPSIjRjJGOUY5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAuNjkgNC42MyBMIDAuNjkgOTc4LjY2IEwgNTQ5LjMxIDk3OC42NiBMIDU0OS4zMSA0LjYzIEMgNTQ5LjMxIDIuNDUgNTQ3LjU1IDAuNjkgNTQ1LjM3IDAuNjkgTCA0LjYzIDAuNjkgQyAyLjQ1IDAuNjkgMC42OSAyLjQ1IDAuNjkgNC42MyBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDYuMDEgOTgzLjI5KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjMzLjgxZW07LS1sdHgtZm8taGVpZ2h0OjAuNjllbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjkuNjEiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDkuNjEpIiB3aWR0aD0iNDY3Ljg0Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMS5GNy5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzMuODFlbTsiPgo8c3BhbiBpZD0iQTEuRjcucGljMS4xLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPkNhdCBCcmVlZCBJZGVudGlmaWNhdGlvbiB3aXRoIFNlbGYtcmVmbGVjdGlvbjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA2LjAxIDYuMDEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzcuMDNlbTstLWx0eC1mby1oZWlnaHQ6NjkuOTFlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9Ijk2Ny4zNCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOTY3LjM0KSIgd2lkdGg9IjUxMi4zOSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iQTEuRjcucGljMS4yIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM3LjAzZW07Ij4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl90b3AiIHN0eWxlPSJ3aWR0aDoyNDEuMXB0OyI+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPnVzZXI8L3NwYW4+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPldoaWNoIGJyZWVkIG9mIGNhdCBpcyB0aGlzPwo8YnIgY2xhc3M9Imx0eF9icmVhayI+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4xLjMuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPmFzc2lzdGFudAo8YnIgY2xhc3M9Imx0eF9icmVhayI+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4xLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkYwMDQwOyI+Jmx0O3RoaW5rJmd0Ozwvc3Bhbj48L3NwYW4+PC9zcGFuPjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuMS40IiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KVGhlIGNhdCBpbiB0aGUgcGljdHVyZSBhcHBlYXJzIHRvIGJlIGEgZG9tZXN0aWMgc2hvcnRoYWlyLiBUaGlzIGlzIGEgY29tbW9uIGJyZWVkIHRoYXQgY2FuIGhhdmUgYSB2YXJpZXR5IG9mIGNvYXQgcGF0dGVybnMgYW5kIGNvbG9ycy4gPC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4yIiBjbGFzcz0ibHR4X2Jsb2NrIGx0eF9hbGlnbl9sZWZ0IGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fdG9wIiBzdHlsZT0id2lkdGg6MTM2LjFwdDsiPjxpbWcgc3JjPSIyNjA3LjEwOTY2djEvZmlndXJlcy9jYXQucG5nIiBpZD0iQTEuRjcucGljMS5nMSIgY2xhc3M9Imx0eF9ncmFwaGljcyBsdHhfaW1nX2xhbmRzY2FwZSIgc3R5bGU9ImFzcGVjdC1yYXRpbzoxODgvMTI3OyIgd2lkdGg9IjE4OCIgaGVpZ2h0PSIxMjciIGFsdD0iUmVmZXIgdG8gY2FwdGlvbiI+CjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuMi4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPgo8L3NwYW4+CjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuMyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjcucGljMS4yLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+ClRoZSBzcGVjaWZpYyBtYXJraW5ncyBhbmQgY29sb3JhdGlvbiBzdWdnZXN0IGl0IGNvdWxkIGJlIGEgY2FsaWNvIG9yIGEgdG9ydG9pc2VzaGVsbCwgd2hpY2ggYXJlIGNvbW1vbiBwYXR0ZXJucyB3aXRoaW4gdGhlIGRvbWVzdGljIHNob3J0aGFpciBjYXRlZ29yeS4gSG93ZXZlciwgd2l0aG91dCBtb3JlIGRldGFpbGVkIGluZm9ybWF0aW9uIG9yIGEgY2xlYXJlciB2aWV3IG9mIHRoZSBjYXTigJlzIHBoeXNpY2FsIGZlYXR1cmVzLCBpdOKAmXMgbm90IHBvc3NpYmxlIHRvIGRldGVybWluZSB0aGUgZXhhY3QgYnJlZWQgd2l0aCBjZXJ0YWludHkuCjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjAwNDA7Ij4mbHQ7L3RoaW5rJmd0Ozwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi4zLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGODA0MDsiPlxib3hlZHt0YWJieX0KPGJyIGNsYXNzPSJsdHhfYnJlYWsiPjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjcucGljMS4yLjQiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi40LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPnVzZXIKPGJyIGNsYXNzPSJsdHhfYnJlYWsiPjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuNC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij4gQSB2ZXJpZmllciBkaXNhZ3JlZXMgd2l0aCB5b3VyIHJlc3BvbnNlLiBUaGluayBhZ2Fpbi4KPGJyIGNsYXNzPSJsdHhfYnJlYWsiPjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjcucGljMS4yLjUiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi41LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPmFzc2lzdGFudDxzcGFuIGlkPSJBMS5GNy5waWMxLjIuNS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi41LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkYwMDQwOyI+Jmx0O3RoaW5rJmd0Ozwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi41LjEuMS4yIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCI+R2l2ZW4gdGhlIHZlcmlmaWVy4oCZcyBkaXNhZ3JlZW1lbnQsIEkgd2lsbCByZS1ldmFsdWF0ZSA8L3NwYW4+dGhlIGNhdOKAmXMgYnJlZWQgYmFzZWQgb24gdGhlIHZpc2libGUgY2hhcmFjdGVyaXN0aWNzLiBUaGUgY2F0IGluIHRoZSBwaWN0dXJlIGhhcyBhIG1peCBvZiB3aGl0ZSBhbmQgYmxhY2sgZnVyIHdpdGggc29tZSBicm93biBwYXRjaGVzLCB3aGljaCBpcyB0eXBpY2FsIG9mIGEgY2FsaWNvIG9yIHRvcnRvaXNlc2hlbGwgcGF0dGVybi4gSG93ZXZlciwgdGhlIHNwZWNpZmljIG1hcmtpbmdzIGFuZCBjb2xvcmF0aW9uIGNhbiB2YXJ5IHdpZGVseSB3aXRoaW4gdGhlc2UgcGF0dGVybnMsIGFuZCB3aXRob3V0IG1vcmUgZGV0YWlsZWQgaW5mb3JtYXRpb24gb3IgYSBjbGVhcmVyIHZpZXcgb2YgdGhlIGNhdOKAmXMgcGh5c2ljYWwgZmVhdHVyZXMsIGl04oCZcyBub3QgcG9zc2libGUgdG8gZGV0ZXJtaW5lIHRoZSBleGFjdCBicmVlZCB3aXRoIGNlcnRhaW50eS4KVGhlIGNhdOKAmXMgZWFycyBhcmUgcG9pbnRlZCwgd2hpY2ggaXMgYSBjb21tb24gZmVhdHVyZSBpbiBtYW55IGJyZWVkcywgaW5jbHVkaW5nIGRvbWVzdGljIHNob3J0aGFpcnMuIFRoZSBvdmVyYWxsIGFwcGVhcmFuY2UsIGluY2x1ZGluZyB0aGUgY29hdCBwYXR0ZXJuLCBzdWdnZXN0cyBpdCBjb3VsZCBiZSBhIGRvbWVzdGljIHNob3J0aGFpciwgd2hpY2ggaXMgYSBjb21tb24gYW5kIGRpdmVyc2UgYnJlZWQgdGhhdCBjYW4gaGF2ZSBhIHZhcmlldHkgb2YgY29hdCBwYXR0ZXJucyBhbmQgY29sb3JzLgo8c3BhbiBpZD0iQTEuRjcucGljMS4yLjUuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIj5HaXZlbiB0aGUgdmVyaWZpZXLigJlzIGRpc2FncmVlbWVudDwvc3Bhbj4sIEkgd2lsbCBjb25zaWRlciB0aGUgcG9zc2liaWxpdHkgdGhhdCB0aGUgY2F0IGNvdWxkIGJlIGEgZG9tZXN0aWMgc2hvcnRoYWlyIHdpdGggYSBjYWxpY28gb3IgdG9ydG9pc2VzaGVsbCBwYXR0ZXJuLgo8c3BhbiBpZD0iQTEuRjcucGljMS4yLjUuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjAwNDA7Ij4mbHQ7L3RoaW5rJmd0Ozwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi41LjEuMS41IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkY4MDQwOyI+XGJveGVke2NhbGljb30KPGJyIGNsYXNzPSJsdHhfYnJlYWsiPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi42IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuNi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjgwMDA7Ij51c2VyCjxiciBjbGFzcz0ibHR4X2JyZWFrIj48c3BhbiBpZD0iQTEuRjcucGljMS4yLjYuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+dmVyaWZpZXIgZGlzYWdyZWVzIHdpdGggeW91ciByZXNwb25zZS4gVGhpbmsgYWdhaW4uCjxiciBjbGFzcz0ibHR4X2JyZWFrIj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi43IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuNy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjgwMDA7Ij5hc3Npc3RhbnQ8c3BhbiBpZD0iQTEuRjcucGljMS4yLjcuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+CjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuNy4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGMDA0MDsiPiZsdDt0aGluayZndDs8L3NwYW4+CkdpdmVuIHRoZSB2ZXJpZmllcuKAmXMgZGlzYWdyZWVtZW50LCBJIHdpbGwgcmUtZXZhbHVhdGUgdGhlIGNhdOKAmXMgYnJlZWQgYmFzZWQgb24gdGhlIHZpc2libGUgY2hhcmFjdGVyaXN0aWNzLiBbLi4uXSA8c3BhbiBpZD0iQTEuRjcucGljMS4yLjcuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIj5HaXZlbiB0aGUgdmVyaWZpZXLigJlzIGRpc2FncmVlbWVudCwgSSB3aWxsIGNvbnNpZGVyPC9zcGFuPiB0aGUgcG9zc2liaWxpdHkgdGhhdCB0aGUgY2F0IGNvdWxkIGJlIGEgZG9tZXN0aWMgc2hvcnRoYWlyIHdpdGggYSBjYWxpY28gb3IgdG9ydG9pc2VzaGVsbCBwYXR0ZXJuLCA8c3BhbiBpZD0iQTEuRjcucGljMS4yLjcuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIj5idXQgSSB3aWxsIGFsc28gYWNrbm93bGVkZ2UgdGhlIHBvc3NpYmlsaXR5PC9zcGFuPiB0aGF0IGl0IGNvdWxkIGJlIGFub3RoZXIgYnJlZWQgd2l0aCBzaW1pbGFyIG1hcmtpbmdzLgo8c3BhbiBpZD0iQTEuRjcucGljMS4yLjcuMS4xLjQiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjAwNDA7Ij4mbHQ7L3RoaW5rJmd0Ozwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi43LjEuMS41IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkY4MDQwOyI+XGJveGVke2NhbGljb30KPGJyIGNsYXNzPSJsdHhfYnJlYWsiPjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi44IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuOC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjAwMDA7Ij5Hcm91bmQgVHJ1dGg6PHNwYW4gaWQ9IkExLkY3LnBpYzEuMi44LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiA8c3BhbiBpZD0iQTEuRjcucGljMS4yLjguMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij5jYWxpY288L3NwYW4+IDwvc3Bhbj58IFNjb3JlOjxzcGFuIGlkPSJBMS5GNy5waWMxLjIuOC4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gPHNwYW4gaWQ9IkExLkY3LnBpYzEuMi44LjEuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkY4MDQwOyI+MS4wPC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9nPjwvc3ZnPg==)

Figure 7: Self-reflection of trained model in an example from
ThinkLite-70K ([Wang et al., 2025b](#bib.bib42)).

### A.2 Full Prompt for VLM Rollout and Inference

We construct the initial prompt header
([Figure 10](#A1.F10 "In A.2 Full Prompt for VLM Rollout and Inference ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning"))
for SVR-R1 to include carefully designed requirements, explicit
formatting instructions, and a one-shot example with detailed reasoning
steps. This structure is intended to enhance multimodal reasoning with
chain of thoughts ([Wei et al., 2022](#bib.bib1)), drawing inspiration
from previous work ReFocus ([Fu et al., 2025](#bib.bib15)).

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEuRjgucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmciIGhlaWdodD0iMTU0Ljg5IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNTUwIDE1NC44OSIgd2lkdGg9IjU1MCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwxNTQuODkpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48Y2xpcHBhdGggaWQ9InBnZmNwMTQiPjxwYXRoIGQ9Ik0gLTIyNjcwLjU0IC0yMjY3MC41NCBMIDIyNjcwLjU0IC0yMjY3MC41NCBMIDIyNjcwLjU0IDIyNjcwLjU0IEwgLTIyNjcwLjU0IDIyNjcwLjU0IFogTSAwIDQuNjMgTCAwIDE1MC4yNiBDIDAgMTUyLjgxIDIuMDcgMTU0Ljg5IDQuNjMgMTU0Ljg5IEwgNTQ1LjM3IDE1NC44OSBDIDU0Ny45MyAxNTQuODkgNTUwIDE1Mi44MSA1NTAgMTUwLjI2IEwgNTUwIDQuNjMgQyA1NTAgMi4wNyA1NDcuOTMgMCA1NDUuMzcgMCBMIDQuNjMgMCBDIDIuMDcgMCAwIDIuMDcgMCA0LjYzIFoiIC8+PC9jbGlwcGF0aD48ZyBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiMwMDYwNjA7IiBmaWxsPSIjMDA2MDYwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAgNC42MyBMIDAgMTUwLjI2IEMgMCAxNTIuODEgMi4wNyAxNTQuODkgNC42MyAxNTQuODkgTCA1NDUuMzcgMTU0Ljg5IEMgNTQ3LjkzIDE1NC44OSA1NTAgMTUyLjgxIDU1MCAxNTAuMjYgTCA1NTAgNC42MyBDIDU1MCAyLjA3IDU0Ny45MyAwIDU0NS4zNyAwIEwgNC42MyAwIEMgMi4wNyAwIDAgMi4wNyAwIDQuNjMgWiBNIDAuNjkgNC42MyBMIDAuNjkgMTM2LjAyIEwgNTQ5LjMxIDEzNi4wMiBMIDU0OS4zMSA0LjYzIEMgNTQ5LjMxIDIuNDUgNTQ3LjU1IDAuNjkgNTQ1LjM3IDAuNjkgTCA0LjYzIDAuNjkgQyAyLjQ1IDAuNjkgMC42OSAyLjQ1IDAuNjkgNC42MyBaIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojRjJGOUY5OyIgZmlsbD0iI0YyRjlGOSIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwLjY5IDQuNjMgTCAwLjY5IDEzNi4wMiBMIDU0OS4zMSAxMzYuMDIgTCA1NDkuMzEgNC42MyBDIDU0OS4zMSAyLjQ1IDU0Ny41NSAwLjY5IDU0NS4zNyAwLjY5IEwgNC42MyAwLjY5IEMgMi40NSAwLjY5IDAuNjkgMi40NSAwLjY5IDQuNjMgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA2LjAxIDE0MC42NSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozMy44MWVtOy0tbHR4LWZvLWhlaWdodDowLjY5ZW07LS1sdHgtZm8tZGVwdGg6MGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSI5LjYxIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5LjYxKSIgd2lkdGg9IjQ2Ny44NCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iQTEuRjgucGljMS4xIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjMzLjgxZW07Ij4KPHNwYW4gaWQ9IkExLkY4LnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMS5GOC5waWMxLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkZGRkY7Ij5TZWxmLVZlcmlmaWVyIFR1cm48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNi4wMSA2LjAxKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjM3LjAzZW07LS1sdHgtZm8taGVpZ2h0OjkuMDFlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjEyNC42OSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTI0LjY5KSIgd2lkdGg9IjUxMi4zOSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iQTEuRjgucGljMS4yIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM3LjAzZW07Ij4KPHNwYW4gaWQ9IkExLkY4LnBpYzEuMi4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GOC5waWMxLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjAwMDA7Ij4mbHQ7SW1hZ2UmZ3Q7PHNwYW4gaWQ9IkExLkY4LnBpYzEuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiA8L3NwYW4+Jmx0O1F1ZXN0aW9uJmd0OzxzcGFuIGlkPSJBMS5GOC5waWMxLjIuMS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPiZsdDtJbml0aWFsIFByb21wdCBIZWFkJmd0OzxzcGFuIGlkPSJBMS5GOC5waWMxLjIuMS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GOC5waWMxLjIuMiIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjgucGljMS4yLjIuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+Jmx0O1ByZXZpb3VzIFJlc3BvbnNlcyBpbiBBc3Npc3RhbnQmZ3Q7PHNwYW4gaWQ9IkExLkY4LnBpYzEuMi4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkY4LnBpYzEuMi4zIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GOC5waWMxLjIuMy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjAwMDA7Ij5Vc2VyOjxzcGFuIGlkPSJBMS5GOC5waWMxLjIuMy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gJnF1b3Q7RXZhbHVhdGUgd2hldGhlciB5b3VyIHJlc3BvbnNlIGlzIGNvcnJlY3QuIElmIHllcywgb3V0cHV0ICZsdDtZRVMmZ3Q7LCBpZiBubywgb3V0cHV0ICZsdDtOTyZndDsuIERvIG5vdCBtYWtlIGFkZGl0aW9uYWwgc3RhdGVtZW50cy4mcXVvdDsKPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GOC5waWMxLjIuNCIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjgucGljMS4yLjQuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+QXNzaXN0YW50OjxzcGFuIGlkPSJBMS5GOC5waWMxLjIuNC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvZz48L3N2Zz4=)

Figure 8: Self-Verifier Turn, with Previous Queries and Response.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEuRjkucGljMSIgY2xhc3M9Imx0eF9waWN0dXJlIGx0eF9jZW50ZXJpbmciIGhlaWdodD0iMTQwLjk3IiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB2aWV3Ym94PSIwIDAgNTUwIDE0MC45NyIgd2lkdGg9IjU1MCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIwLjRwdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwxNDAuOTcpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48Y2xpcHBhdGggaWQ9InBnZmNwMTUiPjxwYXRoIGQ9Ik0gLTIyNjcwLjU0IC0yMjY3MC41NCBMIDIyNjcwLjU0IC0yMjY3MC41NCBMIDIyNjcwLjU0IDIyNjcwLjU0IEwgLTIyNjcwLjU0IDIyNjcwLjU0IFogTSAwIDQuNjMgTCAwIDEzNi4zNCBDIDAgMTM4LjkgMi4wNyAxNDAuOTcgNC42MyAxNDAuOTcgTCA1NDUuMzcgMTQwLjk3IEMgNTQ3LjkzIDE0MC45NyA1NTAgMTM4LjkgNTUwIDEzNi4zNCBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIiAvPjwvY2xpcHBhdGg+PGcgZmlsbC1ydWxlPSJldmVub2RkIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA2MDYwOyIgZmlsbD0iIzAwNjA2MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDQuNjMgTCAwIDEzNi4zNCBDIDAgMTM4LjkgMi4wNyAxNDAuOTcgNC42MyAxNDAuOTcgTCA1NDUuMzcgMTQwLjk3IEMgNTQ3LjkzIDE0MC45NyA1NTAgMTM4LjkgNTUwIDEzNi4zNCBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIE0gMC42OSA0LjYzIEwgMC42OSAxMTkuNDIgTCA1NDkuMzEgMTE5LjQyIEwgNTQ5LjMxIDQuNjMgQyA1NDkuMzEgMi40NSA1NDcuNTUgMC42OSA1NDUuMzcgMC42OSBMIDQuNjMgMC42OSBDIDIuNDUgMC42OSAwLjY5IDIuNDUgMC42OSA0LjYzIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGMkY5Rjk7IiBmaWxsPSIjRjJGOUY5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAuNjkgNC42MyBMIDAuNjkgMTE5LjQyIEwgNTQ5LjMxIDExOS40MiBMIDU0OS4zMSA0LjYzIEMgNTQ5LjMxIDIuNDUgNTQ3LjU1IDAuNjkgNTQ1LjM3IDAuNjkgTCA0LjYzIDAuNjkgQyAyLjQ1IDAuNjkgMC42OSAyLjQ1IDAuNjkgNC42MyBaIiAvPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDYuMDEgMTI2LjczKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjMzLjgxZW07LS1sdHgtZm8taGVpZ2h0OjAuNjllbTstLWx0eC1mby1kZXB0aDowLjE5ZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjEyLjMiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDkuNjEpIiB3aWR0aD0iNDY3Ljg0Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMS5GOS5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzMuODFlbTsiPgo8c3BhbiBpZD0iQTEuRjkucGljMS4xLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExLkY5LnBpYzEuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF9ib2xkIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGRkZGRjsiPlNlbGYtR2VuZXJhdG9yIFR1cm4gd2l0aCBSZXRoaW5raW5nPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDYuMDEgNi4wMSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNy4wM2VtOy0tbHR4LWZvLWhlaWdodDo3LjgxZW07LS1sdHgtZm8tZGVwdGg6MGVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIxMDguMDgiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEwOC4wOCkiIHdpZHRoPSI1MTIuMzkiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkExLkY5LnBpYzEuMiIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNy4wM2VtOyI+CjxzcGFuIGlkPSJBMS5GOS5waWMxLjIuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjkucGljMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+Jmx0O0ltYWdlJmd0OzxzcGFuIGlkPSJBMS5GOS5waWMxLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gPC9zcGFuPiZsdDtRdWVzdGlvbiZndDs8c3BhbiBpZD0iQTEuRjkucGljMS4yLjEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IDwvc3Bhbj4mbHQ7SW5pdGlhbCBQcm9tcHQgSGVhZCZndDs8c3BhbiBpZD0iQTEuRjkucGljMS4yLjEuMS4zIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjkucGljMS4yLjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkY5LnBpYzEuMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPiZsdDtQcmV2aW91cyBSZXNwb25zZXMgaW4gQXNzaXN0YW50Jmd0OzxzcGFuIGlkPSJBMS5GOS5waWMxLjIuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GOS5waWMxLjIuMyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjkucGljMS4yLjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+VXNlcjo8c3BhbiBpZD0iQTEuRjkucGljMS4yLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+ICZxdW90O0EgdmVyaWZpZXIgZGlzYWdyZWVzIHdpdGggeW91ciByZXNwb25zZS4gVGhpbmsgYWdhaW4uJnF1b3Q7Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjkucGljMS4yLjQiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkY5LnBpYzEuMi40LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPkFzc2lzdGFudDo8c3BhbiBpZD0iQTEuRjkucGljMS4yLjQuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L2c+PC9zdmc+)

Figure 9: Self-Generator Turn with the Rethinking Trigger.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEuRjEwLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSBsdHhfY2VudGVyaW5nIiBoZWlnaHQ9IjUyNy43OSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDU1MCA1MjcuNzkiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsNTI3Ljc5KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGNsaXBwYXRoIGlkPSJwZ2ZjcDE2Ij48cGF0aCBkPSJNIC0yMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAyMjY3MC41NCBMIC0yMjY3MC41NCAyMjY3MC41NCBaIE0gMCA0LjYzIEwgMCA1MjMuMTcgQyAwIDUyNS43MiAyLjA3IDUyNy43OSA0LjYzIDUyNy43OSBMIDU0NS4zNyA1MjcuNzkgQyA1NDcuOTMgNTI3Ljc5IDU1MCA1MjUuNzIgNTUwIDUyMy4xNyBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIiAvPjwvY2xpcHBhdGg+PGcgZmlsbC1ydWxlPSJldmVub2RkIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA2MDYwOyIgZmlsbD0iIzAwNjA2MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDQuNjMgTCAwIDUyMy4xNyBDIDAgNTI1LjcyIDIuMDcgNTI3Ljc5IDQuNjMgNTI3Ljc5IEwgNTQ1LjM3IDUyNy43OSBDIDU0Ny45MyA1MjcuNzkgNTUwIDUyNS43MiA1NTAgNTIzLjE3IEwgNTUwIDQuNjMgQyA1NTAgMi4wNyA1NDcuOTMgMCA1NDUuMzcgMCBMIDQuNjMgMCBDIDIuMDcgMCAwIDIuMDcgMCA0LjYzIFogTSAwLjY5IDQuNjMgTCAwLjY5IDUwNi4yNCBMIDU0OS4zMSA1MDYuMjQgTCA1NDkuMzEgNC42MyBDIDU0OS4zMSAyLjQ1IDU0Ny41NSAwLjY5IDU0NS4zNyAwLjY5IEwgNC42MyAwLjY5IEMgMi40NSAwLjY5IDAuNjkgMi40NSAwLjY5IDQuNjMgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0YyRjlGOTsiIGZpbGw9IiNGMkY5RjkiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC42OSA0LjYzIEwgMC42OSA1MDYuMjQgTCA1NDkuMzEgNTA2LjI0IEwgNTQ5LjMxIDQuNjMgQyA1NDkuMzEgMi40NSA1NDcuNTUgMC42OSA1NDUuMzcgMC42OSBMIDQuNjMgMC42OSBDIDIuNDUgMC42OSAwLjY5IDIuNDUgMC42OSA0LjYzIFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNi4wMSA1MTMuNTYpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzMuODFlbTstLWx0eC1mby1oZWlnaHQ6MC42OWVtOy0tbHR4LWZvLWRlcHRoOjAuMTllbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTIuMyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOS42MSkiIHdpZHRoPSI0NjcuODQiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkExLkYxMC5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzMuODFlbTsiPgo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMS5GMTAucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+SW5pdGlhbCBQcm9tcHQgSGVhZCwgZm9yIFZMTSBJbmZlcmVuY2Ugb3IgUm9sbG91dDwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA2LjAxIDYuMDEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzcuMDNlbTstLWx0eC1mby1oZWlnaHQ6MzUuNzdlbTstLWx0eC1mby1kZXB0aDowZW07Zm9udC1zaXplOjEwcHQ7IiBoZWlnaHQ9IjQ5NC45MSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNDk0LjkxKSIgd2lkdGg9IjUxMi4zOSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMiIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozNy4wM2VtOyI+CjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjEiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjAwMDA7Ij4mbHQ7SW1hZ2UmZ3Q7PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gPC9zcGFuPiZsdDtRdWVzdGlvbiZndDs8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xLjEuMiIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMiIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPlVzZXI6PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjMiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMy4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij5Zb3Ugd2lsbCBiZSBnaXZlbiBhIHRhYmxlIGZpZ3VyZTogaW1hZ2VfMSBhbmQgYSBxdWVzdGlvbiwgcGxlYXNlIGFuc3dlciB0aGUgcXVlc3Rpb24gdXNpbmcgdGhlIGluZm9ybWF0aW9uIGluIHRoZSBpbWFnZS48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4zLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNCIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi40LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPiMgUkVRVUlSRU1FTlRTICM6PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjUiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjAwNDA7Ij4xLiBUaGUgZ2VuZXJhdGVkIGFjdGlvbnMgY2FuIHJlc29sdmUgdGhlIGdpdmVuIHVzZXIgcmVxdWVzdCAjIFVTRVIgUkVRVUVTVCAjIHBlcmZlY3RseS4gVGhlIHVzZXIgcmVxdWVzdCBpcyByZWFzb25hYmxlIGFuZCBjYW4gYmUgc29sdmVkLiBUcnkgeW91ciBiZXN0IHRvIHNvbHZlIHRoZSByZXF1ZXN0LjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjUuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj4yLiBJZiB5b3UgdGhpbmsgeW91IGdvdCB0aGUgYW5zd2VyLCB1c2UgPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNS4xLjIiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij5BTlNXRVI6PC9zcGFuPiAmbHQ7eW91ciBhbnN3ZXImZ3Q7IFBsZWFzZSBleHRyYWN0IHRoZSBmaW5hbCBhbnN3ZXIgaW4gPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNS4xLjMiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij5GSU5BTCBBTlNXRVI6PC9zcGFuPiAmbHQ7ZmluYWwgYW5zd2VyJmd0OyBhbmQgZW5kcyB3aXRoIDxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjUuMS40IiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkY4MDQwOyI+VEVSTUlOQVRFPC9zcGFuPi4gMy4gUGxlYXNlIGluY2x1ZGUgdGhlIHJlYXNvbmluZyBzdGVwIGFmdGVyIFRIT1VHSFQ8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi41LjEuNSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPi4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjYiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjgwMDA7Ij4jIEVYQU1QTEUgIzo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi42LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi43LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPiMgVVNFUiBSRVFVRVNUICM6ICZsdDtpbWFnZV8xJmd0OyBXaG8gaGFkIHRoZSBzYW1lIGdhbWUgdmVyc2lvbiBhcyBKb2huIFJvdGg/PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuNy4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjgiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuOC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjAwNDA7Ij5USE9VR0hUIDA6IFRvIGlkZW50aWZ5IHdobyBoYWQgdGhlIHNhbWUgZ2FtZSB2ZXJzaW9uIGFzIEpvaG4gUm90aCwgSSBuZWVkIHRvIGZvY3VzIG9uIHRoZSDigJlHYW1lIFZlcnNpb27igJkgY29sdW1uIGFuZCB0aGUg4oCZTWFuYWdlciBOYW1l4oCZIGNvbHVtbi4gQWxzbywgSSBuZWVkIHRvIGZvY3VzIG9uIGFsbCB0aGUgcm93cyBzbyBJIGRvIG5vdCBuZWVkIHRvIGZvY3VzIG9uIHNvbWUgc3BlY2lmaWMgcm93cy4gSSBjYW4gc2VlIHRoZSDigJlHYW1lIFZlcnNpb27igJkgY29sdW1uIGFuZCB0aGUg4oCZTWFuYWdlciBOYW1l4oCZIGNvbHVtbiBtb3JlIGNsZWFybHkuIFRoZSBnYW1lIHZlcnNpb24gb2YgSm9obiBSb3RoIGlzIOKAmXYxLjLigJkuIE90aGVyIHBlb3BsZSB3aXRoIHRoZSBzYW1lIGdhbWUgdmVyc2lvbiBhcmUg4oCZQWxpY2UgU21pdGjigJkgYW5kIOKAmUJvYiBKb2huc29u4oCZLgo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi44LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuOSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi45LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGODA0MDsiPkFOU1dFUjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjkuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+OiA8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi45LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+4oCZQWxpY2UgU21pdGjigJkgYW5kIOKAmUJvYiBKb2huc29u4oCZIGhhZCB0aGUgc2FtZSBnYW1lIHZlcnNpb24gYXMgSm9obiBSb3RoIGFyZS48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xMCIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xMC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij5GSU5BTCBBTlNXRVI6PHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMTAuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+IDxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjEwLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+QWxpY2UgU21pdGggfHwgQm9iIEpvaG5zb24uIDwvc3Bhbj48L3NwYW4+VEVSTUlOQVRFPHNwYW4gaWQ9IkExLkYxMC5waWMxLjIuMTAuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjgwMDA7Ij4jIEVORCBFWEFNUExFICM8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjEyIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTAucGljMS4yLjEyLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPkFzc2lzdGFudDo8c3BhbiBpZD0iQTEuRjEwLnBpYzEuMi4xMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvZz48L3N2Zz4=)

Figure 10: Full Prompt for VLM Inference or Rollout, with an In-context
Example and Clear Requirements.

### A.3 Full Prompt for Reward Judge

SVR-R1 expects an outcome-based binary reward. For complex,
semi–open-form visual question answering tasks, such as the visual table
and chart reasoning dataset in ReFocus, we follow prior work ([Fu et
al., 2025](#bib.bib15)) and employ a LLM judge to directly compare the
final prediction with the ground truth
([Figure 11](#A1.F11 "In A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")).
For this evaluation, we instruct the LLM to output a binary decision
based solely on the prediction and the ground-truth answer. We use
gpt-oss-120b ([Team, 2025b](#bib.bib48)), which performs well on this
matching task with reasoning traces
([Figure 12](#A1.F12 "In A.3 Full Prompt for Reward Judge ‣ Appendix A Appendix ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")),
and runs efficiently on a single 80 GB GPU using vLLM as the serving
engine. Importantly, we do not introduce any additional information via
the judge: the model is not asked to answer the question; it only
determines whether the prediction matches the ground truth. To our
knowledge, integrating an LLM-judge reward into RL training for VLMs is
novel in this domain; prior work  ([Zhou et al., 2025a](#bib.bib4);
[Chen et al., 2025a](#bib.bib5); [Zhang et al., 2025](#bib.bib6); [Huang
et al., 2025a](#bib.bib7); [Liu et al., 2025](#bib.bib8); [Deng et al.,
2025](#bib.bib9); [Wang et al., 2025a](#bib.bib10); [Wang et al.,
2025b](#bib.bib42)) has focused primarily on verifiable rewards
applicable to strictly verifiable settings (e.g., geometry mathematics
([Chen et al., 2021](#bib.bib41))). Incorporating a computationally
intensive judge into RL training without incurring substantial cost is
non-trivial, and we hope our method and open-source framework will
encourage broader consideration of semi-verifiable tasks in
vision–language domains.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEuRjExLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSBsdHhfY2VudGVyaW5nIiBoZWlnaHQ9IjU3OC4yMiIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDU1MCA1NzguMjIiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsNTc4LjIyKSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGNsaXBwYXRoIGlkPSJwZ2ZjcDE3Ij48cGF0aCBkPSJNIC0yMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAyMjY3MC41NCBMIC0yMjY3MC41NCAyMjY3MC41NCBaIE0gMCA0LjYzIEwgMCA1NzMuNTkgQyAwIDU3Ni4xNSAyLjA3IDU3OC4yMiA0LjYzIDU3OC4yMiBMIDU0NS4zNyA1NzguMjIgQyA1NDcuOTMgNTc4LjIyIDU1MCA1NzYuMTUgNTUwIDU3My41OSBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIiAvPjwvY2xpcHBhdGg+PGcgZmlsbC1ydWxlPSJldmVub2RkIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA2MDYwOyIgZmlsbD0iIzAwNjA2MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDQuNjMgTCAwIDU3My41OSBDIDAgNTc2LjE1IDIuMDcgNTc4LjIyIDQuNjMgNTc4LjIyIEwgNTQ1LjM3IDU3OC4yMiBDIDU0Ny45MyA1NzguMjIgNTUwIDU3Ni4xNSA1NTAgNTczLjU5IEwgNTUwIDQuNjMgQyA1NTAgMi4wNyA1NDcuOTMgMCA1NDUuMzcgMCBMIDQuNjMgMCBDIDIuMDcgMCAwIDIuMDcgMCA0LjYzIFogTSAwLjY5IDQuNjMgTCAwLjY5IDU1Ni42NiBMIDU0OS4zMSA1NTYuNjYgTCA1NDkuMzEgNC42MyBDIDU0OS4zMSAyLjQ1IDU0Ny41NSAwLjY5IDU0NS4zNyAwLjY5IEwgNC42MyAwLjY5IEMgMi40NSAwLjY5IDAuNjkgMi40NSAwLjY5IDQuNjMgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0YyRjlGOTsiIGZpbGw9IiNGMkY5RjkiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC42OSA0LjYzIEwgMC42OSA1NTYuNjYgTCA1NDkuMzEgNTU2LjY2IEwgNTQ5LjMxIDQuNjMgQyA1NDkuMzEgMi40NSA1NDcuNTUgMC42OSA1NDUuMzcgMC42OSBMIDQuNjMgMC42OSBDIDIuNDUgMC42OSAwLjY5IDIuNDUgMC42OSA0LjYzIFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNi4wMSA1NjMuOTgpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzMuODFlbTstLWx0eC1mby1oZWlnaHQ6MC42OWVtOy0tbHR4LWZvLWRlcHRoOjAuMTllbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTIuMyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOS42MSkiIHdpZHRoPSI0NjcuODQiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzMuODFlbTsiPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+UHJvbXB0IGZvciBMTE0gUmV3YXJkIEp1ZGdlPC9zcGFuPjwvc3Bhbj4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDYuMDEgOS4wOSkiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozNy4wM2VtOy0tbHR4LWZvLWhlaWdodDozOS4xOWVtOy0tbHR4LWZvLWRlcHRoOjAuMjJlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iNTQ1LjM0IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA1NDIuMjYpIiB3aWR0aD0iNTEyLjM5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM3LjAzZW07Ij4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPlVzZXI6PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij5Zb3UgYXJlIGdpdmVuIGEgcHJlZGljdGlvbiBmb3IgdGhlIHF1ZXN0aW9uLiBZb3UgbmVlZCB0byByYXRlIGl0IGdpdmVuIHRoZSBjb3JyZWN0IGFuc3dlci4gRGlzcmVnYXJkIHRoZSBmb3JtYXQsIGFuZCBvbmx5IHJhdGUgYmFzZWQgb24gdGhlIGNvbnRlbnQuIElmIHlvdSB0aGluayB0aGUgcHJlZGljdGlvbiBpcyBjb3JyZWN0LCBpLmUuIHNhbWUgYXMgdGhlIGNvcnJlY3QgYW5zd2VyLCB0aGVuIHJldHVybiAxLCBvdGhlcndpc2UgcmV0dXJuIDAuIFJldHVybiAwIG9yIDEgb25seS48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPiMgRXhhbXBsZTxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi40IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjQuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+IyBRdWVzdGlvbjogV2hhdCBpcyB0aGUgaGVpZ2h0IG9mIHRoZSB0b3dlcj88c3BhbiBpZD0iQTEuRjExLnBpYzEuMi40LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuNSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi41LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPiMgUHJlZGljdGlvbjogQU5TV0VSOiBUaGUgdG93ZXIgaXMgYnVpbHQgaW4gQ2hpbmEgZnJvbSAyMDAgeWVhcnMgYWdvLiBUaGUgdG90YWwgaGlnaHQgb2YgdGhlIHRvd2VyIGlzIDE4MCBNZXRlcnMuIDxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjUuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkY4MDQwOyI+VEVSTUlOQVRFPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuNS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjYiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuNi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij4jIENvcnJlY3QgQW5zd2VyOiA4MCBNZXRlcnM8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi42LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuNyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi43LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGODA0MDsiPiMgWW91ciBSZXNwb25zZTogMDxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjcuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi44IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjguMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkY4MDAwOyI+IyBFeGFtcGxlPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuOC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjkiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuOS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij4jIFF1ZXN0aW9uOiBXaGF0IGlzIHRoZSBkaWZmZXJlbmNlPzxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjkuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMCIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij4jIFByZWRpY3Rpb246IEFOU1dFUjogNjklLiA8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij5URVJNSU5BVEU8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMC4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjExIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjExLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPiMgQ29ycmVjdCBBbnN3ZXI6IDY5PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMiIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMi4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij4jIFlvdXIgUmVzcG9uc2U6IDE8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjEzIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjEzLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGODAwMDsiPiMgRXhhbXBsZTxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjEzLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTQiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTQuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+IyBRdWVzdGlvbjogV2hhdCBpcyB0aGUgaW5jcmVhc2U/PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTQuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xNSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xNS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwRkY7Ij4jIFByZWRpY3Rpb246IEFOU1dFUjogMyUuIDxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjE1LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGODA0MDsiPlRFUk1JTkFURTxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjE1LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTYiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTYuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+IyBDb3JyZWN0IEFuc3dlcjogMC4wMzxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjE2LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTciIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTcuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojQkY4MDQwOyI+IyBZb3VyIFJlc3BvbnNlOiAxPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMTcuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xOCIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xOC4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRjgwMDA7Ij4jIEV4YW1wbGU8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4xOC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjE5IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjE5LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPiMgUXVlc3Rpb246IFdoYXQgaXMgdGhlIHJhdGlvPzxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjE5LjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjAiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjAuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+IyBQcmVkaWN0aW9uOiBBTlNXRVI6IDEuNTQxLiA8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yMC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNCRjgwNDA7Ij5URVJNSU5BVEU8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yMC4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIxIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIxLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPiMgQ29ycmVjdCBBbnN3ZXI6IDEuNTQ8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIyIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIyLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGODA0MDsiPiMgWW91ciBSZXNwb25zZTogMTxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIyLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjMiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjMuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+IyBRdWVzdGlvbjoge3F1ZXN0aW9ufTxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjIzLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjQiIGNsYXNzPSJsdHhfcCBsdHhfYWxpZ25fbGVmdCI+PHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjQuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMEZGOyI+IyBQcmVkaWN0aW9uOiB7cHJlZGljdGlvbn08c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yNC4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjI1IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjI1LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPiMgQ29ycmVjdCBBbnN3ZXI6IHtzb2x1dGlvbn08c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yNS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjI2IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjI2LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPkFzc2lzdGFudDo8c3BhbiBpZD0iQTEuRjExLnBpYzEuMi4yNi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4KPC9zcGFuPjwvc3Bhbj48L3NwYW4+CjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjI3IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTEucGljMS4yLjI3LjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0JGODA0MDsiPiMgWW91ciBSZXNwb25zZTog4oCZ4oCZ4oCZPHNwYW4gaWQ9IkExLkYxMS5waWMxLjIuMjcuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L2c+PC9zdmc+)

Figure 11: Prompt for LLM Judge in Semi-open Questions.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTEuRjEyLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSBsdHhfY2VudGVyaW5nIiBoZWlnaHQ9IjE3Ny4yNiIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDU1MCAxNzcuMjYiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMTc3LjI2KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGNsaXBwYXRoIGlkPSJwZ2ZjcDE4Ij48cGF0aCBkPSJNIC0yMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAyMjY3MC41NCBMIC0yMjY3MC41NCAyMjY3MC41NCBaIE0gMCA0LjYzIEwgMCAxNzIuNjMgQyAwIDE3NS4xOCAyLjA3IDE3Ny4yNiA0LjYzIDE3Ny4yNiBMIDU0NS4zNyAxNzcuMjYgQyA1NDcuOTMgMTc3LjI2IDU1MCAxNzUuMTggNTUwIDE3Mi42MyBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIiAvPjwvY2xpcHBhdGg+PGcgZmlsbC1ydWxlPSJldmVub2RkIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA2MDYwOyIgZmlsbD0iIzAwNjA2MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDQuNjMgTCAwIDE3Mi42MyBDIDAgMTc1LjE4IDIuMDcgMTc3LjI2IDQuNjMgMTc3LjI2IEwgNTQ1LjM3IDE3Ny4yNiBDIDU0Ny45MyAxNzcuMjYgNTUwIDE3NS4xOCA1NTAgMTcyLjYzIEwgNTUwIDQuNjMgQyA1NTAgMi4wNyA1NDcuOTMgMCA1NDUuMzcgMCBMIDQuNjMgMCBDIDIuMDcgMCAwIDIuMDcgMCA0LjYzIFogTSAwLjY5IDQuNjMgTCAwLjY5IDE1NS43IEwgNTQ5LjMxIDE1NS43IEwgNTQ5LjMxIDQuNjMgQyA1NDkuMzEgMi40NSA1NDcuNTUgMC42OSA1NDUuMzcgMC42OSBMIDQuNjMgMC42OSBDIDIuNDUgMC42OSAwLjY5IDIuNDUgMC42OSA0LjYzIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGMkY5Rjk7IiBmaWxsPSIjRjJGOUY5IiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggc3R5bGU9InN0cm9rZTpub25lIiBkPSJNIDAuNjkgNC42MyBMIDAuNjkgMTU1LjcgTCA1NDkuMzEgMTU1LjcgTCA1NDkuMzEgNC42MyBDIDU0OS4zMSAyLjQ1IDU0Ny41NSAwLjY5IDU0NS4zNyAwLjY5IEwgNC42MyAwLjY5IEMgMi40NSAwLjY5IDAuNjkgMi40NSAwLjY5IDQuNjMgWiIgLz48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA2LjAxIDE2My4wMikiPjxmb3JlaWdub2JqZWN0IHN0eWxlPSItLWx0eC1mby13aWR0aDozMy44MWVtOy0tbHR4LWZvLWhlaWdodDowLjY5ZW07LS1sdHgtZm8tZGVwdGg6MC4xOWVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIxMi4zIiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA5LjYxKSIgd2lkdGg9IjQ2Ny44NCI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPgo8c3BhbiBpZD0iQTEuRjEyLnBpYzEuMSIgY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIHN0eWxlPSJ3aWR0aDozMy44MWVtOyI+CjxzcGFuIGlkPSJBMS5GMTIucGljMS4xLjEiIGNsYXNzPSJsdHhfcCI+PHNwYW4gaWQ9IkExLkYxMi5waWMxLjEuMS4xIiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfYm9sZCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiNGRkZGRkY7Ij5HUFQtb3NzIEp1ZGdlIFNhbXBsZSwgd2l0aCBSZWFzb248L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNi4wMSA5LjA5KSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjM3LjAzZW07LS1sdHgtZm8taGVpZ2h0OjEwLjIxZW07LS1sdHgtZm8tZGVwdGg6MC4yMmVtO2ZvbnQtc2l6ZToxMHB0OyIgaGVpZ2h0PSIxNDQuMzciIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDE0MS4yOSkiIHdpZHRoPSI1MTIuMzkiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkExLkYxMi5waWMxLjIiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzcuMDNlbTsiPgo8c3BhbiBpZD0iQTEuRjEyLnBpYzEuMi4xIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTIucGljMS4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+Q29udGVudDo8c3BhbiBpZD0iQTEuRjEyLnBpYzEuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiAxCjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTEuRjEyLnBpYzEuMi4yIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMS5GMTIucGljMS4yLjIuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+cmVhc29uaW5nIGNvbnRlbnQ6PHNwYW4gaWQ9IkExLkYxMi5waWMxLjIuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9Ii0tbHR4LWZnLWNvbG9yOiMwMDAwMDA7Ij4gJnF1b3Q7V2UgbmVlZCB0byBjb21wYXJlIHByZWRpY3Rpb24gY29udGVudCB3aXRoIGNvcnJlY3QgYW5zd2VyLiBQcmVkaWN0aW9uOiAmcXVvdDtUaGUgZGlmZmVyZW5jZSBpbiB2YWx1ZSBiZXR3ZWVuIHRoZSBncmVlbiBiYXIgKENhbWVyb29uKSBhbmQgdGhlIG9yYW5nZSBiYXIgKExvdyBpbmNvbWUpIGlzIDAuMDguJnF1b3Q7IENvcnJlY3QgYW5zd2VyOiAmcXVvdDswLjA4JnF1b3Q7LiBUaGUgcHJlZGljdGlvbiBpbmNsdWRlcyB0aGUgc2FtZSBudW1lcmljIHZhbHVlIDAuMDgsIGV2ZW4gdGhvdWdoIGV4dHJhIHRleHQuIEFjY29yZGluZyB0byBndWlkZWxpbmVzLCB3ZSBjb25zaWRlciBjb3JyZWN0IGlmIHNhbWUgYXMgY29ycmVjdCBhbnN3ZXIgKG1heWJlIG51bWVyaWMgZXF1aXZhbGVuY2UpLiBTaG91bGQgYmUgY29uc2lkZXJlZCBjb3JyZWN0LCBzbyByZXR1cm4gMS4mcXVvdDsKPC9zcGFuPjwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvZz48L3N2Zz4=)

Figure 12: GPT‑OSS provides high‑quality judgments, accompanied by
step‑by‑step reasoning.

## Appendix B ThinkLite-VL

As we mention in the Sec. 4.1, we experiment beyond specific table and
chart tasks to more general reasoning, utilizing ThinkLite-VL-70K
dataset from recent multimodal reasoning work ([Wang et al.,
2025b](#bib.bib42)), which includes multimodal reasoning tasks from
multimodal reasoning ([Lu et al., 2021](#bib.bib52); [Chen et al.,
2022](#bib.bib53); [Seo et al., 2015](#bib.bib54)), natural image
understanding ([Kahou et al., 2018](#bib.bib55); [Lu et al.,
2022a](#bib.bib56); [Marino et al., 2019](#bib.bib57)), and chart
interpretation ([Lu et al., 2022b](#bib.bib58); [Lu et al.,
2023](#bib.bib59)). Although SVR-R1 demonstrates improvements over
standard GRPO training, we were unable to achieve the absolute high
accuracy reported in ([Wang et al., 2025b](#bib.bib42)) in standard GRPO
training, because reproducible training and evaluation have not yet been
fully released.

### B.1 Full Prompt for VLM Rollout and Inference

For a fair comparison with the original paper ([Wang et al.,
2025b](#bib.bib42)), we use the same prompt for VLM inference and
rollout during RL training, following the standard RL-VLM
setup ([Figure 13](#A2.F13 "In B.1 Full Prompt for VLM Rollout and Inference ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")).
The prompt format clearly separates the reasoning steps from the final
answer, with the answer presented in a boxed layout.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iQTIuRjEzLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSBsdHhfY2VudGVyaW5nIiBoZWlnaHQ9IjEzNy4yOCIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDU1MCAxMzcuMjgiIHdpZHRoPSI1NTAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsMTM3LjI4KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGNsaXBwYXRoIGlkPSJwZ2ZjcDE5Ij48cGF0aCBkPSJNIC0yMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAtMjI2NzAuNTQgTCAyMjY3MC41NCAyMjY3MC41NCBMIC0yMjY3MC41NCAyMjY3MC41NCBaIE0gMCA0LjYzIEwgMCAxMzIuNjUgQyAwIDEzNS4yMSAyLjA3IDEzNy4yOCA0LjYzIDEzNy4yOCBMIDU0NS4zNyAxMzcuMjggQyA1NDcuOTMgMTM3LjI4IDU1MCAxMzUuMjEgNTUwIDEzMi42NSBMIDU1MCA0LjYzIEMgNTUwIDIuMDcgNTQ3LjkzIDAgNTQ1LjM3IDAgTCA0LjYzIDAgQyAyLjA3IDAgMCAyLjA3IDAgNC42MyBaIiAvPjwvY2xpcHBhdGg+PGcgZmlsbC1ydWxlPSJldmVub2RkIj48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMDA2MDYwOyIgZmlsbD0iIzAwNjA2MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIHN0eWxlPSJzdHJva2U6bm9uZSIgZD0iTSAwIDQuNjMgTCAwIDEzMi42NSBDIDAgMTM1LjIxIDIuMDcgMTM3LjI4IDQuNjMgMTM3LjI4IEwgNTQ1LjM3IDEzNy4yOCBDIDU0Ny45MyAxMzcuMjggNTUwIDEzNS4yMSA1NTAgMTMyLjY1IEwgNTUwIDQuNjMgQyA1NTAgMi4wNyA1NDcuOTMgMCA1NDUuMzcgMCBMIDQuNjMgMCBDIDIuMDcgMCAwIDIuMDcgMCA0LjYzIFogTSAwLjY5IDQuNjMgTCAwLjY5IDExNS43MyBMIDU0OS4zMSAxMTUuNzMgTCA1NDkuMzEgNC42MyBDIDU0OS4zMSAyLjQ1IDU0Ny41NSAwLjY5IDU0NS4zNyAwLjY5IEwgNC42MyAwLjY5IEMgMi40NSAwLjY5IDAuNjkgMi40NSAwLjY5IDQuNjMgWiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0YyRjlGOTsiIGZpbGw9IiNGMkY5RjkiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBzdHlsZT0ic3Ryb2tlOm5vbmUiIGQ9Ik0gMC42OSA0LjYzIEwgMC42OSAxMTUuNzMgTCA1NDkuMzEgMTE1LjczIEwgNTQ5LjMxIDQuNjMgQyA1NDkuMzEgMi40NSA1NDcuNTUgMC42OSA1NDUuMzcgMC42OSBMIDQuNjMgMC42OSBDIDIuNDUgMC42OSAwLjY5IDIuNDUgMC42OSA0LjYzIFoiIC8+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNi4wMSAxMjMuMDUpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzMuODFlbTstLWx0eC1mby1oZWlnaHQ6MC42OWVtOy0tbHR4LWZvLWRlcHRoOjAuMTllbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTIuMyIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgOS42MSkiIHdpZHRoPSI0NjcuODQiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij4KPHNwYW4gaWQ9IkEyLkYxMy5waWMxLjEiIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBzdHlsZT0id2lkdGg6MzMuODFlbTsiPgo8c3BhbiBpZD0iQTIuRjEzLnBpYzEuMS4xIiBjbGFzcz0ibHR4X3AiPjxzcGFuIGlkPSJBMi5GMTMucGljMS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X2JvbGQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkZGRkZGOyI+SW5pdGlhbCBQcm9tcHQgSGVhZCwgZm9yIFZMTSBJbmZlcmVuY2Ugb3IgUm9sbG91dDwvc3Bhbj48L3NwYW4+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA2LjAxIDYuMDEpIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MzcuMDNlbTstLWx0eC1mby1oZWlnaHQ6Ny41NGVtOy0tbHR4LWZvLWRlcHRoOjBlbTtmb250LXNpemU6MTBwdDsiIGhlaWdodD0iMTA0LjM5IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMDQuMzkpIiB3aWR0aD0iNTEyLjM5Ij48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+CjxzcGFuIGlkPSJBMi5GMTMucGljMS4yIiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgc3R5bGU9IndpZHRoOjM3LjAzZW07Ij4KPHNwYW4gaWQ9IkEyLkYxMy5waWMxLjIuMSIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTIuRjEzLnBpYzEuMi4xLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6I0ZGMDAwMDsiPiZsdDtJbWFnZSZndDs8c3BhbiBpZD0iQTIuRjEzLnBpYzEuMi4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPiA8L3NwYW4+Jmx0O1F1ZXN0aW9uJmd0OzxzcGFuIGlkPSJBMi5GMTMucGljMS4yLjEuMS4yIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTIuRjEzLnBpYzEuMi4yIiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMi5GMTMucGljMS4yLjIuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+VXNlcjo8c3BhbiBpZD0iQTIuRjEzLnBpYzEuMi4yLjEuMSIgY2xhc3M9Imx0eF90ZXh0IiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDAwMDsiPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj4KPHNwYW4gaWQ9IkEyLkYxMy5waWMxLjIuMyIgY2xhc3M9Imx0eF9wIGx0eF9hbGlnbl9sZWZ0Ij48c3BhbiBpZD0iQTIuRjEzLnBpYzEuMi4zLjEiIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBzdHlsZT0iLS1sdHgtZmctY29sb3I6IzAwMDBGRjsiPllvdSBGSVJTVCB0aGluayBhYm91dCB0aGUgcmVhc29uaW5nIHByb2Nlc3MgYXMgYW4gaW50ZXJuYWwgbW9ub2xvZ3VlIGFuZCB0aGVuIHByb3ZpZGUgdGhlIGZpbmFsCmFuc3dlci4gVGhlIHJlYXNvbmluZyBwcm9jZXNzIE1VU1QgQkUgZW5jbG9zZWQgd2l0aGluICZsdDt0aGluayZndDsgJmx0Oy90aGluayZndDsgdGFncy4gVGhlIGZpbmFsCmFuc3dlciBNVVNUIEJFIHB1dCBpbiAvYm94LjxzcGFuIGlkPSJBMi5GMTMucGljMS4yLjMuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8c3BhbiBpZD0iQTIuRjEzLnBpYzEuMi40IiBjbGFzcz0ibHR4X3AgbHR4X2FsaWduX2xlZnQiPjxzcGFuIGlkPSJBMi5GMTMucGljMS4yLjQuMSIgY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojRkYwMDAwOyI+QXNzaXN0YW50OjxzcGFuIGlkPSJBMi5GMTMucGljMS4yLjQuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSItLWx0eC1mZy1jb2xvcjojMDAwMDAwOyI+Cjwvc3Bhbj48L3NwYW4+PC9zcGFuPgo8L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L2c+PC9zdmc+)

Figure 13: Full Prompt for VLM Inference or Rollout for ThinkLite-VL
Dataset.

### B.2 Experiment

Dataset Split: 70K Split. The training set comprises 70k examples
spanning Geometry3K, GeoQA, and GEOS (math questions with image input)
([Lu et al., 2021](#bib.bib52); [Chen et al., 2022](#bib.bib53); [Seo et
al., 2015](#bib.bib54)); FigureQA, ScienceQA, and OK-VQA for image
understanding ([Kahou et al., 2018](#bib.bib55); [Lu et al.,
2022a](#bib.bib56); [Marino et al., 2019](#bib.bib57)); and IconQA and
TabMWP for chart understanding ([Lu et al., 2022b](#bib.bib58); [Lu et
al., 2023](#bib.bib59)).

Difficult Data Selection: 11K Split. A key contribution of ([Wang et
al., 2025b](#bib.bib42)) is that sample difficulty critically influences
RFT effectiveness. They employ Monte Carlo Tree Search (MCTS) to select
hard examples for VLM RL training, constructing an 11k high-difficulty
subset from the full 70k dataset and reporting improved sample
efficiency and higher test accuracy, compared to training in 70K split.

![Refer to caption](2607.10966v1/figures/difficult-turns.png)

Figure 14: Mean number of Turns vs. Training Steps on train/val splits.
We allow a maximum of five verification turns, counting each generation
and verification round.

Experiment Setup. For the Thinking-VL dataset, we strictly follow the
hyperparameter configuration in ([Wang et al., 2025b](#bib.bib42)),
which adheres to the EasyR1 default RL-VLM setup. We use the AdamW
optimizer ([Loshchilov and Hutter, 2019](#bib.bib50)) with an initial
learning rate of $`1\times 10^{-6}`$. We employ a micro-batch size of 4
per GPU and a mini-batch size of 128 per update, with an overall
training batch size of 512. We train Qwen-2.5-VL 7B model on 8×8 A100
GPUs (80GB). We use bf16 precision and set the decoding temperature to
1.0 during rollouts to encourage exploration. The rollout group size is
32. We include a KL-divergence term in the loss with coefficient
$`\beta=1\times 10^{-2}`$, larger than that used for table and chart
training. We set the maximum number of self-verification rounds to 3 for
the 70k split and 5 for the difficult 11k split. We enable asynchronous
rollouts for SVR-R1’s multi-turn training and use a dynamic batch size
for higher efficiency. For the reward verifier, we follow the authors’
implementation and use mathruler , assigning 0.1 to reward for format
(including thinking steps) and 0.9 for the outcome.

### B.3 Supplementary Findings

SVR-R1 fails to outperform when trained on the 11K difficult split. As
mentioned in the paper, repeatedly rethinking questions that are too
difficult contributes little to training, as the correct answer may
remain unreachable even with unlimited self-verification turns. When
training on the 11K difficult split selected by MCTS, we observe no
improvement in reasoning accuracy, while the number of verification
turns grows dramatically during RL training
([Figure 14](#A2.F14 "In B.2 Experiment ‣ Appendix B ThinkLite-VL ‣ SVR-R1: Bootstrapping Multi-modal Reasoning with Self-verification in Reinforcement Learning")).
This contrasts with the 70K dataset, where we observe decreasing turns
and better reasoning performance during training, consistent with the
results and findings shown in the table and chart experiments.

### B.4 Implementation Details

We train each model using the AdamW optimizer ([Loshchilov and Hutter,
2019](#bib.bib50)) with an initial learning rate of $`1\times 10^{-6}`$.
Given the large image input size and long text prompts (up to 16k
tokens), we employ a micro-batch size of 2 per GPU and mini-batch size
of 256 (for chart) or 128 (for table) for one update. The 3B and 7B
model are trained on 8×8 A100 GPUs with 80GB memory, We use bf16
precision and set the decoding temperature to 1.0 during rollouts for
exploration. We set the rollout group size to 16. We enable
KL-divergence term in loss and set the co-efficient $`\beta`$ to be
$`1\times 10^{-3}`$. We manually set the maximum self-verification round
to 3. We enable asynchronous rollouts for SVR-R1’s multi-turn training
as SVR-R1’s multiple self-verification loops within each rollout may
lead to inefficient GPU utilization if processed strictly synchronously.
For additional experiments on ThinkLite, we strictly follow the setup
from ([Wang et al., 2025b](#bib.bib42)), with full details provided in
the supplementary materials.

### B.5 Computation

Limited Computational Overhead. Generation and verification reuse the
same model parameters, so modern frameworks do not need to move weights
to or from the GPU, resulting in minimal additional computational
overhead, only 10% wall clock time if properly set up.

![Refer to caption](2607.10966v1/figures/3b_table_turns_compare.png)

Figure 15: SVR-R1 surpasses standard GRPO on table tasks. Average Turns
vs. Training Steps.
````
