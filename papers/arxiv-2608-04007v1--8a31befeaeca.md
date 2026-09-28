---
identifier: arxiv:2608.04007v1
title: "TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"
authors:
  - Changle Qu
  - Sunhao Dai
  - Hengyi Cai
  - Yuqi Zhou
  - Xinran Chen
  - Simon
  - Jun Xu
published: "2026-08-04T17:59:21+00:00"
url: https://arxiv.org/abs/2608.04007v1
source: arxiv
doi: null
arxiv_id: 2608.04007v1
categories:
  - cs.AI
  - cs.CL
---

# TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning

Changle Qu    Sunhao Dai    Hengyi Cai    Yuqi Zhou    Xinran Chen   
Simon    Jun Xu

###### Abstract

Tool-Integrated Reasoning (TIR) enables LLMs to solve complex tasks
through iterative tool interactions. However, existing reinforcement
learning methods often rely on trajectory-level supervision, limiting
fine-grained credit assignment in long-horizon TIR scenarios. On-policy
self-distillation offers denser signals through teacher branches with
privileged context, but existing approaches typically derive such
context from ground-truth answers or retrieved skills, which may not
reflect the states actually visited by the agent. Moreover, token-level
supervision fails to capture the turn-level structure of tool
interactions. To address this, we propose TurnSight, a turn-level
hindsight self-distillation framework that derives supervision directly
from execution-conditioned hindsight. It then constructs multiple
hindsight views with different lookahead horizons and selects reliable
supervision through cross-horizon directional agreement. Finally, the
selected hindsight signal is normalized across sibling rollouts and used
to adaptively modulate RL advantages while preserving their original
optimization direction. Extensive experiments on three benchmarks
demonstrate the effectiveness of TurnSight. Our codes are available
at https://github.com/quchangle1/TurnSight.

¹Gaoling School of Artificial Intelligence, Renmin University of China
²Baidu Inc.

{changlequ,junxu}@ruc.edu.cn

## Introduction

Tool-Integrated Reasoning (TIR) enables large language models (LLMs) to
solve complex tasks by interleaving reasoning with external tool
interactions ([Gou et al. 2024](#bib.bib19); [Qu et al.
2025a](#bib.bib29); [Qu et al. 2025b](#bib.bib20)). Unlike single-step
generation, a TIR agent repeatedly selects tools, constructs arguments,
observes execution results, and adapts its subsequent behavior. These
interactions substantially expand the capabilities of LLMs, but also
create a difficult temporal credit-assignment problem: the final task
outcome reveals whether a trajectory succeeded, yet provides little
information about which intermediate tool interactions are useful,
redundant, or harmful.

Reinforcement learning with verifiable rewards (RLVR) provides a natural
framework for training TIR agents ([Chang et al. 2026](#bib.bib30);
[Jiang et al. 2026](#bib.bib31)). However, as shown in
Figure [1](#Sx1.F1 "Figure 1 ‣ Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning")(a),
outcome-based methods such as GRPO typically propagate a single
trajectory-level advantage to all generated actions, assigning similar
credit to decisions with very different causal effects ([Shao et al.
2024](#bib.bib1); [Zeng et al. 2025](#bib.bib27)). Recent methods
provide denser supervision by matching generated tool calls against
annotated or reference tool trajectories ([Qian et al.
2025](#bib.bib22); [Qu et al. 2026](#bib.bib15)). Although effective,
such supervision assumes that the reference action remains appropriate
at the state visited by the learned policy. This assumption becomes
fragile in multi-turn interaction: once the agent makes a different tool
call, the environment changes, and subsequent states may diverge from
the reference trajectory. Thus, effective credit assignment for TIR
requires supervision that is both fine-grained and state-aligned.

Figure 1: Comparison of credit assignment strategies for TIR. (a)
Trajectory-level RL assigns identical credit to all turns. (b) Existing
OPSD may use state-misaligned privileged context and produce conflicting
token-level signals. (c) TurnSight derives coherent turn-level credit
from execution-conditioned hindsight across multiple lookahead depths.

On-policy self-distillation (OPSD) ([Shenfeld et al. 2026](#bib.bib3);
[Li et al. 2026b](#bib.bib6); [Kim et al. 2026](#bib.bib7)) offers a
promising way to obtain such supervision. It evaluates student-generated
behavior using a privileged view available only during training, thereby
providing feedback on on-policy states rather than prescribing an
external trajectory. However, applying OPSD to multi-turn TIR is
nevertheless non-trivial. As shown in
Figure [1](#Sx1.F1 "Figure 1 ‣ Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning")(b),
existing methods often enrich the teacher with globally informative
context, such as ground-truth answers, reference rollouts, or distilled
skills ([Zhao et al. 2026](#bib.bib4); [Hübotter et al.
2026](#bib.bib2); [Lu et al. 2026](#bib.bib9); [Ye et al.
2026b](#bib.bib5)). These signals describe how to solve the task, but
not whether a tool interaction was appropriate at the state where it
occurred. This mismatch is especially pronounced in TIR, where each tool
call changes the external environment and may cause subsequent states to
diverge from the reference path. In this setting, the trajectory itself
contains the most state-aligned supervision, with tool execution
outcomes serving as hindsight signals.

State alignment alone is insufficient, however. Existing
self-distillation methods generally produce token-level signals, whereas
the fundamental decision unit in TIR is a complete tool interaction,
comprising reasoning, tool selection, and argument construction. Within
a turn, token-level supervision can vary in magnitude or even sign due
to formatting tokens and argument-level variations. Optimizing these
signals independently can therefore assign conflicting credit within the
same interaction. This motivates our turn-level formulation, which
aggregates hindsight supervision at the interaction level to provide
more coherent credit assignment.

In this paper, we propose TurnSight, a turn-level hindsight
self-distillation method for multi-turn TIR. Instead of relying on
external reference trajectories, TurnSight constructs hindsight
supervision directly from on-policy tool execution outcomes. To better
match the decision granularity of multi-turn TIR, it first aggregates
token-level hindsight evidence into coherent turn-level signals. It then
constructs multiple hindsight views with different lookahead horizons
and selects reliable supervision through cross-horizon directional
agreement. Finally, the selected hindsight signal is normalized across
sibling rollouts and used to construct bounded, sign-aware weights that
modulate RL advantages without changing their optimization direction. As
a result, TurnSight introduces no auxiliary imitation objective, is
independent of the underlying reward construction, and can be seamlessly
integrated with existing policy-gradient RL algorithms.

Extensive experiments on both in-domain and out-of-domain benchmarks
verify the effectiveness of TurnSight and its ability to generalize
across diverse tool-use scenarios. In summary, our main contributions
are as follows:

$`\bullet`$ We formulate two requirements for credit assignment in
multi-turn TIR: supervision should be aligned with on-policy execution
states and coherent at the interaction level.

$`\bullet`$ We propose TurnSight, which derives execution-conditioned,
multi-horizon hindsight assessments to modulate RL advantages without
changing their direction.

$`\bullet`$ Extensive experiments on both in-domain and out-of-domain
benchmarks demonstrate the effectiveness, robustness, and generalization
ability of TurnSight.

## Related Work

### Tool-Integrated Reasoning

Recent RLVR methods, particularly GRPO-style algorithms ([Shao et al.
2024](#bib.bib1); [Yu et al. 2025](#bib.bib35)), have emerged as an
effective paradigm for improving TIR by directly optimizing models with
task-level feedback ([Wang et al. 2025](#bib.bib24); [Xue et al.
2026](#bib.bib26); [Qu et al. 2025b](#bib.bib20)). Early approaches
typically rely on sparse outcome rewards ([Li et al. 2025](#bib.bib21);
[Feng et al. 2025](#bib.bib23)) or trajectory-level supervision ([Qian
et al. 2025](#bib.bib22); [Zhang et al. 2025](#bib.bib25); [Zeng et al.
2025](#bib.bib27); [Wei et al. 2025](#bib.bib28)), where all reasoning
turns within a trajectory receive identical advantage signals, limiting
fine-grained credit assignment. Recent efforts improve this by
introducing turn-level supervision through LLM-based evaluation ([Wang
et al. 2026a](#bib.bib32)) or by comparing predicted tool calls with
ground-truth tool executions ([Qu et al. 2026](#bib.bib15)). However,
these methods rely on annotated or reference tool trajectories, making
optimization closer to imitation learning and limiting exploration. In
contrast, TurnSight combines RL with on-policy self-distillation to
achieve fine-grained credit assignment without ground-truth trajectories
while preserving exploration.

### On-Policy Self-Distillation

On-policy self-distillation evaluates student-generated trajectories
using a privileged teacher view constructed from the same model with
additional information, providing dense supervision on the states
visited by the student ([Sang et al. 2026](#bib.bib37); [Ma et al.
2026](#bib.bib10); [Li et al. 2026a](#bib.bib38); [Wang et al.
2026b](#bib.bib39)). Rather than directly matching the teacher
distribution, recent methods use teacher–student disagreement to
reweight RL advantages, allocate fine-grained credit, or bound
distillation interventions ([Yang et al. 2026](#bib.bib8); [Zhang et al.
2026](#bib.bib11); [Ding et al. 2026](#bib.bib12); [Zhou et al.
2026b](#bib.bib14); [Zhou et al. 2026a](#bib.bib13)). Despite these
advances, existing methods often rely on globally informative teacher
signals and token-level distillation, which may not provide
state-aligned supervision at the natural turn-level granularity of TIR.
In contrast, we propose TurnSight, which constructs turn-level hindsight
supervision directly from on-policy tool execution outcomes, enabling
fine-grained and state-aligned credit assignment.

Figure 2: The illustration of our proposed turn-level hindsight
self-distillation framework TurnSight for TIR.

## Preliminaries

In this section, we first formulate the task of TIR, followed by the
agentic RL framework and the OPSD mechanism.

### Task Formulation

Given a user prompt $`q\sim\mathcal{D}`$ and a set of $`M`$ tools
$`\mathcal{T}=\{t_{1},\ldots,t_{M}\}`$, where each tool is specified by
its name and argument schema, an agent with policy $`\pi_{\theta}`$
interacts with an environment to produce a trajectory
$`\tau=(s_{1},\ldots,s_{K})`$. We represent a non-terminal turn as
$`s_{k}=(n_{k},\mathcal{C}_{k},o_{k})`$, where $`n_{k}`$ is the
natural-language reasoning and $`\mathcal{C}_{k}`$ is the set of tool
calls issued in that turn. Each call contains a tool name and a JSON
argument object. A trajectory terminates when the policy emits its final
answer or reaches the pre-defined maximum turn limit $`L`$. For the
terminal turn, $`\mathcal{C}_{K}=\emptyset`$ and $`o_{K}=\emptyset`$.
Our goal is to achieve fine-grained turn-level credit assignment by
leveraging execution-aware supervision.

### Agentic Reinforcement Learning

We optimize the agent policy $`\pi_{\theta}`$ using GRPO ([Shao et al.
2024](#bib.bib1)). For each prompt $`q\sim\mathcal{D}`$, the behavior
policy $`\pi_{\theta_{\mathrm{old}}}`$ samples a group of $`G`$
interaction trajectories $`\{\tau_{i}\}_{i=1}^{G}`$. Rather than
training an additional critic to predict state values, GRPO constructs
policy-gradient advantages by comparing the rewards of trajectories
generated for the same prompt. This group-wise comparison provides a
prompt-dependent baseline and reduces the cost of value-function
estimation. The optimization objective is written as

|     |     |     |
| --- | --- | --- |
|     |

````math
\begin{split}&\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{q\sim D,\{\tau_{i}\}\sim\pi_{\theta_{\text{old}}}(\cdot|q)}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_{i}|}\sum_{t=1}^{|\tau_{i}|}\bigg[\min(\\
&\rho_{i,t}\hat{A}_{i,t},\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t})-\beta\mathbb{D}_{\text{KL}}(\pi_{\theta}|\pi_{\text{ref}})\bigg],\end{split}
``` |  |

where
$`\rho_{i,t}=\frac{\pi_{\theta}(\tau_{i,t}|q,\tau_{i,<t})}{\pi_{\theta_{\text{old}}}(\tau_{i,t}|q,\tau_{i,<t})}`$
is the importance sampling ratio, $`\widehat{A}_{i,t}`$ is the
normalized group-relative advantage, $`\epsilon`$ is the clipping
parameter, and $`\beta`$ controls the KL regularization toward the
reference policy $`\pi_{\mathrm{ref}}`$. Since trajectories contain both
model-generated actions and environment observations, only tokens
generated by the policy are optimized, while observation tokens are
excluded from gradient computation.

### On-Policy Self-Distillation

OPSD scores a trajectory sampled by the student under two information
conditions. At token position $`t`$, the student distribution
$`p_{t}^{S}=\pi_{\theta}(\cdot\mid c_{t})`$ uses only rollout
information, while the detached teacher distribution
$`p_{t}^{T}=\pi_{T}(\cdot\mid c_{t}^{+})`$ additionally accesses
privileged training-time information. For a sampled token $`y_{t}`$, we
define the log-probability gap as

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\delta_{t}=\log\pi_{T}(y_{t}\mid c_{t}^{+})-\log\pi_{\theta}(y_{t}\mid c_{t}).
``` |  | (1) |

The gap $`\delta_{t}`$ captures how privileged information changes
confidence in the realized token. Unlike direct distribution matching,
which lets the teacher determine the update direction, our method uses
$`\delta_{t}`$ as hindsight evidence that modulates RL advantage
magnitudes while preserving their directions.

## Our Approach: TurnSight

In this section, we will provide a detailed introduction to our
TurnSight. The overall framework is illustrated in
Figure [2](#Sx2.F2 "Figure 2 ‣ On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").

### Execution-Conditioned Hindsight Construction

Execution-Conditioned Privileged Context. In multi-turn TIR, each tool
execution changes the environment and affects subsequent states.
Existing privileged context sources, such as ground-truth answers and
reference rollouts, are not conditioned on the student’s trajectory and
may provide misaligned supervision. We therefore construct privileged
context directly from the execution outcomes generated by the student’s
own trajectory, which are naturally aligned with its visited states and
provide informative hindsight for subsequent policy optimization.
Specifically, consider a token $`y_{i,t}`$ in turn $`k`$ of trajectory
$`\tau_{i}`$. The student scores this token under its original causal
context $`c_{i,t}`$, which contains only the information available
during rollout. For a lookahead depth $`d`$, we collect the execution
hindsight block

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{H}_{i,k}^{(d)}=\Big((\mathcal{C}_{i,k},o_{i,k}),\ldots,(\mathcal{C}_{i,\bar{k}},o_{i,\bar{k}})\Big),
``` |  |

where $`\mathcal{C}_{i,j}`$ denotes the tool calls issued by the student
in turn $`j`$, $`o_{i,j}`$ denotes the corresponding tool responses, and
$`\bar{k}=\min(k+d-1,K_{i})`$. So the privileged context is

|     |                                                      |     |
|-----|------------------------------------------------------|-----|
|     |
       ``` math
       c_{i,t}^{+(d)}=c_{i,t}\oplus\mathcal{H}_{i,k}^{(d)},
       ```                                                   |     |

where the hindsight block is used only during training while the
reference branch teacher-forces the original student response. We
instantiate this branch with a frozen initial reference policy
$`\pi_{\mathrm{ref}}`$. For each lookahead depth
$`d\in\mathcal{D}_{H}`$, we compute the detached sampled-token gap

|  |  |  |
|----|----|----|
|  |
``` math
\delta_{i,t}^{(d)}=\log\pi_{\mathrm{ref}}(y_{i,t}\mid c_{i,t}^{+(d)})-\log\pi_{\theta}(y_{i,t}\mid c_{i,t}).
``` |  |

A positive gap indicates that observing the execution consequences
increases the reference branch’s confidence in the realized token,
whereas a negative gap indicates the opposite.

Turn-Level Aggregation. Although hindsight gaps are computed at the
token level, the natural decision unit in multi-turn TIR is an
interaction turn, where reasoning, tool selection, and argument
generation together constitute one tool-use decision. In practice,
token-level hindsight signals within the same turn often vary
substantially or even disagree due to formatting tokens and argument
variations. Applying these signals independently may therefore produce
inconsistent credit assignment. To preserve the semantic coherence of
each interaction, we aggregate token-level hindsight into a turn-level
signal. Let $`\mathcal{Y}_{i,k}=\{t:\kappa_{i}(t)=k\}`$ denote the token
positions belonging to turn $`k`$. For each lookahead depth, we compute

|  |  |  |
|----|----|----|
|  |
``` math
\bar{\delta}_{i,k}^{(d)}=\frac{1}{|\mathcal{Y}_{i,k}|}\sum_{t\in\mathcal{Y}_{i,k}}\delta_{i,t}^{(d)}.
``` |  |

The resulting $`\bar{\delta}_{i,k}^{(d)}`$ provides a single hindsight
assessment for the entire interaction turn, which is subsequently shared
by all policy tokens in that turn and serves as the basic evidence unit
for multi-lookahead teacher selection.

### Multi-Lookahead Teacher Selection

Multi-Lookahead Hindsight Views. The amount of future information
required to evaluate a tool-use decision varies across interaction
states in multi-turn TIR. Some decisions, such as validating tool
selection or argument correctness, can be assessed immediately from
execution feedback, while others require subsequent tool interactions to
reveal their utility. Consequently, no single hindsight horizon is
suitable for every interaction state: short horizons may overlook
delayed effects, while long horizons may introduce unrelated future
interactions and dilute the evidence for the current decision.

To capture these diverse temporal dependencies, we construct multiple
execution-conditioned hindsight contexts with lookahead depths
$`\mathcal{D}_{H}=\{1,2,3\}`$. Each lookahead depth instantiates an
independent hindsight teacher by exposing the frozen reference policy to
different amounts of future execution information, yielding a turn-level
hindsight signal $`\bar{\delta}_{i,k}^{(d)}`$. Collectively, these
teachers provide complementary assessments of the same tool-use
decision, with shallow horizons emphasizing immediate execution quality
and deeper ones reflecting downstream trajectory contributions.

Direction-Consistent Teacher Selection. Since teachers observe different
amounts of future information, their hindsight assessments may disagree
for the same interaction turn. Treating all teachers equally may weaken
informative supervision, while selecting the strongest teacher alone may
amplify an isolated and unreliable assessment. We therefore perform
teacher selection in two stages: we first determine the reliable
supervision direction through cross-horizon agreement and then select
the strongest teacher within that direction. For each turn, we first map
teacher signal to its direction

|     |                                                                  |     |
|-----|------------------------------------------------------------------|-----|
|     |
       ``` math
       s_{i,k}^{(d)}=\begin{cases}+1,&\bar{\delta}_{i,k}^{(d)}\geq 0,\\
       -1,&\bar{\delta}_{i,k}^{(d)}<0.\end{cases}
       ```                                                               |     |

We then determine the consensus direction by majority voting over
$`\{s_{i,k}^{(d)}\}_{d\in\mathcal{D}_{H}}`$ and select the teacher with
the largest hindsight signal among those consistent with the consensus:

|  |  |  |
|----|----|----|
|  |
``` math
d_{i,k}^{*}=\arg\max_{\begin{subarray}{c}d\in\mathcal{D}_{H}:\\
s_{i,k}^{(d)}=v_{i,k}\end{subarray}}\left|\bar{\delta}_{i,k}^{(d)}\right|,\qquad\bar{\delta}_{i,k}^{*}=\bar{\delta}_{i,k}^{(d_{i,k}^{*})},
``` |  |

where $`v_{i,k}`$ denotes the majority direction.

Directional agreement filters out isolated teachers whose assessments
conflict with the majority of hindsight views, while selecting the
strongest consistent teacher preserves the most informative supervision
without diluting it through indiscriminate fusion. The
$`\bar{\delta}_{i,k}^{*}`$ is used as the final hindsight signal and is
subsequently passed to group normalization.

### Group-Normalized Weight Construction

Group-Relative Evidence Normalization. Although the selected hindsight
signal $`\bar{\delta}_{i,k}^{*}`$ provides a turn-level assessment of
each tool interaction, its magnitude is not directly comparable across
trajectories. The teacher–student gap is affected by factors such as
prompt difficulty, trajectory length, turn position, and training
progress. Consequently, directly using the raw gap may overemphasize
naturally large-scale examples instead of reflecting their relative
hindsight evidence.

Inspired by GRPO, we normalize the selected hindsight evidence within
each prompt group. Instead of grouping interactions by turn index, we
compute a shared mean $`\mu_{q_{i}}^{\delta}`$ and standard deviation
$`\sigma_{q_{i}}^{\delta}`$ over the selected token-level gaps from all
valid turns and policy tokens across all sibling rollouts sampled for
query $`q_{i}`$. The normalized signal is

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\delta}_{i,k}=\frac{\bar{\delta}_{i,k}^{*}-\mu_{q_{i}}^{\delta}}{\sigma_{q_{i}}^{\delta}+\epsilon}.
``` |  |

The normalized signal measures the relative hindsight evidence of the
current turn against all valid interactions across sibling on-policy
rollouts for the same query, reducing prompt-specific scale variation
and making the signal more comparable across queries and throughout
training.

Bounded Sign-Aware Weighting. The normalized hindsight signal is used to
modulate the RL credit assigned by the base advantage. We therefore
construct a bounded weight

|  |  |  |
|----|----|----|
|  |
``` math
w_{i,t}=1+\epsilon_{w}\tanh\!\left(\mathrm{sign}\!\left(A_{i,t}^{\mathrm{base}}\right)\widehat{\delta}_{i,k}\right).
``` |  |

The sign term aligns hindsight evidence with the update direction
determined by the base advantage. Consequently, agreement between
hindsight and the RL signal increases the update magnitude
($`w_{i,t}>1`$), whereas disagreement attenuates it ($`w_{i,t}<1`$). To
mitigate the influence of extreme hindsight signals, we further
introduce a bounded modulation mechanism, where $`\epsilon_{w}`$
controls the adjustment range:

|     |                                              |     |
|-----|----------------------------------------------|-----|
|     |
       ``` math
       w_{i,t}\in[1-\epsilon_{w},\,1+\epsilon_{w}].
       ```                                           |     |

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Methods | FTRL |  |  |  | BFCL |  |  |  |  |  |  | ToolHop | Avg. |
|  | Solve-P | Solve-R | Solve-F1 | Avg. | Multi-Turn |  |  |  | Agentic |  | Avg. | Acc. |  |
|  |  |  |  |  | Base | MF | MP | LC | Search | Memory |  |  |  |
| Qwen3-4B |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Vanilla | 30.78 | 29.65 | 25.85 | 28.76 | 37.00 | 36.00 | 31.50 | 22.50 | 9.00 | 19.14 | 22.91 | 23.22 | 24.96 |
| GRPO | 31.61 | 32.64 | 28.49 | 30.91 | 42.00 | 34.00 | 29.00 | 23.50 | 8.00 | 19.35 | 22.90 | 21.98 | 25.26 |
| ToolRL | 34.07 | 31.63 | 27.94 | 31.21 | 43.00 | 41.00 | 27.50 | 25.50 | 8.50 | 24.09 | 25.27 | 23.65 | 26.71 |
| MatchTIR | 37.77 | 36.44 | 33.51 | 35.91 | 51.00 | 44.00 | 30.50 | 38.00 | 18.50 | 23.01 | 30.82 | 37.56 | 34.76 |
| SDPO | 31.35 | 32.66 | 27.71 | 30.57 | 43.50 | 41.50 | 31.00 | 29.50 | 8.00 | 22.15 | 25.73 | 26.53 | 27.61 |
| RLSD | 34.19 | 32.63 | 31.13 | 32.65 | 45.50 | 39.00 | 33.50 | 31.00 | 12.00 | 21.93 | 27.11 | 33.30 | 31.02 |
| SDAR | 33.67 | 37.03 | 31.95 | 34.22 | 47.00 | 40.50 | 30.50 | 37.00 | 16.00 | 22.80 | 29.07 | 37.36 | 33.55 |
| SOD | 31.89 | 34.91 | 30.66 | 32.49 | 38.50 | 34.00 | 28.00 | 24.50 | 9.50 | 21.72 | 23.43 | 22.28 | 26.06 |
| TurnSight (Ours) | 44.61 | 40.16 | 38.44 | 41.00 | 50.50 | 46.50 | 36.00 | 41.00 | 21.00 | 24.73 | 33.18 | 38.29 | 37.51 |
| Qwen3-8B |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Vanilla | 28.08 | 36.55 | 29.74 | 31.46 | 46.50 | 49.50 | 38.50 | 36.00 | 10.50 | 21.29 | 29.26 | 34.07 | 31.60 |
| GRPO | 35.97 | 39.78 | 35.02 | 36.92 | 45.50 | 43.00 | 31.50 | 32.00 | 8.50 | 22.15 | 26.66 | 29.21 | 30.93 |
| ToolRL | 35.22 | 43.72 | 37.53 | 38.82 | 56.00 | 49.00 | 33.00 | 35.50 | 17.00 | 20.64 | 31.10 | 35.68 | 35.20 |
| MatchTIR | 40.88 | 46.64 | 40.83 | 42.78 | 57.00 | 49.00 | 37.00 | 39.50 | 20.00 | 23.87 | 33.78 | 40.54 | 39.03 |
| SDPO | 38.13 | 42.72 | 36.20 | 39.02 | 43.00 | 41.00 | 33.50 | 32.00 | 9.00 | 20.00 | 25.94 | 35.21 | 33.39 |
| RLSD | 36.92 | 40.53 | 36.98 | 38.14 | 46.50 | 45.50 | 33.00 | 36.00 | 16.00 | 26.02 | 30.63 | 38.29 | 35.69 |
| SDAR | 35.02 | 44.23 | 36.72 | 38.66 | 54.00 | 51.00 | 38.50 | 38.50 | 20.00 | 25.38 | 34.09 | 41.37 | 38.04 |
| SOD | 30.08 | 40.92 | 32.58 | 34.53 | 50.00 | 44.50 | 34.50 | 35.00 | 13.00 | 19.14 | 28.54 | 33.64 | 32.23 |
| TurnSight (Ours) | 46.99 | 50.71 | 43.07 | 46.92 | 59.50 | 51.50 | 42.00 | 45.00 | 23.50 | 27.74 | 37.56 | 41.58 | 42.02 |

Table 1: Main results of TurnSight and baseline methods across three
benchmarks using two LLM backbones. For BFCL, MF, MP, and LC denote the
Miss Function, Miss Parameter, and Long Context subsets, respectively.
The best and second-best results in each column are shown in bold and
underlined, respectively.

### Policy Optimization

Advantage Integration. To incorporate turn-level hindsight while
preserving the original RL objective, we interpolate the base advantage
with its hindsight-modulated counterpart. For a token at position $`t`$
belonging to turn $`k`$ within the $`i`$-th rollout, the integrated
advantage is defined as

|  |  |  |
|----|----|----|
|  |
``` math
\widetilde{A}_{i,t}=A_{i,t}^{\mathrm{base}}\left[(1-\lambda)+\lambda w_{i,t}\right],
``` |  |

where $`\lambda`$ controls the strength of hindsight modulation.

Objective Function. We retain the standard GRPO objective and replace
the original advantage with $`\widetilde{A}_{i,t}`$:

|  |  |  |
|----|----|----|
|  |
``` math
\begin{split}\mathcal{J}_{\text{TurnSight}}(\theta)&=\mathbb{E}_{q\sim D,\{\tau_{i}\}\sim\pi_{\theta_{\text{old}}}(\cdot|q)}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_{i}|}\sum_{t=1}^{|\tau_{i}|}\bigg[\min(\\
&\rho_{i,t}\widetilde{A}_{i,t},\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\widetilde{A}_{i,t})\bigg].\end{split}
``` |  |

## Experiments

In this section, we first describe the experimental setups and then
conduct an extensive evaluation of our TurnSight.

### Experimental Setups

Training Data. We use the FTRL dataset ([Ye et al. 2026a](#bib.bib33))
for all post-training experiments. It provides more than 2,000
automatically generated problems paired with executable tool
environments and programmatically verifiable feedback. The dataset
covers single-step tool calls, sequential reasoning over multiple tools,
and parallel tool invocations.

Evaluation Benchmarks. We evaluate TurnSight on three benchmarks
covering both in-domain and out-of-domain settings. FTRL serves as the
in-domain benchmark, whereas BFCL ([Patil et al. 2025](#bib.bib34)) and
ToolHop ([Ye et al. 2025](#bib.bib16)) are used for out-of-domain
evaluation. Together, these benchmarks assess the model’s ability to
generalize beyond the task distributions and patterns encountered during
training.

Baselines. To evaluate the effectiveness of TurnSight, we compare it
with two families of baselines. The first includes RL methods for TIR:
Vanilla ([Yang et al. 2025](#bib.bib17)), GRPO ([Shao et al.
2024](#bib.bib1)), ToolRL ([Qian et al. 2025](#bib.bib22)), and
MatchTIR ([Qu et al. 2026](#bib.bib15)), which provide supervision at
the outcome, trajectory, and turn levels, respectively. The second
comprises representative self-distillation methods, including
SDPO ([Hübotter et al. 2026](#bib.bib2)), RLSD ([Yang et al.
2026](#bib.bib8)), SDAR ([Lu et al. 2026](#bib.bib9)), and SOD ([Zhong
et al. 2026](#bib.bib36)), which leverage privileged contexts to guide
RL optimization.

Implementation Details. We conduct experiments with Qwen3-4B and
Qwen3-8B as the policy backbones ([Yang et al. 2025](#bib.bib17)). All
experiments are implemented with the verl framework ([Sheng et al.
2025](#bib.bib18)) and initialized directly from the corresponding base
checkpoints, without an intermediate supervised fine-tuning stage. Each
optimization batch contains 32 queries, with 16 trajectories sampled per
query and at most 10 interaction turns per trajectory. We do not apply
an explicit KL-divergence regularizer to constrain the policy updates
and train for 3 epochs on 8 NVIDIA A800 GPUs with 80 GB memory each. For
TurnSight, both the mixing coefficient $`\lambda`$ and the modulation
bound $`\epsilon_{w}`$ are set to $`0.5`$. We construct 3
execution-conditioned teacher views using 3 lookahead depths. All
teacher and reference branches are initialized from the corresponding
policy backbone and remain frozen throughout RL training. Since RLSD and
TurnSight are agnostic to the underlying RL algorithm and can be
combined with different policy-optimization methods, we use MatchTIR as
the RL backbone for both.

### Experimental Results

Table [1](#Sx4.T1 "Table 1 ‣ Group-Normalized Weight Construction ‣ Our Approach: TurnSight ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning")
reports the results on three benchmarks, from which we make the
following observations:

$`\bullet`$ Firstly, RL consistently improves tool-use performance,
while outcome-only optimization generalizes poorly. Compared with the
vanilla Qwen3 models, all RL post-training methods achieve clear gains
on the in-domain FTRL benchmark, demonstrating the effectiveness of RL
for optimizing multi-turn tool-use behaviors. However, GRPO often
exhibits noticeably weaker performance on the out-of-domain BFCL and
ToolHop benchmarks, indicating that outcome-level rewards are
insufficient for identifying which intermediate tool interactions should
receive credit.

$`\bullet`$ Secondly, fine-grained credit assignment consistently leads
to stronger and more transferable policies. Compared with vanilla GRPO,
methods equipped with structured tool-level supervision achieve
consistently better performance across both model scales, confirming the
importance of fine-grained credit signals for long-horizon TIR. The
strong performance of MatchTIR further shows that accurate turn-level
supervision is highly effective, making it a strong baseline for
evaluating on-policy hindsight methods.

$`\bullet`$ Thirdly, simply combining on-policy self-distillation with
RL is insufficient for effective TIR. Although SDPO, RLSD, and SDAR
consistently outperform vanilla GRPO, they still lag behind methods with
explicit turn-level credit assignment. This suggests that privileged
information from ground-truth trajectories or successful rollouts cannot
directly provide accurate supervision for every intermediate decision,
especially in long-horizon tool interactions. Moreover, token-level
self-distillation may introduce conflicting credit signals within the
same interaction turn. These results highlight the importance of
aggregating hindsight supervision at the turn level to provide reliable
credit assignment.

$`\bullet`$ Finally, TurnSight achieves the best overall performance
while remaining fully on-policy. Across the three benchmarks, TurnSight
consistently outperforms all baselines and establishes new
state-of-the-art results on the 8B model, improving the previous best
method by 7.7% in overall average performance. Notably, the largest
improvements are observed on challenging BFCL subsets such as Long
Context and Miss Parameter, where successful tool use requires
accurately attributing credit across multiple interaction turns. These
results demonstrate that TurnSight effectively combines the exploration
benefits of on-policy RL with fine-grained hindsight supervision,
enabling more accurate credit assignment without relying on reference
trajectories.

| Variant                       | Solve-P | Solve-R | Solve-F1 | Avg.  |
|-------------------------------|---------|---------|----------|-------|
| TurnSight                     | 46.99   | 50.71   | 43.07    | 46.92 |
| *w/o* turn-level aggregation  | 41.61   | 47.53   | 40.54    | 43.23 |
| *w/o* group normalization     | 42.84   | 48.02   | 40.09    | 43.65 |
| *w/o* multi-teacher selection | 44.26   | 50.24   | 42.35    | 45.62 |

Table 2: Ablation study of the core components.

### Ablation Study

To quantify the contribution of each component, we conduct an ablation
study on Qwen3-8B, with results reported in
Table [2](#Sx5.T2 "Table 2 ‣ Experimental Results ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
Removing any component consistently degrades performance, confirming the
significance of each element:

![Refer to caption](2608.04007v1/context-composition.png)

Figure 3: Effect of hindsight context composition. Darker cells indicate
stronger performance.

Figure 4: Comparison of fixed-lookahead teachers, teacher fusion, and
our direction-consistent teacher selection.

*w/o* turn-level aggregation refers to a variant that removes turn-level
aggregation and instead selects the best execution-conditioned hindsight
view for each token independently, using token-level signals to modulate
the RL advantage. The significant performance drop shows that multi-turn
TIR is naturally organized around interaction turns rather than
individual tokens. Applying supervision independently at the token level
can introduce conflicting credit signals within the same tool
interaction, whereas aggregating hindsight evidence at turn level
produces more reliable credit assignment.

*w/o* group normalization is a variant that removes group normalization
and directly uses the selected turn-level hindsight signal to modulate
the RL advantage. The noticeable degradation demonstrates that the
magnitude of hindsight signals varies substantially across rollout
groups. Group normalization calibrates these signals into a comparable
scale, leading to more stable and reliable credit assignment.

*w/o* multi-teacher selection denotes the variant that replaces the
multi-lookahead teacher with a single execution-conditioned teacher
using one fixed lookahead depth. The performance drop indicates that
different lookahead depths provide complementary hindsight information:
shorter horizons capture the immediate quality of a tool interaction,
while longer horizons reveal delayed consequences that only become
apparent after subsequent reasoning and tool use.

Overall, these results demonstrate that the effectiveness of TurnSight
comes from the coordinated design of turn-level aggregation, group
normalization, and multi-lookahead supervision, which together produce
hindsight signals that are more coherent, comparable, and temporally
informative.

### Further Analysis

Hindsight Context Composition. To evaluate which type of privileged
information provides the most effective hindsight supervision, we
compare three teacher context configurations: tool result only, tool
result + ground-truth answer, and ground-truth answer only, while
keeping all other settings unchanged. As shown in
Figure [3](#Sx5.F3 "Figure 3 ‣ Ablation Study ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
using only the tool result consistently achieves the best performance.
Interestingly, adding the ground-truth answer further degrades
performance, suggesting that more privileged information does not
necessarily provide better supervision. We attribute this to the fact
that tool results provide direct, state-aligned evidence for evaluating
the current tool interaction, whereas the ground-truth answer contains
trajectory-level information that may obscure turn-level causal
contributions. These results highlight execution feedback as the most
effective privileged signal for hindsight-based credit assignment.

Lookahead Teacher Selection. To evaluate the effect of hindsight horizon
and teacher combination strategies, we compare fixed-lookahead teachers
with depths of one, two, and three turns, a teacher-fusion variant, and
our direction-consistent teacher selection strategy. As shown in
Figure [4](#Sx5.F4 "Figure 4 ‣ Ablation Study ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
the one-step teacher performs best among the fixed-horizon variants,
while longer lookahead depths consistently reduce performance,
suggesting that excessive future context introduces irrelevant
interactions that dilute turn-specific supervision. Although teacher
fusion provides a slight improvement over the best fixed teacher, it
cannot effectively resolve conflicting hindsight signals across
different horizons. In contrast, our direction-consistent selection
strategy achieves the best performance on all metrics, outperforming
both the strongest fixed teacher and teacher fusion. These results
demonstrate that multi-horizon hindsight is most effective when
informative future evidence is selectively retained rather than
indiscriminately aggregated.

Figure 5: Effect of temporal self-distillation coverage.

Temporal Coverage of Hindsight. To evaluate how the temporal coverage of
hindsight supervision affects multi-turn TIR, we compare three variants
that apply self-distillation to the first five turns, last five turns,
or all turns of each trajectory. As shown in
Figure [5](#Sx5.F5 "Figure 5 ‣ Further Analysis ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
supervising the entire trajectory consistently achieves the best
performance, outperforming the first-five-turn and last-five-turn
variants by 2.36% and 3.88% in average performance, respectively.
Moreover, supervising the first five turns consistently outperforms the
last five, suggesting that early tool-use decisions have a larger impact
by shaping subsequent interaction states. Nevertheless, the clear
advantage of full-trajectory supervision indicates that reliable credit
assignment requires execution-conditioned hindsight throughout the
entire interaction.

Figure 6: Sensitivity analysis of performance to hyper-parameters. (a)
shows the impact of mixing coefficient $`\lambda`$. (b) illustrates the
effect of modulation bound $`\epsilon_{w}`$.

Hyper-parameter Analysis. To examine the sensitivity of TurnSight, we
independently vary two key hyper-parameters, the mixing coefficient
$`\lambda`$ and modulation bound $`\epsilon_{w}`$, and evaluate their
effects on FTRL. As shown in
Figure [6](#Sx5.F6 "Figure 6 ‣ Further Analysis ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
both exhibit a consistent unimodal trend, achieving the best performance
at $`\lambda=0.5`$ and $`\epsilon_{w}=0.5`$. Increasing $`\lambda`$
initially improves performance, indicating that turn-level hindsight
signals complement trajectory-level advantages. However, an overly large
$`\lambda`$ degrades performance, as excessive emphasis on local credit
may weaken global task-level guidance. This highlights the need to
balance fine-grained interaction supervision with overall outcome
optimization. Similarly, $`\epsilon_{w}`$ determines the strength of
hindsight modulation. Small values restrict the contribution of
execution-conditioned signals, whereas overly large values may amplify
noise in imperfect teacher–student gaps. The optimal intermediate
setting therefore strikes an effective balance between exploiting
hindsight information and maintaining stable RL optimization.

## Conclusion

In this paper, we identify two key requirements for credit assignment in
TIR: alignment with on-policy execution states and coherence at the
interaction level. To address this, we propose TurnSight, a turn-level
hindsight self-distillation framework that derives
execution-conditioned, multi-horizon hindsight assessments from
on-policy tool execution outcomes. By leveraging these assessments to
modulate RL advantages without altering their optimization direction,
TurnSight enables fine-grained and reliable credit assignment while
preserving the exploration capability of RL. Extensive experiments on
both in-domain and out-of-domain benchmarks demonstrate the
effectiveness, robustness, and generalization ability of TurnSight
across diverse multi-turn TIR scenarios.

## References

- Chang et al. (2026) Q. Chang, Z. Zhang, P. Hu, J. Du, J. Ma, Y.
  Pan, J. Zhang, Q. Liu, and J. Gao THOR: tool-integrated hierarchical
  optimization via RL for mathematical reasoning. In The Fourteenth
  International Conference on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=0Af7UiJISU) Cited by:
  [Introduction](#Sx1.p2.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Ding et al. (2026) T. Ding, J. Xin, and J. P. D. l. C. Weinstein Keep
  policy gradient in charge: sibling-guided credit distillation for
  long-horizon tool-use agents. arXiv preprint arXiv:2606.12634. Cited
  by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Feng et al. (2025) J. Feng, S. Huang, X. Qu, G. Zhang, Y. Qin, B.
  Zhong, C. Jiang, J. Chi, and W. Zhong Retool: reinforcement learning
  for strategic tool use in llms. arXiv preprint arXiv:2504.11536. Cited
  by: [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Gou et al. (2024) Z. Gou, Z. Shao, Y. Gong, Y. Yang, M. Huang, N.
  Duan, W. Chen, et al. Tora: a tool-integrated reasoning agent for
  mathematical problem solving. In Proceedings of the 12th International
  Conference on Learning Representations (ICLR). Cited by:
  [Introduction](#Sx1.p1.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Hübotter et al. (2026) J. Hübotter, F. Lübeck, L. D. Behric, A.
  Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K.
  Buening, C. Guestrin, and A. Krause Reinforcement learning via
  self-distillation. In Forty-third International Conference on Machine
  Learning, External Links:
  [Link](https://openreview.net/forum?id=QkfkxyRizZ) Cited by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Jiang et al. (2026) D. Jiang, Y. Lu, Z. Li, Z. Lyu, P. Nie, H.
  Wang, A. Su, H. Chen, K. Zou, C. Du, T. Pang, and W. Chen VerlTool:
  towards holistic agentic reinforcement learning with tool use.
  Transactions on Machine Learning Research. Note: External Links: ISSN
  2835-8856, [Link](https://openreview.net/forum?id=g2LCOW43Md) Cited
  by:
  [Introduction](#Sx1.p2.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Kim et al. (2026) J. Kim, X. Luo, M. Kim, S. Lee, D. Kim, J. Jeon, D.
  Li, and Y. Yang Why does self-distillation (sometimes) degrade the
  reasoning capability of llms?. arXiv preprint arXiv:2603.24472. Cited
  by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Li et al. (2026a) G. Li, T. Yang, J. Fang, M. Song, M. Zheng, H.
  Guo, D. Zhang, J. Wang, and T. Chua Unifying group-relative and
  self-distillation policy optimization via sample routing. arXiv
  preprint arXiv:2604.02288. Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Li et al. (2025) X. Li, H. Zou, and P. Liu Torl: scaling
  tool-integrated rl. arXiv preprint arXiv:2503.23383. Cited by:
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Li et al. (2026b) Y. Li, Y. Zuo, B. He, J. Zhang, C. Xiao, C. Qian, T.
  Yu, H. Gao, W. Yang, Z. Liu, et al. Rethinking on-policy distillation
  of large language models: phenomenology, mechanism, and recipe. arXiv
  preprint arXiv:2604.13016. Cited by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Lu et al. (2026) Z. Lu, Z. Yao, Z. Han, Z. Wang, J. Wu, Q. Gu, X.
  Cai, W. Lu, J. Xiao, Y. Zhuang, and Y. Shen Self-distilled agentic
  reinforcement learning. External Links: 2605.15155 Cited by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Ma et al. (2026) Y. Ma, Z. Liang, B. Chen, Z. Qian, H. Dai, L. Mao, X.
  Zhang, C. Lei, and W. Ou SD-Search: on-policy hindsight
  self-distillation for search-augmented reasoning. External Links:
  2605.18299 Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Patil et al. (2025) S. G. Patil, H. Mao, C. Cheng-Jie Ji, F. Yan, V.
  Suresh, I. Stoica, and J. E. Gonzalez The berkeley function calling
  leaderboard (bfcl): from tool use to agentic evaluation of large
  language models. In Forty-second International Conference on Machine
  Learning, Cited by: [Appendix
  A](#A1.p4.1 "Appendix A Benchmark and Evaluation Details ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p2.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Qian et al. (2025) C. Qian, E. C. Acikgoz, Q. He, H. WANG, X. Chen, D.
  Hakkani-Tür, G. Tur, and H. Ji ToolRL: reward is all tool learning
  needs. In The Thirty-ninth Annual Conference on Neural Information
  Processing Systems, External Links:
  [Link](https://openreview.net/forum?id=eOLdGbXT6t) Cited by:
  [Introduction](#Sx1.p2.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Qu et al. (2026) C. Qu, S. Dai, H. Cai, J. Xu, S. Wang, and D. Yin
  MatchTIR: fine-grained supervision for tool-integrated reasoning via
  bipartite matching. In Proceedings of the 64th Annual Meeting of the
  Association for Computational Linguistics (Volume 1: Long Papers),
  Cited by:
  [Introduction](#Sx1.p2.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Qu et al. (2025a) C. Qu, S. Dai, X. Wei, H. Cai, S. Wang, D. Yin, J.
  Xu, and J. Wen From exploration to mastery: enabling LLMs to master
  tools via self-driven interactions. In The Thirteenth International
  Conference on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=QKBu1BOAwd) Cited by:
  [Introduction](#Sx1.p1.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Qu et al. (2025b) C. Qu, S. Dai, X. Wei, H. Cai, S. Wang, D. Yin, J.
  Xu, and J. Wen Tool learning with large language models: a survey.
  Frontiers of Computer Science 19 (8), pp. 198343. Cited by:
  [Introduction](#Sx1.p1.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Sang et al. (2026) H. Sang, Y. Xu, Z. Zhou, R. He, Z. Wang, and J. Sun
  Crisp: compressed reasoning via iterative self-policy distillation.
  arXiv preprint arXiv:2603.05433. Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Shao et al. (2024) Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H.
  Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo DeepSeekMath: pushing the
  limits of mathematical reasoning in open language models. External
  Links: 2402.03300 Cited by:
  [Introduction](#Sx1.p2.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Agentic Reinforcement
  Learning](#Sx3.SSx2.p1.1 "Agentic Reinforcement Learning ‣ Preliminaries ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Shenfeld et al. (2026) I. Shenfeld, M. Damani, J. Hübotter, and P.
  Agrawal Self-distillation enables continual learning. In Forty-third
  International Conference on Machine Learning, External Links:
  [Link](https://openreview.net/forum?id=qA6FgH0nnZ) Cited by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Sheng et al. (2025) G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R.
  Zhang, Y. Peng, H. Lin, and C. Wu HybridFlow: a flexible and efficient
  RLHF framework. In Proceedings of the Twentieth European Conference on
  Computer Systems, pp. 1279–1297. Cited by: [Appendix
  B](#A2.p1.1 "Appendix B Additional Implementation Details ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p4.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Wang et al. (2026a) G. Wang, S. Dai, G. Ye, Z. Gan, W. Yao, Y.
  Deng, X. Wu, and Z. Ying Information gain-based policy optimization: a
  simple and effective approach for multi-turn search agents. In The
  Fourteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=qkWP6phrvZ)
  Cited by: [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Wang et al. (2026b) H. Wang, G. Wang, H. Xiao, Y. Zhou, Y. Pan, J.
  Wang, K. Xu, Y. Wen, X. Ruan, X. Chen, et al. Skill-sd:
  skill-conditioned self-distillation for multi-turn llm agents. arXiv
  preprint arXiv:2604.10674. Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Wang et al. (2025) H. Wang, C. Qian, W. Zhong, X. Chen, J. Qiu, S.
  Huang, B. Jin, M. Wang, K. Wong, and H. Ji Otc: optimal tool calls via
  reinforcement learning. arXiv e-prints, pp. arXiv–2504. Cited by:
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Wei et al. (2025) Y. Wei, X. Yu, Y. Weng, T. Pan, A. Li, and L. Du
  Autotir: autonomous tools integrated reasoning via reinforcement
  learning. arXiv preprint arXiv:2507.21836. Cited by: [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Xue et al. (2026) Z. Xue, L. Zheng, Q. Liu, Y. Li, X. Zheng, Z. MA,
  and B. An SimpleTIR: end-to-end reinforcement learning for multi-turn
  tool-integrated reasoning. In The Fourteenth International Conference
  on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=EplNy91Xqh) Cited by:
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Yang et al. (2025) A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B.
  Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report.
  External Links: 2505.09388 Cited by: [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p4.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Yang et al. (2026) C. Yang, C. Qin, Q. Si, M. Chen, N. Gu, D. Yao, Z.
  Lin, W. Wang, J. Wang, and N. Duan Self-distilled RLVR. External
  Links: 2604.03128 Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Ye et al. (2025) J. Ye, Z. Du, X. Yao, W. Lin, Y. Xu, Z. Chen, Z.
  Wang, S. Zhu, Z. Xi, S. Yuan, et al. ToolHop: a query-driven benchmark
  for evaluating large language models in multi-hop tool use. In
  Proceedings of the 63rd Annual Meeting of the Association for
  Computational Linguistics, pp. 2995–3021. Cited by: [Appendix
  A](#A1.p6.1 "Appendix A Benchmark and Evaluation Details ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p2.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Ye et al. (2026a) J. Ye, C. Jiang, Z. Du, Y. Xu, X. Yao, Z. Xi, X.
  Fan, Q. Zhang, T. Gui, X. Huang, and J. Chen Feedback-driven tool-use
  improvements in large language models via automated build
  environments. In Findings of the Association for Computational
  Linguistics, ACL 2026, San Diego, California, United States, July 2-7,
  2026, pp. 2293–2323. Cited by: [Appendix
  A](#A1.p2.1 "Appendix A Benchmark and Evaluation Details ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Experimental
  Setups](#Sx5.SSx1.p1.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Ye et al. (2026b) T. Ye, L. Dong, X. Wu, S. Huang, and F. Wei
  On-policy context distillation for language models. arXiv preprint
  arXiv:2602.12275. Cited by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Yu et al. (2025) Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, YuYue, W.
  Dai, T. Fan, G. Liu, J. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G.
  Sheng, Y. Tong, C. Zhang, M. Zhang, R. Zhang, W. Zhang, H. Zhu, J.
  Zhu, J. Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J.
  Liu, W. Ma, Y. Zhang, L. Yan, Y. Wu, and M. Wang DAPO: an open-source
  LLM reinforcement learning system at scale. In The Thirty-ninth Annual
  Conference on Neural Information Processing Systems, Cited by:
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zeng et al. (2025) Y. Zeng, X. Ding, Y. Hou, Y. Wang, L. Du, J.
  Dai, Q. Ding, D. Tang, D. Tu, W. Liu, et al. Tool zero: training
  tool-augmented llms via pure rl from scratch. In Findings of the
  Association for Computational Linguistics: EMNLP 2025, pp. 9135–9147.
  Cited by:
  [Introduction](#Sx1.p2.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning"),
  [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zhang et al. (2025) S. Zhang, Y. Dong, J. Zhang, J. Kautz, B.
  Catanzaro, A. Tao, Q. Wu, Z. Yu, and G. Liu Nemotron-research-tool-n1:
  tool-using language models with reinforced reasoning. arXiv preprint
  arXiv:2505.00024. Cited by: [Tool-Integrated
  Reasoning](#Sx2.SSx1.p1.1 "Tool-Integrated Reasoning ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zhang et al. (2026) Y. Zhang, X. Lin, and C. Wu StepOPSD: step-aware
  online preference distillation for agent reinforcement learning.
  External Links: 2605.27140 Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zhao et al. (2026) S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang, F.
  Chen, and A. Grover Self-distilled reasoner: on-policy
  self-distillation for large language models. In Forty-third
  International Conference on Machine Learning, External Links:
  [Link](https://openreview.net/forum?id=Jpxfof0EaS) Cited by:
  [Introduction](#Sx1.p3.1 "Introduction ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zhong et al. (2026) Q. Zhong, M. Zheng, M. Song, X. Lin, J. Sun, H.
  Jiang, X. Wang, and J. Fang Sod: step-wise on-policy distillation for
  small language model agents. arXiv preprint arXiv:2605.07725. Cited
  by: [Experimental
  Setups](#Sx5.SSx1.p3.1 "Experimental Setups ‣ Experiments ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zhou et al. (2026a) Y. Zhou, L. Zhang, Y. Wu, M. Wang, B. Peng, J.
  Liu, X. Fan, and Z. Zhao SAGE-OPD: selective agent-guided intervention
  for multi-turn on-policy distillation. External Links: 2606.19659
  Cited by: [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").
- Zhou et al. (2026b) Y. Zhou, K. Zheng, H. Li, D. Peng, C. Xu, and J.
  Chen TurnOPD: making on-policy distillation turn-aware for efficient
  long-horizon agent training. External Links: 2607.05804 Cited by:
  [On-Policy
  Self-Distillation](#Sx2.SSx2.p1.1 "On-Policy Self-Distillation ‣ Related Work ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning").

## Appendix

## Appendix A Benchmark and Evaluation Details

We use FTRL for post-training and in-domain evaluation, while BFCL and
ToolHop assess generalization to unseen tool interfaces and interaction
patterns.
Table [3](#A1.T3 "Table 3 ‣ Appendix A Benchmark and Evaluation Details ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning")
summarizes the resulting data composition. The reported counts refer to
executable queries or task instances.

| Benchmark | Usage |  Split or Subset | Instances |
|----|----|----|----|
| FTRL | Post-training |  All training environments | 2,215 |
| FTRL | In-domain evaluation |  Single-Hop / Parallel Single-Hop / Multi-Hop / Parallel Multi-Hop | 50 / 50 / 50 / 50 |
| BFCL | Out-of-domain evaluation |  Base / Missing Function / Missing Parameter / Long Context | 200 / 200 / 200 / 200 |
| BFCL | Out-of-domain evaluation |  Web Search / Memory | 200 / 465 |
| ToolHop | Out-of-domain evaluation |  Multi-hop tool-use queries | 995 |

Table 3: Data used for post-training and evaluation. Instance counts are
reported for each benchmark split or subset.

FTRL.¹¹ 1 https://github.com/bytedance/FTRL FTRL ([Ye et al.
2026a](#bib.bib33)) provides executable tool-use tasks with
automatically constructed queries, local tool environments, and
programmatically verifiable feedback. We use all 2,215 training
instances for post-training and evaluate on four test categories, each
containing 50 queries. These categories differ in their dependency
structures: Single-Hop tasks require a single tool-mediated step;
Parallel Single-Hop tasks contain independent requests that can be
resolved without cross-step dependencies; Multi-Hop tasks require later
actions to use earlier execution results; and Parallel Multi-Hop tasks
combine independent branches with sequential dependencies.

FTRL evaluates tool-use behavior at the subtask level through Solve-P,
Solve-R, and Solve-F1. Let $`N_{\mathrm{call}}`$ denote the number of
generated tool calls, $`N_{\mathrm{solved}}`$ the number of successfully
resolved subtasks, and $`N_{\mathrm{req}}`$ the total number of required
subtasks. The metrics are defined as

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathrm{Solve\text{-}P}=\begin{cases}N_{\mathrm{solved}}/N_{\mathrm{call}},&N_{\mathrm{call}}>0,\\
1,&N_{\mathrm{call}}=0.\end{cases}
``` |  | (2) |

|     |                                                               |     |     |
|-----|---------------------------------------------------------------|-----|-----|
|     |
       ``` math
       \mathrm{Solve\text{-}R}=N_{\mathrm{solved}}/N_{\mathrm{req}}.
       ```                                                            |     | (3) |

and

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathrm{Solve\text{-}F1}=\frac{2\,\mathrm{Solve\text{-}P}\,\mathrm{Solve\text{-}R}}{\mathrm{Solve\text{-}P}+\mathrm{Solve\text{-}R}}.
``` |  | (4) |

Because post-training and evaluation use the same environment family,
FTRL measures in-domain improvements in executable multi-turn tool
reasoning.

BFCL.²² 2
https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
BFCL ([Patil et al. 2025](#bib.bib34)) evaluates function-calling and
agentic behavior under diverse interaction conditions. Our evaluation
includes four multi-turn subsets and two agentic subsets. The multi-turn
portion comprises Base, Missing Function, Missing Parameter, and Long
Context. These subsets respectively assess stateful function calling,
behavior under unavailable tools, behavior under incomplete arguments,
and tool selection from lengthy specifications containing distracting
information. The agentic portion includes Web Search, which requires
external information acquisition, and Memory, which evaluates the reuse
of information obtained earlier in an interaction.

For multi-turn tasks, correctness is determined by the resulting
environment state and, when applicable, the validity of the response
path. State-modifying requests are checked through backend-state
verification, whereas read-oriented requests must also satisfy the
benchmark’s response requirements. The agentic subsets use exact
matching on the designated answer field. We report the official accuracy
of each subset together with their average.

ToolHop.³³ 3 https://huggingface.co/datasets/bytedance-research/ToolHop
ToolHop ([Ye et al. 2025](#bib.bib16)) contains 995 multi-hop queries
and 3,912 locally executable tools. Its tasks require agents to compose
dependent tool calls over a tool collection that differs from the FTRL
training environment. We therefore use ToolHop to examine whether the
learned credit-assignment strategy transfers to new tools and
interaction structures. Performance is measured by final-answer
accuracy, with a prediction counted as correct only when its answer
matches the verified target after the required tool interactions.

## Appendix B Additional Implementation Details

All experiments are implemented with the verl framework ([Sheng et al.
2025](#bib.bib18)).
Table [4](#A2.T4 "Table 4 ‣ Appendix B Additional Implementation Details ‣ TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning")
reports the shared training and rollout configuration for Qwen3-4B and
Qwen3-8B. Both models are initialized from their corresponding base
checkpoints and trained for three epochs without an explicit KL penalty.

| Configuration                        | Value                 |
|--------------------------------------|-----------------------|
| Query batch size                     | 32                    |
| Validation batch size                | 256                   |
| Maximum prompt length                | 10,000 tokens         |
| Maximum response length              | 13,000 tokens         |
| Learning rate                        | $`1\times 10^{-6}`$   |
| PPO mini-batch size                  | 32                    |
| Explicit KL coefficient              | 0                     |
| Rollout engine                       | vLLM                  |
| Sampling temperature                 | 1.0                   |
| Rollout GPU utilization              | 0.7                   |
| Rollouts per query                   | 16                    |
| Maximum interaction turns            | 10                    |
| Training epochs                      | 3                     |
| Checkpoint interval                  | 5 iterations          |
| Evaluation interval                  | 5 iterations          |
| Training hardware                    | $`8\times`$ A800-80GB |
| Lookahead depths $`\mathcal{D}_{H}`$ | $`\{1,2,3\}`$         |
| Mixing coefficient $`\lambda`$       | 0.5                   |
| Modulation bound $`\epsilon_{w}`$    | 0.5                   |

Table 4: Training, rollout, and method-specific hyperparameters shared
across experiments.

Prompt-Group Normalization. For each query, we form one normalization
pool from the selected hindsight gaps at every valid policy-token
position across all 16 sampled rollouts. A single mean and standard
deviation are computed from this pool and shared by all turns in the
prompt group; trajectories are not partitioned according to turn index.
Invalid or masked positions, including environment-generated observation
tokens, do not contribute to these statistics. This implementation
allows trajectories of different lengths to be normalized together
without introducing artificial values for turns they do not contain.

Multi-Horizon Teacher Evaluation. The hindsight teachers share a frozen
reference policy but differ in the amount of future execution
information included in their contexts. Specifically, we construct
teacher views with lookahead depths $`1`$, $`2`$, and $`3`$. Each
teacher scores the student’s sampled response through teacher forcing,
after which the token-level gaps are aggregated into turn-level
hindsight assessments. The resulting statistics, teacher-selection
outputs, and advantage-modulation weights are detached from the
computational graph. Environment-generated observations are excluded
from the policy loss because they are not actions sampled by the agent.
All privileged contexts and teacher computations are used only during
training; deployment requires only the learned policy.

Evaluation Protocol. All compared methods use the same benchmark
environments and decoding settings. For the BFCL Web Search subset,
search results are obtained through the Google Search endpoint of the
Serper API.⁴⁴ 4 https://serper.dev/ The remaining BFCL subsets, together
with FTRL and ToolHop, are evaluated in their released local execution
environments. This shared protocol ensures that differences in
performance are attributable to the learned policies rather than changes
in tool execution or evaluation conditions.
````
