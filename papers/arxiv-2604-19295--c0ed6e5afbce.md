---
identifier: arxiv:2604.19295
title: "TEMPO: Scaling Test-time Training for Large Reasoning Models"
authors:
  - Qingyang Zhang
  - Xinke Kong
  - Haitao Wu
  - Qinghua Hu
  - Minghao Wu
  - Baosong Yang
  - Yu Cheng
  - Yun Luo
  - Ganqu Cui
  - Changqing Zhang
published: "2026-04-21T00:00:00+00:00"
url: https://huggingface.co/papers/2604.19295
source: huggingface
doi: null
arxiv_id: "2604.19295"
categories: []
---

# TEMPO: Scaling Test-time Training for Large Reasoning Models

Qingyang Zhang ^(†)^(†)thanks: Equal contribution.    Xinke
Kong⁰⁰footnotemark: 0    Haitao Wu    Qinghua Hu    Minghao Wu   
Baosong Yang    Yu Cheng    Yun Luo ^(†)^(†)thanks: Co-supervised.
Correspondence to Yun Luo, Ganqu Cui and Changqing Zhang.    Ganqu
Cui⁰⁰footnotemark: 0    Changqing Zhang⁰⁰footnotemark: 0    Tianjin
University Tongyi Lab    Alibaba Group The Chinese University of Hong
Kong Shanghai AI Lab

###### Abstract

Test-time training (TTT) adapts model parameters on unlabeled test
instances during inference time, which continuously extends capabilities
beyond the reach of offline training. Despite initial gains, existing
TTT methods for LRMs plateau quickly and do not benefit from additional
test-time compute. Without external calibration, the self-generated
reward signal increasingly drifts as the policy model evolves, leading
to both performance plateaus and diversity collapse. We propose TEMPO, a
TTT framework that interleaves policy refinement on unlabeled questions
with periodic critic recalibration on a labeled dataset. By formalizing
this alternating procedure through the Expectation-Maximization (EM)
algorithm, we reveal that prior methods can be interpreted as incomplete
variants that omit the crucial recalibration step. Reintroducing this
step tightens the evidence lower bound (ELBO) and enables sustained
improvement. Across diverse model families (Qwen3 and OLMO3) and
reasoning tasks, TEMPO improves OLMO3-7B on AIME 2024 from 33.0% to
51.1% and Qwen3-14B from 42.3% to 65.8%, while maintaining high
diversity. Code is available at [this
url](https://github.com/QingyangZhang/TEMPO).

[  Project Page](https://qingyangzhang.github.io/tempo-homepage) \|  [
 GitHub](https://github.com/QingyangZhang/TEMPO) \|  [
 HuggingFace](https://huggingface.co/collections/qingyangzhang/tempo)

![Refer to caption](2604.19295v1/cover.png)

Figure 1: Scalability of TEMPO on the AIME benchmark. TEMPO alternates
between an E-Step (critic recalibration on labeled data) and an M-Step
(policy refinement on unlabeled test questions), guided by a critic that
provides quality-aware scores. Representative self-rewarding TTT
baselines such as TTRL (grey curves) plateau and collapse after initial
gains. In contrast, TEMPO (blue curve) sustains a consistent upward
trajectory over 350 steps by periodically grounding the critic in
external supervision, demonstrating that additional test-time compute
translates into scalable performance gains on complex open problems.

## 1 Introduction

Large reasoning models (LRMs) have demonstrated remarkable capabilities
on complex reasoning tasks, including logic puzzle-solving and
Olympiad-level mathematics and physics [Cui et al. (2025)](#bib.bib20);
[Chen et al. (2025)](#bib.bib1); [Luo et al. (2026)](#bib.bib2). These
models achieve their performance by allocating extensive computation at
test time through extended reasoning chains. However, their parameters
remain static after training, which prevents them from incorporating
knowledge acquired during test-time experience. Test-time training (TTT)
addresses this limitation by enabling models to update their parameters
on unlabeled test data, thereby extending their reasoning capabilities
beyond the original training distribution. Recent methods such as
EMPO [Zhang et al. (2025c)](#bib.bib12), TTRL [Zuo et al.
(2025)](#bib.bib26), and Theta-Evolve [Wang et al. (2025)](#bib.bib9)
employ self-generated reward signals such as entropy, majority voting,
or self-consistency to refine reasoning policies via reinforcement
learning without ground-truth labels. Practical implementations such as
Cursor’s real-time reinforcement learning for Composer demonstrate the
efficacy of test-time training in dynamically adapting model to complex,
interactive coding environments [Jackson et al. (2026)](#bib.bib6).

Despite promising initial results, existing TTT methods for LRMs exhibit
two fundamental limitations. First, they rely on heuristic reward
signals that are intrinsically bounded by the model’s initial
capabilities, leading to performance plateaus as self-improvement
saturates [Zuo et al. (2026)](#bib.bib10). Second, these methods tend to
collapse output diversity in pursuit of higher average performance,
ultimately degrading reasoning quality [Zhang et al.
(2025d)](#bib.bib11). Both issues share a common root cause: there are
no ground-truth labels available at test time, and the reward signal
must be inferred from the model’s own outputs. As the model becomes
increasingly confident in a narrow set of reasoning patterns, these
heuristic signals systematically overestimate the quality of
self-generated responses, creating a self-reinforcing loop that drives
both saturation and diversity collapse.

To this end, we propose Test-time Expectation-Maximization Policy
Optimization (TEMPO), a TTT framework that decouples reward generation
from policy optimization through an alternating actor-critic design
(Figure [1](#S0.F1 "Figure 1 ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")).
TEMPO operates in two stages: (i) Policy Refinement, where the actor
generates reasoning trajectories on unlabeled test prompts and optimizes
against rewards from a critic model, enabling on-the-fly adaptation to
novel problems; and (ii) Critic Recalibration, where the critic is
periodically updated using verifiable rewards from a labeled dataset. By
maintaining a grounded critic, TEMPO provides a stable training signal
that avoids the reward drift responsible for prior failures. We
formalize this alternating procedure through the
Expectation-Maximization (EM) algorithm. The key insight is that
response correctness is an unobserved latent variable at test time.
Thus, optimizing the policy without ground-truth labels is naturally
framed as maximizing a lower bound on the expected reward. In this view,
the critic recalibration corresponds to the E-step (estimating the
posterior distribution over correct responses), while policy
optimization corresponds to the M-step (updating model parameters given
those estimates). This framing reveals that existing self-rewarding
methods are degenerate variants that execute only the M-step, causing
the estimated posterior to drift from true correctness. Restoring the
E-step through periodic calibration tightens the lower bound and
sustains improvement over extended training horizons.

Experimental results validate both the effectiveness and scalability of
TEMPO. On AIME 2024 and 2025, TEMPO improves OLMO3-7B from 33.0% and
26.3% to 51.1% and 37.0%. For Qwen3-14B, TEMPO pushes the accuracy from
42.3% and 37.1% to 65.8% and 44.6%, achieves an absolute gain of 23.5
and 7.5 percentage points, respectively. More importantly, TEMPO
maintains high pass@K scores where baselines suffer from diversity
collapse. Beyond mathematics, TEMPO generalizes to non-math reasoning
domains, including logic puzzles and STEM tasks, confirming that the
alternating critic-policy design is not domain-specific.

Our contributions are summarized as follows:

- •
  We propose TEMPO, a test-time training framework that achieves
  sustained performance gains through an alternating actor-critic
  optimization, avoiding the diversity collapse and performance plateaus
  of prior LRMs self-training methods (Sec.
  [3](#S3 "3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")).
- •
  We provide a unified analysis that characterizes existing TTT methods
  as incomplete EM procedures that omit the crucial posterior
  recalibration. By identifying the missing E-step as the root cause of
  scalability failures, this perspective yields a principled remedy:
  restoring periodic critic calibration on labeled data (Sec.
  [5](#S5 "5 Discussion ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")).
- •
  We conduct extensive experiments across model families, scales
  (OLMO3-7B, Qwen3-8B, and Qwen3-14B), and five reasoning benchmarks
  spanning math, logic puzzles, and STEM. TEMPO demonstrates both
  superior accuracy and preserved output diversity, confirming that the
  alternating design generalizes beyond mathematical reasoning (Sec.
  [4](#S4 "4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")).

## 2 Related Work

This section surveys two lines of prior work that motivate our approach:
self-rewarding RL methods that avoid ground-truth labels but suffer from
reward self-reinforcement, and existing TTT methods for reasoning models
that share the structural deficiency of omitting reward calibration.

#### Self-rewarding reinforcement learning.

RLVR, first formalized by Tulu-V3 [Lambert et al. (2024)](#bib.bib21),
has emerged as the dominant paradigm for incentivizing reasoning
capability in LLMs [Guo et al. (2025)](#bib.bib17); [Shao et al.
(2024)](#bib.bib18); [Cui et al. (2025)](#bib.bib20), but its reliance
on labeled data motivates self-rewarding alternatives. Previous works
leverage intrinsic rewards such as entropy [Zhang et al.
(2025c)](#bib.bib12); [Gao et al. (2025)](#bib.bib16),
self-certainty [Zhao et al. (2025)](#bib.bib15), or reasoning
topology [Wang et al. (2026)](#bib.bib8) for self-training, without
dependency on external supervision. For example, SARL [Wang et al.
(2026)](#bib.bib8) constructs rewards from the graph structure of
intermediate reasoning steps, optimizing to encourage locally coherent
and efficient thinking. LaSeR [Yang et al. (2026)](#bib.bib14)
demonstrates that the logit of the last token can serve as an effective
self-rewarding signal, achieving superior reasoning accuracy and
inference-time scaling with only one additional token of computation.
However, these methods tend to collapse the output distribution and
plateau as the reward signal becomes self-reinforcing [Zhang et al.
(2025d)](#bib.bib11); [Zuo et al. (2026)](#bib.bib10). Our approach
avoids this by decoupling reward generation (a critic periodically
re-calibrated on labeled data) from policy optimization (on unlabeled
test data), preventing the self-reinforcement loop.

#### Test-time training for reasoning models.

TTT originated in computer vision, where models continuously update
their parameters on each test instance at inference time to fill the gap
of distribution shifts [Grandvalet and Bengio (2004)](#bib.bib19); [Wang
et al. (2021)](#bib.bib22); [Zhang et al. (2025b)](#bib.bib23). In the
LLM reasoning domain, recent methods apply test-time RL using
self-generated signals: TTRL [Zuo et al. (2025)](#bib.bib26) uses
majority voting for pseudo-labels, Intuitor [Zhao et al.
(2025)](#bib.bib15) and EMPO [Zhang et al. (2025c)](#bib.bib12) use
entropy-based rewards. These methods share a structural deficiency: they
perform only policy refinement while neglecting reward calibration,
which causes the reward signal to drift from true correctness as the
policy evolves. TEMPO addresses this by interleaving critic
recalibration (E-step) with policy optimization (M-step), maintaining a
tight ELBO and enabling sustained improvement beyond the ceilings of
prior methods.

## 3 Method

We propose Test-time Expectation-Maximization Policy Optimization
(TEMPO), a TTT framework that initializes actor and critic via RLVR on
labeled data $`D_{L}`$, then continuously improves on unlabeled test
questions by alternating between critic calibration and policy
refinement following the EM algorithm. This section is organized as
follows: we first formalize the problem setup
(Section [3.1](#S3.SS1 "3.1 Problem Setup ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")),
then derive an EM-inspired variational lower bound as optimization
objective
(Section [3.2](#S3.SS2 "3.2 Variational Lower Bound Objective ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")),
and finally detail the alternating E-step critic calibration
(Section [3.3](#S3.SS3 "3.3 Expectation Step: Posterior Estimation via Critic ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"))
and M-step policy optimization
(Section [3.4](#S3.SS4 "3.4 Maximization Step: Policy Optimization ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")).

### 3.1 Problem Setup

We consider a setting where the model has access to a labeled dataset
$`D_{L}`$ containing ground-truth answers and a set of unlabeled test
questions $`D_{u}`$ where the correct responses are unknown. Our goal is
to enable LRMs to continuously self-improve during the test phase.
Formally, we aim to maximize the expected log-probability of generating
a correct response given a question $`x`$. Let $`\theta`$ denote the
parameters of the LRM, and $`P(\text{Correct}|x;\theta)`$ represent the
probability that the model produces a correct output for a given input
$`x`$. The global objective function is defined as follows:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

       ``` math
       J(\theta)=\mathbb{E}_{x}[\log P(\text{Correct}|x;\theta)],
       ```                                                         |     | (1) |

where the marginal probability $`P(\text{Correct}|x;\theta)`$ is
obtained by marginalizing over all possible generated responses $`y`$ as

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
P(\text{Correct}|x;\theta)=\sum_{y}P(\text{Correct}|x,y)\pi_{\theta}(y|x).
``` |  | (2) |

Here, $`\pi_{\theta}(y|x)`$ denotes the policy, which represents the
output distribution of the LRMs.

### 3.2 Variational Lower Bound Objective

We first derive the evidence lower bound (ELBO) that enables
optimization when ground-truth correctness is unobserved, showing how
the EM framework decomposes the objective into an estimable lower bound
and a KL divergence term.

The fundamental challenge in test-time training is that the response
correctness for $`x\in D_{u}`$ is unobserved. In such scenarios, the
optimal response distribution $`P(y|x,\text{Correct})`$ is a latent
variable. To optimize $`J(\theta)`$ under these conditions, we employ
the Expectation-Maximization (EM) framework. By introducing an auxiliary
distribution $`q(y|x)`$, we derive the Evidence Lower Bound (ELBO):

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle J(\theta)`$ | $`\displaystyle=\sum_{x\in D_{u}}\log\sum_{y}q(y|x)\frac{P(\text{Correct}|y,x)\pi_{\theta}(y|x)}{q(y|x)}\ `$ |  | (3) |
|  |  | $`\displaystyle=\sum_{x\in D_{u}}\left(\sum_{y}q(y|x)\log\frac{P(\text{Correct}|y,x)\pi_{\theta}(y|x)}{q(y|x)}+KL(q(y|x)||P(y|x,\text{Correct}))\right).`$ |  | (4) |

Intuitively, this decomposition says: maximizing the lower bound
corresponds to (i) increasing the expected log-likelihood of responses
that are likely to be correct, and (ii) bringing the auxiliary
distribution $`q`$ closer to the true posterior
$`P(y|x,\text{Correct})`$.

By omitting the non-negative KL divergence term, we obtain the objective
$`\mathcal{L}(q,\theta)`$, where $`J(\theta)\geq\mathcal{L}(q,\theta)`$.
The equality holds if and only if $`q(y|x)`$ perfectly matches the
posterior distribution $`P(y|x,\text{Correct})`$. This allows for
iterative refinement by alternating between estimating the distribution
of correct responses and maximizing model likelihood.

### 3.3 Expectation Step: Posterior Estimation via Critic

Then we describe how to train a critic model on labeled data to
approximate the posterior distribution over correct responses, thereby
grounding the reward signal in external supervision. In the E-step, we
keep the current policy $`\pi_{\theta_{0}}`$ fixed and seek the optimal
auxiliary distribution $`q^{*}(y|x)`$ that maximizes the lower bound.
This optimal distribution corresponds to the posterior probability of
the response conditioned on its correctness:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
q(y|x)=P(y|x,\text{Correct})=\frac{P(\text{Correct}|y,x)\pi_{\theta_{0}}(y|x)}{P(\text{Correct}|x)}.
``` |  | (5) |

To approximate the unknown term $`P(\text{Correct}|y,x)`$, we train a
critic model $`V_{\phi}(x,y_{t})\in\mathbb{R}`$ using the labeled data
$`D_{L}`$, where $`t\geq 0`$ is the token index. The critic is optimized
by minimizing the MSE between its token-level predictions and the
ground-truth outcomes. For each response $`y`$ associated with a query
$`x\in D_{L}`$, the critic is trained to perform token-level value
estimation. Formally, the critic parameters $`\phi`$ are updated by

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{critic}}(\phi)=\mathbb{E}_{(x,y,\mathcal{I})\sim D_{L}}\|V_{\phi}(x,y_{t})-\mathcal{I}\|_{2}^{2},
``` |  | (6) |

where $`\mathcal{I}\in\{0,1\}`$ denotes the binary correctness
indicator. This training regime ensures the critic serves as a reliable
proxy for the expected correctness of generated response $`y`$.

Once optimized, the critic provides a tractable surrogate for the
posterior distribution. Since the critic is trained to predict outcome
correctness, its last-token value $`V_{\phi}(x,y_{T})`$ directly
reflects the likelihood of a correct response, enabling the optimal
auxiliary distribution to be approximated by reweighting the model’s
current outputs with the critic scores as follows:

|     |                                                       |     |     |
|-----|-------------------------------------------------------|-----|-----|
|     |
       ``` math
       q(y|x)\propto V_{\phi}(x,y_{T})\pi_{\theta_{0}}(y|x),
       ```                                                    |     | (7) |

where $`T`$ is the response length of $`y`$. This step effectively
identifies high-quality responses from the model’s own generations to
serve as a surrogate for the missing ground-truth labels.

### 3.4 Maximization Step: Policy Optimization

Finally, we formulate the policy update as a weighted maximum likelihood
estimation using critic-derived rewards, and implement it via a policy
gradient RL framework with token-level advantage estimation. In the
M-step, we fix the auxiliary distribution $`q(y|x)`$ and update the
model parameters $`\theta`$ by maximizing the lower bound
$`\mathcal{L}(q,\theta)`$. The optimization problem is formulated as
follows:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\theta_{\text{new}}=\arg\max_{\theta}\sum_{x\in D_{u}}\sum_{y}q(y|x)\log\left(P(\text{Correct}|y,x)\pi_{\theta}(y|x)\right).
``` |  | (8) |

By removing terms independent of $`\theta`$, the objective simplifies to
a weighted maximum likelihood estimation. Substituting the approximation
of $`q^{*}(y|x)`$ derived in the E-step, the update rule becomes

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\theta_{\text{new}}=\arg\max_{\theta}\sum_{x\in D_{u}}\sum_{y}V_{\phi}(x,y_{T})\pi_{\theta_{0}}(y|x)\log\pi_{\theta}(y|x).
``` |  | (9) |

Given that $`\pi_{\theta_{0}}(y|x)`$ represents the sampling
distribution, the inner summation can be interpreted as an expectation.
We further simplify the objective by focusing on the weighted
log-likelihood of the sampled responses as

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\theta_{\text{new}}=\arg\max_{\theta}\sum_{x\in D_{u}}\sum_{y}V_{\phi}(x,y_{T})\log\pi_{\theta}(y|x).
``` |  | (10) |

This optimization is then solved using policy gradient methods,
facilitating the continuous self-refinement of the policy model based on
its reasoning trajectory on the unlabeled open questions.

To effectively optimize this objective while ensuring variance
reduction, we implement the M-step update via a policy gradient-based RL
framework. In this setting, the value prediction of the critic at the
terminal token of the sequence is treated as the ground-truth external
reward $`R=V_{\phi}(x,y)`$ for the entire response. To derive a stable
training signal, we utilize the critic’s intermediate value predictions
$`V_{\phi}(x,y_{1:t})`$ as token-varying baselines $`b_{t}`$. The
advantage $`A_{t}`$ for each token $`y_{t}`$ is defined as the
discrepancy between the final realized reward and the expected value at
step $`t`$:

|     |                              |     |      |
|-----|------------------------------|-----|------|
|     |
       ``` math
       A_{t}=R-V_{\phi}(x,y_{1:t}).
       ```                           |     | (11) |

This formulation ensures that tokens contributing to a higher last-token
value prediction receive a positive reinforcement signal, while those
leading to deviations from the predicted value are penalized. The policy
optimization objective for the M-step is

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\mathcal{L}_{\text{policy}}(\theta)=-\mathbb{E}_{x\in D_{u},y\sim\pi_{\theta}}\left[\sum_{t=1}^{T}A_{t}\log\pi_{\theta}(y_{t}|x,y_{<t})\right].
``` |  | (12) |

By alternating between the critic recalibration (E-step) and the policy
refinement (M-step), the model achieves continuous, scalable
self-improvement on unlabeled open reasoning problems. The full
alternating procedure is summarized in
Figure [2](#S3.F2 "Figure 2 ‣ 3.4 Maximization Step: Policy Optimization ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")
and
Algorithm [1](#algorithm1 "In 3.4 Maximization Step: Policy Optimization ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").

Input: Labeled dataset $`D_{L}`$, unlabeled test set $`D_{u}`$, initial
policy $`\pi_{\theta_{0}}`$, critic $`V_{\phi_{0}}`$, total iterations
$`N`$

Output: Updated policy $`\pi_{\theta_{N}}`$

/\* Stage 1: Initialization \*/

1 Train actor $`\pi_{\theta_{0}}`$ and critic $`V_{\phi_{0}}`$ via RLVR
on $`D_{L}`$;

2 for *$`k=1`$ to $`N`$* do

   /\* Stage 2: Alternating test-time training \*/

   /\* E-step: Critic recalibration \*/

    3 Sample $`(x,y,\mathcal{I})`$ from $`D_{L}`$;

    4 Update critic $`\phi`$ by minimizing
$`\mathcal{L}_{\text{critic}}(\phi)`$
(Eq. ([6](#S3.E6 "In 3.3 Expectation Step: Posterior Estimation via Critic ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")));

   /\* M-step: Policy refinement \*/

    5 Sample $`x\sim D_{u}`$, generate responses $`y\sim\pi_{\theta}`$;

    6 Compute advantages $`A_{t}=V_{\phi}(x,y_{T})-V_{\phi}(x,y_{1:t})`$
(Eq. ([11](#S3.E11 "In 3.4 Maximization Step: Policy Optimization ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")));

    7 Update policy $`\theta`$ by minimizing
$`\mathcal{L}_{\text{policy}}(\theta)`$
(Eq. ([12](#S3.E12 "In 3.4 Maximization Step: Policy Optimization ‣ 3 Method ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")));

8 end for

9 return $`\pi_{\theta_{N}}`$;

Algorithm 1 Test-time Expectation-Maximization Policy Optimization
(TEMPO)

![Refer to caption](2604.19295v1/tempo.png)

Figure 2: TEMPO alternates between (i) Critic Recalibration (E-step):
the critic is periodically updated using verifiable rewards from
$`D_{L}`$ to maintain a grounded and informative reward signal, and (ii)
Policy Refinement (M-step): the actor generates reasoning trajectories
on unlabeled questions $`D_{u}`$ and optimizes against critic-derived
rewards. This alternating EM-style procedure enables sustained
self-improvement beyond the RLVR plateau.

## 4 Experiments

We empirically validate TEMPO across four dimensions. We first describe
the experimental setup
(Section [4.1](#S4.SS1 "4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")),
then demonstrate sustained scalability beyond RLVR ceilings
(Section [4.2](#S4.SS2 "4.2 Scalability (RQ1) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")),
show that TEMPO preserves output diversity while baselines collapse
(Section [4.3](#S4.SS3 "4.3 Diversity (RQ2) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")),
validate generalization to non-math reasoning tasks
(Section [4.4](#S4.SS4 "4.4 Versatility (RQ3) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")),
and finally ablate the alternating training design to confirm the
necessity of each component
(Section [4.5](#S4.SS5 "4.5 Ablation (RQ4) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")).

### 4.1 Experimental Setup

We evaluate TEMPO on both mathematical reasoning and general domain
reasoning tasks, covering a diverse set of base models, datasets, and
baselines.

- •
  For math experiments, we select Qwen3 [Yang et al. (2025)](#bib.bib3)
  and OLMO3 [Olmo et al. (2025)](#bib.bib4) as the base models. We first
  initialize the actor and critic models with PPO on DAPO-Math-17K [Yu
  et al. (2025)](#bib.bib24) as the labeled training dataset, using
  standard RLVR to establish a starting point. Subsequently, we perform
  test-time training driven by TEMPO on a test set consisting of AIME
  2024, AIME 2025, and Beyond AIME [\[ByteDance-Seed\]
  (2025)](#bib.bib5). For a more comprehensive evaluation, we introduce
  AIME 2026 and OlymMath [Zhang et al. (2025a)](#bib.bib7) as holdout
  test sets to assess generalization. For the RL training recipes, we
  set the batch size to 256 and the mini-batch size to 64 for models up
  to 8B. A batch size of 128 and a mini-batch size of 32 for the 14B
  model due to GPU memory constraints. The maximum response length is
  16K. To stabilize off-policy training, we implement the sequence clip
  mechanism with a dual-clip ratio of $`3\times 10^{-4}`$ and
  $`5\times 10^{-4}`$ as suggested by GSPO [Zheng et al.
  (2025)](#bib.bib13). We compare TEMPO against several representative
  baselines, including standard RLVR trained via PPO and representative
  self-training methods TTRL [Zuo et al. (2025)](#bib.bib26) and
  EMPO [Zhang et al. (2025c)](#bib.bib12). We report avg@16 accuracy
  (average accuracy over 16 independent samples per problem) and pass@8
  (the fraction of problems solved by at least one of 8 samples).
- •
  For general domain reasoning tasks, we initialize the actor and critic
  via PPO on Dolci-RL-Zero-General [Olmo et al. (2025)](#bib.bib4), a
  12.8K labeled corpus. Subsequently, we perform test-time training
  driven by TEMPO on a test set consisting of BigBenchHard [Suzgun et
  al. (2023)](#bib.bib27), AGI Eval [Zhong et al. (2024)](#bib.bib28),
  and ZebraLogic [Lin et al. (2025)](#bib.bib29). For a more
  comprehensive evaluation, we introduce GPQA-Diamond [Rein et al.
  (2024)](#bib.bib25) as a holdout test set to assess generalization.
  All RL training hyperparameters remain identical to the math
  experiments. To ensure high-fidelity assessment across these diverse
  domains, we employ gpt-oss-120b [Agarwal et al. (2025)](#bib.bib30) as
  the judge model for correctness verification. Since the volume of BBH,
  AGI Eval, and ZebraLogic is already sufficient to ensure stable
  evaluation, we solely report Avg@1 accuracy for them, while reporting
  Avg@8 and Pass@8 for the highly complex GPQA-Diamond benchmark.

### 4.2 Scalability (RQ1)

We first evaluate whether TEMPO can sustainably improve model
performance beyond the RLVR training ceiling by leveraging unlabeled
test data. We compare TEMPO against the RLVR baseline and two
representative self-rewarding TTT methods (TTRL and EMPO) across three
model families and five benchmarks. Results are shown in
Table [1](#S4.T1 "Table 1 ‣ 4.2 Scalability (RQ1) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").

TEMPO significantly outperforms all baselines across model scales and
benchmarks. For instance, on the AIME 24 dataset, TEMPO improves the
avg@16 accuracy of OLMO3-7B-Base from 33.0% to 51.1%, and Qwen3-14B-Base
from 42.3% to 65.8%. Gains are particularly pronounced on the most
challenging benchmarks (AIME 24 and AIME 25), where prior methods show
the largest degradation relative to TEMPO. This scalability stems from
the EM-based alternating structure. By periodically recalibrating the
critic on labeled data, TEMPO prevents the reward signal from drifting
as the policy evolves, which is the failure mode that causes prior
methods to plateau once the model becomes overconfident in a narrow set
of reasoning patterns. In contrast, the grounded critic in TEMPO
continues to provide informative gradients, enabling the model to
explore high-quality reasoning paths for challenging questions at
test-time.

We further evaluate TEMPO’s ability to surpass the RLVR performance
upper bound by continuing from a converged OLMO3 model (pre-trained for
192 steps on DAPO-Math-17K). As shown in
Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Ablation (RQ4) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
further RLVR training with PPO yields negligible gains, whereas
TEMPO-based test-time training produces a consistent performance surge
over 200 iterations. The widening gap between these two curves confirms
that TEMPO translates additional test-time compute into measurable
capability gains, transcending the limits of standard RLVR.

|  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
| Method | Beyond AIME |  | AIME 24 |  | AIME 25 |  | AIME 26 |  | OlymMath |  |
|  | Acc | Pass@K | Acc | Pass@K | Acc | Pass@K | Acc | Pass@K | Acc | Pass@K |
| Frontier Models |  |  |  |  |  |  |  |  |  |  |
| Oat-Zero-7B | 9.4 | 19.4 | 30.2 | 46.1 | 12.3 | 33.7 | 16.7 | 26.3 | 11.1 | 22.6 |
| MiMo-Zero-RL-7B | 14.6 | 33.1 | 37.7 | 63.9 | 32.3 | 51.9 | 35.0 | 52.8 | 16.7 | 36.1 |
| OLMO3.1-Zero-RL-7B | 13.8 | 32.1 | 31.9 | 56.3 | 26.5 | 39.7 | 24.0 | 42.4 | 14.3 | 42.3 |
| OLMO3-7B-Base |  |  |  |  |  |  |  |  |  |  |
| $`\hookrightarrow`$ Zero-RL (PPO) | 17.6 | 38.8 | 33.0 | 56.1 | 26.3 | 41.1 | 26.7 | 42.8 | 18.9 | 43.3 |
|    $`\hookrightarrow`$ TTRL | 21.8 | 22.3 | 40.8 | 45.6 | 27.1 | 30.7 | 22.8 | 39.2 | 18.9 | 33.0 |
|    $`\hookrightarrow`$ EMPO | 21.3 | 28.4 | 41.6 | 43.3 | 26.7 | 29.5 | 23.6 | 39.7 | 18.7 | 32.9 |
|    $`\hookrightarrow`$ TEMPO | 24.5 | 44.0 | 51.1 | 61.6 | 37.0 | 52.5 | 30.1 | 49.4 | 23.5 | 51.6 |
|    Improvement via TTT | +6.9 | +5.2 | +18.1 | +5.5 | +10.7 | +11.4 | +3.4 | +6.6 | +4.6 | +8.3 |
| Qwen3-8B-Base |  |  |  |  |  |  |  |  |  |  |
| $`\hookrightarrow`$ Zero-RL (PPO) | 15.6 | 33.6 | 26.3 | 53.0 | 25.4 | 44.8 | 21.9 | 43.7 | 15.0 | 39.9 |
|    $`\hookrightarrow`$ TTRL | 18.7 | 20.0 | 29.0 | 30.0 | 32.8 | 33.3 | 13.8 | 25.0 | 11.4 | 25.3 |
|    $`\hookrightarrow`$ EMPO | 16.7 | 23.3 | 32.3 | 26.7 | 33.3 | 35.4 | 19.4 | 33.3 | 13.1 | 26.7 |
|    $`\hookrightarrow`$ TEMPO | 20.0 | 36.7 | 42.7 | 61.1 | 40.8 | 60.4 | 24.2 | 50.7 | 18.7 | 43.3 |
|    Improvement via TTT | +4.4 | +3.1 | +16.4 | +8.1 | +15.4 | +15.6 | +2.3 | +7.0 | +3.7 | +3.4 |
| Qwen3-14B-Base |  |  |  |  |  |  |  |  |  |  |
| $`\hookrightarrow`$ Zero-RL (PPO) | 24.9 | 50.0 | 42.3 | 69.1 | 37.1 | 59.0 | 38.1 | 70.0 | 24.2 | 51.6 |
|    $`\hookrightarrow`$ TTRL | 25.5 | 29.4 | 53.1 | 56.7 | 40.8 | 45.8 | 31.7 | 43.0 | 18.3 | 31.7 |
|    $`\hookrightarrow`$ EMPO | 27.6 | 31.4 | 55.6 | 59.7 | 44.6 | 46.7 | 28.3 | 49.5 | 17.7 | 31.9 |
|    $`\hookrightarrow`$ TEMPO | 29.3 | 46.3 | 65.8 | 73.3 | 44.6 | 60.0 | 38.8 | 70.0 | 25.8 | 50.2 |
|    Improvement via TTT | +4.4 | -3.7 | +23.5 | +4.2 | +7.5 | +1.0 | +0.7 | 0.0 | +1.6 | -1.4 |

Table 1: Main results on mathematical reasoning benchmarks. We report
avg@16 accuracy and pass@8 over 16 independent samples across five
benchmarks and the absolute Improvement via TTT of TEMPO over the
Zero-RL baseline.

### 4.3 Diversity (RQ2)

#### TEMPO improves model performance without compromising diversity.

A critical yet often overlooked aspect of test-time training is the
preservation of output diversity. While many methods achieve short-term
avg@16 accuracy gains, they frequently suffer from diversity collapse,
i.e., the model converges to a narrow set of reasoning patterns, causing
pass@k to degrade even as mean@k improves. This phenomenon fundamentally
limits the scalability of test-time training, as additional samples
yield diminishing returns. As reported in
Table [1](#S4.T1 "Table 1 ‣ 4.2 Scalability (RQ1) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
TEMPO consistently maintains high pass@k scores across all benchmarks,
while representative baselines exhibit significant diversity
degradation. For instance, on the Qwen3-14B-Base model, TEMPO achieving
a pass@k of 73.0 on AIME 24 and 64.3 on AIME 25, substantially
outperforming TTRL’s 56.7 and 43.3, respectively. Similarly, EMPO
records 60.0 on Beyond AIME and 46.7 on AIME 25, trailing TEMPO by 7.6
and 17.6 points.

This performance gap stems from fundamental differences in how each
method constructs its self-training signal. EMPO and TTRL rely on
entropy or self-consistency to encourage consensus, which inherently
favors the most common reasoning path regardless of its quality. As
training progresses, the model becomes increasingly confident in its
dominant mode, suppressing alternative valid solutions and causing the
output distribution to collapse. In contrast, TEMPO employs a
dynamically calibrated critic that assigns continuous, quality-aware
scores to each generated response. This mechanism naturally preserves
diversity among high-quality solutions while down-weighting incorrect
but frequently generated patterns.

Figure 3: TEMPO preserves model diversity. We compare TEMPO with TTRL on
Beyond AIME pass@16. While TTRL consistently degrades pass@16 throughout
training, TEMPO maintains and steadily improves pass@16. This reveals a
fundamental distinction: prior methods trade away exploration capacity
for short-term performance, whereas TEMPO sustains genuine reasoning
diversity as a foundation for continued self-improvement.

Figure 4: TEMPO continues to improve beyond reported results. The
numbers reported in our main table correspond to TEMPO trained on
Qwen3-14B for 224 steps. As shown here, performance on both AIME 2024
and AIME 2025 has not plateaued at that checkpoint. The avg@16 accuracy
continues to rise with further training. This suggests that the results
we report are conservative, and TEMPO has the potential for even more
gains given additional compute.

### 4.4 Versatility (RQ3)

#### TEMPO is applicable for general reasoning tasks beyond math.

We investigate whether the effectiveness of TEMPO extends beyond
mathematical problem-solving. To this end, we evaluate both the
OLMO3-7B-Base and Qwen3-8B-Base models across four diverse,
general-reasoning domains: BigBenchHard (BBH), AGI Eval, ZebraLogic, and
the expert-level GPQA-Diamond. As shown in
Table [2](#S4.T2 "Table 2 ‣ TEMPO is applicable for general reasoning tasks beyond math. ‣ 4.4 Versatility (RQ3) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
TEMPO demonstrates significant versatility and robust performance across
both OLMO3 and Qwen3 families.

Specifically, TEMPO achieves substantial absolute gains of +21.4 on BBH
and +24.5 on AGI Eval for OLMO3-7B. These enhancements enable the 7B
model to surpass specialized frontier models like General-Reasoner-7B
and MiMo-Zero-RL-7B. On the Qwen3-8B model, despite its much higher
starting performance, TEMPO still yields consistent improvements,
particularly on the logic-intensive ZebraLogic (+8.2) and the
expert-level GPQA-Diamond (+5.0 Avg@8). This suggests that TEMPO is not
merely overfitting to specific reasoning patterns but is effectively
enhancing the underlying logical capabilities across different base
model architectures.

When compared to other test-time training methods, TEMPO demonstrates a
more balanced and reliable performance profile. Prior self-training
methods often exhibit systemic sensitivity to the initial policy’s
capability or the domain’s complexity. In contrast, TEMPO achieves
massive gains on OLMO3 across the board, outperforming TTRL by over 20
points on BBH and AGI Eval. Furthermore, on the Qwen3 model, TEMPO
maintains an edge in AGI Eval and ZebraLogic. These results indicate
that the alternating training design in TEMPO is particularly robust
when the initial policy is less mature, providing a grounded signal that
prevents the performance stagnation or collapse observed in prior
self-training baselines.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Method | BBH | AGI | Zebra | GPQA-Diamond |  |
|  | (Avg@1) | (Avg@1) | (Avg@1) | (Avg@8) | (Pass@8) |
| Frontier Models |  |  |  |  |  |
| Olmo-3-7B-RL-Zero-General | 56.5 | 51.9 | 25.7 | 28.9 | 69.0 |
| MiMo-Zero-RL-7B | 61.4 | 53.6 | 30.3 | 18.8 | 45.8 |
| General-Reasoner-7B | 65.6 | 63.6 | 25.9 | 35.1 | 68.6 |
| General-Reasoner-14B | 78.2 | 73.4 | 44.5 | 44.4 | 70.3 |
| OLMO3-7B-Base |  |  |  |  |  |
| $`\hookrightarrow`$ Zero-RL (PPO) | 46.8 | 37.9 | 22.2 | 21.9 | 62.1 |
|    $`\hookrightarrow`$ TTRL | 45.4 | 38.2 | 22.2 | 28.5 | 67.6 |
|    $`\hookrightarrow`$ EMPO | 52.9 | 50.2 | 23.5 | 27.7 | 61.6 |
|    $`\hookrightarrow`$ TEMPO | 68.2 | 62.4 | 35.1 | 32.4 | 69.4 |
|    Improvement via TTT | +21.4 | +24.5 | +12.9 | +10.5 | +7.3 |
| Qwen3-8B-Base |  |  |  |  |  |
| $`\hookrightarrow`$ Zero-RL (PPO) | 69.9 | 65.7 | 25.7 | 32.2 | 62.4 |
|    $`\hookrightarrow`$ TTRL | 74.9 | 68.5 | 31.7 | 41.1 | 73.0 |
|    $`\hookrightarrow`$ EMPO | 66.7 | 65.1 | 26.3 | 39.8 | 70.8 |
|    $`\hookrightarrow`$ TEMPO | 74.2 | 70.1 | 33.9 | 37.2 | 65.3 |
|    Improvement via TTT | +4.3 | +4.4 | +8.2 | +5.0 | +2.9 |

Table 2: Generalization to reasoning tasks beyond math-only domains. We
report Avg@1 for BigBenchHard, AGI Eval, and ZebraLogic, alongside Avg@8
and Pass@8 over 8 independent samples for GPQA-Diamond. We also
highlight the absolute Improvement via TTT of TEMPO over the Zero-RL
baseline across diverse domains.

### 4.5 Ablation (RQ4)

We conduct two ablation studies to isolate the contributions of key
design choices in TEMPO: (1) Frozen critic: the critic is trained once
on $`D_{L}`$ and kept fixed throughout all policy updates, removing the
E-step recalibration; (2) Supervised continuation: the model continues
training on the labeled dataset $`D_{L}`$ using standard PPO without any
test-time updates on unlabeled data.

Figure 5: The Superiority of Test-time training. Starting from a
converged OLMO3 model (192 PPO steps on DAPO-Math-17K), we compare
continuing supervised PPO on the same labeled data (blue) with TEMPO
test-time training on unlabeled questions (orange). Supervised PPO
saturates immediately with negligible gains, while TEMPO achieves a
steady 15+ point avg@16 accuracy improvement over 200 iterations,
confirming that test-time training on novel problems pushes the
performance boundary.

Figure 6: Necessity of alternating critic recalibration. We compare the
full TEMPO (orange) with a frozen-critic variant (blue) where the critic
is trained once on $`D_{L}`$ and never updated. The frozen critic
initially matches TEMPO but plateaus after ~100 iterations as it becomes
misaligned with the evolving policy, while the full model continues to
improve. This confirms that periodic E-step recalibration is essential
for sustained self-improvement.

As shown in
Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Ablation (RQ4) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
the supervised-only continuation (supervised PPO) quickly saturates and
yields negligible gains over 200 additional training steps, confirming
that the model has already converged on the labeled distribution. In
stark contrast, TEMPO exhibits a steady and consistent upward trajectory
from the same starting point. The widening gap between these two curves
grows from near-zero at step 0 to over 15 avg@16 accuracy points by step
200. This gap represents performance gains that are entirely
attributable to test-time training on unlabeled open questions. This
result demonstrates that once a model has converged on its supervised
training data, further optimization on the same distribution cannot
unlock additional capability. Only exposure to novel, challenging
test-time problems can push the model beyond its established boundaries.

Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Ablation (RQ4) ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models")
reveals the critical role of the alternating training design. The
frozen-critic variant initially matches the performance of the full
TEMPO, confirming that a well-calibrated critic can provide useful
signals in the early stages. However, as training progresses, its
improvement curve flattens and eventually plateaus, diverging sharply
from the full model’s sustained growth. This degradation occurs because
a static critic gradually becomes misaligned with the evolving policy:
as the actor generates increasingly sophisticated reasoning paths, the
frozen critic—trained on an earlier, less capable policy’s outputs—fails
to accurately evaluate these new patterns. The resulting mismatch
between the critic’s scores and the true correctness of responses
introduces noise into the policy gradient, ultimately stalling further
improvement. This ablation validates that the E-step critic
recalibration is not a mere implementation detail but a necessary
condition for sustained self-improvement: without periodic grounding on
labeled data, the critic’s evaluations drift, and the entire
self-training loop collapses.

## 5 Discussion

This section interprets representative TTT methods through the lens of
our EM framework, showing how TTRL and EMPO reduce to degenerate cases
that omit the E-step, and explaining why our dynamically calibrated
critic avoids the self-reinforcement trap.

#### A unified perspective on LRM test-time training.

The proposed TEMPO framework offers a principled
Expectation-Maximization (EM) interpretation of test-time training for
LRMs. By iteratively alternating between posterior estimation (E-step)
and policy optimization (M-step), TEMPO ensures that the Evidence Lower
Bound (ELBO) remains a tight surrogate for the true objective
$`J(\theta)`$, thereby preventing the optimization from diverging as the
model self-improves. This theoretical lens reveals a critical insight:
several representative test-time training methods, including EMPO and
TTRL, can be understood as heuristic and degenerate instances of the EM
algorithm. Specifically, these methods effectively execute only the
M-step using self-generated pseudo-labels for policy updates while
entirely neglecting the E-step that should calibrate the quality of
those labels.

To make this connection concrete, we reconsider how TTRL constructs its
training signal. In TTRL, the auxiliary distribution $`q(y|x)`$, which
in the EM framework should approximate the true posterior
$`P(y|x,\text{Correct})`$, is reduced to a binary indicator based on
majority consensus:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
q(y|x)\propto\mathbbm{1}(y\in\mathcal{Y}_{\rm majority})\cdot\pi_{\theta_{0}}(y|x),
``` |  | (13) |

where $`\mathbbm{1}(\cdot)`$ assigns unit weight to responses that agree
with the majority and zero weight to all others. This formulation has
two fundamental limitations. First, the consensus set
$`\mathcal{Y}_{\rm majority}`$ is determined solely by the model’s
current policy. Second, because the training signal is self-generated,
it becomes increasingly self-reinforcing: as the model grows more
confident in a particular reasoning pattern, that pattern dominates the
consensus, which in turn further amplifies the same pattern in
subsequent updates. This positive feedback loop is the root cause of the
performance plateaus and diversity collapse observed in TTRL.

In contrast, TEMPO addresses both limitations through a dynamically
calibrated critic $`V_{\phi}`$. Rather than a binary vote, the critic
provides a continuous, quality-aware score for each response, enabling
fine-grained differentiation among generated samples. More importantly,
because the critic is periodically recalibrated on labeled data during
the E-step, its evaluations remain grounded in external supervision
rather than drifting with the model’s own biases. This design ensures
that the ELBO stays tight throughout training, allowing TEMPO to sustain
meaningful self-improvement over hundreds of iterations without
succumbing to the self-reinforcement trap that limits prior methods.

## 6 Limitations

Despite its advantages, TEMPO has several limitations that warrant
discussion. First, the alternating E/M-step procedure requires
maintaining both an actor and a critic model, which increases GPU memory
and computational overhead compared to single-model TTT methods such as
TTRL. Second, the critic recalibration relies on access to a labeled
dataset $`D_{L}`$. And the size and distribution of $`D_{L}`$ may affect
how well the critic generalizes to out-of-domain test questions. Third,
our experiments are conducted on math, STEM, and puzzle reasoning tasks;
the applicability of TEMPO to other domains such as code generation
remains to be validated. Finally, while the EM perspective provides a
principled framing, our theoretical analysis does not include formal
convergence guarantees for the alternating optimization, which we leave
for future work.

## 7 Conclusion

We present TEMPO, a scalable test-time training framework for LRMs
through an alternating actor-critic optimization. By framing TTT as an
EM-style procedure, we identified the missing E-step, i.e., periodic
critic recalibration on labeled data as the key deficiency underlying
the performance plateaus and diversity collapse of prior baselines. Our
theoretical perspective unifies existing TTT approaches as incomplete
variants of the EM algorithm, and our empirical results demonstrate that
TEMPO consistently outperforms baselines across model scales and
reasoning domains while preserving output diversity. Future work will
explore formal convergence guarantees for the alternating procedure,
extend the framework to agentic tasks, and investigate the trade-off
between frequently-calibrated critic and computational efficiency.

## 8 Acknowledgement

This work is supported by Shanghai Artificial Intelligence Laboratory.
The authors thank the P1 team in Shanghai AI Lab for their extensive
support of this work, including computational resources, training
recipes and insightful discussions.

## References

- \[1\] G. Cui, L. Yuan, Z. Wang, H. Wang, W. Li, B. He, Y. Fan, T.
  Yu, Q. Xu, W. Chen, et al. (2025) Process reinforcement through
  implicit rewards. arXiv preprint arXiv:2502.01456. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[2\] J. Chen, Q. Cheng, F. Yu, H. Wan, Y. Zhang, S. Zheng, J. Yao, Q.
  Zhang, H. He, Y. Luo, et al. (2025) P1: mastering physics olympiads
  with reinforcement learning. arXiv preprint arXiv:2511.13612. Cited
  by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[3\] Y. Luo, F. Wang, Q. Cheng, F. Yu, H. Lei, J. Yan, C. Li, J.
  Chen, Y. Zhao, H. Wan, et al. (2026) P1-vl: bridging visual perception
  and scientific reasoning in physics olympiads. arXiv preprint
  arXiv:2602.09443. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[4\] Q. Zhang, H. Wu, C. Zhang, P. Zhao, and Y. Bian (2025) Right
  question is already half the answer: fully unsupervised llm reasoning
  incentivization. Advances in neural information processing systems.
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Test-time training for reasoning models. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[5\] Y. Zuo, K. Zhang, S. Qu, L. Sheng, X. Zhu, B. Qi, Y. Sun, G.
  Cui, N. Ding, and B. Zhou (2025) TTRL: test-time reinforcement
  learning. arXiv preprint arXiv:2504.16084. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Test-time training for reasoning models. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[6\] Y. Wang, S. Su, Z. Zeng, E. Xu, L. Ren, X. Yang, Z. Huang, X.
  He, L. Ma, B. Peng, H. Cheng, P. He, W. Chen, S. Wang, S. S. Du,
  and Y. Shen (2025) ThetaEvolve: test-time learning on open problems.
  arXiv preprint 2511.23473. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[7\] J. Jackson, B. Trapani, N. Wang, and W. Zhu (2026) Improving
  composer through real-time rl. Note:
  [https://cursor.com/blog/real-time-rl-for-composer](https://cursor.com/blog/real-time-rl-for-composer)Accessed:
  2026-04-12 Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[8\] Y. Zuo, B. He, Z. Liu, S. Zhao, Z. Fu, J. Yang, K. Zhang, Y.
  Fan, G. Cui, C. Qian, X. Chen, Y. Sun, X. Lv, X. Zhu, L. Sheng, R.
  Li, H. Gao, Y. Zhang, L. Yuan, Z. Liu, B. Zhou, and N. Ding (2026) How
  far can unsupervised RLVR scale LLM training?. In The Fourteenth
  International Conference on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=VesLZukY5E) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[9\] Y. Zhang, Z. Zhang, H. Guan, Y. Cheng, Y. Duan, C. Wang, Y.
  Wang, S. Zheng, and J. He (2025) No free lunch: rethinking internal
  feedback for llm reasoning. External Links: 2506.17219,
  [Link](https://arxiv.org/abs/2506.17219) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[10\] N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F.
  Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S.
  Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C.
  Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H.
  Hajishirzi (2024) Tulu 3: pushing frontiers in open language model
  post-training. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[11\] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S.
  Ma, P. Wang, X. Bi, et al. (2025) Deepseek-r1: incentivizing reasoning
  capability in llms via reinforcement learning. arXiv preprint
  arXiv:2501.12948. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[12\] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M.
  Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of
  mathematical reasoning in open language models. arXiv preprint
  arXiv:2402.03300. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[13\] Z. Gao, L. Chen, H. Luo, J. Zhou, and B. Dai (2025) One-shot
  entropy minimization. arXiv preprint arXiv:2505.20282. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[14\] X. Zhao, Z. Kang, A. Feng, S. Levine, and D. Song (2025)
  Learning to reason without external rewards. arXiv preprint
  arXiv:2505.19590. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Test-time training for reasoning models. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[15\] Y. Wang, B. Li, D. Cho, R. Zhang, F. Sui, and A. Grama (2026)
  SARL: label-free reinforcement learning by rewarding reasoning
  topology. arXiv preprint arXiv:2603.27977. Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[16\] W. Yang, W. Liu, R. Xie, Y. Guo, L. Wu, S. Yang, and Y.
  Lin (2026) LaSeR: reinforcement learning with last-token
  self-rewarding. In Internation Conference on Learning Representations,
  Cited by:
  [§2](#S2.SS0.SSS0.Px1.p1.1 "Self-rewarding reinforcement learning. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[17\] Y. Grandvalet and Y. Bengio (2004) Semi-supervised learning by
  entropy minimization. Advances in neural information processing
  systems 17. Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Test-time training for reasoning models. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[18\] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T.
  Darrell (2021) Tent: fully test-time adaptation by entropy
  minimization. In International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=uXl3bZLkr3c)
  Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Test-time training for reasoning models. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[19\] Q. Zhang, Y. Bian, X. Kong, P. Zhao, and C. Zhang (2025) COME:
  test-time adaption by conservatively minimizing entropy. In
  International Conference on Learning Representations, Cited by:
  [§2](#S2.SS0.SSS0.Px2.p1.1 "Test-time training for reasoning models. ‣ 2 Related Work ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[20\] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C.
  Gao, C. Huang, C. Lv, et al. (2025) Qwen3 technical report. arXiv
  preprint arXiv:2505.09388. External Links:
  [Link](https://arxiv.org/abs/2505.09388) Cited by: [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[21\] T. Olmo A. Ettinger et al. (2025) Olmo 3. arXiv preprint
  arXiv:2512.13961. External Links:
  [Link](https://arxiv.org/abs/2512.13961) Cited by: [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models"),
  [2nd
  item](#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[22\] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, T. Fan, G.
  Liu, L. Liu, X. Liu, et al. (2025) DAPO: an open-source llm
  reinforcement learning system at scale. arXiv preprint
  arXiv:2503.14476. Cited by: [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[23\] \[ByteDance-Seed\] (2025) BeyondAIME: advancing math reasoning
  evaluation beyond high school olympiads. Hugging Face. Note:
  [\[https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME\](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME)](https://%5Bhttps://huggingface.co/datasets/ByteDance-Seed/BeyondAIME%5D(https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME))
  Cited by: [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[24\] B. Zhang, H. Sun, Y. Min, Z. Chen, W. X. Zhao, Z. Liu, Z.
  Wang, L. Fang, and J. Wen (2025) Challenging the boundaries of
  reasoning: an olympiad-level math benchmark for large language models.
  arXiv preprint arXiv:2503.21380. External Links:
  [Link](https://arxiv.org/abs/2503.21380) Cited by: [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[25\] C. Zheng, S. Liu, M. Li, X. Chen, B. Yu, C. Gao, K. Dang, Y.
  Liu, R. Men, A. Yang, et al. (2025) Group sequence policy
  optimization. arXiv preprint arXiv:2507.18071. Cited by: [1st
  item](#S4.I1.i1.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[26\] M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W.
  Chung, A. Chowdhery, Q. Le, E. Chi, D. Zhou, et al. (2023) Challenging
  big-bench tasks and whether chain-of-thought can solve them. In
  Findings of the Association for Computational Linguistics: ACL 2023,
  pp. 13003–13051. Cited by: [2nd
  item](#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[27\] W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A.
  Saied, W. Chen, and N. Duan (2024) Agieval: a human-centric benchmark
  for evaluating foundation models. In Findings of the association for
  computational linguistics: NAACL 2024, pp. 2299–2314. Cited by: [2nd
  item](#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[28\] B. Y. Lin, R. Le Bras, K. Richardson, A. Sabharwal, R.
  Poovendran, P. Clark, and Y. Choi (2025) ZebraLogic: on the scaling
  limits of llms for logical reasoning. In International Conference on
  Machine Learning, pp. 37889–37905. Cited by: [2nd
  item](#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[29\] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J.
  Dirani, J. Michael, and S. R. Bowman (2024) Gpqa: a graduate-level
  google-proof q&a benchmark. In First Conference on Language Modeling,
  Cited by: [2nd
  item](#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
- \[30\] S. Agarwal, L. Ahmad, J. Ai, S. Altman, A. Applebaum, E.
  Arbus, R. K. Arora, Y. Bai, B. Baker, H. Bao, et al. (2025)
  Gpt-oss-120b & gpt-oss-20b model card. arXiv preprint
  arXiv:2508.10925. Cited by: [2nd
  item](#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ TEMPO: Scaling Test-time Training for Large Reasoning Models").
````
