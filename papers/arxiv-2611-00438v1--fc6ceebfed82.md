---
identifier: arxiv:2611.00438v1
title: "Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"
authors:
  - Tianwei Gao
  - Garrett G. Wen
  - Huiying Zhong
  - Lihua Lei
  - Weijie Su
  - Zhun Deng
published: "2026-10-30T22:45:28+00:00"
url: https://arxiv.org/abs/2611.00438v1
source: arxiv
doi: null
arxiv_id: 2611.00438v1
categories:
  - cs.LG
  - stat.ML
---

# Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems

Tianwei Gao Affiliation: University of North Carolina at Chapel Hill   
Garrett G. Wen Affiliation: Yale University    Huiying Zhong
Affiliation: Massachusetts Institute of Technology    Lihua Lei
Affiliation: Stanford University    Weijie Su Affiliation: University of
Pennsylvania    Zhun Deng Affiliation: University of North Carolina at
Chapel Hill

###### Abstract

Performativity arises when prediction-informed decisions change the
data-generating process, creating feedback between deployment and
learning. This phenomenon occurs in policy-making and in self-evolving
AI systems that shape their own future training data. Statistical
inference is challenging because estimation errors in earlier predictors
affect subsequent data distributions and propagate into later updates.
We develop an end-to-end framework for statistical inference under
performativity. For repeated risk minimization and repeated gradient
descent, we establish central limit theorems at fixed deployment rounds
and derive covariance recursions that capture both current-round
sampling variation and uncertainty inherited from earlier deployments.
We develop consistent covariance estimators using policy perturbations
and score matching. These results can yield asymptotically valid
confidence regions for population iterates and the performative stable
point. We further extend prediction-powered inference to this dynamic
setting, combining labeled observations with predictions on additional
unlabeled data to improve statistical efficiency. We evaluate our
framework in synthetic, semi-synthetic educational, and self-evolving
large language model routing settings. Our experiments demonstrate that
ignoring propagated uncertainty can lead to substantial undercoverage
even after repeated retraining. Together, these results provide
statistical tools for learning systems that help determine the
distributions from which they learn. Our work establish a complete
statistical inference under performativity, introducing new challenges
and inference settings that we believe will provide substantial value to
policy-making, statistics, and machine learning.

## 1 Introduction

Prediction-informed decisions can shape the very outcomes they seek to
predict. This phenomenon, known as performativity, arises when deploying
a predictive model changes the underlying data-generating process. For
example, lending policies based on predicted default risk can alter
borrowers’ consumption and repayment behavior, thereby influencing the
outcomes used to train future predictors.

The same feedback arises in self-evolving large language model (LLM)
systems, where the current model guides the generation and selection of
data for subsequent learning \[[Zelikman et al., 2022](#bib.bib1);
[Singh et al., 2023](#bib.bib2); [Wang et al., 2025](#bib.bib3);
[Zweiger et al., 2026](#bib.bib6)\]. By influencing which prompts and
responses are generated and retained, these systems participate in
producing their own supervision. Updating the model can therefore change
not only its predictions but also the distribution of its future
training data.

In both settings, deployment and learning form a feedback loop: the
deployed predictor shapes the data collected for retraining, and the
updated predictor can induce a further distributional shift. This raises
a statistical question that goes beyond prediction accuracy and
convergence: how can we quantify uncertainty in predictors along this
evolving sequence and in the equilibrium they may approach?

The framework of performative prediction formalizes this dependence
between predictive models and the distributions they induce \[[Perdomo
et al., 2020](#bib.bib12); [Lin and Zrnic, 2023](#bib.bib5); [Perdomo,
2025](#bib.bib41)\]. For a model parameter $`\theta`$ in a parameter
space $`\Theta`$, the performative risk is defined as

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
\operatorname{PR}(\theta):=\mathbb{E}_{z\sim\mathcal{D}(\theta)}\bigl[\ell(z;\theta)\bigr],
``` |  | (1) |

where $`z=(x,y)\in\mathcal{X}\times\mathcal{Y}`$ is an input-output
pair, $`\mathcal{D}(\theta)`$ is the distribution induced by deploying
$`\theta`$, and $`\ell(z;\theta)`$ is the loss incurred on observation
$`z`$. Finding a performatively optimal point
$`\theta_{\text{PO}}\in\arg\min_{\theta\in\Theta}\operatorname{PR}(\theta)`$
is challenging: the distribution map $`\mathcal{D}(\cdot)`$ is generally
unknown, and the performative risk can be nonconvex even when the loss
is convex in $`\theta`$ \[[Miller et al., 2021](#bib.bib42)\]. One
approach is to estimate this map within a specified parametric family
and optimize the resulting plug-in performative risk \[[Lin and Zrnic,
2023](#bib.bib5)\]. This approach requires an explicit model of the
deployment response, and its optimality guarantees depend on how well
the chosen family captures that response.

A natural alternative is a performative stable point \[[Perdomo et al.,
2020](#bib.bib12)\], which satisfies

|  |  |  |
|----|----|----|
|  |
``` math
\theta_{\text{PS}}\in\arg\min_{\theta\in\Theta}\mathbb{E}_{z\sim\mathcal{D}(\theta_{\text{PS}})}\bigl[\ell(z;\theta)\bigr].
``` |  |

At this equilibrium, the predictor minimizes risk under the distribution
induced by its own deployment, so further population-level retraining
leaves it unchanged. Under suitable regularity conditions, the stable
point is unique and can be approached through repeated retraining
without explicitly modeling the distribution map. When
deployment-induced shifts are sufficiently mild, it is also close to a
performatively optimal point, making it a useful proxy for the
Stackelberg equilibrium in strategic classification. These properties
make performative stability a meaningful target for both policy-making
and self-evolving systems whose deployment shapes subsequent
supervision.

Convergence to a stable point, however, does not by itself quantify
uncertainty. In policy-making, $`\theta`$ often represents an actionable
quantity, such as a tax rate or a credit-score cutoff. Confidence
regions and hypothesis tests are therefore important complements to
point estimates and convergence guarantees. More generally, statistical
inference is needed both for the population predictors at intermediate
deployment rounds and for the equilibrium they approach.

The central challenge is that uncertainty itself propagates through the
feedback loop. Estimation errors in earlier predictors affect subsequent
data distributions and, consequently, the variability of later updates.
Accounting only for the sampling uncertainty within the current round
can therefore miss an important source of variation. Our analysis
characterizes this propagation through the distributional score, which
measures the sensitivity of the log density of $`\mathcal{D}(\theta)`$
to the deployed parameter $`\theta`$. Estimating this sensitivity from
data is a key step toward constructing valid confidence regions under
performative feedback.

Previous work on performative prediction has largely emphasized
prediction performance and convergence analysis. A closely related
contribution by [Cutler et al. \[2024\]](#bib.bib43) establishes
asymptotic normality for averaged stochastic gradient iterates as the
number of updates tends to infinity. This regime differs from
applications in which only a limited number of costly deployments may be
feasible, such as policy-making. Moreover, their results do not provide
a data-driven procedure for estimating the covariance required for
inference. Our focus is instead on inference at fixed deployment rounds
as the sample size within each round grows, together with covariance
estimation that makes the resulting procedures implementable.

Our preliminary NeurIPS paper \[[Li et al., 2025](#bib.bib4)\]
established, to the best of our knowledge, the first end-to-end
statistical inference framework for repeated risk minimization (RRM)
under performativity and introduced prediction-powered inference (PPI)
into this setting. The present work extends this framework to repeated
gradient descent (RGD) and develops applications to self-evolving LLM
systems.

##### Our contributions.

We develop a framework for statistical inference that accounts for the
dependence between estimation, deployment, and subsequent data
collection. Our contributions are threefold.

1.  1.
    Statistical inference for performative learning trajectories and
    stable points. We study RRM and RGD, two iterative procedures for
    approaching $`\theta_{\text{PS}}`$ (Section
    [2](#S2 "2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
    and establish central limit theorems for their empirical iterates at
    each fixed deployment round. The limiting covariance recursions
    account for both current-round sampling variation and uncertainty
    propagated from earlier deployments. When the distributional score
    is unknown, we estimate it through policy perturbations and score
    matching, and establish consistency of the resulting covariance
    estimator under the stated regularity and score-model conditions
    (Section
    [3.2](#S3.SS2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).
    This approach estimates the sensitivity needed for inference
    directly, without fitting a full parametric distribution map. The
    resulting estimators yield asymptotically valid confidence regions
    for the population iterates and, after accounting for the remaining
    iteration bias, for the performative stable point.
2.  2.
    Prediction-powered inference under performativity. We extend PPI
    \[[Angelopoulos et al., 2023a](#bib.bib14)\] to the dynamic
    performative setting by combining labeled observations with
    predictions on additional unlabeled data (Appendix
    [A](#A1 "Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).
    We derive the corresponding central limit theorem and covariance
    recursion along the RRM trajectory, and select the PPI weight at
    each round to minimize the trace of the asymptotic covariance with
    earlier weights held fixed. This provides a data-driven way to
    improve statistical efficiency while accounting for
    deployment-induced feedback. Such gains are particularly relevant
    when reliable labeled feedback is difficult to collect, as in
    survey-based applications \[[Huang et al., 2015](#bib.bib16)\].
3.  3.
    Evaluation across policy feedback and self-evolving LLM systems.

    We evaluate our framework in three settings with distinct feedback
    mechanisms (Section
    [5](#S5 "5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")):
    a synthetic model in which deployment changes the population
    composition; a semi-synthetic educational setting grounded in a
    study of growth-mindset interventions \[[Yeager et al.,
    2019](#bib.bib7); [Athey and Wager, 2019](#bib.bib9)\], where
    predictions guide interventions that affect subsequent student
    outcomes; and a self-evolving LLM routing task. Importantly, in the
    third setting, we formalize a connection between performative
    prediction and self-evolving AI by modeling how the deployed router
    determines which responses are sampled and thereby shapes the
    supervision used for its own subsequent updates. This formulation
    enables statistical inference for the router’s population learning
    trajectory and performative stable point. Across these settings,
    conventional confidence regions for a one-step empirical risk
    minimizer can substantially undercover the performative stable point
    despite achieving near-nominal coverage of their own target. A
    controlled synthetic study with stronger feedback further
    demonstrates that repeated retraining alone does not ensure
    calibrated inference: covariance estimation must also account for
    uncertainty propagated across deployments.

Together, these results extend statistical inference from models trained
on a fixed distribution to learning procedures that help determine the
distributions from which they learn. The framework connects uncertainty
quantification for policy feedback with inference for self-evolving AI
systems, while enabling more efficient use of limited labeled data
through prediction-powered inference.

### 1.1 Related Work

##### Performative prediction.

Performativity describes the phenomenon whereby predictions influence
the outcomes they aim to predict. [Perdomo et al. \[2020\]](#bib.bib12)
were the first to formalize performative prediction in the
supervised-learning setting; their work, along with the majority of
subsequent papers \[[Mendler-Dünner et al., 2020](#bib.bib35);
[Drusvyatskiy and Xiao, 2020](#bib.bib36); [Mofakhami et al.,
2023](#bib.bib37); [Oesterheld et al., 2023](#bib.bib38); [Taori and
Hashimoto, 2022](#bib.bib39); [Khorsandi et al., 2024](#bib.bib40);
[Perdomo, 2025](#bib.bib41); [Bracale et al., 2024](#bib.bib13)\], have
been focused on performative stability and proposed algorithms for
learning performative stable parameters. On the other hand, performative
optimality requires much stricter conditions (e.g. distributional
assumptions to ensure the convexity of $`\text{PR}(\theta)`$) than
performative stability, a few papers address the problem of finding
performative optimal parameters. [Miller et al. \[2021\]](#bib.bib42)
introduce a two-stage method that learns a distribution map to locate
the performative optimal parameter. [Kim and Perdomo
\[2022\]](#bib.bib33) study performative optimality in outcome-only
performative settings. Finally, [Hardt and Mendler-Dünner
\[2023\]](#bib.bib34) provide a comprehensive overview of learning
algorithms, optimization methods, and applications for performative
prediction. Unlike these prior works on performative prediction, which
focus on prediction accuracy, our work is dedicated to constructing
powerful and statistically valid inference procedures under the
performative framework.

##### Prediction-powered inference.

[Angelopoulos et al. \[2023b\]](#bib.bib24) first introduced the
prediction-powered inference (PPI) framework, which leverages black-box
machine learning models to construct valid confidence intervals (CIs)
for statistical quantities. Since then, PPI has been extended and
applied in various settings. Closely related to our strategies,
[Angelopoulos et al. \[2023a\]](#bib.bib14) propose PPI++, a more
computationally efficient procedure that enhances predictability by
accommodating a wider range of models on unlabeled data, while
guaranteeing performance (e.g. CI width) no worse than that of classical
inference methods. Other extensions include Stratified PPI \[[Fisch et
al., 2024](#bib.bib25)\], which incorporates simple data stratification
strategies into basic PPI estimates; Cross PPI \[[Zrnic and Candès,
2023](#bib.bib26)\], which obtains confidence intervals with
significantly lower variability by including model training; Bayesian
PPI \[[Hofer et al., 2024](#bib.bib27)\] and FAB-PPI \[[Cortinovis and
Caron, 2025](#bib.bib28)\], which propose frameworks for PPI based on
Bayesian inference. PPI is also connected to topics such as
semi-parametric inference and missing-data imputation \[[Tsiatis,
2006](#bib.bib32); [Robins and Rotnitzky, 1995](#bib.bib29); [Demirtas,
2018](#bib.bib30); [Song et al., 2023](#bib.bib31)\]. Our work is the
first one to study PPI under performativity, and we validate the PPI
framework in the performative setting both theoretically and
empirically.

##### Inference in performativity.

[Bracale et al. \[2024\]](#bib.bib13) study identifiability and
estimation error under a specific microfoundation model with
performativity. Yet it doesn’t address confidence interval construction.
A closely related work is \[[Cutler et al., 2024](#bib.bib43)\], which
also establishes the asymptotic normality and minimax optimality for
performative settings. It focused on averaged stochastic gradient
iterates under performativity as time $`t\to\infty`$, and did not
provide fixed round central limit theorem. Moreover, \[[Cutler et al.,
2024](#bib.bib43)\] do not provide a data-driven approach for covariance
estimation and thus lacks an end-to-end inference framework. In
contrast, our work explicitly handles score function estimation and
provides an end-to-end inference method for constructing confidence
intervals, which is firstly addressed by our preliminary article \[[Li
et al., 2025](#bib.bib4)\], to the best of our knowledge.

## 2 Background and Setup

We consider the setting in which the current parameter changes the
distribution of the next training run. Let $`\Theta=\mathbb R^{d}`$, let
$`\mathcal{D}(\theta)`$ denote the distribution induced by parameter
$`\theta\in\Theta`$, and let $`\ell(z;\theta)`$ denote the loss of a
candidate parameter $`\theta`$ on an observation $`z`$, with gradient
$`\nabla\ell(z;\cdot)`$ and Hessian $`\nabla^{2}\ell(z;\cdot)`$ in its
parameter argument.

Let $`\theta_{0}\in\Theta`$ be fixed and set
$`\widehat{\theta}_{0}=\theta_{0}`$. At round $`t\geqslant 0`$,
conditional on all observations and estimators constructed before that
round, the labeled observations $`Z_{t,1},\ldots,Z_{t,n_{t}}`$ are
i.i.d. from $`\mathcal{D}(\widehat{\theta}_{t})`$. At round $`t`$,
$`\widehat{\theta}_{t}`$ therefore determines the sampling distribution,
and batch $`t`$ is used to construct $`\widehat{\theta}_{t+1}`$.

### 2.1 Repeated Risk Minimization

Define the population update
$`G(\widetilde{\theta}):=\arg\min_{\theta\in\Theta}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\ell(Z;\theta)].`$
Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the minimizer is unique. Recall that the main objective of interest is
the performative stable point, which satisfies the following
relationship:

|  |  |  |
|----|----|----|
|  |
``` math
\theta_{\mathrm{PS}}=G(\theta_{\mathrm{PS}})=\arg\min_{\theta\in\Theta}\mathbb E_{Z\sim\mathcal{D}(\theta_{\mathrm{PS}})}[\ell(Z;\theta)].
``` |  |

Repeated risk minimization (RRM) is a simple algorithm that can
efficiently find $`\theta_{\text{PS}}`$. Specifically, one starts with
an arbitrary $`\theta_{0}`$ and repeats the procedure
$`\theta_{t+1}=G(\theta_{t}),t\geqslant 0,`$ for $`t\in\mathbb{N}`$.
Under some regularity conditions, this update is well-defined and
provably converges to a unique $`\theta_{\text{PS}}`$ at a linear rate.

###### Theorem 2.1 (Informal, adopted from \[[Perdomo et al., 2020](#bib.bib12)\]).

If the loss is smooth, strongly convex, and the mapping
$`{\cal D}(\cdot)`$ satisfies certain Lipchitz conditions, then:

(a). For any $`\theta^{\prime}\in\Theta`$, $`G(\theta^{\prime})`$ is
unique and well-defined, and $`G:\Theta\to\Theta`$ is a contraction map.

(b). $`\theta_{\text{PS}}`$ is uniquely defined as the fixed point of
$`G`$, and repeated risk minimization converges to
$`\theta_{\text{PS}}`$ at a linear rate.

##### Repeated empirical risk minimization (RERM).

At round $`t`$, repeated empirical risk minimization computes

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\theta}_{t+1}=\arg\min_{\theta\in\Theta}\frac{1}{n_{t}}\sumop\displaylimits_{i=1}^{n_{t}}\ell(Z_{t,i};\theta).
``` |  |

Conditional on the preceding observations and estimators, this empirical
minimizer estimates $`G(\widehat{\theta}_{t})`$, whereas the
corresponding deterministic population iterate is
$`\theta_{t+1}=G(\theta_{t})`$. This gives the population trajectory
$`\theta_{0}\to\theta_{1}\to\cdots`$ and the empirical trajectory
$`\theta_{0}\to\widehat{\theta}_{1}\to\cdots`$. Their difference
satisfies
$`\widehat{\theta}_{t+1}-\theta_{t+1}=\bigl(\widehat{\theta}_{t+1}-G(\widehat{\theta}_{t})\bigr)+\bigl(G(\widehat{\theta}_{t})-G(\theta_{t})\bigr),`$
where the first term represents sampling error in the current empirical
minimization. The second represents the effect of the preceding
estimation error on the population update. Our inference results account
for both contributions when characterizing the asymptotic distribution
of $`\widehat{\theta}_{t+1}-\theta_{t+1}`$.

### 2.2 Repeated Gradient Descent

Repeated gradient descent (RGD) uses the population gradient under the
distribution induced by the current parameter \[[Perdomo et al.,
2020](#bib.bib12)\] :

|  |  |  |
|----|----|----|
|  |
``` math
\theta_{t+1}=\theta_{t}-\alpha\mathbb E_{Z\sim\mathcal{D}(\theta_{t})}\left[\nabla\ell(Z;\theta_{t})\right],\qquad t\geqslant 0,
``` |  |

where $`\alpha>0`$ is the learning rate.

###### Theorem 2.2 (Informal, adopted from \[[Perdomo et al., 2020](#bib.bib12)\]).

If the loss is smooth, strongly convex, and the mapping
$`{\cal D}(\cdot)`$ satisfies certain Lipchitz conditions, then
$`\theta_{\text{PS}}`$ is uniquely defined and repeated gradient descent
converges to $`\theta_{\text{PS}}`$ at a linear rate.

##### Repeated empirical gradient descent (REGD).

Starting from $`\widehat{\theta}_{0}=\theta_{0}`$, repeated empirical
gradient descent updates

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\theta}_{t+1}=\widehat{\theta}_{t}-\frac{\alpha}{n_{t}}\sumop\displaylimits_{i=1}^{n_{t}}\nabla\ell(Z_{t,i};\widehat{\theta}_{t}).
``` |  |

We use $`\{\theta_{t}\}_{t=0}^{T}`$ and
$`\{\widehat{\theta}_{t}\}_{t=0}^{T}`$ for the population and empirical
RGD trajectories. For both RERM and REGD, we study inference on
$`\widehat{\theta}_{t}`$ at each fixed $`t\geqslant 1`$, and we let
$`n_{t}=n`$ for all rounds.

### 2.3 Assumptions and Notation

We introduce the following foundational assumptions for our setup. We
assume that $`{\cal Z}`$ is a subset of Euclidean space with norm
$`\|\cdot\|_{2}`$.

###### Assumption 2.3 (Loss and distribution map).

The following conditions hold.

\(a\) (Local Boundedness) For every compact $`K\subset\Theta`$, there
exists a measurable $`A_{K}:\mathcal{Z}\to[0,\infty)`$ such that
$`|\ell(z;\theta)|\leqslant A_{K}(z),\theta\in K,`$ and
$`\sup_{\widetilde{\theta}\in K}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[A_{K}(Z)^{2}]<\infty.`$

\(b\) (Local Lipschitzness) For every compact $`K\subset\Theta`$, there
exists a measurable $`L_{K}:\mathcal{Z}\to[0,\infty)`$ such that
$`|\ell(z;\theta)-\ell(z;\theta^{\prime})|\leqslant L_{K}(z)\|\theta-\theta^{\prime}\|_{2},\theta,\theta^{\prime}\in K,`$
and
$`\sup_{\widetilde{\theta}\in K}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[L_{K}(Z)^{3}]<\infty.`$

\(c\) (Joint Smoothness) The loss is twice continuously differentiable
in its second argument, and for all $`z,z^{\prime}\in\mathcal{Z}`$ and
$`\theta,\theta^{\prime}\in\Theta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\|\nabla\ell(z;\theta)-\nabla\ell(z;\theta^{\prime})\|_{2}\leqslant\beta\|\theta-\theta^{\prime}\|_{2},\qquad\|\nabla\ell(z;\theta)-\nabla\ell(z^{\prime};\theta)\|_{2}\leqslant\beta\|z-z^{\prime}\|_{2}.
``` |  |

\(d\) (Strong Convexity) For every $`z\in\mathcal{Z}`$, the map
$`\theta\mapsto\ell(z;\theta)`$ is $`\gamma`$-strongly convex: for all
$`\theta,\theta^{\prime}\in\Theta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\ell(z;\theta)\geqslant\ell(z;\theta^{\prime})+\nabla\ell(z;\theta^{\prime})^{\top}(\theta-\theta^{\prime})+\frac{\gamma}{2}\|\theta-\theta^{\prime}\|_{2}^{2}.
``` |  |

\(e\) ($`\varepsilon`$-Sensitivity) The distribution map is
$`\varepsilon`$-sensitive:
$`W_{1}\{\mathcal{D}(\theta),\mathcal{D}(\theta^{\prime})\}\leqslant\varepsilon\|\theta-\theta^{\prime}\|_{2},\theta,\theta^{\prime}\in\Theta.`$

Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[Perdomo et al. \[2020, Theorem 3.5\]](#bib.bib12) gives
$`\|G(\theta)-G(\theta^{\prime})\|_{2}\leqslant\frac{\varepsilon\beta}{\gamma}\|\theta-\theta^{\prime}\|_{2}.`$
Hence $`G`$ has a unique fixed point when $`\varepsilon\beta<\gamma`$.

###### Assumption 2.4 (Differentiability of the distribution map).

\(a\) Each $`\mathcal{D}(\theta)`$ has a positive density
$`p(\cdot;\theta)`$ with respect to a common dominating measure. The map
$`\theta\mapsto\log p(z;\theta)`$ is differentiable. For every compact
$`K\subset\Theta`$, there exists $`L_{s,K}<\infty`$ such that
$`\sup_{\theta\in K}\|\nabla_{\theta}\log p(z;\theta)\|_{2}\leqslant L_{s,K},z\in\mathcal{Z}.`$

\(b\) The map
$`(\widetilde{\theta},\theta)\longmapsto\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;\theta)]`$
is continuously differentiable.

##### Notation

For $`K\in\mathbb N_{+}`$, let $`[K]=\{1,\ldots,K\}`$. We write
$`\xrightarrow{P}`$ and $`\xrightarrow{d}`$ for convergence in
probability and in distribution. For sets $`A,B\subset\mathbb R^{d}`$,
let $`A+B=\{a+b:a\in A,b\in B\}`$. We write $`\mathcal{N}(\mu,\Sigma)`$
for a Gaussian distribution, $`I_{d}`$ for the $`d\times d`$ identity
matrix, $`\mathcal{B}(c,r)`$ for the Euclidean ball centered at $`c`$
with radius $`r`$. For a differentiable map
$`F:\mathbb R^{d}\to\mathbb R^{d}`$, $`\nabla F`$ denotes its Jacobian
matrix; for a matrix $`A`$, $`\|A\|_{2}`$ denotes its spectral norm.

## 3 Statistical Inference for RRM

This section develops inference for the population iterate
$`\theta_{t}`$ based on the RERM iterate $`\widehat{\theta}_{t}`$, with
$`t`$ fixed and $`n\to\infty`$. Because batch $`t`$ is drawn from
$`\mathcal{D}(\widehat{\theta}_{t})`$, the error of
$`\widehat{\theta}_{t+1}`$ contains both the sampling error of the
current round and the error inherited from earlier rounds. Section
[3.1](#S3.SS1 "3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives the resulting central limit theorem, Section
[3.2](#S3.SS2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
estimates its covariance, and Section
[3.3](#S3.SS3 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
turns the confidence region for $`\theta_{t}`$ into one for
$`\theta_{\mathrm{PS}}`$.

### 3.1 Central Limit Theorem of RRM

We start from the decomposition in Section
[2.1](#S2.SS1 "2.1 Repeated Risk Minimization ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\widehat{\theta}_{t+1}-\theta_{t+1}=\bigl(\widehat{\theta}_{t+1}-G(\widehat{\theta}_{t})\bigr)+\bigl(G(\widehat{\theta}_{t})-G(\theta_{t})\bigr).`$
The first term is the error of an M-estimator. Conditional on all
preceding observations and estimators, $`\widehat{\theta}_{t+1}`$
minimizes the empirical risk of $`n`$ i.i.d. observations from
$`\mathcal{D}(\widehat{\theta}_{t})`$, and $`G(\widehat{\theta}_{t})`$
minimizes the corresponding population risk. For
$`\widetilde{\theta},\theta\in\Theta`$, define

|  |  |  |
|----|----|----|
|  |
``` math
H_{\widetilde{\theta}}(\theta):=\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla^{2}\ell(Z;\theta)],\qquad\Sigma_{\widetilde{\theta}}(\theta):=H_{\widetilde{\theta}}(\theta)^{-1}\operatorname{Cov}_{Z\sim\mathcal{D}(\widetilde{\theta})}\bigl(\nabla\ell(Z;\theta)\bigr)H_{\widetilde{\theta}}(\theta)^{-1}.
``` |  |

By standard results for M-estimators, the conditional distribution of
$`\sqrt{n}\{\widehat{\theta}_{t+1}-G(\widehat{\theta}_{t})\}`$ is
approximately
$`\mathcal{N}\bigl(0,\Sigma_{\widehat{\theta}_{t}}(G(\widehat{\theta}_{t}))\bigr)`$,
and $`\Sigma_{\widehat{\theta}_{t}}(G(\widehat{\theta}_{t}))`$ tends to
$`\Sigma_{\theta_{t}}(\theta_{t+1})`$ as
$`\widehat{\theta}_{t}\to\theta_{t}`$. The second term is approximately
$`\nabla G(\theta_{t})(\widehat{\theta}_{t}-\theta_{t})`$, where
$`\nabla G`$ is given by the following proposition.

###### Proposition 3.1.

Under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`G`$ is continuously differentiable, and

|  |  |  |
|----|----|----|
|  |
``` math
\nabla G(\theta)=-H_{\theta}(G(\theta))^{-1}\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\nabla\ell(Z;G(\theta))\,\nabla_{\theta}\log p(Z;\theta)^{\top}\bigr].
``` |  |

The expectation measures how the gradient of the risk at $`G(\theta)`$
changes when the distribution moves with $`\theta`$.

###### Theorem 3.2 (Central limit theorem for RERM).

Under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each fixed $`t\geqslant 1`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\mathcal{N}(0,V_{t}),
``` |  |

where $`V_{0}=0`$ and
$`V_{t+1}=\nabla G(\theta_{t})\,V_{t}\,\nabla G(\theta_{t})^{\top}+\Sigma_{\theta_{t}}(\theta_{t+1}),t\geqslant 0.`$

If $`\mathcal{D}(\theta)`$ does not depend on $`\theta`$, then
$`\nabla G=0`$ and $`V_{t}=\Sigma_{\theta_{t-1}}(\theta_{t})`$, the
covariance of a single M-estimator. Otherwise the inherited term
$`\nabla G(\theta_{t})V_{t}\nabla G(\theta_{t})^{\top}`$ is positive
semidefinite, so treating each round as a separate M-estimation problem
understates the uncertainty of $`\widehat{\theta}_{t+1}`$.

### 3.2 Estimation of the Asymptotic Covariance

The matrices $`H_{\theta_{t}}(\theta_{t+1})`$ and
$`\Sigma_{\theta_{t}}(\theta_{t+1})`$ are defined under
$`\mathcal{D}(\theta_{t})`$ and can be estimated from batch $`t`$. The
derivative $`\nabla G(\theta_{t})`$ also involves the score
$`\nabla_{\theta}\log p(z;\theta)`$, which describes how the
distribution changes with the parameter. When this score is known, as in
the routing experiment of Section
[5.3](#S5.SS3 "5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we use it directly in the estimator below. Otherwise, we estimate it by
score matching.

Let $`s(z,\theta;\psi)\in\mathbb R^{d}`$, with coordinates $`s_{j}`$, be
a model for $`\nabla_{\theta}\log p(z;\theta)`$ indexed by
$`\psi\in\Psi`$, where $`\Psi\subset{\mathbb R}^{p}`$ is a subset of
$`p`$-dimensional Euclidean space. We choose $`\psi`$ to minimize

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\|\nabla_{\theta}\log p(Z;\theta)-s(Z,\theta;\psi)\|_{2}^{2}\bigr].
``` |  |

After expanding the square, the term
$`\mathbb E_{Z\sim\mathcal{D}(\theta)}[\|\nabla_{\theta}\log p(Z;\theta)\|_{2}^{2}]`$
does not depend on $`\psi`$, but the cross term involves the unknown
score. [Hyvärinen and Dayan \[2005\]](#bib.bib23) removes such a term
for the score in $`z`$ by integrating by parts in $`z`$. Here the score
is taken in $`\theta`$, so we instead move the derivative outside the
expectation.

###### Lemma 3.3.

Under Assumption
[B.10](#A2.Thmtheorem10 "Assumption B.10. ‣ B.1.6 Regularity for Score Matching ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\nabla_{\theta}\log p(Z;\theta)^{\top}s(Z,\theta;\psi)\bigr]=\sumop\displaylimits_{j=1}^{d}\Bigl\{\frac{\partial}{\partial\theta^{(j)}}\mathbb E_{Z\sim\mathcal{D}(\theta)}[s_{j}(Z,\theta;\psi)]-\mathbb E_{Z\sim\mathcal{D}(\theta)}[\partial_{\theta^{(j)}}s_{j}(Z,\theta;\psi)]\Bigr\},
``` |  |

where $`\theta^{(j)}`$ is the $`j`$th coordinate of $`\theta`$.

In the first term, the derivative acts on both the distribution and the
argument of $`s_{j}`$; the second term removes the latter. Only the
first term requires observations at parameters other than $`\theta`$.

We estimate this derivative by policy perturbation \[[Munro et al.,
2021](#bib.bib17); [Viviano and Rudder, 2024](#bib.bib18)\]. At round
$`t`$, in addition to batch $`t`$, we use the parameters
$`\widehat{\theta}_{t}+\eta e_{j}`$, $`j\in[d]`$, where $`\eta>0`$ and
$`e_{j}`$ is the $`j`$th standard basis vector. Conditional on all
preceding observations and estimators,
$`Z^{(j)}_{t,1},\ldots,Z^{(j)}_{t,k}`$ are i.i.d. from
$`\mathcal{D}(\widehat{\theta}_{t}+\eta e_{j})`$, independently across
$`j`$ and of batch $`t`$.

We replace the expectations in Lemma
[3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
by sample averages and the derivative by a finite difference, and drop
the term that does not depend on $`\psi`$. Let
$`\widehat{\psi}_{n,k}(\widehat{\theta}_{t})`$ be an approximate
minimizer over $`\psi\in\Psi`$ of

|  |  |  |  |  |
|----|----|----|----|----|
|  |  | $`\displaystyle\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\Bigl\{\|s(Z_{t,i},\widehat{\theta}_{t};\psi)\|_{2}^{2}+2\sumop\displaylimits_{j=1}^{d}\partial_{\theta^{(j)}}s_{j}(Z_{t,i},\widehat{\theta}_{t};\psi)\Bigr\}`$ |  | (2) |
|  |  | $`\displaystyle-\frac{2}{\eta}\sumop\displaylimits_{j=1}^{d}\Bigl\{\frac{1}{k}\sumop\displaylimits_{m=1}^{k}s_{j}(Z^{(j)}_{t,m},\widehat{\theta}_{t}+\eta e_{j};\psi)-\frac{1}{n}\sumop\displaylimits_{i=1}^{n}s_{j}(Z_{t,i},\widehat{\theta}_{t};\psi)\Bigr\},`$ |  |  |

with optimization error $`a_{n,k}`$ as in Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d).

Using batch $`t`$, which also gives $`\widehat{\theta}_{t+1}`$, define

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla G}_{t}:=-\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t+1})\,s\bigl(Z_{t,i},\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr)^{\top},
``` |  |

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\Sigma}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1}):=\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\Bigl\{\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t+1})\nabla\ell(Z_{t,i};\widehat{\theta}_{t+1})^{\top}\Bigr\}\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1},
``` |  |

where
$`\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1}):=n^{-1}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{t,i};\widehat{\theta}_{t+1})`$.
Since the gradients average to zero at $`\widehat{\theta}_{t+1}`$, the
middle factor of $`\widehat{\Sigma}`$ is their sample covariance.
Following Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
set $`\widehat{V}_{0}=0`$ and

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\widehat{V}_{t+1}:=\widehat{\nabla G}_{t}\,\widehat{V}_{t}\,\widehat{\nabla G}_{t}^{\top}+\widehat{\Sigma}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1}),\qquad t\geqslant 0.
``` |  | (3) |

The proof decomposes the error into empirical and population terms.
Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
controls the differences between $`\theta_{k}`$, $`\theta_{k+1}`$ and
$`\widehat{\theta}_{k}`$, $`\widehat{\theta}_{k+1}`$. Theorem
[B.15](#A2.Thmtheorem15 "Theorem B.15. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
controls the score error uniformly over the perturbed parameters. Taken
together, the above result shows the consistency of
$`\widehat{\nabla G}_{t}`$ obtained by policy perturbation.

The next result requires the score model to contain the true score and
the fitted parameters to stay bounded (Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
together with moment and smoothness conditions on $`H`$ and on products
of loss gradients and scores (Assumptions
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.13](#A2.Thmtheorem13 "Assumption B.13. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).

###### Theorem 3.4.

Suppose Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[B.10](#A2.Thmtheorem10 "Assumption B.10. ‣ B.1.6 Regularity for Score Matching ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
and
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")–[B.13](#A2.Thmtheorem13 "Assumption B.13. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold. If $`\eta\to 0`$ and $`\eta\sqrt{\min(n,k)}\to\infty`$, then for
each fixed $`t\geqslant 0`$,

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t}-\nabla G(\theta_{t})\|_{2}^{2}=O_{P}\Bigl(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta+a_{n,k}\Bigr).
``` |  |

The four terms are the sampling error of batch $`t`$, the sampling error
of the perturbed batches divided by $`\eta`$, the bias of the finite
difference, and the optimization error.

###### Corollary 3.5.

Under the conditions of Theorem
[3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\widehat{V}_{t}\xrightarrow{P}V_{t}`$ for each fixed $`t\geqslant 1`$.
If, in addition, $`V_{t}`$ is nonsingular, then the region

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\mathcal{R}}_{t}:=\bigl\{\theta\in\Theta:n(\widehat{\theta}_{t}-\theta)^{\top}\widehat{V}_{t}^{-1}(\widehat{\theta}_{t}-\theta)\leqslant\chi^{2}_{d,1-\delta}\bigr\}
``` |  |

satisfies
$`\lim_{n\to\infty}\mathbb P(\theta_{t}\in\widehat{\mathcal{R}}_{t})=1-\delta`$,
where $`\chi^{2}_{d,1-\delta}`$ is the $`(1-\delta)`$-quantile of the
$`\chi^{2}_{d}`$ distribution.

### 3.3 Confidence Regions for the Performative Stable Point

The region $`\widehat{\mathcal{R}}_{t}`$ covers the population iterate
$`\theta_{t}`$, not $`\theta_{\mathrm{PS}}`$. When
$`\varepsilon\beta<\gamma`$, the contraction of $`G`$ in Section
[2.3](#S2.SS3 "2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives
$`\|\theta_{t}-\theta_{\mathrm{PS}}\|_{2}\leqslant\Bigl(\frac{\varepsilon\beta}{\gamma}\Bigr)^{t}\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}.`$
Enlarging $`\widehat{\mathcal{R}}_{t}`$ by this deterministic bias gives
a bias-aware confidence region \[[Armstrong and Kolesár,
2018](#bib.bib19); [Armstrong et al., 2020](#bib.bib21); [Imbens and
Wager, 2019](#bib.bib22); [Noack and Rothe, 2019](#bib.bib20)\].

###### Corollary 3.6 (Confidence region for $`\theta_{\mathrm{PS}}`$).

Suppose the conditions of Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold, $`\varepsilon\beta<\gamma`$, and
$`\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}\leqslant B`$. Then

|  |  |  |
|----|----|----|
|  |
``` math
\liminf_{n\to\infty}\mathbb P\Bigl\{\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}\Bigl(0,B\Bigl(\frac{\varepsilon\beta}{\gamma}\Bigr)^{t}\Bigr)\Bigr\}\geqslant 1-\delta.
``` |  |

Since $`\theta_{\mathrm{PS}}`$ minimizes
$`\theta\mapsto\mathbb E_{Z\sim\mathcal{D}(\theta_{\mathrm{PS}})}[\ell(Z;\theta)]`$,
strong convexity gives
$`\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}\leqslant\frac{1}{\gamma}\sup_{z\in\mathcal{Z}}\|\nabla\ell(z;\theta_{0})\|_{2}.`$
The radius decreases geometrically in $`t`$, whereas
$`\widehat{\mathcal{R}}_{t}`$ has diameter of order $`n^{-1/2}`$. The
enlargement is therefore negligible once
$`(\varepsilon\beta/\gamma)^{t}`$ is small relative to $`n^{-1/2}`$.

##### Comparison with \[[Perdomo et al., 2020](#bib.bib12)\].

[Perdomo et al. \[2020\]](#bib.bib12) study repeated empirical risk
minimization and prove that, for a sufficiently large number of
iterations $`T`$, with probability $`1-\delta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\theta}_{T}-\theta_{\text{PS}}\|\leqslant C(\log(\frac{1}{\delta}))^{\frac{1}{m}}n^{-\frac{1}{m}},
``` |  |

where $`C`$ is an absolute constant and $`m\geqslant 2`$ is the
dimension of $`z=(x,y)`$. Our $`1-\delta`$ confidence rectangle for
$`\theta_{T}`$ is
$`\widehat{\mathcal{R}}_{T}(n,\delta)=[\widehat{\theta}_{T,j}\pm z_{1-\delta/(2d)}\sqrt{\widehat{V}_{T,jj}/n}]_{j=1}^{d}`$,
where $`z_{1-\delta/(2d)}`$ is the $`1-\delta/(2d)`$ quantile of the
standard normal distribution. The bias-aware region for
$`\theta_{\mathrm{PS}}`$ is obtained by the enlargement in Corollary
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
For each $`j\in[d]`$, the Gaussian component satisfies

|  |  |  |
|----|----|----|
|  |
``` math
|\widehat{\theta}_{T,j}-\theta_{T,j}|\leqslant C(\log(\frac{d}{\delta}))^{\frac{1}{2}}n^{-\frac{1}{2}}\quad\text{for any }j\in[d],
``` |  |

where $`C`$ is an absolute constant. Compared to the
$`O(n^{-\frac{1}{m}})`$ width by \[[Perdomo et al., 2020](#bib.bib12)\],
our interval achieves a better rate $`O(n^{-\frac{1}{2}})`$ when $`m>2`$
(ignoring the exponentially small term in the bias-aware adjustment).
Thus, our proposed inference method provides a narrower confidence
interval and improved statistical efficiency.

## 4 Statistical Inference for RGD

This section develops inference for the population RGD iterate
$`\theta_{t}`$ based on the REGD iterate $`\widehat{\theta}_{t}`$, with
$`t`$ fixed and $`n\to\infty`$. The structure follows Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"):
Section
[4.1](#S4.SS1 "4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives the central limit theorem, Section
[4.2](#S4.SS2 "4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
estimates its covariance, and Section
[4.3](#S4.SS3 "4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives confidence regions for $`\theta_{\mathrm{PS}}`$. We use $`V_{t}`$,
$`\widehat{V}_{t}`$, and $`\widehat{\mathcal{R}}_{t}`$ for the RGD
counterparts of the quantities in Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

### 4.1 Central Limit Theorem of RGD

For $`\theta\in\Theta`$, define the population gradient
$`S(\theta):=\mathbb E_{Z\sim\mathcal{D}(\theta)}[\nabla\ell(Z;\theta)],`$
so that the RGD update in Section
[2.2](#S2.SS2 "2.2 Repeated Gradient Descent ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
is $`\theta_{t+1}=\theta_{t}-\alpha S(\theta_{t})`$. Subtracting it from
the REGD update gives

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\theta}_{t+1}-\theta_{t+1}=-\alpha\Bigl\{\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t})-S(\widehat{\theta}_{t})\Bigr\}+\bigl[\{\widehat{\theta}_{t}-\alpha S(\widehat{\theta}_{t})\}-\{\theta_{t}-\alpha S(\theta_{t})\}\bigr].
``` |  |

The first term is the sampling error of the current round. Conditional
on all preceding observations and estimators, $`\widehat{\theta}_{t}`$
is fixed and
$`\nabla\ell(Z_{t,1};\widehat{\theta}_{t}),\ldots,\nabla\ell(Z_{t,n};\widehat{\theta}_{t})`$
are i.i.d. with mean $`S(\widehat{\theta}_{t})`$. Unlike the RERM case,
we could directly apply the central limit theorem to get the approximate
conditional distribution of $`\sqrt{n}`$ times the first term
$`\mathcal{N}\bigl(0,\alpha^{2}\operatorname{Cov}_{Z\sim\mathcal{D}(\widehat{\theta}_{t})}(\nabla\ell(Z;\widehat{\theta}_{t}))\bigr)`$,
and this covariance tends to
$`\alpha^{2}\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}(\nabla\ell(Z;\theta_{t}))`$
as $`\widehat{\theta}_{t}\to\theta_{t}`$. The second term is the effect
of the preceding estimation error on the population update. By
Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
$`S`$ is continuously differentiable, so the second term is
approximately
$`(I_{d}-\alpha\nabla S(\theta_{t}))(\widehat{\theta}_{t}-\theta_{t})`$.

###### Theorem 4.1 (Central limit theorem for REGD).

Under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each fixed $`t\geqslant 1`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\mathcal{N}(0,V_{t}),
``` |  |

where $`V_{0}=0`$ and
$`V_{t+1}=(I_{d}-\alpha\nabla S(\theta_{t}))\,V_{t}\,(I_{d}-\alpha\nabla S(\theta_{t}))^{\top}+\alpha^{2}\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}\bigl(\nabla\ell(Z;\theta_{t})\bigr),t\geqslant 0.`$

The recursion has the form of Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
with $`\nabla G(\theta_{t})`$ replaced by the Jacobian
$`I_{d}-\alpha\nabla S(\theta_{t})`$ of the RGD update and
$`\Sigma_{\theta_{t}}(\theta_{t+1})`$ replaced by the covariance of one
gradient step.

### 4.2 Estimation of the Asymptotic Covariance

The covariance
$`\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}(\nabla\ell(Z;\theta_{t}))`$
can be estimated from batch $`t`$. The Jacobian $`\nabla S(\theta_{t})`$
also depends on how the distribution changes with the parameter, since
$`\theta`$ enters $`S(\theta)`$ both through the loss and through
$`\mathcal{D}(\theta)`$. Under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
differentiating both gives

|  |  |  |
|----|----|----|
|  |
``` math
\nabla S(\theta)=H_{\theta}(\theta)+\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\nabla\ell(Z;\theta)\,\nabla_{\theta}\log p(Z;\theta)^{\top}\bigr],
``` |  |

where the second term is computed as in Proposition
[3.1](#S3.Thmtheorem1 "Proposition 3.1. ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
It involves the same score as $`\nabla G`$. As in Section
[3.2](#S3.SS2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we use the score directly when it is known; otherwise we replace it by
$`s(\cdot,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))`$,
where $`\widehat{\psi}_{n,k}(\widehat{\theta}_{t})`$ is computed from
batch $`t`$ and the perturbed batches at
$`\widehat{\theta}_{t}+\eta e_{j}`$, $`j\in[d]`$, as in
([2](#S3.E2 "Equation 2 ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).

Using batch $`t`$, define

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla S}_{t}:=\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\Bigl\{\nabla^{2}\ell(Z_{t,i};\widehat{\theta}_{t})+\nabla\ell(Z_{t,i};\widehat{\theta}_{t})\,s\bigl(Z_{t,i},\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr)^{\top}\Bigr\}.
``` |  |

Following Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
set $`\widehat{V}_{0}=0`$ and, for $`t\geqslant 0`$,

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle\widehat{V}_{t+1}:={}`$ | $`\displaystyle(I_{d}-\alpha\widehat{\nabla S}_{t})\,\widehat{V}_{t}\,(I_{d}-\alpha\widehat{\nabla S}_{t})^{\top}`$ |  | (4) |
|  |  | $`\displaystyle+\frac{\alpha^{2}}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t})\nabla\ell(Z_{t,i};\widehat{\theta}_{t})^{\top}-(\widehat{\theta}_{t+1}-\widehat{\theta}_{t})(\widehat{\theta}_{t+1}-\widehat{\theta}_{t})^{\top}.`$ |  |  |

By the REGD update, $`\widehat{\theta}_{t}-\widehat{\theta}_{t+1}`$ is
$`\alpha`$ times the average of the gradients
$`\nabla\ell(Z_{t,i};\widehat{\theta}_{t})`$, so the last two terms
equal $`\alpha^{2}`$ times their sample covariance.

###### Theorem 4.2.

Under the conditions of Theorem
[3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each fixed $`t\geqslant 0`$,

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla S}_{t}-\nabla S(\theta_{t})\|_{2}^{2}=O_{P}\Bigl(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta+a_{n,k}\Bigr).
``` |  |

The rate and the origins of its four terms are the same as in Theorem
[3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

###### Corollary 4.3.

Under the conditions of Theorem
[4.2](#S4.Thmtheorem2 "Theorem 4.2. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\widehat{V}_{t}\xrightarrow{P}V_{t}`$ for each fixed $`t\geqslant 1`$.
If, in addition, $`V_{t}`$ is nonsingular, then the region
$`\widehat{\mathcal{R}}_{t}`$ of Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
computed from the REGD iterate $`\widehat{\theta}_{t}`$ and
$`\widehat{V}_{t}`$ in
([4](#S4.E4 "Equation 4 ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
satisfies
$`\lim_{n\to\infty}\mathbb P(\theta_{t}\in\widehat{\mathcal{R}}_{t})=1-\delta`$.

### 4.3 Confidence Regions for the Performative Stable Point

As in Section
[3.3](#S3.SS3 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we enlarge $`\widehat{\mathcal{R}}_{t}`$ by a bound on
$`\|\theta_{t}-\theta_{\mathrm{PS}}\|_{2}`$. Define

|  |  |  |
|----|----|----|
|  |
``` math
\rho_{\mathrm{RGD}}:=1-\alpha\beta\Bigl\{\frac{\gamma}{\beta+\gamma}-\varepsilon(1+1.5\alpha\beta)\Bigr\}.
``` |  |

If $`\varepsilon<\gamma/\{(\beta+\gamma)(1+1.5\alpha\beta)\}`$ and
$`\alpha\leqslant 2/(\beta+\gamma)`$, then $`\varepsilon\beta<\gamma`$,
so $`\theta_{\mathrm{PS}}`$ is well defined, and
$`\rho_{\mathrm{RGD}}\in(0,1)`$. In this case, [Perdomo et al. \[2020,
Theorem 3.8\]](#bib.bib12) gives
$`\|\theta_{t}-\theta_{\mathrm{PS}}\|_{2}\leqslant\rho_{\mathrm{RGD}}^{t}\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}`$.

###### Corollary 4.4 (Confidence region for $`\theta_{\mathrm{PS}}`$).

Suppose the conditions of Corollary
[4.3](#S4.Thmtheorem3 "Corollary 4.3. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold, $`\varepsilon<\gamma/\{(\beta+\gamma)(1+1.5\alpha\beta)\}`$,
$`\alpha\leqslant 2/(\beta+\gamma)`$, and
$`\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}\leqslant B`$. Then

|  |  |  |
|----|----|----|
|  |
``` math
\liminf_{n\to\infty}\mathbb P\bigl\{\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}(0,B\rho_{\mathrm{RGD}}^{t})\bigr\}\geqslant 1-\delta.
``` |  |

The bound on $`B`$ after Corollary
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
applies here, since it does not depend on the algorithm. Compared with
Corollary
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
only the contraction factor changes, from $`\varepsilon\beta/\gamma`$ to
$`\rho_{\mathrm{RGD}}`$. Since $`\rho_{\mathrm{RGD}}\geqslant 1/2`$,
whereas $`\varepsilon\beta/\gamma`$ can be close to zero, RGD may need
more rounds than RRM before the bias is negligible.

## 5 Experiment

In this section, we provide empirical support for our theoretical
results through three experimental settings. More experimental details,
as well as additional experiments on the benefits of score matching and
PPI, are deferred to Appendix
[C](#A3 "Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

### 5.1 Controlled Synthetic Studies

We first consider a synthetic setting motivated by a stylized lending
market, where prediction-based lending decisions affect borrower
participation. The lender sets interest rates based on predicted default
risk. If lower-risk borrowers are more sensitive to borrowing costs,
higher rates disproportionately discourage their participation,
increasing the share of higher-risk borrowers. Through this selection
response mechanism, the deployed predictor shapes the population used
for subsequent training. Formally, let $`\mathcal{X}=[0,1]`$ denote the
space of borrower risk score and $`\mathcal{Y}=\{0,1\}`$ the outcome
space, with $`Y=1`$ indicating default. The instance space is
$`\mathcal{Z}=\mathcal{X}\times\mathcal{Y}`$. We model the predicted
default probability as $`h_{\theta}(x)=\sigma(\theta x)`$, where
$`\theta\in\Theta:=\mathbb R`$ is a scalar parameter and
$`\sigma(u)=(1+e^{-u})^{-1}`$ is the sigmoid function.

We capture this mechanism through a two-component mixture. Specifically,
deployment of $`\theta`$ induces risk score distribution
$`X\sim(1-a(\theta))\,\mathrm{Unif}(0,1)+a(\theta)\,B(5,2),`$ where
$`a(\theta)=0.8\sigma(\kappa(\theta-b))`$ is the deployment-dependent
mixture weight of the high-risk borrower component, and we set the
conditional distribution of binary label $`Y`$ to be
$`Y\mid X=x\sim\mathrm{Ber}\big(\sigma(-1+2x)\big).`$ These
distributions jointly define the distribution map $`{\cal D}(\theta)`$.
Thus, deployment changes the borrower risk score distribution: the beta
component has higher risk characteristics on average, and its weight
increases with $`\theta`$, while leaving the conditional outcome
distribution fixed. The predictor is fitted using regularized logistic
loss,

|  |  |  |
|----|----|----|
|  |
``` math
\ell((x,y);\theta)=\log(1+e^{\theta x})-y\theta x+\frac{\tau}{2}\|\theta\|_{2}^{2},
``` |  |

where $`\tau>0`$ is a regularization parameter. Our inference target is
the population iterate $`\theta_{t}`$ and performative stable point
$`\theta_{\text{PS}}`$.

#### 5.1.1 Mild Performativity Feedback

We first consider a setting with mild performative feedback to examine
the roles of repeated retraining and uncertainty propagation in
inference for the performative stable point. We initialize both
algorithms at $`\theta_{0}=1`$, set $`\kappa=\tau=1`$, $`b=0`$, and use
step size $`\alpha=0.5`$ for RGD. Each deployment supplies $`n=2000`$
fresh labeled observations per round, and we conduct 10,000 independent
Monte Carlo repetitions for each method. We build confidence intervals
for population trajectory and performative stable point using Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[4.3](#S4.Thmtheorem3 "Corollary 4.3. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
and evaluate coverage and interval width at fixed reporting round, where
the population trajectories $`\{\theta_{t}\}_{t=1}^{T}`$ and the PS
point $`\theta_{\mathrm{PS}}`$ are numerically computed from the
specified distribution and serve as evaluation references. All
experiments include two baselines. First, we consider a one-step ERM
baseline that fits the predictor once using observations from
$`\mathcal{D}(\theta_{0})`$ entirely without accounting for
performativity. Its estimator coincides with the first RRM update, and
its conventional confidence interval targets $`\theta_{1}`$ without an
adjustment for the discrepancy between $`\theta_{1}`$ and
$`\theta_{\mathrm{PS}}`$. Second, we consider a per-round M-estimator
baseline that constructs each interval using only the uncertainty
estimate from the current round’s M-estimator. It uses the same RRM
point estimates and update samples, but replaces the recursive
covariance estimator with an empirical estimate of the current-round
covariance $`\Sigma_{\theta_{t}}(\theta_{t+1})`$. This is equivalent to
setting $`\widehat{\nabla G}_{t}=0`$ in the covariance recursion,
thereby omitting uncertainty inherited from earlier rounds through their
effects on subsequent deployment distributions. For the performative
stable point, it retains the same deterministic bias enlargement.
Details of score estimation, contraction constants, and bias bounds are
provided in Appendix
[C.1](#A3.SS1 "C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Results. Table
[1](#S5.T1 "Table 1 ‣ 5.1.1 Mild Performativity Feedback ‣ 5.1 Controlled Synthetic Studies ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
reports the results at the reporting rounds. Empirical trajectory
coverage is close to the 95% nominal level for both RRM and RGD, and the
corresponding confidence intervals for PS point have coverage slightly
higher than nominal level, consistent with the conservative enlargement
used to account for the remaining population-iteration bias in Corollary
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
The one-step ERM baseline achieves near-nominal coverage of its own
target $`\theta_{1}`$ but shows significant undercoverage for the PS
point. This illustrates that valid inference for a one-step ERM target
does not by itself provide valid inference for the performative stable
point, highlighting the need to account for the feedback induced by
subsequent deployments. The per-round M-estimator, which omits
uncertainty propagated from earlier rounds, achieves nearly identical
coverage of both trajectory and the performative stable point at the RRM
reporting round. Thus, repeated retraining remains important for
approaching the performative stable point even when the contribution of
propagated uncertainty is small. We next consider a controlled
experiment with stronger feedback to demonstrate the importance of
retaining this contribution in covariance estimation. RRM also exhibits
faster empirical convergence than RGD in this setting as we have
remarked at the end of Section
[4.3](#S4.SS3 "4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Method |  Reporting round |  Coverage of $`\theta_{t}`$ (%) |  Coverage of $`\theta_{\mathrm{PS}}`$ (%) |  Trajectory interval width |  Stable-point interval width |
| One-step ERM | 1 | 94.65 | 81.98 | 0.026 | 0.026 |
| Per-round M-estimator | 6 | 94.78 | 95.23 | 0.025 | 0.026 |
| RRM | 6 | 94.78 | 95.24 | 0.025 | 0.026 |
| RGD | 78 | 94.99 | 97.09 | 0.016 | 0.017 |

Table 1: Empirical coverage and mean widths of nominal $`95\%`$
confidence intervals for the population iterate $`\theta_{t}`$ and
performative stable point $`\theta_{\mathrm{PS}}`$ in the pure-synthetic
experiment.

Figure 1: Pure-Synthetic Experiment. Left Panel: Empirical coverage of
$`\theta_{t}`$ across rounds; Right Panel: Empirical coverage of
performative stable point $`\theta_{\text{PS}}`$ across rounds. Vertical
dashed line implies the fixed reporting round of RRM selected by
$`B_{t}<10^{-3}`$.

#### 5.1.2 Uncertainty Propagation under Strong Performative Feedback.

We next examine the necessity of accounting for uncertainty propagated
through performative feedback. As shown in Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the covariance of an RRM iterate includes the contribution
$`\nabla G(\theta_{t})V_{t}\nabla G(\theta_{t})^{\top}`$ from earlier
rounds, where $`\nabla G(\theta_{t})`$ depends on the distributional
score. Omitting this contribution treats each round as a separate
M-estimation problem and can understate uncertainty. To study this
effect under stronger feedback, we set $`\kappa=80`$, $`b=0.05`$, and
the activation function becomes
$`a(\theta)=0.8\sigma\bigl(80(\theta-0.05)\bigr)`$. Each deployment
supplies $`n=10{,}000`$ fresh labeled observations, and we conduct
10,000 independent Monte Carlo repetitions for each algorithm. We use a
correctly specified basis score model, whose construction, fitting
procedure and an ablation on candidate score function family in this
setup are provided in Appendix
[C.1](#A3.SS1 "C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Results. In this stronger-feedback setting, empirical trajectory
coverage is close to the nominal level for both RRM and RGD, while
stable-point coverage is slightly higher, consistent with the
conservative bias enlargement in Corollaries
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
The one-step ERM baseline achieves near-nominal coverage of its own
target $`\theta_{1}`$ but exhibits severe undercoverage for the
performative stable point, highlighting the importance of repeated
retraining under distributional feedback. The per-round M-estimator
produces narrower intervals and exhibits substantial undercoverage for
both the population trajectory and the performative stable point. Unlike
the weaker-feedback setting, the contribution of propagated uncertainty
is therefore substantial. These results illustrate that approaching the
performative stable point through repeated retraining does not alone
ensure valid inference: the covariance estimator must also account for
uncertainty accumulated through deployment feedback.

Figure 2: Pure-Synthetic Controlled Experiment with Strong Feedback.
Left Panel: Empirical coverage of $`\theta_{t}`$ across rounds; Right
Panel: Empirical coverage of performative stable point
$`\theta_{\text{PS}}`$ across rounds. Vertical dashed line implies the
fixed reporting round of RRM selected by $`B_{t}<10^{-3}`$.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Method |  Reporting round |  Coverage of $`\theta_{t}`$ (%) |  Coverage of $`\theta_{\mathrm{PS}}`$ (%) |  Trajectory interval width |  Stable-point interval width |
| One-step ERM | 1 | 95.04 | 4.43 | 0.012 | 0.012 |
| Per-round M-estimator | 20 | 88.77 | 92.11 | 0.011 | 0.013 |
| RRM | 20 | 94.67 | 96.60 | 0.014 | 0.015 |
| RGD | 39 | 94.90 | 98.14 | 0.010 | 0.012 |

Table 2: Empirical coverage and mean widths of interval in controlled
synthetic experiment.

### 5.2 Semi-synthetic Experiment

We next consider a semi-synthetic setup on the Learning Mindset dataset
\[[Athey and Wager, 2019](#bib.bib9)\], an educational dataset modeling
how growth-mindset interventions, which encourage students to view
intellectual abilities as developable through effort and learning, will
affect students’ academic achievement. Before intervention assignment,
the deployed predictor uses students’ characteristics $`X`$ to estimate
the probability that their subsequent academic achievement will meet or
exceed a predefined threshold. These predictions guide intervention
allocation: students with lower predicted success probabilities are more
likely to receive support. Because interventions can affect achievement,
deployment changes the distribution of the outcomes the predictor aims
to predict and uses for subsequent retraining, creating performative
feedback.

Formally, let $`{\cal X}`$ denote the space of student profiles and
$`{\cal Y}=\{0,1\}`$ the outcome space, with $`Y=1`$ indicating
achievement at or above a predefined academic score threshold. Let
$`W\in\{0,1\}`$ indicate intervention assignment, with $`W=1`$ denoting
receipt of intervention. The predictor is
$`h_{\theta}(x)=\sigma(\theta^{\top}\phi(x))`$, where
$`\phi(x)\in\mathbb R^{4}`$ is a feature representation constructed from
student profiles using PCA and $`\theta\in\Theta=\mathbb R^{4}`$ is the
model parameter. The predictions guide intervention assignment: students
with lower predicted success probabilities are more likely to receive
intervention. Specifically, intervention is assigned with probability

|  |  |  |
|----|----|----|
|  |
``` math
e_{\theta}(x)=\delta+(1-2\delta)\sigma\big(-\kappa\theta^{\top}\phi(x)\big),
``` |  |

where $`\kappa>0`$ controls the responsiveness of assignment to the
predictor, and $`0<\delta<1/2`$ ensures that both intervention and
non-intervention probabilities are bounded below by $`\delta`$. Given
student profile $`X=x`$ and intervention assignment $`W=w`$, we generate
final achievement $`Y\sim\text{Ber}(p(x,w))`$, where $`p(x,w)`$ is the
success probability supplied by a neural simulator fitted to the data
and held fixed throughout the experiment. Thus, changing the predictor
changes intervention assignment and therefore changes the conditional
outcome distribution the predictor aims to learn via intervention
effect. The predictor is trained using the regularized logistic loss
after each deployment.

|  |  |  |
|----|----|----|
|  |
``` math
\ell((x,y);\theta)=\log\!\left(1+e^{\theta^{\top}\phi(x)}\right)-y\theta^{\top}\phi(x)+\frac{\tau}{2}\|\theta\|_{2}^{2},\qquad\tau=1.
``` |  |

Each deployment supplies $`n=2000`$ fresh labeled observations. In this
experiment, for covariance estimation, we use the analytic
distributional score computed from the known intervention rule
$`e_{\theta}(X)`$ and the recorded intervention assignments $`W`$. The
required population moments are estimated from the observed deployment
samples.

Results. Table
[3](#S5.T3 "Table 3 ‣ 5.2 Semi-synthetic Experiment ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
reports coverage and mean Euclidean diameters at fixed reporting rounds:
round $`24`$ for RRM and round $`31`$ for RGD, applied uniformly across
all $`10000`$ Monte Carlo repetitions. Trajectory coverage is close to
the nominal $`95\%`$ level and the confidence regions for performative
stable point attain coverage slightly larger than nominal level due to
conservative bias-aware enlargement. The one-step ERM baseline achieves
near-nominal coverage of its own target $`\theta_{1}`$ but substantially
undercovers the performative stable point. The per-round M-estimator
yields results close to those of RRM, with slightly smaller confidence
regions and lower empirical coverage, consistent with its omission of
uncertainty propagated from earlier rounds. Figure
[3](#S5.F3 "Figure 3 ‣ 5.2 Semi-synthetic Experiment ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
complements these summaries by showing coverage across deployment
rounds.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Method |  Reporting round |  Coverage of $`\theta_{t}`$ (%) |  Coverage of $`\theta_{\mathrm{PS}}`$ (%) |  Trajectory region diameter |  Stable-point region diameter |
| One-step ERM | 1 | 95.16 | 80.35 | 0.046 | 0.046 |
| Per-round M-estimator | 24 | 94.82 | 96.42 | 0.046 | 0.048 |
| RRM | 24 | 95.05 | 96.58 | 0.046 | 0.048 |
| RGD | 31 | 95.02 | 96.61 | 0.043 | 0.045 |

Table 3: Empirical coverage and mean Euclidean diameters of nominal
$`95\%`$ confidence regions for the population iterate $`\theta_{t}`$
and performative stable point $`\theta_{\mathrm{PS}}`$ in the
semi-synthetic experiment.

Figure 3: Semi-Synthetic Experiment. Left Panel: Empirical coverage of
$`\theta_{t}`$ across rounds; Right Panel: Empirical coverage of
performative stable point $`\theta_{\text{PS}}`$ across rounds. Vertical
dashed line implies the fixed reporting round of RRM selected by
$`B_{t}<10^{-3}`$.

### 5.3 Self-improving LLM routing

Finally, we consider a self-evolving model routing task on real datasets
consisting of mathematical word problems benchmark GSM8K \[[Cobbe et
al., 2021](#bib.bib10)\] and commonsense knowledge benchmark
CommonsenseQA \[[Talmor et al., 2019](#bib.bib8)\]. We use a frozen
Qwen2.5-1.5B model with two generation modes: fast mode and slow mode.

We consider a stylized model of self-evolving routing based on
Best-of-$`N`$ (BoN) selection. Given an input question, the deployed
router independently routes the question to a mode for each of $`N`$
candidates and collects the response. We evaluate the responses using a
fixed utility and assign the winning mode as the supervision label. The
router is then updated to learn to predict this label from the input.
Performativity arises because deployment affects the supervision label
distribution, thereby altering the distribution of the winning-mode
labels the router aims to predict.

Specifically, fix positive integer $`N\geqslant 2`$, let $`{\cal X}`$ be
the problem space, $`{\cal A}=\{0,1\}`$ the routing choice space, with
routing choice $`A=1`$ denoting routing to slow mode. We adopt a
logistic router $`h_{\theta}(x)=\sigma(\theta^{\top}\phi(x))`$, where
$`\phi:\mathcal{X}\to{\mathbb R}^{5}`$ is a five-dimensional feature
vector map. The routing probability is defined by
$`\pi_{\theta}(1\mid x)=(1-\xi)h_{\theta}(x)+\frac{\xi}{2},`$ where
$`\xi=0.2`$ provides a positive routing probability to each mode bounded
away from $`0`$. For each problem $`X`$, routing choices
$`A_{1},\ldots,A_{N}`$ are i.i.d. sampled conditionally on $`X`$ using
$`\pi_{\theta}(\cdot|X)`$ and draw the corresponding mode response
$`R_{j}`$. The highest-scoring candidate supplies the supervision label
for the next BoN update: $`J^{*}=\arg\max_{j\in[N]}u(X,R_{j}),`$ where
$`u`$ is a fixed utility function. We fit the router to these
winning-mode labels using regularized logistic loss,

|  |  |  |
|----|----|----|
|  |
``` math
\ell(Z;\theta)=w(Z)\big[\log\left(1+e^{\theta^{\top}\phi(X)}\right)-A_{{J}^{*}}\theta^{\top}\phi(X)\big]+\frac{\tau}{2}\|\theta\|_{2}^{2},
``` |  |

where $`Z=(X,A_{1},R_{1},\ldots,A_{N},R_{N})`$ and $`w(Z)\in[0,1]`$ is a
utility advantage weight, with larger weight indicating stronger
preference to the winning mode. Changing the deployed parameter alters
the candidate distribution and hence the conditional distribution of
$`A_{J^{*}}`$, even though the question population and mode-specific
response distributions remain fixed. We apply RRM and RGD to this
setting and evaluate inference for their population trajectories and
performative stable points. For covariance estimation, we use the
analytic distributional score computed from the known routing
probabilities and the sampled mode choices $`A_{1},\ldots,A_{N}`$. The
required population moments are estimated from the observed deployment
samples.

Results. Table
[4](#S5.T4 "Table 4 ‣ 5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
reports coverage and mean Euclidean diameters for best-of-$`5`$ routing
with $`n=10,000`$ at fixed reporting rounds: round $`55`$ for RRM and
round $`198`$ for RGD. Trajectory coverage is $`94.74\%`$ for RRM and
$`94.75\%`$ for RGD, close to the nominal $`95\%`$ coverage level. The
confidence regions for the performative stable point attain coverage
above the nominal level, consistent with the conservative enlargement
used to account for the remaining population-iteration bias. The
one-step ERM baseline achieves near-nominal coverage of its own target
$`\theta_{1}`$, but significantly undercovers the performative stable
point, while the per-round M-estimator yields results very close to
those of RRM. Figure
[4](#S5.F4 "Figure 4 ‣ 5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
complements these summaries by showing coverage across deployment
rounds.

Table
[5](#S5.T5 "Table 5 ‣ 5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
summarizes changes in routing probability relative to the initial
equal-mixture policy and average utility at the performative stable
point on the merged question pool. The mean total variation distance is
$`3.00\%`$, with routing shifting toward the slow mode, and the overall
expected utility increases from $`0.483`$ to $`0.500`$. Table
[6](#S5.T6 "Table 6 ‣ 5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
compares the slow-mode utility advantage for individual response draws
and best-of-$`5`$ candidate sets. Best-of-$`5`$ increases this advantage
on GSM8K but reduces it on CommonsenseQA. This pattern is consistent
with the larger routing shift and utility gain observed on GSM8K.
Detailed experimental configurations are provided in Appendix
[C.3](#A3.SS3 "C.3 LLM Routing ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Figure 4: LLM routing with Best-of-$`5`$ selection and $`n=10000`$.
Left: empirical coverage of $`\theta_{t}`$; right: empirical coverage of
$`\theta_{\mathrm{PS}}`$. Both methods are shown through round $`198`$.
The vertical dashed line marks the fixed reporting round for RRM $`55`$,
selected by $`B_{t}<10^{-3}`$. Horizontal dashed lines indicate nominal
$`95\%`$ coverage; shaded bands are pointwise $`95\%`$ exact binomial
Monte Carlo confidence intervals.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Method |  Reporting round |  Coverage of $`\theta_{t}`$ (%) |  Coverage of $`\theta_{\mathrm{PS}}`$ (%) |  Trajectory region diameter |  Stable-point region diameter |
| One-step ERM | 1 | 95.25 | 84.86 | 0.015 | 0.015 |
| Per-round M-estimator | 55 | 94.71 | 98.60 | 0.015 | 0.016 |
| RRM | 55 | 94.74 | 98.61 | 0.015 | 0.016 |
| RGD | 198 | 94.75 | 98.95 | 0.014 | 0.016 |

Table 4: Empirical coverage and mean Euclidean diameters of nominal
$`95\%`$ confidence regions in the LLM routing experiment.

|  |  |  |  |  |
|----|----|----|----|----|
| Question subset |  Slow-mode routing probability (%) |  Mean routing deviation (%) |  Initial utility |  Stable-point router utility |
| All questions | 52.99 | 3.00 | 0.483 | 0.500 |
| GSM8K | 56.46 | 6.46 | 0.413 | 0.452 |
| CommonsenseQA | 50.21 | 0.22 | 0.539 | 0.539 |

Table 5: Routing deviation and utility in the LLM routing experiment,
evaluated at the population performative stable point. Routing deviation
is the mean total variation distance from the initial equal-mixture
policy. Utility is the expected value of answer correctness minus a
token-cost penalty of $`0.1`$ per $`1{,}000`$ generated tokens,
evaluated for one routed response-bank entry.

|  |  |  |  |  |
|----|----|----|----|----|
| Dataset |  Mean fast utility |  Mean slow utility |  Single-draw advantage |  Best-of-$`5`$ advantage |
| GSM8K | 0.1125 | 0.7128 | 0.6003 | 0.6535 |
| CommonsenseQA | 0.5074 | 0.5709 | 0.0635 | 0.0186 |

Table 6: Utility comparison. The single-draw advantage is mean slow
utility minus mean fast utility. The Best-of-$`5`$ advantage is the
expected difference between the best slow and fast candidate utilities
under the initial equal-mixture policy, conditional on both modes
appearing.

### 5.4 Feedback Strength and Uncertainty Propagation

The experiments illustrate two consequences of performative feedback for
inference. First, a predictor fitted to the initial deployment
distribution need not coincide with the performative stable point, as
demonstrated by the one-step ERM comparisons. Second, repeated
retraining propagates estimation errors through subsequent deployment
distributions. For RRM, this propagation is governed by $`\nabla G`$,
which contributes
$`\nabla G(\theta_{t})V_{t}\nabla G(\theta_{t})^{\top}`$ to the
covariance of the next iterate. The per-round M-estimator omits this
positive semidefinite contribution and can therefore underestimate
uncertainty, even after the population trajectory has approached the
performative stable point. For RGD, uncertainty propagates through the
corresponding update Jacobian $`I_{d}-\alpha\nabla S(\theta_{t})`$.

The importance of this correction varies across experimental settings.
In the synthetic experiment with $`\kappa=1`$ and the two practical
settings, the RRM map has relatively small local derivatives. The
proposed method consequently produces confidence regions close in
diameter to those of the per-round M-estimator, while accounting for
uncertainty from earlier rounds. Under the stronger feedback induced by
$`\kappa=80`$, omitting this contribution leads to substantial
undercoverage, and incorporating the score term markedly improves
calibration. These findings explain why repeated retraining can remain
important for approaching the performative stable point even when the
additional covariance correction is small.

In applications, the strength of this feedback is generally unknown.
Even when the score is analytically available, as in our semi-synthetic
and LLM evolving experiments, the Jacobian remains unknown because it
depends on population moments involving the loss, which we have to
estimate from observed data. The estimator developed in Section
[3.2](#S3.SS2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
consistently estimates $`\nabla G`$ under the stated regularity and
score-model richness assumptions, allowing this sensitivity to be
assessed from data. Table
[7](#S5.T7 "Table 7 ‣ 5.4 Feedback Strength and Uncertainty Propagation ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
evaluates the estimator used at each RRM reporting round against the
population derivative at the same realized deployment. For each
reporting round $`T`$, we report the population Jacobian norm
$`\|\nabla G(\widehat{\theta}_{T-1})\|_{2}`$ averaged across
repetitions, together with the Jacobian estimation RMSE. The norm
summarizes local feedback strength near the performative stable point,
whereas the RMSE measures estimation accuracy. Here the score
specification and Jacobian RMSE refer to the proposed estimator;
coverage refers to the per-round M-estimator with the same deterministic
bias enlargement. This sensitivity governs the covariance contribution
$`\nabla G(\theta_{T-1})V_{T-1}\nabla G(\theta_{T-1})^{\top}`$ from
uncertainty inherited at round $`T`$. These diagnostics indicate
relatively mild feedback in the $`\kappa=1`$ synthetic experiment and
the two practical settings, and stronger feedback in the $`\kappa=80`$
experiment. All four configurations substantially improve on the
zero-Jacobian baseline in Jacobian estimation RMSE. We defer the
experimental details to Appendix
[C.4](#A3.SS4 "C.4 Jacobian estimation diagnostics. ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Setting | Score | Round |  Mean population Jacobian norm |  Jacobian RMSE |  Per-round M-estimator coverage of $`\theta_{\mathrm{PS}}`$ (%) |
| Synthetic ($`\kappa=1`$) | Basis | 6 | 0.0079 | 0.0061 | 95.23 |
| Synthetic ($`\kappa=80`$) | Basis | 20 | 0.5719 | 0.0329 | 92.11 |
| Semi-synthetic | Analytic | 24 | 0.1312 | 0.0441 | 96.42 |
| LLM routing | Analytic | 55 | 0.0661 | 0.0105 | 98.60 |

Table 7: Feedback strength, Jacobian estimation error, and coverage of
the performative stable point by the per-round M-estimator, using 10,000
Monte Carlo repetitions.

Our estimator of $`\nabla G`$ therefore provides a policy designer with
information about how changes in the deployed predictor affect the next
population update. The same estimate enters the covariance recursion,
accounting for feedback without requiring its strength to be known in
advance. In the settings with mild feedback, this correction entails
little additional interval width; when feedback is stronger, it
addresses an important source of undercoverage. These estimates
characterize local sensitivity at the observed deployments, while the
uniform contraction bounds used for stable-point bias control are
established separately.

## 6 Conclusion

In this paper, we study statistical inference under performativity. We
derive asymptotic distributions for the RRM and RGD trajectories and
construct confidence regions for the performative stable point. We
further extend prediction-powered inference to the dynamic setting under
performativity. Our confidence regions for $`\theta_{\mathrm{PS}}`$
account for the remaining iteration bias; direct inference methods for
$`\theta_{\mathrm{PS}}`$ are left for future work. We evaluate our
framework in synthetic, semi-synthetic educational, and self-evolving
LLM routing settings, demonstrating the practical need to account for
performativity in statistical inference.

## 7 Acknowledgements

We would like to acknowledge the helpful discussion and suggestions from
Tijana Zrinc, Juan Perdomo, Anastasios Angelopoulos, and Steven Wu.

## References

- Angelopoulos et al. (2023a) A. N. Angelopoulos, J. C. Duchi, and T.
  Zrnic PPI++: efficient prediction-powered inference. arXiv preprint
  arXiv:2311.01453. Cited by:
  [§A.1](#A1.SS1.p1.1 "A.1 Setup of PPI under Performativity ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§A.2](#A1.SS2.p1.2 "A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§A.3](#A1.SS3.p1.1 "A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [item
  2](#S1.I1.i2.p1.1 "In Our contributions. ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Angelopoulos et al. (2023b) A. N. Angelopoulos, S. Bates, C.
  Fannjiang, M. I. Jordan, and T. Zrnic Prediction-powered inference.
  Science 382, pp. 669 – 674. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:256105365) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Armstrong et al. (2020) T. B. Armstrong, M. Kolesár, and S. Kwon
  Bias-aware inference in regularized regression models. arXiv preprint
  arXiv:2012.14823. Cited by:
  [§3.3](#S3.SS3.p1.1 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Armstrong and Kolesár (2018) T. B. Armstrong and M. Kolesár Optimal
  inference in a class of regression models. Econometrica 86 (2),
  pp. 655–683. Cited by:
  [§3.3](#S3.SS3.p1.1 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Athey and Wager (2019) S. Athey and S. Wager Estimating treatment
  effects with causal forests: an application. Observational studies 5
  (2), pp. 37–51. Cited by:
  [§C.2.1](#A3.SS2.SSS1.p1.1 "C.2.1 Dataset and deployment population. ‣ C.2 Semi-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [item
  3](#S1.I1.i3.p2.1 "In Our contributions. ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§5.2](#S5.SS2.p1.1 "5.2 Semi-synthetic Experiment ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Bracale et al. (2024) D. Bracale, S. Maity, F. M. Polo, S.
  Somerstep, M. Banerjee, and Y. Sun Microfoundation inference for
  strategic prediction. arXiv preprint arXiv:2411.08998. Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1.1](#S1.SS1.SSS0.Px3.p1.1 "Inference in performativity. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Cobbe et al. (2021) K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H.
  Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al.
  Training verifiers to solve math word problems. arXiv preprint
  arXiv:2110.14168. Cited by:
  [§5.3](#S5.SS3.p1.1 "5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Cortinovis and Caron (2025) S. Cortinovis and F. Caron FAB-ppi:
  frequentist, assisted by bayes, prediction-powered inference. ArXiv
  abs/2502.02363. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:276107406) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Cutler et al. (2024) J. Cutler, M. Diaz, and D. Drusvyatskiy
  Stochastic approximation with decision-dependent distributions:
  asymptotic normality and optimality. Journal of Machine Learning
  Research 25 (90), pp. 1–49. Cited by:
  [§1.1](#S1.SS1.SSS0.Px3.p1.1 "Inference in performativity. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p8.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Demirtas (2018) H. Demirtas Flexible imputation of missing data.
  Journal of Statistical Software 85, pp. 1–5. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:65237735) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Drusvyatskiy and Xiao (2020) D. Drusvyatskiy and L. Xiao Stochastic
  optimization with decision-dependent distributions. Math. Oper. Res.
  48, pp. 954–998. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:227127621) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Fisch et al. (2024) A. Fisch, J. Maynez, R. A. Hofer, B. Dhingra, A.
  Globerson, and W. W. Cohen Stratified prediction-powered inference for
  hybrid language model evaluation. ArXiv abs/2406.04291. External
  Links: [Link](https://api.semanticscholar.org/CorpusID:270285671)
  Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Hardt and Mendler-Dünner (2023) M. Hardt and C. Mendler-Dünner
  Performative prediction: past and future. ArXiv abs/2310.16608.
  External Links:
  [Link](https://api.semanticscholar.org/CorpusID:264451701) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Hofer et al. (2024) R. A. Hofer, J. Maynez, B. Dhingra, A. Fisch, A.
  Globerson, and W. W. Cohen Bayesian prediction-powered inference.
  ArXiv abs/2405.06034. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:269740945) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Huang et al. (2015) J. L. Huang, M. Liu, and N. A. Bowling
  Insufficient effort responding: examining an insidious confound in
  survey data.. Journal of Applied Psychology 100 (3), pp. 828. Cited
  by: [item
  2](#S1.I1.i2.p1.1 "In Our contributions. ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Hyvärinen and Dayan (2005) A. Hyvärinen and P. Dayan Estimation of
  non-normalized statistical models by score matching.. Journal of
  Machine Learning Research 6 (4). Cited by:
  [§3.2](#S3.SS2.p2.2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Imbens and Wager (2019) G. Imbens and S. Wager Optimized regression
  discontinuity designs. Review of Economics and Statistics 101 (2),
  pp. 264–278. Cited by:
  [§3.3](#S3.SS3.p1.1 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Khorsandi et al. (2024) P. J. Khorsandi, R. Gupta, M. Mofakhami, S.
  Lacoste-Julien, and G. Gidel Tight lower bounds and improved
  convergence in performative prediction. ArXiv abs/2412.03671. External
  Links: [Link](https://api.semanticscholar.org/CorpusID:274514654)
  Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Kim and Perdomo (2022) M. P. Kim and J. C. Perdomo Making decisions
  under outcome performativity. ArXiv abs/2210.01745. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:252693241) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Li et al. (2025) X. Li, Y. Li, H. Zhong, L. Lei, and Z. Deng
  Statistical inference under performativity. In The Thirty-ninth Annual
  Conference on Neural Information Processing Systems, External Links:
  [Link](https://openreview.net/forum?id=vbn7VvloTd) Cited by:
  [§1.1](#S1.SS1.SSS0.Px3.p1.1 "Inference in performativity. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p9.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Lin and Zrnic (2023) L. Lin and T. Zrnic Plug-in performative
  optimization. arXiv preprint arXiv:2305.18728. Cited by:
  [§1](#S1.p4.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p4.2 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Mendler-Dünner et al. (2020) C. Mendler-Dünner, J. C. Perdomo, T.
  Zrnic, and M. Hardt Stochastic optimization for performative
  prediction. ArXiv abs/2006.06887. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:219636100) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Miller et al. (2021) J. Miller, J. C. Perdomo, and T. Zrnic Outside
  the echo chamber: optimizing the performative risk. In International
  Conference on Machine Learning, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:231942295) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p4.2 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Mofakhami et al. (2023) M. Mofakhami, I. Mitliagkas, and G. Gidel
  Performative prediction with neural networks. In International
  Conference on Artificial Intelligence and Statistics, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:253180829) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Munro et al. (2021) E. Munro, S. Wager, and K. Xu Treatment effects in
  market equilibrium. arXiv preprint arXiv:2109.11647 5. Cited by:
  [§3.2](#S3.SS2.p4.1 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Noack and Rothe (2019) C. Noack and C. Rothe Bias-aware inference in
  fuzzy regression discontinuity designs. arXiv preprint
  arXiv:1906.04631. Cited by:
  [§3.3](#S3.SS3.p1.1 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Oesterheld et al. (2023) C. Oesterheld, J. Treutlein, E. Cooper,
  and R. Hudson Incentivizing honest performative predictions with
  proper scoring rules. ArXiv abs/2305.17601. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:258960363) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Perdomo (2025) J. C. Perdomo Revisiting the predictability of
  performative, social events. ArXiv abs/2503.11713. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:277066781) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p4.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Perdomo et al. (2020) J. Perdomo, T. Zrnic, C. Mendler-Dünner, and M.
  Hardt Performative prediction. In International Conference on Machine
  Learning, pp. 7599–7609. Cited by:
  [§B.1.2](#A2.SS1.SSS2.p9.2.1 "Proof. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.2.4](#A2.SS2.SSS4.p1.2.1 "Proof of Corollary . ‣ B.2.4 Proof of Corollary ‣ B.2 Proofs for Section ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p4.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§1](#S1.p5.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§2.2](#S2.SS2.p1.1 "2.2 Repeated Gradient Descent ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§2.3](#S2.SS3.p2.1 "2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [Theorem
  2.1](#S2.Thmtheorem1 "Theorem 2.1 (Informal, adopted from [ , ]). ‣ 2.1 Repeated Risk Minimization ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [Theorem
  2.2](#S2.Thmtheorem2 "Theorem 2.2 (Informal, adopted from [ , ]). ‣ 2.2 Repeated Gradient Descent ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§3.3](#S3.SS3.SSS0.Px1 "Comparison with [ , ]. ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§3.3](#S3.SS3.SSS0.Px1.p1.1 "Comparison with [ , ]. ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§3.3](#S3.SS3.SSS0.Px1.p1.3 "Comparison with [ , ]. ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§4.3](#S4.SS3.p1.2 "4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Robins and Rotnitzky (1995) J. M. Robins and A. Rotnitzky
  Semiparametric efficiency in multivariate regression models with
  missing data. Journal of the American Statistical Association 90,
  pp. 122–129. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:121261196) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Singh et al. (2023) A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P.
  Patil, X. Garcia, P. J. Liu, J. Harrison, J. Lee, K. Xu, et al. Beyond
  human data: scaling self-training for problem-solving with language
  models. arXiv preprint arXiv:2312.06585. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Song et al. (2023) S. Song, Y. Lin, and Y. Zhou A general m-estimation
  theory in semi-supervised framework. Journal of the American
  Statistical Association 119, pp. 1065 – 1075. External Links:
  [Link](https://api.semanticscholar.org/CorpusID:257266562) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Talmor et al. (2019) A. Talmor, J. Herzig, N. Lourie, and J. Berant
  CommonsenseQA: a question answering challenge targeting commonsense
  knowledge. In Proceedings of the 2019 Conference of the North American
  Chapter of the Association for Computational Linguistics: Human
  Language Technologies, Volume 1 (Long and Short Papers), J.
  Burstein, C. Doran, and T. Solorio (Eds.), Minneapolis, Minnesota,
  pp. 4149–4158. External Links:
  [Link](https://aclanthology.org/N19-1421/),
  [Document](https://dx.doi.org/10.18653/v1/N19-1421) Cited by:
  [§5.3](#S5.SS3.p1.1 "5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Taori and Hashimoto (2022) R. Taori and T. Hashimoto Data feedback
  loops: model-driven amplification of dataset biases. In International
  Conference on Machine Learning, External Links:
  [Link](https://api.semanticscholar.org/CorpusID:252118610) Cited by:
  [§1.1](#S1.SS1.SSS0.Px1.p1.1 "Performative prediction. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Tsiatis (2006) A. A. Tsiatis Semiparametric theory and missing data.
  Springer, New York. External Links: ISBN 978-0-387-35458-3,
  [Link](https://doi.org/10.1007/0-387-37345-4) Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Van der Vaart (2000) A. W. Van der Vaart Asymptotic statistics. Vol.
  3, Cambridge university press. Cited by:
  [§B.1.2](#A2.SS1.SSS2.p9.1.1 "Proof. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.1.4](#A2.SS1.SSS4.p10.1.1 "Proof of Theorem . ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.1.4](#A2.SS1.SSS4.p12.1.1 "Proof of Theorem . ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.1.4](#A2.SS1.SSS4.p17.1.1 "Proof of Theorem . ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.1.4](#A2.SS1.SSS4.p17.2.1 "Proof of Theorem . ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.1.4](#A2.SS1.SSS4.p2.1.1 "Proof of Lemma . ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
  [§B.1.9](#A2.SS1.SSS9.p3.1.1 "Proof. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Viviano and Rudder (2024) D. Viviano and J. Rudder Policy design in
  experiments with unknown interference. arXiv preprint arXiv:2011.08174
  4. Cited by:
  [§3.2](#S3.SS2.p4.1 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Wang et al. (2022) X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S.
  Narang, A. Chowdhery, and D. Zhou Self-consistency improves chain of
  thought reasoning in language models. arXiv preprint arXiv:2203.11171.
  Cited by:
  [§C.3.1](#A3.SS3.SSS1.p2.1 "C.3.1 Dataset and response-bank construction. ‣ C.3 LLM Routing ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Wang et al. (2025) Y. Wang, X. Wang, C. Wang, J. Fang, Q. Wang, J.
  Chu, X. Meng, S. Yang, L. Qin, Y. Zhang, et al. Temporal
  self-rewarding language models: decoupling chosen-rejected via
  past-future. arXiv preprint arXiv:2508.06026. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Yeager et al. (2019) D. S. Yeager, P. Hanselman, G. M. Walton, J. S.
  Murray, R. Crosnoe, C. Muller, E. Tipton, B. Schneider, C. S.
  Hulleman, C. P. Hinojosa, et al. A national experiment reveals where a
  growth mindset improves achievement. Nature 573 (7774), pp. 364–369.
  Cited by: [item
  3](#S1.I1.i3.p2.1 "In Our contributions. ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Zelikman et al. (2022) E. Zelikman, Y. Wu, J. Mu, and N. Goodman Star:
  bootstrapping reasoning with reasoning. Advances in Neural Information
  Processing Systems 35, pp. 15476–15488. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Zrnic and Candès (2023) T. Zrnic and E. J. Candès
  Cross-prediction-powered inference. Proceedings of the National
  Academy of Sciences of the United States of America 121. External
  Links: [Link](https://api.semanticscholar.org/CorpusID:263134612)
  Cited by:
  [§1.1](#S1.SS1.SSS0.Px2.p1.1 "Prediction-powered inference. ‣ 1.1 Related Work ‣ 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
- Zweiger et al. (2026) A. Zweiger, J. Pari, H. Guo, Y. Kim, and P.
  Agrawal Self-adapting language models. Advances in Neural Information
  Processing Systems 38, pp. 74084–74115. Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

###### Contents

1.  [1
    Introduction](#S1 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [1.1 Related
        Work](#S1.SS1 "In 1 Introduction ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
2.  [2 Background and
    Setup](#S2 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [2.1 Repeated Risk
        Minimization](#S2.SS1 "In 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [2.2 Repeated Gradient
        Descent](#S2.SS2 "In 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [2.3 Assumptions and
        Notation](#S2.SS3 "In 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
3.  [3 Statistical Inference for
    RRM](#S3 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [3.1 Central Limit Theorem of
        RRM](#S3.SS1 "In 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [3.2 Estimation of the Asymptotic
        Covariance](#S3.SS2 "In 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [3.3 Confidence Regions for the Performative Stable
        Point](#S3.SS3 "In 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
4.  [4 Statistical Inference for
    RGD](#S4 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [4.1 Central Limit Theorem of
        RGD](#S4.SS1 "In 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [4.2 Estimation of the Asymptotic
        Covariance](#S4.SS2 "In 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [4.3 Confidence Regions for the Performative Stable
        Point](#S4.SS3 "In 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
5.  [5
    Experiment](#S5 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [5.1 Controlled Synthetic
        Studies](#S5.SS1 "In 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [5.1.1 Mild Performativity
            Feedback](#S5.SS1.SSS1 "In 5.1 Controlled Synthetic Studies ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [5.1.2 Uncertainty Propagation under Strong Performative
            Feedback.](#S5.SS1.SSS2 "In 5.1 Controlled Synthetic Studies ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [5.2 Semi-synthetic
        Experiment](#S5.SS2 "In 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [5.3 Self-improving LLM
        routing](#S5.SS3 "In 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    4.  [5.4 Feedback Strength and Uncertainty
        Propagation](#S5.SS4 "In 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
6.  [6
    Conclusion](#S6 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
7.  [7
    Acknowledgements](#S7 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
8.  [References](#bib "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
9.  [A Prediction-Powered Inference under
    Performativity](#A1 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [A.1 Setup of PPI under
        Performativity](#A1.SS1 "In Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [A.2 Central Limit Theorem of
        PPI](#A1.SS2 "In Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [A.3 Choice of the Weights and Covariance
        Estimation](#A1.SS3 "In Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    4.  [A.4 Confidence Regions for the Performative Stable
        Point](#A1.SS4 "In Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
10. [B Theoretical
    Details](#A2 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [B.1 Details of Section : Theory of Inference under
        Performativity](#A2.SS1 "In Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [B.1.1 Proof of Proposition
            ](#A2.SS1.SSS1 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [B.1.2 Consistency of the RERM
            Trajectory](#A2.SS1.SSS2 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        3.  [B.1.3 Preparation
            Lemmas](#A2.SS1.SSS3 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        4.  [B.1.4 Proof of CLT for RRM (Theorem
            )](#A2.SS1.SSS4 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        5.  [B.1.5 Proof of Corollary
            ](#A2.SS1.SSS5 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        6.  [B.1.6 Regularity for Score
            Matching](#A2.SS1.SSS6 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        7.  [B.1.7 Proof of Lemma
            ](#A2.SS1.SSS7 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        8.  [B.1.8 Policy
            Perturbation](#A2.SS1.SSS8 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        9.  [B.1.9 Proof of Theorem
            ](#A2.SS1.SSS9 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        10. [B.1.10 Proof of Consistency of Perturbation-based estimator
            (Theorem
            )](#A2.SS1.SSS10 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        11. [B.1.11 Proof of Corollary
            ](#A2.SS1.SSS11 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        12. [B.1.12 Discussion on Assumptions for Perturbation
            Method](#A2.SS1.SSS12 "In B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [B.2 Proofs for Section
        ](#A2.SS2 "In Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [B.2.1 Proof of Theorem
            ](#A2.SS2.SSS1 "In B.2 Proofs for Section ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [B.2.2 Proof of Theorem
            ](#A2.SS2.SSS2 "In B.2 Proofs for Section ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        3.  [B.2.3 Proof of Corollary
            ](#A2.SS2.SSS3 "In B.2 Proofs for Section ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        4.  [B.2.4 Proof of Corollary
            ](#A2.SS2.SSS4 "In B.2 Proofs for Section ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [B.3 Details of Appendix : Theory of Prediction-Powered
        Inference under
        Performativity](#A2.SS3 "In Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [B.3.1 Proof of Theorem
            ](#A2.SS3.SSS1 "In B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [B.3.2 Proof of Corollary
            ](#A2.SS3.SSS2 "In B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        3.  [B.3.3 Proof of Corollary
            ](#A2.SS3.SSS3 "In B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
11. [C Experimental
    Details](#A3 "In Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    1.  [C.1 Pure-Synthetic
        Experiment](#A3.SS1 "In Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [C.1.1 Data
            Generation](#A3.SS1.SSS1 "In C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [C.1.2 Score
            estimation.](#A3.SS1.SSS2 "In C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        3.  [C.1.3 Contraction and bias
            bounds.](#A3.SS1.SSS3 "In C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        4.  [C.1.4 Inference and
            evaluation.](#A3.SS1.SSS4 "In C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        5.  [C.1.5 Prediction-Powered
            Inference](#A3.SS1.SSS5 "In C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        6.  [C.1.6 Score matching
            diagnostics](#A3.SS1.SSS6 "In C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    2.  [C.2 Semi-Synthetic
        Experiment](#A3.SS2 "In Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [C.2.1 Dataset and deployment
            population.](#A3.SS2.SSS1 "In C.2 Semi-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [C.2.2 Deployment and experimental
            protocol.](#A3.SS2.SSS2 "In C.2 Semi-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        3.  [C.2.3 Inference and
            evaluation.](#A3.SS2.SSS3 "In C.2 Semi-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    3.  [C.3 LLM
        Routing](#A3.SS3 "In Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        1.  [C.3.1 Dataset and response-bank
            construction.](#A3.SS3.SSS1 "In C.3 LLM Routing ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        2.  [C.3.2 Deployment and experimental
            protocol.](#A3.SS3.SSS2 "In C.3 LLM Routing ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
        3.  [C.3.3 Inference and
            evaluation.](#A3.SS3.SSS3 "In C.3 LLM Routing ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
    4.  [C.4 Jacobian estimation
        diagnostics.](#A3.SS4 "In Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

## Appendix A Prediction-Powered Inference under Performativity

In this section, we extend prediction-powered inference (PPI) to this
dynamic performative setting. First, Section
[A.1](#A1.SS1 "A.1 Setup of PPI under Performativity ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
introduces the setup of PPI under performativity, Section
[A.2](#A1.SS2 "A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives the central limit theorem for population trajectory, Section
[A.3](#A1.SS3 "A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
describes a weight tuning procedure and an estimator of the covariance,
and Section
[A.4](#A1.SS4 "A.4 Confidence Regions for the Performative Stable Point ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives confidence regions for $`\theta_{\mathrm{PS}}`$. We retain the
notation in Section
[2](#S2 "2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

### A.1 Setup of PPI under Performativity

Prediction-powered inference (PPI) combines labeled observations with
additional unlabeled observations and a prediction rule
$`f:\mathcal{X}\to\mathcal{Y}`$ \[[Angelopoulos et al.,
2023a](#bib.bib14)\]. We treat $`f`$ as fixed throughout and set
$`\widehat{\theta}_{0}^{\mathrm{PPI}}=\theta_{0}`$. At round $`t`$,
conditional on all preceding observations and estimators, let

|  |  |  |
|----|----|----|
|  |
``` math
\{(X_{t,i},Y_{t,i})\}_{i=1}^{n_{t}}\quad\text{and}\quad\{X^{u}_{t,j}\}_{j=1}^{N_{t}}
``` |  |

be independent samples from
$`\mathcal{D}(\widehat{\theta}_{t}^{\mathrm{PPI}})`$ and its
$`X`$-marginal, respectively. For $`\lambda\in[0,1]`$, define

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\theta}_{t+1}^{\mathrm{PPI}}(\lambda):=\arg\min_{\theta\in\Theta}\left[\frac{1}{n_{t}}\sumop\displaylimits_{i=1}^{n_{t}}\bigl\{\ell(X_{t,i},Y_{t,i};\theta)-\lambda\ell(X_{t,i},f(X_{t,i});\theta)\bigr\}+\frac{\lambda}{N_{t}}\sumop\displaylimits_{j=1}^{N_{t}}\ell(X^{u}_{t,j},f(X^{u}_{t,j});\theta)\right].
``` |  |

Given a data-dependent $`\widehat{\lambda}_{t+1}\in[0,1]`$, define
$`\widehat{\theta}_{t+1}^{\mathrm{PPI}}:=\widehat{\theta}_{t+1}^{\mathrm{PPI}}(\widehat{\lambda}_{t+1}).`$
For fixed $`\lambda`$, the conditional population minimizer is
$`G(\widehat{\theta}_{t}^{\mathrm{PPI}})`$, so the deterministic
population trajectory remains $`\theta_{t+1}=G(\theta_{t})`$. Setting
$`\widehat{\lambda}_{t+1}=0`$ gives the empirical RRM update.

In the following context, we write $`\widehat{\theta}_{t}`$ and
$`\widehat{\theta}_{t}(\lambda)`$ for
$`\widehat{\theta}_{t}^{\mathrm{PPI}}`$ and
$`\widehat{\theta}_{t}^{\mathrm{PPI}}(\lambda)`$, and use $`V_{t}`$,
$`\widehat{V}_{t}`$, and $`\widehat{\mathcal{R}}_{t}`$ for the PPI
counterparts of the quantities in Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

### A.2 Central Limit Theorem of PPI

As noted in Section
[A.1](#A1.SS1 "A.1 Setup of PPI under Performativity ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for a fixed weight $`\lambda`$, the conditional population minimizer of
the PPI objective at round $`t`$ is $`G(\widehat{\theta}_{t})`$. The
decomposition in Section
[2.1](#S2.SS1 "2.1 Repeated Risk Minimization ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
therefore still applies,
$`\widehat{\theta}_{t+1}-\theta_{t+1}=\bigl(\widehat{\theta}_{t+1}-G(\widehat{\theta}_{t})\bigr)+\bigl(G(\widehat{\theta}_{t})-G(\theta_{t})\bigr),`$
and the second term is the same as for RERM. Only the first term
changes. For $`\widetilde{\theta},\theta\in\Theta`$ and
$`\lambda\in[0,1]`$, define

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\Sigma_{\widetilde{\theta}}(\theta;\lambda):={}`$ | $`\displaystyle H_{\widetilde{\theta}}(\theta)^{-1}\bigl\{\operatorname{Cov}_{Z\sim\mathcal{D}(\widetilde{\theta})}\bigl(\nabla\ell(Z;\theta)-\lambda\nabla\ell(X,f(X);\theta)\bigr)`$ |  |
|  |  | $`\displaystyle+r\lambda^{2}\operatorname{Cov}_{Z\sim\mathcal{D}(\widetilde{\theta})}\bigl(\nabla\ell(X,f(X);\theta)\bigr)\bigr\}H_{\widetilde{\theta}}(\theta)^{-1},`$ |  |

where $`Z=(X,Y)`$. By the M-estimation results of [Angelopoulos et al.
\[2023a\]](#bib.bib14), for a fixed $`\lambda`$, the conditional
distribution of
$`\sqrt{n}\{\widehat{\theta}_{t+1}(\lambda)-G(\widehat{\theta}_{t})\}`$
is approximately
$`\mathcal{N}\bigl(0,\Sigma_{\widehat{\theta}_{t}}(G(\widehat{\theta}_{t});\lambda)\bigr)`$.
The two covariances come from the labeled observations and from the
unlabeled covariates, respectively. When $`\lambda=0`$, the unlabeled
covariates are not used and
$`\Sigma_{\widetilde{\theta}}(\theta;0)=\Sigma_{\widetilde{\theta}}(\theta)`$.

###### Theorem A.1 (Central limit theorem for PPI).

Suppose Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
and
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold, $`n/N\to r\in[0,\infty)`$, and
$`\widehat{\lambda}_{t}\xrightarrow{P}\lambda_{t}`$ for each
$`t\geqslant 1`$, where $`\lambda_{t}\in[0,1]`$ are deterministic. Then
for each fixed $`t\geqslant 1`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\mathcal{N}(0,V_{t}),
``` |  |

where $`V_{0}=0`$ and
$`V_{t+1}=\nabla G(\theta_{t})\,V_{t}\,\nabla G(\theta_{t})^{\top}+\Sigma_{\theta_{t}}(\theta_{t+1};\lambda_{t+1}),t\geqslant 0.`$

Assumption
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
makes the PPI objective strongly convex for every $`\lambda\in[0,1]`$
and imposes the moment conditions of Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
on the loss at the predicted labels. The recursion is that of Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
with $`\Sigma_{\theta_{t}}(\theta_{t+1})`$ replaced by
$`\Sigma_{\theta_{t}}(\theta_{t+1};\lambda_{t+1})`$; setting
$`\widehat{\lambda}_{t}=0`$ recovers Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

### A.3 Choice of the Weights and Covariance Estimation

Following [Angelopoulos et al. \[2023a\]](#bib.bib14), we choose the
weight in each round to minimize the trace of the covariance added in
that round,

|  |  |  |
|----|----|----|
|  |
``` math
\lambda_{t+1}^{*}:=\argmin_{\lambda\in[0,1]}\operatorname{Tr}\Sigma_{\theta_{t}}(\theta_{t+1};\lambda),\qquad t\geqslant 0.
``` |  |

Since the inherited term in Theorem
[A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
does not depend on $`\lambda_{t+1}`$, $`\lambda_{t+1}^{*}`$ also
minimizes $`\operatorname{Tr}V_{t+1}`$ given
$`\lambda_{1},\ldots,\lambda_{t}`$. Since $`\lambda=0`$ is allowed,
$`\operatorname{Tr}\Sigma_{\theta_{t}}(\theta_{t+1};\lambda_{t+1}^{*})\leqslant\operatorname{Tr}\Sigma_{\theta_{t}}(\theta_{t+1})`$:
the covariance added in each round has no larger trace than for RERM.

We estimate $`\lambda_{t+1}^{*}`$ from batch $`t`$. Expand the first
covariance in $`\Sigma_{\widetilde{\theta}}(\theta;\lambda)`$ into the
covariance of $`\nabla\ell(Z;\theta)`$, its cross-covariances with
$`\nabla\ell(X,f(X);\theta)`$, and $`\lambda^{2}`$ times the covariance
of $`\nabla\ell(X,f(X);\theta)`$. For $`\theta\in\Theta`$ and
$`\lambda\in[0,1]`$, let
$`\widehat{\Sigma}_{\widehat{\theta}_{t}}(\theta;\lambda)`$ be obtained
from $`\Sigma_{\widehat{\theta}_{t}}(\theta;\lambda)`$ by replacing
$`H_{\widehat{\theta}_{t}}(\theta)`$ with the average Hessian
$`n^{-1}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{t,i};\theta)`$,
$`r`$ with $`n/N`$, and each covariance with its sample version: those
involving $`\nabla\ell(Z;\theta)`$ use the $`n`$ labeled observations of
batch $`t`$, and the covariance of $`\nabla\ell(X,f(X);\theta)`$ uses
all $`n+N`$ covariates $`X_{t,i}`$ and $`X^{u}_{t,j}`$. Choose any
$`\widetilde{\lambda}_{t+1}\in[0,1]`$ and compute the preliminary
estimator $`\widehat{\theta}_{t+1}(\widetilde{\lambda}_{t+1})`$. We set

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\lambda}_{t+1}:=\argmin_{\lambda\in[0,1]}\operatorname{Tr}\widehat{\Sigma}_{\widehat{\theta}_{t}}\bigl(\widehat{\theta}_{t+1}(\widetilde{\lambda}_{t+1});\lambda\bigr),\qquad\widehat{\theta}_{t+1}:=\widehat{\theta}_{t+1}(\widehat{\lambda}_{t+1}).
``` |  |

Both traces are convex quadratic functions of $`\lambda`$, so both
minimizers have closed forms (Appendix
[B.3](#A2.SS3 "B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).
Following Theorem
[A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
set $`\widehat{V}_{0}=0`$ and

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\widehat{V}_{t+1}:=\widehat{\nabla G}_{t}\,\widehat{V}_{t}\,\widehat{\nabla G}_{t}^{\top}+\widehat{\Sigma}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1};\widehat{\lambda}_{t+1}),\qquad t\geqslant 0,
``` |  | (5) |

where $`\widehat{\nabla G}_{t}`$ is computed as in Section
[3.2](#S3.SS2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
from the labeled observations of batch $`t`$ and the perturbed batches.

###### Corollary A.2.

Suppose the conditions of Theorem
[3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Assumption
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold, $`n/N\to r\in[0,\infty)`$, and
$`\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t-1})}(\nabla\ell(X,f(X);\theta_{t}))\neq 0`$
for each $`t\geqslant 1`$. Then for each fixed $`t\geqslant 1`$,
$`\widehat{\lambda}_{t}\xrightarrow{P}\lambda_{t}^{*}`$, so Theorem
[A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
holds with $`\lambda_{t}=\lambda_{t}^{*}`$, and
$`\widehat{V}_{t}\xrightarrow{P}V_{t}`$. If, in addition, $`V_{t}`$ is
nonsingular, then the region $`\widehat{\mathcal{R}}_{t}`$ of Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
computed from $`\widehat{\theta}_{t}`$ and $`\widehat{V}_{t}`$ in
([5](#A1.E5 "Equation 5 ‣ A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
satisfies
$`\lim_{n\to\infty}\mathbb P(\theta_{t}\in\widehat{\mathcal{R}}_{t})=1-\delta`$.

### A.4 Confidence Regions for the Performative Stable Point

The population trajectory is the RRM trajectory, so the enlargement in
Section
[3.3](#S3.SS3 "3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
applies similarly.

###### Corollary A.3 (Confidence region for $`\theta_{\mathrm{PS}}`$).

Suppose the conditions of Corollary
[A.2](#A1.Thmtheorem2 "Corollary A.2. ‣ A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold, $`\varepsilon\beta<\gamma`$, and
$`\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}\leqslant B`$. Then

|  |  |  |
|----|----|----|
|  |
``` math
\liminf_{n\to\infty}\mathbb P\Bigl\{\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}\Bigl(0,B\Bigl(\frac{\varepsilon\beta}{\gamma}\Bigr)^{t}\Bigr)\Bigr\}\geqslant 1-\delta.
``` |  |

## Appendix B Theoretical Details

This appendix gives the assumptions and proofs omitted from Sections
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")–[A](#A1 "Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
We first prove the RERM limit theory and the consistency of its
covariance estimator, and then give the corresponding arguments for RGD
and PPI.

When $`\widetilde{\theta}`$ is fixed conditional on the preceding
observations, $`\widehat{\mathbb E}_{\widetilde{\theta},m}`$ denotes the
empirical average of $`m`$ conditionally i.i.d. observations from
$`\mathcal{D}(\widetilde{\theta})`$.

### B.1 Details of Section [3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"): Theory of Inference under Performativity

We provide the omitted details in Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

#### B.1.1 Proof of Proposition [3.1](#S3.Thmtheorem1 "Proposition 3.1. ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Proposition [3.1](#S3.Thmtheorem1 "Proposition 3.1. ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

By Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
the map

|  |  |  |
|----|----|----|
|  |
``` math
(\widetilde{\theta},\theta)\longmapsto\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;\theta)]
``` |  |

is continuously differentiable. Its Jacobian in $`\theta`$ is
$`H_{\widetilde{\theta}}(\theta)`$, and
$`H_{\widetilde{\theta}}(\theta)\succeq\gamma I_{d}`$ by Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d).
Since $`G(\widetilde{\theta})`$ is the unique solution of

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;G(\widetilde{\theta}))]=0,
``` |  |

the implicit function theorem gives

|  |  |  |
|----|----|----|
|  |
``` math
\nabla G(\theta)=-H_{\theta}(G(\theta))^{-1}\nabla_{\widetilde{\theta}}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;G(\theta))]\Big|_{\widetilde{\theta}=\theta}.
``` |  |

It remains to compute the last derivative. For $`\widetilde{\theta}`$ in
a compact convex neighborhood $`K`$ of $`\theta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;G(\theta))]=\mathbb E_{Z\sim\mathcal{D}(\theta)}\left[\nabla\ell(Z;G(\theta))\frac{p(Z;\widetilde{\theta})}{p(Z;\theta)}\right].
``` |  |

The gradient of the ratio in $`\widetilde{\theta}`$ is

|  |  |  |
|----|----|----|
|  |
``` math
\frac{p(Z;\widetilde{\theta})}{p(Z;\theta)}\nabla_{\theta}\log p(Z;\widetilde{\theta}).
``` |  |

By Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a)
and the mean value theorem, its norm is at most
$`L_{s,K}\exp\{L_{s,K}\operatorname{diam}(K)\}`$, uniformly over
$`\widetilde{\theta}\in K`$. Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
bounds $`\|\nabla\ell(Z;G(\theta))\|_{2}`$ by an integrable function.
Dominated convergence therefore allows differentiation under the
expectation. Since the density ratio equals one at
$`\widetilde{\theta}=\theta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\nabla_{\widetilde{\theta}}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;G(\theta))]\Big|_{\widetilde{\theta}=\theta}=\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\nabla\ell(Z;G(\theta))\nabla_{\theta}\log p(Z;\theta)^{\top}\bigr].
``` |  |

Substituting this identity into the preceding display proves the result.
∎

#### B.1.2 Consistency of the RERM Trajectory

Let us denote:

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\mathcal{L}_{\check{\theta}}(\theta):={\operatorname*{\mathbb{E}}}_{\check{\theta}}\ell(z;\theta),\quad\mathcal{L}_{\check{\theta},n}(\theta):=\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\ell(z_{i};\theta),`$ |  |

where the samples $`z_{i}=(x_{i},y_{i})\sim\mathcal{D}(\check{\theta})`$
are i.i.d. drawn from the distribution under $`\check{\theta}`$.

###### Lemma B.1.

Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each $`\delta>0`$ and compact subsets
$`K_{1},K_{2}\subset\Theta=\mathbb{R}^{d}`$ , the following convergence
holds as $`n\to\infty`$:

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\check{\theta}\in K_{1}}P_{\check{\theta}}\left(\sup_{\theta\in K_{2}}\big|\mathcal{L}_{\check{\theta},n}(\theta)-\mathcal{L}_{\check{\theta}}(\theta)\big|>\delta\right)\to 0,
``` |  |

###### Proof.

Define $`K:=K_{1}\cup K_{2}`$. For $`r>0`$, by the compactness of
$`K_{2}`$, let $`\{\theta_{1},\ldots,\theta_{N}\}`$ be an $`r`$-net of
$`K_{2}`$. Then for each $`{\check{\theta}}\in K_{1},\theta\in K_{2}`$,
there exists a $`\theta_{j}`$, such that
$`\|\theta-\theta_{j}\|_{2}\leqslant r`$, thus recall $`\ell(Z;\theta)`$
is local $`L_{K}(Z)`$-Lipschitz,

|  |  |  |
|----|----|----|
|  |
``` math
\big|\mathcal{L}_{\check{\theta},n}(\theta)-\mathcal{L}_{\check{\theta},n}(\theta_{j})\big|\leqslant\widehat{\mathbb{E}}_{\check{\theta},n}L_{K}(Z)\|\theta_{j}-\theta\|_{2},\quad\big|\mathcal{L}_{\check{\theta}}(\theta)-\mathcal{L}_{\check{\theta}}(\theta_{j})\big|\leqslant{\mathbb{E}}_{\check{\theta}}L_{K}(Z)\|\theta_{j}-\theta\|_{2},
``` |  |

we have

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\sup_{\theta\in K_{2}}\big|\mathcal{L}_{\check{\theta},n}(\theta)-\mathcal{L}_{\check{\theta}}(\theta)\big|\leqslant\sup_{1\leqslant j\leqslant N}\big|\mathcal{L}_{\check{\theta},n}(\theta_{j})-\mathcal{L}_{\check{\theta}}(\theta_{j})\big|+r\big(\widehat{\mathbb{E}}_{\check{\theta},n}L_{K}(Z)+\mathbb{E}_{\check{\theta}}L_{K}(Z)\big).
``` |  | (6) |

For the first term on the right hand side of the above formula, note
that $`K=K_{1}\cup K_{2}`$, by Chebyshev’s inequality and a union bound,

|  |  |  |
|----|----|----|
|  |
``` math
P_{{\check{\theta}}}(\max_{1\leqslant j\leqslant N}|\mathcal{L}_{{\check{\theta}},n}(\theta_{j})-\mathcal{L}_{{\check{\theta}}}(\theta_{j})|>\delta/2)\leqslant\frac{4N}{n\delta^{2}}\max_{1\leqslant j\leqslant N}\mathbb{E}_{{\check{\theta}}}|\ell(Z;\theta_{j})|^{2}\leqslant\frac{4N}{n\delta^{2}}\sup_{{\check{\theta}}\in K}\mathbb{E}_{{\check{\theta}}}|A_{K}(Z)|^{2},
``` |  |

when $`n\to\infty`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{{\check{\theta}}\in K_{1}}P_{{\check{\theta}}}(\max_{1\leqslant j\leqslant N}|\mathcal{L}_{\check{\theta},n}(\theta_{j})-\mathcal{L}_{{\check{\theta}}}(\theta_{j})|>\delta)\to 0.
``` |  |

For the second term on the right hand side of formula Eq.
([6](#A2.E6 "Equation 6 ‣ Proof. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
for fixed $`r>0`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{{\check{\theta}}\in K_{1}}P_{{\check{\theta}}}(|\widehat{\mathbb{E}}_{{\check{\theta}},n}L_{K}(Z)-\mathbb{E}_{{\check{\theta}}}L_{K}(Z)|>{\delta})\leqslant\frac{1}{n\delta^{2}}\sup_{{\check{\theta}}\in K_{1}}\mathbb{E}_{{\check{\theta}}}|L_{K}(Z)|^{2}.
``` |  |

Then for each fixed $`\delta`$, select $`r>0`$ sufficiently small, so
that
$`r\sup_{\check{\theta}\in K}\mathbb{E}_{{\check{\theta}}}|L_{K}(Z)|<\frac{\delta}{4}`$,
then

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\sup_{{\check{\theta}}\in K_{1}}P_{{\check{\theta}}}(r\big(\widehat{\mathbb{E}}_{{\check{\theta}},n}L_{K}(Z)+\mathbb{E}_{{\check{\theta}}}L_{K}(Z)\big)>{\delta})`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\sup_{{\check{\theta}}\in K_{1}}P_{{\check{\theta}}}(r\big(\widehat{\mathbb{E}}_{{\check{\theta}},n}L_{K}(Z)-\mathbb{E}_{{\check{\theta}}}L_{K}(Z)\big)>{\delta}/2)`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\frac{4r^{2}N}{n\delta^{2}}\sup_{{\check{\theta}}\in K_{1}}\mathbb{E}_{{\check{\theta}}}|L_{K}(Z)|^{2}.`$ |  |

By the above inequalities for the terms on the right hand side of
formula Eq.
([6](#A2.E6 "Equation 6 ‣ Proof. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
we have

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{{\check{\theta}}\in K_{1}}P_{\check{\theta}}\left(\sup_{\theta\in K_{2}}\big|\widehat{\mathbb{E}}_{{\check{\theta}},n}\ell(Z;\theta)-\mathbb{E}_{\check{\theta}}\ell(Z;\theta)\big|>\delta\right)\to 0.
``` |  |

∎

###### Lemma B.2.

Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for a compact subset $`K\subset\Theta`$, then for each $`n>0`$, denote
$`\widehat{G}({\check{\theta}})=\argmin_{\theta\in\Theta}\mathcal{L}_{{\check{\theta}},n}(\theta)`$,
when $`R\to\infty`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{R\to\infty}\sup_{n\geqslant 1}\sup_{{\check{\theta}}\in K}P_{{\check{\theta}}}(\|\widehat{G}({\check{\theta}})-G({\check{\theta}})\|_{2}>R)=0.
``` |  |

###### Proof of Lemma [B.2](#A2.Thmtheorem2 "Lemma B.2. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

By the optimality condition,

|  |  |  |
|----|----|----|
|  |
``` math
\big\langle\nabla_{\theta}{\mathcal{L}}_{{\check{\theta}},n}(\widehat{G}({\check{\theta}})),G({\check{\theta}})-\widehat{G}({\check{\theta}})\big\rangle\leqslant 0.
``` |  |

By strong convexity, we have

|  |  |  |
|----|----|----|
|  |
``` math
\big\langle\nabla_{\theta}{\mathcal{L}}_{{\check{\theta}},n}(\widehat{G}({\check{\theta}}))-\nabla_{\theta}{\mathcal{L}}_{{\check{\theta}},n}({G}({\check{\theta}})),\widehat{G}({\check{\theta}})-{G}({\check{\theta}})\big\rangle\geqslant\gamma\|G({\check{\theta}})-\widehat{G}({\check{\theta}})\|_{2}^{2},
``` |  |

This implies

|  |  |  |
|----|----|----|
|  |
``` math
\big\langle\nabla_{\theta}{\mathcal{L}}_{{\check{\theta}},n}({G}({\check{\theta}})),G({\check{\theta}})-\widehat{G}({\check{\theta}})\big\rangle\geqslant\gamma\|G({\check{\theta}})-\widehat{G}({\check{\theta}})\|_{2}^{2},
``` |  |

Thus,

|  |  |  |
|----|----|----|
|  |
``` math
\|\nabla_{\theta}\mathcal{L}_{{\check{\theta}},n}(G({\check{\theta}}))\|_{2}\geqslant\gamma\|G({\check{\theta}})-\widehat{G}({\check{\theta}})\|.
``` |  |

Therefore, for compact set $`K`$, there exists $`R_{1}>0`$, such that
$`\|G({\check{\theta}})\|\leqslant R_{1}`$, and denote
$`K_{1}:=B(0,R_{1})\cup K`$. By Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each $`R>0`$,

|  |  |  |
|----|----|----|
|  |
``` math
P_{{\check{\theta}}}\bigg(\big\|\nabla_{\theta}\mathcal{L}_{{\check{\theta}},n}(G({\check{\theta}}))\big\|_{2}>R\bigg)\leqslant\frac{1}{R^{2}}\mathbb{E}_{{\check{\theta}}}\big[\|\nabla_{\theta}\mathcal{L}_{\check{\theta},n}(G({\check{\theta}}))\|_{2}^{2}\big]\leqslant\frac{1}{R^{2}}\sup_{{\check{\theta}}\in K_{1}}\mathbb{E}_{{\check{\theta}}}\big[L_{K_{1}}(Z)^{2}\big],
``` |  |

let $`R\to\infty`$, we know that

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{n\geqslant 1}\sup_{{\check{\theta}}\in K}P_{{\check{\theta}}}(\|\widehat{G}({\check{\theta}})-G({\check{\theta}})\|_{2}>R)\to 0
``` |  |

∎

###### Proposition B.3.

Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\widehat{\theta}_{t}\xrightarrow{P}\theta_{t}`$ for each fixed
$`t\geqslant 1`$.

###### Proof.

We use $`\widehat{G}`$ defined in Lemma
[B.2](#A2.Thmtheorem2 "Lemma B.2. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and prove the result by induction on $`t`$. By consistency of
$`M`$-estimator (Theorem 5.9 in \[[Van der Vaart, 2000](#bib.bib15)\]),
we have $`\widehat{\theta}_{1}\xrightarrow{P}\theta_{1}`$. We assume
that as $`n\to\infty`$,
$`\widehat{\theta}_{t-1}\xrightarrow{P}\theta_{t-1}`$, and set off to
prove that $`\widehat{\theta}_{t}\xrightarrow{P}\theta_{t}`$.

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\theta_{t}-\widehat{\theta}_{t}\|`$ | $`\displaystyle=\|G(\theta_{t-1})-\widehat{G}(\widehat{\theta}_{t-1})\|`$ |  |
|  |  | $`\displaystyle\leqslant\|G(\widehat{\theta}_{t-1})-\widehat{G}(\widehat{\theta}_{t-1})\|+\|G(\theta_{t-1})-G(\widehat{\theta}_{t-1})\|`$ |  |
|  |  | $`\displaystyle\leqslant\|G(\widehat{\theta}_{t-1})-\widehat{G}(\widehat{\theta}_{t-1})\|+\varepsilon\frac{\beta}{\gamma}\|\theta_{t-1}-\widehat{\theta}_{t-1}\|,`$ |  |

where the last inequality follows from the results derived by [Perdomo
et al. \[2020\]](#bib.bib12), under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we have
$`\|G(\theta)-G(\theta^{\prime})\|\leqslant\frac{\varepsilon\beta}{\gamma}\|\theta-\theta^{\prime}\|`$.

For any $`\delta>0`$, by inductive assumption,
$`\widehat{\theta}_{t-1}\xrightarrow{P}\theta_{t-1}`$, thus there exists
compact subset $`K_{1}\subset\Theta`$, such that

|     |                                                 |     |
|-----|-------------------------------------------------|-----|
|     |
       ``` math
       P(\widehat{\theta}_{t-1}\notin K_{1})<\delta/2.
       ```                                              |     |

By Lemma
[B.2](#A2.Thmtheorem2 "Lemma B.2. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
there exists compact $`K_{2}\subset\Theta`$, such that
$`G(K_{1})\subset K_{2}`$ and that
$`P(\widehat{\theta}_{t-1}\in K_{1},\widehat{G}(\widehat{\theta}_{t-1})\notin K_{2})<\delta`$,
thus in the following, we only need to prove the situation where
$`\widehat{\theta}_{t-1}\in K_{1}`$ and
$`\widehat{G}(\widehat{\theta}_{t-1})\in K_{2}`$.

Note that
$`\mathcal{L}_{{\check{\theta}},n}(\widehat{G}({\check{\theta}}))-\mathcal{L}_{{\check{\theta}},n}(G({\check{\theta}}))\leqslant 0`$,
we have

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\mathcal{L}_{{\check{\theta}}}(\widehat{G}({\check{\theta}}))-\mathcal{L}_{{\check{\theta}}}({G}({\check{\theta}}))`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\big|\mathcal{L}_{{\check{\theta}}}(\widehat{G}({\check{\theta}}))-\mathcal{L}_{{\check{\theta}},n}(\widehat{G}({\check{\theta}}))\big|+\mathcal{L}_{{\check{\theta}},n}(\widehat{G}({\check{\theta}}))-\mathcal{L}_{{\check{\theta}},n}(G({\check{\theta}}))+\big|\mathcal{L}_{{\check{\theta}},n}({G}({\check{\theta}}))-\mathcal{L}_{{\check{\theta}}}({G}({\check{\theta}}))\big|`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle 2\sup_{\theta\in K_{2}}\big|\mathcal{L}_{{\check{\theta}},n}(\theta)-\mathcal{L}_{{\check{\theta}}}(\theta)\big|,`$ |  |

and
$`\mathcal{L}_{{\check{\theta}}}(\widehat{G}({\check{\theta}}))-\mathcal{L}_{{\check{\theta}}}({G}({\check{\theta}}))\geqslant\frac{\gamma}{2}\|\widehat{G}({\check{\theta}})-G({\check{\theta}})\|^{2}_{2}`$,
combine these two inequalities, we obtain that for each
$`{\check{\theta}}\in K_{1}`$,

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{G}({\check{\theta}})-G({\check{\theta}})\|_{2}^{2}\leqslant\frac{4}{\gamma}\sup_{\theta\in K_{2}}\big|\mathcal{L}_{{\check{\theta}},n}(\theta)-\mathcal{L}_{{\check{\theta}}}(\theta)\big|.
``` |  |

We then have for each $`\delta>0`$,

|  |  |  |
|----|----|----|
|  |
``` math
P_{{\check{\theta}}}(\|\widehat{G}({\check{\theta}})-G({\check{\theta}})\|>\delta)<P_{\check{\theta}}\left(\sup_{\theta\in K_{2}}\big|\mathcal{L}_{{\check{\theta}},n}(\theta)-\mathcal{L}_{{\check{\theta}}}(\theta)\big|>\frac{\gamma\delta^{2}}{4}\right),
``` |  |

by Lemma
[B.1](#A2.Thmtheorem1 "Lemma B.1. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the following uniform convergence formula holds,

|  |  |  |
|----|----|----|
|  |
``` math
P\bigg(\sup_{\theta\in K_{2}}\big|\mathcal{L}_{{\check{\theta}},n}(\theta)-\mathcal{L}_{{\check{\theta}}}(\theta)\big|>\frac{\gamma\delta^{2}}{4}\bigg|\widehat{\theta}_{t-1}\bigg)\leqslant\sup_{{\check{\theta}}\in K_{1}}P_{\check{\theta}}\left(\sup_{\theta\in K_{2}}\big|\mathcal{L}_{{\check{\theta}},n}(\theta)-\mathcal{L}_{{\check{\theta}}}(\theta)\big|>\frac{\gamma\delta^{2}}{4}\right)\to 0,
``` |  |

it implies
$`\|\widehat{G}(\widehat{\theta}_{t-1})-{G}(\widehat{\theta}_{t-1})\|_{2}\xrightarrow{P}0`$.
∎

#### B.1.3 Preparation Lemmas

In this subsection, we provide some technical lemmas that will be used
in our proof of Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

###### Lemma B.4.

For a function sequence $`R_{n}:\mathbb{R}^{d}\to\mathbb{C}`$, assume:

1.  1.
    There exists a constant $`C>0`$, such that,
    $`\sup_{n}\sup_{x\in\mathbb{R}^{d}}|R_{n}(x)|<C`$;
2.  2.
    On any compact subset $`K\subset\mathbb{R}^{d}`$,
    $`\sup_{\xi\in K}|R_{n}(\xi)|\to 0`$.

Then for any random vector sequence $`\{\xi_{n}\}`$ taking values in
$`\mathbb{R}^{d}`$ and $`\xi_{n}=O_{P}(1)`$,

|     |                                  |     |
|-----|----------------------------------|-----|
|     |
       ``` math
       \mathbb{E}|R_{n}(\xi_{n})|\to 0.
       ```                               |     |

###### Proof of Lemma [B.4](#A2.Thmtheorem4 "Lemma B.4. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

For any $`\varepsilon>0`$, take a compact subset
$`K\subset{\mathbb R}^{d}`$ such that
$`P(\xi_{n}\notin K)\leqslant\varepsilon`$ holds for each
$`n\in\mathbb{N}`$, and take $`N`$ sufficiently large, such that
$`\sup_{\xi\in K}|R_{n}(\xi)|\leqslant\varepsilon`$ holds for
$`n\geqslant N`$. Then we have

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb{E}|R_{n}(\xi_{n})|\leqslant\sup_{\xi\in K}|R_{n}(\xi)|+CP(\xi_{n}\notin K)\leqslant(1+C)\varepsilon,\quad n\geqslant N.
``` |  |

∎

###### Lemma B.5.

Assume that $`\check{\theta}_{0}\in\Theta`$ is a fixed parameter in
$`\Theta`$, and $`\{\check{\theta}_{n}\}_{n=1}^{\infty}`$ is a sequence
of random variables such that
$`\sqrt{n}(\check{\theta}_{n}-\check{\theta}_{0})\to\mathcal{N}(0,A_{1})`$
for a positive semidefinite matrix $`A_{1}\in\mathbb{R}^{d\times d}`$.
Conditional on $`\check{\theta}_{n}`$, let
$`\{Z_{1,n}^{(j)}\}_{j=1}^{n}`$ be i.i.d. from
$`\mathcal{D}(\check{\theta}_{n})`$. For a function
$`f_{\theta}:\mathcal{Z}\to\mathbb{R}^{q}`$, where $`q`$ is a positive
integer, define $`F({\theta})=\mathbb{E}_{\theta}[f_{\theta}(Z)]`$,
$`V_{f}(\theta)=\mathrm{Cov}_{\theta}[f_{\theta}(Z)]`$. If for any
compact subset $`K\subset\Theta`$, $`F(\theta)`$ is bounded and
continuously differentiable on an open neighborhood of $`K`$, and
$`V_{f}(\theta)`$ is continuous on $`K`$, and there exists constant
$`C_{K}`$, such that
$`\sup_{\theta\in K}\mathbb{E}_{\theta}[\|f_{\theta}(Z)\|^{3}_{2}]<C_{K}`$,
then we have

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}\big(\frac{1}{n}\sumop\displaylimits_{j=1}^{n}f_{\check{\theta}_{n}}(Z_{1,n}^{(j)})-F(\check{\theta}_{0})\big)\xrightarrow{d}\mathcal{N}\bigg(0,V_{f}(\check{\theta}_{0})+\nabla_{\theta}F(\check{\theta}_{0})A_{1}\big(\nabla_{\theta}F(\check{\theta}_{0})\big)^{\top}\bigg).
``` |  |

###### Proof of Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Denote
$`h_{2,n}=\sqrt{n}\big(\frac{1}{n}\sumop\displaylimits_{j=1}^{n}f_{\check{\theta}_{n}}(Z_{1,n}^{(j)})-F(\check{\theta}_{0})\big),`$
the characteristic function of $`h_{2,n}`$ is

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\phi_{h_{2,n}}(s)`$ | $`\displaystyle=\mathbb{E}\bigg[\mathbb{E}_{\check{\theta}_{n}}\big[\exp(is^{\top}h_{2,n})\big|\check{\theta}_{n}\big]\bigg]`$ |  |
|  |  | $`\displaystyle=\mathbb{E}\bigg[\left\{\mathbb{E}_{\check{\theta}_{n}}\left[\exp\big(\frac{is^{\top}}{\sqrt{n}}(f_{\check{\theta}_{n}}(Z)-F(\check{\theta}_{0}))\big)\right]\right\}^{n}\bigg].`$ |  |

Fix a compact subset $`K\subset\Theta`$. For fixed $`s`$, when
$`\sqrt{n}\to\infty`$, by the assumption, for any $`\theta\in K`$, we
have

|  |  |  |
|----|----|----|
|  |
``` math
\bigg|\mathbb{E}_{\theta}[\exp(\frac{1}{\sqrt{n}}{is}^{\top}f_{{\theta}}(Z))]-1-\frac{1}{\sqrt{n}}is^{\top}F({\theta})+\frac{1}{2n}s^{\top}\mathbb{E}_{\theta}[f_{\theta}(Z)f_{\theta}^{\top}(Z)]s\bigg|\leqslant C\|\frac{s}{\sqrt{n}}\|_{2}^{3}.
``` |  |

Thus by the boundedness of $`F(\theta)`$ on $`K`$, the following
expansion holds uniformly for $`\theta,\theta^{\prime}\in K`$:

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle{\mathbb\displaystyle E}_{\theta}\big[\exp\big(\frac{is^{\top}}{\sqrt{n}}(f_{\theta}(Z)-F(\theta^{\prime}))\big)\big]`$ |  |
|  | $`\displaystyle=`$ | $`\displaystyle 1+\frac{is^{\top}}{\sqrt{n}}\big(F(\theta)-F(\theta^{\prime})\big)-\frac{1}{2n}s^{\top}\big[V_{f}(\theta)+\big(F(\theta)-F(\theta^{\prime})\big)\big(F(\theta)-F(\theta^{\prime})\big)^{\top}\big]s+O(\|\frac{s}{\sqrt{n}}\|_{2}^{3})`$ |  |

For $`s\in\mathbb{R}^{q},\theta\in\Theta`$, define
$`\chi(s,\theta):=\exp(-\frac{1}{2}s^{\top}V_{f}(\theta)s)`$ and

|  |  |  |
|----|----|----|
|  |
``` math
\chi_{n}(s,\theta):=\bigg(\mathbb{E}_{\theta}\big[\exp\big(\frac{is^{\top}}{\sqrt{n}}(f_{\theta}(Z)-F(\theta))\big)\big]\bigg)^{n},
``` |  |

Then for any compact set $`K\subset\Theta`$ and fixed $`s`$,
$`\sup_{\theta\in K}|\chi_{n}(s,\theta)-\chi(s,\theta)|\to 0`$ as
$`n\to\infty`$.

Note that $`\chi,\chi_{n}`$ are bounded, by Lemma
[B.4](#A2.Thmtheorem4 "Lemma B.4. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
take $`\xi_{n}=\check{\theta}_{n}`$ and
$`R_{n}(\theta)=\chi_{n}(s,\theta)-\chi(s,\theta)`$, we have for each
fixed $`s`$,

|  |  |  |
|----|----|----|
|  |
``` math
\chi_{n}(s,\check{\theta}_{n})-\chi(s,\check{\theta}_{n})\xrightarrow{L_{1}}0.
``` |  |

Further, we have

|  |  |  |
|----|----|----|
|  |
``` math
\phi_{h_{2,n}}(s)=\mathbb{E}\bigg[\exp\big(is^{\top}\sqrt{n}\big[F(\check{\theta}_{n})-F(\check{\theta}_{0})\big]\big)\chi_{n}(s,\check{\theta}_{n})\bigg]
``` |  |

Applying Delta method to the first term, we have

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}s^{\top}\big(F(\check{\theta}_{n})-F(\check{\theta}_{0})\big)\xrightarrow{d}\mathcal{N}\bigg(0,s^{\top}\nabla_{\theta}F(\check{\theta}_{0})A_{1}\big(\nabla_{\theta}F(\check{\theta}_{0})\big)^{\top}s\bigg).
``` |  |

For the second term, we note that
$`\chi(s,\check{\theta}_{n})\xrightarrow{P}\chi(s,\check{\theta}_{0})`$
by continuous mapping theorem. Then by Portmanteau Lemma, we have the
following convergence:

|  |  |  |
|----|----|----|
|  | $`\displaystyle\phi_{h_{2,n}}(s)\to\exp\Bigg(-\frac{1}{2}s^{\top}\bigg(V_{f}(\check{\theta}_{0})+\nabla_{\theta}F(\check{\theta}_{0})A_{1}\big(\nabla_{\theta}F(\check{\theta}_{0})\big)^{\top}\bigg)s\Bigg).`$ |  |

Note that the right hand side is the characteristic function of a
multi-variate normal distribution, by Levy’s continuity theorem, the
above pointwise convergence of characteristic function implies:

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}\big(\frac{1}{n}\sumop\displaylimits_{j=1}^{n}f_{\check{\theta}_{n}}(Z_{1,n}^{(j)})-F(\check{\theta}_{0})\big)\xrightarrow{d}\mathcal{N}\bigg(0,V_{f}(\check{\theta}_{0})+\nabla_{\theta}F(\check{\theta}_{0})A_{1}\big(\nabla_{\theta}F(\check{\theta}_{0})\big)^{\top}\bigg).
``` |  |

∎

###### Lemma B.6.

Under Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each compact subset $`K\subset\Theta`$, denote $`K_{1}`$ its compact
convex hull, define $`L_{s,K}^{\prime}:=L_{s,K_{1}}e^{DL_{s,K_{1}}}`$
where $`D=\text{diam}(K)`$ and $`L_{s,K_{1}}`$ is the upper bound of the
norm of score function $`\nabla_{\theta}\log p(z;\theta)`$ in Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
then for any $`\check{\theta},\theta\in K`$ and $`z\in\mathcal{Z}`$, we
have

|  |  |  |
|----|----|----|
|  |
``` math
\|\dfrac{p(z;\theta)}{p(z;\check{\theta})}-1\|\leqslant L_{s,K}^{\prime}\|\theta-\check{\theta}\|_{2}.
``` |  |

###### Proof.

For any $`\check{\theta},\theta\in K`$ and $`z\in\mathcal{Z}`$, by
mean-value theorem we have

|  |  |  |
|----|----|----|
|  |
``` math
|\log p(z;\check{\theta})-\log p(z;\theta)|\leqslant L_{s,K_{1}}\|\check{\theta}-\theta\|_{2},
``` |  |

this leads to

|  |  |  |
|----|----|----|
|  |
``` math
e^{-L_{s,K_{1}}\|\theta-\check{\theta}\|_{2}}\leqslant p(z;\theta)/p(z;\check{\theta})\leqslant e^{L_{s,K_{1}}\|\theta-\check{\theta}\|_{2}},
``` |  |

then, we have

|  |  |  |
|----|----|----|
|  |
``` math
\|\dfrac{p(z;\theta)}{p(z;\check{\theta})}-1\|\leqslant e^{L_{s,K_{1}}\|\theta-\check{\theta}\|_{2}}-1.
``` |  |

Since $`K`$ is bounded, the mean-value theorem gives

|  |  |  |
|----|----|----|
|  |
``` math
e^{L_{s,K_{1}}\|\theta-\check{\theta}\|_{2}}-1\leqslant e^{DL_{s,K_{1}}}L_{s,K_{1}}\|\theta-\check{\theta}\|_{2}.
``` |  |

Therefore,
$`\|\dfrac{p(z;\theta)}{p(z;\check{\theta})}-1\|\leqslant L_{s,K}^{\prime}\|\theta-\check{\theta}\|_{2}.`$
∎

#### B.1.4 Proof of CLT for RRM (Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"))

We first prove the following local expansion lemma with dynamic
underlying distribution.

###### Lemma B.7.

For a function $`f:\mathcal{Z}\to\mathbb R`$ and
$`\check{\theta}\in\Theta`$, let

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb G_{n,\check{\theta}}f:=\sqrt{n}\left\{\frac{1}{n}\sumop\displaylimits_{j=1}^{n}f(Z_{j})-\mathbb E_{\check{\theta}}[f(Z)]\right\},
``` |  |

where $`Z_{1},\ldots,Z_{n}`$ are i.i.d. from
$`\mathcal{D}(\check{\theta})`$. Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
given two random vector sequences
$`\{\check{\theta}_{n}\}_{n=1}^{\infty},\{\theta_{n}\}_{n=1}^{\infty}`$
with $`\check{\theta}_{n}\xrightarrow{P}\check{\theta}_{0}`$ and
$`\theta_{n}\xrightarrow{P}\theta_{0}`$, if $`h_{n}=O_{p}^{*}(1)`$, then
we have

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb G_{n,\check{\theta}_{n}}\left[\sqrt{n}\left(\ell_{\theta_{n}+h_{n}/\sqrt{n}}-\ell_{\theta_{n}}\right)-h_{n}^{\top}\nabla\ell_{\theta_{n}}\right]=o_{P}^{*}(1)
``` |  |

with
$`\mathbb{G}_{n,\check{\theta}_{n}}f=\sqrt{n}\left\{\frac{1}{n}\sumop\displaylimits_{j=1}^{n}f(Z_{j,n})-\mathbb E_{\check{\theta}_{n}}[f(Z)]\right\}`$,
where, conditional on $`(\check{\theta}_{n},\theta_{n})`$, the
observations $`Z_{1,n},\ldots,Z_{n,n}`$ are i.i.d. from
$`\mathcal{D}(\check{\theta}_{n})`$.

###### Proof of Lemma [B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

We follow the proof of Lemma 19.31 in [Van der Vaart
\[2000\]](#bib.bib15). For $`M>0`$, we define
$`B_{M}:=\big\{h\in\mathbb{R}^{d}:\|h\|_{2}\leqslant M\big\}`$.

Step 1. Uniform bound on $`B_{M}`$. Fix a nonrandom $`h\in B_{M}`$. For
fixed $`\theta,\check{\theta}\in\Theta`$ and a constant
$`h\in\mathbb{R}^{d}`$, denote

|  |  |  |
|----|----|----|
|  |
``` math
r_{n,\theta,h}(z)=\sqrt{n}\big(\ell_{\theta+h/\sqrt{n}}(z)-\ell_{\theta}(z)\big)-h^{\top}\nabla_{\theta}\ell_{\theta}(z),
``` |  |

then we have
$`\mathbb{E}_{\check{\theta}}[\mathbb{G}_{n,\check{\theta}}r_{n,\theta,h}]=0`$.
Note that by mean-value theorem, for each $`z`$, there exists a constant
$`c_{z}\in[0,1]`$ and
$`r_{n,\theta,h}(z)=h^{\top}\big(\nabla_{\theta}\ell_{\theta+\frac{c_{z}h}{\sqrt{n}}}(z)-\nabla_{\theta}\ell_{\theta}(z)\big)`$,
under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the joint smoothness of $`\ell`$ implies that
$`|r_{n,\theta,h}(z)|\leqslant\frac{\beta}{\sqrt{n}}\|h\|_{2}^{2}`$,
thus

|  |  |  |
|----|----|----|
|  |
``` math
\text{Var}_{\check{\theta}}(\mathbb{G}_{n,\check{\theta}}r_{n,\theta,h})=\text{Var}_{\check{\theta}}(r_{n,\theta,h}(Z))\leqslant\mathbb{E}_{\check{\theta}}\|r_{n,\theta,h}(Z)\|^{2}\leqslant\frac{\beta^{2}}{n}\|h\|_{2}^{4}.
``` |  |

Step 2. Equicontinuity on $`B_{M}`$. For fixed
$`\theta,\check{\theta}\in\Theta`$, $`z\in\mathcal{Z}`$ and
$`h,h^{\prime}\in B_{M}`$,

|  |  |  |
|----|----|----|
|  |
``` math
r_{n,\theta,h}(z)=\intop\nolimits_{0}^{1}h^{\top}\left(\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta}(z)\right)ds
``` |  |

by the joint smoothness assumption (Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c)),
we have

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle|r_{n,\theta,h}(z)-r_{n,\theta,h^{\prime}}(z)|`$ |  |
|  |  | $`\displaystyle=\bigg|\intop\nolimits_{0}^{1}(h-h^{\prime})^{\top}\big[\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta}(z)\big]ds+\intop\nolimits_{0}^{1}(h^{\prime})^{\top}\big[\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta+sh^{\prime}/\sqrt{n}}(z)\big]ds\bigg|`$ |  |
|  |  | $`\displaystyle\leqslant\bigg|\intop\nolimits_{0}^{1}(h-h^{\prime})^{\top}\big[\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta}(z)\big]ds\bigg|+\bigg|\intop\nolimits_{0}^{1}(h^{\prime})^{\top}\big[\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta+sh^{\prime}/\sqrt{n}}(z)\big]ds\bigg|`$ |  |
|  |  | $`\displaystyle\leqslant\|h-h^{\prime}\|_{2}\cdot\intop\nolimits_{0}^{1}\|\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta}(z)\|_{2}ds+\|h^{\prime}\|_{2}\intop\nolimits_{0}^{1}\|\nabla_{\theta}\ell_{\theta+sh/\sqrt{n}}(z)-\nabla_{\theta}\ell_{\theta+sh^{\prime}/\sqrt{n}}(z)\|_{2}ds`$ |  |
|  |  | $`\displaystyle\leqslant\|h-h^{\prime}\|_{2}\frac{\|h\|_{2}}{\sqrt{n}}\beta+\|h^{\prime}\|_{2}\frac{\beta\|h-h^{\prime}\|_{2}}{\sqrt{n}}`$ |  |
|  |  | $`\displaystyle\leqslant\frac{2M\beta}{\sqrt{n}}\|h-h^{\prime}\|_{2}.`$ |  |

Thus, the following inequality holds for any $`\check{\theta}\in\Theta`$

|  |  |  |
|----|----|----|
|  |
``` math
\big|\mathbb{G}_{n,\check{\theta}}\big(r_{n,\theta,h}-r_{n,\theta,h^{\prime}}\big)\big|\leqslant\sqrt{n}\widehat{\mathbb{E}}_{\check{\theta},n}|r_{n,\theta,h}-r_{n,\theta,h^{\prime}}|+\sqrt{n}\mathbb{E}_{\check{\theta}}|r_{n,\theta,h}-r_{n,\theta,h^{\prime}}|\leqslant 4M\beta\|h-h^{\prime}\|_{2}
``` |  |

Fix $`\varepsilon>0`$, take $`\delta=\frac{\varepsilon}{8M\beta}>0`$,
and a finite $`\delta`$-net $`\{h_{1},\ldots,h_{N}\}`$ of $`B_{M}`$.
Then for each $`h\in B_{M}`$, there exists $`j\in[1:N]`$, such that
$`\|h-h_{j}\|_{2}\leqslant\delta`$, then for each
$`\check{\theta}\in\Theta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\big|\mathbb{G}_{n,\check{\theta}}(r_{n,\theta,h}-r_{n,\theta,h_{j}})\big|\leqslant 4M\beta\|h-h_{j}\|_{2}\leqslant 4M\beta\delta\leqslant\frac{\varepsilon}{2}.
``` |  |

This shows

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta,\check{\theta}\in\Theta}P^{*}\big(\sup_{h\in B_{M}}|\mathbb{G}_{n,\check{\theta}}r_{n,\theta,h}|>\varepsilon\big)\leqslant\sup_{\theta,\check{\theta}\in\Theta}P\big(\max_{1\leqslant j\leqslant N}|\mathbb{G}_{n,\check{\theta}}r_{n,\theta,h_{j}}|>\frac{\varepsilon}{2}\big),
``` |  |

Using the variance bound in Step 1, we have

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta,\check{\theta}\in\Theta}P\big(\max_{1\leqslant j\leqslant N}|\mathbb{G}_{n,\check{\theta}}r_{n,\theta,h_{j}}|>\frac{\varepsilon}{2}\big)\leqslant\dfrac{4N\beta^{2}M^{4}}{n\varepsilon^{2}}.
``` |  |

Finally, given random sequence $`\check{\theta}_{n}`$ and
$`\theta_{n}`$, and general $`h_{n}=O_{p}^{*}(1)`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
P\left(\sup_{\|h\|\leqslant M}|\mathbb G_{n,\check{\theta}_{n}}r_{n,\theta_{n},h}|>\varepsilon\,\middle|\,\check{\theta}_{n},\theta_{n}\right)\leqslant\dfrac{4N\beta^{2}M^{4}}{n\varepsilon^{2}},
``` |  |

taking expectation and by boundedness of $`h_{n}`$, for any fixed
$`a>0`$, we have $`\limsup_{n}P^{*}(\|h_{n}\|_{2}>M)\leqslant a`$, thus

|  |  |  |
|----|----|----|
|  |
``` math
\limsup_{n}P^{*}(|\mathbb{G}_{n,\check{\theta}_{n}}r_{n,\theta_{n},h_{n}}|>\varepsilon)\leqslant\limsup_{n}P^{*}(\|h_{n}\|_{2}>M)+\limsup_{n}P^{*}(\sup_{h\in B_{M}}|\mathbb{G}_{n,\check{\theta}_{n}}r_{n,\theta_{n},h}|>\varepsilon)\leqslant a,
``` |  |

As $`a\downarrow 0`$, this proves the desired convergence for the fixed
threshold $`\varepsilon`$. ∎

###### Lemma B.8.

Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for fixed $`\check{\theta}_{0}\in\Theta`$ and a sequence of random
variables assume
$`\sqrt{n}(\check{\theta}_{n}-\check{\theta}_{0})=O_{p}(1)`$ and
$`h_{n}=O_{p}^{\ast}(1)`$ is a random sequence, we have

|  |  |  |
|----|----|----|
|  |
``` math
{n}\big(\mathbb{E}_{\check{\theta}_{n}}-\mathbb{E}_{{\check{\theta}_{0}}}\big)\bigg[\ell_{G(\check{\theta}_{0})+\frac{h_{n}}{\sqrt{n}}}(Z)-\ell_{G(\check{\theta}_{0})}(Z)-\frac{h_{n}^{\top}}{\sqrt{n}}\nabla\ell_{G(\check{\theta}_{0})}(Z)\bigg]=o_{p}^{*}(1)
``` |  |

###### Proof.

By Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we have for any compact set $`K\subset\Theta`$ containing
$`\check{\theta}_{0}`$, when $`\check{\theta}\in K`$, for any fixed
$`h\in\mathbb{R}^{d}`$,

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle{\displaystyle n}\bigg|\big(\mathbb{E}_{\check{\theta}}-\mathbb{E}_{{\check{\theta}_{0}}}\big)\big[\ell_{G(\check{\theta}_{0})+\frac{h}{\sqrt{n}}}(Z)-\ell_{G(\check{\theta}_{0})}(Z)-\frac{h^{\top}}{\sqrt{n}}\nabla\ell_{G(\check{\theta}_{0})}(Z)\big]\bigg|`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle n\mathbb{E}_{\check{\theta}_{0}}\bigg[\bigg|\big(\frac{p(Z;\check{\theta})}{p(Z;\check{\theta}_{0})}-1\big)\big(\ell_{G(\check{\theta}_{0})+\frac{h}{\sqrt{n}}}(Z)-\ell_{G(\check{\theta}_{0})}(Z)-\frac{h^{\top}}{\sqrt{n}}\nabla_{\theta}\ell_{G(\check{\theta}_{0})}(Z)\big)\bigg|\bigg]`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\sqrt{n}\mathbb{E}_{\check{\theta}_{0}}\bigg[L_{s,K}^{\prime}\big\|\check{\theta}-\check{\theta}_{0}\big\|_{2}\times\bigg|\sqrt{n}\big(\ell_{G(\check{\theta}_{0})+\frac{h}{\sqrt{n}}}(Z)-\ell_{G(\check{\theta}_{0})}(Z)\big)-{h^{\top}}\nabla_{\theta}\ell_{G(\check{\theta}_{0})}(Z)\bigg|\bigg]`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\sqrt{n}\mathbb{E}_{\check{\theta}_{0}}\bigg[L_{s,K}^{\prime}\big\|\check{\theta}-\check{\theta}_{0}\big\|_{2}\cdot\frac{\beta}{\sqrt{n}}\|{h}\|_{2}^{2}\bigg]`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\beta L_{s,K}^{\prime}\|h\|_{2}^{2}\cdot\|\check{\theta}-\check{\theta}_{0}\|_{2},`$ |  |

Define
$`R_{n}(\check{\theta},h)={n}\big(\mathbb{E}_{\check{\theta}}-\mathbb{E}_{{\check{\theta}_{0}}}\big)\big[\ell_{G(\check{\theta}_{0})+\frac{h}{\sqrt{n}}}-\ell_{G(\check{\theta}_{0})}-\frac{h^{\top}}{\sqrt{n}}\nabla\ell_{G(\check{\theta}_{0})}\big]`$.
Fix $`\varepsilon>0`$. By the fact that
$`\sqrt{n}(\check{\theta}_{n}-\check{\theta}_{0})=O_{p}(1)`$ and that
$`h_{n}=O_{p}^{*}(1)`$, there exists $`M>0`$ such that,
$`\|\check{\theta}_{0}\|_{2}\leqslant M`$,
$`\sup_{n}P\big(\|\sqrt{n}(\check{\theta}_{n}-\check{\theta}_{0})\|_{2}>M\big)\leqslant\varepsilon,`$
and $`\sup_{n}P^{*}(\|h_{n}\|_{2}>M)\leqslant\varepsilon`$, then

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle P^{*}(|R_{n}(\check{\theta}_{n},h_{n})|>2\varepsilon)`$ | $`\displaystyle\leqslant P^{*}\big(\sup_{\|h\|_{2}\leqslant M}\sup_{\sqrt{n}\|\check{\theta}-\check{\theta}_{0}\|_{2}\leqslant M}|R_{n}(\check{\theta},h)|>\varepsilon\big)`$ |  |
|  |  | $`\displaystyle+P\big(\|\sqrt{n}(\check{\theta}_{n}-\check{\theta}_{0})\|_{2}>M\big)+P^{*}(\|h_{n}\|_{2}>M).`$ |  |
|  |  | $`\displaystyle\leqslant P^{*}\big(\sup_{\|h\|_{2}\leqslant M}\sup_{\sqrt{n}\|\check{\theta}-\check{\theta}_{0}\|_{2}\leqslant M}|R_{n}(\check{\theta},h)|>\varepsilon\big)+2\varepsilon,`$ |  |

Set $`K=B(0,2M)\subset\mathbb{R}^{d}`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
P^{*}\big(\sup_{\|h\|_{2}\leqslant M}\sup_{\sqrt{n}\|\check{\theta}-\check{\theta}_{0}\|_{2}\leqslant M}|R_{n}(\check{\theta},h)|>\varepsilon\big)\leqslant\frac{2}{\sqrt{n}}\beta M^{3}L_{s,K}^{\prime},
``` |  |

thus when $`n\to\infty`$, the first term tends to $`0`$. Therefore, we
have proved that

|  |  |  |
|----|----|----|
|  |
``` math
n\big(\mathbb{E}_{\check{\theta}_{n}}-\mathbb{E}_{{\check{\theta}_{0}}}\big)\bigg[\ell_{G(\check{\theta}_{0})+\frac{h_{n}}{\sqrt{n}}}(Z)-\ell_{G(\check{\theta}_{0})}(Z)-\frac{h_{n}^{\top}}{\sqrt{n}}\nabla\ell_{G(\check{\theta}_{0})}(Z)\bigg]=o_{p}^{*}(1),
``` |  |

∎

We now make the decomposition preceding Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
precise.

###### Theorem B.9 (Restatement of Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).

Under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for each fixed $`t\geqslant 1`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\mathcal{N}(0,V_{t}),
``` |  |

where $`V_{0}=0`$ and
$`V_{t+1}=\nabla G(\theta_{t})V_{t}\nabla G(\theta_{t})^{\top}+\Sigma_{\theta_{t}}(\theta_{t+1}),\qquad t\geqslant 0.`$

###### Proof of Theorem [3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

We prove
$`\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\to\mathcal{N}(0,V_{t})`$ by
induction on $`t`$, where $`V_{t}`$ is defined as in Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
By Theorem 5.23 in \[[Van der Vaart, 2000](#bib.bib15)\],
$`\sqrt{n}(\widehat{\theta}_{1}-\theta_{1})\xrightarrow{d}\mathcal{N}(0,V_{1})`$,
where $`V_{1}=\Sigma_{\theta_{0}}(\theta_{1})`$. Assume that
$`\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\mathcal{N}(0,V_{t})`$.
We prove the result for $`t+1`$. Set
$`\widetilde{\theta}_{t+1,n}=G(\widehat{\theta}_{t})`$. We first
localize $`\widehat{\theta}_{t+1}`$ around
$`\widetilde{\theta}_{t+1,n}`$.

Step 1. Asymptotic localization. We show that
$`\sqrt{n}(\widehat{\theta}_{t+1}-\widetilde{{\theta}}_{t+1,n})=O_{p}(1)`$.

We follow the proof of Theorem 5.52 in \[[Van der Vaart,
2000](#bib.bib15)\]. The induction hypothesis and the delta method imply
that $`\sqrt{n}(\widetilde{\theta}_{t+1,n}-\theta_{t+1})`$ is
asymptotically normal and hence bounded in probability. It remains to
show
$`\sqrt{n}\big(\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n}\big)=O_{p}(1)`$.

Since $`\widetilde{\theta}_{t+1,n}=G(\widehat{\theta}_{t})`$ minimizes
$`\theta\mapsto\mathbb E_{\widehat{\theta}_{t}}[\ell_{\theta}(Z)]`$ and
$`\widehat{\theta}_{t+1}`$ minimizes
$`\theta\mapsto\widehat{\mathbb E}_{\widehat{\theta}_{t},n}[\ell_{\theta}(Z)]`$,

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb{E}_{\widehat{\theta}_{t}}[\nabla_{\theta}\ell_{\widetilde{\theta}_{t+1,n}}(Z)]=0,
``` |  |

and

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\widehat{\theta}_{t+1}}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant 0
``` |  |

For $`n\geqslant 1`$, we define a partition of
$`\Theta\backslash\{\widetilde{\theta}_{t+1,n}\}`$ by setting
$`S_{j}^{(n)}=\{\theta:2^{j-1}<\sqrt{n}\|\theta-\widetilde{\theta}_{t+1,n}\|\leqslant 2^{j}\}`$
with $`j`$ ranging over integers. For any $`\rho>0`$, choose a compact
subset $`K\subset\Theta`$ and set
$`A_{n}=\{\widehat{\theta}_{t}\in K,\widetilde{\theta}_{t+1,n}\in K\}`$
so that $`P(A_{n}^{c})<\rho`$ for all sufficiently large $`n`$. On
$`A_{n}`$, both $`\widehat{\theta}_{t}`$ and
$`\widetilde{\theta}_{t+1,n}`$ lie in $`K`$.

For fixed $`M\in\mathbb{R}`$, if
$`\sqrt{n}\|\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n}\|>2^{M}`$,
then $`\widehat{\theta}_{t+1}\in S_{j}^{(n)}`$ for some $`j>M`$, and

|  |  |  |
|----|----|----|
|  |
``` math
\inf_{\theta\in S_{j}^{(n)}}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\widehat{\theta}_{t+1}}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant 0.
``` |  |

This implies that for any $`\eta>0`$, the following inequality holds,

|  |  |  |
|----|----|----|
|  |
``` math
P(\sqrt{n}\|\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n}\|>2^{M})\leqslant\sumop\displaylimits_{\begin{subarray}{c}j>M\\
2^{j}\leqslant\sqrt{n}\eta\end{subarray}}P^{*}\big(A_{n},\inf_{\theta\in S_{j}^{(n)}}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant 0\big)+P(2\|\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n}\|>\eta)+P(A_{n}^{c}).
``` |  |

By Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the second term on the right hand side tends to $`0`$ when
$`n\to\infty`$ for each $`\eta>0`$. By the strong convexity of $`\ell`$
in $`\Theta`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
{\mathbb{E}}_{\widehat{\theta}_{t}}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\geqslant\frac{\gamma}{2}\|\theta-\widetilde{\theta}_{t+1,n}\|^{2},
``` |  |

Conditioning on $`A_{n}`$ and $`\widehat{\theta}_{t}`$,

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\sumop\displaylimits_{\begin{subarray}{c}j>M\\
2^{j}\leqslant\sqrt{n}\eta\end{subarray}}P^{*}\bigg(A_{n},\inf_{\theta\in S_{j}^{(n)}}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant 0\bigg|\widehat{\theta}_{t}\bigg)`$ |  |
|  | $`\displaystyle\leqslant`$ | $`\displaystyle\sumop\displaylimits_{\begin{subarray}{c}j>M\\
2^{j}\leqslant\sqrt{n}\eta\end{subarray}}P^{*}\bigg(A_{n},\inf_{\theta\in S_{j}^{(n)}}\mathbb{G}_{n,\widehat{\theta}_{t}}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant-\frac{\gamma}{2}\sqrt{n}\inf_{\theta\in S_{j}^{(n)}}\|\theta-\widetilde{\theta}_{t+1,n}\|^{2}\bigg|\widehat{\theta}_{t}\bigg)`$ |  |
|  | $`\displaystyle\lesssim`$ | $`\displaystyle\sumop\displaylimits_{\begin{subarray}{c}j>M\\
2^{j}\leqslant\sqrt{n}\eta\end{subarray}}{2^{-2j}}{\sqrt{n}}\mathbb{E}^{*}\bigg[\mathbf{1}_{A_{n}}\cdot\big|\inf_{\theta\in S_{j}^{(n)}}\mathbb{G}_{n,\widehat{\theta}_{t}}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\big|\bigg|\widehat{\theta}_{t}\bigg]`$ |  |
|  | $`\displaystyle\lesssim`$ | $`\displaystyle\sumop\displaylimits_{\begin{subarray}{c}j>M\\
2^{j}\leqslant\sqrt{n}\eta\end{subarray}}{2^{-2j}}{\sqrt{n}}\mathbb{E}^{*}\bigg[\mathbf{1}_{A_{n}}\sup_{\theta\in S_{j}^{(n)}}\big|\mathbb{G}_{n,\widehat{\theta}_{t}}(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}})\big|\bigg|\widehat{\theta}_{t}\bigg]`$ |  |

Note that when $`j>M`$ and $`2^{j}\leqslant\sqrt{n}\eta`$, the set
$`S_{j}^{(n)}\subset K^{2\eta}`$, where
$`K^{2\eta}:=\{\theta:\text{dist}(\theta,K)\leqslant 2\eta\}`$ is the
compact enlargement of the fixed compact set $`K`$. Conditional on
$`A_{n}`$ and $`\widehat{\theta}_{t}`$,
$`\widetilde{\theta}_{t+1,n}=G(\widehat{\theta}_{t})`$ is fixed and lies
in $`K`$.

For $`\delta\in(0,2\eta]`$ and $`\check{\theta}\in K`$, set
$`\mathcal{F}_{\check{\theta},\delta}=\{\ell(\theta)-\ell(G(\check{\theta})):\|\theta-G(\check{\theta})\|_{2}\leqslant\delta,\theta\in K^{2\eta}\}`$.
On $`A_{n}`$, take $`\check{\theta}=\widehat{\theta}_{t}`$. Then
$`G(\check{\theta})\in K`$, and $`\mathcal{F}_{\check{\theta},\delta}`$
has envelope $`\delta L_{K^{2\eta}}`$, where $`L_{K^{2\eta}}`$ is given
by Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
Since
$`\sup_{\check{\theta}\in K^{2\eta}}\mathbb{E}_{\check{\theta}}\|L_{K^{2\eta}}(Z)\|^{2}<M_{1}^{2}`$,
Corollary 19.35 in \[[Van der Vaart, 2000](#bib.bib15)\] gives

|  |  |  |
|----|----|----|
|  | $`\displaystyle\mathbb{E}^{*}\bigg[\sup_{\|\theta-G(\widehat{\theta}_{t})\|\leqslant\delta}\big|\mathbb{G}_{n,\widehat{\theta}_{t}}(\ell_{\theta}-\ell_{G(\widehat{\theta}_{t})})\big|\bigg|\widehat{\theta}_{t}\bigg]\lesssim\mathcal{J}_{[]}\big(\delta\|L_{K^{2\eta}}(z)\|_{L_{2}(\widehat{\theta}_{t})},\mathcal{F}_{\widehat{\theta}_{t},\delta},L_{2}(\widehat{\theta}_{t})\big),`$ |  |

where for any $`\check{\theta}\in\Theta`$, $`L_{2}(\check{\theta})`$
stands for $`L_{2}`$ distance with respect to the distribution
$`\mathcal{D}(\check{\theta})`$. By Example 19.7 in \[[Van der Vaart,
2000](#bib.bib15)\], we know that there exists a constant $`C`$ that
depends only on $`K^{2\eta}`$, $`d`$, and the following inequality holds
uniformly for $`\check{\theta}\in K`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\check{\theta}\in K}N_{[]}(\varepsilon,\mathcal{F}_{\check{\theta},\delta},L_{2}({\check{\theta}}))\leqslant(\frac{C\delta M_{1}}{\varepsilon})^{d}.
``` |  |

Define the uniform metric entropy over $`K`$ as below for
$`a,\delta>0`$:

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{J}_{[]}\big(a,\mathcal{F}_{\check{\theta},\delta},L_{2}\big)=\intop\nolimits_{0}^{a}\sqrt{\log\sup_{\check{\theta}\in K}N_{[]}\big(\varepsilon,\mathcal{F}_{\check{\theta},\delta},L_{2}(\check{\theta})\big)}d\varepsilon,
``` |  |

then the following inequality holds for some fixed constant $`C_{1}`$
depend on $`K^{2\eta},d`$,

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{J}_{[]}\big(\delta M_{1},\mathcal{F}_{\check{\theta},\delta},L_{2}\big)\leqslant\intop\nolimits_{0}^{\delta M_{1}}\sqrt{d\log(\frac{\delta M_{1}}{\varepsilon})+C_{1}}\mathrm{d}\varepsilon=\delta\intop\nolimits_{0}^{M_{1}}\sqrt{d\log(\frac{M_{1}}{\varepsilon})+C_{1}}\mathrm{d}\varepsilon.
``` |  |

Taking $`\delta=\frac{2^{j+1}}{\sqrt{n}}`$ gives, for any $`M>0`$,

|  |  |  |
|----|----|----|
|  |
``` math
P^{*}\big(A_{n},\inf_{\theta\in S_{j}^{(n)}}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant 0\big)\lesssim 2^{-2j}\sqrt{n}\,\mathbb{E}\bigg[\mathbb{E}\big[\mathbf{1}_{A_{n}}\sup_{\theta\in S_{j}^{(n)}}|\mathbb{G}_{n,\widehat{\theta}_{t}}(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}})|\big|\widehat{\theta}_{t}\big]\bigg]
``` |  |

and thus

|  |  |  |
|----|----|----|
|  |
``` math
\sumop\displaylimits_{\begin{subarray}{c}j>M\\
2^{j}\leqslant\sqrt{n}\eta\end{subarray}}P^{*}\big(A_{n},\inf_{\theta\in S_{j}^{(n)}}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big(\ell_{\theta}-\ell_{\widetilde{\theta}_{t+1,n}}\big)\leqslant 0\big)\lesssim\sumop\displaylimits_{j>M}2^{-j}\leqslant{2^{1-M}}.
``` |  |

Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives
$`P(2\|\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n}\|>\eta)\to 0`$.
Hence

|  |  |  |
|----|----|----|
|  |
``` math
\limsup_{n}P(\sqrt{n}\|\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n}\|>2^{M})\lesssim\rho+2^{1-M}.
``` |  |

Letting first $`M\to\infty`$ and then $`\rho\to 0`$ proves that
$`\sqrt{n}(\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n})`$ is
bounded in probability.

Step 2. Asymptotic normality. for a fixed function
$`f:{\cal Z}\to{\mathbb R}^{d}`$, define
$`g_{t,n}[f]:=\sqrt{n}\left\{\widehat{E}_{\widehat{\theta}_{t},n}[f(Z)]-E_{\theta_{t}}[f(Z)]\right\}.`$
Take
$`\widehat{h}_{t+1,n}=\sqrt{n}(\widehat{\theta}_{t+1}-\theta_{t+1})`$
and
$`\tilde{h}_{t+1,n}=-H_{\theta_{t}}(\theta_{t+1})^{-1}g_{t,n}\nabla\ell(\theta_{t+1})`$.
By Step 1, we have
$`\sqrt{n}(\widehat{\theta}_{t+1}-\widetilde{\theta}_{t+1,n})=O_{p}(1)`$,
by induction hypothesis and applying Delta method to function $`G`$, we
have $`\sqrt{n}(\widetilde{\theta}_{t+1,n}-\theta_{t+1})=O_{p}(1)`$.
Combining them gives that $`\widehat{h}_{t+1,n}`$ is $`O_{p}(1)`$.

For $`\check{\theta}\in\Theta`$, set
$`f_{\check{\theta}}=H_{\theta_{t}}(\theta_{t+1})^{-1}\nabla_{\theta}\ell_{\theta_{t+1}}(Z)`$
and
$`F(\check{\theta})=\mathbb{E}_{\check{\theta}}f_{\check{\theta}}(Z)`$.
Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
implies that $`F`$ is continuously differentiable, while Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives the required third-moment bound. Note that by Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a)
and Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we have for each fixed compact set $`K_{1}`$,
$`\left\|\mathbb{E}_{\vartheta^{\prime}}[ff^{\top}]-\mathbb{E}_{\vartheta}[ff^{\top}]\right\|_{2}\leqslant L^{\prime}_{s,K_{1}}\|\vartheta^{\prime}-\vartheta\|_{2}\mathbb{E}_{\vartheta}\|f\|_{2}^{2}.`$
Together with continuously differentiability of $`F`$, we have
$`\text{Cov}_{\theta}f_{\theta}`$ is continuous. Lemma
[B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
then implies that $`\tilde{h}_{t+1,n}`$ is asymptotically normal and
hence $`O_{p}^{*}(1)`$.

Now, by Lemma
[B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Lemma
[B.8](#A2.Thmtheorem8 "Lemma B.8. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
taking the random sequences in Lemma
[B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
to be $`\check{\theta}_{n}=\widehat{\theta}_{t}`$ and
$`\theta_{n}\equiv\theta_{t+1}`$ (we remark that
$`\widehat{\theta}_{t}`$ is a random sequence depending on sample number
per-round $`n`$), we have

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle n\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\ell_{\theta_{t+1}+\widehat{h}_{t+1,n}/\sqrt{n}}-\ell_{\theta_{t+1}}\big]=\frac{1}{2}\widehat{h}_{t+1,n}^{\top}H_{\theta_{t}}(\theta_{t+1})\widehat{h}_{t+1,n}+\widehat{h}_{t+1,n}^{\top}g_{t,n}[\nabla_{\theta}\ell_{\theta_{t+1}}]+o_{p}(1)`$ |  |
|  |  | $`\displaystyle n\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\ell_{\theta_{t+1}+\widetilde{h}_{t+1,n}/\sqrt{n}}-\ell_{\theta_{t+1}}\big]=-\frac{1}{2}\big(g_{t,n}[\nabla\ell_{\theta_{t+1}}]\big)^{\top}H_{\theta_{t}}(\theta_{t+1})^{-1}\big(g_{t,n}[\nabla\ell_{\theta_{t+1}}]\big)+o_{p}(1).`$ |  |

Now by the above formulas and the definition of
$`\widehat{\theta}_{t+1}`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
\big(\widehat{h}_{t+1,n}+H_{\theta_{t}}(\theta_{t+1})^{-1}g_{t,n}[\nabla\ell_{\theta_{t+1}}]\big)^{\top}H_{\theta_{t}}(\theta_{t+1})\big(\widehat{h}_{t+1,n}+H_{\theta_{t}}(\theta_{t+1})^{-1}g_{t,n}[\nabla\ell_{\theta_{t+1}}]\big)+o_{p}(1)\leqslant 0,
``` |  |

since $`H_{\theta_{t}}(\theta_{t+1})`$ is positive definite, we have

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{h}_{t+1,n}+H_{\theta_{t}}(\theta_{t+1})^{-1}g_{t,n}[\nabla_{\theta}\ell_{\theta_{t+1}}]=o_{p}(1).
``` |  |

We have by Lemma
[B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Slutsky Theorem,

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}(\widehat{\theta}_{t+1}-\theta_{t+1})\to\mathcal{N}(0,\Sigma_{\theta_{t}}(\theta_{t+1})+\nabla G(\theta_{t})V_{t}\nabla G(\theta_{t})^{\top}\big),
``` |  |

The covariance on the right is $`V_{t+1}`$ by the recursion in Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
∎

#### B.1.5 Proof of Corollary [3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Corollary [3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Since $`\varepsilon\beta<\gamma`$, the contraction of $`G`$ gives

|  |  |  |
|----|----|----|
|  |
``` math
\|\theta_{t}-\theta_{\mathrm{PS}}\|_{2}\leqslant\left(\frac{\varepsilon\beta}{\gamma}\right)^{t}\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}\leqslant B\left(\frac{\varepsilon\beta}{\gamma}\right)^{t}.
``` |  |

Hence $`\theta_{t}\in\widehat{\mathcal{R}}_{t}`$ implies

|  |  |  |
|----|----|----|
|  |
``` math
\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}\left(0,B(\varepsilon\beta/\gamma)^{t}\right).
``` |  |

Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
therefore gives

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\liminf_{n\to\infty}\mathbb P\left\{\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}\left(0,B(\varepsilon\beta/\gamma)^{t}\right)\right\}\geqslant\lim_{n\to\infty}\mathbb P(\theta_{t}\in\widehat{\mathcal{R}}_{t})=1-\delta.`$ |  |

For the bound after the corollary, the optimality of
$`\theta_{\mathrm{PS}}`$ gives

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\theta_{\mathrm{PS}})}[\nabla\ell(Z;\theta_{\mathrm{PS}})]=0.
``` |  |

Strong convexity and the Cauchy–Schwarz inequality imply

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\gamma\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}^{2}`$ | $`\displaystyle\leqslant\mathbb E_{Z\sim\mathcal{D}(\theta_{\mathrm{PS}})}[\nabla\ell(Z;\theta_{0})]^{\top}(\theta_{0}-\theta_{\mathrm{PS}})`$ |  |
|  |  | $`\displaystyle\leqslant\sup_{z\in\mathcal{Z}}\|\nabla\ell(z;\theta_{0})\|_{2}\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}.`$ |  |

Dividing by $`\gamma\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}`$ proves the
stated bound when $`\theta_{0}\neq\theta_{\mathrm{PS}}`$; the remaining
case is immediate. ∎

#### B.1.6 Regularity for Score Matching

The following conditions justify differentiating under the integral in
Lemma
[3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

###### Assumption B.10.

For each $`j\in[d]`$, $`\theta\in\Theta`$, and $`\psi\in\Psi`$, the
following hold, where integrability is with respect to the dominating
measure in Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a).

\(a\) The functions $`z\mapsto p(z;\theta)s_{j}(z,\theta;\psi)`$ and
$`z\mapsto p(z;\theta)\,\partial_{\theta^{(j)}}s_{j}(z,\theta;\psi)`$
are integrable.

\(b\) For almost every $`z`$, the map
$`\theta\mapsto p(z;\theta)s_{j}(z,\theta;\psi)`$ is differentiable in
$`\theta^{(j)}`$.

\(c\) For $`\theta^{\prime}`$ in a neighborhood of $`\theta`$,
$`\bigl|\partial_{\theta^{(j)}}\{p(z;\theta^{\prime})s_{j}(z,\theta^{\prime};\psi)\}\bigr|`$
is bounded by an integrable function of $`z`$ that does not depend on
$`\theta^{\prime}`$.

#### B.1.7 Proof of Lemma [3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Lemma [3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Integrals are with respect to the dominating measure in Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a).
Fix $`j\in[d]`$. Since
$`p(z;\theta)\,\partial_{\theta^{(j)}}\log p(z;\theta)=\partial_{\theta^{(j)}}p(z;\theta)`$,
the product rule gives

|  |  |  |
|----|----|----|
|  |
``` math
p(z;\theta)\,\partial_{\theta^{(j)}}\log p(z;\theta)\,s_{j}(z,\theta;\psi)=\partial_{\theta^{(j)}}\bigl\{p(z;\theta)s_{j}(z,\theta;\psi)\bigr\}-p(z;\theta)\,\partial_{\theta^{(j)}}s_{j}(z,\theta;\psi).
``` |  |

Integrating over $`z`$, and moving the derivative outside the integral
by Assumption
[B.10](#A2.Thmtheorem10 "Assumption B.10. ‣ B.1.6 Regularity for Score Matching ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we obtain

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\partial_{\theta^{(j)}}\log p(Z;\theta)\,s_{j}(Z,\theta;\psi)\bigr]=\frac{\partial}{\partial\theta^{(j)}}\mathbb E_{Z\sim\mathcal{D}(\theta)}[s_{j}(Z,\theta;\psi)]-\mathbb E_{Z\sim\mathcal{D}(\theta)}[\partial_{\theta^{(j)}}s_{j}(Z,\theta;\psi)].
``` |  |

Summing over $`j\in[d]`$ proves the lemma. ∎

#### B.1.8 Policy Perturbation

This subsection gives the assumptions and proofs for the
policy-perturbation estimator in Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

For a deterministic $`\theta`$, let $`\widehat{J}_{n,k}(\theta;\psi)`$
denote the expression in
([2](#S3.E2 "Equation 2 ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
computed from $`n`$ observations from $`\mathcal{D}(\theta)`$ and $`k`$
observations from each $`\mathcal{D}(\theta+\eta e_{j})`$, $`j\in[d]`$,
plus the constant

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\theta)}[\|\nabla_{\theta}\log p(Z;\theta)\|_{2}^{2}].
``` |  |

The constant does not change the minimizer, so the estimator in Section
[3.2](#S3.SS2 "3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
is also an approximate minimizer of $`\widehat{J}_{n,k}`$. Define the
squared score error

|  |  |  |
|----|----|----|
|  |
``` math
J(\theta;\psi):=\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\|\nabla_{\theta}\log p(Z;\theta)-s(Z,\theta;\psi)\|_{2}^{2}\bigr]
``` |  |

and its finite-difference version

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle J_{\eta}(\theta;\psi):={}`$ | $`\displaystyle\mathbb\displaystyle E_{Z\sim\mathcal{D}(\theta)}\left[\|s(Z,\theta;\psi)\|_{2}^{2}+2\sumop\displaylimits_{j=1}^{d}\partial_{\theta^{(j)}}s_{j}(Z,\theta;\psi)+\|\nabla_{\theta}\log p(Z;\theta)\|_{2}^{2}\right]`$ |  |
|  |  | $`\displaystyle-\frac{2}{\eta}\sumop\displaylimits_{j=1}^{d}\left\{\mathbb E_{Z\sim\mathcal{D}(\theta+\eta e_{j})}[s_{j}(Z,\theta+\eta e_{j};\psi)]-\mathbb E_{Z\sim\mathcal{D}(\theta)}[s_{j}(Z,\theta;\psi)]\right\}.`$ |  |

Lemma
[3.3](#S3.Thmtheorem3 "Lemma 3.3. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives $`J_{\eta}(\theta;\psi)\to J(\theta;\psi)`$ as $`\eta\to 0`$. The
next assumptions make this convergence uniform over the parameters used
below. In the following context, for $`R>0`$, we define
$`{}_{R}:=\{\psi\in\Psi:\|\psi\|_{2}\leqslant R\}`$.

###### Assumption B.11.

We assume that the score function $`s`$, distribution map
$`\mathcal{D}`$ have the following properties:

(a). For any compact subset $`K\subset\Theta`$, $`R>0`$, there is a
positive constant $`C_{K,R}>0`$, such that

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}\sup_{\|\psi\|_{2}<R}\left|\frac{\partial^{2}}{\partial(\theta^{(i)})^{2}}\mathbb{E}_{\theta}[s_{i}(z,\theta;\psi)]\right|\leqslant C_{K,R}<\infty.
``` |  |

(b). (Enveloping function) For any compact subset $`K`$, and $`R>0`$,
there exists a function $`L_{K,R}:\mathcal{Z}\to\mathbb{R}`$ and
$`B_{K,R}:\mathcal{Z}\to\mathbb{R}`$, such that: denote

|  |  |  |
|----|----|----|
|  |
``` math
g_{0}(z,\theta;\psi):=\|s(z,\theta;\psi)\|^{2}_{2},\ g_{1}^{(i)}(z,\theta,\psi)=s_{i}(z,\theta;\psi),\ g_{2}^{(i)}(z,\theta;\psi)=\partial_{\theta_{i}}s_{i}(z,\theta;\psi),\ i\in[d],
``` |  |

then for any function
$`g(z,\theta;\psi)\in\{g_{0}(z,\theta;\psi),g_{1}^{(i)}(z,\theta;\psi),g_{2}^{(i)}(z,\theta;\psi):i\in[d]\}`$
for any $`z\in\mathcal{Z}`$, $`\theta,\theta^{\prime}\in K`$,
$`\psi,\psi^{\prime}\in B(R)`$, the following inequality holds:

|     |                                         |     |
|-----|-----------------------------------------|-----|
|     |
       ``` math
       |g(z,\theta,\psi)|\leqslant B_{K,R}(z),
       ```                                      |     |

|  |  |  |
|----|----|----|
|  |
``` math
|g(z,\theta,\psi)-g(z,\theta^{\prime},\psi^{\prime})|\leqslant L_{K,R}(z)\big(\|\theta-\theta^{\prime}\|_{2}+\|\psi-\psi^{\prime}\|_{2}\big),
``` |  |

with
$`\sup_{\theta\in K^{1}}\mathbb{E}_{\theta}[B_{K,R}{(Z)}^{2}+L_{K,R}{(Z)}^{2}]<\infty,`$
where $`K^{1}`$ is the $`1`$-enlargement of $`K\subset\Theta`$.

(c). (Richness of class.) For any compact set $`K\subset\Theta`$, there
exists $`R_{K}>0`$, such that for each $`\theta\in K`$, there exists a
$`\psi^{\ast}(\theta)\in{}_{R_{K}}`$ satisfying

|     |                                                                  |     |
|-----|------------------------------------------------------------------|-----|
|     |
       ``` math
       s(z,\theta;\psi^{\ast}(\theta))=\nabla_{\theta}\log p(z;\theta).
       ```                                                               |     |

(d). (Approximate minimization and tightness.) For deterministic
$`\theta`$, let $`\widehat{\psi}_{n,k}(\theta)`$ be the empirical
score-matching estimator. For every compact $`K\subset\Theta`$, it can
be selected so that, for some $`a_{n,k}\to 0`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}\mathbb P\left\{\widehat{J}_{n,k}(\theta;\widehat{\psi}_{n,k}(\theta))>\inf_{\psi\in\Psi}\widehat{J}_{n,k}(\theta;\psi)+a_{n,k}\right\}\longrightarrow 0
``` |  |

as $`\min(n,k)\to\infty`$. In addition,

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{R\to\infty}\limsup_{\min(n,k)\to\infty}\sup_{\theta\in K}\mathbb P\{\|\widehat{\psi}_{n,k}(\theta)\|_{2}>R\}=0.
``` |  |

###### Assumption B.12.

For every compact $`K\subset\Theta`$, the following conditions hold.

\(a\)
$`(\widetilde{\theta},\theta)\mapsto H_{\widetilde{\theta}}(\theta)`$ is
Lipschitz on $`K\times K`$.

\(b\)
$`(\widetilde{\theta},\theta)\mapsto\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;\theta)\nabla_{\theta}\log p(Z;\widetilde{\theta})^{\top}]`$
is Lipschitz on $`K\times K`$.

\(c\) There exists a measurable $`L_{K}^{H}:\mathcal{Z}\to[0,\infty)`$
such that, for all $`\theta,\theta^{\prime}\in K`$ and
$`z\in\mathcal{Z}`$,

|  |  |  |
|----|----|----|
|  |
``` math
\|\nabla^{2}\ell(z;\theta)-\nabla^{2}\ell(z;\theta^{\prime})\|_{2}\leqslant L_{K}^{H}(z)\|\theta-\theta^{\prime}\|_{2},
``` |  |

and
$`\sup_{\widetilde{\theta}\in K}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[L_{K}^{H}(Z)]<\infty.`$

###### Assumption B.13.

For each compact subset $`K\subset\Theta`$, constant $`R>0`$, define

|  |  |  |
|----|----|----|
|  |
``` math
h_{ij}(z;\theta,\check{\theta};\psi):=\partial_{\theta^{(i)}}\ell(z;\theta)s_{(j)}(z,\check{\theta};\psi),
``` |  |

where $`i,j\in[d]`$. We assume that there exists function
$`Q_{K,R}:\mathcal{Z}\to\mathbb{R}`$, and
$`L_{K,R}^{\text{prod}}:\mathcal{Z}\to\mathbb{R}`$, such that for any
$`\theta,\check{\theta},\theta^{\prime},\check{\theta}^{\prime}\in K`$,
$`\psi,\psi^{\prime}\in{}_{R}`$,

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle|h_{ij}(z;\theta,\check{\theta};\psi)|\leqslant Q_{K,R}(z),`$ |  |
|  |  | $`\displaystyle|h_{ij}(z;\theta,\check{\theta};\psi)-h_{ij}(z;\theta^{\prime},\check{\theta}^{\prime};\psi^{\prime})|\leqslant L_{K,R}^{\text{prod}}(z)\big(\|\theta-\theta^{\prime}\|_{2}+\|\check{\theta}-\check{\theta}^{\prime}\|_{2}+\|\psi-\psi^{\prime}\|_{2}\big),`$ |  |

with
$`\sup_{\check{\theta}\in K}\mathbb{E}_{\check{\theta}}[Q_{K,R}(Z)^{2}+L_{K,R}^{\text{prod}}(Z)^{2}]<\infty`$.

#### B.1.9 Proof of Theorem [B.15](#A2.Thmtheorem15 "Theorem B.15. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Lemma B.14.

Under Assumption
[B.10](#A2.Thmtheorem10 "Assumption B.10. ‣ B.1.6 Regularity for Score Matching ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a),
we have for any compact set $`K`$ and $`R>0`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}\sup_{\psi\in{}_{R}}|J(\theta;\psi)-{J}_{\eta}(\theta;\psi)|\leqslant 2d\eta C_{K^{2\eta},R}.
``` |  |

where $`C_{K^{2\eta},R}`$ is the constant in Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and $`K^{2\eta}`$ is the compact enlargement
$`K^{2\eta}:=\{\theta:\text{dist}(\theta,K)\leqslant 2\eta\}`$ of $`K`$.

###### Proof.

By mean-value theorem, there exists $`\eta^{(i)}\in[0,\eta]`$,
$`1\leqslant i\leqslant d`$ , such that

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\big|J(\theta;\psi)-{J}_{\eta}(\theta;\psi)\big|`$ | $`\displaystyle=2\bigg|\sumop\displaylimits_{i=1}^{d}\frac{1}{\eta}\bigg({\mathbb{E}}_{\theta+\eta e_{i}}\big[s_{i}(z,\theta+\eta e_{i};\psi)\big]-{\mathbb{E}}_{\theta}\big[s_{i}(z,\theta;\psi)\big]\bigg)-\sumop\displaylimits_{i=1}^{d}\big[\frac{\partial}{\partial\theta^{(i)}}\mathbb{E}_{\theta}[s_{i}(z,\theta;\psi)]\big]\bigg|`$ |  |
|  |  | $`\displaystyle=2\bigg|\sumop\displaylimits_{i=1}^{d}\big[\frac{\partial}{\partial\theta^{(i)}}\mathbb{E}_{\theta+\eta^{(i)}e_{i}}[s_{i}(z,\theta+\eta^{(i)}e_{i};\psi)]\big]-\sumop\displaylimits_{i=1}^{d}\big[\frac{\partial}{\partial\theta^{(i)}}\mathbb{E}_{\theta}[s_{i}(z,\theta;\psi)]\big]\bigg|`$ |  |
|  |  | $`\displaystyle\leqslant 2C_{K^{2\eta},R}(\sumop\displaylimits_{i=1}^{d}\eta^{(i)})`$ |  |
|  |  | $`\displaystyle\leqslant 2d\eta C_{K^{2\eta},R},`$ |  |

the first inequality follows from a direct application of mean-value
theorem. ∎

###### Theorem B.15.

For fixed $`T\geqslant 1`$, under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
[B.10](#A2.Thmtheorem10 "Assumption B.10. ‣ B.1.6 Regularity for Score Matching ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
and
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),

|  |  |  |
|----|----|----|
|  |
``` math
\max_{0\leqslant j\leqslant T}J(\widehat{\theta}_{j};\widehat{\psi}_{n,k}(\widehat{\theta}_{j}))=O_{P}\left(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta+a_{n,k}\right).
``` |  |

###### Proof.

Fix $`\delta>0`$. Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives a compact set $`K`$ such that

|     |                                                                    |     |
|-----|--------------------------------------------------------------------|-----|
|     |
       ``` math
       P\{\widehat{\theta}_{j}\in K,\ 0\leqslant j\leqslant T\}>1-\delta.
       ```                                                                 |     |

Choose $`R`$ larger than the constant in Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c)
for this set $`K`$. By Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d),
$`R`$ may be chosen so that, uniformly over deterministic
$`\theta\in K`$, the fitted parameter belongs to _(R) and satisfies the
approximate-minimization inequality with probability tending to one.

By Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Dudley’s uniform entropy bound, c.f. Corollary 19.35 in \[[Van der
Vaart, 2000](#bib.bib15)\], we have

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\sup_{\theta\in K}\mathbb{E}_{\theta}\bigg[\sup_{\psi\in{}_{R}}\bigg|\widehat{\mathbb{E}}_{\theta,n}\big[\|s(z,\theta;\psi)\|^{2}\big]-\mathbb{E}_{\theta}\big[\|s(z,\theta;\psi)\|^{2}\big]\bigg|\bigg]`$ | $`\displaystyle\lesssim\frac{1}{\sqrt{n}},`$ |  |
|  | $`\displaystyle\sup_{\theta\in K}\mathbb{E}_{\theta}\bigg[\sup_{\psi\in{}_{R}}\bigg|\widehat{\mathbb{E}}_{\theta,n}\big[\dfrac{\partial}{\partial\theta^{(i)}}s_{i}(z,\theta;\psi)\big]-\mathbb{E}_{\theta}\big[\dfrac{\partial}{\partial\theta^{(i)}}s_{i}(z,\theta;\psi)\big]\bigg|\bigg]`$ | $`\displaystyle\lesssim\frac{1}{\sqrt{n}},`$ |  |
|  | $`\displaystyle\sup_{\theta\in K}\mathbb{E}_{\theta}\bigg[\sup_{\psi\in{}_{R}}\bigg|\widehat{\mathbb{E}}_{\theta,k}\big[s_{i}(z,\theta;\psi)\big]-\mathbb{E}_{\theta}\big[s_{i}(z,\theta;\psi)\big]\bigg|\bigg]`$ | $`\displaystyle\lesssim\frac{1}{\sqrt{k}}.`$ |  |

The same bounds apply to the samples from
$`\mathcal{D}(\theta+\eta e_{i})`$, because these parameters belong to
the compact enlargement of $`K`$ for all sufficiently small $`\eta`$.
Markov’s inequality and Lemma
[B.14](#A2.Thmtheorem14 "Lemma B.14. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
therefore give, for every $`\varepsilon>0`$, a constant $`C<\infty`$
such that, for all sufficiently large $`n,k`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}P\left\{\sup_{\psi\in{}_{R}}|\widehat{J}_{n,k}(\theta;\psi)-J(\theta;\psi)|>C\left(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta\right)\right\}<\varepsilon.
``` |  |

We apply this uniform tail bound separately at each trajectory point.

Conditional on all observations preceding batch $`j`$,
$`\widehat{\theta}_{j}`$ is fixed and the samples used to fit
$`\widehat{\psi}_{n,k}(\widehat{\theta}_{j})`$ are drawn at that
deployment parameter. The preceding bound and Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d)
therefore apply at $`\widehat{\theta}_{j}`$. Since $`T`$ is fixed, a
union bound makes them hold simultaneously for
$`0\leqslant j\leqslant T`$. On this event, let
$`\psi^{\ast}(\widehat{\theta}_{j})\in{}_{R}`$ be given by Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c).
Then

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle J(\widehat{\theta}_{j};\widehat{\psi}_{n,k}(\widehat{\theta}_{j}))`$ | $`\displaystyle\leqslant 2\sup_{\psi\in{}_{R}}|\widehat{J}_{n,k}(\widehat{\theta}_{j};\psi)-J(\widehat{\theta}_{j};\psi)|+a_{n,k},`$ |  |

because $`J(\widehat{\theta}_{j};\psi^{\ast}(\widehat{\theta}_{j}))=0`$.
Taking the maximum over the fixed set of indices proves the result.

∎

#### B.1.10 Proof of Consistency of Perturbation-based estimator (Theorem [3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"))

###### Proof of Theorem [3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

By Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d),
$`\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\succeq\gamma I_{d}`$
and
$`H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\succeq\gamma I_{d}`$.
Both inverses therefore have spectral norm at most $`1/\gamma`$.

We compare the estimator with $`\nabla G(\theta_{t})`$ through six
terms. Fix $`\delta>0`$. Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives a compact set $`K`$ such that

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb P\{\widehat{\theta}_{t},\widehat{\theta}_{t+1},\theta_{t},\theta_{t+1}\in K\}>1-\delta.
``` |  |

The uniform tail bound in Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d)
then allows us to choose $`R>0`$ such that

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb P\{\|\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\|_{2}\leqslant R\}\geqslant\delta
``` |  |

for all sufficiently large $`n,k`$. We work on the intersection of these
two events.

Let

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla G}_{t,1}:=-{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\nabla_{\theta}\ell(z;\widehat{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big].
``` |  |

Step 1: Convergence of
$`\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t}\|_{2}`$

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t}\|_{2}`$ | $`\displaystyle\lesssim\big\|{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}-\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\big\|_{2}\cdot\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\big\|\nabla_{\theta}\ell(z;\widehat{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big\|_{2}\big],`$ |  |

We next verify the rate for the empirical Hessian. By Assumption
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c),

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\|\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\|_{2}`$ |  |
|  |  | $`\displaystyle\leqslant\left\{\frac{1}{n}\sumop\displaylimits_{i=1}^{n}L_{K}^{H}(Z_{t,i})+\mathbb E_{\widehat{\theta}_{t}}[L_{K}^{H}(Z)]\right\}\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2}`$ |  |
|  |  | $`\displaystyle+\left\|\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{t,i};\theta_{t+1})-H_{\widehat{\theta}_{t}}(\theta_{t+1})\right\|_{2}.`$ |  |

The quantity in braces is $`O_{P}(1)`$. Conditional on the preceding
observations, the last term is an average of independent centered
matrices. Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c)
bounds their spectral norms, so this term is $`O_{P}(n^{-1/2})`$.
Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
now gives

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\|_{2}=O_{P}(n^{-1/2}).
``` |  |

Since both Hessians are bounded below by $`\gamma I_{d}`$,

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\|H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}-\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\|_{2}\leqslant\gamma^{-2}\|\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\|_{2}=O_{P}(n^{-1/2}).`$ |  |

Assumption
[B.13](#A2.Thmtheorem13 "Assumption B.13. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
also gives

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\mathbb E}_{\widehat{\theta}_{t},n}\big[\|\nabla\ell(z;\widehat{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\|_{2}\big]=O_{P}(1).
``` |  |

This implies that

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t}\|_{2}=O_{P}(\frac{1}{\sqrt{n}}).
``` |  |

Let

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla G}_{t,2}:=-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big].
``` |  |

Step 2: Convergence of
$`\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t,2}\|`$.

Since
$`\|H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\|_{2}\leqslant 1/\gamma`$,
we have

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t,2}\|_{2}`$ | $`\displaystyle\lesssim\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2}\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\big\|s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))\big\|_{2}\big]`$ |  |
|  |  | $`\displaystyle=\frac{1}{n}\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2}\sumop\displaylimits_{l=1}^{n}\big\|s(Z_{t,l},\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))\big\|_{2}.`$ |  |

Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
bounds the last sample average uniformly on $`K\times{}_{R}`$, so it is
$`O_{P}(1)`$. Therefore,

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t,2}\|_{2}=O_{P}(\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2}),
``` |  |

thus combining the above result with Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives
$`\|\widehat{\nabla G}_{t,1}-\widehat{\nabla G}_{t,2}\|_{2}=O_{P}(\frac{1}{\sqrt{n}}).`$
Let

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla G}_{t,3}:=-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\mathbb{E}_{\widehat{\theta}_{t}}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big].
``` |  |

Step 3: Convergence of
$`\|\widehat{\nabla G}_{t,2}-\widehat{\nabla G}_{t,3}\|`$.

By the uniform empirical-process bound from Assumption
[B.13](#A2.Thmtheorem13 "Assumption B.13. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we have

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\widehat{\nabla G}_{t,2}-\widehat{\nabla G}_{t,3}\|_{2}`$ | $`\displaystyle\lesssim\bigg\|\widehat{\mathbb{E}}_{\widehat{\theta}_{t},n}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big]-\mathbb{E}_{\widehat{\theta}_{t}}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big]\bigg\|_{2}`$ |  |
|  |  | $`\displaystyle=O_{P}(\frac{1}{\sqrt{n}}).`$ |  |

Let

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla G}_{t,4}:=-H_{\theta_{t}}(\theta_{t+1})^{-1}\mathbb{E}_{\widehat{\theta}_{t}}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big].
``` |  |

Step 4: Convergence of
$`\|\widehat{\nabla G}_{t,3}-\widehat{\nabla G}_{t,4}\|_{2}`$. By
Assumption
[B.13](#A2.Thmtheorem13 "Assumption B.13. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we have

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb{E}_{\widehat{\theta}_{t}}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))^{\top}\big]=O_{P}(1),
``` |  |

thus

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\widehat{\nabla G}_{t,3}-\widehat{\nabla G}_{t,4}\|_{2}`$ | $`\displaystyle=O_{P}\big(\|H_{\theta_{t}}(\theta_{t+1})^{-1}-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\|_{2}\big)`$ |  |
|  |  | $`\displaystyle=O_{P}\bigg(\big\|H_{\theta_{t}}(\theta_{t+1})^{-1}\big(H_{\theta_{t}}(\theta_{t+1})-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\big)H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})^{-1}\big\|_{2}\bigg),`$ |  |

Using Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d)
and Assumption
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a),

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t,3}-\widehat{\nabla G}_{t,4}\|_{2}\lesssim\|\widehat{\theta}_{t}-\theta_{t}\|_{2}+\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2},
``` |  |

thus by Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t,3}-\widehat{\nabla G}_{t,4}\|_{2}=O_{P}(\frac{1}{\sqrt{n}}).
``` |  |

Let

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\nabla G}_{t,5}:=-H_{\theta_{t}}(\theta_{t+1})^{-1}\mathbb{E}_{\widehat{\theta}_{t}}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})\nabla_{\theta}\log p(z;\widehat{\theta}_{t})^{\top}\big].
``` |  |

Step 5: Convergence of
$`\|\widehat{\nabla G}_{t,4}-\widehat{\nabla G}_{t,5}\|_{2}`$.

By Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
the loss gradients have uniformly bounded second moments on $`K`$, and

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\widehat{\nabla G}_{t,4}-\widehat{\nabla G}_{t,5}\|^{2}`$ | $`\displaystyle\lesssim\bigg\{\mathbb{E}_{\widehat{\theta}_{t}}\bigg[\big\|\nabla_{\theta}\ell(z;\theta_{t+1})\big\|_{2}\times\big\|s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))-\nabla_{\theta}\log p(z;\widehat{\theta}_{t})\big\|_{2}\bigg]\bigg\}^{2}`$ |  |
|  |  | $`\displaystyle\leqslant\mathbb{E}_{\widehat{\theta}_{t}}\big[\big\|\nabla_{\theta}\ell(z;\theta_{t+1})\big\|_{2}^{2}\big]`$ |  |
|  |  | $`\displaystyle\times\mathbb{E}_{\widehat{\theta}_{t}}\big[\big\|s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))-\nabla_{\theta}\log p(z;\widehat{\theta}_{t})\big\|_{2}^{2}\big]\quad\mathrm{(by\ Cauchy-Schwarz\ inequality)}`$ |  |
|  |  | $`\displaystyle\lesssim\mathbb{E}_{\widehat{\theta}_{t}}\big[\big\|s(z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t}))-\nabla_{\theta}\log p(z;\widehat{\theta}_{t})\big\|_{2}^{2}\big]\quad\mathrm{(by\ compactness\ of\ K)}`$ |  |
|  |  | $`\displaystyle=O_{P}(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta+a_{n,k})\quad\mathrm{(by\ Theorem\ \ref{thm:discuss1})}.`$ |  |

Step 6: Convergence of
$`\|\widehat{\nabla G}_{t}-\nabla G(\theta_{t})\|`$.

Finally, recall that

|  |  |  |
|----|----|----|
|  |
``` math
\nabla G(\theta_{t})=-H_{\theta_{t}}(\theta_{t+1})^{-1}\mathbb{E}_{{\theta}_{t}}\big[\nabla_{\theta}\ell(z;{\theta}_{t+1})\nabla_{\theta}\log p(z;\theta_{t})^{\top}\big],
``` |  |

Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d)
and Assumption
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
give

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t,5}-\nabla G(\theta_{t})\|_{2}\lesssim\|\widehat{\theta}_{t}-\theta_{t}\|_{2}=O_{P}(n^{-1/2}),
``` |  |

where the last equality follows from Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
Combining the six bounds gives

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{\nabla G}_{t}-\nabla G(\theta_{t})\|_{2}^{2}=O_{P}(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta+a_{n,k}).
``` |  |

∎

#### B.1.11 Proof of Corollary [3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Corollary [3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Fix $`t\geqslant 0`$. Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives $`\widehat{\theta}_{t}\xrightarrow{P}\theta_{t}`$ and
$`\widehat{\theta}_{t+1}\xrightarrow{P}\theta_{t+1}`$. Let $`K`$ be a
compact convex set whose interior contains $`\theta_{t}`$ and
$`\theta_{t+1}`$. With probability tending to one,
$`\widehat{\theta}_{t},\widehat{\theta}_{t+1}\in K`$.

We first show
$`\widehat{\Sigma}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\xrightarrow{P}\Sigma_{\theta_{t}}(\theta_{t+1}).`$
On the event $`\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2}\leqslant r`$,

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\|\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})-H_{\theta_{t}}(\theta_{t+1})\|_{2}`$ |  |
|  |  | $`\displaystyle\leqslant\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\sup_{\|\theta-\theta_{t+1}\|_{2}\leqslant r}\|\nabla^{2}\ell(Z_{t,i};\theta)-\nabla^{2}\ell(Z_{t,i};\theta_{t+1})\|_{2}`$ |  |
|  |  | $`\displaystyle+\left\|\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{t,i};\theta_{t+1})-H_{\widehat{\theta}_{t}}(\theta_{t+1})\right\|_{2}+\|H_{\widehat{\theta}_{t}}(\theta_{t+1})-H_{\theta_{t}}(\theta_{t+1})\|_{2}.`$ |  |

By Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c),
the Hessian is continuous in its parameter argument and has spectral
norm at most $`\beta`$. Dominated convergence, together with Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
makes the first term arbitrarily small as $`r\downarrow 0`$. Conditional
on the preceding observations, the second term is an average of $`n`$
i.i.d. bounded matrices with mean
$`H_{\widehat{\theta}_{t}}(\theta_{t+1})`$, so it is
$`O_{P}(n^{-1/2})`$. The third term tends to zero by Assumption
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a).
Hence

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\xrightarrow{P}H_{\theta_{t}}(\theta_{t+1}).
``` |  |

The same decomposition applies to the middle factor of
$`\widehat{\Sigma}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})`$. On
$`K`$, Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)–(c)
gives $`\|\nabla\ell(z;\theta)\|_{2}\leqslant L_{K}(z)`$ and

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\|\nabla\ell(z;\widehat{\theta}_{t+1})\nabla\ell(z;\widehat{\theta}_{t+1})^{\top}-\nabla\ell(z;\theta_{t+1})\nabla\ell(z;\theta_{t+1})^{\top}\|_{2}\leqslant 2\beta L_{K}(z)\|\widehat{\theta}_{t+1}-\theta_{t+1}\|_{2}.`$ |  |

The third-moment bound on $`L_{K}`$ gives the required uniform
integrability. The conditional weak law and Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
therefore yield

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t+1})\nabla\ell(Z_{t,i};\widehat{\theta}_{t+1})^{\top}`$ |  |
|  |  | $`\displaystyle\xrightarrow{P}\mathbb E_{Z\sim\mathcal{D}(\theta_{t})}[\nabla\ell(Z;\theta_{t+1})\nabla\ell(Z;\theta_{t+1})^{\top}]`$ |  |
|  |  | $`\displaystyle=\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}(\nabla\ell(Z;\theta_{t+1})),`$ |  |

where the last equality uses
$`\mathbb E_{Z\sim\mathcal{D}(\theta_{t})}[\nabla\ell(Z;\theta_{t+1})]=0`$.
Finally,
$`\widehat{H}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1})\succeq\gamma I_{d}`$
by Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d),
so continuity of matrix inversion proves the claimed convergence of
$`\widehat{\Sigma}`$.

Theorem
[3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives $`\widehat{\nabla G}_{t}\xrightarrow{P}\nabla G(\theta_{t})`$.
Since $`\widehat{V}_{0}=V_{0}=0`$, the recursion in
([3](#S3.E3 "Equation 3 ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")),
induction on $`t`$, and the continuous mapping theorem give
$`\widehat{V}_{t}\xrightarrow{P}V_{t}`$ for each fixed $`t\geqslant 1`$.

If $`V_{t}`$ is nonsingular, then $`\widehat{V}_{t}`$ is nonsingular
with probability tending to one and
$`\widehat{V}_{t}^{-1}\xrightarrow{P}V_{t}^{-1}`$. Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Slutsky’s lemma give

|  |  |  |
|----|----|----|
|  |
``` math
n(\widehat{\theta}_{t}-\theta_{t})^{\top}\widehat{V}_{t}^{-1}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\chi^{2}_{d}.
``` |  |

The continuity of the $`\chi^{2}_{d}`$ distribution function proves
$`\mathbb P(\theta_{t}\in\widehat{\mathcal{R}}_{t})\to 1-\delta`$. ∎

#### B.1.12 Discussion on Assumptions for Perturbation Method

Now we show that the richness assumption and tight optimizer assumption
(Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c-d))
hold when we have a correctly-specified generalized affine score model.

###### Assumption B.16.

\(a\) We assume that $`s(z,\theta;\psi)`$ is linear in $`\psi`$:
Specifically, there exists fixed functions
$`A:\mathcal{Z}\times\Theta\to\mathbb{R}^{d\times p}`$ and
$`B:\mathcal{Z}\times\Theta\to\mathbb{R}^{d}`$, such that for each
$`\theta,\check{\theta}`$,
$`s(z,\theta;\psi)=A(z,\theta)\psi+B(z,\theta)`$, and for any compact
set $`K`$, there exists constant $`\gamma_{K}>0`$ such that, for each
$`\theta,\check{\theta}\in K`$,

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb{E}_{\check{\theta}}[A(z,\theta)^{\top}A(z,\theta)]\succeq\gamma_{K}I_{p}.
``` |  |

Further, we assume that on any compact subset $`K\subset\Theta`$ and
$`R>0`$, there exists $`C_{K,R}:\mathcal{Z}\to\mathbb{R}`$ and
$`L_{K,R}:\mathcal{Z}\to\mathbb{R}`$, such that denote

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle g_{1}^{(i)}(z,\theta;\psi):=\|A(z,\theta)\psi+B(z,\theta)\|^{2}_{2},\ g_{2}^{(i)}(z,\theta;\psi)=A_{i}(z,\theta)\psi+B_{i}(z,\theta),\`$ |  |
|  |  | $`\displaystyle g_{3}^{(i)}(z,\theta;\psi)=\partial_{\theta^{(i)}}A_{i}(z,\theta)\psi+\partial_{\theta^{(i)}}B_{i}(z,\theta),\ i\in[d],`$ |  |

where we denote $`A_{i}`$ the $`i`$-th row of matrix-valued function
$`A`$, and $`B_{i}`$ the $`i`$-th component of vector valued function
$`B`$. Then for any function
$`g(z,\theta;\psi)\in\{g_{j}^{(i)}(z,\theta;\psi):i\in[d],j\in[3]\}`$
for any $`z\in\mathcal{Z}`$, $`\theta,\theta^{\prime}\in K`$,
$`\psi,\psi^{\prime}\in{}_{R}`$, the following inequality holds:

|     |                                    |     |
|-----|------------------------------------|-----|
|     |
       ``` math
       |g(z,\theta)|\leqslant C_{K,R}(z),
       ```                                 |     |

|  |  |  |
|----|----|----|
|  |
``` math
|g(z,\theta;\psi)-g(z,\theta^{\prime};\psi^{\prime})|\leqslant L_{K,R}(z)\big(\|\theta-\theta^{\prime}\|_{2}+\|\psi-\psi^{\prime}\|_{2}\big),
``` |  |

with
$`\sup_{\theta\in K^{1}}\mathbb{E}_{\theta}[C_{K,R}{(Z)}^{2}+L_{K,R}{(Z)}^{2}]<\infty,`$
where $`K^{1}`$ is the $`1`$-enlargement of $`K\subset\Theta`$.

\(b\) We assume that for a compact subset $`K\subset\Theta`$, there
exists $`R_{K}>0`$ such that for each $`{\theta}\in K`$, there exists a
$`\psi^{*}(\theta)\in{}_{R_{K}}`$, such that for any
$`z\in\mathcal{Z}`$, the following identity holds,

|  |  |  |
|----|----|----|
|  |
``` math
A(z,\theta)\psi^{*}(\theta)+B(z,\theta)=\nabla_{\theta}\log p(z;\theta).
``` |  |

###### Remark B.17.

We remark that richness assumption (Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c))
holds due to Assumption
[B.16](#A2.Thmtheorem16 "Assumption B.16. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b).
Further, under Assumption
[B.16](#A2.Thmtheorem16 "Assumption B.16. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the score matching objectives can be rewritten as

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle{J}_{\eta}(\theta;\psi):`$ | $`\displaystyle=\mathbb{E}_{\theta}[\|A(z,\theta)\psi+B(z,\theta)\|^{2}]+2\sumop\displaylimits_{i=1}^{d}\mathbb{E}_{\theta}\big[\big(\partial_{\theta^{(i)}}A_{i}(z,\theta)\big)\psi+\partial_{\theta^{(i)}}B_{i}(z,\theta)\big]+\mathbb E_{\theta}[\|\nabla_{\theta}\log p(z;\theta)\|_{2}^{2}]`$ |  |
|  |  | $`\displaystyle-\frac{2}{\eta}\sumop\displaylimits_{i=1}^{d}\bigg({\mathbb{E}}_{\theta+\eta e_{i}}\big[A_{i}(z,\theta+\eta e_{i})\big]-{\mathbb{E}}_{\theta}\big[A_{i}(z,\theta)\big]\bigg)\psi`$ |  |
|  |  | $`\displaystyle-\frac{2}{\eta}\sumop\displaylimits_{i=1}^{d}\bigg({\mathbb{E}}_{\theta+\eta e_{i}}\big[B_{i}(z,\theta+\eta e_{i})\big]-{\mathbb{E}}_{\theta}\big[B_{i}(z,\theta)\big]\bigg),`$ |  |
|  | $`\displaystyle\widehat{J}_{n,k}(\theta;\psi):`$ | $`\displaystyle=\widehat{\mathbb{E}}_{\theta,n}[\|A(z,\theta)\psi+B(z,\theta)\|^{2}]+2\sumop\displaylimits_{i=1}^{d}\widehat{\mathbb{E}}_{\theta,n}\big[\big(\partial_{\theta^{(i)}}A_{i}(z,\theta)\big)\psi+\partial_{\theta^{(i)}}B_{i}(z,\theta)\big]+\mathbb E_{\theta}[\|\nabla_{\theta}\log p(z;\theta)\|_{2}^{2}]`$ |  |
|  |  | $`\displaystyle-\frac{2}{\eta}\sumop\displaylimits_{i=1}^{d}\bigg(\widehat{\mathbb{E}}_{\theta+\eta e_{i},k}\big[A_{i}(z,\theta+\eta e_{i})\big]-\widehat{\mathbb{E}}_{\theta,n}\big[A_{i}(z,\theta)\big]\bigg)\psi`$ |  |
|  |  | $`\displaystyle-\frac{2}{\eta}\sumop\displaylimits_{i=1}^{d}\bigg(\widehat{\mathbb{E}}_{\theta+\eta e_{i},k}\big[B_{i}(z,\theta+\eta e_{i})\big]-\widehat{\mathbb{E}}_{\theta,n}\big[B_{i}(z,\theta)\big]\bigg).`$ |  |

We note that $`\widehat{J}_{n,k}(\theta,\psi)`$ is convex in $`\psi`$,
and $`J_{\eta}(\theta,\psi),J(\theta,\psi)`$ are
$`\gamma_{K}`$-strictly-convex in $`\psi`$ given compact subset $`K`$.

We now prove that under this assumption, the tight optimizer assumption
(Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d))
can be achieved.

###### Lemma B.18.

Under Assumption
[B.10](#A2.Thmtheorem10 "Assumption B.10. ‣ B.1.6 Regularity for Score Matching ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a)
and Assumption
[B.16](#A2.Thmtheorem16 "Assumption B.16. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we define $`\widehat{\psi}_{n,k}(\theta)`$ as below: when
$`\widehat{J}_{n,k}`$ is strictly-convex in $`\psi`$, we take
$`\widehat{\psi}_{n,k}(\theta)`$ to be the empirical minimizer of
$`\widehat{J}_{n,k}`$, otherwise we set it to be $`0`$. Then for any
compact subset $`K\subset\Theta`$, there exists $`R_{K,1}>0`$ such that
as $`\eta\to 0`$, $`\min(n,k)\to\infty`$ and
$`\eta\sqrt{\min(n,k)}\to\infty`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}P(\|\widehat{\psi}_{n,k}(\theta)\|>R_{K,1})\to 0.
``` |  |

###### Proof of Lemma [B.18](#A2.Thmtheorem18 "Lemma B.18. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Fix $`\delta>0`$. By Assumption
[B.16](#A2.Thmtheorem16 "Assumption B.16. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a),
for a fixed compact subset $`K\subset\Theta`$ and $`\theta\in K`$, the
quadratic term in $`\widehat{J}_{n,k}(\theta;\psi)`$ is

|  |  |  |
|----|----|----|
|  |
``` math
\psi^{\top}\widehat{\mathbb{E}}_{\theta,n}[A(z,\theta)^{\top}A(z,\theta)]\psi,
``` |  |

by uniform law of large numbers, when $`\min(n,k)\to\infty`$, the matrix
$`\widehat{\mathbb{E}}_{\theta,n}[A(z,\theta)^{\top}A(z,\theta)]`$
converges in probability uniformly to
$`\mathbb{E}_{\theta}[A(z,\theta)^{\top}A(z,\theta)]`$, which is
positive definite, thus

|  |  |  |
|----|----|----|
|  |
``` math
\inf_{\check{\theta}\in K}P\big(\widehat{J}_{n,k}(\check{\theta},\psi)\text{ is strongly convex in }\psi\big)\to 1,
``` |  |

as $`\min(n,k)\to\infty`$. This implies that the
$`\widehat{\psi}_{n,k}(\check{\theta})`$ is exact minimizer with
probability converging to $`1`$. Therefore we shall assume
$`\widehat{J}_{n,k}`$ is strongly convex, and thus
$`\widehat{\psi}_{n,k}(\theta)`$ is uniquely defined. By Assumption
[B.16](#A2.Thmtheorem16 "Assumption B.16. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
for fixed compact set $`K`$ there exists a constant $`R_{K}`$, such that
$`\psi^{*}(\theta)`$ the minimizer of $`J(\theta;\psi)`$ lies in the
ball $`{}_{R_{K}}`$ for any $`\theta`$. By strong convexity, for
$`\psi\in\Psi`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
J(\theta;\psi)-J(\theta;\psi^{\ast}(\theta))\geqslant\frac{\gamma_{K}}{2}\|\psi-\psi^{*}(\theta)\|_{2}^{2},
``` |  |

Thus take $`R_{K,1}=R_{K}+1`$, then for any $`\psi`$ with
$`\|\psi\|_{2}=R_{K,1}`$,

|     |                                                                 |     |
|-----|-----------------------------------------------------------------|-----|
|     |
       ``` math
       J(\theta;\psi)-J(\theta;\psi^{*}(\theta))\geqslant\gamma_{K}/2.
       ```                                                              |     |

Now for notation simplicity, we write $`\widehat{\psi}_{n,k}`$ and
$`\psi^{*}`$ for $`\widehat{\psi}_{n,k}(\theta)`$ and
$`\psi^{*}(\theta)`$ respectively. When $`\widehat{\psi}_{n,k}`$ is
outside $`{}_{R_{K,1}}`$, set $`\bar{\psi}_{n,k}`$ to be the
intersection of the sphere $`\partial{}_{R_{K,1}}`$ and the line segment
connecting $`\widehat{\psi}_{n,k}`$ and $`\psi^{*}`$. Then since
$`\|\bar{\psi}_{n,k}\|_{2}=R_{K,1}`$, we have

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle P(\|\widehat{\psi}_{n,k}(\theta)\|_{2}>R_{K,1})`$ | $`\displaystyle\leqslant P\big(J(\theta,\bar{\psi}_{n,k})-J(\theta,\psi^{*})\geqslant\frac{\gamma_{K}}{2}\big)`$ |  |

Since $`\widehat{\psi}_{n,k}`$ is a global minimizer of
$`\widehat{J}_{n,k}`$, convexity gives
$`\widehat{J}_{n,k}(\theta;\bar{\psi}_{n,k})\leqslant\widehat{J}_{n,k}(\theta;\psi^{\ast})`$.
Therefore,

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle J(\theta,\bar{\psi}_{n,k})-J(\theta,{\psi}^{*})`$ | $`\displaystyle=J(\theta,\bar{\psi}_{n,k})-\widehat{J}_{n,k}(\theta,\bar{\psi}_{n,k})+\big(\widehat{J}_{n,k}(\theta,\bar{\psi}_{n,k})-\widehat{J}_{n,k}(\theta,\psi^{*})\big)-\big(J(\theta,\psi^{*})-\widehat{J}_{n,k}(\theta,\psi^{*})\big)`$ |  |
|  |  | $`\displaystyle\leqslant J(\theta,\bar{\psi}_{n,k})-\widehat{J}_{n,k}(\theta,\bar{\psi}_{n,k})-\big(J(\theta,\psi^{*})-\widehat{J}_{n,k}(\theta,\psi^{*})\big)`$ |  |
|  |  | $`\displaystyle\leqslant 2\sup_{\psi\in{}_{R_{K,1}}}\big|J(\theta,\psi)-\widehat{J}_{n,k}(\theta,\psi)\big|`$ |  |
|  |  | $`\displaystyle\leqslant 2\sup_{\psi\in{}_{R_{K,1}}}\big|J_{\eta}(\theta,\psi)-\widehat{J}_{n,k}(\theta,\psi)\big|+2\sup_{\psi\in{}_{R_{K,1}}}\big|J_{\eta}(\theta,\psi)-{J}(\theta,\psi)\big|`$ |  |
|  |  | $`\displaystyle\leqslant 2\sup_{\psi\in{}_{R_{K,1}}}\big|J_{\eta}(\theta,\psi)-\widehat{J}_{n,k}(\theta,\psi)\big|+C_{K}d\eta,`$ |  |

where the last inequality follows from Lemma
[B.14](#A2.Thmtheorem14 "Lemma B.14. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
Note that under Assumption
[B.16](#A2.Thmtheorem16 "Assumption B.16. ‣ B.1.12 Discussion on Assumptions for Perturbation Method ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the score function $`s(z,\theta;\psi)`$ satisfies Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
thus as in the proof of Theorem
[B.15](#A2.Thmtheorem15 "Theorem B.15. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we have the following bound holds under $`P_{\theta}`$, uniformly over
$`\theta\in K`$:

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\psi\in{}_{R}}\big|\widehat{J}_{n,k}(\theta;\psi)-{J}_{\eta}(\theta;\psi)\big|=O_{P}(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}).
``` |  |

Thus when $`\min(k,n)\to\infty`$, and $`\eta\to 0`$, we have

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\sup_{\theta\in K}P\big(\|\widehat{\psi}_{n,k}\|>R_{K,1}\big)`$ | $`\displaystyle\leqslant\sup_{\theta\in K}P_{\theta}\big(\sup_{\psi\in{}_{R_{K,1}}}\big|J_{\eta}(\theta,\psi)-\widehat{J}_{n,k}(\theta,\psi)\big|+C_{K}d\eta>\frac{\gamma_{K}}{2}\big)\to 0.`$ |  |

∎

### B.2 Proofs for Section [4](#S4 "4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

#### B.2.1 Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

For $`\theta\in\Theta`$ and $`z\in\mathcal{Z}`$, let

|     |                                                   |     |
|-----|---------------------------------------------------|-----|
|     |
       ``` math
       f_{\theta}(z):=\theta-\alpha\nabla\ell(z;\theta).
       ```                                                |     |

Then
$`\widehat{\theta}_{t+1}=n^{-1}\sumop\displaylimits_{i=1}^{n}f_{\widehat{\theta}_{t}}(Z_{t,i})`$,
and the two quantities in Lemma
[B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
are

|  |  |  |
|----|----|----|
|  |
``` math
F(\theta)=\theta-\alpha S(\theta),\qquad V_{f}(\theta)=\alpha^{2}\operatorname{Cov}_{Z\sim\mathcal{D}(\theta)}\bigl(\nabla\ell(Z;\theta)\bigr).
``` |  |

In particular, $`\theta_{t+1}=F(\theta_{t})`$ and
$`\nabla F(\theta)=I_{d}-\alpha\nabla S(\theta)`$.

We verify the conditions of Lemma
[B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
Fix a compact set $`K\subset\Theta`$. By Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
$`S`$ is continuously differentiable, and hence $`F`$ is continuously
differentiable and bounded on $`K`$. To prove that $`V_{f}`$ is
continuous, it remains to consider the second moment of the gradients.
For $`\theta,\theta^{\prime}\in K`$, the difference between this second
moment at $`\theta^{\prime}`$ and at $`\theta`$ is

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\mathbb\displaystyle E_{Z\sim\mathcal{D}(\theta^{\prime})}\left[\nabla\ell(Z;\theta^{\prime})\nabla\ell(Z;\theta^{\prime})^{\top}-\nabla\ell(Z;\theta)\nabla\ell(Z;\theta)^{\top}\right]`$ |  |
|  |  | $`\displaystyle+\mathbb E_{Z\sim\mathcal{D}(\theta)}\left[\nabla\ell(Z;\theta)\nabla\ell(Z;\theta)^{\top}\left\{\frac{p(Z;\theta^{\prime})}{p(Z;\theta)}-1\right\}\right].`$ |  |

Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)–(c)
bounds the spectral norm of the matrix in the first expectation by
$`2\beta L_{K}(Z)\|\theta^{\prime}-\theta\|_{2}`$. Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
bounds the factor in braces in the second expectation by a constant
multiple of $`\|\theta^{\prime}-\theta\|_{2}`$. The moment bound in
Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
therefore proves continuity. The same assumption also gives

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}\mathbb E_{Z\sim\mathcal{D}(\theta)}\bigl[\|f_{\theta}(Z)\|_{2}^{3}\bigr]<\infty.
``` |  |

We now proceed by induction on $`t`$. For $`t=0`$,
$`\widehat{\theta}_{0}=\theta_{0}`$ and $`V_{0}=0`$. Suppose that
$`\sqrt{n}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\mathcal{N}(0,V_{t})`$.
Conditional on all preceding observations and estimators,
$`Z_{t,1},\ldots,Z_{t,n}`$ are i.i.d. from
$`\mathcal{D}(\widehat{\theta}_{t})`$. Lemma
[B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
with $`\check{\theta}_{n}=\widehat{\theta}_{t}`$,
$`\check{\theta}_{0}=\theta_{t}`$, and $`A_{1}=V_{t}`$, gives

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\sqrt{n}(\widehat{\theta}_{t+1}-\theta_{t+1})`$ | $`\displaystyle=\sqrt{n}\left\{\frac{1}{n}\sumop\displaylimits_{i=1}^{n}f_{\widehat{\theta}_{t}}(Z_{t,i})-F(\theta_{t})\right\}`$ |  |
|  |  | $`\displaystyle\xrightarrow{d}\mathcal{N}\left(0,V_{f}(\theta_{t})+\nabla F(\theta_{t})V_{t}\nabla F(\theta_{t})^{\top}\right).`$ |  |

The covariance on the right is $`V_{t+1}`$ by the recursion in Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
∎

#### B.2.2 Proof of Theorem [4.2](#S4.Thmtheorem2 "Theorem 4.2. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Theorem [4.2](#S4.Thmtheorem2 "Theorem 4.2. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

We first derive the formula for $`\nabla S`$ in Section
[4.2](#S4.SS2 "4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
The map

|  |  |  |
|----|----|----|
|  |
``` math
(\widetilde{\theta},\theta)\longmapsto\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[\nabla\ell(Z;\theta)]
``` |  |

is continuously differentiable by Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
and $`S(\theta)`$ is its value at $`\widetilde{\theta}=\theta`$. Its
Jacobian in the second argument is $`H_{\theta}(\theta)`$. Its Jacobian
in the first argument is

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\theta)}[\nabla\ell(Z;\theta)\nabla_{\theta}\log p(Z;\theta)^{\top}],
``` |  |

by the likelihood-ratio argument in the proof of Proposition
[3.1](#S3.Thmtheorem1 "Proposition 3.1. ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
Adding the two Jacobians gives the formula in Section
[4.2](#S4.SS2 "4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Fix $`t\geqslant 0`$. Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives $`\|\widehat{\theta}_{t}-\theta_{t}\|_{2}=O_{P}(n^{-1/2})`$, with
equality to zero when $`t=0`$. The proof of Theorem
[B.15](#A2.Thmtheorem15 "Theorem B.15. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
applied to the finite set
$`\{\widehat{\theta}_{j}:0\leqslant j\leqslant t\}`$ and using Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
in place of Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
gives

|  |  |  |
|----|----|----|
|  |
``` math
J\bigl(\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr)=O_{P}\left(\frac{1}{\sqrt{n}}+\frac{1}{\eta\sqrt{\min(n,k)}}+\eta+a_{n,k}\right).
``` |  |

Let $`K`$ be a compact set whose interior contains $`\theta_{t}`$. By
the preceding convergence and Assumption
[B.11](#A2.Thmtheorem11 "Assumption B.11. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(d),
we may restrict attention to an event on which
$`\widehat{\theta}_{t}\in K`$ and
$`\|\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\|_{2}\leqslant R`$, for a
fixed $`R<\infty`$, with probability arbitrarily close to one. On this
event, we split $`\widehat{\nabla S}_{t}-\nabla S(\theta_{t})`$ into
four terms.

*Hessian term.* Conditional on all preceding observations and
estimators, $`\nabla^{2}\ell(Z_{t,i};\widehat{\theta}_{t})`$,
$`i\in[n]`$, are i.i.d. with mean
$`H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t})`$. Their spectral norms
are bounded by $`\beta`$ under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c),
so

|  |  |  |
|----|----|----|
|  |
``` math
\left\|\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{t,i};\widehat{\theta}_{t})-H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t})\right\|_{2}=O_{P}(n^{-1/2}).
``` |  |

Assumption
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a)
also gives

|  |  |  |
|----|----|----|
|  |
``` math
\|H_{\widehat{\theta}_{t}}(\widehat{\theta}_{t})-H_{\theta_{t}}(\theta_{t})\|_{2}\lesssim\|\widehat{\theta}_{t}-\theta_{t}\|_{2}=O_{P}(n^{-1/2}).
``` |  |

*Sampling term.* For the product of the loss gradient and the fitted
score, Assumption
[B.13](#A2.Thmtheorem13 "Assumption B.13. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives a square-integrable envelope and a square-integrable Lipschitz
bound over $`K\times K\times{}_{R}`$. The same uniform empirical-process
bound used in the proof of Theorem
[B.15](#A2.Thmtheorem15 "Theorem B.15. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
therefore gives

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\left\|\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t})s\bigl(Z_{t,i},\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr)^{\top}\right.`$ |  |
|  |  | $`\displaystyle\left.-\mathbb E_{Z\sim\mathcal{D}(\widehat{\theta}_{t})}\left[\nabla\ell(Z;\widehat{\theta}_{t})s\bigl(Z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr)^{\top}\right]\right\|_{2}=O_{P}(n^{-1/2}).`$ |  |

*Score term.* By the Cauchy–Schwarz inequality, the squared spectral
norm of

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\widehat{\theta}_{t})}\left[\nabla\ell(Z;\widehat{\theta}_{t})\left\{s\bigl(Z,\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr)-\nabla_{\theta}\log p(Z;\widehat{\theta}_{t})\right\}^{\top}\right]
``` |  |

is bounded by

|  |  |  |
|----|----|----|
|  |
``` math
\mathbb E_{Z\sim\mathcal{D}(\widehat{\theta}_{t})}[\|\nabla\ell(Z;\widehat{\theta}_{t})\|_{2}^{2}]J\bigl(\widehat{\theta}_{t};\widehat{\psi}_{n,k}(\widehat{\theta}_{t})\bigr).
``` |  |

The first factor is bounded on $`K`$ by Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
so this term contributes
$`O_{P}(n^{-1/2}+\{\eta\sqrt{\min(n,k)}\}^{-1}+\eta+a_{n,k})`$ to the
squared error.

*Distribution term.* Assumption
[B.12](#A2.Thmtheorem12 "Assumption B.12. ‣ B.1.8 Policy Perturbation ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
gives

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\left\|\mathbb E_{Z\sim\mathcal{D}(\widehat{\theta}_{t})}[\nabla\ell(Z;\widehat{\theta}_{t})\nabla_{\theta}\log p(Z;\widehat{\theta}_{t})^{\top}]-\mathbb E_{Z\sim\mathcal{D}(\theta_{t})}[\nabla\ell(Z;\theta_{t})\nabla_{\theta}\log p(Z;\theta_{t})^{\top}]\right\|_{2}`$ |  |
|  |  | $`\displaystyle\lesssim\|\widehat{\theta}_{t}-\theta_{t}\|_{2}=O_{P}(n^{-1/2}).`$ |  |

The squared norm of the sum is at most four times the sum of the four
squared norms. Combining the preceding bounds proves the result. ∎

#### B.2.3 Proof of Corollary [4.3](#S4.Thmtheorem3 "Corollary 4.3. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Corollary [4.3](#S4.Thmtheorem3 "Corollary 4.3. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Fix $`t\geqslant 0`$. By Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\widehat{\theta}_{t}\xrightarrow{P}\theta_{t}`$ and
$`\widehat{\theta}_{t+1}\xrightarrow{P}\theta_{t+1}`$. The REGD update
therefore gives

|  |  |  |
|----|----|----|
|  |
``` math
\frac{\widehat{\theta}_{t}-\widehat{\theta}_{t+1}}{\alpha}\xrightarrow{P}\frac{\theta_{t}-\theta_{t+1}}{\alpha}=S(\theta_{t}).
``` |  |

Conditional on all preceding observations and estimators,

|  |  |  |
|----|----|----|
|  |
``` math
\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t})\nabla\ell(Z_{t,i};\widehat{\theta}_{t})^{\top}
``` |  |

is an average of i.i.d. matrices. On a compact neighborhood $`K`$ of
$`\theta_{t}`$, their spectral norms are bounded by $`L_{K}(Z)^{2}`$.
The third-moment bound in Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
gives uniform integrability, and the conditional weak law of large
numbers gives

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla\ell(Z_{t,i};\widehat{\theta}_{t})\nabla\ell(Z_{t,i};\widehat{\theta}_{t})^{\top}-\mathbb E_{Z\sim\mathcal{D}(\widehat{\theta}_{t})}[\nabla\ell(Z;\widehat{\theta}_{t})\nabla\ell(Z;\widehat{\theta}_{t})^{\top}]\xrightarrow{P}0.`$ |  |

The continuity established in the proof of Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
then shows that the last two terms in
([4](#S4.E4 "Equation 4 ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"))
converge in probability to

|  |  |  |
|----|----|----|
|  |
``` math
\alpha^{2}\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}\bigl(\nabla\ell(Z;\theta_{t})\bigr).
``` |  |

Theorem
[4.2](#S4.Thmtheorem2 "Theorem 4.2. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives $`\widehat{\nabla S}_{t}\xrightarrow{P}\nabla S(\theta_{t})`$.
Since $`\widehat{V}_{0}=V_{0}=0`$, induction in the recursion
([4](#S4.E4 "Equation 4 ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"))
and the continuous mapping theorem give
$`\widehat{V}_{t}\xrightarrow{P}V_{t}`$ for each fixed $`t\geqslant 1`$.
If $`V_{t}`$ is nonsingular, then $`\widehat{V}_{t}`$ is nonsingular
with probability tending to one and
$`\widehat{V}_{t}^{-1}\xrightarrow{P}V_{t}^{-1}`$. Theorem
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Slutsky’s lemma give

|  |  |  |
|----|----|----|
|  |
``` math
n(\widehat{\theta}_{t}-\theta_{t})^{\top}\widehat{V}_{t}^{-1}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\chi_{d}^{2},
``` |  |

which proves the stated coverage. ∎

#### B.2.4 Proof of Corollary [4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Corollary [4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

We first verify the two claims about $`\rho_{\mathrm{RGD}}`$. Since
$`\alpha\leqslant 2/(\beta+\gamma)`$ and
$`(\beta+\gamma)^{2}\geqslant 4\beta\gamma`$,

|  |  |  |
|----|----|----|
|  |
``` math
\frac{\alpha\beta\gamma}{\beta+\gamma}\leqslant\frac{2\beta\gamma}{(\beta+\gamma)^{2}}\leqslant\frac{1}{2}.
``` |  |

It follows that $`\rho_{\mathrm{RGD}}\geqslant 1/2`$. The condition on
$`\varepsilon`$ makes the term in braces in the definition of
$`\rho_{\mathrm{RGD}}`$ positive, so $`\rho_{\mathrm{RGD}}<1`$. It also
implies $`\varepsilon<\gamma/\beta`$, and hence $`\theta_{\mathrm{PS}}`$
is well defined. By [Perdomo et al. \[2020, Theorem 3.8\]](#bib.bib12),

|  |  |  |
|----|----|----|
|  |
``` math
\|\theta_{t}-\theta_{\mathrm{PS}}\|_{2}\leqslant\rho_{\mathrm{RGD}}^{t}\|\theta_{0}-\theta_{\mathrm{PS}}\|_{2}\leqslant B\rho_{\mathrm{RGD}}^{t}.
``` |  |

Thus $`\theta_{t}\in\widehat{\mathcal{R}}_{t}`$ implies
$`\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}(0,B\rho_{\mathrm{RGD}}^{t})`$.
Corollary
[4.3](#S4.Thmtheorem3 "Corollary 4.3. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
therefore gives

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\liminf_{n\to\infty}\mathbb P\left\{\theta_{\mathrm{PS}}\in\widehat{\mathcal{R}}_{t}+\mathcal{B}(0,B\rho_{\mathrm{RGD}}^{t})\right\}\geqslant\lim_{n\to\infty}\mathbb P(\theta_{t}\in\widehat{\mathcal{R}}_{t})=1-\delta.`$ |  |

∎

### B.3 Details of Appendix [A](#A1 "Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"): Theory of Prediction-Powered Inference under Performativity

As in Section
[A](#A1 "Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\widehat{\theta}_{t}`$ and $`\widehat{\theta}_{t}(\lambda)`$ denote
the PPI iterates $`\widehat{\theta}_{t}^{\mathrm{PPI}}`$ and
$`\widehat{\theta}_{t}^{\mathrm{PPI}}(\lambda)`$. For
$`\check{\theta},\theta\in\Theta`$, let

|  |  |  |  |  |  |
|----|----|----|----|----|----|
|  | $`\displaystyle\mathcal{L}_{\check{\theta}}(\theta)`$ | $`\displaystyle:=\mathbb E_{Z\sim\mathcal{D}(\check{\theta})}[\ell(Z;\theta)],`$ | $`\displaystyle\mathcal{L}_{\check{\theta}}^{f}(\theta)`$ | $`\displaystyle:=\mathbb E_{Z\sim\mathcal{D}(\check{\theta})}[\ell(X,f(X);\theta)],`$ |  |
|  | $`\displaystyle\mathcal{L}_{\check{\theta},n}(\theta)`$ | $`\displaystyle:=\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\ell(X_{i},Y_{i};\theta),`$ | $`\displaystyle\mathcal{L}_{\check{\theta},n}^{f}(\theta)`$ | $`\displaystyle:=\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\ell(X_{i},f(X_{i});\theta),`$ |  |
|  | $`\displaystyle\widetilde{\mathcal{L}}_{\check{\theta},N}^{f}(\theta)`$ | $`\displaystyle:=\frac{1}{N}\sumop\displaylimits_{j=1}^{N}\ell(X_{j}^{u},f(X_{j}^{u});\theta),`$ | $`\displaystyle\widehat{\mathcal{L}}_{\check{\theta},n,N}(\theta;\lambda)`$ | $`\displaystyle:=\mathcal{L}_{\check{\theta},n}(\theta)+\lambda\{\widetilde{\mathcal{L}}_{\check{\theta},N}^{f}(\theta)-\mathcal{L}_{\check{\theta},n}^{f}(\theta)\}.`$ |  |

Here $`(X_{i},Y_{i})`$ are i.i.d. from $`\mathcal{D}(\check{\theta})`$,
and $`X_{j}^{u}`$ are independently drawn from its $`X`$-marginal.

###### Assumption B.19.

The following conditions hold.

\(a\) For some $`\gamma^{f}>0`$ and every $`x\in\mathcal{X}`$, the map
$`\theta\mapsto\ell(x,f(x);\theta)`$ is $`\gamma^{f}`$-strongly convex.

\(b\) For every $`(x,y)\in\mathcal{Z}`$, the map
$`\theta\mapsto\ell(x,y;\theta)-\ell(x,f(x);\theta)`$ is convex.

\(c\) For every compact $`K\subset\Theta`$, there exist measurable
$`A_{K}^{f},L_{K}^{f}:\mathcal{X}\to[0,\infty)`$ such that, for all
$`\theta,\theta^{\prime}\in K`$ and $`x\in\mathcal{X}`$,

|  |  |  |
|----|----|----|
|  |
``` math
|\ell(x,f(x);\theta)|\leqslant A_{K}^{f}(x),\qquad|\ell(x,f(x);\theta)-\ell(x,f(x);\theta^{\prime})|\leqslant L_{K}^{f}(x)\|\theta-\theta^{\prime}\|_{2},
``` |  |

and

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\widetilde{\theta}\in K}\mathbb E_{Z\sim\mathcal{D}(\widetilde{\theta})}[A_{K}^{f}(X)^{3}+L_{K}^{f}(X)^{3}]<\infty.
``` |  |

###### Remark B.20.

Assumption
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(a)–(b)
holds when
$`\ell(x,y;\theta)=\frac{1}{2}(y-\theta^{\top}x)^{2}+\frac{\gamma^{f}}{2}\|\theta\|_{2}^{2}`$.

For $`\theta\in\Theta`$, write $`\ell_{\theta}(Z):=\ell(Z;\theta)`$ and
$`\ell_{\theta}^{f}(Z):=\ell(X,f(X);\theta)`$. We write
$`\mathbb E_{\theta}^{u}`$ for expectation under the $`X`$-marginal of
$`\mathcal{D}(\theta)`$. Under Assumption
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
$`\ell_{\theta}(z)-\ell_{\theta}^{f}(z)`$ is convex in $`\theta`$ for
every $`z\in\mathcal{Z}`$. For a function
$`h:\mathcal{Z}\to\mathbb R^{q}`$ and $`\check{\theta}\in\Theta`$,
define the labeled and unlabeled empirical processes by

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle{\mathbb\displaystyle G}_{n,\check{\theta}}h:=\sqrt{n}\big(\frac{1}{n}\sumop\displaylimits_{j=1}^{n}h(Z_{j})-\mathbb{E}_{\check{\theta}}h(Z)\big),`$ |  |
|  |  | $`\displaystyle{\mathbb{\displaystyle G}}^{f}_{N,\check{\theta}}h:=\sqrt{N}\big(\frac{1}{N}\sumop\displaylimits_{j=1}^{N}h(X_{j}^{u},f(X_{j}^{u}))-\mathbb{E}_{\check{\theta}}^{u}h(X,f(X))\big),`$ |  |

where $`\{Z_{j}\}_{j=1}^{n}`$ are i.i.d. from
$`\mathcal{D}(\check{\theta})`$ and $`\{X_{j}^{u}\}_{j=1}^{N}`$ are
i.i.d. from $`\mathcal{D}_{X}(\check{\theta})`$.

###### Lemma B.21.

Under Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
for any compact subset $`K_{1},K_{2}\subset\Theta`$, constant
$`\delta>0`$, when $`n\to\infty`$ and $`\frac{n}{N}\to r`$ for some
$`r\geqslant 0`$, the following uniform LLN holds:

|  |  |  |
|----|----|----|
|  | $`\displaystyle\sup_{\check{\theta}\in K_{1}}P_{\check{\theta}}\big(\sup_{\lambda\in[0,1]}\sup_{\theta\in K_{2}}\big|\widehat{\mathcal{L}}_{\check{\theta},n,N}(\theta;\lambda)-\mathcal{L}_{\check{\theta}}(\theta)\big|>\delta\big)\to 0.`$ |  |

###### Proof of Lemma [B.21](#A2.Thmtheorem21 "Lemma B.21. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Since
$`\widehat{\mathcal{L}}_{\check{\theta},n,N}(\theta;\lambda)=\mathcal{L}_{\check{\theta},n}(\theta)+\lambda\big(\widetilde{\mathcal{L}}_{\check{\theta},N}^{f}(\theta)-\mathcal{L}_{\check{\theta},n}^{f}(\theta)\big)`$,
we have

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\mathcal{L}}_{\check{\theta},n,N}(\theta;\lambda)-\mathcal{L}_{\check{\theta}}(\theta)=\mathcal{L}_{\check{\theta},n}(\theta)-\mathcal{L}_{\check{\theta}}(\theta)+\lambda\big(\widetilde{\mathcal{L}}_{\check{\theta},N}^{f}(\theta)-\mathcal{L}_{\check{\theta}}^{f}(\theta)\big)-\lambda\big(\mathcal{L}_{\check{\theta},n}^{f}(\theta)-\mathcal{L}_{\check{\theta}}^{f}(\theta)\big).
``` |  |

By the proof of Lemma
[B.1](#A2.Thmtheorem1 "Lemma B.1. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the three differences converge uniformly to $`0`$ over
$`\lambda\in[0,1]`$ as $`n,N\to\infty`$. ∎

Write $`\widehat{\mathcal{L}}_{\check{\theta}}(\theta;\lambda)`$ for
$`\widehat{\mathcal{L}}_{\check{\theta},n,N}(\theta;\lambda)`$. The
population objective remains $`\mathcal{L}_{\check{\theta}}`$, with
minimizer $`G(\check{\theta})`$.

###### Proposition B.22 (Consistency of the PPI iterates).

Suppose Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
hold, $`n/N\to r\in[0,\infty)`$, and $`\widehat{\lambda}_{s}\in[0,1]`$
for all $`s`$. Then, for each fixed $`t\geqslant 1`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\lambda\in[0,1]}\|\widehat{\theta}_{t}(\lambda)-\theta_{t}\|_{2}\xrightarrow{P}0.
``` |  |

In particular, $`\widehat{\theta}_{t}\xrightarrow{P}\theta_{t}`$.

###### Proof of Proposition [B.22](#A2.Thmtheorem22 "Proposition B.22 (Consistency of the PPI iterates). ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

For $`\check{\theta}\in\Theta`$ and $`\lambda\in[0,1]`$, let

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{G}_{\lambda}(\check{\theta}):=\argmin_{\theta\in\Theta}\widehat{\mathcal{L}}_{\check{\theta}}(\theta;\lambda).
``` |  |

We prove the result by induction, starting from
$`\widehat{\theta}_{0}=\theta_{0}`$.

*Step 1: uniform localization.* Fix a compact set
$`K_{1}\subset\Theta`$. Since

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\mathcal{L}}_{\check{\theta}}(\theta;\lambda)=(1-\lambda)\mathcal{L}_{\check{\theta},n}(\theta)+\lambda\widehat{\mathbb E}_{\check{\theta},n}[\ell_{\theta}-\ell_{\theta}^{f}]+\lambda\widetilde{\mathcal{L}}_{\check{\theta},N}^{f}(\theta),
``` |  |

Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
imply that this objective is $`\min\{\gamma,\gamma^{f}\}`$-strongly
convex. Hence

|  |  |  |
|----|----|----|
|  |
``` math
\min\{\gamma,\gamma^{f}\}\|\widehat{G}_{\lambda}(\check{\theta})-G(\check{\theta})\|_{2}\leqslant\|\nabla_{\theta}\widehat{\mathcal{L}}_{\check{\theta}}(G(\check{\theta});\lambda)\|_{2}.
``` |  |

The Lipschitz continuity of $`G`$ gives a compact set $`K`$ containing
$`K_{1}`$ and $`G(K_{1})`$. Uniformly over $`\lambda\in[0,1]`$, the
expression on the right is bounded by the sum of

|  |  |  |
|----|----|----|
|  |
``` math
\|\nabla\mathcal{L}_{\check{\theta},n}(G(\check{\theta}))\|_{2},\quad\|\widehat{\mathbb E}_{\check{\theta},n}[\nabla\ell_{G(\check{\theta})}-\nabla\ell_{G(\check{\theta})}^{f}]\|_{2},\quad\|\nabla\widetilde{\mathcal{L}}_{\check{\theta},N}^{f}(G(\check{\theta}))\|_{2}.
``` |  |

Their second moments are bounded uniformly over
$`\check{\theta}\in K_{1}`$ by the moment conditions on $`L_{K}`$ and
$`L_{K}^{f}`$. Therefore

|  |  |  |
|----|----|----|
|  |
``` math
\lim_{R\to\infty}\limsup_{n,N\to\infty}\sup_{\check{\theta}\in K_{1}}P_{\check{\theta}}\left\{\sup_{\lambda\in[0,1]}\|\widehat{G}_{\lambda}(\check{\theta})-G(\check{\theta})\|_{2}>R\right\}=0.
``` |  |

*Step 2: uniform consistency.* Suppose
$`\widehat{\theta}_{t}\xrightarrow{P}\theta_{t}`$, which holds for
$`t=0`$. Choose compact sets $`K_{1},K_{2}\subset\Theta`$ so that, with
probability tending to one, $`\widehat{\theta}_{t}\in K_{1}`$ and
$`\widehat{G}_{\lambda}(\widehat{\theta}_{t})\in K_{2}`$ for every
$`\lambda\in[0,1]`$; Step 1 permits this choice. For
$`\check{\theta}\in K_{1}`$, the minimizing property of
$`\widehat{G}_{\lambda}(\check{\theta})`$ and the $`\gamma`$-strong
convexity of $`\mathcal{L}_{\check{\theta}}`$ give

|  |  |  |
|----|----|----|
|  |
``` math
\|\widehat{G}_{\lambda}(\check{\theta})-G(\check{\theta})\|_{2}^{2}\leqslant\frac{4}{\gamma}\sup_{\theta\in K_{2}}|\widehat{\mathcal{L}}_{\check{\theta}}(\theta;\lambda)-\mathcal{L}_{\check{\theta}}(\theta)|.
``` |  |

Lemma
[B.21](#A2.Thmtheorem21 "Lemma B.21. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
uniformly in $`\lambda`$, therefore yields

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\lambda\in[0,1]}\|\widehat{G}_{\lambda}(\widehat{\theta}_{t})-G(\widehat{\theta}_{t})\|_{2}\xrightarrow{P}0.
``` |  |

Since $`G`$ is continuous,

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\sup_{\lambda\in[0,1]}\|\widehat{\theta}_{t+1}(\lambda)-\theta_{t+1}\|_{2}`$ | $`\displaystyle\leqslant\sup_{\lambda\in[0,1]}\|\widehat{G}_{\lambda}(\widehat{\theta}_{t})-G(\widehat{\theta}_{t})\|_{2}+\|G(\widehat{\theta}_{t})-G(\theta_{t})\|_{2}\xrightarrow{P}0.`$ |  |

This completes the induction. Taking $`\lambda=\widehat{\lambda}_{t+1}`$
gives the final assertion. ∎

###### Lemma B.23.

Under Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
let $`\{\check{\theta}_{n}\}`$ and $`\{\theta_{n}\}`$ be random vector
sequences in $`\Theta`$ such that

|  |  |  |
|----|----|----|
|  |
``` math
\check{\theta}_{n}\xrightarrow{P}\check{\theta}_{0},\qquad\theta_{n}\xrightarrow{P}\theta_{0}.
``` |  |

Conditional on $`(\check{\theta}_{n},\theta_{n})`$, suppose that the
labeled observations $`Z_{1,n},\ldots,Z_{n,n}`$ are i.i.d. from
$`\mathcal{D}(\check{\theta}_{n})`$, and the unlabeled covariates
$`X^{u}_{1,n},\ldots,X^{u}_{N,n}`$ are i.i.d. from its $`X`$-marginal
and independent of the labeled observations. If $`h_{n}=O_{P}^{*}(1)`$,
$`n,N\to\infty`$, and $`n/N\to r\in[0,\infty)`$, then

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\mathbb\displaystyle G_{n,\check{\theta}_{n}}\left[\sqrt{n}\left(\ell_{\theta_{n}+h_{n}/\sqrt{n}}-\ell_{\theta_{n}}\right)-h_{n}^{\top}\nabla\ell_{\theta_{n}}\right]=o_{P}^{*}(1),`$ |  |
|  |  | $`\displaystyle\mathbb\displaystyle G_{n,\check{\theta}_{n}}\left[\sqrt{n}\left(\ell^{f}_{\theta_{n}+h_{n}/\sqrt{n}}-\ell^{f}_{\theta_{n}}\right)-h_{n}^{\top}\nabla\ell^{f}_{\theta_{n}}\right]=o_{P}^{*}(1),`$ |  |
|  |  | $`\displaystyle\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}_{n}}\left[\sqrt{n}\left(\ell^{f}_{\theta_{n}+h_{n}/\sqrt{n}}-\ell^{f}_{\theta_{n}}\right)-h_{n}^{\top}\nabla\ell^{f}_{\theta_{n}}\right]=o_{P}^{*}(1).`$ |  |

Here the empirical processes are centered at the expectations under
$`\mathcal{D}(\check{\theta}_{n})`$ and its $`X`$-marginal,
respectively.

###### Proof of Lemma [B.23](#A2.Thmtheorem23 "Lemma B.23. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

The first convergence follows from Lemma
[B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
Its proof, applied to $`z=(x,f(x))`$, gives the second convergence. We
prove the third using the same argument.

For $`M>0`$, let

|     |                                                   |     |
|-----|---------------------------------------------------|-----|
|     |
       ``` math
       B_{M}=\{h\in\mathbb R^{d}:\|h\|_{2}\leqslant M\}.
       ```                                                |     |

For fixed $`\theta,\check{\theta}\in\Theta`$ and nonrandom
$`h\in B_{M}`$, define

|  |  |  |
|----|----|----|
|  |
``` math
r^{f}_{n,\theta,h}(x):=\sqrt{n}\left\{\ell(x,f(x);\theta+h/\sqrt{n})-\ell(x,f(x);\theta)\right\}-h^{\top}\nabla_{\theta}\ell(x,f(x);\theta).
``` |  |

By Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c),

|  |  |  |
|----|----|----|
|  |
``` math
|r^{f}_{n,\theta,h}(x)|\leqslant\frac{\beta}{\sqrt{n}}\|h\|_{2}^{2}.
``` |  |

The centered empirical process has mean zero, and

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\operatorname{Var}_{\check{\theta}}\left(\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}}r^{f}_{n,\theta,h}\right)`$ | $`\displaystyle=\frac{n}{N}\,\operatorname{Var}_{\check{\theta}}\left(r^{f}_{n,\theta,h}(X)\right)\leqslant\frac{\beta^{2}}{N}\|h\|_{2}^{4}.`$ |  |

Thus, for each fixed $`h`$, the process converges to zero in probability
uniformly over $`\theta,\check{\theta}`$.

To make this convergence uniform over $`B_{M}`$, use the integral
representation

|  |  |  |
|----|----|----|
|  |
``` math
r^{f}_{n,\theta,h}(x)=\intop\nolimits_{0}^{1}h^{\top}\left\{\nabla_{\theta}\ell(x,f(x);\theta+sh/\sqrt{n})-\nabla_{\theta}\ell(x,f(x);\theta)\right\}\,ds.
``` |  |

As in the proof of Lemma
[B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c)
gives, for $`h,h^{\prime}\in B_{M}`$,

|  |  |  |
|----|----|----|
|  |
``` math
|r^{f}_{n,\theta,h}(x)-r^{f}_{n,\theta,h^{\prime}}(x)|\leqslant\frac{2M\beta}{\sqrt{n}}\|h-h^{\prime}\|_{2}.
``` |  |

Consequently,

|  |  |  |
|----|----|----|
|  |
``` math
\left|\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}}\left(r^{f}_{n,\theta,h}-r^{f}_{n,\theta,h^{\prime}}\right)\right|\leqslant 4M\beta\|h-h^{\prime}\|_{2}.
``` |  |

Fix $`\varepsilon>0`$, set $`\delta=\varepsilon/(8M\beta)`$, and let
$`\{h_{1},\ldots,h_{J}\}`$ be a finite $`\delta`$-net of $`B_{M}`$. The
net cardinality $`J`$ is independent of $`n,N,\theta,\check{\theta}`$.
The preceding inequality, a union bound, and the variance bound yield

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\sup_{\theta,\check{\theta}\in\Theta}P_{\check{\theta}}\left(\sup_{h\in B_{M}}\left|\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}}r^{f}_{n,\theta,h}\right|>\varepsilon\right)\leqslant\sup_{\theta,\check{\theta}\in\Theta}P_{\check{\theta}}\left(\max_{1\leqslant j\leqslant J}\left|\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}}r^{f}_{n,\theta,h_{j}}\right|>\frac{\varepsilon}{2}\right)\leqslant\frac{4J\beta^{2}M^{4}}{N\varepsilon^{2}}.`$ |  |

The supremum over $`B_{M}`$ is measurable: by the displayed Lipschitz
bound, it equals the supremum over a countable dense subset.

By the conditional sampling assumption, the same bound applies almost
surely after conditioning on $`(\check{\theta}_{n},\theta_{n})`$:

|  |  |  |
|----|----|----|
|  |
``` math
P\left(\sup_{h\in B_{M}}\left|\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}_{n}}r^{f}_{n,\theta_{n},h}\right|>\varepsilon\,\middle|\,\check{\theta}_{n},\theta_{n}\right)\leqslant\frac{4J\beta^{2}M^{4}}{N\varepsilon^{2}}.
``` |  |

Averaging this inequality and localizing $`h_{n}`$ gives

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle P^{*}\left(\left|\sqrt{\frac{n}{N}}\,\mathbb G^{f}_{N,\check{\theta}_{n}}r^{f}_{n,\theta_{n},h_{n}}\right|>\varepsilon\right)\quad\leqslant P^{*}(\|h_{n}\|_{2}>M)+\frac{4J\beta^{2}M^{4}}{N\varepsilon^{2}}.`$ |  |

For any $`a>0`$, tightness of $`h_{n}`$ permits choosing $`M`$ such that

|     |                                               |     |
|-----|-----------------------------------------------|-----|
|     |
       ``` math
       \limsup_{n}P^{*}(\|h_{n}\|_{2}>M)\leqslant a.
       ```                                            |     |

For this fixed $`M`$, letting $`n,N\to\infty`$ bounds the limiting upper
probability by $`a`$. Letting $`a\downarrow 0`$ proves the third
convergence. ∎

###### Lemma B.24.

Suppose
$`\sqrt{n}(\check{\theta}_{n}-\check{\theta}_{0})\xrightarrow{d}\mathcal{N}(0,A_{1})`$
for a positive semidefinite matrix $`A_{1}\in\mathbb R^{d\times d}`$,
and $`\lambda_{n}\xrightarrow{P}\lambda_{0}`$. Conditional on
$`\check{\theta}_{n}`$, the labeled observations are i.i.d. from
$`\mathcal{D}(\check{\theta}_{n})`$, and the unlabeled covariates are
i.i.d. from its $`X`$-marginal and independent of the labeled
observations. For $`h_{\theta}:\mathcal{Z}\to\mathbb R^{q}`$, define

|  |  |  |  |  |  |
|----|----|----|----|----|----|
|  | $`\displaystyle F_{h}(\theta)`$ | $`\displaystyle:=\mathbb E_{\theta}[h_{\theta}(Z)],`$ | $`\displaystyle h_{\theta}^{f}(Z)`$ | $`\displaystyle:=h_{\theta}(X,f(X)),`$ |  |
|  | $`\displaystyle F_{h}^{f}(\theta)`$ | $`\displaystyle:=\mathbb E_{\theta}[h_{\theta}^{f}(Z)],`$ | $`\displaystyle V_{\lambda,h}(\theta)`$ | $`\displaystyle:=\operatorname{Cov}_{\theta}\{h_{\theta}(Z)-\lambda h_{\theta}^{f}(Z)\},`$ |  |
|  | $`\displaystyle V_{\lambda,h}^{f}(\theta)`$ | $`\displaystyle:=\lambda^{2}\operatorname{Cov}_{\theta}\{h_{\theta}^{f}(Z)\}.`$ |  |  |  |

Suppose, for every compact $`K\subset\Theta`$, that $`F_{h}`$ is bounded
and continuously differentiable on an open neighborhood of $`K`$,
$`V_{\lambda,h}`$ and $`V_{\lambda,h}^{f}`$ are continuous on $`K`$, and

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}\mathbb E_{\theta}[\|h_{\theta}(Z)\|_{2}^{3}+\|h_{\theta}^{f}(Z)\|_{2}^{3}]<\infty.
``` |  |

Then, as $`n,N\to\infty`$ with $`n/N\to r\geqslant 0`$,

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle\mathbb\displaystyle G_{n,\check{\theta}_{n}}[h_{\check{\theta}_{n}}-\lambda_{n}h_{\check{\theta}_{n}}^{f}]+\lambda_{n}\sqrt{\frac{n}{N}}\mathbb G^{f}_{N,\check{\theta}_{n}}h_{\check{\theta}_{n}}+\sqrt{n}\{F_{h}(\check{\theta}_{n})-F_{h}(\check{\theta}_{0})\}`$ |  |
|  |  | $`\displaystyle\xrightarrow{d}\mathcal{N}\left(0,V_{\lambda_{0},h}(\check{\theta}_{0})+rV_{\lambda_{0},h}^{f}(\check{\theta}_{0})+\nabla F_{h}(\check{\theta}_{0})A_{1}\nabla F_{h}(\check{\theta}_{0})^{\top}\right).`$ |  |

###### Proof of Lemma [B.24](#A2.Thmtheorem24 "Lemma B.24. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Set $`g_{\theta}=h_{\theta}-\lambda_{0}h_{\theta}^{f}`$ and
$`F_{g}(\theta)=\mathbb E_{\theta}[g_{\theta}(Z)]`$. For fixed
$`s\in\mathbb R^{q}`$, define

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\chi(s,\theta)`$ | $`\displaystyle:=\exp\{-\tfrac{1}{2}s^{\top}V_{\lambda_{0},h}(\theta)s\},`$ |  |
|  | $`\displaystyle\chi_{n}(s,\theta)`$ | $`\displaystyle:=\left[\mathbb E_{\theta}\exp\left\{\frac{is^{\top}}{\sqrt{n}}(g_{\theta}(Z)-F_{g}(\theta))\right\}\right]^{n},`$ |  |
|  | $`\displaystyle\chi^{f}(s,\theta)`$ | $`\displaystyle:=\exp\{-\tfrac{r}{2}s^{\top}V_{\lambda_{0},h}^{f}(\theta)s\},`$ |  |
|  | $`\displaystyle\chi_{n,N}^{f}(s,\theta)`$ | $`\displaystyle:=\left[\mathbb E_{\theta}^{u}\exp\left\{\frac{i\sqrt{n}\lambda_{0}s^{\top}}{N}(h_{\theta}^{f}(Z)-F_{h}^{f}(\theta))\right\}\right]^{N}.`$ |  |

The third-moment bound and the continuity of the two covariances give,
uniformly on compact sets, $`\chi_{n}(s,\theta)-\chi(s,\theta)\to 0`$
and $`\chi_{n,N}^{f}(s,\theta)-\chi^{f}(s,\theta)\to 0`$. Since
$`\check{\theta}_{n}\xrightarrow{P}\check{\theta}_{0}`$, Lemma
[B.4](#A2.Thmtheorem4 "Lemma B.4. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and the continuity of $`\chi`$ and $`\chi^{f}`$ yield

|  |  |  |
|----|----|----|
|  |
``` math
\chi_{n}(s,\check{\theta}_{n})\xrightarrow{L^{1}}\chi(s,\check{\theta}_{0}),\qquad\chi_{n,N}^{f}(s,\check{\theta}_{n})\xrightarrow{L^{1}}\chi^{f}(s,\check{\theta}_{0}).
``` |  |

Let $`\xi_{n}`$ denote the random vector on the left side of the
conclusion, and let $`\xi_{n}^{0}`$ be the same vector with
$`\lambda_{n}`$ replaced by $`\lambda_{0}`$. Conditional independence of
the two samples gives

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbb E[e^{is^{\top}\xi_{n}^{0}}]=\mathbb E\big[`$ | $`\displaystyle\exp\{i\sqrt{n}s^{\top}(F_{h}(\check{\theta}_{n})-F_{h}(\check{\theta}_{0}))\}\times\chi_{n}(s,\check{\theta}_{n})\chi_{n,N}^{f}(s,\check{\theta}_{n})\big].`$ |  |

The delta method and the preceding $`L^{1}`$ convergences show that this
characteristic function converges to

|  |  |  |
|----|----|----|
|  |
``` math
\exp\left[-\frac{1}{2}s^{\top}\left\{V_{\lambda_{0},h}(\check{\theta}_{0})+rV_{\lambda_{0},h}^{f}(\check{\theta}_{0})+\nabla F_{h}(\check{\theta}_{0})A_{1}\nabla F_{h}(\check{\theta}_{0})^{\top}\right\}s\right].
``` |  |

It remains to replace $`\lambda_{0}`$ by $`\lambda_{n}`$. Their
difference is $`(\lambda_{0}-\lambda_{n})\Delta_{n}`$, where

|  |  |  |
|----|----|----|
|  |
``` math
\Delta_{n}:=\mathbb G_{n,\check{\theta}_{n}}h_{\check{\theta}_{n}}^{f}-\sqrt{\frac{n}{N}}\mathbb G^{f}_{N,\check{\theta}_{n}}h_{\check{\theta}_{n}}.
``` |  |

For every compact $`K\subset\Theta`$, the third-moment condition gives

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\theta\in K}\mathbb E_{\theta}[\|h_{\theta}^{f}(Z)\|_{2}^{2}]<\infty,
``` |  |

and the same bound holds under the $`X`$-marginal. Since
$`\check{\theta}_{n}`$ lies in such a compact set with probability
tending to one, the conditional second moments of both empirical
processes are bounded, and hence $`\Delta_{n}=O_{p}(1)`$. Therefore
$`(\lambda_{0}-\lambda_{n})\Delta_{n}=o_{p}(1)`$. The characteristic
functions of $`\xi_{n}`$ and $`\xi_{n}^{0}`$ have the same limit, which
proves the result. ∎

#### B.3.1 Proof of Theorem [A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Theorem [A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

We prove the result by induction on $`t\geqslant 0`$. For $`t=0`$,
$`\widehat{\theta}_{0}=\theta_{0}`$ and $`V_{0}=0`$. Suppose the
conclusion holds at round $`t`$. Conditional on all preceding
observations and estimators, $`Z_{t,1},\ldots,Z_{t,n}`$ are i.i.d. from
$`\mathcal{D}(\widehat{\theta}_{t})`$, and
$`X^{u}_{t,1},\ldots,X^{u}_{t,N}`$ are independently drawn from its
$`X`$-marginal. Below, $`\lambda=\widehat{\lambda}_{t+1}`$.

*Step 1: localization.* We first show

|  |  |  |
|----|----|----|
|  |
``` math
\sqrt{n}\{\widehat{\theta}_{t+1}-G(\widehat{\theta}_{t})\}=O_{p}(1).
``` |  |

The localization argument in Step 1 of the proof of Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
applies with Proposition
[B.22](#A2.Thmtheorem22 "Proposition B.22 (Consistency of the PPI iterates). ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
in place of Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
The centered PPI objective is the sum of a labeled and an unlabeled
empirical process. On
$`\{\theta:\|\theta-G(\widehat{\theta}_{t})\|_{2}\leqslant\delta\}`$,
their increments have envelopes $`\delta\{L_{K}(z)+L_{K}^{f}(x)\}`$ and
$`\delta L_{K}^{f}(x)`$, respectively. The same bracketing bound
therefore applies to both processes, uniformly over $`\lambda\in[0,1]`$;
moreover, $`\sqrt{n/N}`$ is bounded.

*Step 2: the leading term.* Let

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\widetilde{h}:={}`$ | $`\displaystyle-H_{\theta_{t}}(\theta_{t+1})^{-1}\sqrt{n}\Bigg[\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\Big\{\nabla\ell(Z_{t,i};\theta_{t+1})-\lambda\nabla\ell(X_{t,i},f(X_{t,i});\theta_{t+1})\Big\}`$ |  |
|  |  | $`\displaystyle+\frac{\lambda}{N}\sumop\displaylimits_{j=1}^{N}\nabla\ell(X^{u}_{t,j},f(X^{u}_{t,j});\theta_{t+1})\Bigg].`$ |  |

The expression in brackets is the gradient of the PPI objective at
$`\theta_{t+1}`$. Since
$`\mathbb E_{\theta_{t}}[\nabla\ell(Z;\theta_{t+1})]=0`$ and the labeled
and unlabeled prediction terms have the same conditional mean, its
scaled value is the random vector in Lemma
[B.24](#A2.Thmtheorem24 "Lemma B.24. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
with

|  |  |  |
|----|----|----|
|  |
``` math
h_{\vartheta}(z)=\nabla\ell(z;\theta_{t+1}),\quad\check{\theta}_{n}=\widehat{\theta}_{t},\quad\check{\theta}_{0}=\theta_{t},\quad A_{1}=V_{t},\quad\lambda_{n}=\widehat{\lambda}_{t+1},\quad\lambda_{0}=\lambda_{t+1}.
``` |  |

We verify the conditions of the lemma. The map
$`F_{h}(\vartheta)=\mathbb E_{Z\sim\mathcal{D}(\vartheta)}[\nabla\ell(Z;\theta_{t+1})]`$
is continuously differentiable by Assumption
[2.4](#S2.Thmtheorem4 "Assumption 2.4 (Differentiability of the distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b),
and the proof of Proposition
[3.1](#S3.Thmtheorem1 "Proposition 3.1. ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives

|  |  |  |
|----|----|----|
|  |
``` math
\nabla F_{h}(\theta_{t})=-H_{\theta_{t}}(\theta_{t+1})\nabla G(\theta_{t}).
``` |  |

Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives continuity of the labeled covariances. Integrating its
density-ratio bound over $`y`$ gives the same bound for the
$`X`$-marginal densities and hence continuity of the unlabeled
covariance. The third-moment condition follows from Assumptions
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(b)
and
[B.19](#A2.Thmtheorem19 "Assumption B.19. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c).
Lemma
[B.24](#A2.Thmtheorem24 "Lemma B.24. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and the definition of
$`\Sigma_{\theta_{t}}(\theta_{t+1};\lambda_{t+1})`$ now give

|     |                                                     |     |
|-----|-----------------------------------------------------|-----|
|     |
       ``` math
       \widetilde{h}\xrightarrow{d}\mathcal{N}(0,V_{t+1}).
       ```                                                  |     |

*Step 3: expansion.* Let
$`\widehat{h}=\sqrt{n}(\widehat{\theta}_{t+1}-\theta_{t+1})`$. Step 1,
the induction hypothesis, and the delta method applied to $`G`$ imply
that $`\widehat{h}=O_{p}(1)`$. For $`h=O_{p}(1)`$, consider $`n`$ times
the increase of the PPI objective from $`\theta_{t+1}`$ to
$`\theta_{t+1}+h/\sqrt{n}`$. The two prediction terms have the same
conditional mean, so only the labeled loss contributes to the population
part. The center-independent form of Lemma
[B.23](#A2.Thmtheorem23 "Lemma B.23. ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
controls the three empirical remainders, and Lemma
[B.8](#A2.Thmtheorem8 "Lemma B.8. ‣ B.1.4 Proof of CLT for RRM (Theorem ) ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
replaces the current distribution by $`\mathcal{D}(\theta_{t})`$ in the
remaining second-order term. Uniformly for $`h=O_{p}(1)`$, the increase
is therefore

|  |  |  |  |
|----|----|----|----|
|  |  | $`\displaystyle-h^{\top}H_{\theta_{t}}(\theta_{t+1})\widetilde{h}+n\,\mathbb E_{Z\sim\mathcal{D}(\theta_{t})}\left[\ell(Z;\theta_{t+1}+h/\sqrt{n})-\ell(Z;\theta_{t+1})\right]+o_{p}(1)`$ |  |
|  |  | $`\displaystyle=-h^{\top}H_{\theta_{t}}(\theta_{t+1})\widetilde{h}+\frac{1}{2}h^{\top}H_{\theta_{t}}(\theta_{t+1})h+o_{p}(1).`$ |  |

The second equality uses
$`\mathbb E_{\theta_{t}}[\nabla\ell(Z;\theta_{t+1})]=0`$ and Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c).
Evaluating this expansion at $`\widehat{h}`$ and $`\widetilde{h}`$, the
minimizing property of $`\widehat{\theta}_{t+1}`$ gives

|  |  |  |
|----|----|----|
|  |
``` math
\frac{1}{2}(\widehat{h}-\widetilde{h})^{\top}H_{\theta_{t}}(\theta_{t+1})(\widehat{h}-\widetilde{h})\leqslant o_{p}(1).
``` |  |

Since $`H_{\theta_{t}}(\theta_{t+1})\succeq\gamma I_{d}`$,
$`\widehat{h}=\widetilde{h}+o_{p}(1)`$. Step 2 and Slutsky’s lemma prove
the claim at round $`t+1`$. ∎

#### B.3.2 Proof of Corollary [A.2](#A1.Thmtheorem2 "Corollary A.2. ‣ A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Corollary [A.2](#A1.Thmtheorem2 "Corollary A.2. ‣ A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

Fix $`t\geqslant 0`$. Proposition
[B.22](#A2.Thmtheorem22 "Proposition B.22 (Consistency of the PPI iterates). ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\theta}_{t}\xrightarrow{P}\theta_{t},\qquad\widehat{\theta}_{t+1}(\widetilde{\lambda}_{t+1})\xrightarrow{P}\theta_{t+1},\qquad\widehat{\theta}_{t+1}\xrightarrow{P}\theta_{t+1}.
``` |  |

*Step 1: plug-in covariance.* We first show that, when $`\theta`$ is
either $`\widehat{\theta}_{t+1}(\widetilde{\lambda}_{t+1})`$ or
$`\widehat{\theta}_{t+1}`$,

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{\lambda\in[0,1]}\left\|\widehat{\Sigma}_{\widehat{\theta}_{t}}(\theta;\lambda)-\Sigma_{\theta_{t}}(\theta_{t+1};\lambda)\right\|_{2}\xrightarrow{P}0.
``` |  |

Both matrices are polynomials of degree two in $`\lambda`$. Their
coefficients are products of the inverse Hessian with covariances and
cross-covariances of $`\nabla\ell(Z;\theta)`$ and
$`\nabla\ell(X,f(X);\theta)`$. The proof of Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
gives convergence of the Hessian and each coefficient involving
$`\nabla\ell(Z;\theta)`$. The same argument applies to the remaining
coefficients with $`L_{K}(z)`$ replaced by $`L_{K}(z)+L_{K}^{f}(x)`$,
using Assumption
[2.3](#S2.Thmtheorem3 "Assumption 2.3 (Loss and distribution map). ‣ 2.3 Assumptions and Notation ‣ 2 Background and Setup ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")(c)
also at $`z=(x,f(x))`$. Integrating the density-ratio bound in Lemma
[B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.1.3 Preparation Lemmas ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
over $`y`$ gives the corresponding bound for the $`X`$-marginal, so the
argument also applies to the $`n+N`$ covariates. Since $`n/N\to r`$ and
$`\lambda\in[0,1]`$, convergence of the finitely many coefficients gives
the displayed uniform convergence.

*Step 2: the weights.* Since
$`\operatorname{Tr}(A^{\top})=\operatorname{Tr}(A)`$,

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\operatorname{Tr}\Sigma_{\theta_{t}}(\theta_{t+1};\lambda)={}`$ | $`\displaystyle\operatorname{Tr}\Sigma_{\theta_{t}}(\theta_{t+1})`$ |  |
|  |  | $`\displaystyle-2\lambda\operatorname{Tr}\Bigl\{H_{\theta_{t}}(\theta_{t+1})^{-1}\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}\bigl(\nabla\ell(Z;\theta_{t+1}),\nabla\ell(X,f(X);\theta_{t+1})\bigr)H_{\theta_{t}}(\theta_{t+1})^{-1}\Bigr\}`$ |  |
|  |  | $`\displaystyle+(1+r)\lambda^{2}\operatorname{Tr}\Bigl\{H_{\theta_{t}}(\theta_{t+1})^{-1}\operatorname{Cov}_{Z\sim\mathcal{D}(\theta_{t})}\bigl(\nabla\ell(X,f(X);\theta_{t+1})\bigr)H_{\theta_{t}}(\theta_{t+1})^{-1}\Bigr\}.`$ |  |

The coefficient of $`\lambda^{2}`$ is positive by the condition in the
corollary and the positive definiteness of
$`H_{\theta_{t}}(\theta_{t+1})`$. Thus $`\lambda_{t+1}^{*}`$ is the
ratio of the trace multiplying $`-2\lambda`$ to the coefficient of
$`\lambda^{2}`$, truncated to $`[0,1]`$. The sample criterion has the
same quadratic form, with the covariances specified in Section
[A.3](#A1.SS3 "A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and $`r`$ replaced by $`n/N`$. Step 1, evaluated at
$`\widehat{\theta}_{t+1}(\widetilde{\lambda}_{t+1})`$, shows that its
coefficients converge to the population coefficients. With probability
tending to one, its quadratic coefficient is positive. Continuity of the
truncated ratio then gives

|     |                                                          |     |
|-----|----------------------------------------------------------|-----|
|     |
       ``` math
       \widehat{\lambda}_{t+1}\xrightarrow{P}\lambda_{t+1}^{*}.
       ```                                                       |     |

The same argument holds at every fixed round. Therefore Theorem
[A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
applies with $`\lambda_{s}=\lambda_{s}^{*}`$ at each fixed $`s`$.

*Step 3: covariance and coverage.* Theorems
[B.15](#A2.Thmtheorem15 "Theorem B.15. ‣ B.1.9 Proof of Theorem ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[3.4](#S3.Thmtheorem4 "Theorem 3.4. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
continue to hold along the PPI trajectory: their proofs use Theorem
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Proposition
[B.3](#A2.Thmtheorem3 "Proposition B.3. ‣ B.1.2 Consistency of the RERM Trajectory ‣ B.1 Details of Section : Theory of Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
only through the $`\sqrt{n}`$-consistency and consistency of the
iterates, which Theorem
[A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Proposition
[B.22](#A2.Thmtheorem22 "Proposition B.22 (Consistency of the PPI iterates). ‣ B.3 Details of Appendix : Theory of Prediction-Powered Inference under Performativity ‣ Appendix B Theoretical Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
provide. Hence
$`\widehat{\nabla G}_{t}\xrightarrow{P}\nabla G(\theta_{t})`$. Steps 1
and 2 also give

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\Sigma}_{\widehat{\theta}_{t}}(\widehat{\theta}_{t+1};\widehat{\lambda}_{t+1})\xrightarrow{P}\Sigma_{\theta_{t}}(\theta_{t+1};\lambda_{t+1}^{*}).
``` |  |

Since $`\widehat{V}_{0}=V_{0}=0`$, induction in
([5](#A1.E5 "Equation 5 ‣ A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"))
and the continuous mapping theorem yield
$`\widehat{V}_{t}\xrightarrow{P}V_{t}`$. If $`V_{t}`$ is nonsingular,
then $`\widehat{V}_{t}`$ is positive definite with probability tending
to one and $`\widehat{V}_{t}^{-1}\xrightarrow{P}V_{t}^{-1}`$. Theorem
[A.1](#A1.Thmtheorem1 "Theorem A.1 (Central limit theorem for PPI). ‣ A.2 Central Limit Theorem of PPI ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and Slutsky’s lemma give

|  |  |  |
|----|----|----|
|  |
``` math
n(\widehat{\theta}_{t}-\theta_{t})^{\top}\widehat{V}_{t}^{-1}(\widehat{\theta}_{t}-\theta_{t})\xrightarrow{d}\chi_{d}^{2},
``` |  |

which proves the stated coverage. ∎

#### B.3.3 Proof of Corollary [A.3](#A1.Thmtheorem3 "Corollary A.3 (Confidence region for 𝜃_PS). ‣ A.4 Confidence Regions for the Performative Stable Point ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")

###### Proof of Corollary [A.3](#A1.Thmtheorem3 "Corollary A.3 (Confidence region for 𝜃_PS). ‣ A.4 Confidence Regions for the Performative Stable Point ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

The population trajectory is the RRM trajectory. Hence the proof of
Corollary
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
applies with Corollary
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
replaced by Corollary
[A.2](#A1.Thmtheorem2 "Corollary A.2. ‣ A.3 Choice of the Weights and Covariance Estimation ‣ Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
∎

## Appendix C Experimental Details

In this section, we provide implementation details for the experiments
in Section
[5](#S5 "5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

### C.1 Pure-Synthetic Experiment

#### C.1.1 Data Generation

We use the mixture distribution and conditional outcome model described
in Section
[5.1](#S5.SS1 "5.1 Controlled Synthetic Studies ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
that is, when deploying $`\theta`$:

|  |  |  |
|----|----|----|
|  |
``` math
X\sim(1-a(\theta))\,\mathrm{Unif}(0,1)+a(\theta)\,B(5,2),\quad Y\mid X=x\sim\mathrm{Ber}\big(\sigma(-1+2x)\big).
``` |  |

The predictor is fitted using regularized logistic loss,

|  |  |  |
|----|----|----|
|  |
``` math
\ell((x,y);\theta)=\log(1+e^{\theta x})-y\theta x+\frac{\tau}{2}\|\theta\|_{2}^{2},
``` |  |

where $`\tau=1`$ is the regularization parameter. We take
$`\theta_{0}=1`$ and learning rate $`\alpha=0.5`$.

#### C.1.2 Score estimation.

We implement the score estimator in Section
[3](#S3 "3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
In the mild-feedback setting, each deployment uses 20,000 independent
auxiliary observations from each of the current and perturbed
distributions, totaling 40,000 observations per score fit, with
perturbation scale $`\eta=0.1`$. The basis estimator uses
$`b(x)=(1,2x-1)^{\top}`$ and the score model
$`s(z,\theta;\psi)=\psi^{\top}b(x)`$. We remark that this linear basis
does not contain the exact distributional score of the mixture model.
The near-nominal trajectory coverage observed in this setting is
consistent with the small contribution of propagated uncertainty,
despite the score approximation error.

In the strong-feedback setting, to better resolve the sharper deployment
response, we use a smaller perturbation scale $`\eta=0.01`$ and increase
the auxiliary sample size to 100,000 observations per distribution to
limit estimation variability. We use a correctly specified basis to
isolate the role of uncertainty propagation in coverage. The effects of
basis specification and auxiliary sample size are examined in the
score-matching ablation later in Appendix
[C.1.6](#A3.SS1.SSS6 "C.1.6 Score matching diagnostics ‣ C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
The basis score model is

|  |  |  |
|----|----|----|
|  |
``` math
s(z,\theta;\psi)=\psi_{0}+\psi_{1}s_{\theta}(x),\qquad s_{\theta}(x)=\partial_{\theta}\log p_{\theta}(x),
``` |  |

where $`p_{\theta}`$ is the deployment-dependent covariate density. This
basis contains the exact distributional score, allowing us to examine
covariance estimation under a correctly specified score model. Its
coefficients are nevertheless estimated from the auxiliary samples.

In both settings, the auxiliary batches are independent of the labeled
observations used for the predictor update, and the score-model
coefficients are refitted at every deployment.

#### C.1.3 Contraction and bias bounds.

We verify the contraction conditions for the synthetic experiment. Since
$`\tau=1`$,

|     |                                                                   |     |
|-----|-------------------------------------------------------------------|-----|
|     |
       ``` math
       \nabla_{\theta}\ell((x,y);\theta)=x\{\sigma(\theta x)-y\}+\theta,
       ```                                                                |     |

|  |  |  |
|----|----|----|
|  |
``` math
\nabla_{\theta}^{2}\ell((x,y);\theta)=1+x^{2}\sigma^{\prime}(\theta x)\in[1,5/4].
``` |  |

Thus the loss is $`1`$-strongly convex and $`5/4`$-smooth in $`\theta`$.
To bound its smoothness in the data, let
$`c_{\sigma}:=\sup_{u\in\mathbb R}|u|\sigma^{\prime}(u).`$ For
$`u\geqslant 0`$, the derivative of $`u\sigma^{\prime}(u)`$ has the sign
of $`1-u\tanh(u/2)`$. Consequently, the maximum is attained at the
unique positive solution of $`u\tanh(u/2)=1,`$ giving
$`u\approx 1.543405`$ and $`c_{\sigma}\approx 0.223872`$. We have

|  |  |  |
|----|----|----|
|  |
``` math
\partial_{x}\nabla_{\theta}\ell((x,y);\theta)=\sigma(\theta x)-y+\theta x\sigma^{\prime}(\theta x).
``` |  |

Therefore,
$`\left|\partial_{x}\nabla_{\theta}\ell((x,y);\theta)\right|\leqslant 1+c_{\sigma}`$,
and for $`y,y^{\prime}\in{\cal Y}`$, we have

|  |  |  |
|----|----|----|
|  |
``` math
\left|\nabla_{\theta}\ell((x,y);\theta)-\nabla_{\theta}\ell((x,y^{\prime});\theta)\right|\leqslant|x|\cdot|y-y^{\prime}|\leqslant|y-y^{\prime}|.
``` |  |

Combining these bounds, for $`z=(x,y)`$ and
$`z^{\prime}=(x^{\prime},y^{\prime})`$,

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\left|\nabla_{\theta}\ell(z;\theta)-\nabla_{\theta}\ell(z^{\prime};\theta)\right|`$ | $`\displaystyle\leqslant(1+c_{\sigma})|x-x^{\prime}|+|y-y^{\prime}|`$ |  |
|  |  | $`\displaystyle\leqslant\sqrt{(1+c_{\sigma})^{2}+1}\,\|z-z^{\prime}\|_{2},`$ |  |

where the last inequality follows from Cauchy–Schwarz.

A valid joint-smoothness constant is consequently

|  |  |  |
|----|----|----|
|  |
``` math
\beta=\max\left\{\frac{5}{4},\sqrt{(1+c_{\sigma})^{2}+1}\right\}\approx 1.580462,\qquad\gamma=1.
``` |  |

Next, the mixture weight satisfies
$`|a^{\prime}(\theta)|=0.8\,\sigma(\theta)\{1-\sigma(\theta)\}\leqslant 0.2.`$
The uniform and beta components share the same conditional outcome
distribution. Their joint total variation distance therefore equals the
total variation distance between their covariate distributions, which
satisfies

|  |  |  |
|----|----|----|
|  |
``` math
\frac{1}{2}\intop\nolimits_{0}^{1}\left|30x^{4}(1-x)-1\right|\,dx<0.41.
``` |  |

Couple the mixture indicators at two deployments using a common uniform
random variable. When they agree, use the same observation from the
selected component. When they differ, maximally couple the two component
distributions, so the observations differ with probability at most
$`0.41`$. Since the joint support $`[0,1]\times\{0,1\}`$ has diameter
$`\sqrt{2}`$, this gives

|  |  |  |
|----|----|----|
|  |
``` math
W_{1}\{\mathcal{D}(\theta),\mathcal{D}(\theta^{\prime})\}\leqslant 0.41\sqrt{2}\,|a(\theta)-a(\theta^{\prime})|\leqslant 0.082\sqrt{2}\,|\theta-\theta^{\prime}|.
``` |  |

Hence we may take $`\varepsilon=0.082\sqrt{2}\approx 0.115966`$. The RRM
contraction bound is therefore

|  |  |  |
|----|----|----|
|  |
``` math
\rho_{\mathrm{RRM}}=\frac{\varepsilon\beta}{\gamma}\approx 0.18327914<1.
``` |  |

For RGD, the step size $`\alpha=0.5`$ satisfies

|  |  |  |
|----|----|----|
|  |
``` math
\alpha\leqslant\frac{2}{\beta+\gamma}\approx 0.775055,\quad\varepsilon<\frac{\gamma}{(\beta+\gamma)(1+1.5\alpha\beta)}\approx 0.177330.
``` |  |

The contraction results of Perdomo et al. (2020, Theorems 3.5 and 3.8)
therefore apply, with

|  |  |  |
|----|----|----|
|  |
``` math
\rho_{\mathrm{RGD}}=1-\alpha\beta\left\{\frac{\gamma}{\beta+\gamma}-\varepsilon(1+1.5\alpha\beta)\right\}\approx 0.89402797<1.
``` |  |

For either population trajectory, with its corresponding $`\rho`$,

|  |  |  |
|----|----|----|
|  |
``` math
|\theta_{t}-\theta_{\mathrm{PS}}|\leqslant\rho^{t}|\theta_{0}-\theta_{\mathrm{PS}}|.
``` |  |

Following Corollaries
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
we bound the initial distance using only known model parameters. Since
$`X\in[0,1]`$, $`Y\in\{0,1\}`$, and $`0\leqslant\sigma(u)\leqslant 1`$,

|  |  |  |
|----|----|----|
|  |
``` math
\left|\nabla_{\theta}\ell((x,y);\theta_{0})\right|=\left|x\{\sigma(\theta_{0}x)-y\}+\tau\theta_{0}\right|\leqslant 1+\tau|\theta_{0}|.
``` |  |

Consequently, uniformly over $`\check{\theta}\in\Theta`$,

|  |  |  |
|----|----|----|
|  |
``` math
\left|\mathbb E_{\check{\theta}}[\nabla_{\theta}\ell(Z;\theta_{0})]\right|\leqslant 1+\tau|\theta_{0}|.
``` |  |

Strong convexity of the population loss under
$`\mathcal{D}(\theta_{\mathrm{PS}})`$ and its first-order condition at
$`\theta_{\mathrm{PS}}`$ therefore give

|  |  |  |
|----|----|----|
|  |
``` math
|\theta_{0}-\theta_{\mathrm{PS}}|\leqslant\frac{1}{\gamma}\left|\mathbb E_{\theta_{\mathrm{PS}}}[\nabla_{\theta}\ell(Z;\theta_{0})]\right|\leqslant\frac{1+\tau|\theta_{0}|}{\gamma}.
``` |  |

Define

|     |                                                                |     |
|-----|----------------------------------------------------------------|-----|
|     |
       ``` math
       \widetilde{B}:=|\theta_{0}|+\frac{1+\tau|\theta_{0}|}{\gamma}.
       ```                                                             |     |

Then both $`|\theta_{0}|`$ and $`|\theta_{\mathrm{PS}}|`$ are bounded by
$`\widetilde{B}`$. For the present configuration,
$`\theta_{0}=\tau=\gamma=1`$, so $`\widetilde{B}=3`$. Combining this
bound with contraction yields

|  |  |  |
|----|----|----|
|  |
``` math
|\theta_{t}-\theta_{\mathrm{PS}}|\leqslant\rho^{t}\bigl(|\theta_{0}|+|\theta_{\mathrm{PS}}|\bigr)\leqslant 2\widetilde{B}\,\rho^{t}=6\rho^{t}=:B_{t}.
``` |  |

This bound requires no evaluation of the stable point or expectations
under its deployment distribution. Consequently, whenever an interval
contains $`\theta_{t}`$, enlarging each endpoint outward by $`B_{t}`$
produces an interval containing $`\theta_{\mathrm{PS}}`$.

Model-specific contraction bounds. For the $`\kappa=80`$ controlled
experiment, we bound the population update derivatives directly. Let
$`G(q)`$ denote the population RRM update and
$`S(q)=\mathbb E_{\mathcal{D}(q)}[\nabla_{\theta}\ell(Z;q)]`$. Writing
$`f_{q}(x)=1+a(q)\{30x^{4}(1-x)-1\}`$, define

|  |  |  |
|----|----|----|
|  |
``` math
H_{q}(v)=1+\intop\nolimits_{0}^{1}x^{2}\sigma^{\prime}(vx)f_{q}(x)\,dx,\qquad\Delta(v)=\intop\nolimits_{0}^{1}x\{\sigma(vx)-\sigma(2x-1)\}\{30x^{4}(1-x)-1\}\,dx.
``` |  |

Since the ridge coefficient is $`\tau=1`$ and the logistic loss is
convex, $`H_{q}(v)\geqslant 1`$ uniformly in $`q`$ and $`v`$.
Differentiating the first-order condition for $`G`$ and the population
gradient $`S`$ yields

|  |  |  |
|----|----|----|
|  |
``` math
G^{\prime}(q)=-\frac{a^{\prime}(q)\Delta(G(q))}{H_{q}(G(q))},\qquad S^{\prime}(q)=H_{q}(q)+a^{\prime}(q)\Delta(q).
``` |  |

Using uniform numerical bounds, we have

|  |  |  |
|----|----|----|
|  |
``` math
\sup_{q\in\mathbb R}|G^{\prime}(q)|\leqslant 0.632751,\qquad 0.402321\leqslant S^{\prime}(q)\leqslant 1.123810.
``` |  |

Therefore, with $`\alpha=0.5`$,

|  |  |  |
|----|----|----|
|  |
``` math
\rho_{\mathrm{RRM}}\leqslant 0.632751<1,\ \rho_{\mathrm{RGD}}\leqslant\max\{|1-0.5(0.402321)|,\,|1-0.5(1.123810)|\}\leqslant 0.798840<1.
``` |  |

The mean value theorem establishes global contraction. Together with
$`\widetilde{B}=3`$, these bounds give $`B_{t}=6\rho^{t}`$, so the
criterion $`B_{t}<10^{-3}`$ selects rounds 20 and 39 for RRM and RGD,
respectively.

###### Remark C.1.

For Corollary
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
the same enlargement argument applies whenever a deterministic bound

|     |                                                                |     |
|-----|----------------------------------------------------------------|-----|
|     |
       ``` math
       \|\theta_{t}-\theta_{\mathrm{PS}}\|_{2}\leqslant B_{0}\rho^{t}
       ```                                                             |     |

is established, with $`B_{0}\geqslant 0`$ and $`\rho\in(0,1)`$.
Specifically, if, for a fixed $`t`$,

|  |  |  |
|----|----|----|
|  |
``` math
\liminf_{n\to\infty}\ProbOp\{\theta_{t}\in\widehat{R}_{t}\}\geqslant 1-\delta,
``` |  |

then

|  |  |  |
|----|----|----|
|  |
``` math
\liminf_{n\to\infty}\ProbOp\{\theta_{\mathrm{PS}}\in\widehat{R}_{t}+B(0,B_{0}\rho^{t})\}\geqslant 1-\delta.
``` |  |

Indeed, whenever $`\widehat{R}_{t}`$ contains $`\theta_{t}`$, its
enlargement by $`B(0,B_{0}\rho^{t})`$ contains $`\theta_{\mathrm{PS}}`$.
Thus, independently established model-specific contraction bounds can
replace the generic contraction factors in the bias-enlargement
arguments of Corollaries
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
provided the conditions ensuring trajectory-region coverage hold.

#### C.1.4 Inference and evaluation.

Population trajectories, covariance references, and performative stable
points are computed separately for the two synthetic settings using
512-point Gaussian quadrature. For the mild-feedback setting with
$`\kappa=1`$, this gives $`\theta_{\mathrm{PS}}\approx 0.053190`$. We
construct nominal $`95\%`$ confidence intervals for the population
iterate $`\theta_{t}`$ using the recursive covariance estimator. For
inference on $`\theta_{\mathrm{PS}}`$, each interval is expanded by

|     |                                                       |     |
|-----|-------------------------------------------------------|-----|
|     |
       ``` math
       B_{t}=2\widetilde{B}\,\rho^{t},\qquad\widetilde{B}=3,
       ```                                                    |     |

where $`\rho`$ is the contraction factor of the corresponding update
method. We use the support-based bound $`\widetilde{B}=3`$ and the
deterministic reporting rule

|     |                                                         |     |
|-----|---------------------------------------------------------|-----|
|     |
       ``` math
       t_{\mathrm{report}}=\min\{t\geqslant 1:B_{t}<10^{-3}\}.
       ```                                                      |     |

For the mild-feedback setting ($`\kappa=1`$), the reporting rule selects
round 6 for RRM and round 78 for RGD, using contraction bounds
$`\rho_{\mathrm{RRM}}\approx 0.18327914`$ and
$`\rho_{\mathrm{RGD}}\approx 0.89402797`$. Both methods are simulated
for 78 rounds. For the strong-feedback setting ($`\kappa=80`$), the same
reporting rule selects round 20 for RRM and round 39 for RGD, using the
model-specific contraction bounds
$`\rho_{\mathrm{RRM}}\approx 0.632751`$ and
$`\rho_{\mathrm{RGD}}\approx 0.798840`$. Both methods are simulated for
40 rounds.

Within each setting and method, the reporting round is fixed across all
Monte Carlo repetitions.

One-step ERM baseline. We include a performativity-agnostic baseline
that fits the predictor once using $`n`$ independent observations
$`Z_{1},\ldots,Z_{n}\sim\mathcal{D}(\theta_{0})`$ at the fixed initial
deployment:

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{\theta}_{1}=\arg\min_{\theta}\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\ell(Z_{i};\theta).
``` |  |

This estimator coincides with the first RRM update and targets
$`\theta_{1}=\arg\min_{\theta}\mathbb E_{\mathcal{D}(\theta_{0})}[\ell(Z;\theta)]`$.
We construct its confidence region using the ordinary sandwich
covariance estimator. Writing
$`g_{i}=\nabla_{\theta}\ell(Z_{i};\widehat{\theta}_{1})`$ and
$`\bar{g}=n^{-1}\sumop\displaylimits_{i=1}^{n}g_{i}`$, define

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{H}_{1}=\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{i};\widehat{\theta}_{1}),\qquad\widehat{\Omega}_{1}=\widehat{\text{Cov}}[\nabla\ell(Z;\widehat{\theta_{1}})],\qquad\widehat{V}_{1}=\widehat{H}_{1}^{-1}\widehat{\Omega}_{1}\widehat{H}_{1}^{-\top}.
``` |  |

We evaluate this same region against both $`\theta_{1}`$ and
$`\theta_{\mathrm{PS}}`$ without bias enlargement to account for
performativity agnostic in estimation. Its nominal coverage concerns
$`\theta_{1}`$; coverage of $`\theta_{\mathrm{PS}}`$ measures the
consequence of ignoring subsequent performative feedback.

Per-round M-estimator baseline. We also consider a baseline that uses
the same RRM point estimates and update samples but accounts only for
current-round sampling uncertainty. At round $`t`$, observations
$`Z_{t-1,1},\ldots,Z_{t-1,n}`$ are drawn independently from
$`\mathcal{D}(\widehat{\theta}_{t-1})`$ conditional on the preceding
deployment. Define

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{H}_{t}=\frac{1}{n}\sumop\displaylimits_{i=1}^{n}\nabla^{2}\ell(Z_{t-1,i};\widehat{\theta}_{t}),\qquad\widehat{\Omega}_{t}=\widehat{\operatorname{Cov}}\bigl[\nabla\ell(Z_{t-1,i};\widehat{\theta}_{t})\bigr].
``` |  |

The baseline replaces the recursive covariance estimator with the
current-round sandwich estimator

|  |  |  |
|----|----|----|
|  |
``` math
\widehat{V}_{t}^{\mathrm{M}}=\widehat{H}_{t}^{-1}\widehat{\Omega}_{t}\widehat{H}_{t}^{-\top},
``` |  |

which estimates $`\Sigma_{\theta_{t-1}}(\theta_{t})`$. This is
equivalent to setting $`\widehat{\nabla G}_{t-1}=0`$ in the covariance
recursion, thereby omitting the propagated uncertainty term

|     |                                                             |     |
|-----|-------------------------------------------------------------|-----|
|     |
       ``` math
       \nabla G(\theta_{t-1})V_{t-1}\nabla G(\theta_{t-1})^{\top}.
       ```                                                          |     |

Thus, repeated retraining still accounts for the changing deployment
distribution, but covariance estimation treats the preceding deployment
as fixed. We construct nominal confidence regions using
$`\widehat{V}_{t}^{\mathrm{M}}/n`$ and evaluate their coverage of
$`\theta_{t}`$. For inference on the performative stable point, we
retain the same deterministic bias enlargement as our RRM method. This
comparison isolates the effect of accounting for uncertainty propagated
across deployments.

#### C.1.5 Prediction-Powered Inference

We evaluate the benefits of prediction-powered inference method
introduced in Appendix
[A](#A1 "Appendix A Prediction-Powered Inference under Performativity ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
using the fixed pseudo-labeler $`f(x)=\mathbf{1}(x\geqslant 0.6)`$ and
varying the unlabeled-to-labeled sample ratio $`N/n`$ over
$`\{1,2,5,10,20\}`$. Table
[8](#A3.T8 "Table 8 ‣ C.1.5 Prediction-Powered Inference ‣ C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
reports the mean and RMSE of the selected weight, showing that its mean
closely tracks the population-optimal $`\lambda^{\star}`$. Figure
[5](#A3.F5 "Figure 5 ‣ C.1.5 Prediction-Powered Inference ‣ C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
complements these results by comparing stable-point interval widths,
selected weights, and coverage across the sample-size ratios.

|  |  |  |  |
|----|----|----|----|
| $`N/n`$ |  Population $`\lambda^{\star}`$ |  Mean $`\widehat{\lambda}`$ |  RMSE of $`\widehat{\lambda}`$ |
| 1 | 0.107863 | 0.107762 | 0.011998 |
| 2 | 0.143818 | 0.143778 | 0.015899 |
| 5 | 0.179772 | 0.179700 | 0.019544 |
| 10 | 0.196115 | 0.196447 | 0.021505 |
| 20 | 0.205454 | 0.205390 | 0.022777 |

Table 8: Population-optimal and empirically selected PPI weights in the
pure-synthetic experiment, with $`n=2000`$ and fixed round $`t={6}`$.
Means and RMSEs use 10,000 Monte Carlo repetitions. For each $`N/n`$,
$`\lambda^{\star}`$ is the common reference evaluated at the
deterministic population deployment for round 6.

Figure 5: Prediction-powered inference in the pure-synthetic experiment.
From left to right: mean stable-point interval width, mean selected
$`\widehat{\lambda}`$, and stable-point coverage as $`N/n`$ varies. The
gray dashed curve shows the population-optimal $`\lambda^{\star}`$. The
horizontal dashed line marks nominal $`95\%`$ coverage; shaded bands are
pointwise exact $`95\%`$ binomial Monte Carlo confidence intervals.

#### C.1.6 Score matching diagnostics

To examine how score estimation affects inference through the
uncertainty propagation term, we conduct a score-matching ablation in
the strong-feedback synthetic setting of Section
[5.1.2](#S5.SS1.SSS2 "5.1.2 Uncertainty Propagation under Strong Performative Feedback. ‣ 5.1 Controlled Synthetic Studies ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
with activation function
$`a(\theta)=0.8\sigma\bigl(80(\theta-0.05)\bigr)`$. We vary the
score-model specification and auxiliary sample size to assess their
effects on score-estimation error and confidence-region coverage. Within
each algorithm, all score specifications use the same update samples and
parameter trajectories, isolating the effect of score estimation on
covariance estimation and the resulting inference.

We test the following methods for score function estimation:

1.  1.
    Basis: an oracle-informed linear family
    $`s_{b,c}(x,\theta)=b+c\,s_{\theta}(x)`$, where
    $`s_{\theta}(x)=\partial_{\theta}\log p_{\theta}(x)`$ is the exact
    score. This family contains the true score and includes an
    intercept. Both coefficients are fitted without penalization at each
    deployment, retaining the policy-derivative term in the
    score-matching objective.
2.  2.
    Misspecified basis: a polynomial basis of degree 12 in $`x`$,
    refitted at each deployment.
3.  3.
    Exact: the analytically evaluated exact score, with no
    score-estimation error. The covariance estimator still uses
    empirical moments and therefore remains subject to sampling
    variability.
4.  4.
    Zero: the zero-valued constant function as a trivial estimator of
    the score function.

We fix the finite-difference step at $`\eta=0.01`$ and vary the
auxiliary sample size over $`m\in\{10^{3},10^{4},10^{5}\}`$. Each fit
uses $`m`$ independent observations from each of the base and perturbed
distributions, totaling $`2m`$ observations. All score methods use the
same sampled parameter trajectories, so their comparison isolates the
effect of score estimation on the confidence intervals. The empirical
experiment uses $`n=10{,}000`$ labeled observations per deployment and
10,000 Monte Carlo repetitions. Trajectory coverage uses nominal
$`95\%`$ Wald intervals, whereas performative stable point coverage uses
the same intervals enlarged by $`B_{t}`$ at each endpoint.
Score-estimation RMSE is evaluated at the deployment parameter used for
the reported update. Results are evaluated at fixed round $`t=20`$ for
RRM and $`t=39`$ for RGD, selected by

|  |  |  |
|----|----|----|
|  |
``` math
t_{\mathrm{report}}=\min\{t\geqslant 1:B_{t}<10^{-3}\},\qquad B_{t}=6\rho^{t}.
``` |  |

Here we use the model-specific global contraction bounds
$`\rho_{\mathrm{RRM}}\approx 0.632751`$ and
$`\rho_{\mathrm{RGD}}\approx 0.798840`$ obtained as in previous section.
The corresponding bias bounds are approximately $`6.35\times 10^{-4}`$
and $`9.42\times 10^{-4}`$, respectively.

Tables
[9](#A3.T9 "Table 9 ‣ C.1.6 Score matching diagnostics ‣ C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[10](#A3.T10 "Table 10 ‣ C.1.6 Score matching diagnostics ‣ C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
report score-estimation RMSE and coverage of the population iterate
$`\theta_{t}`$ and performative stable point $`\theta_{\mathrm{PS}}`$
for RRM and RGD, respectively. Figure
[6](#A3.F6 "Figure 6 ‣ C.1.6 Score matching diagnostics ‣ C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
compares coverage across the same methods and auxiliary sample sizes.
Increasing $`m`$ substantially reduces score-estimation RMSE for both
fitted families. The correctly specified, oracle-informed basis yields
trajectory coverage close to the exact-score method, whereas the
misspecified polynomial basis retains significant trajectory
undercoverage. After bias enlargement, both fitted families attain
stable-point coverage above the nominal $`95\%`$ level, while the
zero-score approximation remains below nominal coverage for both
algorithms. Both fitted families improve coverage substantially over the
zero-score approximation and the score estimation error substantially
decreases as the sample number $`m`$ increases, although coverage need
not increase monotonically with $`m`$. These results demonstrate that
perturbation-based score estimation can substantially improve coverage
over the trivial zero-score approximation.

|  |  |  |  |  |
|----|----|----|----|----|
| Score method | $`m`$ | Score RMSE |  Coverage of $`\theta_{t}`$ (%) |  Coverage of $`\theta_{\mathrm{PS}}`$ (%) |
| Exact | – | 0.000 | 94.56 | 96.59 |
| Zero | – | 14.268 | 88.77 | 92.11 |
| Basis | 1,000 | 3.568 | 94.49 | 96.56 |
| Basis | 10,000 | 1.122 | 94.67 | 96.63 |
| Basis | 100,000 | 0.359 | 94.67 | 96.60 |
| Misspecified basis | 1,000 | 16.657 | 93.69 | 95.96 |
| Misspecified basis | 10,000 | 5.019 | 93.39 | 95.74 |
| Misspecified basis | 100,000 | 2.442 | 93.22 | 95.64 |

Table 9: Score estimation and coverage for RRM in the $`\kappa=80`$
strong feedback experiment, with $`n=10{,}000`$, $`\eta=0.01`$, and
10,000 Monte Carlo repetitions, evaluated at $`t=20`$. Each fitted score
uses $`m`$ independent auxiliary observations per distribution, totaling
$`2m`$. Score RMSE is the square root of the average integrated squared
error across repetitions. Trajectory coverage uses nominal $`95\%`$ Wald
intervals; stable-point intervals are enlarged at each endpoint by
$`B_{t}=6\rho^{t}`$, with the round selected by $`B_{t}<10^{-3}`$.

|  |  |  |  |  |
|----|----|----|----|----|
| Score method | $`m`$ | Score RMSE |  Coverage of $`\theta_{t}`$ (%) |  Coverage of $`\theta_{\mathrm{PS}}`$ (%) |
| Exact | – | 0.000 | 94.96 | 98.13 |
| Zero | – | 14.342 | 83.96 | 92.59 |
| Basis | 1,000 | 3.518 | 95.01 | 97.92 |
| Basis | 10,000 | 1.106 | 94.96 | 98.14 |
| Basis | 100,000 | 0.346 | 94.90 | 98.14 |
| Misspecified basis | 1,000 | 16.714 | 93.22 | 97.13 |
| Misspecified basis | 10,000 | 5.042 | 93.15 | 97.43 |
| Misspecified basis | 100,000 | 2.475 | 93.22 | 97.40 |

Table 10: Score estimation and coverage for RGD in the $`\kappa=80`$
strong feedback experiment, with $`n=10{,}000`$, $`\eta=0.01`$, and
10,000 Monte Carlo repetitions, evaluated at $`t=39`$. Each fitted score
uses $`m`$ independent auxiliary observations per distribution, totaling
$`2m`$. Score RMSE is the square root of the average integrated squared
error across repetitions. Trajectory coverage uses nominal $`95\%`$ Wald
intervals; stable-point intervals are enlarged at each endpoint by
$`B_{t}=6\rho^{t}`$, with the round selected by $`B_{t}<10^{-3}`$.

Figure 6: Coverage under different score estimators. RRM and RGD are
evaluated at fixed rounds 20 and 39, respectively. Filled circles show
trajectory coverage using ordinary Wald intervals; open squares show
stable-point coverage after bias enlargement. Error bars are pointwise
$`95\%`$ Wilson Monte Carlo intervals. The dashed line marks nominal
$`95\%`$ coverage.

### C.2 Semi-Synthetic Experiment

In this subsection, we provide implementation details for semi-synthetic
experiment (Section
[5.2](#S5.SS2 "5.2 Semi-synthetic Experiment ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")).

#### C.2.1 Dataset and deployment population.

The Learning Mindset dataset \[[Athey and Wager, 2019](#bib.bib9)\]
contains 10,391 observations recording student characteristics,
intervention assignments, and academic achievement. We allocate 2,000
observations to simulator training and 500 to validation, with splits
stratified by school and intervention assignment. The binary achievement
threshold is the median achievement score in the simulator-training
sample. The remaining 7,891 student profiles form a construction set for
generating deployment contexts.

We generate 100,000 synthetic contexts by independently sampling
construction profiles with replacement and perturbing their five
continuous school-level covariates. Specifically, the perturbation is
$`0.05LU`$, where $`LL^{\top}`$ is the empirical covariance matrix of
these covariates in the construction set and $`U`$ is uniformly
distributed on the five-dimensional ball of radius $`\sqrt{7}`$, so that
$`\operatorname{Cov}(U)=I_{5}`$. Perturbed values are reflected into
their respective construction-set ranges, while categorical and ordinal
characteristics retain their sampled values. Perturbations are applied
before the fixed encoding and PCA transformations. These 100,000
generated contexts form a single fixed deployment pool. Empirical
RRM/RGD updates sample uniformly with replacement from this pool, while
population reference trajectories and stable points are computed using
the same pool.

##### Simulator.

Student characteristics are encoded into 26 covariates, comprising six
standardized numerical variables and 20 categorical indicator variables.
Encoding parameters are fitted using the simulator-training sample. The
outcome simulator maps these covariates through a shared tanh layer with
24 hidden units and two treatment-specific logistic output heads.
Training minimizes binary log loss for the observed intervention
assignments using Adam with learning rate $`0.006`$, batch size $`256`$,
and an $`L_{2}`$ weight-penalty coefficient of $`2\times 10^{-4}`$.

Training runs for at most 1,000 epochs, with early stopping after 80
consecutive epochs without a validation log-loss improvement exceeding
$`10^{-5}`$. The retained checkpoint is selected using validation loss
and remains fixed throughout the deployment experiment. In the fitted
simulator used here, training terminates after 120 epochs and the
retained checkpoint comes from epoch 40.

##### Feature representation.

For the deployed predictor, we retain three principal components fitted
to the encoded training covariates and whiten them using their training
standard deviations. Let $`\phi_{0}(x)\in\mathbb R^{3}`$ denote these
whitened coordinates. We define

|  |  |  |
|----|----|----|
|  |
``` math
\phi(x)=\frac{s_{\phi}}{c}\big(1,\ \phi_{0}(x)\big)^{\top},\qquad c=\max_{i\in\mathcal{I}_{\mathrm{train}}}\left\|\big(1,\ \phi_{0}(x_{i})\big)^{\top}\right\|_{2},
``` |  |

where $`\mathcal{I}_{\mathrm{train}}`$ indexes the simulator-training
data, $`c\approx 4.67`$, and the feature multiplier is
$`s_{\phi}\approx 3.54`$. The resulting predictor has four parameters,
including the intercept. The retained components explain approximately
$`61.688\%`$ of the variance in the encoded training covariates. The
encoding, PCA transformation, whitening scales, and feature
normalization remain fixed across deployment rounds and repetitions.

#### C.2.2 Deployment and experimental protocol.

We initialize $`\theta_{0}=\mathbf{0}`$, use regularization $`\tau=1`$,
and set the RGD step size to $`\alpha=0.8`$. Intervention assignment
uses $`\delta=0.1`$ and $`\kappa=6`$. At each deployment, student
profiles are sampled independently and uniformly with replacement from
the fixed deployment pool of 100,000 augmented contexts. Intervention
assignments and binary outcomes are then generated according to the
deployment law described in the main text, using the frozen outcome
simulator.

We conduct 10,000 independent Monte Carlo repetitions for each method
and labeled sample size $`n=2000`$. Fresh observations are generated at
every deployment. The simulator and feature transformation remain fixed
across repetitions.

##### Score estimation.

In this setup, we can directly evaluate the analytic score at deployment
parameter $`q`$:

|  |  |  |
|----|----|----|
|  |
``` math
s_{q}(x,w)=\frac{w-e_{q}(x)}{e_{q}(x)(1-e_{q}(x))}\nabla_{q}e_{q}(x).
``` |  |

This expression depends on the intervention policy and does not require
outcome labels.

#### C.2.3 Inference and evaluation.

Population trajectories, covariance references, and the performative
stable point are computed by averaging uniformly over the fixed pool of
100,000 generated contexts and summing over the possible
intervention–outcome states under the frozen simulator. Confidence
regions are constructed from observations sampled from this same pool,
and coverage is evaluated against its population iterates and
performative stable point.

We construct nominal $`95\%`$ Wald confidence ellipsoids for the
four-dimensional population iterate using the feasible covariance
estimators in Corollaries
[3.5](#S3.Thmtheorem5 "Corollary 3.5. ‣ 3.2 Estimation of the Asymptotic Covariance ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.3](#S4.Thmtheorem3 "Corollary 4.3. ‣ 4.2 Estimation of the Asymptotic Covariance ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
with a $`\chi^{2}_{4}`$ cutoff. For stable-point inference, we enlarge
each ellipsoid by a Euclidean ball, following Corollaries
[3.6](#S3.Thmtheorem6 "Corollary 3.6 (Confidence region for 𝜃_PS). ‣ 3.3 Confidence Regions for the Performative Stable Point ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.4](#S4.Thmtheorem4 "Corollary 4.4 (Confidence region for 𝜃_PS). ‣ 4.3 Confidence Regions for the Performative Stable Point ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

##### Contraction bounds.

We remark here that as in Appendix
[C.1](#A3.SS1 "C.1 Pure-Synthetic Experiment ‣ Appendix C Experimental Details ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
explicitly differentiating the conditional mean and the population
updates gives sharper contraction bounds for the semi-synthetic model.
For the fixed covariate distribution, write

|  |  |  |
|----|----|----|
|  |
``` math
Q=\mathbb{E}[\phi(X)\phi(X)^{\top}],\qquad L_{\phi}=\lambda_{\max}(Q),\qquad c=\frac{(1-2\delta)\kappa}{4}.
``` |  |

Let $`p(x,w)=\ProbOp(Y=1\mid X=x,W=w)`$ be the conditional outcome
probability produced by the fixed simulator. Under deployment $`u`$, the
intervention probability is

|     |                                                             |     |
|-----|-------------------------------------------------------------|-----|
|     |
       ``` math
       e_{u}(x)=\delta+(1-2\delta)\sigma(-\kappa u^{\top}\phi(x)).
       ```                                                          |     |

Define the conditional outcome mean under this deployment by

|  |  |  |
|----|----|----|
|  |
``` math
m_{u}(x):=\mathbb{E}_{{\cal D}(u)}[Y\mid X=x]=p(x,0)+e_{u}(x)\{p(x,1)-p(x,0)\}.
``` |  |

Differentiating then gives

|  |  |  |
|----|----|----|
|  |
``` math
\nabla_{u}m_{u}(x)=-(1-2\delta)\kappa\{p(x,1)-p(x,0)\}\sigma^{\prime}(-\kappa u^{\top}\phi(x))\phi(x).
``` |  |

Differentiating the first-order condition defining $`G`$ and the
population gradient $`S`$ gives

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\nabla G(u)`$ | $`\displaystyle=H_{u}(G(u))^{-1}\mathbb{E}[\phi(X)\nabla_{u}m_{u}(X)^{\top}],`$ |  |
|  | $`\displaystyle\nabla S(u)`$ | $`\displaystyle=H_{u}(u)-\mathbb{E}[\phi(X)\nabla_{u}m_{u}(X)^{\top}].`$ |  |

Because $`|p(x,1)-p(x,0)|\leqslant 1`$ and
$`0\leqslant\sigma^{\prime}\leqslant 1/4`$, the matrix
$`\mathbb{E}[\phi(X)\nabla_{u}m_{u}(X)^{\top}]`$ is symmetric with
spectral norm at most $`cL_{\phi}`$, while
$`\tau I_{d}\preceq H_{u}(v)\preceq(\tau+L_{\phi}/4)I_{d}`$.
Consequently,

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\|\nabla G(u)\|_{2}`$ | $`\displaystyle\leqslant\frac{cL_{\phi}}{\tau},\|I_{d}-\alpha\nabla S(u)\|_{2}\leqslant\max\left\{|1-\alpha(\tau-cL_{\phi})|,\,\left|1-\alpha\{\tau+(1/4+c)L_{\phi}\}\right|\right\}.`$ |  |

These bounds hold uniformly over $`u\in\mathbb R^{d}`$ and give global
contraction factors for RRM and RGD, respectively, whenever

|     |                                                                |     |
|-----|----------------------------------------------------------------|-----|
|     |
       ``` math
       \tau>cL_{\phi},\qquad 0<\alpha<\frac{2}{\tau+(1/4+c)L_{\phi}}.
       ```                                                             |     |

For the fixed deployment pool, $`L_{\phi}\approx 0.63`$,
$`\rho_{\mathrm{RRM}}\approx 0.76`$, and
$`\rho_{\mathrm{RGD}}\approx 0.81`$. With $`\theta_{0}=\mathbf{0}`$, the
bias radius used for each method is
$`B_{t}=\frac{\sqrt{L_{\phi}}}{\tau}\rho^{t},`$ where $`\rho`$ is the
corresponding contraction bound. We use the deterministic reporting rule

|     |                                                         |     |
|-----|---------------------------------------------------------|-----|
|     |
       ``` math
       t_{\mathrm{report}}=\min\{t\geqslant 1:B_{t}<10^{-3}\}.
       ```                                                      |     |

This selects fixed round 24 for RRM and fixed round 31 for RGD. The same
reporting round is used for every repetition within each method. Each
simulation saves 100 rounds; the reported results use these fixed rounds
rather than empirical stopping rounds. The trajectory-region diameter is

|  |  |  |
|----|----|----|
|  |
``` math
2\sqrt{\frac{\chi^{2}_{4,0.95}\lambda_{\max}(\widehat{V}_{t})}{n}},
``` |  |

where $`\widehat{V}_{t}`$ is the empirical variance estimated using
Theorems
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems"),
and the stable-point-region diameter adds $`2B_{t}`$.

### C.3 LLM Routing

This section provides implementation details about the LLM routing
experiment in Section
[5.3](#S5.SS3 "5.3 Self-improving LLM routing ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").

#### C.3.1 Dataset and response-bank construction.

We assemble 8,792 questions from GSM8K and 10,962 questions from the
training and validation splits of CommonsenseQA. Of these, 500 questions
are reserved for calibration of the feature transformation. The
remaining 19,254 questions, comprising 8,572 GSM8K questions and 10,682
CommonsenseQA questions, form the fixed deployment pool. Empirical
updates and population reference calculations use this same deployment
pool.

For each question, the response bank contains ten entries under each of
two modes of the frozen Qwen2.5-1.5B model. The fast mode uses direct
answer generation on both datasets. For GSM8K, the slow mode uses
reasoning followed by a final answer, retaining all ten draws per mode
from the existing bank without filtering by correctness. Inspired by
self-consistency decoding \[[Wang et al., 2022](#bib.bib11)\], we
construct the CommonsenseQA slow mode by plurality voting over five
independently sampled direct answers. Ties in this vote are broken
uniformly at random using a fixed seed, without consulting reference
answers. Thus, ten slow entries require fifty component generations per
CommonsenseQA question. The token cost of each voted entry is the sum of
the token counts of its five component generations.

Generation uses temperature $`0.6`$ and top-$`p=0.95`$, with maximum
lengths of 8,192 reasoning tokens and 64 final-answer tokens. Final
answers are constrained to the appropriate format: numerical answers for
GSM8K and option letters for CommonsenseQA. Reference answers and worked
solutions are excluded from the prompts. Incorrect and duplicate
responses are retained with their sampling multiplicities.

The bank contains 395,080 mode-level entries: 10,000 for calibration
from 500 questions and 385,080 for empirical deployment. A voted slow
answer counts as one bank entry. The model and response bank remain
fixed throughout the routing experiment.

##### Question representation.

We represent each question by the model’s 1,536-dimensional final-layer
hidden state at the last token of a question-only prompt, including the
answer choices for CommonsenseQA. Before PCA, we concatenate two
unit-valued one-hot dataset indicators with this embedding, giving a
1,538-dimensional input. We fit PCA using only the 500 calibration
questions and retain four components, which explain $`74.852\%`$ of the
calibration variance. Each component is standardized using its
calibration mean and standard deviation and clipped to $`[-3,3]`$. We
additionally radially clip the resulting four-dimensional vector to have
Euclidean norm at most $`\sqrt{19}`$. Writing
$`\phi_{0}(x)\in\mathbb R^{4}`$ for these clipped coordinates, we define

|  |  |  |
|----|----|----|
|  |
``` math
\phi(x)=c_{\phi}(1,\phi_{0}(x)^{\top})^{\top},\qquad{c_{\phi}=\sqrt{\frac{45.125}{20}}\approx 1.50.}
``` |  |

Thus, the router parameter belongs to $`\Theta=\mathbb R^{5}`$. The
representation satisfies $`\|\phi(x)\|_{2}^{2}\leqslant 45.125`$.
Dataset identity enters the router through the joint PCA representation
and is also retained for stratified reporting. The fitted transformation
remains fixed across deployment rounds, sample sizes, and repetitions.

#### C.3.2 Deployment and experimental protocol.

We initialize $`\theta_{0}=\mathbf{0}`$, use regularization $`\tau=1`$,
and set the RGD step size to $`\alpha=0.8`$. The logistic predictor and
routing probabilities are

|  |  |  |
|----|----|----|
|  |
``` math
h_{\theta}(x)=\sigma(\theta^{\top}\phi(x)),\qquad\pi_{\theta}(1\mid x)=(1-\xi)h_{\theta}(x)+\frac{\xi}{2},\qquad\xi=0.2,
``` |  |

where mode $`1`$ denotes the dataset-specific slow mode: reasoning
followed by a final answer for GSM8K, and five-answer voting for
CommonsenseQA. The exploration mixture governs candidate generation,
while the regularized logistic loss uses $`h_{\theta}`$.

At each deployment, we sample $`n`$ questions independently and
uniformly with replacement. For each question, we independently sample
five modes from $`\pi_{\theta}(\cdot\mid x)`$ and then sample one
response uniformly from the ten stored entries for each selected mode.
Response sampling is also with replacement. For the reported experiment,
utility is correctness minus $`0.1`$ times generated tokens divided by
$`1000`$:

|  |  |  |
|----|----|----|
|  |
``` math
u(x,r)=\mathbf{1}\{r\text{ is correct}\}-0.1\,\frac{\operatorname{tokens}(r)}{1000}.
``` |  |

The candidate with the highest utility wins, with ties resolved in favor
of the first candidate. Its mode supplies the binary supervision label.

Let $`J^{*}`$ denote the winning candidate, and write
$`U_{j}=u(X,R_{j})`$. Define the highest observed utility within each
represented mode: $`M_{a}=\max_{j:A_{j}=a}U_{j},\ a\in\{0,1\}.`$ We
weight the logistic loss by the observed utility gap:

|  |  |  |
|----|----|----|
|  |
``` math
w(Z)=\begin{cases}|M_{1}-M_{0}|/C,&\text{if both modes appear},\\
0,&\text{otherwise}.\end{cases}\qquad C=1+\frac{0.1\times 8256}{1000}=1.8256.
``` |  |

Here, $`8256`$ is the maximum total token count for a GSM8K slow-mode
response. It also bounds the token count of every other response-bank
entry, including CommonsenseQA voting entries, which use at most
$`5\times 64=320`$ tokens. Consequently, utilities lie in
$`[-0.8256,1]`$, so $`|M_{1}-M_{0}|\leqslant 1.8256=C`$ and
$`0\leqslant w(Z)\leqslant 1`$. The observation-level loss is

|  |  |  |
|----|----|----|
|  |
``` math
\ell(Z;\theta)=w(Z)\left[\log\left(1+e^{\theta^{\top}\phi(X)}\right)-A_{J^{*}}\theta^{\top}\phi(X)\right]+\frac{\tau}{2}\|\theta\|_{2}^{2}.
``` |  |

Candidate sets containing only one mode, and mixed-mode candidate sets
whose best fast and slow utilities are equal, have zero weight. All
$`n`$ observations remain in the empirical-average denominator; we do
not divide by the sum of the weights.

Fresh question–candidate observations are sampled at every deployment.
We report $`n=10000`$ labeled question-level observations per deployment
and 10,000 independent Monte Carlo repetitions. Each observation
contains five candidates. Simulations retain 198 rounds for both RRM and
RGD.

##### Distribution score and covariance estimation.

For the joint observation

|     |                                       |     |
|-----|---------------------------------------|-----|
|     |
       ``` math
       Z=(X,A_{1},R_{1},\ldots,A_{N},R_{N}),
       ```                                    |     |

question sampling and the mode-specific response distributions do not
depend on $`\theta`$. Consequently, the distribution score is available
analytically:

|  |  |  |
|----|----|----|
|  |
``` math
s_{q}(Z)=\frac{\left\{\sumop\displaylimits_{j=1}^{N}A_{j}-N\pi_{q}(1\mid X)\right\}(1-\xi)h_{q}(X)\{1-h_{q}(X)\}}{\pi_{q}(1\mid X)\{1-\pi_{q}(1\mid X)\}}\phi(X).
``` |  |

The results reported here use this analytic score, so no score fitting
or auxiliary perturbation batches are required. Scores are evaluated at
the deployment parameter that generated the batch, while loss gradients
and Hessians are evaluated at the updated parameter for RRM and at the
same deployment parameter for RGD.

##### Population references.

All population calculations are conditional on the fixed merged pool of
19,254 questions, response bank, and feature transformation. We consider
best-of-$`N`$ selection, with $`N=5`$ in the reported experiment. For a
deployed parameter $`q`$, define

|  |  |  |
|----|----|----|
|  |
``` math
h_{q}(x)=\sigma(q^{\top}\phi(x)),\qquad b_{q}(x)=\pi_{q}(1\mid x)=\frac{\xi}{2}+(1-\xi)h_{q}(x).
``` |  |

The probability of drawing exactly $`j`$ slow-mode candidates is

|  |  |  |
|----|----|----|
|  |
``` math
v_{j}(h_{q}(x))=\binom{N}{j}b_{q}(x)^{j}\{1-b_{q}(x)\}^{N-j},\qquad j=0,\ldots,N.
``` |  |

Write $`\mathbb E_{j}[\cdot\mid X]`$ for expectation over candidate
randomness conditional on exactly $`j`$ slow candidates and $`N-j`$ fast
candidates. Responses are sampled independently and uniformly with
replacement from the corresponding frozen banks. This conditional law
does not depend on $`q`$. Since single-mode draws have zero weight, the
population loss gradient at a candidate parameter $`\theta`$ is

|  |  |  |
|----|----|----|
|  |
``` math
\mathcal{S}(q,\theta)=\tau\theta+\mathbb E_{X}\!\left[\sumop\displaylimits_{j=1}^{N-1}v_{j}(h_{q}(X))\mathbb E_{j}\!\left[w(h_{\theta}(X)-A_{J^{\ast}})\mid X\right]\phi(X)\right].
``` |  |

Here $`A_{J^{\ast}}`$ is the winning mode, and $`w`$ is the normalized
absolute utility gap between the best slow and fast candidates. The
conditional expectations are computed exactly from the frozen response
banks using the distributions of the within-mode maximum utilities.
Thus, the population RRM map $`G(q)`$ solves $`\mathcal{S}(q,G(q))=0`$,
while the population RGD map is $`q-\alpha\mathcal{S}(q,q)`$.

We compute population trajectories, covariance references, and the
performative stable point by averaging uniformly over the merged
question pool and integrating over candidate randomness. These reference
calculations are performed after empirical sampling.

#### C.3.3 Inference and evaluation.

We construct nominal $`95\%`$ joint confidence ellipsoids for the
five-dimensional population iterate using the covariance recursions in
Theorems
[3.2](#S3.Thmtheorem2 "Theorem 3.2 (Central limit theorem for RERM). ‣ 3.1 Central Limit Theorem of RRM ‣ 3 Statistical Inference for RRM ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
and
[4.1](#S4.Thmtheorem1 "Theorem 4.1 (Central limit theorem for REGD). ‣ 4.1 Central Limit Theorem of RGD ‣ 4 Statistical Inference for RGD ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems").
For inference on the performative stable point, each ellipsoid is
enlarged by a Euclidean ball of radius $`B_{t}`$. Coverage is evaluated
using membership in this ellipsoid-plus-ball region. Its Euclidean
diameter is twice the sum of the largest ellipsoid semi-axis and
$`B_{t}`$.

Contraction bounds. As in the semi-synthetic experiment, we obtain
contraction bounds specific to the routing model by differentiating the
population RRM and RGD updates under the exact probabilistic model.
Conditional on $`X`$, the number of slow candidates follows a binomial
distribution with parameters $`N`$ and
$`\xi/2+(1-\xi)\sigma(u^{\top}\phi(X))`$. Conditional on this count, the
response-bank distribution is independent of the deployed parameter.
Using these count probabilities, $`0\leqslant w\leqslant 1`$, and the
convention $`w=0`$ for single-mode draws, we bound the update Jacobians
uniformly over $`u\in{\mathbb R}^{d}`$. The resulting scalar derivative
bounds are verified using Bernstein polynomial bounds on $`[0,1]`$.

For $`N=5`$, $`\xi=0.2`$, $`\tau=1`$, $`\alpha=0.8`$, and
$`L_{\phi}=\lambda_{\max}(\mathbb{E}[\phi(X)\phi(X)^{\top}])\leqslant 2.643`$,
the resulting contraction factors are

|  |  |  |
|----|----|----|
|  |
``` math
\rho_{\mathrm{RRM}}\approx 0.851901,\qquad\rho_{\mathrm{RGD}}\approx 0.956687.
``` |  |

These constants depend only on the candidate count, exploration, weight
bound, feature second-moment bound, and algorithm parameters; they
require neither the stable point nor the response banks’ correctness
rates or utility-gap moments.

For $`n=10000`$, the deterministic reporting criterion $`B_{t}<10^{-3}`$
selects round 55 for RRM and round 198 for RGD, where we have

|     |                           |     |
|-----|---------------------------|-----|
|     |
       ``` math
       B_{t}=6.297670\,\rho^{t}.
       ```                        |     |

These rounds are shared by all Monte Carlo repetitions. Coverage is
pointwise in the deployment round and conditional on the frozen question
pools, response bank, and feature transformation.

### C.4 Jacobian estimation diagnostics.

For each setting, we evaluate the RRM Jacobian estimator at its fixed
reporting round $`T`$. In repetition $`r`$, the estimator
$`\widehat{\nabla G}_{T-1}^{(r)}`$ is compared with the population
derivative $`\nabla G(\widehat{\theta}_{T-1}^{(r)})`$ at the same
realized deployment. Population derivatives are computed using numerical
quadrature in the synthetic settings and by averaging over the fixed
deployment pools and integrating over the remaining randomness in the
semi-synthetic and LLM settings. These calculations are used only for
evaluation.

With $`R=10{,}000`$ repetitions, we report mean population Jacobian norm
$`\frac{1}{R}\sumop\displaylimits_{r=1}^{R}\left\|\nabla G(\widehat{\theta}_{T-1}^{(r)})\right\|_{2},`$
and Jacobian RMSE

|  |  |  |
|----|----|----|
|  |
``` math
\left[\frac{1}{R}\sumop\displaylimits_{r=1}^{R}\left\|\widehat{\nabla G}_{T-1}^{(r)}-\nabla G(\widehat{\theta}_{T-1}^{(r)})\right\|_{2}^{2}\right]^{1/2},
``` |  |

where $`\|\cdot\|_{2}`$ denotes the spectral norm. Thus, the RMSE
measures matrix estimation error rather than the difference between the
estimated and population norms. For the zero-Jacobian baseline, we
replace $`\widehat{\nabla G}_{T-1}^{(r)}`$ by zero in the same
calculation. Its RMSEs are approximately $`0.0079`$, $`0.5737`$,
$`0.1312`$, and $`0.0661`$ for the mild synthetic, strong synthetic,
semi-synthetic, and LLM settings, respectively. The coverage column in
Table
[7](#S5.T7 "Table 7 ‣ 5.4 Feedback Strength and Uncertainty Propagation ‣ 5 Experiment ‣ Statistical Inference under Performativity: From Policy Feedback to Self-Evolving Systems")
reports the per-round M-estimator’s coverage of the performative stable
point, retaining the same deterministic bias enlargement as RRM.
````
