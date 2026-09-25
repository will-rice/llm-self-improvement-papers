---
identifier: arxiv:2510.13501v1
title: "Confidence as a Reward: Transforming LLMs into Reward Models"
authors:
  - He Du
  - Bowen Li
  - Chengxing Xie
  - Chang Gao
  - Kai Chen
  - Dacheng Tao
published: "2025-10-15T12:51:47+00:00"
url: https://arxiv.org/abs/2510.13501v1
source: arxiv
doi: null
arxiv_id: 2510.13501v1
categories:
  - cs.AI
---

# Confidence as a Reward: Transforming LLMs into Reward Models

He Du Email: <elyndendu@gmail.com>    Bowen Li ^(†)^(†)thanks:
Corresponding author. Affiliation: Shanghai AI Laboratory
Email: <libowen.ne@gmail.com>    Chengxing Xie
Email: <chenkai@pjlab.org.cn>    Chang Gao Affiliation: The Chinese
University of Hong Kong    Kai Chen¹¹footnotemark: 1
Affiliation: Shanghai AI Laboratory    Dacheng Tao Affiliation: Nanyang
Technological University Affiliation: Fudan University
Affiliation: Xidian University

###### Abstract

Reward models enhance the reasoning capabilities of large language
models (LLMs) but typically require extensive curated data and costly
training, often bringing challenges. LLM-as-a-Judge offers a
training-free alternative by using LLMs’ intrinsic reasoning to evaluate
responses, but it still lags behind trained reward models. In this work,
we propose Confidence-as-a-Reward (CRew), a simple yet effective
training-free approach that uses a model’s token-level confidence in the
final answer as a reward proxy, particularly for close-ended problems.
Additionally, we introduce CRew-DPO, a training method that constructs
preference data from confidence scores and correctness signals. We
validate our approach through extensive experiments on mathematical
tasks. CRew achieves the best performance among training-free reward
models on MATH500 and RewardMATH, even surpassing most trained reward
models, highlighting its effectiveness as a reward proxy. Additionally,
we show a strong correlation between CRew evaluations and model
reasoning performance. Furthermore, CRew can be used as a data filtering
strategy by selecting high-quality training samples. Finetuning with
CRew-DPO further enhances judging capabilities and outperforms existing
self-training methods.

|     |
| --- |
|     |

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable performance on
complex reasoning tasks ([Guo et al., 2025](#bib.bib12); [OpenAI,
2024](#bib.bib25)), significantly advancing fields such as mathematical
problem-solving, logical inference ([Lightman et al., 2023](#bib.bib19);
[Tafjord et al., 2020](#bib.bib34)), and decision-making systems ([Gao
et al., 2023a](#bib.bib9)). This progress has sparked growing interest
in reward modeling, a critical component in refining LLM
capabilities ([Huang et al., 2023](#bib.bib15); [Xia et al.,
2024](#bib.bib39); [Setlur et al., 2024b](#bib.bib31)). Reward models
evaluate the quality of model-generated responses by typically
determining their preference over baseline solutions, playing a vital
role in reinforcement learning based training ([Schulman et al.,
2017](#bib.bib29); [Ouyang et al., 2022](#bib.bib26)). Beyond optimizing
policy updates, reward models also enable test-time scaling by selecting
the most promising solution among multiple generated candidates, thereby
enhancing overall model reliability and effectiveness ([Hosseini et al.,
2024](#bib.bib14); [Snell et al., 2024](#bib.bib33)).

Despite their utility, training reward models presents several
fundamental challenges ([Kim et al., 2024](#bib.bib16); [Lambert et al.,
2024](#bib.bib17)). First, developing a well-calibrated reward model
typically requires extensive curated datasets, which can be costly and
time-consuming to construct ([Lightman et al., 2023](#bib.bib19); [Wang
et al., 2023](#bib.bib36); [Xia et al., 2024](#bib.bib39)). Second,
training these models is computationally expensive and challenging, as
it requires the reward model to learn subtle reasoning patterns  ([Yuan
et al., 2024a](#bib.bib42); [Zhang et al., 2024a](#bib.bib44)). Third,
trained reward models may overfit to specific datasets, capturing
surface-level heuristics rather than true judging capabilities, failing
to generalize well beyond their training distribution ([Gao et al.,
2023b](#bib.bib10)). An alternative approach, LLM-as-a-Judge ([Li
et al., 2024](#bib.bib18); [Gu et al., 2024](#bib.bib11)), leverages
LLMs’ intrinsic generation capabilities to evaluate responses without
explicit reward model training. However, existing study suggests that
this method still underperforms compared to trained reward
models ([Zhang et al., 2024b](#bib.bib45)).

This raises a research question: Can LLMs themselves serve as reward
models in a training-free manner, and how well can they perform? In this
work, we introduce Confidence-as-a-Reward (CRew), a simple yet effective
approach that utilizes the model’s token-level confidence of the final
answer as a reward proxy, primarily for close-ended problems where the
final answer can be effortless identified and objectively verified. As
illustrated in the upper part of
Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
instead of relying on extensive datasets and additional training,
CRew directly extracts the probability of the final answer tokens and
computes their mean confidence score, harnessing the model’s inherent
reasoning capabilities to assess response quality. Beyond the reward
estimation, we propose a corresponding training method, CRew-DPO, to
further enhance the model’s judging capabilities. As shown in the lower
part of
Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
CRew-DPO constructs preference data by sampling data from the model
itself and leveraging both confidence scores and solution correctness.
This preference data is then used for DPO training, enabling the model
to refine its reward function without requiring external annotations.

We conduct comprehensive experiments on mathematical reasoning tasks, as
they are among the most important and representative benchmarks. Using
the same base model, Qwen2.5-7B-Instruct, CRew achieves the best
performance among training-free reward modeling approaches on MATH500
[Lightman et al. (2023)](#bib.bib19) when applied with reward-weighted
self-consistency and Best-of-N selection. It also significantly
outperforms those methods on the RewardMATH benchmark [Kim et al.
(2024)](#bib.bib16), demonstrating its effectiveness as a reward proxy.
Scaling to larger model variants (14B, 32B, and 72B), perhaps
surprisingly, CRew remains among the top-performing approaches on
RewardMATH across all model types, even surpassing most reward models
trained on massive datasets. Additionally, we observe a strong positive
correlation between a model’s mathematical reasoning performance on MATH
and its evaluation capability as measured by CRew on RewardMATH, a
finding we validate across the Llama-3 and Qwen2.5 model families.
Moreover, CRew can serve as a data filtering strategy, significantly
improving model finetuning by selecting high-quality training samples.
Beyond training-free evaluation, we finetune Qwen2.5-7B-Instruct using
CRew-DPO. CRew-DPO not only outperforms existing self-training
methods—including vanilla DPO, ReST$`{}^{\text{EM}}`$ ([Singh et al.,
2023](#bib.bib32)), and LLM-as-a-Judge on MATH500 and GSM8K, but also
achieves significantly higher gains on RewardMATH, further validating
the effectiveness of confidence-based self-supervised learning.

Our main contributions are as follows:

      • We introduce CRew, a novel confidence-based reward model for
training-free solution evaluation, along with CRew-DPO, a corresponding
training approach.

      • We demonstrate that CRew serves as an effective reward proxy for
mathematical reasoning tasks, showing a strong correlation with model
performance, and utility for data filtering.

      • We validate the effectiveness of CRew-DPO through training
experiments, showing significant improvements in evaluation capability
over existing self-training methods.

## 2 Related Work

ORM and PRM. The output generated by ORM based models typically focuses
solely on the final result when calculating rewards, without paying
attention to the finer details of the process ([Yuan et al.,
2024a](#bib.bib42); [Cai et al., 2024](#bib.bib3); [Liu et al.,
2024](#bib.bib22); [Dai et al., 2023](#bib.bib5); [Yang et al.,
2024](#bib.bib41); [Wang et al., 2024](#bib.bib35)). Generally, pairs
consisting of correct and incorrect reasoning data are constructed and
used for training with pairwise loss ([Bradley & Terry,
1952](#bib.bib2)). Our method essentially transforms a mathematical
model into an ORM with the confidence format.  [Lightman et al.
(2023)](#bib.bib19) suggests that PRM focuses on every step of the
reasoning process and assigns a reward value to each step. Training
often requires a large amount of manually labeled, fine-grained
supervisory data for each step ([Xia et al., 2024](#bib.bib39);
[o1 Team, 2024](#bib.bib24)). To address the heavy reliance on manually
labeled data,  [Wang et al. (2023)](#bib.bib36),  [Zhang et al.
(2024a)](#bib.bib44) and other works mimic reinforcement learning by
defining the reasoning process as a Markov Decision Process (MDP). This
approach allows the use of multiple rollout data to train PRM ([Xie
et al., 2024](#bib.bib40); [Feng et al., 2023](#bib.bib7)).

Generative Verifier.  [Zhang et al. (2024b)](#bib.bib45) uses the
probability of a “Yes” or “No” token indicating the correctness of a
solution as the reward value. This approach leverages the text
generation capabilities of pretrained large language models (LLMs)
through the CoT (Chain-of-Thought) in the critic process ([Wei et al.,
2022](#bib.bib38)).  [Wang & Zhou (2024)](#bib.bib37) focuses on the
process of sampling tokens, referring to the difference in probability
between the top-1 and top-2 choices in the final answer part as
confidence. It argues that the greater this confidence, the more it
indicates that the model has arrived at the answer through a series of
reasoning processes. Our method differs from the above methods by
directly using the probability of the final answer in the solution as
the reward. This allows us to obtain fine-grained rewards while
maintaining consistency with the reasoning process. For the model, it
can directly obtain the reward after reasoning, without the need for an
additional calculation process.

LLM-as-a-Judge. “LLM-as-a-Judge” is not actually a single method but
rather a broad concept. In a wider sense, any approach that utilizes the
inherent attributes and capabilities of LLMs falls under this category,
mainly based on prompt-based methods ([Fu et al., 2023](#bib.bib8); [Wei
et al., 2022](#bib.bib38); [Dong et al., 2024](#bib.bib6); [Lin & Chen,
2023](#bib.bib20); [Bai et al., 2022](#bib.bib1); [Ling et al.,
2024](#bib.bib21); [Zheng et al., 2023](#bib.bib46)). It relies on the
model’s judgment ability, which is gained through large-scale
pretraining.  [Yuan et al. (2024b)](#bib.bib43) demonstrates that such
rewards can, to some extent, reflect the quality of reasoning. However,
due to its training-free nature, it may encounter issues of instability
and difficulty in handling complex tasks ([Li et al., 2024](#bib.bib18);
[Gu et al., 2024](#bib.bib11)).

![Refer to caption](2510.13501v1/main_fig.png)

Figure 1: An Overview of Our Approach. The upper part illustrates how to
calculate the confidence reward of a given solution. The bottom part
describes how training on close-end tasks can lead to the
self-improvement of the model’s evaluation ability. For a given
question, the model outputs multiple solutions, each solution consists
of a rationale and an answer. The answers are extracted from all the
solutions to calculate the confidence. Finally, the top-K pairs with the
largest confidence difference among all correct and incorrect solution
pairs are selected for DPO training.

## 3 Methodology

### 3.1 CRew

Existing reward methods—both discriminative and generative—often
directly evaluate the entire solution, which can bring several issues.
For example, learning to evaluate the whole solution is relatively
difficult and usually requires more training. Additionally, key tokens
within the solution may be obscured, resulting in suboptimal reward
calculations.

To address the mentioned issue, our method introduces two improvements:

      • Focus on the final answer tokens: We only consider the key
tokens corresponding to the final answer, providing a more accurate
reward.

      • Align reward computation with the model’s reasoning process: The
reward calculation process is identical to the model’s reasoning
process, allowing for better utilization of the model’s capabilities,
which have been honed through extensive training.

#### What is confidence?

As shown in the upper part of
Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
to capture the most important information from the solution, we focus on
extracting tokens corresponding to final answers (e.g. content within
\boxed{} in MATH tasks) and compute their mean probability as
confidence. This confidence metric leverages the model’s inherent
reasoning capabilities through Chain-of-Thought generation, effectively
transforming its reasoning process into a quantifiable reward signal
that reflects solution certainty.

#### Why confidence can be used as a reward?

Let’s start by defining an existing model $`\pi`$, and for a given
question $`q`$ with gold solution $`s`$ containing the gold answer
$`a`$. This model generated solution $`\hat{s}`$ consists of two parts:
the rationale $`\hat{r}`$ and the final answer $`\hat{a}`$, so that
$`\hat{s}=<\hat{r},\hat{a}>`$. At the same time, we define a
verification function $`v`$, where $`v(\hat{s},q)=1`$ if the solution is
correct, and $`v(\hat{s},q)=0`$ otherwise.

Close-ended reasoning tasks’ characteristic implies that the probability
of a solution being correct is equal to the probability that the final
answer is the gold answer:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

       ``` math
       P(v(\hat{s},q)=1)=P(\hat{a}=a)
       ```                             |     | (1) |

However, the equivalence between a correct solution and the answer being
the gold answer is not sufficient to demonstrate that confidence is an
excellent reward. Therefore, we will next explain the conditions that a
reward needs to satisfy to be considered effective.

We define a reward function $`R`$ as a binary classification function,
where ideally for a solution $`s`$, if $`s`$ is correct, $`R(s)=1`$;
otherwise, $`R(s)=0`$. To approach this goal, one way is to ensure that
at least the following conditions are met:

For question $`q`$, suppose we have two reward functions, $`R_{1}`$ and
$`R_{2}`$, where $`R_{2}`$ is the better one, and define a set
$`S=\{S^{+},S^{-}\}`$ that contains all possible solutions. For all
possible correct solutions $`s^{+}\in S^{+}`$ and incorrect solutions
$`s^{-}\in S^{-}`$, we require:

|     |                                                                         |                     |     |     |
| --- | ----------------------------------------------------------------------- | ------------------- | --- | --- |
|     | $`\displaystyle\mathbb{E}_{s^{+}\in S^{+}}[R_{2}(s^{+})-R_{1}(s^{+})]`$ | $`\displaystyle>0`$ |     |     |
|     | $`\displaystyle\mathbb{E}_{s^{-}\in S^{-}}[R_{2}(s^{-})-R_{1}(s^{-})]`$ | $`\displaystyle<0`$ |     | (2) |

At the same time, we present a hypothesis involving two models,
$`\pi_{1}`$ and $`\pi_{2}`$ , where $`\pi_{2}`$ has stronger reasoning
capabilities than $`\pi_{1}`$ . The hypothesis is as follows:

|     |                                                                                       |                                                                                        |     |     |
| --- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --- | --- |
|     | $`\displaystyle\mathbb{E}_{\hat{s}_{2}\sim\pi_{2}}[P_{\pi_{2}}(v(\hat{s}_{2},q)=1)]`$ | $`\displaystyle>\mathbb{E}_{\hat{s}_{1}\sim\pi_{1}}[P_{\pi_{1}}(v(\hat{s}_{1},q)=1)]`$ |     |     |
|     | $`\displaystyle\mathbb{E}_{\hat{s}_{2}\sim\pi_{2}}[P_{\pi_{2}}(v(\hat{s}_{2},q)=0)]`$ | $`\displaystyle<\mathbb{E}_{\hat{s}_{1}\sim\pi_{1}}[P_{\pi_{1}}(v(\hat{s}_{1},q)=0)]`$ |     | (3) |

Therefore, by simply substituting confidence into Hypothesis
[3](#S3.Ex2 "In Why confidence can be used as a reward? ‣ 3.1 CRew ‣ 3 Methodology ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
we can easily derive Requirement
[2](#S3.Ex1 "In Why confidence can be used as a reward? ‣ 3.1 CRew ‣ 3 Methodology ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
which means that confidence can be a form of ideal reward function (The
complete derivation can be found in the
[A.4](#A1.SS4 "A.4 Derivation ‣ Appendix A Appendix ‣ Confidence as a Reward: Transforming LLMs into Reward Models")).
This also demonstrates that, there is a correlation between the model’s
reasoning ability and its evaluation capability.

### 3.2 CRew-DPO

Based on confidence, the new reward calculation method, we propose a
corresponding training approach, CRew-DPO. CRew-DPO  is a self-training
approach that does not require a large amount of manually labeled data,
yet significantly improves confidence performance.

Data Preparation. As shown in the bottom part of
Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
given a seed dataset consisting of a series of questions with gold
solutions. We first define the current model as $`\pi_{t}`$, and the
seed dataset with gold answers as $`D=\{(q_{i},s_{i})\}`$. The current
model samples $`N`$ different solutions for each question and computes
the confidence for each solution:

|     |     |     |
| --- | --- | --- |
|     |

````math
c_{i}^{n}=\pi_{t}(a_{i}^{n}|r_{i}^{n},q_{i})\quad n\in\{1,2,\dots,N\},
``` |  |

Finally we get a new dataset:

|  |  |  |
|----|----|----|
|  |
``` math
\hat{D}=\{(q_{i},\{s_{i}^{n},c_{i}^{n}\})\}\quad n\in\{1,2,\dots,N\}.
``` |  |

The solutions generated for all questions in $`\hat{D}`$ are classified
into two sets, $`\hat{D}^{chosen}`$ and $`\hat{D}^{rejected}`$ , based
on whether they match the gold answer. Specifically,
$`\hat{D}^{chosen}`$ contains solutions that are correct, and
$`\hat{D}^{rejected}`$ contains solutions that are incorrect. Then, for
each question, the solutions from $`\hat{D}^{chosen}`$ and
$`\hat{D}^{rejected}`$ are matched, forming a set that contains all
possible correct-incorrect solution pairs:

|  |  |  |
|----|----|----|
|  | $`\displaystyle\hat{D}^{pairs}=\{(q_{i},\{s_{i}^{m},c_{i}^{m}\}_{chosen},\{s_{i}^{n},c_{i}^{n}\}_{rejected})\},\quad m+n\in\{2,\dots,N\}.`$ |  |

Once $`\hat{D}^{pairs}`$ is obtained, we select the top-K pairs with the
largest confidence differences between the correct and incorrect
solutions. These pairs are then used to form the training set
$`\hat{D}^{train}`$ for subsequent preference optimization.

Preference Optimization. With the constructed pair training set
$`\hat{D}^{train}`$, the next stage is to improve the model’s evaluation
capability through preference optimization. We update the initial model
on $`\hat{D}^{train}`$ using DPO loss ([Rafailov et al.,
2024](#bib.bib28)), which results in the new model.

The reason for this training approach is to enable the model to better
associate the correctness of the solution with the level of confidence.
Specifically, a correct solution should correspond to a higher
confidence, while an incorrect solution should have a lower confidence.

## 4 Experiments with CRew

In this section, our experiments focus on the superiority of using
confidence as a reward, including its excellent evaluation performance,
strong correlation with the model’s inherent reasoning capabilities and
its ability to help filter training data.

### 4.1 Setup

#### Datasets

For the most important and representative reasoning task—mathematics, we
focus on two widely used datasets: the popular grade-school math test
dataset GSM8K ([Cobbe et al., 2021](#bib.bib4)) and the more challenging
MATH dataset ([Hendrycks et al., 2021](#bib.bib13)). For evaluating
mathematical reasoning tasks on MATH, we select the representative
MATH500 dataset ([Lightman et al., 2023](#bib.bib19)) from the same
source to achieve higher evaluation efficiency. Specifically, when
evaluating the performance of the reward model, in addition to using
indirect methods like majority voting based on rewards, we also use a
direct dataset called RewardMATH ([Kim et al., 2024](#bib.bib16)). This
dataset is built on MATH500 and includes 1 correct solution and 9
incorrect solutions for each question, randomly generated by 14 popular
open-source and closed-source models. The objective of RewardMATH is to
select the correct solution from a total of 10 solutions (for example,
please refer to the
Table [8](#A1.T8 "Table 8 ‣ A.3 Examples ‣ Appendix A Appendix ‣ Confidence as a Reward: Transforming LLMs into Reward Models")),
which directly reflects the reward model’s ability to assess
mathematical reasoning solutions.

### 4.2 CRew as a Strong Reward Proxy

One of the advantages of confidence is that it is a training-free
reward. Training-free reward means that for a solution generated by a
model, there is no need to train an additional reward model to obtain
the reward value for that solution.

In this section, we compare confidence with several different
training-free reward methods on MATH500 and RewardMATH datasets, such as
LLM-as-a-Judge, generative verifier, and perplexity. For LLM-as-a-Judge,
we follow the prompt from  [Yuan et al. (2024b)](#bib.bib43), asking the
model to score its own output. For the generative verifier
method ([Zhang et al., 2024b](#bib.bib45)), after the model generates a
solution, we ask the same model (which has not been trained on specific
evaluation tasks) whether the answer is correct, and then uses the
probability of the Yes or No token in the model’s response as the
reward. Similar to the calculation of confidence, perplexity—one of the
basic properties of model output—can also be used as a form of reward.

The two parts below
Table [1](#S4.T1 "Table 1 ‣ 4.2 CRew as a Strong Reward Proxy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models")
show the performance of all training-free reward methods on RewardMATH.
As seen, confidence outperforms the other methods, highlighting its
ability to distinguish between correct and incorrect solutions.
Table [2](#S4.T2 "Table 2 ‣ Figure 2 ‣ 4.4 CRew as a Data Filtering Stretegy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models")
presents the performance of the same model on MATH500 using
SC+Reward(where the normalized reward is used as the weight for voting)
and best-of-N approaches, under different training-free reward
approaches. The result also shows that confidence achieves the best
performance.

After our analysis, the LLM-as-a-Judge method, tends to assign higher
scores during evaluation, which impacts the evaluation process
negatively. Additionally, due to the integer scoring form, it cannot
distinguish between solutions with the same score, resulting in lower
evaluation accuracy. For the generative verifier method, it reduces the
granularity of the reward, but since evaluating solutions has not been
extensively trained for the model, its capability remains relatively
weak. As for perplexity, compared to confidence, experiments show that
the perplexity reward may include the probability of many irrelevant
tokens, which leads to a less accurate evaluation of the solution.

It is evident that, whether on MATH500 or RewardMATH, confidence proves
to be a more effective training-free reward, distinguishing between
correct and incorrect solutions more effectively. At the same time, when
evaluating the content generated by the model itself, there is no need
for the overhead of an additional reward calculation process.

|  |  |  |
|----|----|----|
| Reward Model |  | RewardMATH |
| Random |  | 10.00 |
| Trained Classifier-based Reward Models |  |  |
| [GRM-gemma-2B](https://huggingface.co/Ray2333/GRM-Gemma-2B-sftreg) |  | 4.97 |
| Qwen2.5-7B-Instruct-rm^(\$) |  | 6.83 |
| [Oasst-rm-2.1-pythia-1.4b](https://huggingface.co/OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5) |  | 7.04 |
| [Beaver-7b-v2.0-reward](https://huggingface.co/PKU-Alignment/beaver-7b-v2.0-reward) |  | 7.25 |
| [Eurus-RM-7b](https://huggingface.co/openbmb/Eurus-RM-7b) |  | 16.98 |
| [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) |  | 20.50 |
| [Skywork-Reward-Llama3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B) |  | 22.15 |
| [GRM-llama3-8B](https://huggingface.co/Ray2333/GRM-llama3-8B-sftreg) |  | 24.43 |
| [Internlm2-20b-reward](https://huggingface.co/internlm/internlm2-20b-reward) |  | 33.95 |
| [Internlm2-7b-reward](https://huggingface.co/internlm/internlm2-7b-reward) |  | 37.27 |
| Trained Process Reward Models (prod) |  |  |
| [Llemma-7b-prm-prm800k](https://huggingface.co/ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf) |  | 14.08 |
| [ReasonEval-34B](https://huggingface.co/GAIR/ReasonEval-34B) |  | 15.95 |
| [Math-Shepherd-Mistral-7B](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm) |  | 17.18 |
| [ReasonEval-7B](https://huggingface.co/GAIR/ReasonEval-7B) |  | 18.22 |
| Trained Process Reward Models (geo mean) |  |  |
| [Math-Shepherd-Mistral-7B](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm) |  | 15.74 |
| [Llemma-7b-prm-prm800k](https://huggingface.co/ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf) |  | 16.36 |
| [ReasonEval-34B](https://huggingface.co/GAIR/ReasonEval-34B) |  | 18.43 |
| [ReasonEval-7B](https://huggingface.co/GAIR/ReasonEval-7B) |  | 20.29 |
| Training-Free Reward |  |  |
| Perplexity |  | 2.07 |
| LLM-as-a-Judge |  | 13.25 |
| Generative Verifier |  | 16.56 |
| CRew (Ours) |  |  |
| Qwen2.5-14B-Instruct |  | 32.71 |
| Qwen2.5-32B-Instruct |  | 40.17 |
| Qwen2.5-72B-Instruct |  | 38.51 |
| Qwen2.5-7B-Instruct |  | 27.12 |
|      CRew-DPO Iteration 1 |  | 36.02 |
|      CRew-DPO Iteration 2 |  | 38.72 |

Table 1: The results of our confidence reward method compared to
classifier-based ORMs/PRMs reported in the  [Kim et al.
(2024)](#bib.bib16) and training-free reward methods (based on
Qwen2.5-7B-Instruct) on RewardMATH. $`\$`$: The ORM trained on the same
data sampled from the first iteration of CRew-DPO on MATH task, based on
Qwen2.5-7B-Instruct.

### 4.3 Stronger Models are Better Evaluators

To validate the conclusion presented in Section
[3](#S3 "3 Methodology ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
that a stronger reasoning ability of the model corresponds to a stronger
evaluation capability, in this section, we conduct correlation
experiment and demonstrate a strong correlation between the two
capability.

In the experiment, we select eight commonly used open-source models,
from the Llama-3 ([Meta AI, 2024](#bib.bib23)) and Qwen-2.5
families ([Qwen, 2024](#bib.bib27)). We use the scores of all models
under the official 4-shot prompt setting on the MATH testset ([Hendrycks
et al., 2021](#bib.bib13)) as an estimate of their mathematical
capabilities and then test their performance on RewardMATH to assess
their evaluation ability.
Figure [2](#S4.F2 "Figure 2 ‣ 4.4 CRew as a Data Filtering Stretegy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models")
illustrates the relationship between reasoning and evaluation. As shown,
the correlation coefficient between the two reaches 0.83, indicating a
strong correlation. The experiment demonstrates a strong correlation
between the model’s mathematical ability and evaluation capability, and
also means that if we already have a model with strong reasoning
capabilities, a powerful reward model can be achieved without additional
training.

### 4.4 CRew as a Data Filtering Stretegy

Figure 2: The correlation between mathematical reasoning ability and
evaluation ability. Using eight models with varying capabilities from
the Llama-3 and Qwen-2.5 families, the results demonstrate a strong
positive correlation, with a correlation coefficient reaches 0.83. This
demonstrates the consistency between mathematical reasoning ability and
mathematical evaluation ability.

|                     |           |      |     |
|---------------------|-----------|------|-----|
| Method              | MATH500   |      |     |
|                     | SC+Reward | BoN  |     |
| Self-consistency    | 78.2      | \-   |     |
| LLM-as-a-Judge      | 78.2      | 64.0 |     |
| Generative verifier | 78.8      | 67.4 |     |
| Perplexity          | 79.0      | 55.8 |     |
| Confidence          | 79.2      | 72.2 |     |



Table 2: The results of different training-free reward methods on
MATH500 with SC+Reward and Bon. We used Qwen2.5-7B-Instruct ([Qwen,
2024](#bib.bib27)). Specifically, on MATH500, we sample 16 solutions
each question with temperature of 1.0, using the normalized reward as
the weight for voting or choosing. All evaluations are conducted using
our zero-shot prompt, instructing the model to perform step-by-step
reasoning and format the final output accordingly.

In this section, we aim to explore whether there is a correlation
between confidence and data quality. To investigate this, we first
define high-quality data as data that is more beneficial for training a
weaker base model. When all other factors are kept constant, the base
model trained on higher-quality data should perform better in testing.

The experimental process is as follows: for the same dataset (where a
single question has multiple correct solutions), a separate confidence
reward model is used to calculate the confidence for all solutions. The
data with the highest and lowest confidence scores are selected, and the
original gold solution is used as the baseline. The base model is then
trained with the same settings using these three types of data.

We use Qwen2.5-7B-Instruct model to sample 30 solutions for each
question in the MATH training set, which forms the data source required
for our experiment. To eliminate randomness, we conduct three sets of
experiments using three different confidence reward models. Each set
followed the same experimental process, using the same dataset and
training the same base model.

Figure 3: The results of the Llama3-8B-Base model on MATH500 using the
data with the highest and lowest confidence (calculated by three
different critic models.) versus the original training set, all under
the same training settings.

Specifically, since the dataset in the experiment was sampled by
Qwen2.5-7B-Instruct, we selected an additional base model,
Llama3-8B-Base, to exclude potential issues arising from similar data
distributions.

As shown in
Figure [3](#S4.F3 "Figure 3 ‣ 4.4 CRew as a Data Filtering Stretegy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
the manually labeled original training set is difficult for the model to
learn effectively. This supports the conclusion in [Setlur et al.
(2024a)](#bib.bib30) that data sampled by models is easier to learn from
than manually labeled data. It is also noteworthy that models trained on
low-confidence data perform significantly better than those trained on
high-confidence data. Different models calculating confidence
consistently show this result, indicating that solutions that are
correct but where the model is less certain are more suitable for
learning. These solutions may contain more potential variability, making
them higher-quality data for the model.

## 5 Experiments with CRew-DPO

In this section, the goal of our experiments is to validate the
effectiveness of our training method, CRew-DPO. CRew-DPO is a
self-training approach (the model samples data to train itself) that
does not rely on large amounts of externally labeled data. (The training
details can be found in the
Section [A.2](#A1.SS2 "A.2 Training Details ‣ Appendix A Appendix ‣ Confidence as a Reward: Transforming LLMs into Reward Models").)
Compared to other self-training methods, our approach not only
outperforms them in terms of performance on mathematical tasks but also
significantly enhances the model’s evaluation capability.

### 5.1 Setup

#### Baselines

As comparison baselines, we compare other self-training approaches like
DPO, ReST$`{}^{\text{EM}}`$ ([Singh et al., 2023](#bib.bib32)) and
LLM-as-a-Judge ([Yuan et al., 2024b](#bib.bib43)):

      • DPO A correct and incorrect solution pair is randomly selected
from the model’s sampled results, and then DPO training is applied
directly.

      • ReST$`{}^{\text{EM}}`$ For each question, up to 10¹¹ 1 We follow
the original paper’s numerical settings. correct solutions are randomly
selected from the model’s sampled results. During each training
iteration, ReST$`{}^{\text{EM}}`$ performs SFT on the initial model.

      • LLM-as-a-Judge The core idea is to use the model score its own
sampled solutions, in order to obtain preference data, which is then
used for training with DPO.

|                        |         |            |       |            |
|------------------------|---------|------------|-------|------------|
| Task                   | MATH    |            | GSM8K |            |
| Dataset                | MATH500 | RewardMATH | GSM8K | RewardMATH |
| Qwen2.5-7B-Instruct    | 71.2    | 27.12      | 90.37 | 27.12      |
| DPO                    | 73.0    | 29.19      | 92.27 | 27.33      |
| ReST$`{}^{\text{EM}}`$ |         |            |       |            |
|        Iteration 1     | 74.8    | 28.78      | 92.42 | 26.09      |
|        Iteration 2     | 72.4    | 29.81      | 91.81 | 30.43      |
| LLM-as-a-Judge         |         |            |       |            |
|        Iteration 1     | 74.0    | 27.12      | 92.12 | 27.33      |
|        Iteration 2     | 72.4    | 27.12      | 92.12 | 28.57      |
| CRew-DPO               |         |            |       |            |
|        Iteration 1     | 75.0    | 36.02      | 92.42 | 30.43      |
|        Iteration 2     | 71.4    | 38.72      | 92.57 | 29.81      |

Table 3: The comparison of CRew-DPO with other self-training methods in
terms of evaluation ability and mathematical reasoning ability, tested
separately on the MATH and GSM8K tasks, along with the corresponding
RewardMATH scores.

### 5.2 A Significant Improvement in Evaluation Ability

As shown in
Table [3](#S5.T3 "Table 3 ‣ Baselines ‣ 5.1 Setup ‣ 5 Experiments with CRew-DPO ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
consistent with the conclusion validated in
Section [4.3](#S4.SS3 "4.3 Stronger Models are Better Evaluators ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
different self-training methods, while enhancing mathematical ability,
have also led to performance improvements on RewardMATH to some extent.
At the same time, CRew-DPO not only outperforms other baseline methods
on mathematical tasks but also achieves a significantly greater
improvement in evaluation ability. Across two iterations, our method
improves evaluation performance by 8.9% and 11.5%, respectively.

Especially in the last part of
Table [1](#S4.T1 "Table 1 ‣ 4.2 CRew as a Strong Reward Proxy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
it can be seen that after training with CRew-DPO, the 7B model’s
performance on RewardMATH reaches the level of the 72B model,
demonstrating that our approach targetedly enhances evaluation
capability.

Based on these observations, while ReST$`{}^{\text{EM}}`$ can improve
mathematical task performance, the lack of erroneous data during
training hinders the model’s ability to learn how to evaluate a solution
when the answer is unknown. At the same time, performing SFT on the
instruct model after RLHF ([Ouyang et al., 2022](#bib.bib26)) alignment
may, to some extent, degrade the evaluation capability that the model
originally gained through the alignment training. For DPO and
LLM-as-a-Judge, although these methods leverage incorrect data, they do
not use confidence as the selection signal. This fundamentally
introduces a distribution mismatch between the data construction phase
and the evaluation phase, resulting in limited performance improvements.

### 5.3 The Generalization of CRew-DPO

To verify whether CRew-DPO enhances evaluation capability by learning
the intrinsic mathematical ability, in this experiment, we use two
mathematically distinct datasets, MATH and GSM8K, with different
distributions. The experiment is divided into two main components:

Same Actor Model: We use Qwen2.5-7B-Instruct as the actor model. The
actor model will be tested on both MATH and GSM8K, guided by the reward
model.

Different reward models, includes:

      • Confidence reward models trained using our method on MATH or
GSM8K.

      • An untrained model (identical to the actor model) to verify
whether the trained reward model improves evaluation capability compared
to the untrained one.

When testing the model’s evaluation capability, we continue to use the
SC+Reward method described earlier.

As shown in
FIgure [4](#S5.F4 "Figure 4 ‣ 5.4 Comparison with Trained Reward Models ‣ 5 Experiments with CRew-DPO ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
first, whether in the MATH or GSM8K tasks, using the trained confidence
reward model to guide SC+Reward consistently outperforms using the actor
model itself as the reward model. This demonstrates that CRew-DPO indeed
leads to an improvement in evaluation capability. Furthermore, when the
training task of the confidence reward model differs from the task it
guides the actor’s reasoning on, the confidence reward model trained
with our method still brings performance improvements. This indicates
that the improvement in evaluation capability brought by
CRew-DPO generalizes across different mathematical tasks.

### 5.4 Comparison with Trained Reward Models

Figure 4: A comparison between the confidence reward model trained on
MATH, the confidence reward model trained on GSM8K, and the untrained
model itself. The two subplots on the left and right respectively show
the performance of the actor model on MATH and GSM8K under the guidance
of different reward models. The testing method here is the same as
described earlier, i.e., SC+Confidence, with a temperature of 1.0 and 16
solutions sampled for each question. Self confidence means using the
untrained actor itself to calculate the confidence.

Previous experiments have already shown that models with strong
mathematical abilities can also serve as excellent reward models under
the confidence reward. In this section, we further demonstrate the
significant enhancement of evaluation ability through confidence by
comparing it with mainstream trained ORMs and PRMs on RewardMATH.

As shown in
Table [1](#S4.T1 "Table 1 ‣ 4.2 CRew as a Strong Reward Proxy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
under the confidence format, even models that are not specifically
trained can perform as well as the top reward models. After two
iterations of training CRew-DPO, the model’s performance surpasses all
other reward models that trained on manually labeled datasets, such as
PRM800K  ([Lightman et al., 2023](#bib.bib19)), or synthetic data, like
 [Wang et al. (2023)](#bib.bib36), etc. This suggests that training a
reward model does not fully leverage the inherent mathematical abilities
of the original model.

Meanwhile, as shown in
Table [1](#S4.T1 "Table 1 ‣ 4.2 CRew as a Strong Reward Proxy ‣ 4 Experiments with CRew ‣ Confidence as a Reward: Transforming LLMs into Reward Models"),
using the same data constructed by our training experiment on the MATH
task, we trained a binary classifier ORM based on Qwen2.5-7B-Instruct.
It can be observed that training a new ORM on the same data does not
perform as well as the confidence-based approach. This indicates that
our approach is more efficient than direct training in enhancing
evaluation capabilities, and better transforms the model’s inherent
reasoning ability into evaluation ability.

## 6 Conclusion

In this paper, we propose a new reward method—CRew. For close-end tasks,
CRew employs the confidence of the final answer tokens within the
solution as the primary reward metric. Specifically, this confidence is
calculated as the mean probability of the tokens that constitute the
final answer. Beyond proposing this confidence-based reward mechanism,
we also present a corresponding training method, CRew-DPO, which
conducts DPO training based on confidence and correctness. Experimental
results show that confidence outperforms other reward methods, proving
to be an effective reward. It can utilize the model’s own reasoning
ability and is effective for data filtering. Moreover, the training
method we designed further significantly enhances the performance of
confidence.

## 7 Acknowledgement

This project is supported by Shanghai Artificial Intelligence
Laboratory. We are committed to advancing artificial intelligence
technology through cutting-edge research and practical applications,
contributing to technological progress. We are grateful for the valuable
support from Shanghai AI Lab, which enables us to continue exploring new
opportunities in the field of artificial intelligence and develop more
advanced solutions.

## References

- Bai et al. (2022) Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda
  Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia
  Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness
  from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022.
- Bradley & Terry (1952) Ralph Allan Bradley and Milton E Terry. Rank
  analysis of incomplete block designs: I. the method of paired
  comparisons. *Biometrika*, 39(3/4):324–345, 1952.
- Cai et al. (2024) Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen,
  Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al.
  Internlm2 technical report. *arXiv preprint arXiv:2403.17297*, 2024.
- Cobbe et al. (2021) Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
  Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek,
  Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
  math word problems, 2021. *URL https://arxiv.
  org/abs/2110.14168*, 2021.
- Dai et al. (2023) Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji,
  Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe
  reinforcement learning from human feedback. *arXiv preprint
  arXiv:2310.12773*, 2023.
- Dong et al. (2024) Yijiang River Dong, Tiancheng Hu, and Nigel
  Collier. Can llm be a personalized judge? *arXiv preprint
  arXiv:2406.11657*, 2024.
- Feng et al. (2023) Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus
  McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like
  tree-search can guide large language model decoding and training.
  *arXiv preprint arXiv:2309.17179*, 2023.
- Fu et al. (2023) Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei
  Liu. Gptscore: Evaluate as you desire. *arXiv preprint
  arXiv:2302.04166*, 2023.
- Gao et al. (2023a) Chang Gao, Haiyun Jiang, Deng Cai, Shuming Shi, and
  Wai Lam. Strategyllm: Large language models as strategy generators,
  executors, optimizers, and evaluators for problem solving. *arXiv
  preprint arXiv:2311.08803*, 2023a.
- Gao et al. (2023b) Leo Gao, John Schulman, and Jacob Hilton. Scaling
  laws for reward model overoptimization. In *International Conference
  on Machine Learning*, pp. 10835–10866. PMLR, 2023b.
- Gu et al. (2024) Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
  Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao
  Liu, et al. A survey on llm-as-a-judge. *arXiv preprint
  arXiv:2411.15594*, 2024.
- Guo et al. (2025) Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
  Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi,
  et al. Deepseek-r1: Incentivizing reasoning capability in llms via
  reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.
- Hendrycks et al. (2021) Dan Hendrycks, Collin Burns, Saurav Kadavath,
  Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt.
  Measuring mathematical problem solving with the math dataset. *arXiv
  preprint arXiv:2103.03874*, 2021.
- Hosseini et al. (2024) Arian Hosseini, Xingdi Yuan, Nikolay Malkin,
  Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-star:
  Training verifiers for self-taught reasoners. *arXiv preprint
  arXiv:2402.06457*, 2024.
- Huang et al. (2023) Jie Huang, Xinyun Chen, Swaroop Mishra,
  Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou.
  Large language models cannot self-correct reasoning yet. *arXiv
  preprint arXiv:2310.01798*, 2023.
- Kim et al. (2024) Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo
  Chae, Jungsoo Won, Dongha Lee, and Jinyoung Yeo. Evaluating robustness
  of reward models for mathematical reasoning. *arXiv preprint
  arXiv:2410.01729*, 2024.
- Lambert et al. (2024) Nathan Lambert, Valentina Pyatkin, Jacob
  Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri,
  Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating
  reward models for language modeling. *arXiv preprint
  arXiv:2403.13787*, 2024.
- Li et al. (2024) Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia
  Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. Llms-as-judges: A
  comprehensive survey on llm-based evaluation methods. *arXiv preprint
  arXiv:2412.05579*, 2024.
- Lightman et al. (2023) Hunter Lightman, Vineet Kosaraju, Yura Burda,
  Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya
  Sutskever, and Karl Cobbe. Let’s verify step by step. *arXiv preprint
  arXiv:2305.20050*, 2023.
- Lin & Chen (2023) Yen-Ting Lin and Yun-Nung Chen. Llm-eval: Unified
  multi-dimensional automatic evaluation for open-domain conversations
  with large language models. *arXiv preprint arXiv:2305.13711*, 2023.
- Ling et al. (2024) Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang,
  Mingu Lee, Roland Memisevic, and Hao Su. Deductive verification of
  chain-of-thought reasoning. *Advances in Neural Information Processing
  Systems*, 36, 2024.
- Liu et al. (2024) Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan,
  Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou.
  Skywork-reward: Bag of tricks for reward modeling in llms. *arXiv
  preprint arXiv:2410.18451*, 2024.
- Meta AI (2024) Meta AI. Meta llama 3.
  [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/), 2024.
  Accessed: 2024-05-22.
- o1 Team (2024) Skywork o1 Team. Skywork-o1 open series.
  [https://huggingface.co/Skywork](https://huggingface.co/Skywork),
  November 2024. URL
  [https://huggingface.co/Skywork](https://huggingface.co/Skywork).
- OpenAI (2024) OpenAI. Learning to reason with llms, 2024. URL
  [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/).
- Ouyang et al. (2022) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
  Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal,
  Katarina Slama, Alex Ray, et al. Training language models to follow
  instructions with human feedback. *Advances in neural information
  processing systems*, 35:27730–27744, 2022.
- Qwen (2024) Team Qwen. Qwen2.5: A party of foundation models,
  September 2024. URL
  [https://qwenlm.github.io/blog/qwen2.5/](https://qwenlm.github.io/blog/qwen2.5/).
- Rafailov et al. (2024) Rafael Rafailov, Archit Sharma, Eric Mitchell,
  Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct
  preference optimization: Your language model is secretly a reward
  model. *Advances in Neural Information Processing Systems*, 36, 2024.
- Schulman et al. (2017) John Schulman, Filip Wolski, Prafulla Dhariwal,
  Alec Radford, and Oleg Klimov. Proximal policy optimization
  algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- Setlur et al. (2024a) Amrith Setlur, Saurabh Garg, Xinyang Geng, Naman
  Garg, Virginia Smith, and Aviral Kumar. Rl on incorrect synthetic data
  scales the efficiency of llm math reasoning by eight-fold. *arXiv
  preprint arXiv:2406.14532*, 2024a.
- Setlur et al. (2024b) Amrith Setlur, Chirag Nagpal, Adam Fisch,
  Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal,
  Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling
  automated process verifiers for llm reasoning. *arXiv preprint
  arXiv:2410.08146*, 2024b.
- Singh et al. (2023) Avi Singh, John D Co-Reyes, Rishabh Agarwal,
  Ankesh Anand, Piyush Patil, Peter J Liu, James Harrison, Jaehoon Lee,
  Kelvin Xu, Aaron Parisi, et al. Beyond human data: Scaling
  self-training for problem-solving with language models. *arXiv
  preprint arXiv:2312.06585*, 2023.
- Snell et al. (2024) Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral
  Kumar. Scaling llm test-time compute optimally can be more effective
  than scaling model parameters. *arXiv preprint
  arXiv:2408.03314*, 2024.
- Tafjord et al. (2020) Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter
  Clark. Proofwriter: Generating implications, proofs, and abductive
  statements over natural language. *arXiv preprint
  arXiv:2012.13048*, 2020.
- Wang et al. (2024) Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao,
  and Tong Zhang. Interpretable preferences via multi-objective reward
  modeling and mixture-of-experts. In *EMNLP*, 2024.
- Wang et al. (2023) Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai,
  Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: A
  label-free step-by-step verifier for llms in mathematical reasoning.
  *arXiv preprint arXiv:2312.08935*, 2023.
- Wang & Zhou (2024) Xuezhi Wang and Denny Zhou. Chain-of-thought
  reasoning without prompting. *arXiv preprint arXiv:2402.10200*, 2024.
- Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
  Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought
  prompting elicits reasoning in large language models. *Advances in
  neural information processing systems*, 35:24824–24837, 2022.
- Xia et al. (2024) Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu,
  and Pengfei Liu. Evaluating mathematical reasoning beyond accuracy.
  *arXiv preprint arXiv:2404.05692*, 2024.
- Xie et al. (2024) Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan,
  Timothy P Lillicrap, Kenji Kawaguchi, and Michael Shieh. Monte carlo
  tree search boosts reasoning via iterative preference learning. *arXiv
  preprint arXiv:2405.00451*, 2024.
- Yang et al. (2024) Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and
  Tong Zhang. Regularizing hidden states enables learning generalizable
  reward model for llms. *arXiv preprint arXiv:2406.10216*, 2024.
- Yuan et al. (2024a) Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding,
  Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai
  Lin, et al. Advancing llm reasoning generalists with preference trees.
  *arXiv preprint arXiv:2404.02078*, 2024a.
- Yuan et al. (2024b) Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho,
  Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding
  language models. *arXiv preprint arXiv:2401.10020*, 2024b.
- Zhang et al. (2024a) Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue,
  Yuxiao Dong, and Jie Tang. Rest-mcts\*: Llm self-training via process
  reward guided tree search. *arXiv preprint arXiv:2406.03816*, 2024a.
- Zhang et al. (2024b) Lunjun Zhang, Arian Hosseini, Hritik Bansal,
  Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative
  verifiers: Reward modeling as next-token prediction. *arXiv preprint
  arXiv:2408.15240*, 2024b.
- Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
  Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li,
  Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot
  arena. *Advances in Neural Information Processing Systems*,
  36:46595–46623, 2023.

## Appendix A Appendix

### A.1 Prompt

[TABLE]

Table 4: Our zero-shot prompt

[TABLE]

Table 5: LLM-as-a-Judge prompt

[TABLE]

Table 6: generative verifier prompt

### A.2 Training Details

We select the Qwen2.5-7B-Instruct ([Qwen, 2024](#bib.bib27)) as initial
model, which has excellent instruction-following capabilities, making it
good for sampling data. During the solution sampling process, we use our
zero-shot prompt and instruct the model to perform step-by-step
reasoning, while also requiring the final answer to be enclosed in a
LaTeX-style \boxed{} to better extract and evaluate the model’s answer.
In the data construction phase, for each question, model samples 30
solutions with temperature of 1.0. After validating with the gold
answer, we pair all correct solutions with incorrect ones and select the
top-K pairs with the largest confidence gap (For the MATH task,
$`K=10`$; for the GSM8K task, $`K=15`$). After collecting the data, we
train using batch size of 128 and a learning rate of 5e-7 for 2 epochs
with DPO, saving a checkpoint every 50 steps. Ultimately, we select the
best checkpoint from all saved models based on performance on the
development set. As for the coefficient $`\beta`$ after testing, we find
that $`\beta=0.3`$ yielded the best performance.

### A.3 Examples

[TABLE]

Table 7: Confidence Example

[TABLE]

Table 8: Confidence Example

### A.4 Derivation

For the given model $`\pi`$, We define the probability that the solution
generated by model $`\pi`$ is correct as
$`P_{\pi}(\hat{s}=\text{True})`$, and the probability that the
corresponding answer is the gold answer as $`P_{\pi}(\hat{a}=a)`$. Thus,
we have:

|     |                                                    |     |     |
|-----|----------------------------------------------------|-----|-----|
|     |
       ``` math
       P_{\pi}(v(\hat{s},q)=1)=P(\hat{a}=a)\pi(\hat{s}|q)
       ```                                                 |     | (1) |

For a given question $`q`$, the probability of the model answering
correctly is:

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle\mathbb{E}_{\hat{s}\sim\pi}[P_{\pi}(v(\hat{s},q)=1)]=\mathbb{E}_{\hat{a}\sim\pi}[P(\hat{a}=a)\pi(\hat{s}|q)]=\mathbb{E}_{\hat{a}\sim\pi}[\mathbb{E}_{\hat{r}\sim\pi}[P(\hat{a}=a)\pi(\hat{a}|\hat{r},q)]]`$ |  | (2) |

Then, for a given question $`q`$, for all possible correct solutions
$`\hat{s}=<\hat{r},\hat{a}>\in S^{+}`$, we have:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
\displaystyle\mathbb{E}_{<\hat{r},\hat{a}>\in S^{+}}[\pi_{2}(\hat{a}|\hat{r},q)-\pi_{1}(\hat{a}|\hat{r},q)]
``` |  |  |
|  |
``` math
\displaystyle=\mathbb{E}_{<\hat{r},\hat{a}>\in S^{+}}[P(\hat{a}=a)\pi_{2}(\hat{a}|\hat{r},q)-P(\hat{a}=a)\pi_{1}(\hat{a}|\hat{r},q)]
``` |  |  |
|  |
``` math
\displaystyle=\mathbb{E}_{\hat{a}}[\mathbb{E}_{\hat{r}}[P(\hat{a}=a)\pi_{2}(\hat{a}|\hat{r},q)-P(\hat{a}=a)\pi_{1}(\hat{a}|\hat{r},q)]]
``` |  |  |
|  |
``` math
\displaystyle=\mathbb{E}_{\hat{a}}[\mathbb{E}_{\hat{r}}[P(\hat{a}=a)\pi_{2}(\hat{a}|\hat{r},q)]]-\mathbb{E}_{\hat{a}}[\mathbb{E}_{\hat{r}}[P(\hat{a}=a)\pi_{1}(\hat{a}|\hat{r},q)]]
``` |  |  |
|  |
``` math
\displaystyle=\mathbb{E}_{\hat{s}_{2}\in S}[P(v(\hat{s}_{2},q)=1))]-\mathbb{E}_{\hat{s}_{1}\in S}[P(v(\hat{s}_{1},q)=1))]
``` |  |  |
|  |
``` math
\displaystyle>0
``` |  | (3) |

The above derivation is based on Eq.
[2](#A1.E2 "In A.4 Derivation ‣ Appendix A Appendix ‣ Confidence as a Reward: Transforming LLMs into Reward Models")
and the fact that

|  |  |  |  |
|----|----|----|----|
|  | $`\displaystyle P(v(\hat{s},q)=1)=1,\quad\forall\hat{s}\in S^{+},`$ |  |  |
|  | $`\displaystyle P(v(\hat{s},q)=1)=0,\quad\forall\hat{s}\in S^{-}`$ |  | (4) |

Similarly, for all incorrect solutions $`\hat{s}=<\hat{r},\hat{a}>`$, we
have:

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\displaystyle\mathbb{E}_{<\hat{r},\hat{a}>\in S}[\pi_{2}(\hat{a}|\hat{r},q)-\pi_{1}(\hat{a}|\hat{r},q)]`$ | $`\displaystyle<0`$ |  | (5) |
````
