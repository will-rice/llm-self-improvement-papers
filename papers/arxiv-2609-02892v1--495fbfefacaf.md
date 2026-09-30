---
identifier: arxiv:2609.02892v1
title: Counterexamples as Feedback for Agent Self-Correction
authors:
  - Sidhesh Badrinarayan
  - Adithya Parthasarathy
published: "2026-07-01T06:08:15+00:00"
url: https://arxiv.org/abs/2609.02892v1
source: arxiv
doi: null
arxiv_id: 2609.02892v1
categories:
  - cs.AI
  - cs.CL
---

# COUNTEREXAMPLES AS FEEDBACK FOR AGENT SELF-CORRECTION

Sidhesh Badrinarayan\ad1    Adithya Parthasarathy\ad1 \add1Senior IEEE
Member

###### Abstract

Single-turn code-generation metrics understate a central property of
deployed agents: whether they can repair a wrong artifact after
receiving concrete feedback. This paper presents A-CEGIS, a lightweight
framework that uses counterexamples as feedback for evaluating
multi-turn refinement in natural-language-to-regex synthesis. An agent
proposes a regex, a deterministic oracle checks it under full-match
semantics, and compact false-positive or false-negative witnesses guide
the next turn. On 30 NL-RX-Turk tasks, diagnostic counterexample
feedback solves 90% of tasks within a four-turn ablation budget,
compared with 17% for zero-shot generation, 27% for generic
self-correction, and 23% for error-only feedback. In a full diagnostic
run with hardening, all tasks are solved on the hidden set by the final
turn, with mean time-to-success of 2.7 turns and robust success of 77%
after targeted probing. These results show that A-CEGIS measures how
efficiently an agent improves across turns while adding a practical
robustness check beyond the original held-out cases.

###### keywords:

multi-turn agents, agent evaluation, regular expression synthesis, LLM
self-correction, LLM benchmarking

## 1 Introduction

Software agents are rarely useful only on their first attempt: they must
use feedback, repair wrong artifacts, and avoid breaking behaviour that
was already correct. Standard one-shot metrics such as pass@1 remain
useful, but they do not measure repair efficiency, stability across
revisions, or whether an apparently solved task remains robust under
harder tests.

We study this problem in natural-language-to-regex synthesis, where
artifacts are compact, brittle, and executable. A small operator change
can alter the accepted language, while correctness can be checked
deterministically against positive and negative strings. This makes
regex synthesis a controlled setting for isolating refinement behaviour
without mixing it with retrieval, interface use, or other
agent-environment effects.

This paper introduces A-CEGIS, a counterexample feedback loop for
measuring agent self-correction. The agent proposes a regex, the oracle
evaluates it with re.fullmatch semantics, and the next prompt receives
concrete false-positive and false-negative witnesses rather than generic
criticism. On 30 NL-RX-Turk tasks, diagnostic A-CEGIS reaches 0.900
hidden-set success within a four-turn ablation budget, compared with
0.167 for zero-shot generation, 0.267 for generic self-correction, and
0.233 for error-only feedback. With a longer diagnostic budget and
targeted hardening, hidden-set success reaches 1.0 and robust success is
0.767. The contributions are a reproducible CEGIS-style loop, trajectory
metrics for convergence and regressions, direct feedback-strategy
ablations, and a robustness probe that exposes failures hidden by
endpoint accuracy.

## 2 Background and Related Work

LLM evaluation has historically emphasised first-attempt quality, exact
match, and single-step task completion. Code-generation benchmarks such
as HumanEval and MBPP established functional correctness as a central
measure for synthesis [b8](#bib.bib8) ; [b9](#bib.bib9) , while
SWE-bench moves evaluation toward realistic software repair
[b12](#bib.bib12) . Recent work argues that multi-turn agents also
require measures of context retention, interaction quality, and error
recovery [b1](#bib.bib1) ; coding benchmarks similarly show that
performance can degrade in multi-turn settings [b2](#bib.bib2) . These
results motivate process-level evaluation rather than endpoint scoring
alone.

Self-refinement methods show that models can improve outputs through
repeated feedback and revision [b3](#bib.bib3) ; [b11](#bib.bib11) , and
ReAct demonstrates the value of interleaving reasoning with environment
actions [b10](#bib.bib10) . However, natural-language feedback can be
noisy or underspecified. AgentBoard and related benchmarks broaden
evaluation to tool-mediated environments [b4](#bib.bib4) , but such
settings can mix repair skill with retrieval, action selection, and
interface effects. A-CEGIS asks a narrower complementary question: given
deterministic evidence of failure, can an agent repair a compact formal
artifact efficiently?

Natural-language-to-regex synthesis fits this question well. NL-RX maps
descriptions to regular expressions [b5](#bib.bib5) , while
programming-by-example and CEGIS work show the value of executable
examples and counterexample-guided revision [b14](#bib.bib14) ;
[b6](#bib.bib6) ; [b13](#bib.bib13) . A-CEGIS adapts that interaction
pattern to language-agent evaluation. It is not a full equivalence
proof, since the verifier uses finite tests and targeted probes, but
every turn is grounded in executable semantics. This lets us measure how
many turns are needed, whether repairs are local, whether revisions
introduce regressions, and whether a hidden-set solution survives
additional probing.

## 3 Methodology

The evaluation harness uses deterministic Python regex evaluation,
synthetic task-local tests, and an LLM interface. A task is
$`(d_{i},g_{i})`$, where $`d_{i}`$ is a natural-language description and
$`g_{i}`$ is a gold Python-compatible regex. For each task, the oracle
builds

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       T_{i}=P_{i}\cup N_{i},
       ```                     |     |

with 12 generated positives and 12 generated negatives in the reported
configuration. Positives are sampled from the gold regex using a
lightweight parser-driven generator; negatives come from mutations and
random samples retained only when rejected by $`g_{i}`$.

### 3.1 Oracle construction

The gold regex is used only as an executable specification. Positive
examples are generated by recursively traversing the parsed expression
when possible, including literals, branches, character classes,
subpatterns, and bounded repetitions. Negative examples are generated
from positive seeds by deletion, insertion, substitution, swapping,
truncation, and random sampling over a printable alphabet, following the
intuition that small edit operations often expose boundary behaviour
[b7](#bib.bib7) . A sampled negative is retained only if it fails the
gold regex under full-string matching.

This construction has two practical consequences. First, the benchmark
can be run without an external theorem prover or automata library.
Secondly, the hidden set is task-local: failures tend to occur near the
semantic material of the target regex rather than in an unrelated global
string distribution. The trade-off is that hidden-set success is not
semantic equivalence. That limitation is deliberate, because it creates
room to measure the difference between sampled success and robustness
under targeted probing.

![](data:image/svg+xml;base64,PHN2ZyBpZD0iUzMuRjEuMS4xLnBpYzEiIGNsYXNzPSJsdHhfcGljdHVyZSIgaGVpZ2h0PSIyMDYuODUiIG92ZXJmbG93PSJ2aXNpYmxlIiB2ZXJzaW9uPSIxLjEiIHZpZXdib3g9IjAgMCA0NjQuMDUgMjA2Ljg1IiB3aWR0aD0iNDY0LjA1Ij48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDIwNi44NSkgbWF0cml4KDEgMCAwIC0xIDAgMCkgdHJhbnNsYXRlKDQ5LjUyLDApIHRyYW5zbGF0ZSgwLDE5Mi43NSkiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTstLWx0eC1maWxsLWNvbG9yOiNEQkVBRkU7IiBzdHJva2U9IiMzNzQxNTEiIGZpbGw9IiNEQkVBRkUiIHN0cm9rZS13aWR0aD0iMC40NXB0Ij48cGF0aCBkPSJNIDQzLjY4IDEzLjc4IEwgLTQzLjY4IDEzLjc4IEMgLTQ2LjczIDEzLjc4IC00OS4yMSAxMS4zIC00OS4yMSA4LjI0IEwgLTQ5LjIxIC04LjI0IEMgLTQ5LjIxIC0xMS4zIC00Ni43MyAtMTMuNzggLTQzLjY4IC0xMy43OCBMIDQzLjY4IC0xMy43OCBDIDQ2LjczIC0xMy43OCA0OS4yMSAtMTEuMyA0OS4yMSAtOC4yNCBMIDQ5LjIxIDguMjQgQyA0OS4yMSAxMS4zIDQ2LjczIDEzLjc4IDQzLjY4IDEzLjc4IFogTSAtNDkuMjEgLTEzLjc4IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgc3Ryb2tlLXdpZHRoPSIwLjQ1cHQiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIC0zMi43NSAtNS43OCkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMy40NSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA2LjczKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMjEuMzkgMCkiPjx0ZXh0IHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMCkiPlRhc2s8L3RleHQ+PC9nPjwvZz48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfcm93IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDAgMTMuNDYpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6NS43OGVtOy0tbHR4LWZvLWhlaWdodDowLjU5ZW07LS1sdHgtZm8tZGVwdGg6MC4xN2VtOyIgd2lkdGg9IjY1LjUiIGhlaWdodD0iOC42MSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCA2LjczKSIgb3ZlcmZsb3c9InZpc2libGUiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iUzMuRjEuMS4xLnBpYzEuNC40LjQuMi4yLjIuMi4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOyI+ZGVzY3JpcHRpb24gPC9zcGFuPjxtYXRoIGlkPSJTMy5GMS4xLjEucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iZF97aX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgbWF0aHNpemU9IjAuNzAwZW0iPmQ8L21pPjxtaSBtYXRoc2l6ZT0iMC43MDBlbSI+aTwvbWk+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+ZF97aX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7LS1sdHgtZmlsbC1jb2xvcjojRENGQ0U3OyIgc3Ryb2tlPSIjMzc0MTUxIiBmaWxsPSIjRENGQ0U3IiBzdHJva2Utd2lkdGg9IjAuNDVwdCI+PHBhdGggZD0iTSAxNzAuMjggMTMuNzggTCA4Mi45MyAxMy43OCBDIDc5Ljg3IDEzLjc4IDc3LjM5IDExLjMgNzcuMzkgOC4yNCBMIDc3LjM5IC04LjI0IEMgNzcuMzkgLTExLjMgNzkuODcgLTEzLjc4IDgyLjkzIC0xMy43OCBMIDE3MC4yOCAtMTMuNzggQyAxNzMuMzQgLTEzLjc4IDE3NS44MiAtMTEuMyAxNzUuODIgLTguMjQgTCAxNzUuODIgOC4yNCBDIDE3NS44MiAxMS4zIDE3My4zNCAxMy43OCAxNzAuMjggMTMuNzggWiBNIDc3LjM5IC0xMy43OCIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIHN0cm9rZS13aWR0aD0iMC40NXB0IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA5OS42NyAtNS4zOSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMi42NykiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA2LjYyKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMTIuNTYgMCkiPjx0ZXh0IHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMCkiPkFnZW50PC90ZXh0PjwvZz48L2c+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X3JvdyIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAwIDEyLjY3KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjQuNzJlbTstLWx0eC1mby1oZWlnaHQ6MC4zN2VtOy0tbHR4LWZvLWRlcHRoOjAuMTdlbTsiIHdpZHRoPSI1My41NyIgaGVpZ2h0PSI2LjA1IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDQuMTcpIiBvdmVyZmxvdz0idmlzaWJsZSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTMy5GMS4xLjEucGljMS41LjUuNS4yLjIuMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7Ij5wcm9wb3NlcyA8L3NwYW4+PG1hdGggaWQ9IlMzLkYxLjEuMS5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMSIgY2xhc3M9Imx0eF9NYXRoIiBhbHR0ZXh0PSJyX3t0fSIgZGlzcGxheT0iaW5saW5lIiBpbnRlbnQ9IjpsaXRlcmFsIj48c2VtYW50aWNzPjxtc3ViPjxtaSBtYXRoc2l6ZT0iMC43MDBlbSI+cjwvbWk+PG1pIG1hdGhzaXplPSIwLjcwMGVtIj50PC9taT48L21zdWI+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4Ij5yX3t0fTwvYW5ub3RhdGlvbj48L3NlbWFudGljcz48L21hdGg+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTstLWx0eC1maWxsLWNvbG9yOiNGRUYzQzc7IiBzdHJva2U9IiMzNzQxNTEiIGZpbGw9IiNGRUYzQzciIHN0cm9rZS13aWR0aD0iMC40NXB0Ij48cGF0aCBkPSJNIDI5Ni44OSAxMy43OCBMIDIwOS41NCAxMy43OCBDIDIwNi40OCAxMy43OCAyMDQgMTEuMyAyMDQgOC4yNCBMIDIwNCAtOC4yNCBDIDIwNCAtMTEuMyAyMDYuNDggLTEzLjc4IDIwOS41NCAtMTMuNzggTCAyOTYuODkgLTEzLjc4IEMgMjk5Ljk1IC0xMy43OCAzMDIuNDMgLTExLjMgMzAyLjQzIC04LjI0IEwgMzAyLjQzIDguMjQgQyAzMDIuNDMgMTEuMyAyOTkuOTUgMTMuNzggMjk2Ljg5IDEzLjc4IFogTSAyMDQgLTEzLjc4IiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgc3Ryb2tlLXdpZHRoPSIwLjQ1cHQiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDIzMC41OSAtNi4wMykiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMy40NSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA2LjczKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgNi45NyAwKSI+PHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+T3JhY2xlPC90ZXh0PjwvZz48L2c+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X3JvdyIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAwIDEzLjQ2KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjQuMDJlbTstLWx0eC1mby1oZWlnaHQ6MC41OWVtOy0tbHR4LWZvLWRlcHRoOjAuMTJlbTsiIHdpZHRoPSI0NS41NSIgaGVpZ2h0PSI4LjExIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDYuNzMpIiBvdmVyZmxvdz0idmlzaWJsZSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTMy5GMS4xLjEucGljMS42LjYuNi4yLjIuMi4yLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7Ij5jaGVja3MgPC9zcGFuPjxtYXRoIGlkPSJTMy5GMS4xLjEucGljMS4zLjMuMy4zLjMuMy4zLjMuMy4zLjMuMy4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEubTEiIGNsYXNzPSJsdHhfTWF0aCIgYWx0dGV4dD0iVF97aX0iIGRpc3BsYXk9ImlubGluZSIgaW50ZW50PSI6bGl0ZXJhbCI+PHNlbWFudGljcz48bXN1Yj48bWkgbWF0aHNpemU9IjAuNzAwZW0iPlQ8L21pPjxtaSBtYXRoc2l6ZT0iMC43MDBlbSI+aTwvbWk+PC9tc3ViPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LXRleCI+VF97aX08L2Fubm90YXRpb24+PC9zZW1hbnRpY3M+PC9tYXRoPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7LS1sdHgtZmlsbC1jb2xvcjojRkVGM0M3OyIgc3Ryb2tlPSIjMzc0MTUxIiBmaWxsPSIjRkVGM0M3IiBzdHJva2Utd2lkdGg9IjAuNDVwdCI+PHBhdGggZD0iTSAyODUuMzggLTU4LjExIEwgMjUzLjIxIC00Mi4wOSBMIDIyMS4wNSAtNTguMTEgTCAyNTMuMjEgLTc0LjEzIFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiBzdHJva2U9IiMwMDAwMDAiIGZpbGw9IiMwMDAwMDAiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTsiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIwLjQ1cHQiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNGRUYzQzc7IiBmaWxsPSIjRkVGM0M3Ij48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MGVtOy0tbHR4LWZvLWhlaWdodDowZW07LS1sdHgtZm8tZGVwdGg6MGVtOyIgd2lkdGg9IjAiIGhlaWdodD0iMCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSIgb3ZlcmZsb3c9InZpc2libGUiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iUzMuRjEuMS4xLnBpYzEuNy43LjcuNC4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0Ij48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNTA2LjQzIC0xMzIuMjMpIiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxMy40NSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA2LjczKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgNi45MiAwKSI+PHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+YWxsPC90ZXh0PjwvZz48L2c+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X3JvdyIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAwIDEzLjQ2KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+cGFzcz88L3RleHQ+PC9nPjwvZz48L2c+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7LS1sdHgtZmlsbC1jb2xvcjojRkZFNEU2OyIgc3Ryb2tlPSIjMzc0MTUxIiBmaWxsPSIjRkZFNEU2IiBzdHJva2Utd2lkdGg9IjAuNDVwdCI+PHBhdGggZD0iTSAxNzAuMjggLTQxLjk2IEwgODIuOTMgLTQxLjk2IEMgNzkuODcgLTQxLjk2IDc3LjM5IC00NC40NCA3Ny4zOSAtNDcuNSBMIDc3LjM5IC02My45OSBDIDc3LjM5IC02Ny4wNCA3OS44NyAtNjkuNTIgODIuOTMgLTY5LjUyIEwgMTcwLjI4IC02OS41MiBDIDE3My4zNCAtNjkuNTIgMTc1LjgyIC02Ny4wNCAxNzUuODIgLTYzLjk5IEwgMTc1LjgyIC00Ny41IEMgMTc1LjgyIC00NC40NCAxNzMuMzQgLTQxLjk2IDE3MC4yOCAtNDEuOTYgWiBNIDc3LjM5IC02OS41MiIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIHN0cm9rZS13aWR0aD0iMC40NXB0IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCA4Ni43IC02My4wMSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNi40MSkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA3LjI2KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMy42MyAwKSI+PHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+RmFsc2UgcG9zLi9uZWcuPC90ZXh0PjwvZz48L2c+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X3JvdyIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAwIDE2LjQyKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+Y291bnRlcmV4YW1wbGVzPC90ZXh0PjwvZz48L2c+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7LS1sdHgtZmlsbC1jb2xvcjojRURFOUZFOyIgc3Ryb2tlPSIjMzc0MTUxIiBmaWxsPSIjRURFOUZFIiBzdHJva2Utd2lkdGg9IjAuNDVwdCI+PHBhdGggZD0iTSAyOTYuODkgLTEwMi40NCBMIDIwOS41NCAtMTAyLjQ0IEMgMjA2LjQ4IC0xMDIuNDQgMjA0IC0xMDQuOTEgMjA0IC0xMDcuOTcgTCAyMDQgLTEyNC40NiBDIDIwNCAtMTI3LjUyIDIwNi40OCAtMTMwIDIwOS41NCAtMTMwIEwgMjk2Ljg5IC0xMzAgQyAyOTkuOTUgLTEzMCAzMDIuNDMgLTEyNy41MiAzMDIuNDMgLTEyNC40NiBMIDMwMi40MyAtMTA3Ljk3IEMgMzAyLjQzIC0xMDQuOTEgMjk5Ljk1IC0xMDIuNDQgMjk2Ljg5IC0xMDIuNDQgWiBNIDIwNCAtMTMwIiAvPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMwMDAwMDA7LS1sdHgtZmlsbC1jb2xvcjojMDAwMDAwOyIgc3Ryb2tlLXdpZHRoPSIwLjQ1cHQiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDIxMS41OSAtMTIyLjk0KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDE1LjM0KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X3JvdyIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAwIDYuNzMpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAyMC4zIDApIj48dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj5UYXJnZXRlZDwvdGV4dD48L2c+PC9nPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCAxNS4zNCkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9jb2wgbHR4X25vcGFkX2wgbHR4X25vcGFkX3IiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMCkiPjx0ZXh0IHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMCkiPmhhcmRlbmluZyBwcm9iZXM8L3RleHQ+PC9nPjwvZz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTstLWx0eC1maWxsLWNvbG9yOiNFREU5RkU7IiBzdHJva2U9IiMzNzQxNTEiIGZpbGw9IiNFREU5RkUiIHN0cm9rZS13aWR0aD0iMC40NXB0Ij48cGF0aCBkPSJNIDI4Ny40OCAtMTc1LjM3IEwgMjUzLjIxIC0xNTguMzEgTCAyMTguOTUgLTE3NS4zNyBMIDI1My4yMSAtMTkyLjQ0IFoiIC8+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTsiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIwLjQ1cHQiPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiNFREU5RkU7IiBmaWxsPSIjRURFOUZFIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MGVtOy0tbHR4LWZvLWhlaWdodDowZW07LS1sdHgtZm8tZGVwdGg6MGVtOyIgd2lkdGg9IjAiIGhlaWdodD0iMCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSIgb3ZlcmZsb3c9InZpc2libGUiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250YWluZXIiPjxzcGFuIGNsYXNzPSJsdHhfZm9yZWlnbm9iamVjdF9jb250ZW50Ij48c3BhbiBpZD0iUzMuRjEuMS4xLnBpYzEuOC44LjguNS4xLjEuMSIgY2xhc3M9Imx0eF90ZXh0Ij48L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgNTA2LjQzIC0zNjcuODIpIiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeCIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNS4zNCkiPjxnIGNsYXNzPSJsdHhfdGlrem1hdHJpeF9yb3ciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIDEgMCA2LjczKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X2NvbCBsdHhfbm9wYWRfbCBsdHhfbm9wYWRfciIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMS4yNyAwKSI+PHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAwKSI+cHJvYmU8L3RleHQ+PC9nPjwvZz48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfcm93IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDAgMTUuMzQpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj48dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj5jbGVhbj88L3RleHQ+PC9nPjwvZz48L2c+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7LS1sdHgtZmlsbC1jb2xvcjojQ0NGQkYxOyIgc3Ryb2tlPSIjMzc0MTUxIiBmaWxsPSIjQ0NGQkYxIiBzdHJva2Utd2lkdGg9IjAuNDVwdCI+PHBhdGggZD0iTSA0MDguNjggLTE2MS42IEwgMzIxLjMzIC0xNjEuNiBDIDMxOC4yNyAtMTYxLjYgMzE1Ljc5IC0xNjQuMDcgMzE1Ljc5IC0xNjcuMTMgTCAzMTUuNzkgLTE4My42MiBDIDMxNS43OSAtMTg2LjY4IDMxOC4yNyAtMTg5LjE1IDMyMS4zMyAtMTg5LjE1IEwgNDA4LjY4IC0xODkuMTUgQyA0MTEuNzQgLTE4OS4xNSA0MTQuMjIgLTE4Ni42OCA0MTQuMjIgLTE4My42MiBMIDQxNC4yMiAtMTY3LjEzIEMgNDE0LjIyIC0xNjQuMDcgNDExLjc0IC0xNjEuNiA0MDguNjggLTE2MS42IFogTSAzMTUuNzkgLTE4OS4xNSIgLz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIHN0cm9rZS13aWR0aD0iMC40NXB0IiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAzNDUuODkgLTE4Mi4xKSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDEzLjQ1KSI+PGcgY2xhc3M9Imx0eF90aWt6bWF0cml4X3JvdyIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAwIDYuNzMpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAxLjg3IDApIj48dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj5Sb2J1c3Q8L3RleHQ+PC9nPjwvZz48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfcm93IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDAgMTMuNDYpIj48ZyBjbGFzcz0ibHR4X3Rpa3ptYXRyaXhfY29sIGx0eF9ub3BhZF9sIGx0eF9ub3BhZF9yIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj48dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDApIj5zb2x1dGlvbjwvdGV4dD48L2c+PC9nPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMzc0MTUxOyIgc3Ryb2tlLXdpZHRoPSIwLjhwdCIgc3Ryb2tlPSIjMzc0MTUxIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDQ5LjUyIDAgTCA2OC41IDAiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzM3NDE1MTsiIGZpbGw9IiMzNzQxNTEiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDY4LjUgMCkiPjxwYXRoIGQ9Ik0gNi4zNyAwIEMgNS41OCAwLjE5IDIuMTUgMS4yOCAwIDIuNDYgTCAwIC0yLjQ2IEMgMi4xNSAtMS4yOCA1LjU4IC0wLjE5IDYuMzcgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMzc0MTUxOyIgc3Ryb2tlLXdpZHRoPSIwLjhwdCIgc3Ryb2tlPSIjMzc0MTUxIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDE3Ni4xMyAwIEwgMTk1LjExIDAiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzM3NDE1MTsiIGZpbGw9IiMzNzQxNTEiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDE5NS4xMSAwKSI+PHBhdGggZD0iTSA2LjM3IDAgQyA1LjU4IDAuMTkgMi4xNSAxLjI4IDAgMi40NiBMIDAgLTIuNDYgQyAyLjE1IC0xLjI4IDUuNTggLTAuMTkgNi4zNyAwIFoiIC8+PC9nPjwvZz48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7IiBzdHJva2Utd2lkdGg9IjAuOHB0IiBzdHJva2U9IiMzNzQxNTEiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gMjUzLjIxIC0xNC4wOSBMIDI1My4yMSAtMzMuMDciIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzM3NDE1MTsiIGZpbGw9IiMzNzQxNTEiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMC4wIC0xLjAgMS4wIDAuMCAyNTMuMjEgLTMzLjA3KSI+PHBhdGggZD0iTSA2LjM3IDAgQyA1LjU4IDAuMTkgMi4xNSAxLjI4IDAgMi40NiBMIDAgLTIuNDYgQyAyLjE1IC0xLjI4IDUuNTggLTAuMTkgNi4zNyAwIFoiIC8+PC9nPjwvZz48ZyBzdHJva2Utd2lkdGg9IjAuOHB0Ij48ZyBzdHlsZT0iLS1sdHgtc3Ryb2tlLWNvbG9yOiMzNzQxNTE7IiBzdHJva2U9IiMzNzQxNTEiPjxwYXRoIHN0eWxlPSJmaWxsOm5vbmUiIGQ9Ik0gMjIxLjc3IC01Ny41MiBMIDE4NC43MSAtNTYuODMiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzM3NDE1MTsiIGZpbGw9IiMzNzQxNTEiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoLTAuOTk5ODIgMC4wMTg3IC0wLjAxODcgLTAuOTk5ODIgMTg0LjcxIC01Ni44MykiPjxwYXRoIGQ9Ik0gNi4zNyAwIEMgNS41OCAwLjE5IDIuMTUgMS4yOCAwIDIuNDYgTCAwIC0yLjQ2IEMgMi4xNSAtMS4yOCA1LjU4IC0wLjE5IDYuMzcgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxOTMuMTQgLTUxLjkzKSIgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIj48Zm9yZWlnbm9iamVjdCBzdHlsZT0iLS1sdHgtZm8td2lkdGg6MS4wNWVtOy0tbHR4LWZvLWhlaWdodDowLjM4ZW07LS1sdHgtZm8tZGVwdGg6MGVtOyIgd2lkdGg9IjExLjYzIiBoZWlnaHQ9IjQuMTciIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgNC4xNykiIG92ZXJmbG93PSJ2aXNpYmxlIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGFpbmVyIj48c3BhbiBjbGFzcz0ibHR4X2ZvcmVpZ25vYmplY3RfY29udGVudCI+PHNwYW4gaWQ9IlMzLkYxLjEuMS5waWMxLjkuOS45LjYuMS4xLjEiIGNsYXNzPSJsdHhfdGV4dCIgc3R5bGU9ImZvbnQtc2l6ZTo3MCU7Ij5ubzwvc3Bhbj48L3NwYW4+PC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTsiIHN0cm9rZS13aWR0aD0iMC44cHQiIHN0cm9rZT0iIzM3NDE1MSI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAxMjYuNjEgLTQxLjY1IEwgMTI2LjYxIC0yMi42NyIgLz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMzc0MTUxOyIgZmlsbD0iIzM3NDE1MSIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgwLjAgMS4wIC0xLjAgMC4wIDEyNi42MSAtMjIuNjcpIj48cGF0aCBkPSJNIDYuMzcgMCBDIDUuNTggMC4xOSAyLjE1IDEuMjggMCAyLjQ2IEwgMCAtMi40NiBDIDIuMTUgLTEuMjggNS41OCAtMC4xOSA2LjM3IDAgWiIgLz48L2c+PC9nPjxnIHN0cm9rZS13aWR0aD0iMC44cHQiPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzM3NDE1MTsiIHN0cm9rZT0iIzM3NDE1MSI+PHBhdGggc3R5bGU9ImZpbGw6bm9uZSIgZD0iTSAyNTMuMjEgLTc0LjU3IEwgMjUzLjIxIC05My41NSIgLz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMzc0MTUxOyIgZmlsbD0iIzM3NDE1MSIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgwLjAgLTEuMCAxLjAgMC4wIDI1My4yMSAtOTMuNTUpIj48cGF0aCBkPSJNIDYuMzcgMCBDIDUuNTggMC4xOSAyLjE1IDEuMjggMCAyLjQ2IEwgMCAtMi40NiBDIDIuMTUgLTEuMjggNS41OCAtMC4xOSA2LjM3IDAgWiIgLz48L2c+PC9nPjxnIHN0eWxlPSItLWx0eC1zdHJva2UtY29sb3I6IzAwMDAwMDstLWx0eC1maWxsLWNvbG9yOiMwMDAwMDA7IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjU4LjM4IC04OS40OSkiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjEuMzRlbTstLWx0eC1mby1oZWlnaHQ6MC4zOGVtOy0tbHR4LWZvLWRlcHRoOjAuMTdlbTsiIHdpZHRoPSIxNC44MiIgaGVpZ2h0PSI2LjA1IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDQuMTcpIiBvdmVyZmxvdz0idmlzaWJsZSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTMy5GMS4xLjEucGljMS4xMC4xMC4xMC43LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOyI+eWVzPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMzc0MTUxOyIgc3Ryb2tlLXdpZHRoPSIwLjhwdCIgc3Ryb2tlPSIjMzc0MTUxIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDI1My4yMSAtMTMwLjMxIEwgMjUzLjIxIC0xNDkuMjkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6IzM3NDE1MTsiIGZpbGw9IiMzNzQxNTEiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1kYXNob2Zmc2V0PSIwLjBwdCIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgdHJhbnNmb3JtPSJtYXRyaXgoMC4wIC0xLjAgMS4wIDAuMCAyNTMuMjEgLTE0OS4yOSkiPjxwYXRoIGQ9Ik0gNi4zNyAwIEMgNS41OCAwLjE5IDIuMTUgMS4yOCAwIDIuNDYgTCAwIC0yLjQ2IEMgMi4xNSAtMS4yOCA1LjU4IC0wLjE5IDYuMzcgMCBaIiAvPjwvZz48L2c+PGcgc3Ryb2tlLXdpZHRoPSIwLjhwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMzc0MTUxOyIgc3Ryb2tlPSIjMzc0MTUxIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDI4Ny45MiAtMTc1LjM3IEwgMzA2LjkgLTE3NS4zNyIgLz48ZyBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojMzc0MTUxOyIgZmlsbD0iIzM3NDE1MSIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgc3Ryb2tlLWRhc2hvZmZzZXQ9IjAuMHB0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMzA2LjkgLTE3NS4zNykiPjxwYXRoIGQ9Ik0gNi4zNyAwIEMgNS41OCAwLjE5IDIuMTUgMS4yOCAwIDIuNDYgTCAwIC0yLjQ2IEMgMi4xNSAtMS4yOCA1LjU4IC0wLjE5IDYuMzcgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAyOTQuMjkgLTE2OC4zMykiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjEuMzRlbTstLWx0eC1mby1oZWlnaHQ6MC4zOGVtOy0tbHR4LWZvLWRlcHRoOjAuMTdlbTsiIHdpZHRoPSIxNC44MiIgaGVpZ2h0PSI2LjA1IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDQuMTcpIiBvdmVyZmxvdz0idmlzaWJsZSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTMy5GMS4xLjEucGljMS4xMS4xMS4xMS44LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOyI+eWVzPC9zcGFuPjwvc3Bhbj48L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PGcgc3Ryb2tlLXdpZHRoPSIwLjhwdCI+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMzc0MTUxOyIgc3Ryb2tlPSIjMzc0MTUxIj48cGF0aCBzdHlsZT0iZmlsbDpub25lIiBkPSJNIDIxOC41MSAtMTc1LjM3IEwgMTI2LjYxIC0xNzUuMzcgTCAxMjYuNjEgLTc4LjQxIiAvPjxnIHN0eWxlPSItLWx0eC1maWxsLWNvbG9yOiMzNzQxNTE7IiBmaWxsPSIjMzc0MTUxIiBzdHJva2UtZGFzaGFycmF5PSJub25lIiBzdHJva2UtZGFzaG9mZnNldD0iMC4wcHQiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIHRyYW5zZm9ybT0ibWF0cml4KDAuMCAxLjAgLTEuMCAwLjAgMTI2LjYxIC03OC40MSkiPjxwYXRoIGQ9Ik0gNi4zNyAwIEMgNS41OCAwLjE5IDIuMTUgMS4yOCAwIDIuNDYgTCAwIC0yLjQ2IEMgMi4xNSAtMS4yOCA1LjU4IC0wLjE5IDYuMzcgMCBaIiAvPjwvZz48L2c+PGcgc3R5bGU9Ii0tbHR4LXN0cm9rZS1jb2xvcjojMDAwMDAwOy0tbHR4LWZpbGwtY29sb3I6IzAwMDAwMDsiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAxNjYuNzQgLTE3MC4yMSkiIGZpbGw9IiMwMDAwMDAiIHN0cm9rZT0iIzAwMDAwMCI+PGZvcmVpZ25vYmplY3Qgc3R5bGU9Ii0tbHR4LWZvLXdpZHRoOjEuMDVlbTstLWx0eC1mby1oZWlnaHQ6MC4zOGVtOy0tbHR4LWZvLWRlcHRoOjBlbTsiIHdpZHRoPSIxMS42MyIgaGVpZ2h0PSI0LjE3IiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDQuMTcpIiBvdmVyZmxvdz0idmlzaWJsZSI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRhaW5lciI+PHNwYW4gY2xhc3M9Imx0eF9mb3JlaWdub2JqZWN0X2NvbnRlbnQiPjxzcGFuIGlkPSJTMy5GMS4xLjEucGljMS4xMi4xMi4xMi45LjEuMS4xIiBjbGFzcz0ibHR4X3RleHQiIHN0eWxlPSJmb250LXNpemU6NzAlOyI+bm88L3NwYW4+PC9zcGFuPjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L2c+PC9nPjwvc3ZnPg==)

Figure 1: A-CEGIS evaluation loop in which candidate regexes are checked
against hidden tests, failures become diagnostic counterexamples, and
hidden-set success triggers targeted robustness probing.

Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Oracle construction ‣ 3 Methodology ‣ COUNTEREXAMPLES AS FEEDBACK FOR AGENT SELF-CORRECTION")
shows the core loop. At turn $`t`$, the agent proposes $`a_{i}^{(t)}`$.
The oracle computes the pass vector $`\mathbf{y}_{i}^{(t)}`$ over
$`m=|T_{i}|`$ cases and pass rate

|     |     |     |
| --- | --- | --- |
|     |

       ``` math
       \rho_{i}^{(t)}=\frac{1}{m}\sum_{j=1}^{m}y_{i,j}^{(t)}.
       ```                                                     |     |

If $`\rho_{i}^{(t)}<1`$, feedback is produced according to the strategy.
In the diagnostic A-CEGIS strategy, feedback contains up to two false
negatives and two false positives, sorted to favour compact witnesses.
This tells the model not only that the regex is wrong but also whether
the language should be broadened or tightened.

The implementation supports a small strategy grid: zero-shot generation,
self-correction with generic language feedback, error-only feedback with
no witnesses, minimal A-CEGIS with one failing example, three-example
A-CEGIS, and diagnostic A-CEGIS. The ablation reported below compares
zero-shot, generic self-correction, error-only feedback, and diagnostic
A-CEGIS on the same task set. The common loop is the same across
strategies; only the information returned after a failed turn changes.

After hidden-set success, hardening searches a targeted probe pool
$`Q_{i}(r)`$ for witnesses where $`g_{i}`$ and $`r`$ disagree. The pool
is built from short generic strings, tokens in the task description,
literals extracted from the gold and candidate regexes, existing
positive and negative examples, and simple transformations such as
prefixing, suffixing, repetition, concatenation, and character mutation.
This makes the probe distribution intentionally local: it stresses
boundaries near the language already implied by the task rather than
sampling arbitrary strings.

### 3.2 Hardening protocol

Hardening is intended to expose semantic errors that the original hidden
set did not sample. For a candidate $`r`$, the probe stage searches for
strings

|     |     |     |
| --- | --- | --- |
|     |

````math
w\in Q_{i}(r)\quad\text{such that}\quad\mathbb{1}[g_{i}(w)]\neq\mathbb{1}[r(w)].
``` |  |

Each mismatch is labelled by direction. If the gold regex accepts $`w`$
but the candidate rejects it, $`w`$ becomes a probe-derived false
negative; if the candidate accepts $`w`$ but the gold rejects it, $`w`$
becomes a probe-derived false positive. In the reported configuration,
each hardening cycle evaluates up to 320 probes, appends at most four
probe-derived counterexamples to the task-local test set, and gives the
agent one repair attempt. This process is repeated for at most two
cycles, producing an expanded set $`T_{i}^{\prime}=T_{i}\cup C_{i}`$.

The final status is stricter than base success. *Hidden success* means
that a candidate passes the original sampled positives and negatives.
*Hardened success* means that the repaired candidate passes the expanded
set after probe-derived examples are added. *Robust success*
additionally requires a final probe pass with no remaining mismatches.
Thus a regex can appear solved on the hidden set and still fail
robustness because it accepts decorated literals, misses minimum-length
cases, or uses an overly broad character class.

The benchmark records endpoint and process metrics. If $`\tau_{i}`$ is
the first successful turn, it reports Pass@1, Pass@final, mean
time-to-success, and MRR. Repair locality is measured by normalised edit
similarity between consecutive regex token sequences,

|  |  |  |
|----|----|----|
|  |
``` math
S_{i}^{(t)}=1-\frac{\mathrm{EditDist}(u^{(t-1)},u^{(t)})}{\max\{|u^{(t-1)}|,|u^{(t)}|,1\}},
``` |  |

and net repair behaviour is measured by the refinement efficiency ratio

|  |  |  |
|----|----|----|
|  |
``` math
\mathrm{RER}=\frac{\sum_{i,t}f_{i}^{(t)}}{\max(1,\sum_{i,t}b_{i}^{(t)})},
``` |  |

where $`f`$ counts fixed cases and $`b`$ counts regressions.

These metrics are designed to describe the refinement trajectory rather
than only its endpoint. Mean time-to-success and MRR capture how quickly
the agent converges. Stability distinguishes local repair from complete
regeneration. RER asks whether the loop fixes more behaviours than it
breaks, which is especially important for formal artifacts: a revision
that solves one counterexample while regressing several earlier cases is
not good repair, even if it looks plausible in natural language.

## 4 Experiments and Results

The experimental artifacts evaluate Gemini 3 Flash Preview on 30
NL-RX-Turk tasks. Each task uses 24 hidden oracle checks before
hardening. We report two complementary settings. First, a four-turn
feedback-strategy ablation compares zero-shot generation, generic
self-correction, error-only feedback, and diagnostic A-CEGIS without
hardening. Secondly, a full diagnostic run allows up to seven turns,
followed by at most two hardening cycles with one repair attempt per
cycle.

The run is intentionally focused on feedback mechanisms rather than
model ranking. It does not claim to compare model families. Instead, it
tests whether a capable model, given structured counterexamples,
exhibits the repair dynamics that A-CEGIS is meant to measure and
whether those dynamics differ from weaker feedback channels. Because
regex evaluation is deterministic, changes in pass rate, stability, and
robustness can be attributed to the candidate sequence rather than to
stochastic environment effects.

|                                     |       |
|-------------------------------------|-------|
| \topruleMetric                      | Value |
| \midruleTasks                       | 30    |
| Pass@1                              | 0.167 |
| Pass@final (hidden set)             | 1.000 |
| Mean turns-to-success               | 2.667 |
| Mean reciprocal rank                | 0.477 |
| Mean initial pass rate              | 0.706 |
| Mean final pass rate                | 1.000 |
| Mean structural stability           | 0.561 |
| Fix events                          | 311   |
| Break events                        | 99    |
| Refinement efficiency ratio         | 2.613 |
| Hardened pass@final                 | 0.767 |
| Probe-clean rate                    | 0.767 |
| Robust pass@final                   | 0.767 |
| Mean hardened final pass rate       | 0.937 |
| Probe-derived counterexamples added | 80    |
| Hardening cycles completed          | 40    |
| \botrule                            |       |

Table 1: Full diagnostic A-CEGIS results over 30 tasks

Table [1](#S4.T1 "Table 1 ‣ 4 Experiments and Results ‣ COUNTEREXAMPLES AS FEEDBACK FOR AGENT SELF-CORRECTION")
shows a large gap between first-turn and final hidden-set success. Only
5 of 30 tasks are solved immediately, yet all are solved by the full
diagnostic loop. The mean initial pass rate of 0.706 indicates that
initial regexes are often close but incomplete; the repair trajectory
turns those partial matches into hidden-set-perfect solutions. RER is
2.61, with 311 fixes against 99 breaks, so refinement is net
constructive rather than oscillatory.

The MRR of 0.477 shows that the gain is not merely a final-turn rescue
after many attempts. Most tasks converge within a few turns, but not on
the first turn. This is exactly the regime in which multi-turn
evaluation is informative: one-shot accuracy is too pessimistic about
eventual repair ability, while final success alone hides the cost and
path of convergence.

|                   |            |       |       |          |
|-------------------|------------|-------|-------|----------|
| \topruleStrategy  | Pass@final | MRR   | RER   | Attempts |
| \midruleZero-shot | 0.167      | 0.167 | 0.000 | 1.00     |
| Self-correction   | 0.267      | 0.211 | 0.674 | 2.77     |
| Error-only        | 0.233      | 0.194 | 0.990 | 2.83     |
| Diag. A-CEGIS     | 0.900      | 0.458 | 2.521 | 2.53     |
| \botrule          |            |       |       |          |

Table 2: Feedback-strategy ablation over 30 tasks with a four-turn
budget

Table [2](#S4.T2 "Table 2 ‣ 4 Experiments and Results ‣ COUNTEREXAMPLES AS FEEDBACK FOR AGENT SELF-CORRECTION")
isolates the effect of feedback content. All strategies use the same
model and task set, and the diagnostic row is evaluated under the same
four-turn budget as the baselines. Returning concrete false-positive and
false-negative witnesses is substantially more effective than asking the
model to self-correct generically or reporting only the number of
hidden-test failures. Diagnostic A-CEGIS solves 27 of 30 tasks within
four turns, compared with 8 for generic self-correction, 7 for
error-only feedback, and 5 for zero-shot generation. It also uses no
more average generation attempts than the weaker multi-turn baselines,
indicating that the improvement comes from more informative feedback
rather than simply more turns.

|                      |                 |
|----------------------|-----------------|
| \topruleSuccess turn | Number of tasks |
| \midrule1            | 5               |
| 2                    | 10              |
| 3                    | 9               |
| 4                    | 3               |
| 5                    | 2               |
| 6                    | 1               |
| \botrule             |                 |

Table 3: Distribution of first successful turn

Table [3](#S4.T3 "Table 3 ‣ 4 Experiments and Results ‣ COUNTEREXAMPLES AS FEEDBACK FOR AGENT SELF-CORRECTION")
shows that most tasks require explicit repair: 25 of 30 are solved after
turn 1, and the mode is turn 2. No task requires more than six turns.
The hardening results then separate hidden-set success from robustness.
Only 23 of 30 tasks remain probe clean; the other seven pass the
original sampled tests but fail on targeted variants involving
boundaries, repetitions, or broad character classes.

The turn distribution also suggests that diagnostic counterexamples
narrow the search space quickly. The system is not repeatedly sampling
unrelated regexes until one happens to pass. Instead, many trajectories
make a small number of semantically meaningful changes, such as
replacing .\* with .+, tightening an alternation, or adding a missing
character-class constraint.

[TABLE]

Table 4: Representative hardening examples

Table [4](#S4.T4 "Table 4 ‣ 4 Experiments and Results ‣ COUNTEREXAMPLES AS FEEDBACK FOR AGENT SELF-CORRECTION")
illustrates the common pattern. The model often captures the coarse
language but expresses it with permissive constructs such as .\*, broad
one-character classes, or unnecessary context around a literal.
Hardening can remove this slack: .\*dog.\*\|\S is tightened to
dog\|\[A-Za-z0-9\_\], and surrounding wildcards are removed from an
expression requiring seven or more letters. These cases explain both the
0.561 stability score and the robustness gap: repairs preserve broad
structure while tightening operators that leak semantics.

### 4.1 Failure modes

The seven non-robust cases fall into three recurring categories. The
first is boundary under-specification, where a regex accidentally
accepts the empty string, a one-character string, or a missing
suffix/prefix case. The second is wildcard overreach, where .\* admits
repeated or decorated strings that were absent from the base hidden set.
The third is class-boundary confusion, where a broad class such as \w or
a negated class includes more symbols than the description intended.

These failures are not merely anecdotal edge cases. They show that
hidden-set success can validate a sampled behaviour profile while still
missing the intended language shape. In that sense, the hardening phase
plays a different role from ordinary testing: it is not just another
larger test set, but a targeted search around likely semantic
boundaries.

### 4.2 Interpretation

The main empirical lesson is that the same run can look excellent or
incomplete depending on the measurement layer. Under hidden tests, the
method solves every task. Under trajectory metrics, it shows efficient
and mostly constructive repair. Under hardening, it reveals that roughly
one quarter of hidden-set-perfect solutions still contain latent
semantic errors. These statements are not contradictory; they describe
different levels of reliability.

For agent evaluation, this distinction is useful. A benchmark that
reports only Pass@1 would miss the model’s ability to recover. A
benchmark that reports only final hidden success would miss the residual
robustness gap. A-CEGIS makes both visible in one compact loop.

## 5 Example A-CEGIS Run

To make the protocol concrete, consider a typical task trajectory. The
model often begins with a regex that captures the main lexical cue but
misses a boundary condition. The oracle evaluates the candidate on all
positives and negatives, separates failures by direction, and returns a
few witnesses. A false negative says that the expression is too narrow;
a false positive says that it is too broad. The next candidate is
therefore conditioned on a behavioural correction rather than a vague
request for improvement.

The traces make the run auditable. A close first candidate may need only
a local edit, such as replacing .\* with .+ after a minimum-length
failure, while a structurally wrong candidate may require a larger
rewrite. Structural stability and RER distinguish these cases by
recording whether edits are local and whether they fix more cases than
they break. The released artifacts include aggregate metrics, per-task
candidate sequences, and probe-derived counterexamples, so a run can be
inspected rather than only scored. The code is available in the public
A-CEGIS repository [b16](#bib.bib16) .

## 6 Discussion

A-CEGIS is best understood as a measurement protocol for agent repair.
Its value is that it separates refinement behaviour from other benchmark
complexity: the artifact is compact, the verifier is deterministic, and
feedback is concrete behavioural evidence. The ablation shows why this
matters. Diagnostic A-CEGIS reaches 0.900 Pass@final within four turns
while generic self-correction and error-only feedback remain below
0.300, indicating that directional witnesses help the model make
targeted edits instead of inferring the failure mode from a scalar
score.

The trajectory metrics add information that endpoint scores cannot
provide. Pass@1 shows the difficulty of initial synthesis, final hidden
success shows convergence, RER records whether revisions are
constructive, and stability distinguishes local repairs from wholesale
rewrites. Targeted hardening then adds an active robustness layer after
hidden-set success by probing strings near description tokens, regex
literals, and existing examples. This checks whether a recovered
expression remains reliable near the semantic boundaries implied by the
task.

A-CEGIS is meant to generalise beyond regexes wherever three ingredients
are available: a compact formal artifact, a deterministic oracle, and
counterexamples that can be rendered back to the model. SQL queries can
be checked against databases, schemas against documents, and
configuration policies against generated states. Recent DocSync work
applies a related critic-guided refinement loop to documentation
maintenance, pairing structural code context with iterative critique to
keep descriptions aligned with implementation [b15](#bib.bib15) . In
each case, the core question is whether an agent can use concrete
evidence to make a targeted repair.

## 7 Conclusion and Future Work

A-CEGIS shows that counterexamples are a powerful feedback signal for
agent self-correction. Instead of scoring only the first answer or final
endpoint, the framework records the proposed artifact, concrete
behavioural failures, revisions, regressions, and robustness under
additional probing. In regex synthesis, diagnostic counterexamples raise
four-turn Pass@final to 0.900 and full diagnostic hidden-set success to
1.0, while hardening exposes a remaining robustness gap. Future work can
scale the benchmark across additional models, add exact equivalence
checks where available, and extend the same compact-artifact,
deterministic-oracle pattern to SQL queries, schemas, configuration
policies, and program fragments.

The main lesson is that refinement quality should be evaluated as a
trajectory, not as a single number. A final answer can pass sampled
tests while still being brittle, and a low first-attempt score can hide
a strong ability to repair once concrete evidence is available. By
preserving the sequence of candidates, counterexamples, fixes, and
breaks, A-CEGIS makes those distinctions visible and gives future
evaluations a practical way to compare not only whether agents succeed,
but how they improve.

## 8 References

## References

- \[1\] Guan, S., Xiong, H., Wang, J., et al.: ‘Evaluating LLM-based
  agents for multi-turn conversations: a survey’, arXiv preprint
  arXiv:2503.22458, 2025. DOI: https://doi.org/10.48550/arXiv.2503.22458
- \[2\] Rawal, R., Chiang, J.Y.F., Shen, C., et al.: ‘Benchmarking
  correctness and security in multi-turn code generation’,
  OpenReview, 2025. DOI: https://doi.org/10.48550/arXiv.2510.13859
- \[3\] Madaan, A., Tandon, N., Gupta, P., et al.: ‘Self-Refine:
  iterative refinement with self-feedback’, arXiv preprint
  arXiv:2303.17651, 2023. DOI: https://doi.org/10.48550/arXiv.2303.17651
- \[4\] Ma, C., Zhang, J., Zhu, Z., et al.: ‘AgentBoard: an analytical
  evaluation board of multi-turn LLM agents’, arXiv preprint
  arXiv:2401.13178, 2024. DOI: https://doi.org/10.48550/arXiv.2401.13178
- \[5\] Locascio, N., Narasimhan, K., DeLeon, E., et al.: ‘Neural
  generation of regular expressions from natural language with minimal
  domain knowledge’. Proc. Conf. on Empirical Methods in Natural
  Language Processing, Austin, TX, USA, November 2016, pp. 1918–1923.
  DOI: https://doi.org/10.18653/v1/D16-1197
- \[6\] Abate, A., David, C., Kroening, D., et al.: ‘Counterexample
  guided inductive synthesis modulo theories’. Proc. Int. Conf. Computer
  Aided Verification, Oxford, UK, July 2018, pp. 270–288. DOI:
  https://doi.org/10.1007/978-3-319-96145-3_15
- \[7\] Levenshtein, V.I.: ‘Binary codes capable of correcting
  deletions, insertions, and reversals’, *Soviet Physics Doklady*, 1966,
  10, (8), pp. 707–710. URL:
  https://ui.adsabs.harvard.edu/abs/1966SPhD...10..707L/abstract,
  accessed 10 May 2026
- \[8\] Chen, M., Tworek, J., Jun, H., et al.: ‘Evaluating large
  language models trained on code’, arXiv preprint
  arXiv:2107.03374, 2021. DOI: https://doi.org/10.48550/arXiv.2107.03374
- \[9\] Austin, J., Odena, A., Nye, M., et al.: ‘Program synthesis with
  large language models’, arXiv preprint arXiv:2108.07732, 2021. DOI:
  https://doi.org/10.48550/arXiv.2108.07732
- \[10\] Yao, S., Zhao, J., Yu, D., et al.: ‘ReAct: synergizing
  reasoning and acting in language models’, in *Proc. Int. Conf.
  Learning Representations*, Kigali, Rwanda, May 2023. DOI:
  https://doi.org/10.48550/arXiv.2210.03629
- \[11\] Shinn, N., Cassano, F., Gopinath, A., et al.: ‘Reflexion:
  language agents with verbal reinforcement learning’, in *Advances in
  Neural Information Processing Systems*, New Orleans, LA, USA,
  December 2023. DOI: https://doi.org/10.48550/arXiv.2303.11366
- \[12\] Jimenez, C.E., Yang, J., Wettig, A., et al.: ‘SWE-bench: can
  language models resolve real-world GitHub issues?’, in *Proc. Int.
  Conf. Learning Representations*, Vienna, Austria, May 2024. DOI:
  https://doi.org/10.48550/arXiv.2310.06770
- \[13\] Alur, R., Bodik, R., Juniwal, G., et al.: ‘Syntax-guided
  synthesis’. Proc. IEEE Int. Conf. Formal Methods in Computer-Aided
  Design, Portland, OR, USA, October 2013, pp. 1–8. DOI:
  https://doi.org/10.1109/FMCAD.2013.6679385
- \[14\] Gulwani, S.: ‘Automating string processing in spreadsheets
  using input-output examples’. Proc. ACM SIGPLAN-SIGACT Symp.
  Principles of Programming Languages, Austin, TX, USA, January 2011,
  pp. 317–330. DOI: https://doi.org/10.1145/1926385.1926423
- \[15\] Badrinarayan, S., Parthasarathy, A.: ‘DocSync: agentic
  documentation maintenance via critic-guided Reflexion’, arXiv preprint
  arXiv:2605.02163, 2026. DOI: https://doi.org/10.48550/arXiv.2605.02163
- \[16\] Badrinarayan, S.: ‘A-CEGIS regex refinement experiments’,
  GitHub repository, 2026. URL: https://github.com/TheSidhesh/ACEGIS
````
