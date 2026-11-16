# Core LLM self-improvement papers

The reference set that defines this repository's scope: language models and
LLM agents that improve from their own outputs or experience.

## Sources

- Tao et al., *A Survey on Self-Evolution of Large Language Models*
  (arXiv:2404.14387). Its §4 (experience acquisition), §5 (experience
  refinement) and §6 (updating) describe the self-evolution loop; the named
  self-improvement methods cited there are listed with the section that first
  cites them. Base models, adapters, memory systems, decoding methods,
  data-augmentation recipes and robotics or domain applications it also cites
  are left out.
- Added: methods and studies published since or outside the survey (SCoRe,
  Meta-Rewarding, the sharpening and Mind the Gap analyses, LLMs Cannot
  Self-Correct Reasoning Yet, Absolute Zero, R-Zero, SEAL, Language Self-Play,
  SPICE, Self-Play SWE-RL, Godel Agent, Darwin Godel Machine), plus five
  surveys of the field.

## Scope

Self-evolving agent methods and studies are in scope; applied papers that build
a self-evolving agent for one domain (finance, medicine, chemistry and so on)
are not. Text language models only: image, video, speech, robotics and other
multimodal self-improvement are out.

## Use in the topic rule

`core_papers.json` holds every paper below. The topic rule's similarity gate
compares each accepted paper with them, and the tests require every one to
pass the keyword rule and clear its similarity threshold against the others.

| Paper | arXiv | Year | Source | Title |
|---|---|---|---|---|
| STaR | [2203.14465](https://arxiv.org/abs/2203.14465) | 2022 | Tao et al. §3.1.1 LLMs | STaR: Bootstrapping Reasoning With Reasoning |
| LMSI | [2210.11610](https://arxiv.org/abs/2210.11610) | 2022 | Tao et al. §20232022 InstructGPT | Large Language Models Can Self-Improve |
| Constitutional AI | [2212.08073](https://arxiv.org/abs/2212.08073) | 2022 | Tao et al. §3.1.1 LLMs | Constitutional AI: Harmlessness from AI Feedback |
| Self-Verification | [2212.09561](https://arxiv.org/abs/2212.09561) | 2022 | Tao et al. §3.1.1 LLMs | Large Language Models are Better Reasoners with Self-Verification |
| Self-Instruct | [2212.10560](https://arxiv.org/abs/2212.10560) | 2022 | Tao et al. §20232022 InstructGPT | Self-Instruct: Aligning Language Models with Self-Generated Instructions |
| Reflexion | [2303.11366](https://arxiv.org/abs/2303.11366) | 2023 | Tao et al. §3.1.1 LLMs | Reflexion: Language Agents with Verbal Reinforcement Learning |
| Self-Refine | [2303.17651](https://arxiv.org/abs/2303.17651) | 2023 | Tao et al. §3.1.1 LLMs | Self-Refine: Iterative Refinement with Self-Feedback |
| Self-Debug | [2304.05128](https://arxiv.org/abs/2304.05128) | 2023 | Tao et al. §3.1.1 LLMs | Teaching Large Language Models to Self-Debug |
| Self-Align | [2305.03047](https://arxiv.org/abs/2305.03047) | 2023 | Tao et al. §3.1.1 LLMs | Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision |
| MoT | [2305.05181](https://arxiv.org/abs/2305.05181) | 2023 | Tao et al. §3.1.1 LLMs | MoT: Memory-of-Thought Enables ChatGPT to Self-Improve |
| CRITIC | [2305.11738](https://arxiv.org/abs/2305.11738) | 2023 | Tao et al. §3.1.1 LLMs | CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing |
| SIRLC | [2305.14483](https://arxiv.org/abs/2305.14483) | 2023 | Tao et al. §3.1.1 LLMs | Language Model Self-improvement by Reinforcement Learning Contemplation |
| SelfEvolve | [2306.02907](https://arxiv.org/abs/2306.02907) | 2023 | Tao et al. §3.1.1 LLMs | SelfEvolve: A Code Evolution Framework via Large Language Models |
| RLCD | [2307.12950](https://arxiv.org/abs/2307.12950) | 2023 | Tao et al. §3.1.1 LLMs | RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment |
| Instruction Backtranslation | [2308.06259](https://arxiv.org/abs/2308.06259) | 2023 | Tao et al. §3.1.1 LLMs | Self-Alignment with Instruction Backtranslation |
| ReST | [2308.08998](https://arxiv.org/abs/2308.08998) | 2023 | Tao et al. §20232022 InstructGPT | Reinforced Self-Training (ReST) for Language Modeling |
| ISR-LLM | [2308.13724](https://arxiv.org/abs/2308.13724) | 2023 | Tao et al. §3.1.1 LLMs | ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning |
| Promptbreeder | [2309.16797](https://arxiv.org/abs/2309.16797) | 2023 | Tao et al. §3.1.1 LLMs | Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution |
| Cannot Self-Correct Yet | [2310.01798](https://arxiv.org/abs/2310.01798) | 2023 | added | Large Language Models Cannot Self-Correct Reasoning Yet |
| SALMON | [2310.05910](https://arxiv.org/abs/2310.05910) | 2023 | Tao et al. §3.1.1 LLMs | SALMON: Self-Alignment with Instructable Reward Models |
| DiverseEvol | [2311.08182](https://arxiv.org/abs/2311.08182) | 2023 | Tao et al. §3.1.1 LLMs | Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning |
| SPIN | [2401.01335](https://arxiv.org/abs/2401.01335) | 2024 | Tao et al. §3.1.1 LLMs | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models |
| Self-Contrast | [2401.02009](https://arxiv.org/abs/2401.02009) | 2024 | Tao et al. §3.1.1 LLMs | Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives |
| Self-Talk | [2401.05033](https://arxiv.org/abs/2401.05033) | 2024 | Tao et al. §3.1.1 LLMs | Bootstrapping LLM-based Task-Oriented Dialogue Agents via Self-Talk |
| AutoAct | [2401.05268](https://arxiv.org/abs/2401.05268) | 2024 | Tao et al. §3.1.1 LLMs | AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning |
| Self-Rewarding | [2401.10020](https://arxiv.org/abs/2401.10020) | 2024 | Tao et al. §3.1.1 LLMs | Self-Rewarding Language Models |
| GRATH | [2401.12292](https://arxiv.org/abs/2401.12292) | 2024 | Tao et al. §3.1.1 LLMs | GRATH: Gradual Self-Truthifying for Large Language Models |
| DITTO | [2401.12474](https://arxiv.org/abs/2401.12474) | 2024 | Tao et al. §3.1.1 LLMs | Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment |
| ICE | [2401.13996](https://arxiv.org/abs/2401.13996) | 2024 | Tao et al. §3.1.1 LLMs | Investigate-Consolidate-Exploit: A General Strategy for Inter-Task Agent Self-Evolution |
| V-STaR | [2402.06457](https://arxiv.org/abs/2402.06457) | 2024 | Tao et al. §3.1.1 LLMs | V-STaR: Training Verifiers for Self-Taught Reasoners |
| SOFT | [2402.07610](https://arxiv.org/abs/2402.07610) | 2024 | Tao et al. §3.1.1 LLMs | Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping |
| Self-Alignment for Factuality | [2402.09267](https://arxiv.org/abs/2402.09267) | 2024 | Tao et al. §3.1.1 LLMs | Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation |
| DLMA | [2402.11907](https://arxiv.org/abs/2402.11907) | 2024 | Tao et al. §3.1.1 LLMs | Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation |
| Agent-Pro | [2402.17574](https://arxiv.org/abs/2402.17574) | 2024 | Tao et al. §3.1.1 LLMs | Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization |
| Survey: self-evolution of LLMs | [2404.14387](https://arxiv.org/abs/2404.14387) | 2024 | survey | A Survey on Self-Evolution of Large Language Models |
| Meta-Rewarding | [2407.19594](https://arxiv.org/abs/2407.19594) | 2024 | added | Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge |
| SCoRe | [2409.12917](https://arxiv.org/abs/2409.12917) | 2024 | added | Training Language Models to Self-Correct via Reinforcement Learning |
| Godel Agent | [2410.04444](https://arxiv.org/abs/2410.04444) | 2024 | added | Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement |
| Sharpening | [2412.01951](https://arxiv.org/abs/2412.01951) | 2024 | added | Self-Improvement in Language Models: The Sharpening Mechanism |
| Mind the Gap | [2412.02674](https://arxiv.org/abs/2412.02674) | 2024 | added | Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models |
| Survey: inference-time self-improvement | [2412.14352](https://arxiv.org/abs/2412.14352) | 2024 | survey | A Survey on LLM Inference-Time Self-Improvement |
| Self-rewarding correction | [2502.19613](https://arxiv.org/abs/2502.19613) | 2025 | added | Self-rewarding correction for mathematical reasoning |
| Absolute Zero | [2505.03335](https://arxiv.org/abs/2505.03335) | 2025 | added | Absolute Zero: Reinforced Self-play Reasoning with Zero Data |
| Darwin Godel Machine | [2505.22954](https://arxiv.org/abs/2505.22954) | 2025 | added | Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents |
| Reflect Retry Reward | [2505.24726](https://arxiv.org/abs/2505.24726) | 2025 | added | Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning |
| SEAL | [2506.10943](https://arxiv.org/abs/2506.10943) | 2025 | added | Self-Adapting Language Models |
| Survey: self-evolving agents | [2507.21046](https://arxiv.org/abs/2507.21046) | 2025 | survey | A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence |
| R-Zero | [2508.05004](https://arxiv.org/abs/2508.05004) | 2025 | added | R-Zero: Self-Evolving Reasoning LLM from Zero Data |
| Survey: self-evolving AI agents | [2508.07407](https://arxiv.org/abs/2508.07407) | 2025 | survey | A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems |
| Language Self-Play | [2509.07414](https://arxiv.org/abs/2509.07414) | 2025 | added | Language Self-Play For Data-Free Training |
| SPICE | [2510.24684](https://arxiv.org/abs/2510.24684) | 2025 | added | SPICE: Self-Play In Corpus Environments Improves Reasoning |
| Self-Play SWE-RL | [2512.18552](https://arxiv.org/abs/2512.18552) | 2025 | added | Toward Training Superintelligent Software Agents through Self-Play SWE-RL |
| Survey: self-improvement technical overview | [2603.25681](https://arxiv.org/abs/2603.25681) | 2026 | survey | Self-Improvement of Large Language Models: A Technical Overview and Future Outlook |
