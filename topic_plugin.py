"""LLM self-improvement relevance rule.

The repository covers language models and LLM agents that improve from their
own outputs or experience: self-training (STaR, ReST), self-refinement and
self-correction (Self-Refine, SCoRe), self-play (SPIN, Absolute Zero),
self-rewarding and self-alignment (Self-Rewarding LMs, Constitutional AI), and
self-evolving agent methods (Reflexion, Godel Agent, Darwin Godel Machine).
Applied papers that build a self-evolving agent for one domain (finance,
medicine, chemistry and so on) are out of scope, as are image, video, speech,
robotics and other multimodal settings.

Every accepted paper is compared with core_papers.json, the core papers listed
in docs/research/core-papers.md. A title that names a self-improvement method
only needs to clear a similarity floor. A paper whose abstract names it at
least twice, without the title saying so, must be close to a core paper.
"""

import functools
import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision

# "self-" compounds that only mean self-improvement.
SELF_SPECIFIC = (
    r"\bself[- ]?(improv\w*|refin\w*|correct\w*|rewarding|reward(s|ed)?|taught|evol\w*"
    r"|instruct\w*|align\w*|verif\w*|feedback|critiqu\w*|critic|reflect\w*|debug\w*"
    r"|bootstrap\w*|referential|generated|talk|contrast|teach\w*|learn\w*|truthif\w*"
    r"|plan\w*)\b"
)

# "self-" compounds that classic reinforcement learning and representation
# learning also use. A title matching only these needs a language model
# mentioned somewhere.
SELF_AMBIGUOUS = r"\bself[- ]?(play|train\w*|distill\w*|adapt\w*)\b"

OTHER_PHRASES = (
    r"bootstrap\w* (reasoning|language models?|llms?)|from zero data|zero[- ]data"
    r"|learn\w* from (its|their) own|(its|their) own (outputs|generations|feedback|mistakes)"
    r"|recursive(ly)? self|constitutional ai|ai feedback|rlaif|meta-rewarding"
    r"|sharpening mechanism|iterative (self-)?refinement|instruction backtranslation"
    r"|learn\w* to (self-)?(evolve|improve)|from scratch"
    r"|without (using )?human (feedback|annotations?|supervision|labels)"
    r"|\b(star|v-star|rest|spin|reflexion|critic|g[öo]del (agent|machine)|absolute zero"
    r"|r-zero)\b"
)

SPECIFIC_PATTERN = re.compile(SELF_SPECIFIC + "|" + OTHER_PHRASES)
TASK_PATTERN = re.compile(SELF_SPECIFIC + "|" + SELF_AMBIGUOUS + "|" + OTHER_PHRASES)
ABSTRACT_MENTIONS = 2

LANGUAGE_MODEL = re.compile(
    r"\b(llms?|slms?|lrms?|language models?|foundation models?|gpt|chatgpt|llama|qwen"
    r"|mistral|deepseek|reasoning models?|transformers?|post-training|rlvr|reasoning"
    r"|conversation\w*|dialog\w*)\b|chain[- ]of[- ]thought|retrieval[- ]augmented"
    r"|(language|coding|code|tool|search|web|llm-based|research|software|computer-use)"
    r" agents?"
)

TITLE_EXCLUSION = re.compile(
    r"\b(image|images|video|audio|speech|vision|visual|diffusion|robot\w*|embodied|gui"
    r"|3d|multimodal|mllms?|vlms?|lvlms?|vision-language|text-to-(image|video))\b"
)

AGENT = re.compile(r"\bagents?\b|\bagentic\b")
APPLICATION_DOMAIN = re.compile(
    r"\b(financ\w*|trading|stock|medical|clinical|health\w*|legal|law|chemistr\w*"
    r"|chemical|materials?|biomedical|molecul\w*|protein|drug|patent|education\w*"
    r"|cyber\w*|security|recommend\w*|e-commerce|customer|port|vehicle|traffic"
    r"|manufactur\w*|supply chain|energy|power grid|geospatial|weather|agricultur\w*"
    r"|travel|marketing|accounting|insurance|mobile|smart contract|telecom|pandemic)\b"
)

CORE_PAPERS = Path(__file__).with_name("core_papers.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Titles naming a self-improvement method below this are mostly speculative
# decoding, game-theoretic self-play or jailbreak papers.
TITLE_SIMILARITY_FLOOR = 0.50
# Abstract-only papers below this mostly mention self-improvement in passing.
SIMILARITY_THRESHOLD = 0.60


def accept_topic(paper: Paper) -> TopicDecision:
    """Apply the LLM self-improvement relevance rule to a paper's title and abstract.

    Args:
        paper: Normalized paper to classify.

    Returns:
        Acceptance decision with the rule that decided it.
    """
    decision = keyword_decision(paper)
    if not decision.accepted:
        return decision
    similarity = float(
        (embed([f"{paper.title}. {paper.abstract}"]) @ core_embeddings().T).max()
    )
    threshold = (
        TITLE_SIMILARITY_FLOOR
        if decision.reason == "accepted by title"
        else SIMILARITY_THRESHOLD
    )
    if similarity < threshold:
        return TopicDecision(
            False, f"similarity to core papers {similarity:.2f} below threshold"
        )
    return decision


def keyword_decision(paper: Paper) -> TopicDecision:
    """Apply the keyword part of the rule.

    Args:
        paper: Normalized paper to classify.

    Returns:
        Acceptance decision with the rule that decided it.
    """
    title = paper.title.lower()
    text = f"{title} {paper.abstract.lower()}"
    excluded = TITLE_EXCLUSION.search(title)
    if excluded is not None:
        return TopicDecision(False, f"title matched excluded term: {excluded[0]}")
    if AGENT.search(title) and APPLICATION_DOMAIN.search(title):
        return TopicDecision(False, "applied agent for one domain")
    if TASK_PATTERN.search(title):
        if not SPECIFIC_PATTERN.search(title) and not LANGUAGE_MODEL.search(text):
            return TopicDecision(False, "missing language model signal")
        return TopicDecision(True, "accepted by title")
    if len(TASK_PATTERN.findall(text)) < ABSTRACT_MENTIONS:
        return TopicDecision(False, "missing self-improvement signal")
    return TopicDecision(True, "accepted by abstract")


@functools.cache
def core_embeddings() -> np.ndarray:
    """Embed the core papers once per process.

    Returns:
        Unit-normalized embeddings, one row per core paper.
    """
    core = json.loads(CORE_PAPERS.read_text(encoding="utf-8"))
    return embed([f"{entry['title']}. {entry['abstract']}" for entry in core])


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with the shared model.

    fastembed runs the model with ONNX Runtime and returns NumPy arrays; torch
    is only installed with the conversion extra, so the gate stays in NumPy.

    Args:
        texts: Texts to embed.

    Returns:
        Unit-normalized embeddings, one row per text.
    """
    vectors = np.stack(list(embedding_model().embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@functools.cache
def embedding_model() -> TextEmbedding:
    """Load the embedding model once per process.

    Returns:
        The sentence embedding model.
    """
    return TextEmbedding(EMBEDDING_MODEL)
