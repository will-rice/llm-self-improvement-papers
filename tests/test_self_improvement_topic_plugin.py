import json
from pathlib import Path

import pytest

from papers_pipeline.config import TopicConfig
from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision, build_topic_gate
from topic_plugin import (
    CORE_PAPERS,
    SIMILARITY_THRESHOLD,
    TITLE_SIMILARITY_FLOOR,
    accept_topic,
    core_embeddings,
    keyword_decision,
)

GATE_PAPERS = json.loads(
    (Path(__file__).parent / "fixtures" / "topic_gate_papers.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize(
    ("title", "abstract", "decision"),
    [
        (
            "Self-Rewarding Language Models",
            "The model judges its own responses to build preference data.",
            TopicDecision(True, "accepted by title"),
        ),
        (
            "Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents",
            "A coding agent rewrites its own code.",
            TopicDecision(True, "accepted by title"),
        ),
        (
            "SPICE: Self-Play In Corpus Environments Improves Reasoning",
            "A single model plays challenger and reasoner.",
            TopicDecision(True, "accepted by title"),
        ),
        (
            "A Self-Play Posterior Sampling Algorithm for Zero-Sum Markov Games",
            "Regret bounds for two-player games.",
            TopicDecision(False, "missing language model signal"),
        ),
        (
            "STELLA: Self-Evolving LLM Agent for Biomedical Research",
            "An agent that grows its own tool library.",
            TopicDecision(False, "applied agent for one domain"),
        ),
        (
            "Sherlock: Self-Correcting Reasoning in Vision-Language Models",
            "Self-correction for VLMs.",
            TopicDecision(False, "title matched excluded term: vision"),
        ),
        (
            "RLCD: Reinforcement Learning from Contrastive Distillation",
            "Aligns language models without using human feedback, from"
            " self-generated preference pairs.",
            TopicDecision(True, "accepted by abstract"),
        ),
        (
            "Scaling Large Language Models",
            "Compute-optimal pretraining laws for LLMs.",
            TopicDecision(False, "missing self-improvement signal"),
        ),
    ],
)
def test_keyword_decision_applies_self_improvement_rule(
    paper: Paper, title: str, abstract: str, decision: TopicDecision
) -> None:
    candidate = paper.model_copy(update={"title": title, "abstract": abstract})

    assert keyword_decision(candidate) == decision


def test_every_core_paper_passes_the_keyword_rule(paper: Paper) -> None:
    core = json.loads(CORE_PAPERS.read_text(encoding="utf-8"))

    rejected = [
        entry["name"]
        for entry in core
        if not keyword_decision(
            paper.model_copy(
                update={"title": entry["title"], "abstract": entry["abstract"]}
            )
        ).accepted
    ]

    assert rejected == []


def test_every_core_paper_clears_its_threshold_against_the_others(
    paper: Paper,
) -> None:
    core = json.loads(CORE_PAPERS.read_text(encoding="utf-8"))
    similarity = core_embeddings() @ core_embeddings().T
    similarity[range(len(similarity)), range(len(similarity))] = 0

    below = [
        entry["name"]
        for entry, score in zip(core, similarity.max(axis=1), strict=True)
        if score
        < (
            TITLE_SIMILARITY_FLOOR
            if keyword_decision(
                paper.model_copy(
                    update={"title": entry["title"], "abstract": entry["abstract"]}
                )
            ).reason
            == "accepted by title"
            else SIMILARITY_THRESHOLD
        )
    ]

    assert below == []


@pytest.mark.parametrize(
    ("fixture", "decision"),
    [
        ("on_topic", TopicDecision(True, "accepted by title")),
        (
            "off_topic",
            TopicDecision(False, "similarity to core papers 0.44 below threshold"),
        ),
    ],
)
def test_accept_topic_gates_on_similarity_to_core_papers(
    paper: Paper, fixture: str, decision: TopicDecision
) -> None:
    entry = GATE_PAPERS[fixture]
    candidate = paper.model_copy(
        update={"title": entry["title"], "abstract": entry["abstract"]}
    )

    assert keyword_decision(candidate).accepted
    assert accept_topic(candidate) == decision


def test_papers_yml_plugin_reference_builds_repository_gate(paper: Paper) -> None:
    gate = build_topic_gate(TopicConfig(plugin="topic_plugin:accept_topic"))
    entry = GATE_PAPERS["on_topic"]
    candidate = paper.model_copy(
        update={"title": entry["title"], "abstract": entry["abstract"]}
    )

    assert gate(candidate) == TopicDecision(True, "accepted by title")
