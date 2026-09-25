import pytest

from papers_pipeline.config import TopicConfig
from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision, build_topic_gate
from topic_plugin import accept_topic


@pytest.mark.parametrize(
    ("title", "abstract", "decision"),
    [
        (
            "Self-Rewarding Language Models",
            "The model judges its own responses to build preference data.",
            TopicDecision(True, "accepted"),
        ),
        (
            "Recursive Self-Improvement of Research Agents",
            "Agents rewrite their own scaffolding across generations.",
            TopicDecision(True, "accepted"),
        ),
        (
            "Iterative Self-Refine for LLMs",
            "Feedback from the same model improves each draft.",
            TopicDecision(True, "accepted"),
        ),
        (
            "Self-Distillation for Vision Transformers",
            "A self-supervised image representation learner.",
            TopicDecision(False, "missing language model signal"),
        ),
        (
            "Filmmaking with Diffusion",
            "Self-play between a critic and generator improves film shots.",
            TopicDecision(False, "missing language model signal"),
        ),
        (
            "Scaling Large Language Models",
            "Compute-optimal pretraining laws for LLMs.",
            TopicDecision(False, "missing self-improvement signal"),
        ),
    ],
)
def test_accept_topic_requires_model_and_self_improvement_signals(
    paper: Paper, title: str, abstract: str, decision: TopicDecision
) -> None:
    candidate = paper.model_copy(update={"title": title, "abstract": abstract})

    assert accept_topic(candidate) == decision


def test_papers_yml_plugin_reference_builds_repository_gate(paper: Paper) -> None:
    gate = build_topic_gate(TopicConfig(plugin="topic_plugin:accept_topic"))
    candidate = paper.model_copy(
        update={"title": "Self-Play Fine-Tuning", "abstract": "An LLM plays itself."}
    )

    assert gate(candidate) == TopicDecision(True, "accepted")
