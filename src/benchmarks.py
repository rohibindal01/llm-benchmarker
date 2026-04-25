"""
LLM Benchmark Scores Database
Static scores based on publicly available research, benchmarks, and evaluations.
Sources: MMLU, HumanEval, MT-Bench, TruthfulQA, HellaSwag, BIG-Bench, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelScore:
    name: str
    provider: str
    version: str
    scores: Dict[str, float]
    metadata: Dict[str, str] = field(default_factory=dict)

    def overall_score(self) -> float:
        weights = {
            "reasoning": 0.18,
            "language": 0.15,
            "factual_accuracy": 0.15,
            "coding": 0.14,
            "creativity": 0.10,
            "safety": 0.12,
            "speed": 0.08,
            "cost": 0.08,
        }
        total = 0.0
        for key, weight in weights.items():
            total += self.scores.get(key, 0) * weight
        return round(total, 2)


# ─── Score Definitions ───────────────────────────────────────────────────────
# Each category score is out of 10.
# Multimodal is stored separately (0-10 or None if not applicable).

MODELS: list[ModelScore] = [
    ModelScore(
        name="GPT-4o",
        provider="OpenAI",
        version="gpt-4o-2024-08",
        scores={
            "reasoning": 9.3,
            "language": 9.4,
            "factual_accuracy": 8.8,
            "coding": 9.0,
            "creativity": 8.7,
            "safety": 9.0,
            "speed": 8.0,
            "cost": 5.5,
            "multimodal": 9.5,
        },
        metadata={
            "context_window": "128K tokens",
            "input_price": "$2.50 / 1M tokens",
            "output_price": "$10.00 / 1M tokens",
            "mmlu": "88.7%",
            "humaneval": "90.2%",
            "best_for": "General, Research, Multimodal",
        },
    ),
    ModelScore(
        name="Claude 3.5 Sonnet",
        provider="Anthropic",
        version="claude-3-5-sonnet-20241022",
        scores={
            "reasoning": 9.2,
            "language": 9.5,
            "factual_accuracy": 9.0,
            "coding": 9.4,
            "creativity": 9.2,
            "safety": 9.6,
            "speed": 8.2,
            "cost": 6.0,
            "multimodal": 8.8,
        },
        metadata={
            "context_window": "200K tokens",
            "input_price": "$3.00 / 1M tokens",
            "output_price": "$15.00 / 1M tokens",
            "mmlu": "88.3%",
            "humaneval": "92.0%",
            "best_for": "Coding, Creative Writing, Safety-critical",
        },
    ),
    ModelScore(
        name="Gemini 1.5 Pro",
        provider="Google",
        version="gemini-1.5-pro-002",
        scores={
            "reasoning": 9.0,
            "language": 9.1,
            "factual_accuracy": 8.7,
            "coding": 8.7,
            "creativity": 8.5,
            "safety": 8.8,
            "speed": 7.8,
            "cost": 6.5,
            "multimodal": 9.3,
        },
        metadata={
            "context_window": "1M tokens",
            "input_price": "$1.25 / 1M tokens (≤128K)",
            "output_price": "$5.00 / 1M tokens",
            "mmlu": "85.9%",
            "humaneval": "84.1%",
            "best_for": "Long-context tasks, Multimodal, Research",
        },
    ),
    ModelScore(
        name="LLaMA 3 70B",
        provider="Meta",
        version="llama-3-70b-instruct",
        scores={
            "reasoning": 8.3,
            "language": 8.5,
            "factual_accuracy": 7.8,
            "coding": 8.0,
            "creativity": 7.9,
            "safety": 7.5,
            "speed": 8.8,
            "cost": 9.5,
            "multimodal": None,
        },
        metadata={
            "context_window": "8K tokens",
            "input_price": "Free / Self-hosted",
            "output_price": "Free / Self-hosted",
            "mmlu": "82.0%",
            "humaneval": "81.7%",
            "best_for": "Low-cost, Self-hosted, Privacy-focused",
        },
    ),
    ModelScore(
        name="Mistral Large",
        provider="Mistral AI",
        version="mistral-large-2411",
        scores={
            "reasoning": 8.5,
            "language": 8.6,
            "factual_accuracy": 8.0,
            "coding": 8.5,
            "creativity": 7.8,
            "safety": 8.2,
            "speed": 9.0,
            "cost": 8.5,
            "multimodal": None,
        },
        metadata={
            "context_window": "128K tokens",
            "input_price": "$2.00 / 1M tokens",
            "output_price": "$6.00 / 1M tokens",
            "mmlu": "84.0%",
            "humaneval": "83.2%",
            "best_for": "Cost-efficient, Coding, Multilingual",
        },
    ),
    ModelScore(
        name="Gemini 1.5 Flash",
        provider="Google",
        version="gemini-1.5-flash-002",
        scores={
            "reasoning": 7.8,
            "language": 8.2,
            "factual_accuracy": 7.9,
            "coding": 7.7,
            "creativity": 7.5,
            "safety": 8.5,
            "speed": 9.5,
            "cost": 9.8,
            "multimodal": 8.5,
        },
        metadata={
            "context_window": "1M tokens",
            "input_price": "$0.075 / 1M tokens",
            "output_price": "$0.30 / 1M tokens",
            "mmlu": "78.9%",
            "humaneval": "74.3%",
            "best_for": "High-volume, Low-latency, Budget",
        },
    ),
    ModelScore(
        name="GPT-4o mini",
        provider="OpenAI",
        version="gpt-4o-mini-2024-07",
        scores={
            "reasoning": 7.9,
            "language": 8.3,
            "factual_accuracy": 7.7,
            "coding": 7.8,
            "creativity": 7.4,
            "safety": 8.6,
            "speed": 9.3,
            "cost": 9.7,
            "multimodal": 7.8,
        },
        metadata={
            "context_window": "128K tokens",
            "input_price": "$0.15 / 1M tokens",
            "output_price": "$0.60 / 1M tokens",
            "mmlu": "82.0%",
            "humaneval": "87.2%",
            "best_for": "Budget, High-throughput, Lightweight tasks",
        },
    ),
    ModelScore(
        name="Mixtral 8x22B",
        provider="Mistral AI",
        version="mixtral-8x22b-instruct",
        scores={
            "reasoning": 8.0,
            "language": 8.1,
            "factual_accuracy": 7.5,
            "coding": 7.9,
            "creativity": 7.6,
            "safety": 7.8,
            "speed": 8.5,
            "cost": 8.8,
            "multimodal": None,
        },
        metadata={
            "context_window": "64K tokens",
            "input_price": "$1.20 / 1M tokens",
            "output_price": "$1.20 / 1M tokens",
            "mmlu": "77.8%",
            "humaneval": "75.0%",
            "best_for": "Open-source quality, Multilingual, Cost-efficiency",
        },
    ),
]
