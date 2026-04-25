"""
Analysis & ranking engine for LLM benchmarks.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from src.benchmarks import ModelScore, MODELS


CATEGORY_DESCRIPTIONS = {
    "reasoning": "Logical reasoning, multi-step problem solving, mathematical accuracy",
    "language": "Grammar & fluency, context retention, coherence in long responses",
    "factual_accuracy": "Hallucination rate, knowledge correctness, ability to justify answers",
    "coding": "Code generation, debugging ability, multi-language support",
    "creativity": "Storytelling quality, idea generation, novelty of responses",
    "safety": "Bias handling, toxicity avoidance, policy compliance",
    "speed": "Response latency, token efficiency, throughput",
    "cost": "Pricing per token, cost-performance tradeoff",
    "multimodal": "Image understanding, audio/video handling (N/A if not applicable)",
}

STRENGTHS_WEAKNESSES = {
    "GPT-4o": {
        "strengths": [
            "Best-in-class multimodal understanding",
            "Excellent reasoning with vision tasks",
            "Strong real-world task performance",
            "Reliable code generation",
        ],
        "weaknesses": [
            "High output cost ($10/1M tokens)",
            "Context window smaller than Gemini",
            "Rate limits on free tier",
        ],
    },
    "Claude 3.5 Sonnet": {
        "strengths": [
            "Top coding benchmark scores (HumanEval 92%)",
            "Largest context window (200K) among closed models",
            "Exceptional safety and alignment",
            "Nuanced creative writing",
        ],
        "weaknesses": [
            "Premium pricing",
            "No native audio/video support",
            "Slower than Flash-tier models",
        ],
    },
    "Gemini 1.5 Pro": {
        "strengths": [
            "Industry-leading 1M token context window",
            "Strong native multimodal (image, audio, video)",
            "Competitive pricing vs GPT-4",
            "Google ecosystem integration",
        ],
        "weaknesses": [
            "Slightly behind on coding tasks",
            "Latency can increase with very long context",
            "Occasionally verbose responses",
        ],
    },
    "LLaMA 3 70B": {
        "strengths": [
            "Completely free and open-source",
            "Self-hostable for full data privacy",
            "Competitive with smaller closed models",
            "Active open-source community",
        ],
        "weaknesses": [
            "Requires significant GPU hardware to self-host",
            "No multimodal support (base model)",
            "Shorter 8K context window",
            "Weaker safety alignment vs closed models",
        ],
    },
    "Mistral Large": {
        "strengths": [
            "Best multilingual performance in class",
            "Fast inference speeds",
            "Strong coding performance for the price",
            "EU-based privacy compliance",
        ],
        "weaknesses": [
            "Smaller ecosystem than OpenAI/Google",
            "No multimodal capabilities",
            "Less extensive tool/function support",
        ],
    },
    "Gemini 1.5 Flash": {
        "strengths": [
            "Extremely low cost ($0.075/1M input tokens)",
            "1M token context at budget price",
            "Fast response times",
            "Multimodal at minimal cost",
        ],
        "weaknesses": [
            "Weaker reasoning vs Pro-tier models",
            "Lower factual accuracy",
            "Not suitable for complex multi-step tasks",
        ],
    },
    "GPT-4o mini": {
        "strengths": [
            "Very affordable ($0.15/1M input)",
            "Strong coding relative to price",
            "High throughput / low latency",
            "Broad API feature parity with GPT-4o",
        ],
        "weaknesses": [
            "Reduced reasoning depth vs GPT-4o",
            "Not ideal for complex tasks",
            "Limited context vs Gemini Flash",
        ],
    },
    "Mixtral 8x22B": {
        "strengths": [
            "Open-source Mixture-of-Experts architecture",
            "Good multilingual support",
            "Cost-effective via API",
            "Flexible deployment options",
        ],
        "weaknesses": [
            "Behind frontier closed models",
            "No multimodal support",
            "64K context limit",
        ],
    },
}

RECOMMENDATIONS = {
    "Research": ("GPT-4o", "Best overall reasoning + multimodal for analyzing papers/data"),
    "Coding": ("Claude 3.5 Sonnet", "Highest HumanEval scores, strong debugging, 200K context"),
    "General Use": ("GPT-4o", "Balanced excellence across all categories with broad ecosystem"),
    "Creative Writing": ("Claude 3.5 Sonnet", "Nuanced storytelling, highest creativity + language scores"),
    "Low-Cost Applications": ("Gemini 1.5 Flash", "Ultra-low price, 1M context, multimodal — best value"),
    "Privacy / Self-Hosted": ("LLaMA 3 70B", "Fully open-source, self-hostable, no data leaves your infra"),
    "Multilingual": ("Mistral Large", "Best multilingual support, EU data compliance"),
}


def rank_models(models: List[ModelScore] = MODELS) -> List[Tuple[int, ModelScore, float]]:
    ranked = sorted(models, key=lambda m: m.overall_score(), reverse=True)
    return [(i + 1, m, m.overall_score()) for i, m in enumerate(ranked)]


def get_category_leaders(models: List[ModelScore] = MODELS) -> Dict[str, Tuple[str, float]]:
    leaders = {}
    for cat in CATEGORY_DESCRIPTIONS:
        best = max(
            (m for m in models if m.scores.get(cat) is not None),
            key=lambda m: m.scores.get(cat, 0),
        )
        leaders[cat] = (best.name, best.scores[cat])
    return leaders


def score_justifications(model: ModelScore) -> Dict[str, str]:
    """Return per-category justification strings."""
    justifications = {
        "GPT-4o": {
            "reasoning": "Scores ~88.7% on MMLU; excels at multi-hop reasoning & math (MATH benchmark ~76%)",
            "language": "Near-perfect fluency, strong long-context coherence up to 128K tokens",
            "factual_accuracy": "TruthfulQA ~78%; occasionally hallucinates on niche topics",
            "coding": "HumanEval 90.2%; broad language coverage; strong debugging",
            "creativity": "High diversity, engaging narratives; occasionally formulaic",
            "safety": "Strong RLHF alignment; passes most red-team evaluations",
            "speed": "~300 tokens/sec on standard; slower with vision tasks",
            "cost": "$2.50 input / $10 output per 1M — premium but justified",
            "multimodal": "Best-in-class image+text reasoning; supports vision and document analysis",
        },
        "Claude 3.5 Sonnet": {
            "reasoning": "88.3% MMLU; top-tier multi-step problem solving; strong math",
            "language": "Exceptional fluency & nuance; 200K context maintained coherently",
            "factual_accuracy": "High TruthfulQA score; rarely hallucinates; excellent citations",
            "coding": "HumanEval 92% — highest in benchmark; strong SWE-bench results",
            "creativity": "Rich storytelling; most nuanced voice among evaluated models",
            "safety": "Highest safety score; Constitutional AI training; minimal bias",
            "speed": "Moderate latency; optimized streaming; fast first-token",
            "cost": "$3 input / $15 output per 1M — premium pricing",
            "multimodal": "Strong image understanding; no native audio/video",
        },
        "Gemini 1.5 Pro": {
            "reasoning": "85.9% MMLU; strong at long-context reasoning; solid math",
            "language": "Natural fluency; best at >100K context retention",
            "factual_accuracy": "Good accuracy; occasional over-confident errors",
            "coding": "HumanEval 84.1%; decent but behind Claude/GPT-4",
            "creativity": "Creative with long inputs; slightly generic short outputs",
            "safety": "Strong Responsible AI practices; Google policy enforced",
            "speed": "Moderate; scales with context length",
            "cost": "$1.25/1M input (≤128K) — competitive",
            "multimodal": "Native video, audio, image; best long-video understanding",
        },
        "LLaMA 3 70B": {
            "reasoning": "82% MMLU; strong for open-source; weaker on multi-step",
            "language": "Excellent fluency for open model; coherent at 8K limit",
            "factual_accuracy": "Decent but higher hallucination rate than closed models",
            "coding": "HumanEval 81.7%; competitive with older closed models",
            "creativity": "Good storytelling; limited by instruction fine-tuning quality",
            "safety": "Community fine-tuned safety; weaker than closed-model RLHF",
            "speed": "Fast on capable hardware; ~8K context helps speed",
            "cost": "Free to run; infra cost is your only expense",
            "multimodal": "Base model is text-only; multimodal variants exist separately",
        },
        "Mistral Large": {
            "reasoning": "84% MMLU; solid structured reasoning; good at logic puzzles",
            "language": "Strong multilingual; 26-language support; coherent 128K context",
            "factual_accuracy": "Reliable; slight accuracy gap vs frontier models",
            "coding": "HumanEval 83.2%; excellent for Python, JS, Rust",
            "creativity": "Competent; less distinctive voice than Claude",
            "safety": "EU-aligned; GDPR compliant; robust refusal patterns",
            "speed": "High throughput; fast inference via Mistral API",
            "cost": "$2/1M input, $6/1M output — excellent value",
            "multimodal": "Text only; no vision or audio",
        },
        "Gemini 1.5 Flash": {
            "reasoning": "78.9% MMLU; weaker complex reasoning; fine for single-step",
            "language": "Good fluency; maintains coherence across 1M context surprisingly well",
            "factual_accuracy": "Lower accuracy than Pro; more prone to confident errors",
            "coding": "HumanEval 74.3%; suitable for simple/moderate tasks",
            "creativity": "Serviceable; lacks depth for complex creative work",
            "safety": "Standard Google safety; solid for most use cases",
            "speed": "Very fast; lowest latency among evaluated models",
            "cost": "$0.075/1M input — cheapest multimodal model evaluated",
            "multimodal": "Image, audio, video support — remarkable at this price point",
        },
        "GPT-4o mini": {
            "reasoning": "82% MMLU; strong for its price tier; suitable for structured tasks",
            "language": "Fluent and coherent; good short-medium context handling",
            "factual_accuracy": "Lower than GPT-4o; hallucinations more frequent",
            "coding": "HumanEval 87.2%; punches above its weight in coding",
            "creativity": "Limited depth; formulaic on complex creative requests",
            "safety": "Inherits GPT-4o safety training; OpenAI moderation applied",
            "speed": "High throughput; 9.3/10 for latency performance",
            "cost": "$0.15/1M input — best-in-class per-dollar coding performance",
            "multimodal": "Vision support; narrower multimodal than GPT-4o",
        },
        "Mixtral 8x22B": {
            "reasoning": "77.8% MMLU; MoE enables efficient parallel reasoning",
            "language": "Solid multilingual; coherent to 64K context",
            "factual_accuracy": "Moderate; weaker than frontier on factual recall",
            "coding": "HumanEval 75%; good for standard tasks",
            "creativity": "Competent; behind dedicated instruction-tuned models",
            "safety": "Open model; community fine-tunes vary in safety quality",
            "speed": "Fast inference due to sparse MoE — only active experts compute",
            "cost": "$1.20/1M flat via API — very cost-effective",
            "multimodal": "Text-only in standard form",
        },
    }
    return justifications.get(model.name, {})
