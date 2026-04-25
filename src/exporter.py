"""
Export benchmark results to JSON and CSV formats.
"""

import json
import csv
import os
from datetime import datetime
from src.benchmarks import MODELS
from src.analysis import rank_models, RECOMMENDATIONS


def export_json(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    ranked = rank_models()
    data = {
        "generated_at": datetime.now().isoformat(),
        "methodology": "Scores derived from MMLU, HumanEval, TruthfulQA, MT-Bench, BIG-Bench Hard",
        "models": [],
        "recommendations": {k: {"model": v[0], "reason": v[1]} for k, v in RECOMMENDATIONS.items()},
    }

    for rank, model, overall in ranked:
        data["models"].append({
            "rank": rank,
            "name": model.name,
            "provider": model.provider,
            "version": model.version,
            "overall_score": overall,
            "scores": model.scores,
            "metadata": model.metadata,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def export_csv(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")

    CATS = ["reasoning", "language", "factual_accuracy", "coding", "creativity", "safety", "speed", "cost", "multimodal"]
    ranked = rank_models()

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Model", "Provider", "Version", "Overall"] + CATS + ["Context", "MMLU", "HumanEval"])
        for rank, model, overall in ranked:
            row = [rank, model.name, model.provider, model.version, overall]
            for cat in CATS:
                val = model.scores.get(cat)
                row.append(val if val is not None else "N/A")
            row += [
                model.metadata.get("context_window", ""),
                model.metadata.get("mmlu", ""),
                model.metadata.get("humaneval", ""),
            ]
            writer.writerow(row)

    return path
