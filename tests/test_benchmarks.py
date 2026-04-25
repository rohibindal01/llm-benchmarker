"""
Unit tests for LLM Benchmarker.
Run: pytest tests/
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmarks import MODELS, ModelScore
from src.analysis import rank_models, get_category_leaders, RECOMMENDATIONS


class TestModels:
    def test_models_loaded(self):
        assert len(MODELS) > 0

    def test_all_models_have_required_fields(self):
        for m in MODELS:
            assert m.name
            assert m.provider
            assert m.version
            assert isinstance(m.scores, dict)

    def test_scores_in_range(self):
        for m in MODELS:
            for cat, score in m.scores.items():
                if score is not None:
                    assert 0 <= score <= 10, f"{m.name}.{cat} = {score} out of range"

    def test_overall_score_in_range(self):
        for m in MODELS:
            overall = m.overall_score()
            assert 0 <= overall <= 10, f"{m.name} overall = {overall} out of range"


class TestAnalysis:
    def test_rank_models_returns_all(self):
        ranked = rank_models()
        assert len(ranked) == len(MODELS)

    def test_ranks_are_sequential(self):
        ranked = rank_models()
        for i, (rank, _, _) in enumerate(ranked):
            assert rank == i + 1

    def test_ranks_are_descending(self):
        ranked = rank_models()
        scores = [s for _, _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_category_leaders_cover_all_cats(self):
        leaders = get_category_leaders()
        assert "reasoning" in leaders
        assert "coding" in leaders
        assert "safety" in leaders

    def test_recommendations_have_valid_models(self):
        model_names = {m.name for m in MODELS}
        for use_case, (model, _) in RECOMMENDATIONS.items():
            assert model in model_names, f"Recommended model '{model}' for '{use_case}' not in MODELS"


class TestExports:
    def test_json_export_creates_file(self, tmp_path):
        from src.exporter import export_json
        path = export_json(str(tmp_path))
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert "models" in data
        assert len(data["models"]) == len(MODELS)

    def test_csv_export_creates_file(self, tmp_path):
        from src.exporter import export_csv
        import csv
        path = export_csv(str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == len(MODELS) + 1  # header + data
