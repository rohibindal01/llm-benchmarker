# 🤖 LLM Benchmark Evaluator

A comprehensive Python tool to compare and evaluate Large Language Models (LLMs) across standardized performance metrics. Generates terminal reports, charts, and data exports — GitHub-ready.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

## 📊 Models Evaluated

| Model | Provider | Context | MMLU | HumanEval |
|---|---|---|---|---|
| GPT-4o | OpenAI | 128K | 88.7% | 90.2% |
| Claude 3.5 Sonnet | Anthropic | 200K | 88.3% | 92.0% |
| Gemini 1.5 Pro | Google | 1M | 85.9% | 84.1% |
| LLaMA 3 70B | Meta | 8K | 82.0% | 81.7% |
| Mistral Large | Mistral AI | 128K | 84.0% | 83.2% |
| Gemini 1.5 Flash | Google | 1M | 78.9% | 74.3% |
| GPT-4o mini | OpenAI | 128K | 82.0% | 87.2% |
| Mixtral 8x22B | Mistral AI | 64K | 77.8% | 75.0% |

---

## 🔬 Evaluation Criteria

Each model is scored out of **10** across 9 categories:

| # | Category | What it Measures |
|---|---|---|
| 1 | **Reasoning** | Logical reasoning, multi-step problem solving, math accuracy |
| 2 | **Language** | Grammar, fluency, context retention, long-response coherence |
| 3 | **Factual Accuracy** | Hallucination rate, knowledge correctness, justification ability |
| 4 | **Coding** | Code generation, debugging, multi-language support |
| 5 | **Creativity** | Storytelling quality, idea generation, novelty |
| 6 | **Safety** | Bias handling, toxicity avoidance, policy compliance |
| 7 | **Speed** | Response time, token efficiency, throughput |
| 8 | **Cost** | Pricing per token, cost-performance tradeoff |
| 9 | **Multimodal** | Image/audio/video handling (N/A if unsupported) |

**Overall Score Weights:**
Reasoning 18% · Language 15% · Factual Accuracy 15% · Coding 14% · Safety 12% · Creativity 10% · Speed 8% · Cost 8%

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-benchmarker.git
cd llm-benchmarker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full report

```bash
python main.py
```

---

## 🖥️ CLI Options

```bash
python main.py                      # Full report: terminal + charts + exports
python main.py --report-only        # Terminal report only (no files generated)
python main.py --charts-only        # Generate PNG charts only
python main.py --export             # JSON + CSV export only
python main.py --no-charts          # Terminal report + exports, skip charts
python main.py --model "GPT-4o"     # Show profile for a specific model
python main.py --output-dir ./out   # Custom output directory
```

---

## 📁 Project Structure

```
llm-benchmarker/
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── benchmarks.py        # Model score database
│   ├── analysis.py          # Ranking + recommendation engine
│   ├── report.py            # Rich terminal report renderer
│   ├── charts.py            # Matplotlib chart generator
│   └── exporter.py          # JSON + CSV export
├── reports/                 # Auto-generated outputs
│   ├── overall_ranking.png
│   ├── radar_comparison.png
│   ├── heatmap.png
│   ├── cost_vs_performance.png
│   ├── benchmark_TIMESTAMP.json
│   └── benchmark_TIMESTAMP.csv
└── tests/
    └── test_benchmarks.py
```

---

## 📈 Output Examples

### Terminal Report
- Color-coded comparison table with score bars
- Overall ranking with visual progress bars
- Category leaders table
- Individual model profiles with justifications
- Use-case recommendation table
- Executive summary

### Charts (PNG)
- `overall_ranking.png` — Horizontal bar chart of all models
- `radar_comparison.png` — Multi-axis spider/radar chart
- `heatmap.png` — Score heatmap (models × categories)
- `cost_vs_performance.png` — Scatter plot tradeoff view

### Exports
- `benchmark_TIMESTAMP.json` — Full structured data
- `benchmark_TIMESTAMP.csv` — Spreadsheet-ready scores

---

## 🏆 Sample Rankings

| Rank | Model | Overall |
|---|---|---|
| 🥇 | Claude 3.5 Sonnet | 9.09 |
| 🥈 | GPT-4o | 9.02 |
| 🥉 | Gemini 1.5 Pro | 8.71 |
| 4 | Mistral Large | 8.55 |
| 5 | LLaMA 3 70B | 8.43 |

---

## 💡 Use-Case Recommendations

| Use Case | Best Model | Why |
|---|---|---|
| 🔬 Research | GPT-4o | Best overall reasoning + multimodal |
| 💻 Coding | Claude 3.5 Sonnet | HumanEval 92%, best debugging |
| 🌐 General Use | GPT-4o | Balanced across all categories |
| ✍️ Creative Writing | Claude 3.5 Sonnet | Highest creativity + language scores |
| 💰 Low-Cost | Gemini 1.5 Flash | $0.075/1M input, multimodal |
| 🔒 Self-Hosted | LLaMA 3 70B | Free, fully open-source |
| 🌍 Multilingual | Mistral Large | 26-language support, EU compliance |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Adding a New Model

Edit `src/benchmarks.py` and add a new `ModelScore` entry:

```python
ModelScore(
    name="Your Model Name",
    provider="Provider",
    version="model-version-string",
    scores={
        "reasoning": 8.5,
        "language": 8.3,
        "factual_accuracy": 8.0,
        "coding": 7.9,
        "creativity": 7.5,
        "safety": 8.2,
        "speed": 9.0,
        "cost": 8.0,
        "multimodal": None,  # None if not applicable
    },
    metadata={
        "context_window": "32K tokens",
        "input_price": "$X.XX / 1M tokens",
        "output_price": "$X.XX / 1M tokens",
        "mmlu": "82.0%",
        "humaneval": "79.0%",
        "best_for": "Your use case",
    },
)
```

Then add strengths/weaknesses in `src/analysis.py` under `STRENGTHS_WEAKNESSES`.

---

## 📚 Data Sources

Scores are derived from publicly available benchmarks:

- **MMLU** (Massive Multitask Language Understanding)
- **HumanEval** (Code generation)
- **TruthfulQA** (Factual accuracy & hallucination)
- **MT-Bench** (Multi-turn instruction following)
- **BIG-Bench Hard** (Challenging reasoning tasks)
- **HellaSwag** (Commonsense reasoning)
- **MATH** (Mathematical problem solving)
- **SWE-Bench** (Software engineering tasks)

