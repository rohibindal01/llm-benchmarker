"""
Terminal-based report renderer using rich.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.rule import Rule
from rich.padding import Padding

from src.benchmarks import MODELS, ModelScore
from src.analysis import (
    rank_models,
    get_category_leaders,
    score_justifications,
    RECOMMENDATIONS,
    STRENGTHS_WEAKNESSES,
    CATEGORY_DESCRIPTIONS,
)

console = Console()

SCORE_COLORS = {
    (9.5, 10.0): "bold green",
    (9.0, 9.5): "green",
    (8.5, 9.0): "bright_green",
    (8.0, 8.5): "yellow",
    (7.5, 8.0): "bright_yellow",
    (7.0, 7.5): "orange3",
    (0.0, 7.0): "red",
}


def score_color(score: float | None) -> str:
    if score is None:
        return "dim"
    for (lo, hi), color in SCORE_COLORS.items():
        if lo <= score <= hi:
            return color
    return "white"


def score_bar(score: float | None, width: int = 8) -> str:
    if score is None:
        return "  N/A  "
    filled = round(score / 10 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {score:.1f}"


def render_header():
    console.print()
    console.print(
        Panel.fit(
            "[bold white]🤖  LLM Benchmark Evaluator  🤖[/bold white]\n"
            "[dim]Comprehensive comparison of frontier & open-source language models[/dim]\n"
            "[dim]Scores sourced from MMLU · HumanEval · TruthfulQA · MT-Bench · BIG-Bench[/dim]",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()


def render_comparison_table():
    console.print(Rule("[bold cyan]📊 Full Score Comparison Table[/bold cyan]", style="cyan"))
    console.print()

    CATS = ["reasoning", "language", "factual_accuracy", "coding", "creativity", "safety", "speed", "cost", "multimodal"]
    SHORT = {
        "reasoning": "Reason", "language": "Lang", "factual_accuracy": "Facts",
        "coding": "Code", "creativity": "Create", "safety": "Safety",
        "speed": "Speed", "cost": "Cost", "multimodal": "Multi",
    }

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_blue",
        border_style="blue",
        expand=True,
    )
    table.add_column("Model", min_width=20, style="bold white")
    table.add_column("Provider", min_width=12, style="dim")
    for cat in CATS:
        table.add_column(SHORT[cat], justify="center", min_width=7)
    table.add_column("Overall ★", justify="center", min_width=9, style="bold yellow")

    ranked = rank_models()
    for rank, model, overall in ranked:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank} ")
        row = [f"{medal} {model.name}", model.provider]
        for cat in CATS:
            s = model.scores.get(cat)
            color = score_color(s)
            row.append(f"[{color}]{score_bar(s, 6)}[/{color}]")
        row.append(f"[bold yellow]{overall:.2f}[/bold yellow]")
        table.add_row(*row)

    console.print(table)
    console.print()


def render_rankings():
    console.print(Rule("[bold cyan]🏆 Overall Rankings[/bold cyan]", style="cyan"))
    console.print()

    ranked = rank_models()
    for rank, model, overall in ranked:
        medal = {1: "🥇 [bold green]", 2: "🥈 [bold white]", 3: "🥉 [bold yellow]"}.get(
            rank, f"  #{rank} [dim]"
        )
        close = "[/bold green]" if rank == 1 else ("[/bold white]" if rank == 2 else ("[/bold yellow]" if rank == 3 else "[/dim]"))
        bar_width = int(overall / 10 * 40)
        bar = "█" * bar_width + "░" * (40 - bar_width)
        console.print(
            f"  {medal}{model.name:<22}{close} "
            f"[bright_blue]{bar}[/bright_blue] "
            f"[bold yellow]{overall:.2f}[/bold yellow]/10  "
            f"[dim]{model.provider}[/dim]"
        )
    console.print()


def render_category_leaders():
    console.print(Rule("[bold cyan]🏅 Category Leaders[/bold cyan]", style="cyan"))
    console.print()

    leaders = get_category_leaders()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold bright_blue")
    table.add_column("Category", min_width=20)
    table.add_column("Description", min_width=40)
    table.add_column("Leader", min_width=22)
    table.add_column("Score", justify="center", min_width=8)

    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        leader_name, leader_score = leaders[cat]
        color = score_color(leader_score)
        table.add_row(
            f"[bold white]{cat.replace('_', ' ').title()}[/bold white]",
            f"[dim]{desc[:48]}...[/dim]" if len(desc) > 48 else f"[dim]{desc}[/dim]",
            f"[cyan]{leader_name}[/cyan]",
            f"[{color}]{leader_score:.1f}[/{color}]",
        )
    console.print(table)
    console.print()


def render_model_profiles():
    console.print(Rule("[bold cyan]📋 Individual Model Profiles[/bold cyan]", style="cyan"))
    console.print()

    ranked = rank_models()
    CATS = ["reasoning", "language", "factual_accuracy", "coding", "creativity", "safety", "speed", "cost", "multimodal"]

    for rank, model, overall in ranked:
        sw = STRENGTHS_WEAKNESSES.get(model.name, {})
        justs = score_justifications(model)

        strengths_text = "\n".join(f"  ✅ {s}" for s in sw.get("strengths", []))
        weaknesses_text = "\n".join(f"  ⚠️  {w}" for w in sw.get("weaknesses", []))

        meta = model.metadata
        meta_lines = (
            f"  Context: {meta.get('context_window', 'N/A')}  |  "
            f"MMLU: {meta.get('mmlu', 'N/A')}  |  "
            f"HumanEval: {meta.get('humaneval', 'N/A')}\n"
            f"  Input: {meta.get('input_price', 'N/A')}  |  "
            f"Output: {meta.get('output_price', 'N/A')}"
        )

        scores_lines = []
        for cat in CATS:
            s = model.scores.get(cat)
            color = score_color(s)
            label = cat.replace("_", " ").title().ljust(16)
            bar = score_bar(s, 10)
            just = justs.get(cat, "")
            scores_lines.append(
                f"  [dim]{label}[/dim] [{color}]{bar}[/{color}]"
                + (f"  [dim italic]{just[:60]}[/dim italic]" if just else "")
            )

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        panel_content = (
            f"[bold white]{model.name}[/bold white]  [dim]v{model.version} — {model.provider}[/dim]  "
            f"[bold yellow]Overall: {overall:.2f}/10[/bold yellow]\n\n"
            f"[dim]{meta_lines}[/dim]\n\n"
            + "\n".join(scores_lines)
            + f"\n\n[bold green]Strengths:[/bold green]\n{strengths_text}"
            + f"\n\n[bold red]Weaknesses:[/bold red]\n{weaknesses_text}"
        )

        console.print(
            Panel(
                panel_content,
                title=f"[bold bright_blue]{medal} Rank #{rank}[/bold bright_blue]",
                border_style="bright_blue" if rank <= 3 else "blue",
                padding=(1, 2),
            )
        )
        console.print()


def render_recommendations():
    console.print(Rule("[bold cyan]💡 Use-Case Recommendations[/bold cyan]", style="cyan"))
    console.print()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold bright_blue", border_style="blue")
    table.add_column("Use Case", min_width=24, style="bold white")
    table.add_column("Best Model", min_width=22, style="bold cyan")
    table.add_column("Rationale", min_width=50)

    icons = {
        "Research": "🔬",
        "Coding": "💻",
        "General Use": "🌐",
        "Creative Writing": "✍️ ",
        "Low-Cost Applications": "💰",
        "Privacy / Self-Hosted": "🔒",
        "Multilingual": "🌍",
    }

    for use_case, (model, reason) in RECOMMENDATIONS.items():
        icon = icons.get(use_case, "•")
        table.add_row(f"{icon} {use_case}", model, f"[dim]{reason}[/dim]")

    console.print(table)
    console.print()


def render_bullet_summary():
    console.print(Rule("[bold cyan]📝 Executive Summary[/bold cyan]", style="cyan"))
    console.print()

    ranked = rank_models()
    top3 = ranked[:3]

    summary = Panel(
        f"[bold white]Top 3 Models (Overall Score):[/bold white]\n"
        + "\n".join(
            f"  {'🥇🥈🥉'[i]}  [cyan]{m.name}[/cyan] ({s:.2f}/10) — [dim]{m.metadata.get('best_for', '')}[/dim]"
            for i, (_, m, s) in enumerate(top3)
        )
        + "\n\n"
        "[bold white]Key Takeaways:[/bold white]\n"
        "  • [green]Claude 3.5 Sonnet[/green] leads in coding (HumanEval 92%) and safety alignment\n"
        "  • [green]GPT-4o[/green] is the strongest general-purpose + multimodal model\n"
        "  • [green]Gemini 1.5 Pro[/green] wins on context length (1M tokens) and multimodal breadth\n"
        "  • [green]Gemini 1.5 Flash[/green] is the best value — cheapest multimodal at $0.075/1M tokens\n"
        "  • [green]LLaMA 3 70B[/green] remains the best open-source option for self-hosted deployments\n"
        "  • [green]Mistral Large[/green] excels at multilingual tasks and EU data compliance\n"
        "  • [green]GPT-4o mini[/green] punches above its weight in coding at very low cost\n\n"
        "[bold white]Scoring Methodology:[/bold white]\n"
        "  Scores derived from MMLU, HumanEval, TruthfulQA, MT-Bench, BIG-Bench Hard,\n"
        "  HellaSwag, MATH, SWE-Bench, and independent evaluator testing.\n"
        "  Weighted average: Reasoning 18% · Language 15% · Facts 15% · Coding 14%\n"
        "  Safety 12% · Speed 8% · Cost 8% · Creativity 10%",
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold bright_blue]Summary[/bold bright_blue]",
    )
    console.print(summary)
    console.print()


def run_report():
    render_header()
    render_comparison_table()
    render_rankings()
    render_category_leaders()
    render_model_profiles()
    render_recommendations()
    render_bullet_summary()
    console.print("[bold green]✅ Report complete.[/bold green]\n")
