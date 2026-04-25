#!/usr/bin/env python3
"""
LLM Benchmark Evaluator — Main CLI Entry Point

Usage:
    python main.py                  # Full report (terminal + charts + exports)
    python main.py --report-only    # Terminal report only
    python main.py --charts-only    # Charts only
    python main.py --export         # JSON + CSV export only
    python main.py --no-charts      # Terminal report + exports, skip charts
    python main.py --model "GPT-4o" # Show profile for a specific model
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="LLM Benchmark Evaluator — Compare frontier & open-source LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--report-only", action="store_true", help="Show terminal report only")
    parser.add_argument("--charts-only", action="store_true", help="Generate charts only")
    parser.add_argument("--export", action="store_true", help="Export JSON + CSV only")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--model", type=str, help="Show profile for a specific model name")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for files")
    args = parser.parse_args()

    from src.report import run_report, console, render_header
    from src.exporter import export_json, export_csv

    # ── Single model profile ──────────────────────────────────────────────
    if args.model:
        from src.benchmarks import MODELS
        from src.analysis import score_justifications, STRENGTHS_WEAKNESSES
        from src.report import render_header, console
        from rich.panel import Panel

        target = next((m for m in MODELS if m.name.lower() == args.model.lower()), None)
        if not target:
            available = ", ".join(m.name for m in MODELS)
            console.print(f"[red]Model '{args.model}' not found.[/red]")
            console.print(f"[dim]Available: {available}[/dim]")
            sys.exit(1)

        render_header()
        from src.report import render_model_profiles
        from src.benchmarks import MODELS as ALL
        from src.analysis import rank_models
        ranked = rank_models()
        for rank, model, overall in ranked:
            if model.name == target.name:
                from src.report import render_model_profiles
                # Print just this model
                console.print(f"\n[bold cyan]Profile: {target.name}[/bold cyan]\n")
                render_model_profiles()
                break
        return

    # ── Charts only ───────────────────────────────────────────────────────
    if args.charts_only:
        from src.charts import generate_all_charts
        console = __import__("rich.console", fromlist=["Console"]).Console()
        console.print("[bold cyan]Generating charts...[/bold cyan]")
        paths = generate_all_charts(args.output_dir)
        for p in paths:
            console.print(f"  [green]✅ {p}[/green]")
        return

    # ── Export only ───────────────────────────────────────────────────────
    if args.export:
        from rich.console import Console
        console = Console()
        console.print("[bold cyan]Exporting data...[/bold cyan]")
        jp = export_json(args.output_dir)
        cp = export_csv(args.output_dir)
        console.print(f"  [green]✅ JSON: {jp}[/green]")
        console.print(f"  [green]✅ CSV:  {cp}[/green]")
        return

    # ── Full run (default) ────────────────────────────────────────────────
    run_report()

    if not args.report_only and not args.no_charts:
        try:
            from src.charts import generate_all_charts
            from rich.console import Console
            c = Console()
            c.print("[bold cyan]📊 Generating charts...[/bold cyan]")
            paths = generate_all_charts(args.output_dir)
            for p in paths:
                c.print(f"  [green]✅ {p}[/green]")
        except ImportError:
            from rich.console import Console
            Console().print("[yellow]⚠️  matplotlib not installed — skipping charts. Run: pip install matplotlib[/yellow]")

    if not args.report_only:
        from rich.console import Console
        c = Console()
        c.print("\n[bold cyan]💾 Exporting data...[/bold cyan]")
        jp = export_json(args.output_dir)
        cp = export_csv(args.output_dir)
        c.print(f"  [green]✅ JSON: {jp}[/green]")
        c.print(f"  [green]✅ CSV:  {cp}[/green]")


if __name__ == "__main__":
    main()
