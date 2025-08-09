import signal
import sys
from pathlib import Path

import pandas as pd
import questionary
from questionary import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


def build_csv_interactive(output_path: str = "eval_config.csv") -> None:
    """
    Enhanced interactive CSV builder with arrow key navigation.

    Args:
        output_path: Path where the CSV file will be saved.
    """
    console = Console()

    # Set up signal handler for graceful exit
    def signal_handler(sig, frame):
        console.print("\n\n[yellow]Interrupted by user. Exiting...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Custom style for questionary
    custom_style = Style(
        [
            ("qmark", "fg:#673ab7 bold"),
            ("question", "bold"),
            ("answer", "fg:#f44336 bold"),
            ("pointer", "fg:#673ab7 bold"),
            ("highlighted", "fg:#673ab7 bold"),
            ("selected", "fg:#cc5454"),
            ("separator", "fg:#cc5454"),
            ("instruction", "fg:#abb2bf"),
            ("text", ""),
            ("disabled", "fg:#858585 italic"),
        ]
    )

    # Clear screen and show header
    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]OpenEuroLLM Evaluation Configuration Builder[/bold cyan]\n"
            "[dim]Use arrow keys to navigate, Enter to select, Ctrl+C to exit[/dim]",
            border_style="cyan",
        )
    )

    # Step 1: Get models with enhanced input
    console.print("\n[bold cyan]📦 Step 1: Add Models[/bold cyan]")

    models = []
    add_more = True

    while add_more:
        try:
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    "➕ Add a model",
                    "📋 View current models"
                    if models
                    else questionary.Choice(
                        "📋 View current models", disabled="No models added yet"
                    ),
                    "✅ Continue to tasks"
                    if models
                    else questionary.Choice(
                        "✅ Continue to tasks", disabled="Add at least one model first"
                    ),
                ],
                style=custom_style,
            ).ask()

            if action is None:  # User pressed Ctrl+C
                console.print("\n[yellow]Cancelled by user.[/yellow]")
                return
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled by user.[/yellow]")
            return

        if action == "➕ Add a model":
            model = questionary.text(
                "Enter model (HuggingFace ID or local path):",
                instruction="(e.g., meta-llama/Llama-2-7b-hf or /path/to/model)",
                style=custom_style,
            ).ask()
            
            if model is None:  # User pressed Ctrl+C
                console.print("\n[yellow]Cancelled by user.[/yellow]")
                return

            if model:
                models.append(model)
                console.print(f"[green]✓ Added: {model}[/green]")

        elif action == "📋 View current models":
            console.print("\n[bold]Current models:[/bold]")
            for i, model in enumerate(models, 1):
                console.print(f"  {i}. [cyan]{model}[/cyan]")
            console.print()

        elif action == "✅ Continue to tasks":
            add_more = False

    # Step 2: Configure tasks
    console.print("\n[bold cyan]📝 Step 2: Configure Tasks[/bold cyan]")

    task_configs = []
    add_more = True

    while add_more:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "➕ Add a task",
                "📋 View current tasks"
                if task_configs
                else questionary.Choice(
                    "📋 View current tasks", disabled="No tasks added yet"
                ),
                "✅ Continue to preview"
                if task_configs
                else questionary.Choice(
                    "✅ Continue to preview", disabled="Add at least one task first"
                ),
            ],
            style=custom_style,
        ).ask()

        if action is None:
            console.print("\n[yellow]Cancelled by user.[/yellow]")
            return

        if action == "➕ Add a task":
            # Direct task input
            task = questionary.text("Enter task name:", style=custom_style).ask()
            if task is None:
                console.print("\n[yellow]Cancelled by user.[/yellow]")
                return

            if task:
                # Get n_shot values
                n_shot_choice = questionary.select(
                    f"Select n_shot configuration for '{task}':",
                    choices=[
                        "0 (zero-shot)",
                        "5 (few-shot)",
                        "0,5 (both)",
                        "0,5,10,25 (multiple)",
                        "📝 Custom values",
                    ],
                    style=custom_style,
                ).ask()

                if n_shot_choice is None:
                    console.print("\n[yellow]Cancelled by user.[/yellow]")
                    return

                if n_shot_choice == "📝 Custom values":
                    n_shots_str = questionary.text(
                        "Enter n_shot values (comma-separated):",
                        instruction="(e.g., 0,5,10)",
                        style=custom_style,
                    ).ask()
                    if n_shots_str is None:
                        console.print("\n[yellow]Cancelled by user.[/yellow]")
                        return
                else:
                    # Extract numbers from the choice
                    import re

                    n_shots_str = ",".join(re.findall(r"\d+", n_shot_choice))

                try:
                    n_shots = [int(x.strip()) for x in n_shots_str.split(",")]
                    task_configs.append((task, n_shots))
                    console.print(
                        f"[green]✓ Added: {task} with n_shot={n_shots}[/green]"
                    )
                except ValueError:
                    console.print("[red]Invalid n_shot values. Skipping.[/red]")

        elif action == "📋 View current tasks":
            console.print("\n[bold]Current tasks:[/bold]")
            for i, (task, n_shots) in enumerate(task_configs, 1):
                console.print(f"  {i}. [green]{task}[/green] → n_shot={n_shots}")
            console.print()

        elif action == "✅ Continue to preview":
            add_more = False

    # Build the dataframe
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Building configuration matrix...", total=None)

        rows = []
        for model in models:
            for task_name, n_shots in task_configs:
                for n_shot in n_shots:
                    rows.append(
                        {"model_path": model, "task_path": task_name, "n_shot": n_shot}
                    )

        df = pd.DataFrame(rows)
        progress.update(task, completed=True)

    # Show preview
    console.print("\n[bold cyan]👁️  Preview[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Task", style="green")
    table.add_column("n_shot", justify="right", style="yellow")

    # Show first 10 rows
    for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
        table.add_row(
            str(idx), str(row["model_path"]), str(row["task_path"]), str(row["n_shot"])
        )

    if len(df) > 10:
        table.add_row("...", "...", "...", "...")

    console.print(table)
    console.print(f"\n[bold]Total configurations: {len(df)}[/bold]")

    # Summary statistics
    console.print("\n[bold cyan]📊 Summary[/bold cyan]")
    console.print(f"  • Models: {len(models)}")
    console.print(f"  • Tasks: {len(task_configs)}")
    console.print(f"  • Total evaluations: {len(df)}")

    # Save confirmation
    save = questionary.confirm(
        f"\nSave configuration to {output_path}?", default=True, style=custom_style
    ).ask()

    if save is None:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        return

    if save:
        # Ensure directory exists
        output_dir = Path(output_path).parent
        if output_dir != Path("."):
            output_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Saving to {output_path}...", total=None)
            df.to_csv(output_path, index=False)
            progress.update(task, completed=True)

        console.print(f"\n[green]✅ Configuration saved to {output_path}[/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Review the configuration: [cyan]cat {output_path}[/cyan]")
        console.print(
            f"  2. Run evaluation: [cyan]oellm schedule-eval --eval_csv_path {output_path}[/cyan]"
        )
    else:
        console.print("\n[yellow]Configuration not saved.[/yellow]")
